"""Tensor contractions: einsum, tensordot, inner, vdot."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.lib_nodes.blas import expand_dot
from hpcagent_bench.translators.numpyto_common.lib_nodes.call_args import (
    axes_kwarg,
    tensordot_axes,
    parse_einsum_subscripts,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import iter_extent_of_
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import (
    alloc_marker,
    attr_call,
    const_,
    name_,
    name_id,
    store_,
    wrap_for_loops,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.scalarize import scalarize_at_iters


def expand_einsum_ellipsis(spec: str, ranks: list[int]) -> str:
    """Rewrite ``...`` in an einsum spec to explicit index letters using each
    operand's rank, so the plain-subscript lowering handles it:
    ``'...ij,...jk->...ik'`` on rank-3 operands -> ``'Aij,Ajk->Aik'`` (the
    broadcast axis becomes a fresh shared index ``A``). Requires an explicit
    ``->`` and that every ``...`` covers the same axis count (numpy's
    differing-rank ellipsis broadcasting isn't modelled) -- else raises.
    """
    spec = spec.replace(" ", "")
    if "->" not in spec:
        raise NotImplementedError("einsum: ellipsis requires an explicit -> output")
    lhs, rhs = spec.split("->")
    ins = lhs.split(",")
    if len(ins) != len(ranks):
        raise NotImplementedError("einsum: operand count != subscript count")
    ell_rank, seen = 0, False
    for sub, r in zip(ins, ranks):
        if "..." in sub:
            er = r - len(sub.replace("...", ""))
            if er < 0:
                raise NotImplementedError("einsum: too many indices for operand rank")
            if seen and er != ell_rank:
                raise NotImplementedError("einsum: differing ellipsis ranks (broadcast) unsupported")
            ell_rank, seen = er, True
    used = set(spec) - set(".,->")
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    fresh = [c for c in letters if c not in used]
    if len(fresh) < ell_rank:
        raise NotImplementedError("einsum: not enough free index letters for ellipsis")
    ell = "".join(fresh[:ell_rank])
    new_ins = [sub.replace("...", ell) for sub in ins]
    return ",".join(new_ins) + "->" + rhs.replace("...", ell)


#: Counter for the scratch buffers :func:`materialize_operands` spills non-Name operands into.
#: Reset per translation unit by :func:`reset_temp_counters`.
OP_SPILL_TEMP = [0]


def materialize_operands(
    operands: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    prefix: str,
    local_dtypes: dict[str, str] | None = None,
    fresh_local_allocs: dict[str, tuple[str, ...]] | None = None,
) -> tuple[list[ast.stmt], list[ast.expr]]:
    """Spill every non-Name operand into a fresh scratch buffer. Several
    expanders (einsum, concatenate) are written against bare-Name operands, but
    a kernel may hand them a slice/call/arithmetic expression -- dwt2d's
    ``np.concatenate((e[:, 1:], e[:, 0:1]), axis=1)``. Copying each into a local
    first lets the bare-Name expansion run unchanged, instead of teaching every
    expander to index arbitrary sub-expressions.

    ``prefix`` is the caller's name stem (``__es_``/``__cc_``): the buffer is
    ``<prefix>op<N>``, its copy-loop iterators ``<prefix>c<i>``.

    Returns ``(prelude_stmts, operand_exprs)``: the copy loops to emit first,
    and the operand list with each spill replaced by its buffer Name. The
    buffer is registered in ``shape_table`` + ``fresh_local_allocs`` so the
    emitter declares and allocates it, inheriting the source array's dtype. An
    operand whose extent won't resolve passes through untouched, so the
    caller's own bare-Name check still reports it.
    """
    prelude: list[ast.stmt] = []
    out: list[ast.expr] = []
    for op in operands:
        if isinstance(op, ast.Name):
            out.append(op)
            continue
        op_ext = iter_extent_of_(op, shape_table)
        if op_ext is None:
            out.append(op)  # unresolved -- the caller's bare-Name check raises
            continue
        OP_SPILL_TEMP[0] += 1
        tmp = f"{prefix}op{OP_SPILL_TEMP[0]}"
        tmp_shape = tuple(ast.unparse(e) for e in op_ext)
        shape_table[tmp] = tmp_shape
        if fresh_local_allocs is not None:
            fresh_local_allocs[tmp] = tmp_shape
        if local_dtypes is not None:
            base = op.value if isinstance(op, ast.Subscript) else op
            base_name = name_id(base)
            if base_name and local_dtypes.get(base_name):
                local_dtypes[tmp] = local_dtypes[base_name]
        cp_iters = [f"{prefix}c{i}" for i in range(len(op_ext))]
        cp_nodes = [name_(c) for c in cp_iters]
        cp_slot = cp_nodes[0] if len(cp_nodes) == 1 else ast.Tuple(elts=cp_nodes, ctx=ast.Load())
        cp_src = scalarize_at_iters(op, cp_nodes, shape_table)
        cp_dst = ast.Subscript(value=name_(tmp), slice=cp_slot, ctx=ast.Store())
        prelude.append(alloc_marker(tmp))
        prelude.extend(wrap_for_loops(cp_iters, list(tmp_shape), [ast.Assign(targets=[cp_dst], value=cp_src)]))
        out.append(name_(tmp))
    return prelude, out


def expand_einsum(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    local_dtypes: dict[str, str] | None = None,
    fresh_local_allocs: dict[str, tuple[str, ...]] | None = None,
) -> list[ast.stmt]:
    """Lower ``np.einsum(subscripts, *operands)`` to a nested loop nest. Output
    indices become nested loops over the result; indices summed away (in the
    inputs but not the output) become inner accumulation loops. Body:
    ``out[out_idx] (+)= prod(operand[its idx letters])``.

    Handles N operands and arbitrary index letters, including a letter repeated
    within one operand (``ii`` -> diagonal ``A[i, i]``). One path subsumes
    matmul ``ij,jk->ik``, transpose ``ij->ji``, trace ``ii->``, diagonal
    ``ii->i``, outer ``i,j->ij`` and sum ``ij->``.

    A non-Name operand (Subscript/Call/BinOp like ``psi_frag[f]``) is first
    spilled into a fresh scratch buffer (the same copy-into-local pattern
    ``expand_copy``/``expand_median`` use), so the bare-Name expansion below
    contracts the buffer.
    """
    if not args or not isinstance(args[0], ast.Constant) or not isinstance(args[0].value, str):
        raise NotImplementedError("einsum needs a literal subscript string")
    spec = args[0].value
    operands = args[1:]
    # Materialize any non-Name operand into a fresh local so the bare-Name expansion (and
    # the ellipsis rank lookup) sees a Name.
    prelude, operands = materialize_operands(
        operands, shape_table, "__es_", local_dtypes=local_dtypes, fresh_local_allocs=fresh_local_allocs
    )
    if "..." in spec:
        # Expand ``...`` to explicit letters from each operand's rank (needs the
        # shape table), then lower the plain form; the parser stays ellipsis-free.
        ranks = []
        for op in operands:
            if not isinstance(op, ast.Name) or shape_table.get(op.id) is None:
                raise NotImplementedError("einsum ellipsis needs bare-Name operands with known shape")
            ranks.append(len(shape_table[op.id]))
        spec = expand_einsum_ellipsis(spec, ranks)
    inputs, output = parse_einsum_subscripts(spec)
    if len(inputs) != len(operands):
        raise NotImplementedError("einsum operand count mismatches subscripts")
    operand_names: list[str] = []
    for op in operands:
        if not isinstance(op, ast.Name):
            raise NotImplementedError("einsum operands must be bare Names")
        operand_names.append(op.id)
    # Map every index letter to its extent symbol (first operand that uses it).
    letter_extent: dict[str, str] = {}
    for spec, name in zip(inputs, operand_names):
        shape = shape_table.get(name)
        if shape is None or len(shape) != len(spec):
            raise NotImplementedError(f"einsum: shape of {name!r} unknown / rank mismatch")
        for letter, dim in zip(spec, shape):
            letter_extent.setdefault(letter, dim)
    out_letters = list(output)
    sum_letters = [c for c in letter_extent if c not in out_letters]
    # Per-letter loop variable.
    var_of = {c: f"__es_{c}" for c in letter_extent}

    def subscript_(name: str, spec: str) -> ast.expr:
        idx = [name_(var_of[c]) for c in spec]
        sl = idx[0] if len(idx) == 1 else ast.Tuple(elts=idx, ctx=ast.Load())
        return ast.Subscript(value=name_(name), slice=sl, ctx=ast.Load())

    # Product of every operand scalarised at its index letters.
    product: ast.expr = subscript_(operand_names[0], inputs[0])
    for name, spec in zip(operand_names[1:], inputs[1:]):
        product = ast.BinOp(left=product, op=ast.Mult(), right=subscript_(name, spec))

    # Output write target.
    if out_letters:
        out_idx = [name_(var_of[c]) for c in out_letters]
        out_sl = out_idx[0] if len(out_idx) == 1 else ast.Tuple(elts=out_idx, ctx=ast.Load())
        out_store = ast.Subscript(value=name_(target.id), slice=out_sl, ctx=ast.Store())
    else:
        out_store = store_(target.id)  # scalar result (trace / full sum)

    # Inner: accumulate the product over the summed letters.
    if sum_letters:
        body: list[ast.stmt] = [ast.AugAssign(target=out_store, op=ast.Add(), value=product)]
        body = wrap_for_loops([var_of[c] for c in sum_letters], [letter_extent[c] for c in sum_letters], body)
        zero = ast.Assign(targets=[copy.deepcopy(out_store)], value=const_(0.0))
        inner: list[ast.stmt] = [zero] + body
    else:
        inner = [ast.Assign(targets=[copy.deepcopy(out_store)], value=product)]

    if out_letters:
        return prelude + wrap_for_loops(
            [var_of[c] for c in out_letters], [letter_extent[c] for c in out_letters], inner
        )
    return prelude + inner


def expand_tensordot(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
    local_dtypes: dict[str, str] | None = None,
    fresh_local_allocs: dict[str, tuple[str, ...]] | None = None,
) -> list[ast.stmt]:
    """``np.tensordot(a, b, axes)`` -> an equivalent einsum.

    ``axes`` is an int K (contract the last K axes of ``a`` with the first K
    of ``b``) or a pair of axis lists. Default ``axes=2``.

    A non-Name operand (conv2d's sliced ``input[:, ki:ki+H_out, kj:kj+W_out,
    :]`` and partially-indexed ``weights[ki, kj]``) is spilled into a fresh
    scratch buffer first, the same ``materialize_operands`` pattern
    :func:`expand_einsum` uses -- under a distinct ``__td_`` prefix so its
    temps/copy-iterators cannot collide with einsum's own ``__es_`` spill
    once the mapped call below runs.
    """
    if len(args) < 2:
        raise NotImplementedError("tensordot needs 2 array args")
    prelude, (a, b) = materialize_operands(
        args[:2], shape_table, "__td_", local_dtypes=local_dtypes, fresh_local_allocs=fresh_local_allocs
    )
    if not (isinstance(a, ast.Name) and isinstance(b, ast.Name)):
        raise NotImplementedError("tensordot operands must be bare Names")
    ra = len(shape_table.get(a.id, ()))
    rb = len(shape_table.get(b.id, ()))
    if not ra or not rb:
        raise NotImplementedError("tensordot: operand shapes unknown")
    axes_node = args[2] if len(args) > 2 else axes_kwarg(kwargs)
    a_ax, b_ax = tensordot_axes(axes_node, ra, rb)
    letters = "abcdefghijklmnopqrstuvwxyz"
    a_spec = list(letters[:ra])
    b_spec = [None] * rb
    # Shared contraction letters: pair a_ax[i] <-> b_ax[i].
    nxt = ra
    for ca, cb in zip(a_ax, b_ax):
        b_spec[cb] = a_spec[ca]
    for i in range(rb):
        if b_spec[i] is None:
            b_spec[i] = letters[nxt]
            nxt += 1
    out_spec = [c for i, c in enumerate(a_spec) if i not in a_ax] + [c for i, c in enumerate(b_spec) if i not in b_ax]
    spec = f"{''.join(a_spec)},{''.join(b_spec)}->{''.join(out_spec)}"
    return prelude + expand_einsum(
        target, [const_(spec), a, b], shape_table, local_dtypes=local_dtypes, fresh_local_allocs=fresh_local_allocs
    )


def expand_inner(target: ast.expr, args: list[ast.expr], shape_table: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    """``np.inner(a, b)`` -> contract the LAST axis of each operand.

    Rank-1 x rank-1 is the plain dot product (routes to :func:`expand_dot`)."""
    if len(args) != 2:
        raise NotImplementedError("np.inner needs 2 args")
    a, b = args
    if not (isinstance(a, ast.Name) and isinstance(b, ast.Name)):
        raise NotImplementedError("np.inner operands must be bare Names")
    ra = len(shape_table.get(a.id, ()))
    rb = len(shape_table.get(b.id, ()))
    if ra == 1 and rb == 1:
        return expand_dot(target, args, shape_table)
    letters = "abcdefghijklmnopqrstuvwxyz"
    a_spec = list(letters[:ra])
    b_spec = list(letters[ra : ra + rb])
    b_spec[-1] = a_spec[-1]  # contract the last axis of each
    out_spec = a_spec[:-1] + b_spec[:-1]
    spec = f"{''.join(a_spec)},{''.join(b_spec)}->{''.join(out_spec)}"
    return expand_einsum(target, [const_(spec), a, b], shape_table)


def expand_vdot(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    local_dtypes: dict[str, str] | None = None,
) -> list[ast.stmt]:
    """``np.vdot(a, b)`` -> ``sum(conj(a) * b)`` over the flattened operands.
    The conjugate is emitted only when the first operand is complex (a no-op
    for reals, and invalid on a real scalar); real operands reduce to the plain
    dot product."""
    if len(args) != 2:
        raise NotImplementedError("np.vdot needs 2 args")
    a, b = args
    if not (isinstance(a, ast.Name) and isinstance(b, ast.Name)):
        raise NotImplementedError("np.vdot operands must be bare Names")
    shape = shape_table.get(a.id)
    if shape is None:
        raise NotImplementedError("np.vdot: operand shape unknown")
    if len(shape) != 1:
        raise NotImplementedError("np.vdot only supports rank-1 operands")
    is_complex = bool(local_dtypes and str(local_dtypes.get(a.id, "")).startswith("complex"))
    it = "__vd"
    a_elem: ast.expr = ast.Subscript(value=name_(a.id), slice=name_(it), ctx=ast.Load())
    if is_complex:
        a_elem = attr_call("np", "conj", [a_elem])
    prod = ast.BinOp(
        left=a_elem, op=ast.Mult(), right=ast.Subscript(value=name_(b.id), slice=name_(it), ctx=ast.Load())
    )
    body = [ast.AugAssign(target=store_(target.id), op=ast.Add(), value=prod)]
    return [ast.Assign(targets=[store_(target.id)], value=const_(0.0)), *wrap_for_loops([it], [shape[0]], body)]
