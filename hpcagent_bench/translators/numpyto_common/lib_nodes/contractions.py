"""Tensor contractions: einsum, tensordot, inner, vdot."""

import ast
import copy
from collections.abc import Sequence

import numpy as np

from hpcagent_bench.translators.numpyto_common.ast_build import name_, store_
from hpcagent_bench.translators.numpyto_common.lib_nodes.blas import BLAS_GEMM_STRIDED_MARKER, expand_dot
from hpcagent_bench.translators.numpyto_common.lib_nodes.call_args import (
    axes_kwarg,
    parse_einsum_subscripts,
    tensordot_axes,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import iter_extent_of
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import (
    alloc_marker,
    attr_call,
    const_,
    const_or_name,
    flat_index_,
    mul_exts,
    name_id,
    wrap_for_loops,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.scalarize import scalarize_at_iters

__all__ = [
    "EINSUM_TEMP",
    "OP_SPILL_TEMP",
    "PATH_SYMBOL_EXTENT",
    "EinsumContraction",
    "einsum_letter_extents",
    "einsum_loop_nest",
    "einsum_path",
    "expand_einsum",
    "expand_einsum_ellipsis",
    "expand_inner",
    "expand_tensordot",
    "expand_vdot",
    "explicit_einsum_spec",
    "gemm_dtype_ok",
    "materialize_operands",
]


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
    for sub, r in zip(ins, ranks, strict=False):
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

#: Counter for the intermediates and relayouts an einsum's contraction adds (:class:`EinsumContraction`).
#: Reset per translation unit by :func:`reset_temp_counters`.
EINSUM_TEMP = [0]


def materialize_operands(
    operands: Sequence[ast.expr],
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
        op_ext = iter_extent_of(op, shape_table)
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
        cp_nodes: list[ast.expr] = [name_(c) for c in cp_iters]
        cp_slot = cp_nodes[0] if len(cp_nodes) == 1 else ast.Tuple(elts=cp_nodes, ctx=ast.Load())
        cp_src = scalarize_at_iters(op, cp_nodes, shape_table)
        cp_dst = ast.Subscript(value=name_(tmp), slice=cp_slot, ctx=ast.Store())
        prelude.append(alloc_marker(tmp))
        prelude.extend(wrap_for_loops(cp_iters, list(tmp_shape), [ast.Assign(targets=[cp_dst], value=cp_src)]))
        out.append(name_(tmp))
    return prelude, out


def expand_einsum(
    target: ast.Name,
    args: Sequence[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    local_dtypes: dict[str, str] | None = None,
    fresh_local_allocs: dict[str, tuple[str, ...]] | None = None,
    blas_float: str | None = None,
    extent_values: dict[str, int] | None = None,
    blas_complex: bool = False,
) -> list[ast.stmt]:
    """Lower ``np.einsum(subscripts, *operands)``. Three or more operands contract pairwise along
    numpy's ``optimize='optimal'`` path (:func:`einsum_path`), each intermediate a fresh temp; a pair
    becomes a strided gemm (:func:`contract_gemm`) when ``blas_float`` names the kernel float a BLAS
    target computes in and the operands are of it (or, under ``blas_complex``, all of one complex dtype),
    else a loop nest (:func:`einsum_loop_nest`).

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
        spec = explicit_einsum_spec(spec, operands, shape_table)
    inputs, output = parse_einsum_subscripts(spec)
    if len(inputs) != len(operands):
        raise NotImplementedError("einsum operand count mismatches subscripts")
    operand_names: list[str] = []
    for op in operands:
        if not isinstance(op, ast.Name):
            raise NotImplementedError("einsum operands must be bare Names")
        operand_names.append(op.id)
    letter_extent = einsum_letter_extents(inputs, operand_names, shape_table)
    dtypes = local_dtypes if local_dtypes is not None else {}
    tags = {dtypes.get(n) for n in operand_names}
    gemm = blas_float is not None and (
        all(gemm_dtype_ok(t, blas_float) for t in tags)
        or (blas_complex and len(tags) == 1 and str(next(iter(tags))).startswith("complex"))
    )
    contraction = EinsumContraction(letter_extent, shape_table, dtypes, fresh_local_allocs, gemm)
    pending = list(zip(operand_names, inputs, strict=True))
    body: list[ast.stmt] = []
    for step in einsum_path(inputs, output, letter_extent, extent_values or {}):
        group = [pending[i] for i in step]
        pending = [p for i, p in enumerate(pending) if i not in step]
        if not pending:
            body += contraction.contract(group, target.id, output)
            break
        # An intermediate keeps the letters a later step or the output still reads, in first-seen order.
        later = set(output).union(*(letters for _, letters in pending))
        kept = "".join(dict.fromkeys(c for _, letters in group for c in letters if c in later))
        temp = contraction.fresh_temp("tmp", kept, [name for name, _ in group])
        body += contraction.contract(group, temp, kept)
        pending.append((temp, kept))
    return prelude + body


#: Extent the path planner assumes for a dimension it cannot read as an integer: the sizes are runtime symbols,
#: and a symbol is a problem size, so it plans as a large one against the literal (physical) extents.
PATH_SYMBOL_EXTENT = 1024


def einsum_path(
    inputs: list[str], output: str, letter_extent: dict[str, str], extent_values: dict[str, int]
) -> list[tuple[int, ...]]:
    """numpy's ``optimize='optimal'`` contraction order (its default memory limit, the largest operand or output),
    planned at :data:`PATH_SYMBOL_EXTENT` for every symbolic extent. Each step names positions in the pending list,
    which drops the step's operands and appends its result, as ``np.einsum_path`` reports it."""
    if len(inputs) <= 2:
        return [tuple(range(len(inputs)))]

    def size(token: str) -> int:
        token = token.strip()
        if token.lstrip("-").isdigit():
            return int(token)
        return extent_values.get(token, PATH_SYMBOL_EXTENT)

    sizes = {c: size(tok) for c, tok in letter_extent.items()}
    # Zero-stride views: the planner reads only the shapes, so nothing the size of an operand is allocated.
    views = [np.broadcast_to(np.float64(0), tuple(sizes[c] for c in letters)) for letters in inputs]
    try:
        path = np.einsum_path(f"{','.join(inputs)}->{output}", *views, optimize="optimal")[0]
    except ValueError:  # a placeholder shape past the address space: keep the single loop nest
        return [tuple(range(len(inputs)))]
    return [tuple(step) for step in path[1:]]


def gemm_dtype_ok(dtype: str | None, blas_float: str) -> bool:
    """An operand a ``blas_float`` gemm reads as it is: untagged (the kernel float) or tagged ``blas_float``."""
    return dtype is None or dtype == blas_float


class EinsumContraction:
    """Lowers the steps of one einsum: a pair to a strided gemm or a loop nest, a larger group to a loop nest.

    Letters index the loop variables ``__es_<letter>``; every buffer the lowering adds is a fresh temp
    registered in the shape table (and ``fresh_local_allocs``) with an allocation marker ahead of its first write.
    """

    __slots__ = ("dtypes", "fresh_local_allocs", "gemm", "letter_extent", "shape_table")

    def __init__(
        self,
        letter_extent: dict[str, str],
        shape_table: dict[str, tuple[str, ...]],
        dtypes: dict[str, str],
        fresh_local_allocs: dict[str, tuple[str, ...]] | None,
        gemm: bool,
    ) -> None:
        self.letter_extent = letter_extent
        self.shape_table = shape_table
        self.dtypes = dtypes
        self.fresh_local_allocs = fresh_local_allocs
        #: The target renders this einsum's pairs as gemms; False: every step keeps its loop nest.
        self.gemm = gemm

    def fresh_temp(self, role: str, letters: str, sources: Sequence[str]) -> str:
        """A new ``__es_<role><n>`` buffer over ``letters``, of the sources' promoted dtype: a complex tag wins, an
        untagged source (the kernel float) leaves the buffer untagged, all-tagged sources promote as numpy does."""
        EINSUM_TEMP[0] += 1
        temp = f"__es_{role}{EINSUM_TEMP[0]}"
        shape = tuple(self.letter_extent[c] for c in letters)
        self.shape_table[temp] = shape
        if self.fresh_local_allocs is not None:
            self.fresh_local_allocs[temp] = shape
        tags = [self.dtypes[source] for source in sources if source in self.dtypes]
        complex_tag = next((t for t in tags if t.startswith("complex")), None)
        if complex_tag is not None:
            self.dtypes[temp] = complex_tag
        elif tags and len(tags) == len(sources):
            # One tag keeps its own spelling (``bool_``); mixed integer/float tags promote as numpy does.
            self.dtypes[temp] = tags[0] if len(set(tags)) == 1 else str(np.result_type(*tags))
        return temp

    def contract(self, group: list[tuple[str, str]], out: str, out_letters: str) -> list[ast.stmt]:
        """``out[out_letters] = einsum(group)``, with the allocation marker of a temp ``out``."""
        marker = [alloc_marker(out)] if out.startswith("__es_") else []
        if len(group) == 2 and self.gemm:
            gemm = self.contract_gemm(group[0], group[1], out, out_letters)
            if gemm is not None:
                return marker + gemm
        return marker + einsum_loop_nest(group, out, out_letters, self.letter_extent)

    def extent(self, letters: str) -> ast.expr:
        """The product of ``letters``' extents (``1`` for none)."""
        return mul_exts(const_or_name(self.letter_extent[c]) for c in letters)

    def permuted(self, name: str, letters: str, want: str) -> tuple[str, list[ast.stmt]]:
        """``name`` laid out as ``want``: itself when it already is, else a fresh temp and the copy filling it."""
        if letters == want:
            return name, []
        temp = self.fresh_temp("perm", want, [name])
        return temp, [alloc_marker(temp), *einsum_loop_nest([(name, letters)], temp, want, self.letter_extent)]

    def contract_gemm(
        self, a: tuple[str, str], b: tuple[str, str], out: str, out_letters: str
    ) -> list[ast.stmt] | None:
        """``out = a . b`` as gemms: one per index of the letters all three share (the batch), over
        ``(batch, M, K) x (batch, K, N) -> (batch, M, N)`` with M the letters only ``a`` and ``out`` carry, N
        only ``b`` and ``out``, K only the two operands. An operand in neither ``(batch, M, K)`` nor
        ``(batch, K, M)`` order (for ``b``, ``(batch, K, N)`` / ``(batch, N, K)``) is first copied into it;
        an ``out`` in neither ``(batch, M, N)`` nor ``(batch, N, M)`` order is computed into a temp and copied.

        None when the pair is not a gemm: a repeated letter, a letter summed within one operand, no
        contracted letter (a broadcast product) or nothing but batch letters (a batched dot product).
        """
        (an, sa), (bn, sb) = a, b
        if any(len(set(s)) != len(s) for s in (sa, sb, out_letters)):
            return None
        if not set(out_letters) <= set(sa) | set(sb) or not (set(sa) ^ set(sb)) <= set(out_letters):
            return None
        batch = "".join(c for c in out_letters if c in sa and c in sb)
        k = "".join(c for c in sa if c in sb and c not in out_letters)
        m_set = {c for c in out_letters if c in sa and c not in sb}
        n_set = {c for c in out_letters if c in sb and c not in sa}
        if not k or not (m_set or n_set):
            return None
        stmts: list[ast.stmt] = []
        m = "".join(c for c in out_letters if c in m_set)
        n = "".join(c for c in out_letters if c in n_set)
        if out_letters == batch + n + m and out_letters != batch + m + n:
            # out holds C transposed: C^T = B^T A^T, the same gemm with the operands' roles swapped.
            (an, sa), (bn, sb), m, n = (bn, sb), (an, sa), n, m
        final: tuple[str, str] | None = None
        if out_letters != batch + m + n:
            final = (out, out_letters)
            out = self.fresh_temp("gemm", batch + m + n, [an, bn])
            stmts.append(alloc_marker(out))
        trans_a = sa == batch + k + m and sa != batch + m + k
        if not trans_a:
            an, copy_a = self.permuted(an, sa, batch + m + k)
            stmts += copy_a
        trans_b = sb == batch + n + k and sb != batch + k + n
        if not trans_b:
            bn, copy_b = self.permuted(bn, sb, batch + k + n)
            stmts += copy_b
        em, en, ek = self.extent(m), self.extent(n), self.extent(k)
        batch_iters = [f"__es_{c}" for c in batch]
        flat = flat_index_(batch_iters, tuple(self.letter_extent[c] for c in batch)) if batch else const_(0)

        def offset(per_batch: ast.expr) -> ast.expr:
            return ast.BinOp(left=copy.deepcopy(flat), op=ast.Mult(), right=per_batch) if batch else const_(0)

        gemm = ast.Call(
            func=name_(BLAS_GEMM_STRIDED_MARKER),
            args=[
                name_(an),
                name_(bn),
                name_(out),
                em,
                en,
                ek,
                const_(int(trans_a)),
                const_(int(trans_b)),
                copy.deepcopy(em if trans_a else ek),
                copy.deepcopy(ek if trans_b else en),
                copy.deepcopy(en),
                offset(ast.BinOp(left=copy.deepcopy(em), op=ast.Mult(), right=copy.deepcopy(ek))),
                offset(ast.BinOp(left=copy.deepcopy(ek), op=ast.Mult(), right=copy.deepcopy(en))),
                offset(ast.BinOp(left=copy.deepcopy(em), op=ast.Mult(), right=copy.deepcopy(en))),
            ],
            keywords=[],
        )
        # A bare Expr, as the 2-D gemm marker is (see hoist_matmul): an Assign would be wrapped elementwise.
        stmts += wrap_for_loops(batch_iters, [self.letter_extent[c] for c in batch], [ast.Expr(value=gemm)])
        if final is not None:
            stmts += einsum_loop_nest([(out, batch + m + n)], final[0], final[1], self.letter_extent)
        return stmts


def einsum_loop_nest(
    group: Sequence[tuple[str, str]], out: str, out_letters: str, letter_extent: dict[str, str]
) -> list[ast.stmt]:
    """``out[out_letters] = einsum(group)`` as ONE loop nest. Output indices become nested loops over the
    result; indices summed away (in the inputs but not the output) become inner accumulation loops. Body:
    ``out[out_idx] (+)= prod(operand[its idx letters])``.

    Handles N operands and arbitrary index letters, including a letter repeated
    within one operand (``ii`` -> diagonal ``A[i, i]``). One path subsumes
    matmul ``ij,jk->ik``, transpose ``ij->ji``, trace ``ii->``, diagonal
    ``ii->i``, outer ``i,j->ij`` and sum ``ij->``.
    """
    out_letters_list = list(out_letters)
    used = dict.fromkeys(c for _, letters in group for c in letters)
    sum_letters = [c for c in used if c not in out_letters_list]
    # Per-letter loop variable.
    var_of = {c: f"__es_{c}" for c in letter_extent}

    def subscript_(name: str, spec: str) -> ast.expr:
        idx: list[ast.expr] = [name_(var_of[c]) for c in spec]
        sl = idx[0] if len(idx) == 1 else ast.Tuple(elts=idx, ctx=ast.Load())
        return ast.Subscript(value=name_(name), slice=sl, ctx=ast.Load())

    # Product of every operand scalarised at its index letters.
    (first, first_spec), *rest = group
    product: ast.expr = subscript_(first, first_spec)
    for name, spec in rest:
        product = ast.BinOp(left=product, op=ast.Mult(), right=subscript_(name, spec))

    # Output write target.
    out_store: ast.Name | ast.Subscript
    if out_letters_list:
        out_idx: list[ast.expr] = [name_(var_of[c]) for c in out_letters_list]
        out_sl = out_idx[0] if len(out_idx) == 1 else ast.Tuple(elts=out_idx, ctx=ast.Load())
        out_store = ast.Subscript(value=name_(out), slice=out_sl, ctx=ast.Store())
    else:
        out_store = store_(out)  # scalar result (trace / full sum)

    # Inner: accumulate the product over the summed letters.
    if sum_letters:
        body: list[ast.stmt] = [ast.AugAssign(target=out_store, op=ast.Add(), value=product)]
        body = wrap_for_loops([var_of[c] for c in sum_letters], [letter_extent[c] for c in sum_letters], body)
        zero = ast.Assign(targets=[copy.deepcopy(out_store)], value=const_(0.0))
        inner: list[ast.stmt] = [zero, *body]
    else:
        inner = [ast.Assign(targets=[copy.deepcopy(out_store)], value=product)]

    if out_letters_list:
        return wrap_for_loops(
            [var_of[c] for c in out_letters_list], [letter_extent[c] for c in out_letters_list], inner
        )
    return inner


def explicit_einsum_spec(spec: str, operands: Sequence[ast.expr], shape_table: dict[str, tuple[str, ...]]) -> str:
    """``spec`` with ``...`` expanded to explicit letters from each operand's rank (the parser stays
    ellipsis-free)."""
    ranks = []
    for op in operands:
        if not isinstance(op, ast.Name) or shape_table.get(op.id) is None:
            raise NotImplementedError("einsum ellipsis needs bare-Name operands with known shape")
        ranks.append(len(shape_table[op.id]))
    return expand_einsum_ellipsis(spec, ranks)


def einsum_letter_extents(
    inputs: list[str], operand_names: list[str], shape_table: dict[str, tuple[str, ...]]
) -> dict[str, str]:
    """Every index letter's extent symbol, from the first operand that uses it."""
    letter_extent: dict[str, str] = {}
    for spec, name in zip(inputs, operand_names, strict=False):
        shape = shape_table.get(name)
        if shape is None or len(shape) != len(spec):
            raise NotImplementedError(f"einsum: shape of {name!r} unknown / rank mismatch")
        for letter, dim in zip(spec, shape, strict=False):
            letter_extent.setdefault(letter, dim)
    return letter_extent


def expand_tensordot(
    target: ast.Name,
    args: Sequence[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
    local_dtypes: dict[str, str] | None = None,
    fresh_local_allocs: dict[str, tuple[str, ...]] | None = None,
    blas_float: str | None = None,
    blas_complex: bool = False,
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
    b_spec = [""] * rb
    # Shared contraction letters: pair a_ax[i] <-> b_ax[i].
    nxt = ra
    for ca, cb in zip(a_ax, b_ax, strict=False):
        b_spec[cb] = a_spec[ca]
    for i in range(rb):
        if not b_spec[i]:
            b_spec[i] = letters[nxt]
            nxt += 1
    out_spec = [c for i, c in enumerate(a_spec) if i not in a_ax] + [c for i, c in enumerate(b_spec) if i not in b_ax]
    spec = f"{''.join(a_spec)},{''.join(b_spec)}->{''.join(out_spec)}"
    return prelude + expand_einsum(
        target,
        [const_(spec), a, b],
        shape_table,
        local_dtypes=local_dtypes,
        fresh_local_allocs=fresh_local_allocs,
        blas_float=blas_float,
        blas_complex=blas_complex,
    )


def expand_inner(target: ast.Name, args: Sequence[ast.expr], shape_table: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
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
    target: ast.Name,
    args: Sequence[ast.expr],
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
    body: list[ast.stmt] = [ast.AugAssign(target=store_(target.id), op=ast.Add(), value=prod)]
    return [ast.Assign(targets=[store_(target.id)], value=const_(0.0)), *wrap_for_loops([it], [shape[0]], body)]
