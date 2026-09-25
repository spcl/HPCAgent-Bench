"""Kept helpers: return shapes and the caller locals a helper call binds."""

import ast
import copy
from collections.abc import Iterator, Mapping

from hpcagent_bench.translators.numpyto_common.ir import ArrayDesc, ScalarDesc, SymbolDesc
from hpcagent_bench.translators.numpyto_common.lib_nodes import iter_extent_of
from hpcagent_bench.translators.numpyto_common.frontend.callsite import (
    caller_side_shape,
    held_before_table,
    shape_symbols,
    value_names,
)
from hpcagent_bench.translators.numpyto_common.frontend.helper_params import DescEntry, DescKey, infer_helper_params
from hpcagent_bench.translators.numpyto_common.frontend.helper_specialize import bind_call_constants, literal_call_arg
from hpcagent_bench.translators.numpyto_common.frontend.shapes import assigns_to, local_array_def, resolve_array_ref


def helper_return_array_shape(
    lhs: ast.expr | None, arr_by: dict[str, ArrayDesc], fn: ast.FunctionDef
) -> tuple[list[str] | None, str | None]:
    """When a captured helper's result is stored into an ARRAY target
    (``X = h(...)`` with X an array, or ``X[:, j] = h(...)``), return the returned
    array's ``(shape_strings, dtype)`` -- so the helper emits an out-param of that
    shape. A scalar / non-array target returns ``(None, None)`` (by-value path).

    Delegates to :func:`resolve_array_ref`, which chases the WHOLE alias chain -- a bare
    ``X`` target is not always a declared param or a direct ``np.zeros`` local: inlining a
    Form-3 helper rebinds its tail return through a renamed local (``X = __inl1_out`` where
    ``__inl1_out = np.zeros(...)``), and a single-hop check stops one alias short, misreading
    an array-returning helper's target as a scalar."""
    if not isinstance(lhs, (ast.Name, ast.Subscript)):
        return None, None
    res = resolve_array_ref(fn, lhs, arr_by)
    return (list(res[0]), res[1]) if res is not None else (None, None)


def name_assign(stmt: ast.stmt) -> bool:
    """Whether ``stmt`` binds exactly one bare Name."""
    return isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)


def statements_in_order(body: list[ast.stmt], nested: bool = False) -> Iterator[tuple[ast.stmt, bool]]:
    """Each statement in SOURCE order, nested blocks included, paired with whether it is nested.

    ``ast.walk`` is breadth-first, which is the wrong order for a forward extent sweep.
    """
    for stmt in body:
        yield stmt, nested
        if isinstance(stmt, (ast.If, ast.For, ast.While)):
            yield from statements_in_order(stmt.body, True)
            yield from statements_in_order(stmt.orelse, True)
        elif isinstance(stmt, ast.With):
            yield from statements_in_order(stmt.body, True)


def propagate_local_extents(hfn: ast.FunctionDef, table: dict[str, tuple[str, ...]]) -> None:
    """Extend ``table`` with each local of ``hfn`` that :func:`iter_extent_of` can size.

    Statement order matters: a local is sized against the names bound before it, so one sweep
    forward resolves a chain (``cumulative`` from ``x``, then ``seg`` from ``cumulative``). A name
    that stops resolving is dropped rather than left holding a stale extent.

    A local bound inside a branch counts too -- conv2d_gelu_global_avg_pool's ``_adaptive_avg_pool2d``
    reshapes into ``y`` under an ``if`` and returns a reduction over it, and skipping the block left
    the helper sized off its INPUT. Only when that is the name's ONLY binding: two branches sizing
    it differently would leave whichever ran last, and a wrong extent allocates a wrong buffer
    instead of declining.
    """
    stmts = [(st, nested) for st, nested in statements_in_order(hfn.body) if name_assign(st)]
    assigns = [(st, nested) for st, nested in stmts if isinstance(st, ast.Assign)]
    named = [(st, st.targets[0], nested) for st, nested in assigns if isinstance(st.targets[0], ast.Name)]
    bindings: dict[str, int] = {}
    for unused, target, nested_ in named:
        bindings[target.id] = bindings.get(target.id, 0) + 1
    # Only the nested locals a RETURN actually depends on, closed transitively over the nested
    # assignments. Sizing every statement in every loop body instead put resnet101's emit an order
    # of magnitude slower for extents no derivation reads.
    needed: set[str] = set()
    for node in ast.walk(hfn):
        if isinstance(node, ast.Return) and node.value is not None:
            needed |= {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
    for stmt, target, nested in reversed(named):
        if nested and target.id in needed:
            needed |= {n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name)}
    for stmt, target, nested in named:
        name = target.id
        if nested and (bindings.get(name, 0) != 1 or name not in needed):
            continue
        ext = iter_extent_of(stmt.value, table)
        if ext is None:
            table.pop(name, None)
        else:
            table[name] = tuple(ast.unparse(d) for d in ext)


def scalar_value_names(hfn: ast.FunctionDef, seed: set[str]) -> set[str]:
    """Names bound to a SCALAR in ``hfn``, seeded with its scalar and symbol PARAMETERS.

    Absence from the extent table conflates "this is a scalar" with "this is an array the sweep
    could not size", and reading a divisor as the second declines a helper whose return shape is
    fully known: ``_avgpool2d`` returns ``acc / (kh * kw)``, and ``kh``/``kw`` -- locals bound from
    scalar parameters -- made the whole return unresolvable, so the out-param fell back to a
    broadcast join over the ARGUMENTS and the caller allocated the pool's INPUT shape.

    Source order is enough to chain: each local is decided against the ones bound before it.
    """
    known = set(seed)
    for stmt, unused in statements_in_order(hfn.body):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        targets = target.elts if isinstance(target, ast.Tuple) else [target]
        if not all(isinstance(t, ast.Name) for t in targets):
            continue
        values = stmt.value.elts if isinstance(stmt.value, ast.Tuple) else [stmt.value] * len(targets)
        if len(values) != len(targets):
            continue
        for name_node, value in zip(targets, values):
            if all(n.id in known for n in ast.walk(value) if isinstance(n, ast.Name)):
                known.add(name_node.id)
    return known


def extent_operands_resolved(value: ast.expr, table: dict[str, tuple[str, ...]], scalars: set[str]) -> bool:
    """Whether every name the expression uses AS AN ARRAY has an extent in ``table``.

    Only the positions that carry an extent are checked -- a direct operand of an arithmetic
    BinOp, and a subscript base. A name in any other position is a scalar (mamba2's ``span``
    inside ``np.full((span, span), ...)``), and demanding an extent for it declines helpers that
    are perfectly resolvable. ``scalars`` names the ones that carry a value rather than an extent
    even in an operand position -- see :func:`scalar_value_names`.
    """
    operands: list[ast.expr] = []
    for node in ast.walk(value):
        if isinstance(node, ast.BinOp) and not isinstance(node.op, ast.MatMult):
            operands.extend([node.left, node.right])
        elif isinstance(node, ast.Subscript):
            operands.append(node.value)
    return not any(isinstance(op, ast.Name) and op.id not in table and op.id not in scalars for op in operands)


def target_shape_is_the_call_itself(
    fn: ast.FunctionDef, lhs: ast.expr | None, arr_by: dict[str, ArrayDesc], name: str
) -> bool:
    """Whether ``lhs``'s only definition is the very call whose result shape is being asked for.

    :func:`resolve_array_ref` chases a local to its first binding, and when that binding IS the
    helper call it falls through to a broadcast join over the call's ARGUMENTS. For an elementwise
    helper that join is the answer; for one that changes shape it is a guess, and a wrong one:
    nbody's ``acc = _acc_from_sep(dx, dy, dz, dist2, mass, G, softening)`` returns
    ``np.hstack((ax, ay, az))``, ``(N, 3)``, and the join over the ``(N, N)`` separations sized the
    out-param ``(N, N)``. The helper then wrote 3 columns into a buffer the caller read at stride
    ``N`` -- every read past the first row was another row's data, and the energies came out wrong
    with nothing failing to compile.

    Where this is true the callee's own body is the authority instead (see
    :func:`helper_return_shape_from_body`). A declared array, an allocation, or a subscript target
    all say something the call does not, and none of them answer True here.
    """
    if not isinstance(lhs, ast.Name) or lhs.id in arr_by:
        return False
    if local_array_def(fn, lhs.id, arr_by) is not None:
        return False
    assigns = assigns_to(fn, lhs.id)
    return (
        bool(assigns)
        and isinstance(assigns[0].value, ast.Call)
        and isinstance(assigns[0].value.func, ast.Name)
        and assigns[0].value.func.id == name
    )


def call_specialized_body(hfn: ast.FunctionDef, pnames: list[str], args: list[ast.expr]) -> ast.FunctionDef:
    """A COPY of ``hfn`` with this call site's literal arguments bound and the guards they decide gone.

    The return classification has to read the body this call site produces, not the generic one.
    ``_conv2d``'s two arms return different extents, and two return shapes retire the array-return
    path outright ("one pointer cannot carry both") -- at a call site where the guard is decided
    and exactly one arm survives. Classified by-value, an array return emits a function typed
    ``double`` that hands back a pointer, and the out-param buffer the caller allocates for it
    takes the shape of whatever else was in scope.

    A copy, because the real body is specialised further down the same pass, after several
    rewrites that must see the parameters this substitution would have removed.
    """
    probe = copy.deepcopy(hfn)
    bind_call_constants(probe, {pn: a for pn, a in zip(pnames, args) if literal_call_arg(a)})
    return probe


def helper_returns_rank0(
    hfn: ast.FunctionDef,
    pnames: list[str],
    args: list[ast.expr],
    arr_by: dict[str, ArrayDesc],
    sca_by: dict[str, ScalarDesc],
    sym_by: dict[str, SymbolDesc],
    fn: ast.FunctionDef | None = None,
) -> bool:
    """Whether every RETURN of ``hfn`` is PROVABLY rank 0 -- a reduction, array in and scalar out.

    :func:`helper_return_shape_from_body` answers ``None`` both for a scalar return and for a
    body it could not size, and a caller that has only a target-side GUESS to fall back on needs
    to know which. Provable means all three: the parameters sized (nothing is known about a body
    whose own arguments did not resolve), every return expression's array operands resolved, and
    the extent calculus then reporting no axes at all.
    """
    returns = [n.value for n in ast.walk(hfn) if isinstance(n, ast.Return) and n.value is not None]
    if not returns:
        return False
    arrays, scalars, symbols = infer_helper_params(pnames, args, arr_by, sca_by, sym_by, fn)
    if not arrays:
        return False
    scalar_names = scalar_value_names(hfn, {d.name for d in (*scalars, *symbols)})
    table = {a.name: tuple(str(s) for s in a.shape) for a in arrays}
    propagate_local_extents(hfn, table)
    if any(not extent_operands_resolved(value, table, scalar_names) for value in returns):
        return False
    return all(iter_extent_of(value, table) is None for value in returns)


def helper_return_shape_from_body(
    hfn: ast.FunctionDef,
    pnames: list[str],
    args: list[ast.expr],
    arr_by: dict[str, ArrayDesc],
    sca_by: dict[str, ScalarDesc],
    sym_by: dict[str, SymbolDesc],
    fn: ast.FunctionDef | None = None,
) -> tuple[list[str] | None, str | None]:
    """``(shape_strings, dtype)`` for a helper whose RETURN EXPRESSION is array-valued.

    The call-site target is the first authority on this, but it only exists when some call writes
    the result into a named array. A helper called solely as another call's argument has no such
    target, and defaulting to a by-value scalar return is not a neutral guess -- it produces a
    function typed ``double`` that returns a pointer.

    The helper's own parameters are enough to size it: their shapes come from the call site, and
    :func:`iter_extent_of` already resolves a return expression against them. Rank 0 means the
    return really is scalar, so ``(None, None)`` keeps the existing path.
    """
    returns = [n.value for n in ast.walk(hfn) if isinstance(n, ast.Return) and n.value is not None]
    if not returns:
        return None, None
    arrays, scalars, symbols = infer_helper_params(pnames, args, arr_by, sca_by, sym_by, fn)
    if not arrays:
        return None, None
    scalar_names = scalar_value_names(hfn, {d.name for d in (*scalars, *symbols)})
    table = {a.name: tuple(str(s) for s in a.shape) for a in arrays}
    # A return expression is built from the helper's own locals (mamba2's
    # ``return seg + np.triu(__full1, 1)``), not from its parameters directly, so sizing it needs
    # those locals too -- propagated forward, since each is sized against the ones before it.
    propagate_local_extents(hfn, table)
    if any(not extent_operands_resolved(value, table, scalar_names) for value in returns):
        # ``iter_extent_of`` answers a BinOp with the operand it COULD size when the other comes
        # back None. That is a serviceable broadcast hint and a wrong allocation: mamba2's
        # ``seg + np.triu(...)`` reported the triangle's ``(span, span)`` for a 4-D result, which
        # sizes the out-param two ranks short of what the body writes into it.
        return None, None
    extents = [iter_extent_of(value, table) for value in returns]
    resolved = [ext for ext in extents if ext is not None]
    if not extents or len(resolved) != len(extents):
        return None, None
    shapes = {tuple(ast.unparse(dim) for dim in ext) for ext in resolved}
    if len(shapes) != 1:
        # Two returns of different extents need two out-params; one pointer cannot carry both.
        return None, None
    shape = list(shapes.pop())
    # The result takes the dtype of the array operand it is computed from -- the same rule
    # ``helper_return_array_shape`` gets for free from the target it writes into.
    return shape, arrays[0].dtype


def structure_key(node: ast.AST) -> str:
    """Value fingerprint of a tree, positions included -- these are rewritten mid-sweep, so an
    ``id()`` key would answer for a tree that has moved on, and sites are ordered on position."""
    return ast.dump(node, include_attributes=True)


def desc_key(table: Mapping[str, DescEntry]) -> DescKey:
    """Value fingerprint of a descriptor table; the descriptors are flat dataclasses."""
    return tuple(sorted((name, repr(desc)) for name, desc in table.items()))


def helper_call_local_arrays(
    owner_fn: ast.FunctionDef,
    helper_defs: list[ast.FunctionDef],
    arr_by: dict[str, ArrayDesc],
    sca_by: dict[str, ScalarDesc],
    sym_by: dict[str, SymbolDesc],
) -> dict[str, ArrayDesc]:
    """``{local: ArrayDesc}`` for every one of ``owner_fn``'s locals bound to a USER-HELPER call.

    The shape is the CALLEE's own return, respelled in the caller's names. Nothing else resolves
    one: the local is not a declared array and not an allocation, so ``resolve_array_ref`` falls
    to ``shape_from_expression``, which reads an unresolved Call as ELEMENTWISE and answers with
    the broadcast join of its ARGUMENTS. ``h1 = _conv2d(x, w, ...)`` therefore came back with
    ``x``'s own ``(batch_size, in_channels, height, width)`` instead of the convolution's
    ``(batch_size, out_channels, oh, ow)`` -- and the two agree at preset S only because
    ``in_channels == out_channels`` there, so the batch-norm tail read a 6x6 buffer at 8x8 strides
    and returned a wrong number rather than failing.

    Bindings are read in SOURCE order and each resolved one joins the table the next resolves
    against, so a chain (``h2 = _batch_norm(h1, ...)``) settles in a single pass. A local bound
    more than once, a call whose arity does not match, and a return whose extents do not respell
    into names the caller holds are all left OUT: absent is what every consumer already handles,
    a guess is what this exists to stop.

    A chain is not only helper calls. conv_transpose3d_scale_batch_norm_global_avg_pool writes
    ``h2 = h1 * scale_factor`` between two of them, and conv2d_gelu_global_avg_pool's whole chain
    becomes plain locals once its first two helpers are inlined. Those links are carried here too
    -- same forward pass, extents through :func:`iter_extent_of` against the table built so far --
    because dropping one drops every helper downstream of it: the argument resolves to nothing, the
    parameter is typed by-value, and the helper indexes a double. This is a CLASSIFYING pass, not a
    guessing one: a statement is read only when every name it reads is already known to be an array
    or a scalar, so the broadcast-hint answer ``iter_extent_of`` gives for a half-resolved operand
    pair is never reached.
    """
    hdefs = {h.name: h for h in helper_defs}
    if not hdefs:
        return {}
    bindings: dict[str, int] = {}
    for node in ast.walk(owner_fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = bindings.get(target.id, 0) + 1
    held = (
        {s.name for s in sym_by.values()}
        | {a.name for a in arr_by.values()}
        | {s.name for s in sca_by.values()}
        | set(shape_symbols(arr_by.values()))
        | {a.arg for a in owner_fn.args.args}
    )
    sites = assigns_in_run_order(owner_fn)
    before = held_before_table(owner_fn)
    # Names already known NOT to carry an extent: the caller's own scalars and shape symbols, plus
    # every loop induction variable. A plain statement is read only when its operands are all in
    # here or in the array table; anything else leaves its target unclassified, which is what keeps
    # a half-resolved expression from being sized by whichever operand happened to answer.
    base_scalars = (
        {s.name for s in sca_by.values()}
        | {s.name for s in sym_by.values()}
        | set(shape_symbols(arr_by.values()))
        | {
            t.id
            for n in ast.walk(owner_fn)
            if isinstance(n, ast.For)
            for t in ast.walk(n.target)
            if isinstance(t, ast.Name)
        }
    )

    def sweep(rebound_ok: bool) -> tuple[dict[str, ArrayDesc], bool]:
        """One forward pass. ``rebound_ok`` also reads a name the body binds more than once,
        keeping its FIRST resolved shape; the flag returned says whether every such name's other
        bindings came back with that same shape, which is what makes the assumption a fixpoint
        rather than a guess."""
        known = dict(arr_by)
        found: dict[str, ArrayDesc] = {}
        scalar_names = set(base_scalars)
        per_binding: dict[str, list[tuple[str, ...]]] = {}
        for site in sites:
            if len(site.targets) != 1 or not isinstance(site.targets[0], ast.Name):
                continue
            name = site.targets[0].id
            call = site.value
            count = bindings.get(name, 0)
            if name in arr_by or count == 0 or (count > 1 and not rebound_ok):
                continue
            site_held = held | before.get(id(site), before[None])
            desc: ArrayDesc | None = None
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id in hdefs
                and not call.keywords
            ):
                desc = helper_call_local(call, hdefs, known, sca_by, sym_by, owner_fn, site_held, name)
            else:
                desc = plain_local_array(site, known, scalar_names, site_held)
                if desc is None and reads_no_array(call, known):
                    scalar_names.add(name)
            if desc is None:
                continue
            per_binding.setdefault(name, []).append(tuple(str(s) for s in desc.shape))
            if name not in found:
                known[name] = desc
                found[name] = desc
        settled = all(
            len(shapes) == bindings[name] and len(set(shapes)) == 1
            for name, shapes in per_binding.items()
            if bindings[name] > 1
        )
        return found, settled

    found, settled = sweep(True)
    # A rebound local whose bindings do NOT agree makes every extent read off the first one
    # suspect, so nothing from that pass is kept -- the strict pass, which never looked at a
    # rebound name, is the answer instead.
    return found if settled else sweep(False)[0]


def helper_call_local(
    call: ast.Call,
    hdefs: dict[str, ast.FunctionDef],
    known: dict[str, ArrayDesc],
    sca_by: dict[str, ScalarDesc],
    sym_by: dict[str, SymbolDesc],
    owner_fn: ast.FunctionDef,
    site_held: set[str],
    name: str,
) -> ArrayDesc | None:
    """The descriptor a caller local gets from the USER HELPER whose call binds it, or ``None``.

    The helper is asked with this site's literal arguments already bound into a COPY of its body,
    the same specialisation :func:`build_helper_kirs` performs before it reads any extent. Asked
    raw, resnet101's ``_conv2d`` binds ``oh`` twice -- once in the 1x1 branch, once in the general
    one -- and ``caller_side_symbol`` declines a symbol that is not bound exactly once, so every
    one of its hundred-odd specialisations went unresolved and each following layer's input with it.
    """
    hfn = hdefs[call.func.id]
    pnames = [a.arg for a in hfn.args.args]
    if len(call.args) != len(pnames):
        return None
    consts = {pn: a for pn, a in zip(pnames, call.args) if literal_call_arg(a)}
    if consts:
        hfn = copy.deepcopy(hfn)
        bind_call_constants(hfn, consts)
    shape, dtype = helper_return_shape_from_body(hfn, pnames, call.args, known, sca_by, sym_by, owner_fn)
    if shape is None:
        return None
    try:
        tokens = caller_side_shape(shape, site_held, pnames, call.args, hfn, call.func.id, known)
    except NotImplementedError:
        return None  # the callee sizes itself through more of its own locals than the chase follows
    free: set[str] = set()
    for token in tokens:
        free |= value_names(ast.parse(str(token), mode="eval"))
    if not free <= site_held:
        return None  # an extent the caller cannot name is not a shape the caller can be told
    return ArrayDesc(name=name, dtype=dtype, shape=tuple(tokens), is_output=False)


def assigns_in_run_order(fn: ast.FunctionDef) -> list[ast.Assign]:
    """Every ``name = ...`` in ``fn``, in the order the statements RUN.

    Not by position: inlining SPLICES a helper's statements into the caller verbatim, so they keep
    the callee DEF's line numbers and sort ahead of the caller statements they actually follow.
    conv_transpose3d's inlined ``_batch_norm`` tail carried line 34 into a body whose surrounding
    statements are at 136 and 146, so a forward pass over sorted positions read it before the
    local it is computed from and resolved nothing.
    """
    out: list[ast.Assign] = []

    def walk(body: list[ast.stmt]) -> None:
        for stmt in body:
            if isinstance(stmt, ast.Assign):
                out.append(stmt)
            elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While, ast.If)):
                walk(stmt.body)
                walk(stmt.orelse)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                walk(stmt.body)

    walk(fn.body)
    return out


def rhs_value_names(value: ast.expr) -> set[str]:
    """Every name a statement's RHS reads as a VALUE. A call's own Name callee (``int(...)``) and
    the numpy module are spellings, not operands, so neither has to be classified."""
    callees = {id(n.func) for n in ast.walk(value) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return {
        n.id
        for n in ast.walk(value)
        if isinstance(n, ast.Name)
        and isinstance(n.ctx, ast.Load)
        and id(n) not in callees
        and n.id not in ("np", "numpy", "math")
    }


def reads_no_array(value: ast.expr, known: dict[str, ArrayDesc]) -> bool:
    """Whether the RHS reads nothing that carries an extent -- so its target is a scalar rather
    than an array this pass merely failed to size (``head_dim = embed_dim // num_heads``)."""
    return not (rhs_value_names(value) & set(known))


def plain_local_array(
    site: ast.Assign, known: dict[str, ArrayDesc], scalar_names: set[str], site_held: set[str]
) -> ArrayDesc | None:
    """The descriptor for ``name = <numpy expression>`` when the caller can be TOLD that shape.

    Declines unless every name the expression reads is already classified -- an array in ``known``
    or a scalar -- because :func:`iter_extent_of` answers a BinOp with whichever operand it could
    size when the other comes back ``None``, and that answer is a wrong allocation, not a hint.
    Declines too when no operand supplies a dtype: the width decides the buffer, and there is
    nothing to read it off.
    """
    value = site.value
    reads = rhs_value_names(value)
    if not reads or not reads <= (set(known) | scalar_names):
        return None
    table = {n: tuple(str(s) for s in a.shape) for n, a in known.items()}
    ext = iter_extent_of(value, table)
    if not ext:
        return None
    tokens = [ast.unparse(dim) for dim in ext]
    free: set[str] = set()
    for token in tokens:
        free |= value_names(ast.parse(token, mode="eval"))
    if not free <= site_held:
        return None  # an extent the caller cannot name is not a shape the caller can be told
    # AST order, not set order: the descriptor has to come out the same on every run.
    dtype = next(
        (known[n.id].dtype for n in ast.walk(value) if isinstance(n, ast.Name) and n.id in known),
        None,
    )
    if dtype is None:
        return None
    return ArrayDesc(name=site.targets[0].id, dtype=dtype, shape=tuple(tokens), is_output=False)
