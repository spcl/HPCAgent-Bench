"""Array constructors: copy, linspace, arange, fromfunction, meshgrid, eye."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import iter_extent_of_
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import (
    alloc_marker,
    const_,
    const_int,
    name_,
    store_,
    wrap_for_loops,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.scalarize import scalarize_at_iters


def expand_copy(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    local_dtypes: dict[str, str] | None = None,
) -> list[ast.stmt]:
    """``out = np.copy(a)`` -> elementwise copy loop into the LHS. Source may be
    a bare Name (``np.copy(a)``) or an array-valued Subscript (``grid[0].copy()``
    -- a row lowered into a fresh local); for the Subscript case the iteration
    extent comes from the *result* shape (un-indexed axes) via
    ``iter_extent_of_``, scalarized per element like the elementwise expanders.
    """
    if not args:
        raise NotImplementedError("np.copy needs an operand")
    src = args[0]
    shape = iter_extent_of_(src, shape_table)
    if not shape:
        raise NotImplementedError("np.copy: source shape unknown")
    # Register + allocate the fresh target, mirroring the matmul/linalg expanders.
    # Without the ``__hpcagent_bench_zeros__`` marker, a copy target with a RUNTIME
    # (symbolic) shape -- eigh Jacobi's ``Cm = np.ascontiguousarray(a)`` or
    # ``__eigh<k>_a = np.ascontiguousarray(Linv @ h_sub @ Linv.T)`` -- is declared
    # a null pointer and never malloc'd, so the copy loop writes through NULL.
    # The emitter treats a static-shape target's marker as a no-op stack
    # declaration, so emitting it unconditionally is safe for both.
    shape_table.setdefault(target.id, tuple(ast.unparse(e) for e in shape))
    if local_dtypes is not None:
        src_name = None
        if isinstance(src, ast.Name):
            src_name = src.id
        elif isinstance(src, ast.Subscript):
            base = src.value
            while isinstance(base, ast.Subscript):
                base = base.value
            if isinstance(base, ast.Name):
                src_name = base.id
        if src_name:
            src_dt = local_dtypes.get(src_name)
            if src_dt:
                local_dtypes[target.id] = src_dt
    iters = [f"__r{i}" for i in range(len(shape))]
    iter_nodes = [name_(i) for i in iters]
    idx = iter_nodes[0] if len(iters) == 1 else ast.Tuple(elts=iter_nodes, ctx=ast.Load())
    sub_src = scalarize_at_iters(src, iter_nodes, shape_table)
    sub_dst = ast.Subscript(value=name_(target.id), slice=idx, ctx=ast.Store())
    body = [ast.Assign(targets=[sub_dst], value=sub_src)]
    return [alloc_marker(target.id)] + wrap_for_loops(iters, shape, body)


def expand_linspace(target: ast.expr, args: list[ast.expr], shape_table: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    """``out = np.linspace(start, stop, n)`` -> ``for i in range(n): out[i] =
    (stop - start) / max(n - 1, 1) * i + start``, then ``out[n - 1] = stop`` when ``n > 1``.

    Both details are numpy's arithmetic, not cosmetics. numpy divides FIRST and scales the index by
    that step (``y = arange(num) * step; y += start``); folding the division to the end --
    ``start + span * i / div`` -- is a different rounding, and it was off by an ulp on the very
    first grid this expansion was used for. numpy then PINS the last sample to ``stop``, because
    the computed one need not land on it exactly. An ulp is not a rounding nit wherever the grid
    seeds an iteration: mandelbrot's escape-time map turned 4.4e-16 at the seed into 1.3 in the
    result. ``max(n - 1, 1)`` is numpy's divisor too, so ``np.linspace(start, stop, 1)`` returns
    ``[start]`` rather than dividing by zero -- which is also why the endpoint pin is guarded:
    at ``n == 1`` numpy keeps ``start`` and pinning would overwrite it with ``stop``.
    """
    if len(args) != 3:
        raise NotImplementedError("np.linspace needs (start, stop, n)")
    start, stop, n = args
    span = ast.BinOp(left=copy.deepcopy(stop), op=ast.Sub(), right=copy.deepcopy(start))
    denom = ast.Call(
        func=name_("max"),
        args=[ast.BinOp(left=copy.deepcopy(n), op=ast.Sub(), right=const_(1)), const_(1)],
        keywords=[],
    )
    step = ast.BinOp(left=span, op=ast.Div(), right=denom)
    expr = ast.BinOp(
        left=ast.BinOp(left=step, op=ast.Mult(), right=name_("__i")), op=ast.Add(), right=copy.deepcopy(start)
    )
    body = [
        ast.Assign(targets=[ast.Subscript(value=name_(target.id), slice=name_("__i"), ctx=ast.Store())], value=expr)
    ]
    last = ast.Subscript(
        value=name_(target.id), slice=ast.BinOp(left=copy.deepcopy(n), op=ast.Sub(), right=const_(1)), ctx=ast.Store()
    )
    return [
        ast.For(
            target=store_("__i"),
            iter=ast.Call(func=name_("range"), args=[copy.deepcopy(n)], keywords=[]),
            body=body,
            orelse=[],
        ),
        ast.If(
            test=ast.Compare(left=copy.deepcopy(n), ops=[ast.Gt()], comparators=[const_(1)]),
            body=[ast.Assign(targets=[last], value=copy.deepcopy(stop))],
            orelse=[],
        ),
    ]


def arange_count(args: list[ast.expr]) -> ast.expr:
    """Element count of ``np.arange(*args)``: ``ceil(span / step)``, clamped at zero.

    The obvious ``(span + step - 1) // step`` is a POSITIVE-STEP identity. Under a negative step
    it over-counts -- ``np.arange(10, 0, -1)`` came out 12, so the iota loop wrote 12 elements
    into an array sized ``stop - start`` = -10: a negative C VLA (which does not compile) and a
    zero-length Fortran array the loop then ran off the end of, returning garbage that graded
    green. ``-((-span) // step)`` is ceil for either sign, given floor division.

    Constant arguments fold here, so the common literal ``arange`` keeps a plain integer extent
    rather than an expression a Fortran declaration would have to be able to evaluate.

    Shared with the shape inference in :class:`~hpcagent_bench.translators.numpyto_common.lib_nodes.call_hoist.CallHoister` so the extent an array is
    declared with and the trip count the loop runs cannot disagree -- disagreeing is what turned
    a wrong count into an out-of-bounds write."""
    if len(args) == 1:
        span, step = args[0], None
    elif len(args) == 2:
        span, step = ast.BinOp(left=args[1], op=ast.Sub(), right=args[0]), None
    elif len(args) == 3:
        span, step = ast.BinOp(left=args[1], op=ast.Sub(), right=args[0]), args[2]
    else:
        raise NotImplementedError("np.arange needs 1-3 args")
    literals = [const_int(a) for a in args]  # accepts a negated literal: step=-1 is UnaryOp(USub, 1)
    if all(v is not None for v in literals):
        start, stop = (0, literals[0]) if len(args) == 1 else (literals[0], literals[1])
        istep = literals[2] if len(args) == 3 else 1
        if istep == 0:
            raise NotImplementedError("np.arange step must be nonzero")
        return const_(max(0, -((start - stop) // istep)))
    # Unit step (the 1- and 2-arg forms, or an explicit step of 1): the count IS the span, so keep
    # the plain expression rather than wrapping it in the ceil form.
    if step is None or const_int(step) == 1:
        return span
    # -((-span) // step): ceil for either sign, since // is emitted as the flooring int_floor.
    return ast.UnaryOp(
        op=ast.USub(), operand=ast.BinOp(left=ast.UnaryOp(op=ast.USub(), operand=span), op=ast.FloorDiv(), right=step)
    )


def expand_arange(target: ast.expr, args: list[ast.expr], shape_table: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    """``out = np.arange(stop)`` -> ``for i in range(stop): out[i] = i``;
    ``np.arange(start, stop)`` -> ``out[i] = start + i``;
    ``np.arange(start, stop, step)`` -> ``out[i] = start + i*step`` over
    :func:`arange_count` elements. The iota value casts to ``out``'s declared dtype on
    assignment. Mirrors :func:`expand_linspace`."""
    if len(args) == 1:
        start, step = const_(0), None
    elif len(args) == 2:
        start, step = args[0], None
    elif len(args) == 3:
        start, step = args[0], args[2]
    else:
        raise NotImplementedError("np.arange needs 1-3 args")
    count = arange_count(args)
    # value(i) = start + i*step  (step omitted -> +i)
    idx = name_("__i")
    scaled = idx if step is None else ast.BinOp(left=idx, op=ast.Mult(), right=step)
    value = scaled if (len(args) == 1) else ast.BinOp(left=start, op=ast.Add(), right=scaled)
    body = [
        ast.Assign(targets=[ast.Subscript(value=name_(target.id), slice=name_("__i"), ctx=ast.Store())], value=value)
    ]
    return [
        ast.For(
            target=store_("__i"), iter=ast.Call(func=name_("range"), args=[count], keywords=[]), body=body, orelse=[]
        )
    ]


class RenameNames(ast.NodeTransformer):
    """Rename bare ``Name`` ids per a mapping (used to bind a fromfunction
    lambda's parameters to the loop iteration variables)."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.mapping:
            return ast.copy_location(ast.Name(id=self.mapping[node.id], ctx=node.ctx), node)
        return node


def expand_fromfunction(
    target: ast.expr, args: list[ast.expr], shape_table: dict[str, tuple[str, ...]]
) -> list[ast.stmt]:
    """``out = np.fromfunction(lambda i, j: f(i, j), (N, M))`` -> ``for i in
    range(N): for j in range(M): out[i, j] = f(i, j)``. The lambda body is
    inlined per-element with its parameters bound to the loop iters, realising
    the lambda as the loop body rather than a separate callable. Captured free
    variables are left untouched."""
    if len(args) < 2 or not isinstance(args[0], ast.Lambda):
        raise NotImplementedError("np.fromfunction needs (lambda, shape)")
    lam, shape_node = args[0], args[1]
    params = [a.arg for a in lam.args.args]
    shape_elts = list(shape_node.elts) if isinstance(shape_node, (ast.Tuple, ast.List)) else [shape_node]
    if len(params) != len(shape_elts):
        raise NotImplementedError("np.fromfunction: lambda arity != shape rank")
    iters = [f"__ff{i}" for i in range(len(params))]
    body_expr = RenameNames(dict(zip(params, iters))).visit(copy.deepcopy(lam.body))
    slot_elts = [name_(v) for v in iters]
    slot = slot_elts[0] if len(iters) == 1 else ast.Tuple(elts=slot_elts, ctx=ast.Load())
    body = [ast.Assign(targets=[ast.Subscript(value=name_(target.id), slice=slot, ctx=ast.Store())], value=body_expr)]
    return wrap_for_loops(iters, shape_elts, body)


#: Synthetic keyword the tuple-unpack in lowering.py attaches to each split
#: ``np.meshgrid`` call, telling this expander which output array it builds.
#: numpy's ``meshgrid`` has no such keyword, so the name is unambiguous.
MESHGRID_AXIS_KW = "__meshgrid_axis__"


def expand_meshgrid(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
) -> list[ast.stmt]:
    """Emit ONE broadcast output of ``np.meshgrid(a0, a1, ..., a_{k-1})``. Given
    1-D inputs of lengths ``(N0, ..., N_{k-1})``: ``indexing='ij'`` gives every
    output shape ``(N0, ..., N_{k-1})`` with ``out_d[i0, ..., i_{k-1}] =
    a_d[i_d]``; ``indexing='xy'`` (numpy's default) swaps axes 0 and 1, so the
    output shape is ``(N1, N0, N2, ...)`` and output ``d`` varies with input axis
    ``perm[d]``.

    A meshgrid call returns a TUPLE of arrays; the lowering-side unpack splits it
    into one call per output and marks each with :data:`MESHGRID_AXIS_KW` so this
    expander builds that output's broadcast-copy loop nest. Each input must be a
    1-D array of known length."""
    kwargs = kwargs or []
    indexing = "xy"
    axis: int | None = None
    for kw in kwargs:
        if kw.arg == "indexing" and isinstance(kw.value, ast.Constant):
            indexing = kw.value.value
        elif kw.arg == MESHGRID_AXIS_KW and isinstance(kw.value, ast.Constant):
            axis = kw.value.value
    if indexing not in ("ij", "xy"):
        raise NotImplementedError(f"np.meshgrid indexing={indexing!r} not supported")
    k = len(args)
    if axis is None or not (0 <= axis < k):
        raise NotImplementedError("np.meshgrid: output axis unresolved")
    # Length of each 1-D input array.
    lengths: list[ast.expr] = []
    for a in args:
        ext = iter_extent_of_(a, shape_table)
        if ext is None or len(ext) != 1:
            raise NotImplementedError("np.meshgrid needs 1-D inputs of known length")
        lengths.append(ext[0])
    # perm maps an OUTPUT axis to the INPUT axis whose length it takes; it is a
    # single swap of 0 and 1 for 'xy' (and its own inverse), identity for 'ij'.
    perm = list(range(k))
    if indexing == "xy" and k >= 2:
        perm[0], perm[1] = 1, 0
    out_dims = [lengths[perm[p]] for p in range(k)]
    iters = [f"__mgi{p}" for p in range(k)]
    # This output varies with input ``axis`` -> along output axis ``perm[axis]``
    # (perm is self-inverse, so the read iterator is ``iters[perm[axis]]``).
    read_iter = name_(iters[perm[axis]])
    src = scalarize_at_iters(copy.deepcopy(args[axis]), [read_iter], shape_table)
    out_slot = name_(iters[0]) if k == 1 else ast.Tuple(elts=[name_(v) for v in iters], ctx=ast.Load())
    body = [ast.Assign(targets=[ast.Subscript(value=name_(target.id), slice=out_slot, ctx=ast.Store())], value=src)]
    return wrap_for_loops(iters, [copy.deepcopy(d) for d in out_dims], body)


def expand_eye(target: ast.expr, args: list[ast.expr], shape_table: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    """``out = np.eye(n)`` -> ``out[i, j] = (i == j) ? 1.0 : 0.0``."""
    if not args:
        raise NotImplementedError("np.eye needs at least 1 arg")
    n = args[0]
    body = [
        ast.Assign(
            targets=[
                ast.Subscript(
                    value=name_(target.id),
                    slice=ast.Tuple(elts=[name_("__i"), name_("__j")], ctx=ast.Load()),
                    ctx=ast.Store(),
                )
            ],
            value=ast.IfExp(
                test=ast.Compare(left=name_("__i"), ops=[ast.Eq()], comparators=[name_("__j")]),
                body=const_(1.0),
                orelse=const_(0.0),
            ),
        )
    ]
    return [
        ast.For(
            target=store_("__i"),
            iter=ast.Call(func=name_("range"), args=[n], keywords=[]),
            body=[
                ast.For(
                    target=store_("__j"),
                    iter=ast.Call(func=name_("range"), args=[n], keywords=[]),
                    body=body,
                    orelse=[],
                )
            ],
            orelse=[],
        )
    ]
