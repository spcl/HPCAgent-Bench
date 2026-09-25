"""Reductions: sum/max/min/mean/prod/any/all/count_nonzero/argmax/argmin/std/var, full or along axes."""

import ast
import copy
from collections.abc import Collection, Mapping, Sequence
from collections.abc import Callable

from hpcagent_bench.translators.numpyto_common.lib_nodes.call_args import eval_axes, read_kwarg, read_axis_keepdims
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import (
    const_,
    const_or_name,
    falsy,
    if_set,
    make_iter_name,
    name_,
    resolve_shape,
    shape_total_product,
    store_,
    truthy,
    wrap_for_loops,
)


def reduction_source_index(
    n_dim: int,
    axes: Collection[int],
    red_iter_map: Mapping[int, str],
    outer_iter_names: Sequence[str],
    reduced: Callable[[int], ast.expr] | None = None,
) -> list[ast.expr]:
    """Subscript entries reading a reduced operand: a reduced axis takes its reduction iterator (or
    ``reduced(axis)``), the rest take the outer iterators in order."""
    out: list[ast.expr] = []
    outer_pos = 0
    for k in range(n_dim):
        if k in axes:
            out.append(name_(red_iter_map[k]) if reduced is None else reduced(k))
        else:
            out.append(name_(outer_iter_names[outer_pos]))
            outer_pos += 1
    return out


def reduction_output_index(
    n_dim: int, axes: Collection[int], outer_iter_names: Sequence[str], keepdims: bool
) -> list[ast.expr]:
    """Subscript entries writing a reduction result: a reduced axis is dropped (index 0 under keepdims), the
    rest take the outer iterators in order."""
    out: list[ast.expr] = []
    outer_pos = 0
    for k in range(n_dim):
        if k in axes:
            if keepdims:
                out.append(const_(0))
        else:
            out.append(name_(outer_iter_names[outer_pos]))
            outer_pos += 1
    return out


def expand_axis_reduction(
    target: ast.expr,
    args: list[ast.expr],
    kwargs: list[ast.keyword] | None,
    shape_table: dict[str, tuple[str, ...]],
    init: ast.expr,
    op_fn: Callable[[ast.expr, ast.expr], ast.expr] | None,
    post_fn: Callable[[ast.expr, ast.expr], ast.stmt] | None = None,
    update_fn: Callable[[ast.expr, ast.expr, ast.expr], ast.stmt] | None = None,
) -> list[ast.stmt]:
    """Generic axis-aware reduction. Lowers ``out = np.X(arr, axis=k,
    keepdims=True)`` into a nested loop, non-reduction axes outside and the
    reduction axis inside; writes through to ``out`` at the kept axes (axis
    ``k`` writes ``out[..., 0, ...]`` when ``keepdims=True``). Full reduction
    (``axis=None``) walks all axes and writes one scalar to ``target``.

    :param post_fn: optional ``(target_lvalue, divisor) -> ast.stmt`` invoked
        after the loop closes; used by mean to divide by the reduction size.
    :param update_fn: optional ``(store, load, src) -> ast.stmt`` overriding the
        default ``store = op_fn(load, src)``; used by if-guarded boolean
        reductions (any/all/count_nonzero), which can't rely on C's
        bool-as-int arithmetic (invalid in Fortran).

    A float sum accumulates in ONE chain, matching the source order rather than numpy's pairwise
    blocking: reassociation is sanctioned, so the extra block loop bought only a scop that pet
    refuses (POLYCC-008) and a scop-external block accumulator it silently drops (POLYCC-009).
    """
    arr = args[0]
    shape = resolve_shape(arr, shape_table)
    refuse_unreadable_axis(args)
    axes, keepdims = read_axis_keepdims(args, kwargs)
    n_dim = len(shape)

    # ``initial=`` (sum/prod/max/min) seeds the reduction instead of the default
    # identity/first element. The loop still walks every element (max/min are
    # idempotent, so re-including index 0 is harmless) -- numpy's
    # ``op(initial, *elements)``.
    initial = read_kwarg(kwargs, "initial")
    if initial is not None:
        init = initial

    refuse_dropped_reduction_kwargs(kwargs)
    if axes is None:
        return full_reduction(target, arr, shape, init, op_fn, post_fn, update_fn)
    axes_norm = normalized_axes(axes, n_dim)
    axes_set = set(axes_norm)
    # Outer iter names walk the kept axes (those NOT in axes_set);
    # one inner iter per reduction axis.
    kept_axes = [k for k in range(n_dim) if k not in axes_set]
    outer_iter_names = [make_iter_name("__ax", i) for i in range(len(kept_axes))]
    red_iter_names = [make_iter_name("__rd", i) for i in range(len(axes_norm))]
    red_iter_map = dict(zip(axes_norm, red_iter_names))

    src_elts = reduction_source_index(n_dim, axes_set, red_iter_map, outer_iter_names)
    src_slot = src_elts[0] if n_dim == 1 else ast.Tuple(elts=src_elts, ctx=ast.Load())
    out_sub, out_load = reduction_output_refs(
        target, reduction_output_index(n_dim, axes_set, outer_iter_names, keepdims)
    )
    src_sub = ast.Subscript(value=name_(arr.id), slice=src_slot, ctx=ast.Load())
    # Init for axis-reductions: ``out[outer..] = init`` (or the
    # zero-th element of the reduction axes for max/min).
    if isinstance(init, ast.Subscript):
        init_src_elts = reduction_source_index(n_dim, axes_set, {}, outer_iter_names, reduced=lambda k: const_(0))
        init_slot = init_src_elts[0] if n_dim == 1 else ast.Tuple(elts=init_src_elts, ctx=ast.Load())
        init_node = ast.Subscript(value=name_(arr.id), slice=init_slot, ctx=ast.Load())
    else:
        init_node = init
    init_stmt = ast.Assign(targets=[out_sub], value=init_node)
    update_stmt = (
        update_fn(out_sub, out_load, src_sub)
        if update_fn
        else ast.Assign(targets=[out_sub], value=op_fn(out_load, src_sub))
    )
    # Inner loop nest over the reduction axes, deepest first.
    inner_stmts: list[ast.stmt] = [update_stmt]
    for ax, rn in zip(reversed(axes_norm), reversed(red_iter_names)):
        inner_stmts = [
            ast.For(
                target=store_(rn),
                iter=ast.Call(func=name_("range"), args=[const_or_name(shape[ax])], keywords=[]),
                body=inner_stmts,
                orelse=[],
            )
        ]
    if post_fn is not None:
        # Divisor for mean: product of the reduction-axis sizes.
        divisor = const_or_name(shape[axes_norm[0]])
        for ax in axes_norm[1:]:
            divisor = ast.BinOp(left=divisor, op=ast.Mult(), right=const_or_name(shape[ax]))
        inner_stmts.append(post_fn(out_sub, divisor))
    body = [init_stmt] + inner_stmts
    bounds = tuple(shape[k] for k in kept_axes)
    if not bounds:
        # No kept axes (all reduced; equivalent to full reduction).
        return body
    return wrap_for_loops(outer_iter_names, bounds, body)


def refuse_unreadable_axis(args: list[ast.expr]) -> None:
    """Here slot 1 REALLY is the axis (``np.sum(a, 1)``), unlike the shared reader's general case, so
    an unreadable one is refused rather than silently becoming a reduction over every axis."""
    if len(args) >= 2 and not (isinstance(args[1], ast.Constant) and args[1].value is None):
        if eval_axes(args[1]) is None:
            raise NotImplementedError(
                f"axis {ast.unparse(args[1])!r} must be a compile-time integer or tuple "
                f"of them (it selects the loop nest)"
            )


def refuse_dropped_reduction_kwargs(kwargs: list[ast.keyword] | None) -> None:
    """``where=`` masks which elements take part and ``dtype=`` pins the ACCUMULATOR type (the
    classic case being a float64 accumulator over a float32 operand, which changes the result, not
    just its storage). The reduction loop honours neither, so neither may be ignored."""
    for dropped in ("where", "dtype", "out"):
        if read_kwarg(kwargs, dropped) is not None:
            raise NotImplementedError(
                f"reduction {dropped}= is not lowered; it changes the result, so it cannot be dropped"
            )


def full_reduction(
    target: ast.expr,
    arr: ast.expr,
    shape: tuple[str, ...],
    init: ast.expr,
    op_fn: Callable[[ast.expr, ast.expr], ast.expr] | None,
    post_fn: Callable[[ast.expr, ast.expr], ast.stmt] | None,
    update_fn: Callable[[ast.expr, ast.expr, ast.expr], ast.stmt] | None,
) -> list[ast.stmt]:
    """``axis=None``: every axis walked, one scalar written to ``target``."""
    n_dim = len(shape)
    iters = [make_iter_name("__r", i) for i in range(n_dim)]
    subscript = ast.Subscript(
        value=name_(arr.id),
        slice=(name_(iters[0]) if n_dim == 1 else ast.Tuple(elts=[name_(i) for i in iters], ctx=ast.Load())),
        ctx=ast.Load(),
    )
    target_load = ast.Name(id=target.id, ctx=ast.Load())
    body = [
        update_fn(target, target_load, subscript)
        if update_fn
        else ast.Assign(targets=[target], value=op_fn(target_load, subscript))
    ]
    loops = wrap_for_loops(iters, shape, body)
    stmts = [ast.Assign(targets=[target], value=init_for(init, arr, n_dim))]
    stmts.extend(loops)
    if post_fn is not None:
        stmts.append(post_fn(target, shape_total_product(shape)))
    return stmts


def normalized_axes(axes: Sequence[int], n_dim: int) -> list[int]:
    """The reduced axes with negatives resolved mod ``n_dim``; out-of-range and duplicate axes are
    refused."""
    axes_norm: list[int] = []
    for a in axes:
        na = a + n_dim if a < 0 else a
        if na < 0 or na >= n_dim:
            raise NotImplementedError(f"axis {a} out of range for ndim {n_dim}")
        if na in axes_norm:
            raise NotImplementedError(f"duplicate axis {a} in reduction tuple")
        axes_norm.append(na)
    return axes_norm


def reduction_output_refs(target: ast.expr, out_elts: list[ast.expr]) -> tuple[ast.expr, ast.expr]:
    """The (store, load) references of the reduction result: the target itself when no axis is kept
    and keepdims is off (a scalar result), else the target subscripted at ``out_elts``."""
    if len(out_elts) == 0:
        return target, ast.Name(id=target.id, ctx=ast.Load())
    if len(out_elts) == 1:
        return (
            ast.Subscript(value=name_(target.id), slice=out_elts[0], ctx=ast.Store()),
            ast.Subscript(value=name_(target.id), slice=out_elts[0], ctx=ast.Load()),
        )
    out_slot = ast.Tuple(elts=out_elts, ctx=ast.Load())
    return (
        ast.Subscript(value=name_(target.id), slice=out_slot, ctx=ast.Store()),
        ast.Subscript(value=name_(target.id), slice=out_slot, ctx=ast.Load()),
    )


def init_for(init: ast.expr, arr: ast.expr, n_dim: int) -> ast.expr:
    """Resolve init for full reduction: rewrite max/min first-element to
    a fully-zeroed subscript if needed."""
    if isinstance(init, ast.Subscript):
        # For full reduction we use arr[0, 0, ..., 0] as the init.
        if n_dim == 1:
            return ast.Subscript(value=name_(arr.id), slice=const_(0), ctx=ast.Load())
        return ast.Subscript(
            value=name_(arr.id), slice=ast.Tuple(elts=[const_(0)] * n_dim, ctx=ast.Load()), ctx=ast.Load()
        )
    return init


def reduction_elem_is_integer(args: list[ast.expr], local_dtypes: dict[str, str] | None) -> bool:
    """True when the reduced array (first arg, a bare Name) is tagged an
    integer / boolean dtype -- numpy upcasts int8/16/32/bool to int64 for
    ``sum`` / ``prod``, so the accumulator must be an integer, not a float."""
    if not local_dtypes or not args or not isinstance(args[0], ast.Name):
        return False
    dt = local_dtypes.get(args[0].id)
    return dt is not None and dt.startswith(("int", "uint", "bool"))


def nan_reduce_op(cmp: type[ast.cmpop]) -> Callable[[ast.expr, ast.expr], ast.expr]:
    """Running max/min update that propagates NaN like numpy: ``x if (x <cmp> acc
    or x != x) else acc``. The ``x != x`` test lets a NaN element win, and once
    the accumulator is NaN it stays (nothing compares ``<cmp>`` against a NaN).
    Matches numpy (``np.max``/``np.min`` return NaN if any element is NaN),
    unlike C's ``fmax``/the ``max`` macro, which suppress NaN."""

    def f_(acc: ast.expr, x: ast.expr) -> ast.expr:
        return ast.IfExp(
            test=ast.BoolOp(
                op=ast.Or(),
                values=[
                    ast.Compare(left=copy.deepcopy(x), ops=[cmp()], comparators=[copy.deepcopy(acc)]),
                    ast.Compare(left=copy.deepcopy(x), ops=[ast.NotEq()], comparators=[copy.deepcopy(x)]),
                ],
            ),
            body=copy.deepcopy(x),
            orelse=copy.deepcopy(acc),
        )

    return f_


def expand_sum(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
    local_dtypes: dict[str, str] | None = None,
) -> list[ast.stmt]:
    is_int = reduction_elem_is_integer(args, local_dtypes)
    if is_int and local_dtypes is not None and isinstance(target, ast.Name):
        local_dtypes[target.id] = "int64"
    return expand_axis_reduction(
        target,
        args,
        kwargs,
        shape_table,
        init=const_(0) if is_int else const_(0.0),
        op_fn=lambda acc, x: ast.BinOp(left=acc, op=ast.Add(), right=x),
    )


def expand_max(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
) -> list[ast.stmt]:
    arr = args[0]
    reject_zero_size_reduction(args, kwargs, shape_table)
    return expand_axis_reduction(
        target,
        args,
        kwargs,
        shape_table,
        init=ast.Subscript(value=arr, slice=const_(0), ctx=ast.Load()),
        op_fn=nan_reduce_op(ast.Gt),
    )


def expand_min(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
) -> list[ast.stmt]:
    arr = args[0]
    reject_zero_size_reduction(args, kwargs, shape_table)
    return expand_axis_reduction(
        target,
        args,
        kwargs,
        shape_table,
        init=ast.Subscript(value=arr, slice=const_(0), ctx=ast.Load()),
        op_fn=nan_reduce_op(ast.Lt),
    )


def reject_zero_size_reduction(
    args: list[ast.expr], kwargs: list[ast.keyword] | None, shape_table: dict[str, tuple[str, ...]]
) -> None:
    """Refuse to lower ``np.max``/``np.min`` over a statically zero-length
    reduction axis: numpy raises ``zero-size array to reduction ... which has no
    identity``, and the seed ``arr[..., 0]`` would read OOB. Raise
    ``NotImplementedError`` instead of emitting an OOB seed. (A symbolic extent
    that's 0 only at runtime can't be caught here.)"""
    if not args or not isinstance(args[0], ast.Name):
        return
    shape = shape_table.get(args[0].id)
    if not shape:
        return
    n_dim = len(shape)
    axes = read_axis_keepdims(args, kwargs)[0]
    red_axes = range(n_dim) if axes is None else [a + n_dim if a < 0 else a for a in axes]
    for ax in red_axes:
        if 0 <= ax < n_dim and str(shape[ax]) == "0":
            raise NotImplementedError("zero-size array to reduction which has no identity")


def expand_mean(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
) -> list[ast.stmt]:
    # Special form ``np.mean(arr[mask])``, ``mask`` a same-length boolean array:
    # boolean fancy indexing produces a dynamic-length compacted view we don't
    # materialise, so emit a masked-sum + count loop directly.
    if (
        args
        and isinstance(args[0], ast.Subscript)
        and isinstance(args[0].value, ast.Name)
        and isinstance(args[0].slice, ast.Name)
    ):
        arr = args[0].value
        mask = args[0].slice
        a_shape = shape_table.get(arr.id)
        if a_shape and len(a_shape) == 1:
            n_ast = const_or_name(a_shape[0])
            iter_name = "__mn_i"
            sum_name = "__mn_sum"
            cnt_name = "__mn_cnt"
            body = [
                ast.If(
                    test=ast.Subscript(value=name_(mask.id), slice=name_(iter_name), ctx=ast.Load()),
                    body=[
                        ast.AugAssign(
                            target=store_(sum_name),
                            op=ast.Add(),
                            value=ast.Subscript(value=name_(arr.id), slice=name_(iter_name), ctx=ast.Load()),
                        ),
                        ast.AugAssign(target=store_(cnt_name), op=ast.Add(), value=const_(1)),
                    ],
                    orelse=[],
                ),
            ]
            return [
                ast.Assign(targets=[store_(sum_name)], value=const_(0.0)),
                ast.Assign(targets=[store_(cnt_name)], value=const_(0)),
                ast.For(
                    target=store_(iter_name),
                    iter=ast.Call(func=name_("range"), args=[n_ast], keywords=[]),
                    body=body,
                    orelse=[],
                ),
                ast.Assign(
                    targets=[store_(target.id)],
                    value=ast.BinOp(left=name_(sum_name), op=ast.Div(), right=name_(cnt_name)),
                ),
            ]
    return expand_axis_reduction(
        target,
        args,
        kwargs,
        shape_table,
        init=const_(0.0),
        op_fn=lambda acc, x: ast.BinOp(left=acc, op=ast.Add(), right=x),
        post_fn=lambda lvalue, divisor: ast.Assign(
            targets=[lvalue],
            value=ast.BinOp(
                left=(
                    lvalue
                    if isinstance(lvalue, ast.Name)
                    else ast.Subscript(value=lvalue.value, slice=lvalue.slice, ctx=ast.Load())
                ),
                op=ast.Div(),
                right=divisor,
            ),
        ),
    )


def expand_prod(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
    local_dtypes: dict[str, str] | None = None,
) -> list[ast.stmt]:
    is_int = reduction_elem_is_integer(args, local_dtypes)
    if is_int and local_dtypes is not None and isinstance(target, ast.Name):
        local_dtypes[target.id] = "int64"
    return expand_axis_reduction(
        target,
        args,
        kwargs,
        shape_table,
        init=const_(1) if is_int else const_(1.0),
        op_fn=lambda acc, x: ast.BinOp(left=acc, op=ast.Mult(), right=x),
    )


def expand_any(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
) -> list[ast.stmt]:
    """``s = np.any(A [, axis=k, keepdims=...])`` -- OR reduction. Init=0; each
    truthy element sets the (integer 0/1) accumulator to 1."""
    return expand_axis_reduction(
        target,
        args,
        kwargs,
        shape_table,
        init=const_(0),
        op_fn=None,
        update_fn=if_set(truthy, lambda load: const_(1)),
    )


def expand_all(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
) -> list[ast.stmt]:
    """``s = np.all(A [, axis=k, keepdims=...])`` -- AND reduction. Init=1; each
    falsy element clears the (integer 0/1) accumulator to 0."""
    return expand_axis_reduction(
        target, args, kwargs, shape_table, init=const_(1), op_fn=None, update_fn=if_set(falsy, lambda load: const_(0))
    )


def expand_count_nonzero(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
) -> list[ast.stmt]:
    """``s = np.count_nonzero(A [, axis=k, keepdims=...])`` -- count of non-zero
    elements. Init=0; each truthy element increments the integer accumulator."""
    return expand_axis_reduction(
        target,
        args,
        kwargs,
        shape_table,
        init=const_(0),
        op_fn=None,
        update_fn=if_set(truthy, lambda load: ast.BinOp(left=load, op=ast.Add(), right=const_(1))),
    )


def expand_argmax(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
) -> list[ast.stmt]:
    """``i = np.argmax(A [, axis=k, keepdims=...])`` -- index of the maximum.
    Only ``axis=None`` (flat, scalar result) and ``axis=int`` are implemented;
    an axis tuple raises ``NotImplementedError`` (a reshape + flat argmax
    expresses it)."""
    return expand_arg_reduction(target, args, shape_table, kwargs, op="argmax")


def expand_argmin(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
) -> list[ast.stmt]:
    return expand_arg_reduction(target, args, shape_table, kwargs, op="argmin")


def expand_arg_reduction(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None,
    op: str,
) -> list[ast.stmt]:
    """``argmax``/``argmin`` shared scaffold. Supports the full ``axis = None /
    int / tuple / list`` matrix: ``None`` is a full reduction to a single flat
    index; an int reduces one axis, keeping the others at the input's extent,
    indexed 0..shape[axis]-1; a tuple ravels the chosen axes and returns the
    flat index across them (output keeps the axes not in the tuple; per
    kept-axes position, walk the reduction axes in source order tracking
    (best_val, best_flat_idx), with the flat index using row-major mapping over
    the reduction-axis sizes). ``keepdims=True`` adds a size-1 axis at each
    reduced position in all three forms.
    """
    if not args or not isinstance(args[0], ast.Name):
        raise NotImplementedError(f"np.{op} needs Name first arg")
    a = args[0]
    shape = resolve_shape(a, shape_table)
    axes, keepdims = read_axis_keepdims(args, kwargs)
    n_dim = len(shape)
    cmp_op = ast.Gt() if op == "argmax" else ast.Lt()
    # Normalise axes -> set + ordered list (for flat-index mapping).
    if axes is None:
        axes_norm = list(range(n_dim))
    else:
        axes_norm = []
        for ax in axes:
            na = ax + n_dim if ax < 0 else ax
            if na < 0 or na >= n_dim:
                raise NotImplementedError(f"np.{op} axis {ax} out of range for ndim {n_dim}")
            if na in axes_norm:
                raise NotImplementedError(f"np.{op} duplicate axis {ax}")
            axes_norm.append(na)
    axes_set = set(axes_norm)
    kept_axes = [k for k in range(n_dim) if k not in axes_set]
    outer_iter_names = [make_iter_name("__aax", i) for i in range(len(kept_axes))]
    red_iter_names = [make_iter_name("__ard", i) for i in range(len(axes_norm))]
    red_iter_map = dict(zip(axes_norm, red_iter_names))
    src_elts = reduction_source_index(n_dim, axes_set, red_iter_map, outer_iter_names)
    src_slot = src_elts[0] if n_dim == 1 else ast.Tuple(elts=src_elts, ctx=ast.Load())
    src_sub = ast.Subscript(value=name_(a.id), slice=src_slot, ctx=ast.Load())
    out_elts = reduction_output_index(n_dim, axes_set, outer_iter_names, keepdims)
    # First-element init: reduction axes pinned at 0, kept axes at the outer iters.
    init_src_elts = reduction_source_index(n_dim, axes_set, {}, outer_iter_names, reduced=lambda k: const_(0))
    init_slot = init_src_elts[0] if n_dim == 1 else ast.Tuple(elts=init_src_elts, ctx=ast.Load())
    init_val = ast.Subscript(value=name_(a.id), slice=init_slot, ctx=ast.Load())
    out_sub, unused = reduction_output_refs(target, out_elts)
    # The running-best temp is named after its TARGET. A fixed name collided across two argmax
    # expansions in one kernel, and when the two operands had different dtypes the second
    # expansion's temp inherited the first's declared type -- cp2k_density_matrix_trs4 compared an
    # integer temp with .eqv. because the other argmax in the same body ran over a boolean plane.
    best_val = f"__ar_val_{target.id}" if isinstance(target, ast.Name) else "__ar_val"
    init_stmts: list[ast.stmt] = [
        ast.Assign(targets=[store_(best_val)], value=init_val),
        ast.Assign(targets=[out_sub], value=const_(0)),
    ]
    # Flat index across the reduction axes (in source order):
    #   ((red_iter[0] * shape[axis1]) + red_iter[1]) * shape[axis2] + red_iter[2] + ...
    flat_idx: ast.expr = name_(red_iter_map[axes_norm[0]])
    for k in range(1, len(axes_norm)):
        flat_idx = ast.BinOp(
            left=ast.BinOp(left=flat_idx, op=ast.Mult(), right=const_or_name(shape[axes_norm[k]])),
            op=ast.Add(),
            right=name_(red_iter_map[axes_norm[k]]),
        )
    # NaN semantics (numpy): argmax/argmin return the index of the FIRST NaN.
    # Update rule ``(best == best) and (src != src or src <cmp> best)``:
    # ``best == best`` goes false once ``best`` is NaN, locking the index at the
    # first NaN; ``src != src`` lets a NaN element win (sets best = NaN); else the
    # ordinary ``src <cmp> best`` drives the arg. (Element 0 already NaN: the seed
    # ``best == best`` is false and the index stays 0.)
    best_not_nan = ast.Compare(left=name_(best_val), ops=[ast.Eq()], comparators=[name_(best_val)])
    src_is_nan = ast.Compare(left=copy.deepcopy(src_sub), ops=[ast.NotEq()], comparators=[copy.deepcopy(src_sub)])
    ordinary = ast.Compare(left=copy.deepcopy(src_sub), ops=[cmp_op], comparators=[name_(best_val)])
    update = ast.If(
        test=ast.BoolOp(op=ast.And(), values=[best_not_nan, ast.BoolOp(op=ast.Or(), values=[src_is_nan, ordinary])]),
        body=[
            ast.Assign(targets=[store_(best_val)], value=copy.deepcopy(src_sub)),
            ast.Assign(targets=[out_sub], value=flat_idx),
        ],
        orelse=[],
    )
    # Wrap the comparison in nested reduction loops, deepest first.
    inner_body: list[ast.stmt] = [update]
    for ax, rn in zip(reversed(axes_norm), reversed(red_iter_names)):
        inner_body = [
            ast.For(
                target=store_(rn),
                iter=ast.Call(func=name_("range"), args=[const_or_name(shape[ax])], keywords=[]),
                body=inner_body,
                orelse=[],
            )
        ]
    body_stmts = init_stmts + inner_body
    if not kept_axes:
        return body_stmts
    bounds = tuple(shape[k] for k in kept_axes)
    return wrap_for_loops(outer_iter_names, bounds, body_stmts)


def expand_std(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
) -> list[ast.stmt]:
    """``s = np.std(A [, axis=k, keepdims=...])`` -- mean + sum of squared
    deviations + sqrt, over the unified axis-aware reduction scaffold (axis=None
    full reduction, axis=k vector along kept axes, keepdims preserves a size-1
    axis at k). Composed as a mean reduction (sum/count) then a
    sum-of-squared-deviations reduction over the same axes then sqrt(sum/count);
    goes through ``expand_axis_reduction`` twice -- once into ``target`` for the
    mean, once into a scratch ``__sd`` for the squared deviations.
    """
    return expand_var_or_std(target, args, shape_table, kwargs, finish="sqrt")


def expand_var(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
) -> list[ast.stmt]:
    """``s = np.var(A [, axis=k, keepdims=...])`` -- mean + sum of
    squared deviations, no sqrt. Shares the scaffold with
    :func:`expand_std` (axis-tuple supported)."""
    return expand_var_or_std(target, args, shape_table, kwargs, finish="none")


def expand_var_or_std(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None,
    finish: str,
) -> list[ast.stmt]:
    """Shared scaffold for ``np.var``/``np.std`` -- variance with optional sqrt
    finalisation. Supports the full ``axis = None/int/tuple/list`` and
    ``keepdims`` matrix: walks kept axes outside, reduces each reduction axis
    inside (deepest first), writes back ``sqrt(sum_of_squared_dev / divisor)``
    per kept-axis position (``divisor`` = product of the reduction-axis sizes).
    """
    if not args or not isinstance(args[0], ast.Name):
        raise NotImplementedError(f"np.{finish or 'var'} needs Name first arg")
    a = args[0]
    shape = resolve_shape(a, shape_table)
    axes, keepdims = read_axis_keepdims(args, kwargs)
    n_dim = len(shape)
    if axes is None:
        axes_norm = list(range(n_dim))
    else:
        axes_norm = []
        for ax in axes:
            na = ax + n_dim if ax < 0 else ax
            if na < 0 or na >= n_dim:
                raise NotImplementedError(f"np.{finish or 'var'} axis {ax} out of range")
            if na in axes_norm:
                raise NotImplementedError(f"np.{finish or 'var'} duplicate axis {ax}")
            axes_norm.append(na)
    axes_set = set(axes_norm)
    kept_axes = [k for k in range(n_dim) if k not in axes_set]

    # Step 1: compute the mean (axis-aware) into ``target``.
    mean_stmts = expand_axis_reduction(
        target,
        args,
        kwargs,
        shape_table,
        init=const_(0.0),
        op_fn=lambda acc, x: ast.BinOp(left=acc, op=ast.Add(), right=x),
        post_fn=lambda lvalue, divisor: ast.Assign(
            targets=[lvalue],
            value=ast.BinOp(
                left=(
                    lvalue
                    if isinstance(lvalue, ast.Name)
                    else ast.Subscript(value=lvalue.value, slice=lvalue.slice, ctx=ast.Load())
                ),
                op=ast.Div(),
                right=divisor,
            ),
        ),
    )

    # Step 2: accumulate squared deviations into ``__sd_acc`` (scalar per
    # kept-axes position), then finalise as ``out = (__sd_acc / divisor)``, sqrt
    # wrapped for std. The mean target slot stays alive through the inner
    # reduction loops; only the outer (kept) loop nest overwrites it.
    outer_iter_names = [make_iter_name("__sax", i) for i in range(len(kept_axes))]
    red_iter_names = [make_iter_name("__srd", i) for i in range(len(axes_norm))]
    red_iter_map = dict(zip(axes_norm, red_iter_names))

    src_elts = reduction_source_index(n_dim, axes_set, red_iter_map, outer_iter_names)
    src_slot = src_elts[0] if n_dim == 1 else ast.Tuple(elts=src_elts, ctx=ast.Load())
    out_elts = reduction_output_index(n_dim, axes_set, outer_iter_names, keepdims)
    is_scalar_target = len(out_elts) == 0
    if is_scalar_target:
        out_sub: ast.expr = target
        out_load: ast.expr = ast.Name(id=target.id, ctx=ast.Load())
    elif len(out_elts) == 1:
        out_sub = ast.Subscript(value=name_(target.id), slice=out_elts[0], ctx=ast.Store())
        out_load = ast.Subscript(value=name_(target.id), slice=out_elts[0], ctx=ast.Load())
    else:
        slot = ast.Tuple(elts=out_elts, ctx=ast.Load())
        out_sub = ast.Subscript(value=name_(target.id), slice=slot, ctx=ast.Store())
        out_load = ast.Subscript(value=name_(target.id), slice=slot, ctx=ast.Load())
    src_sub = ast.Subscript(value=name_(a.id), slice=src_slot, ctx=ast.Load())
    sd_acc = "__sd_acc"
    diff = ast.BinOp(left=src_sub, op=ast.Sub(), right=out_load)
    sq = ast.BinOp(left=diff, op=ast.Mult(), right=diff)
    init_acc = ast.Assign(targets=[store_(sd_acc)], value=const_(0.0))
    add_acc = ast.AugAssign(target=store_(sd_acc), op=ast.Add(), value=sq)
    # Divisor = product of every reduction-axis size, minus ``ddof`` (numpy's
    # ``np.var``/``np.std`` divide by ``N - ddof``; ddof defaults to 0). The
    # mean above always divides by the full ``N`` -- only the variance honors
    # ddof.
    divisor: ast.expr = const_or_name(shape[axes_norm[0]])
    for ax in axes_norm[1:]:
        divisor = ast.BinOp(left=divisor, op=ast.Mult(), right=const_or_name(shape[ax]))
    ddof = read_kwarg(kwargs, "ddof")
    if ddof is not None and not (isinstance(ddof, ast.Constant) and ddof.value == 0):
        divisor = ast.BinOp(left=divisor, op=ast.Sub(), right=copy.deepcopy(ddof))
    finalize_value: ast.expr = ast.BinOp(left=name_(sd_acc), op=ast.Div(), right=divisor)
    if finish == "sqrt":
        finalize_value = ast.Call(func=name_("sqrt"), args=[finalize_value], keywords=[])
    finalize = ast.Assign(targets=[out_sub], value=finalize_value)
    # Wrap inner reduction iters around add_acc, deepest first.
    inner_body: list[ast.stmt] = [add_acc]
    for ax, rn in zip(reversed(axes_norm), reversed(red_iter_names)):
        inner_body = [
            ast.For(
                target=store_(rn),
                iter=ast.Call(func=name_("range"), args=[const_or_name(shape[ax])], keywords=[]),
                body=inner_body,
                orelse=[],
            )
        ]
    body_stmts = [init_acc, *inner_body, finalize]
    if is_scalar_target:
        return mean_stmts + body_stmts
    bounds = tuple(shape[k] for k in kept_axes)
    return mean_stmts + wrap_for_loops(outer_iter_names, bounds, body_stmts)
