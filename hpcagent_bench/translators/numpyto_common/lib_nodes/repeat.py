"""``np.repeat`` with a scalar or per-element count."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.lib_nodes.call_args import axis_literal_or_refuse, kwarg_or_pos
from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import iter_extent_of
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import (
    alloc_marker,
    const_,
    const_or_name,
    make_iter_name,
    name_,
    store_,
)


def diff_operand(expr: ast.expr) -> ast.Name | None:
    """``np.diff(p)`` -> ``p``, else ``None``.

    The one per-element ``np.repeat`` count form whose SUM telescopes without
    touching data: ``sum(np.diff(p)) == p[-1] - p[0]``. ``n``/``axis`` kwargs
    change which sum telescopes (or whether it does at all), so only the bare
    single-arg call is recognised.
    """
    if not (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and isinstance(expr.func.value, ast.Name)
        and expr.func.value.id == "np"
        and expr.func.attr == "diff"
    ):
        return None
    if len(expr.args) != 1 or not isinstance(expr.args[0], ast.Name) or expr.keywords:
        return None
    return expr.args[0]


def expand_repeat_prefix_sum(
    target: ast.expr,
    a: ast.Name,
    a_shape: tuple[str, ...],
    k_arg: ast.expr,
    shape_table: dict[str, tuple[str, ...]],
    local_dtypes: dict[str, str] | None,
    fresh_local_allocs: dict[str, tuple[str, ...]] | None,
) -> list[ast.stmt]:
    """Per-element ``np.repeat`` count -- the destination offset is the
    RUNNING prefix sum of the counts, not ``outer * K`` (that formula reads
    the count array as a scalar multiplier and is wrong the moment two counts
    differ; it also silently skips a zero count instead of writing nothing)::

        pos = 0
        for i in range(<source extent>):
            for r in range(counts[i]):
                out[pos] = src[i]
                pos += 1

    Only a 1-D source is supported -- there is no per-axis running offset
    here for the axis-aware case. ``counts`` must be ``np.diff(p)`` so its
    sum (the result extent) telescopes to ``p[-1] - p[0]``; any other
    per-element form has a data-dependent sum with no static extent and is
    refused rather than guessed (an under-sized buffer is a heap overflow).
    ``np.diff`` is never materialised: ``counts[i]`` is expressed inline as
    ``p[i + 1] - p[i]``, so no auxiliary array is allocated at all.
    """
    if len(a_shape) != 1:
        raise NotImplementedError("np.repeat with a per-element count needs a 1-D source")
    diff_src = diff_operand(k_arg)
    if diff_src is None:
        raise NotImplementedError(
            "np.repeat with a per-element count needs a derivable sum "
            f"(got {ast.unparse(k_arg)}); only np.diff(p) telescopes to p[-1] - p[0]"
        )
    p_shape = shape_table.get(diff_src.id)
    if not p_shape or len(p_shape) != 1:
        raise NotImplementedError("np.repeat: np.diff operand shape unknown")
    last_idx = ast.BinOp(left=const_or_name(p_shape[0]), op=ast.Sub(), right=const_(1))
    total = ast.BinOp(
        left=ast.Subscript(value=copy.deepcopy(diff_src), slice=last_idx, ctx=ast.Load()),
        op=ast.Sub(),
        right=ast.Subscript(value=copy.deepcopy(diff_src), slice=const_(0), ctx=ast.Load()),
    )
    extent_tok = ast.unparse(total)
    shape_table[target.id] = (extent_tok,)
    if fresh_local_allocs is not None:
        fresh_local_allocs[target.id] = (extent_tok,)
    pos, src_iter, cnt_iter = "__rep_pos0", "__rep_i0", "__rep_r0"
    if local_dtypes is not None:
        local_dtypes[pos] = "int64"
    count_at_i = ast.BinOp(
        left=ast.Subscript(
            value=copy.deepcopy(diff_src),
            slice=ast.BinOp(left=name_(src_iter), op=ast.Add(), right=const_(1)),
            ctx=ast.Load(),
        ),
        op=ast.Sub(),
        right=ast.Subscript(value=copy.deepcopy(diff_src), slice=name_(src_iter), ctx=ast.Load()),
    )
    body = [
        ast.Assign(
            targets=[ast.Subscript(value=name_(target.id), slice=name_(pos), ctx=ast.Store())],
            value=ast.Subscript(value=copy.deepcopy(a), slice=name_(src_iter), ctx=ast.Load()),
        ),
        ast.AugAssign(target=store_(pos), op=ast.Add(), value=const_(1)),
    ]
    inner_loop = ast.For(
        target=store_(cnt_iter),
        iter=ast.Call(func=name_("range"), args=[count_at_i], keywords=[]),
        body=body,
        orelse=[],
    )
    outer_loop = ast.For(
        target=store_(src_iter),
        iter=ast.Call(func=name_("range"), args=[const_or_name(a_shape[0])], keywords=[]),
        body=[inner_loop],
        orelse=[],
    )
    init_pos = ast.Assign(targets=[store_(pos)], value=const_(0))
    return [alloc_marker(target.id), init_pos, outer_loop]


def expand_repeat(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
    local_dtypes: dict[str, str] | None = None,
    fresh_local_allocs: dict[str, tuple[str, ...]] | None = None,
) -> list[ast.stmt]:
    """``out = np.repeat(A, K, axis=N)`` -> tile-and-write loop nest. Source
    ``A`` of shape ``(s0, ..., sN, ..., sM-1)`` becomes ``out`` of shape
    ``(s0, ..., sN*K, ..., sM-1)``. Per-element::

        out[i0, ..., iN_outer * K + iN_inner, ..., iM-1]
            = A[i0, ..., iN_outer, ..., iM-1]

    The broadcast-from-size-1 case (``sN == 1``, stockham_fft) still applies
    the formula unchanged: ``iN_outer`` only ranges over 0, so ``A`` reads
    ``[..., 0, ...]`` regardless of the inner index.

    ``axis`` may be int (positional or kwarg) or None (flat-axis repeat: result
    is a flat 1-D array of size ``prod(A.shape) * K``).
    """
    if not args or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.repeat needs Name first arg")
    a = args[0]
    a_shape = shape_table.get(a.id)
    if not a_shape:
        raise NotImplementedError("np.repeat: source shape unknown")
    # ``K`` -- repetitions.
    if len(args) < 2:
        raise NotImplementedError("np.repeat needs repetitions arg")
    k_arg = args[1]
    # A PER-ELEMENT repeat count (``np.repeat(np.arange(M), np.diff(A_indptr))``) is a different
    # lowering: the destination offset is a prefix sum of the counts, not ``outer * K`` (that
    # formula reads the count array as a scalar multiplier and computes the wrong offsets).
    if iter_extent_of(k_arg, shape_table) is not None:
        return expand_repeat_prefix_sum(target, a, a_shape, k_arg, shape_table, local_dtypes, fresh_local_allocs)
    # ``axis`` -- positional [2] or kwarg. An ABSENT axis means numpy's flat repeat; an axis that is
    # merely unreadable must NOT fall into that branch, because flat repeat is a different output
    # shape and a different loop nest, not a degraded version of the same one.
    axis_node = kwarg_or_pos(args, kwargs, 2, "axis")
    axis: int | None = None
    if not (axis_node is None or (isinstance(axis_node, ast.Constant) and axis_node.value is None)):
        axis = axis_literal_or_refuse(axis_node, "np.repeat")
    n_dim = len(a_shape)
    if axis is None:
        # Flat repeat: each scalar element repeated K times.
        # out[flat * K + r] = A[flat]
        iters = [make_iter_name("__rp", i) for i in range(n_dim)]
        rep_iter = make_iter_name("__rep", 0)
        # source subscript
        src_slot = name_(iters[0]) if n_dim == 1 else ast.Tuple(elts=[name_(i) for i in iters], ctx=ast.Load())
        # destination flat index = ((((i0)*s1 + i1)*s2 + ...) * K + r)
        flat_index: ast.expr = name_(iters[0])
        for k in range(1, n_dim):
            flat_index = ast.BinOp(
                left=ast.BinOp(left=flat_index, op=ast.Mult(), right=const_or_name(a_shape[k])),
                op=ast.Add(),
                right=name_(iters[k]),
            )
        dst_index = ast.BinOp(
            left=ast.BinOp(left=flat_index, op=ast.Mult(), right=k_arg), op=ast.Add(), right=name_(rep_iter)
        )
        body = [
            ast.Assign(
                targets=[ast.Subscript(value=name_(target.id), slice=dst_index, ctx=ast.Store())],
                value=ast.Subscript(value=name_(a.id), slice=src_slot, ctx=ast.Load()),
            )
        ]
        # Wrap with the source loops and the repetition loop deepest.
        out = body
        out = [
            ast.For(
                target=store_(rep_iter),
                iter=ast.Call(func=name_("range"), args=[k_arg], keywords=[]),
                body=out,
                orelse=[],
            )
        ]
        for var, bound in zip(reversed(iters), reversed(a_shape)):
            out = [
                ast.For(
                    target=store_(var),
                    iter=ast.Call(func=name_("range"), args=[const_or_name(bound)], keywords=[]),
                    body=out,
                    orelse=[],
                )
            ]
        return out
    # Axis-aware repeat: walk every axis; for axis ``N`` the dest
    # index is ``outer_N * K + inner_N`` while source still reads at
    # ``outer_N``.
    if axis < 0:
        axis += n_dim
    if axis < 0 or axis >= n_dim:
        raise NotImplementedError(f"np.repeat axis {axis} out of range for ndim {n_dim}")
    iters = [make_iter_name("__rp", i) for i in range(n_dim)]
    rep_iter = make_iter_name("__rep", 0)
    src_elts = [name_(iters[i]) for i in range(n_dim)]
    dst_elts: list[ast.expr] = []
    for i in range(n_dim):
        if i == axis:
            dst_elts.append(
                ast.BinOp(
                    left=ast.BinOp(left=name_(iters[i]), op=ast.Mult(), right=k_arg),
                    op=ast.Add(),
                    right=name_(rep_iter),
                )
            )
        else:
            dst_elts.append(name_(iters[i]))
    src_slot = src_elts[0] if n_dim == 1 else ast.Tuple(elts=src_elts, ctx=ast.Load())
    dst_slot = dst_elts[0] if n_dim == 1 else ast.Tuple(elts=dst_elts, ctx=ast.Load())
    body = [
        ast.Assign(
            targets=[ast.Subscript(value=name_(target.id), slice=dst_slot, ctx=ast.Store())],
            value=ast.Subscript(value=name_(a.id), slice=src_slot, ctx=ast.Load()),
        )
    ]
    # Innermost = repetition loop.
    out = body
    out = [
        ast.For(
            target=store_(rep_iter), iter=ast.Call(func=name_("range"), args=[k_arg], keywords=[]), body=out, orelse=[]
        )
    ]
    for var, bound in zip(reversed(iters), reversed(a_shape)):
        out = [
            ast.For(
                target=store_(var),
                iter=ast.Call(func=name_("range"), args=[const_or_name(bound)], keywords=[]),
                body=out,
                orelse=[],
            )
        ]
    return out
