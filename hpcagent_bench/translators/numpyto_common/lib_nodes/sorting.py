"""Sorting and order statistics: sort, median, searchsorted."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.lib_nodes.call_args import kwarg_or_pos, read_axis_keepdims
from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import iter_extent_of
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import (
    const_,
    const_or_name,
    flat_index_,
    name_,
    store_,
    wrap_for_loops,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.scalarize import scalarize_at_iters


def make_sort_routine(buf: str, n: ast.expr, prefix: str) -> list[ast.stmt]:
    """In-place ascending insertion sort over ``buf[0:n]`` (rendered as plain
    loops -- every backend supports them; shared by median / future
    percentile / quantile). ``prefix`` namespaces the loop / temp vars."""
    i, j, key = f"{prefix}_i", f"{prefix}_j", f"{prefix}_key"
    # key = buf[i]; j = i - 1; while j >= 0 and buf[j] > key: buf[j+1]=buf[j]; j-=1; buf[j+1]=key
    inner = [
        ast.Assign(targets=[store_(key)], value=ast.Subscript(value=name_(buf), slice=name_(i), ctx=ast.Load())),
        ast.Assign(targets=[store_(j)], value=ast.BinOp(left=name_(i), op=ast.Sub(), right=const_(1))),
        ast.While(
            test=ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Compare(left=name_(j), ops=[ast.GtE()], comparators=[const_(0)]),
                    ast.Compare(
                        left=ast.Subscript(value=name_(buf), slice=name_(j), ctx=ast.Load()),
                        ops=[ast.Gt()],
                        comparators=[name_(key)],
                    ),
                ],
            ),
            body=[
                ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=name_(buf),
                            slice=ast.BinOp(left=name_(j), op=ast.Add(), right=const_(1)),
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Subscript(value=name_(buf), slice=name_(j), ctx=ast.Load()),
                ),
                ast.AugAssign(target=store_(j), op=ast.Sub(), value=const_(1)),
            ],
            orelse=[],
        ),
        ast.Assign(
            targets=[
                ast.Subscript(
                    value=name_(buf), slice=ast.BinOp(left=name_(j), op=ast.Add(), right=const_(1)), ctx=ast.Store()
                )
            ],
            value=name_(key),
        ),
    ]
    return [
        ast.For(
            target=store_(i),
            iter=ast.Call(func=name_("range"), args=[const_(1), n], keywords=[]),
            body=inner,
            orelse=[],
        )
    ]


def expand_median(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
    local_dtypes: dict[str, str] | None = None,
    fresh_local_allocs: dict[str, tuple[str, ...]] | None = None,
) -> list[ast.stmt]:
    """``np.median(a)`` (full, flattened) -> copy + insertion-sort + pick the
    middle element (mean of the two middles for an even count).

    A scratch buffer ``__md_buf`` of the operand's total size holds the sorted
    copy so the input is not mutated."""
    if not args or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.median needs a bare-Name array")
    # ONLY the full flattened median. An axis was silently ignored: the expander flattened every
    # element and returned one scalar, so ``np.median(A, axis=1)`` computed a whole-array median
    # and reported no error at all.
    if read_axis_keepdims(args, kwargs or [])[0] is not None:
        raise NotImplementedError("np.median(axis=...) is not implemented; only the flattened median is")
    a = args[0]
    shape = shape_table.get(a.id)
    if shape is None:
        raise NotImplementedError("np.median: operand shape unknown")
    total = shape[0] if len(shape) == 1 else "(" + ") * (".join(shape) + ")"
    buf = "__md_buf"
    if fresh_local_allocs is not None:
        fresh_local_allocs[buf] = (total,)
    n_node = const_or_name(total)
    # Flat copy a -> buf.
    cp_iters = [f"__mdc{i}" for i in range(len(shape))]
    flat = flat_index_(cp_iters, shape)
    copy_body = [
        ast.Assign(
            targets=[ast.Subscript(value=name_(buf), slice=flat, ctx=ast.Store())],
            value=ast.Subscript(
                value=name_(a.id),
                slice=(
                    name_(cp_iters[0])
                    if len(shape) == 1
                    else ast.Tuple(elts=[name_(c) for c in cp_iters], ctx=ast.Load())
                ),
                ctx=ast.Load(),
            ),
        )
    ]
    copy_loops = wrap_for_loops(cp_iters, list(shape), copy_body)
    sort = make_sort_routine(buf, n_node, "__md")
    half = ast.BinOp(left=copy.deepcopy(n_node), op=ast.FloorDiv(), right=const_(2))
    mid = ast.Subscript(value=name_(buf), slice=copy.deepcopy(half), ctx=ast.Load())
    mid_lo = ast.Subscript(
        value=name_(buf), slice=ast.BinOp(left=copy.deepcopy(half), op=ast.Sub(), right=const_(1)), ctx=ast.Load()
    )
    # even count -> mean of the two middles; odd -> the single middle.
    even = ast.Compare(
        left=ast.BinOp(left=copy.deepcopy(n_node), op=ast.Mod(), right=const_(2)),
        ops=[ast.Eq()],
        comparators=[const_(0)],
    )
    pick = ast.IfExp(
        test=even,
        body=ast.BinOp(
            left=ast.BinOp(left=mid_lo, op=ast.Add(), right=copy.deepcopy(mid)), op=ast.Div(), right=const_(2.0)
        ),
        orelse=mid,
    )
    store = ast.Assign(targets=[store_(target.id)], value=pick)
    return [*copy_loops, *sort, store]


def expand_sort(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
) -> list[ast.stmt]:
    """``np.sort(a)`` (1-D, ascending) -> copy ``a`` into the target buffer, then
    an in-place insertion sort of that buffer. Only the 1-D form is lowered. The
    target may be an output parameter (``out[:] = np.sort(a)``) or a fresh local
    (``t = np.sort(a)``, allocated via :data:`ELEMENT_WRITE_EXPANDERS`);
    registering ``shape_table[target]`` here makes the sorted buffer's shape
    available to that auto-alloc and any later use of ``t``."""
    if not args or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.sort needs a bare-Name array")
    if not isinstance(target, ast.Name):
        raise NotImplementedError("np.sort target must be a bare Name")
    a = args[0]
    shape = shape_table.get(a.id)
    if shape is None:
        raise NotImplementedError("np.sort: operand shape unknown")
    if len(shape) != 1:
        raise NotImplementedError("np.sort: only a 1-D array is lowered")
    out = target.id
    shape_table.setdefault(out, tuple(shape))
    it = "__srtc"
    copy_body = [
        ast.Assign(
            targets=[ast.Subscript(value=name_(out), slice=name_(it), ctx=ast.Store())],
            value=ast.Subscript(value=name_(a.id), slice=name_(it), ctx=ast.Load()),
        )
    ]
    copy_loops = wrap_for_loops([it], [shape[0]], copy_body)
    sort = make_sort_routine(out, const_or_name(shape[0]), "__srt")
    return [*copy_loops, *sort]


def expand_searchsorted(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
    local_dtypes: dict[str, str] | None = None,
) -> list[ast.stmt]:
    """``np.searchsorted(a, v, side=...)`` -> a binary search per element of ``v``.

    ``a`` is a sorted 1-D array; the result is an int64 array shaped like ``v``, holding for each
    element the index where it would be inserted to keep ``a`` sorted. ``side='left'`` counts the
    entries STRICTLY below the value, ``side='right'`` counts those at or below it -- one comparison
    apart, and the difference is exactly what a bin lookup's ``- 1`` relies on.

    A binary search, not a scan: numpy's is O(log n) per element and the corpus calls this with a
    grid of tens of thousands of edges. A linear count would return the same indices and turn the
    kernel's complexity class into something the reference never had.
    """
    if len(args) < 2 or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.searchsorted needs a bare-Name sorted array and a values operand")
    if not isinstance(target, ast.Name):
        raise NotImplementedError("np.searchsorted target must be a bare Name")
    sorted_shape = shape_table.get(args[0].id)
    if sorted_shape is None or len(sorted_shape) != 1:
        raise NotImplementedError("np.searchsorted: the sorted operand must be a 1-D array of known shape")
    values_shape = iter_extent_of(args[1], shape_table)
    if values_shape is None:
        raise NotImplementedError("np.searchsorted: shape of the values operand unknown")
    side_node = kwarg_or_pos(args, kwargs or [], 2, "side")
    side = "left" if side_node is None else (side_node.value if isinstance(side_node, ast.Constant) else None)
    if side not in ("left", "right"):
        raise NotImplementedError("np.searchsorted: side must be the literal 'left' or 'right'")
    if local_dtypes is not None:
        local_dtypes.setdefault(target.id, "int64")
    shape_table.setdefault(target.id, tuple(ast.unparse(e) for e in values_shape))

    iters = [f"__ss{k}" for k in range(len(values_shape))]
    index = name_(iters[0]) if len(iters) == 1 else ast.Tuple(elts=[name_(v) for v in iters], ctx=ast.Load())
    value = scalarize_at_iters(copy.deepcopy(args[1]), [name_(v) for v in iters], shape_table)
    lo, hi, mid = "__ss_lo", "__ss_hi", "__ss_mid"
    # ``a[mid] <= v`` for side='right', ``a[mid] < v`` for side='left': the first is the count of
    # entries at or below the value, the second the count strictly below it.
    below = ast.Compare(
        left=ast.Subscript(value=name_(args[0].id), slice=name_(mid), ctx=ast.Load()),
        ops=[ast.LtE() if side == "right" else ast.Lt()],
        comparators=[value],
    )
    search = [
        ast.Assign(targets=[store_(lo)], value=const_(0)),
        ast.Assign(targets=[store_(hi)], value=const_or_name(sorted_shape[0])),
        ast.While(
            test=ast.Compare(left=name_(lo), ops=[ast.Lt()], comparators=[name_(hi)]),
            body=[
                ast.Assign(
                    targets=[store_(mid)],
                    value=ast.BinOp(
                        left=ast.BinOp(left=name_(lo), op=ast.Add(), right=name_(hi)),
                        op=ast.FloorDiv(),
                        right=const_(2),
                    ),
                ),
                ast.If(
                    test=below,
                    body=[
                        ast.Assign(
                            targets=[store_(lo)], value=ast.BinOp(left=name_(mid), op=ast.Add(), right=const_(1))
                        )
                    ],
                    orelse=[ast.Assign(targets=[store_(hi)], value=name_(mid))],
                ),
            ],
            orelse=[],
        ),
        ast.Assign(targets=[ast.Subscript(value=name_(target.id), slice=index, ctx=ast.Store())], value=name_(lo)),
    ]
    return wrap_for_loops(iters, list(values_shape), search)
