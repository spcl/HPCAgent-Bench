"""Render an array-valued expression at scalar loop indices."""

import ast

from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import (
    advanced_index_rank,
    ctor_fill_element,
    resolve_negative,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import (
    const_,
    const_or_name,
    name_id,
    slice_step_any,
    step_is_negative,
    step_node,
    slice_axes,
)


def slice_start(ax: ast.Slice, axis_len: ast.expr | None, step: int | ast.expr | None) -> ast.expr | None:
    """First SOURCE index a slice reads. ``lower`` when given (negative resolved
    against ``axis_len``), else 0 -- except under a NEGATIVE step, where numpy
    flips the default and starts at the last element ``axis_len - 1``
    (``a[::-1]``). A reverse slice over an untracked axis cannot be indexed at
    all; refuse loudly rather than emit the forward ``a[i]`` that would silently
    drop the reversal (mirrors lowering's slice-assign rewriters).

    ``step`` is a literal int or a symbolic step expression; only a literal can be the reverse."""
    if ax.lower is not None:
        return resolve_negative(ax.lower, axis_len)
    if step_is_negative(step):
        if axis_len is None:
            raise NotImplementedError("reverse slice needs a known axis length (shape untracked)")
        return ast.BinOp(left=axis_len, op=ast.Sub(), right=const_(1))
    return const_(0)


def strided_index(ivar: ast.expr, start: ast.expr | None, step: int | ast.expr | None) -> ast.expr:
    """Source index of result position ``ivar`` within a slice ``[start::step]``:
    ``start + ivar * step``. Must stay in lockstep with :func:`iter_extent_of_`,
    which counts ``ceil(extent / |step|)`` elements -- an index that ignored
    ``step`` would walk a DIFFERENT (contiguous) run of the same length.

    ``step`` is a literal int or a symbolic step expression; both multiply the position."""
    unit = step is None or (isinstance(step, int) and step == 1)
    pos: ast.expr = ivar if unit else ast.BinOp(left=ivar, op=ast.Mult(), right=step_node(step))
    if isinstance(start, ast.Constant) and start.value == 0:
        return pos
    return ast.BinOp(left=pos, op=ast.Add(), right=start)


def subscript_result_rank(
    axes: list[ast.expr], shape: tuple[str, ...] | None, shape_table: dict[str, tuple[str, ...]]
) -> int:
    """Result-axis count of a subscript, counted by the SAME rules that consume iters below.

    A slice or a newaxis contributes one axis; an advanced-index group contributes its shared
    broadcast rank once; a scalar index contributes none but eats a source axis; and any source
    axis the subscript never mentions is an implicit trailing full slice.
    """
    rank = 0
    src = 0
    grouped = False
    for ax in axes:
        if isinstance(ax, ast.Constant) and ax.value is None:
            rank += 1
            continue
        if isinstance(ax, ast.Slice):
            rank += 1
            src += 1
            continue
        adv = (
            len(shape_table[ax.id])
            if isinstance(ax, ast.Name) and shape_table.get(ax.id)
            else advanced_index_rank(ax, shape_table)
        )
        if adv:
            if not grouped:
                rank += adv
                grouped = True
            src += 1
            continue
        src += 1
    return rank + max(0, (len(shape) if shape else 0) - src)


def scalarize_at_iters(expr: ast.expr, iters: list[ast.expr], shape_table: dict[str, tuple[str, ...]]) -> ast.expr:
    """Render an array-valued expression at the given iter indices. Recursive
    structural lowering, independent of any one numpy op: ``Name(A)`` ->
    ``A[iters]``; ``Subscript(A, axes)`` -> walk axes, each Slice axis consumes
    one iter (offset by ``slice.lower``), scalar axes kept as-is;
    ``BinOp``/``UnaryOp``/``Call``/``IfExp`` recurse on children; ``Constant``
    unchanged.
    """
    if isinstance(expr, ast.Constant):
        return expr
    if isinstance(expr, ast.Name):
        shape = shape_table.get(expr.id)
        if shape is None:
            return expr
        if len(shape) > len(iters):
            return expr
        # Right-align the operand's axes against the iter nest (numpy broadcasts
        # along the leading axes); index any size-1 axis with constant 0, since a
        # length-1 axis broadcasts and must not consume the iter. softmax/mlp's
        # keepdims ``tmp_max`` is (N, H, SM, 1) and must read ``tmp_max[i, j, k,
        # 0]``, not ``tmp_max[..., r3]`` out of bounds.
        offset = len(iters) - len(shape)
        elts = [ast.Constant(value=0) if s == "1" else iters[offset + i] for i, s in enumerate(shape)]
        slot = elts[0] if len(elts) == 1 else ast.Tuple(elts=list(elts), ctx=ast.Load())
        return ast.Subscript(value=expr, slice=slot, ctx=ast.Load())
    if isinstance(expr, ast.Subscript):
        name = name_id(expr.value)
        shape = shape_table.get(name) if name else None
        axes = slice_axes(expr)
        # A subscript on a COMPUTED base (``(reduce_shape != 0)[:, None]``) has no name to index --
        # the BASE is the array. When the subscript is a pure broadcast-reshape (only full slices
        # and newaxis), render it by scalarising the base at the iters its full-slice axes map to;
        # the newaxis axes consume an iter and contribute nothing. Left to the axis walk below the
        # base stayed whole-array under a scalar subscript and reached C as ``(ptr != 0)[i]``, which
        # C++ rejects outright and C compiles into a pointer read. The two sibling rewriters
        # (``SliceToScalarRewriter.visit_Subscript``, ``SubscriptifyNames.visit_Subscript``)
        # already apply this rule to the same spelling; np.where's cond reached neither.
        full = [isinstance(a, ast.Slice) and a.lower is None and a.upper is None and a.step is None for a in axes]
        newax = [isinstance(a, ast.Constant) and a.value is None for a in axes]
        if name is None and axes and all(f or n for f, n in zip(full, newax)) and any(full) and len(axes) <= len(iters):
            offset = len(iters) - len(axes)
            base_iters = [iters[offset + k] for k, f in enumerate(full) if f]
            return scalarize_at_iters(expr.value, base_iters, shape_table)
        new_axes: list[ast.expr] = []
        # numpy broadcasts RIGHT-aligned: an operand whose result rank is below the nest's reads
        # the TRAILING iters. The bare-Name branch above already offsets for that; this one started
        # at iter 0, so ``np.where(cond4d, cxyz[:a, :b, :c], 0)`` read cxyz at the OUTER three loops
        # and every element came from the wrong plane. Equal ranks give offset 0, which is the
        # arithmetic that was already happening.
        iter_idx = max(0, len(iters) - subscript_result_rank(axes, shape, shape_table))
        src_axis = 0  # source-axis pointer (see _iter_extent_of).
        group_iters: list[ast.expr] | None = None  # shared advanced-index iters
        for ax in axes:
            if isinstance(ax, ast.Constant) and ax.value is None:
                # newaxis -- consume one iter from the result-axis side but
                # contribute no source index. The size-1 result axis maps
                # every read to ``source[...]`` (constant).
                if iter_idx < len(iters):
                    iter_idx += 1
                continue
            if isinstance(ax, ast.Slice):
                axis_len = const_or_name(shape[src_axis]) if shape and src_axis < len(shape) else None
                step = slice_step_any(ax)
                lo = slice_start(ax, axis_len, step)
                if iter_idx >= len(iters):
                    return expr  # not enough iters supplied
                ivar = iters[iter_idx]
                iter_idx += 1
                new_axes.append(strided_index(ivar, lo, step))
            elif isinstance(ax, ast.Name) and shape_table.get(ax.id):
                # Fancy-index gather: ``arr[idx]`` -> ``arr[idx[k]]``. Multiple index
                # arrays in one subscript form a numpy advanced-index GROUP that
                # broadcasts to a single result-axis set and SHARES the iters:
                # ``u2[q, r, s]`` -> ``u2[q[m], r[m], s[m]]`` (one iter ``m``, not
                # three). Consumed once, at the first index-array axis, reused after.
                idx_shape = shape_table[ax.id]
                if group_iters is None:
                    if iter_idx + len(idx_shape) > len(iters):
                        return expr  # not enough iters supplied
                    group_iters = iters[iter_idx : iter_idx + len(idx_shape)]
                    iter_idx += len(idx_shape)
                idx_iters = group_iters[-len(idx_shape) :]
                if len(idx_iters) == 1:
                    new_axes.append(ast.Subscript(value=ax, slice=idx_iters[0], ctx=ast.Load()))
                else:
                    new_axes.append(
                        ast.Subscript(value=ax, slice=ast.Tuple(elts=list(idx_iters), ctx=ast.Load()), ctx=ast.Load())
                    )
                src_axis += 1
                continue
            else:
                # Advanced-index EXPRESSION axis (``edge_idx[:, :, 0] - 1``):
                # part of the same broadcast group as any bare-Name index, with
                # shared iters. Recurse to scalarize its nested slices.
                adv_rank = advanced_index_rank(ax, shape_table)
                if adv_rank:
                    if group_iters is None:
                        if iter_idx + adv_rank > len(iters):
                            return expr  # not enough iters supplied
                        group_iters = iters[iter_idx : iter_idx + adv_rank]
                        iter_idx += adv_rank
                    idx_iters = group_iters[-adv_rank:]
                    new_axes.append(scalarize_at_iters(ax, idx_iters, shape_table))
                    src_axis += 1
                    continue
                # Concrete scalar index -- resolve a negative ``arr[-1]`` against
                # the axis length (C / Fortran have no negative indexing): the
                # stencil_*_vc ``w_dist[-1]`` last-weight read.
                axis_len = const_or_name(shape[src_axis]) if shape and src_axis < len(shape) else None
                new_axes.append(resolve_negative(ax, axis_len))
            src_axis += 1
        # If the source has more axes than the Subscript covered, the iter
        # nest may carry additional trailing iters that map straight to
        # the missing source axes.
        while src_axis < (len(shape) if shape else 0) and iter_idx < len(iters):
            new_axes.append(iters[iter_idx])
            iter_idx += 1
            src_axis += 1
        if not new_axes:
            return expr.value
        slot = new_axes[0] if len(new_axes) == 1 else ast.Tuple(elts=new_axes, ctx=ast.Load())
        return ast.Subscript(value=expr.value, slice=slot, ctx=ast.Load())
    if isinstance(expr, ast.BinOp):
        return ast.BinOp(
            left=scalarize_at_iters(expr.left, iters, shape_table),
            op=expr.op,
            right=scalarize_at_iters(expr.right, iters, shape_table),
        )
    if isinstance(expr, ast.UnaryOp):
        return ast.UnaryOp(op=expr.op, operand=scalarize_at_iters(expr.operand, iters, shape_table))
    if isinstance(expr, ast.Compare):
        return ast.Compare(
            left=scalarize_at_iters(expr.left, iters, shape_table),
            ops=expr.ops,
            comparators=[scalarize_at_iters(c, iters, shape_table) for c in expr.comparators],
        )
    if isinstance(expr, ast.BoolOp):
        return ast.BoolOp(op=expr.op, values=[scalarize_at_iters(v, iters, shape_table) for v in expr.values])
    if isinstance(expr, ast.IfExp):
        return ast.IfExp(
            test=scalarize_at_iters(expr.test, iters, shape_table),
            body=scalarize_at_iters(expr.body, iters, shape_table),
            orelse=scalarize_at_iters(expr.orelse, iters, shape_table),
        )
    if isinstance(expr, ast.Call):
        # An array CONSTRUCTOR is not elementwise: every element of ``np.zeros_like(a)`` is 0,
        # whatever ``a`` is. Recursing into the args instead emitted a per-element call to the
        # constructor itself (``__t[i, j] = np.zeros_like(...)``), which no backend can render.
        fill = ctor_fill_element(expr)
        if fill is not None:
            return fill
        # Math intrinsics on array values fall through; the args are
        # array expressions to scalarize.
        return ast.Call(
            func=expr.func, args=[scalarize_at_iters(a, iters, shape_table) for a in expr.args], keywords=expr.keywords
        )
    return expr
