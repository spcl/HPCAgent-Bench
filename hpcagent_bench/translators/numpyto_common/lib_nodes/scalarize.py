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
    ``start + ivar * step``. Must stay in lockstep with :func:`iter_extent_of`,
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
    if isinstance(expr, ast.Name):
        return scalarize_name(expr, iters, shape_table)
    if isinstance(expr, ast.Subscript):
        return scalarize_subscript(expr, iters, shape_table)
    if isinstance(expr, ast.Call):
        # An array CONSTRUCTOR is not elementwise: every element of ``np.zeros_like(a)`` is 0,
        # whatever ``a`` is -- never a per-element call to the constructor itself.
        fill = ctor_fill_element(expr)
        if fill is not None:
            return fill
        # Math intrinsics on array values fall through; the args are
        # array expressions to scalarize.
        return ast.Call(
            func=expr.func, args=[scalarize_at_iters(a, iters, shape_table) for a in expr.args], keywords=expr.keywords
        )
    return scalarize_children(expr, iters, shape_table)


def scalarize_children(expr: ast.expr, iters: list[ast.expr], shape_table: dict[str, tuple[str, ...]]) -> ast.expr:
    """An operator node rebuilt over its scalarised operands; anything else (a Constant) unchanged."""
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
    return expr


def scalarize_name(expr: ast.Name, iters: list[ast.expr], shape_table: dict[str, tuple[str, ...]]) -> ast.expr:
    """``A`` -> ``A[iters]``, the operand's axes right-aligned against the iter nest (numpy broadcasts
    along the leading axes). A size-1 axis broadcasts and is indexed with constant 0 instead of
    consuming an iter: a keepdims ``tmp_max`` of (N, H, SM, 1) reads ``tmp_max[i, j, k, 0]``."""
    shape = shape_table.get(expr.id)
    if shape is None or len(shape) > len(iters):
        return expr
    offset = len(iters) - len(shape)
    elts = [ast.Constant(value=0) if s == "1" else iters[offset + i] for i, s in enumerate(shape)]
    slot = elts[0] if len(elts) == 1 else ast.Tuple(elts=list(elts), ctx=ast.Load())
    return ast.Subscript(value=expr, slice=slot, ctx=ast.Load())


def scalarize_subscript(
    expr: ast.Subscript, iters: list[ast.expr], shape_table: dict[str, tuple[str, ...]]
) -> ast.expr:
    """``A[axes]`` at the iters: each axis indexed by :class:`SubscriptIndexer`, the source's uncovered
    trailing axes taking the remaining iters. numpy broadcasts RIGHT-aligned, so a subscript whose
    result rank is below the nest's reads the TRAILING iters.

    A subscript on a COMPUTED base (``(reduce_shape != 0)[:, None]``) has no name to index -- the BASE
    is the array. When it is a pure broadcast-reshape (only full slices and newaxis) the base is
    scalarised at the iters its full-slice axes map to; the newaxis axes consume an iter and
    contribute nothing. ``SliceToScalarRewriter.visit_Subscript`` and
    ``SubscriptifyNames.visit_Subscript`` apply the same rule to the same spelling."""
    name = name_id(expr.value)
    shape = shape_table.get(name) if name else None
    axes = slice_axes(expr)
    full = [isinstance(a, ast.Slice) and a.lower is None and a.upper is None and a.step is None for a in axes]
    newax = [isinstance(a, ast.Constant) and a.value is None for a in axes]
    if name is None and axes and all(f or n for f, n in zip(full, newax)) and any(full) and len(axes) <= len(iters):
        offset = len(iters) - len(axes)
        base_iters = [iters[offset + k] for k, f in enumerate(full) if f]
        return scalarize_at_iters(expr.value, base_iters, shape_table)
    indexer = SubscriptIndexer(
        iters, max(0, len(iters) - subscript_result_rank(axes, shape, shape_table)), shape, shape_table
    )
    for ax in axes:
        if not indexer.index(ax):
            return expr  # not enough iters supplied
    indexer.fill_trailing()
    if not indexer.axes:
        return expr.value
    slot = indexer.axes[0] if len(indexer.axes) == 1 else ast.Tuple(elts=indexer.axes, ctx=ast.Load())
    return ast.Subscript(value=expr.value, slice=slot, ctx=ast.Load())


class SubscriptIndexer:
    """One subscript's axis walk: the next iter to consume, the next source axis, the index axes
    built so far, and the iters shared by the subscript's advanced-index group.

    Several integer-array indices form ONE numpy advanced-index group that broadcasts to a single
    result-axis set and SHARES its iters: ``u2[q, r, s]`` -> ``u2[q[m], r[m], s[m]]`` (one iter
    ``m``, not three), consumed at the first index-array axis and reused after.
    """

    def __init__(
        self,
        iters: list[ast.expr],
        iter_idx: int,
        shape: tuple[str, ...] | None,
        shape_table: dict[str, tuple[str, ...]],
    ) -> None:
        self.iters = iters
        self.iter_idx = iter_idx
        self.shape = shape
        self.shape_table = shape_table
        self.src_axis = 0
        self.group_iters: list[ast.expr] | None = None
        self.axes: list[ast.expr] = []

    def axis_len(self) -> ast.expr | None:
        if self.shape and self.src_axis < len(self.shape):
            return const_or_name(self.shape[self.src_axis])
        return None

    def index(self, ax: ast.expr) -> bool:
        """Append ``ax``'s index; False when the iters run out."""
        if isinstance(ax, ast.Constant) and ax.value is None:
            # newaxis -- consumes one iter from the result-axis side, contributes no source index.
            if self.iter_idx < len(self.iters):
                self.iter_idx += 1
            return True
        if isinstance(ax, ast.Slice):
            step = slice_step_any(ax)
            lo = slice_start(ax, self.axis_len(), step)
            if self.iter_idx >= len(self.iters):
                return False
            self.axes.append(strided_index(self.iters[self.iter_idx], lo, step))
            self.iter_idx += 1
        elif isinstance(ax, ast.Name) and self.shape_table.get(ax.id):
            # Fancy-index gather: ``arr[idx]`` -> ``arr[idx[k]]``.
            idx_iters = self.group(len(self.shape_table[ax.id]))
            if idx_iters is None:
                return False
            slot = idx_iters[0] if len(idx_iters) == 1 else ast.Tuple(elts=list(idx_iters), ctx=ast.Load())
            self.axes.append(ast.Subscript(value=ax, slice=slot, ctx=ast.Load()))
        elif adv_rank := advanced_index_rank(ax, self.shape_table):
            # Advanced-index EXPRESSION axis (``edge_idx[:, :, 0] - 1``): in the same group, its
            # nested slices scalarised at the shared iters.
            idx_iters = self.group(adv_rank)
            if idx_iters is None:
                return False
            self.axes.append(scalarize_at_iters(ax, idx_iters, self.shape_table))
        else:
            # Concrete scalar index -- a negative ``arr[-1]`` resolved against the axis length
            # (C / Fortran have no negative indexing).
            self.axes.append(resolve_negative(ax, self.axis_len()))
        self.src_axis += 1
        return True

    def group(self, rank: int) -> list[ast.expr] | None:
        """The advanced-index group's trailing ``rank`` iters, taking them on first use."""
        if self.group_iters is None:
            if self.iter_idx + rank > len(self.iters):
                return None
            self.group_iters = self.iters[self.iter_idx : self.iter_idx + rank]
            self.iter_idx += rank
        return self.group_iters[-rank:]

    def fill_trailing(self) -> None:
        """Source axes the subscript did not cover take the nest's remaining trailing iters."""
        while self.src_axis < (len(self.shape) if self.shape else 0) and self.iter_idx < len(self.iters):
            self.axes.append(self.iters[self.iter_idx])
            self.iter_idx += 1
            self.src_axis += 1
