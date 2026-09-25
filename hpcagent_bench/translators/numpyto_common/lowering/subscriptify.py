"""Bare array names inside loops rewritten to element subscripts."""

import ast
import copy
from types import NotImplementedType

from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import iter_extent_of, extent_is_scalar
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import (
    slice_step_any,
    step_is_negative,
    step_node,
    const_or_name,
)
from hpcagent_bench.translators.numpyto_common.lowering.indexing import advanced_runs
from hpcagent_bench.translators.numpyto_common.lowering.shape_reads import is_newaxis
from hpcagent_bench.translators.numpyto_common.subscripts import is_full_slice


def resolve_neg_index(idx: ast.expr, axis_len: ast.expr) -> ast.expr:
    """Resolve a negative constant array index to ``axis_len - K``.

    numpy ``arr[-1]`` reads the last element; C / Fortran have no negative
    indexing, so a literal ``-K`` must become ``dim - K`` (the stencils'
    ``w_dist[-1]`` last-weight read). Both spellings -- ``Constant(-K)`` and
    ``UnaryOp(USub, Constant(K))`` -- are handled; everything else passes
    through unchanged."""
    k = None
    if isinstance(idx, ast.Constant) and isinstance(idx.value, int) and idx.value < 0:
        k = -idx.value
    elif (
        isinstance(idx, ast.UnaryOp)
        and isinstance(idx.op, ast.USub)
        and isinstance(idx.operand, ast.Constant)
        and isinstance(idx.operand.value, int)
    ):
        k = idx.operand.value
    if k is None:
        return idx
    return ast.BinOp(left=copy.deepcopy(axis_len), op=ast.Sub(), right=ast.Constant(value=k))


#: A :class:`SubscriptifyNames` step that does not apply to the subscript.
UNHANDLED = NotImplemented


class SubscriptifyNames(ast.NodeTransformer):
    """Rewrite ``Name(arr)`` references whose shape matches the loop
    nest's bounds into ``Subscript(arr, idx)``."""

    def __init__(self, shape_table, iters) -> None:
        self.shape_table = shape_table
        self.iters = iters

    def visit_Name(self, node: ast.Name) -> ast.AST:
        shape = self.shape_table.get(node.id)
        if not shape:
            return node
        if len(shape) > len(self.iters):
            return node
        # Build the subscript: right-align with the iter nest; each
        # axis subscripts the corresponding iter, EXCEPT size-1
        # axes which subscript constant 0 (broadcast).
        offset = len(self.iters) - len(shape)
        elts = []
        for i, s in enumerate(shape):
            if s == "1":
                elts.append(ast.Constant(value=0))
            else:
                elts.append(ast.Name(id=self.iters[offset + i], ctx=ast.Load()))
        idx = elts[0] if len(elts) == 1 else ast.Tuple(elts=elts, ctx=ast.Load())
        return ast.Subscript(value=node, slice=idx, ctx=ast.Load())

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        """Scalarise subscripts that contain ``:`` slices.

        Forms handled:

        * ``arr[:]`` / ``arr[:, :]`` (all-slice): equivalent to a bare
          Name reference; subscript with the iter vars right-aligned.
        * ``arr[:, j]`` / ``arr[i, :]`` (one slice + one index): replace
          each ``:`` with the next iter (in axis order); keep concrete
          indices as-is (``y -= H[j, k] * Q[:, j]`` lands Q as ``Q[__w0, j]``).
        * ``arr[:-1, :, k]`` (bounded slice + slices + concrete): each
          slice element (whether ``:`` or bounded like ``:-1`` /
          ``1:``) is substituted with the next iter -- the iter loop
          bounds already enforce the slice range, so the subscript
          just needs the iter variable. Concrete indices stay.
        """
        if isinstance(node.ctx, ast.Load):
            steps = (
                (self.fancy_gather, self.lone_slice, self.index_tuple, self.partial_scalar_index)
                if isinstance(node.value, ast.Name)
                else (self.computed_base_reshape,)
            )
            for step in steps:
                scalar = step(node)
                if scalar is not UNHANDLED:
                    return scalar
        self.generic_visit(node)
        return node

    def computed_base_reshape(self, node: ast.Subscript) -> ast.AST | NotImplementedType:
        """A subscript on a NON-Name value (a BinOp / Call result) whose slice is a pure
        broadcast-reshape -- only ``:`` and ``np.newaxis`` (``(q_nb[:, None, :] * fs)[:, :, :, None]``):
        the inner expression scalarised at the iters mapped to the ``:`` axes; each newaxis adds a
        result axis (and consumes an iter) but contributes no source axis."""
        sl0 = node.slice
        elts0 = list(sl0.elts) if isinstance(sl0, ast.Tuple) else [sl0]
        if (
            elts0
            and all(is_full_slice(e) or is_newaxis(e) for e in elts0)
            and any(is_full_slice(e) for e in elts0)
            and len(elts0) <= len(self.iters)
        ):
            offset = len(self.iters) - len(elts0)
            sub_iters = [self.iters[offset + k] for k, e in enumerate(elts0) if is_full_slice(e)]
            return SubscriptifyNames(self.shape_table, sub_iters).visit(copy.deepcopy(node.value))
        return UNHANDLED

    def fancy_gather(self, node: ast.Subscript) -> ast.AST | NotImplementedType:
        """``arr[idx]`` with an INDEX ARRAY ``idx`` (or any array-valued expression over one, such as
        ``table[ri + 1]``) -> ``arr[idx[iter...]]``: ``arr`` itself is NOT subscripted by the iter.
        numpy basic fancy indexing gathers along ``arr``'s LEADING axis -- result shape = idx.shape +
        arr.shape[1:] -- so the index consumes the first ``r`` result axes and the source's trailing
        axes the rest (``momentum[nb]`` on (ncells, 3) -> ``momentum[nb[i], j]``)."""
        sl = node.slice
        idx_shape = None
        if isinstance(sl, ast.Name) and self.shape_table.get(sl.id):
            idx_shape = self.shape_table[sl.id]
        elif not isinstance(sl, (ast.Slice, ast.Tuple)) and not is_newaxis(sl):
            idx_ext = iter_extent_of(sl, self.shape_table)
            if idx_ext is not None and not extent_is_scalar(idx_ext):
                idx_shape = tuple(ast.unparse(e) for e in idx_ext)
        if idx_shape is None:
            return UNHANDLED
        src_shape = self.shape_table.get(node.value.id)
        r = len(idx_shape)
        n_trailing = (len(src_shape) - 1) if src_shape else 0
        result_rank = r + n_trailing
        if result_rank > len(self.iters):
            return UNHANDLED
        offset = len(self.iters) - result_rank
        idx_iters = [self.iters[offset + i] for i in range(r)]
        trail_iters = [ast.Name(id=self.iters[offset + r + i], ctx=ast.Load()) for i in range(n_trailing)]
        # The index EXPRESSION scalarised at its own iters; for a bare Name this is ``sl[idx_iters]``.
        gathered = SubscriptifyNames(self.shape_table, idx_iters).visit(copy.deepcopy(sl))
        full = [gathered] + trail_iters
        slot = full[0] if len(full) == 1 else ast.Tuple(elts=full, ctx=ast.Load())
        return ast.Subscript(value=node.value, slice=slot, ctx=ast.Load())

    def lone_slice(self, node: ast.Subscript) -> ast.AST | NotImplementedType:
        """``arr[:]`` is the bare Name. A strided / reverse ``arr[::k]`` / ``arr[lo::k]`` reads ``start
        + iter*k`` (start = ``lower``, or 0 / axis_len-1 for a negative step when omitted; an
        untracked reverse is refused rather than emitted forward). A bounded ``arr[:k]`` / ``arr[a:b]``
        reads the (right-aligned) iter plus any ``lower`` -- the loop bound already enforces the range."""
        sl = node.slice
        if not isinstance(sl, ast.Slice):
            return UNHANDLED
        if sl.lower is None and sl.upper is None and sl.step is None:
            return self.visit_Name(node.value)
        step = slice_step_any(sl)
        if step is not None and step != 1 and self.iters:
            start: ast.expr | None = sl.lower
            if start is None and step_is_negative(step):
                sh = self.shape_table.get(node.value.id)
                if sh:
                    al = (
                        ast.Constant(value=int(sh[0]))
                        if str(sh[0]).isdigit()
                        else ast.Name(id=str(sh[0]), ctx=ast.Load())
                    )
                    start = ast.BinOp(left=al, op=ast.Sub(), right=ast.Constant(value=1))
            if step_is_negative(step) and start is None:
                raise NotImplementedError(
                    f"reverse slice of {node.value.id!r} needs a known axis length (shape untracked)"
                )
            iterv: ast.expr = ast.Name(id=self.iters[-1], ctx=ast.Load())
            scaled: ast.expr = ast.BinOp(left=iterv, op=ast.Mult(), right=step_node(step))
            idx = scaled if start is None else ast.BinOp(left=scaled, op=ast.Add(), right=start)
            return ast.Subscript(value=node.value, slice=idx, ctx=ast.Load())
        if self.iters:
            iter_node: ast.expr = ast.Name(id=self.iters[-1], ctx=ast.Load())
            if sl.lower is not None and not (isinstance(sl.lower, ast.Constant) and sl.lower.value == 0):
                iter_node = ast.BinOp(left=iter_node, op=ast.Add(), right=sl.lower)
            return ast.Subscript(value=node.value, slice=iter_node, ctx=ast.Load())
        return UNHANDLED

    def index_tuple(self, node: ast.Subscript) -> ast.AST | NotImplementedType:
        """A tuple subscript: all ``:`` is the bare Name; one with index arrays goes through
        :meth:`advanced_tuple`; a mix of slices and concrete indices through :meth:`sliced_tuple`."""
        sl = node.slice
        if not (isinstance(sl, ast.Tuple) and sl.elts):
            return UNHANDLED
        if all(isinstance(e, ast.Slice) and e.lower is None and e.upper is None and e.step is None for e in sl.elts):
            return self.visit_Name(node.value)
        if any(self.index_rank(e) >= 1 for e in sl.elts):
            scalar = self.advanced_tuple(node)
            if scalar is not UNHANDLED:
                return scalar
        if any(isinstance(e, ast.Slice) for e in sl.elts):
            return self.sliced_tuple(node)
        return UNHANDLED

    def index_rank(self, e: ast.expr) -> int:
        """Rank of an advanced index: an index array's rank, or an index EXPRESSION's over one
        (``dxa[ib - 1, :, :]``) -- advanced indexing exactly as the bare ``ib`` is; 0 otherwise."""
        if isinstance(e, ast.Name):
            return len(self.shape_table.get(e.id) or ())
        if isinstance(e, (ast.Slice, ast.Constant)) or is_newaxis(e):
            return 0
        ext = iter_extent_of(e, self.shape_table)
        return len(ext) if ext is not None and not extent_is_scalar(ext) else 0

    def advanced_tuple(self, node: ast.Subscript) -> ast.AST | NotImplementedType:
        """Advanced indices (index arrays AND plain scalars -- numpy counts a bare integer as advanced
        next to an index array) that are ADJACENT BROADCAST into one shared block of result axes; a
        Slice keeps its own axis (``x1[:, PERM]`` with PERM (8,6) -> ``x1[w0, PERM[w1, w2]]``;
        ``A[idx, lev, blk]`` is rank 3, not the sum). Each operand in the run right-aligns against the
        SAME shared iters. Advanced indices separated by a slice / newaxis are refused."""
        sl = node.slice
        runs = advanced_runs(sl.elts)
        if len(runs) > 1:
            raise NotImplementedError(
                f"advanced indices of {node.value.id!r} separated by a slice/newaxis "
                f"({ast.unparse(node)!r}) -- broadcast-to-front placement is not implemented"
            )
        run = set(runs[0])
        run_rank = max((self.index_rank(sl.elts[i]) for i in run), default=0)
        result_axis_count = len(sl.elts) - len(run) + run_rank
        if result_axis_count > len(self.iters):
            return UNHANDLED
        offset = len(self.iters) - result_axis_count
        pos = 0
        giters: list[str] | None = None
        new_elts = []
        for i, e in enumerate(sl.elts):
            if i in run:
                if giters is None:
                    giters = self.iters[offset + pos : offset + pos + run_rank]
                    pos += run_rank
                new_elts.append(SubscriptifyNames(self.shape_table, giters).visit(copy.deepcopy(e)))
                continue
            if isinstance(e, ast.Constant) and e.value is None:
                pos += 1
                continue
            it = ast.Name(id=self.iters[offset + pos], ctx=ast.Load())
            pos += 1
            if (
                isinstance(e, ast.Slice)
                and e.lower is not None
                and not (isinstance(e.lower, ast.Constant) and e.lower.value == 0)
            ):
                it = ast.BinOp(left=it, op=ast.Add(), right=e.lower)
            new_elts.append(it)
        slot = new_elts[0] if len(new_elts) == 1 else ast.Tuple(elts=new_elts, ctx=ast.Load())
        return ast.Subscript(value=node.value, slice=slot, ctx=ast.Load())

    def sliced_tuple(self, node: ast.Subscript) -> ast.AST | NotImplementedType:
        """Slices and concrete indices: each Slice (full or bounded) takes the next iter in axis order,
        right-aligned, plus any ``lower``; a concrete index stays. A newaxis consumes a result axis
        (and its iter) but no source index, so the right-alignment counts it. A source axis the array
        declares as 1 BROADCASTS along its result axis and pins to 0, as :meth:`visit_Name` does for
        a bare Name (``pressure[..., None]`` of an ``(ncells, 1)`` array under an ``(ncells, 4, 3)``
        nest would otherwise read the next cell's row)."""
        sl = node.slice
        n_slices = sum(1 for e in sl.elts if isinstance(e, ast.Slice))
        n_newaxis = sum(1 for e in sl.elts if isinstance(e, ast.Constant) and e.value is None)
        result_rank = n_slices + n_newaxis
        if result_rank > len(self.iters):
            return UNHANDLED
        axis_pos = len(self.iters) - result_rank
        new_elts: list[ast.expr] = []
        src_shape = self.shape_table.get(node.value.id)
        src_axis = 0  # the SOURCE axis each element reads: a slice or an index consumes one
        for e in sl.elts:
            if isinstance(e, ast.Slice):
                if src_shape and src_axis < len(src_shape) and str(src_shape[src_axis]).strip() == "1":
                    new_elts.append(ast.Constant(value=0))
                    axis_pos += 1
                    src_axis += 1
                    continue
                iter_node: ast.expr = ast.Name(id=self.iters[axis_pos], ctx=ast.Load())
                axis_pos += 1
                src_axis += 1
                if e.lower is not None and not (isinstance(e.lower, ast.Constant) and e.lower.value == 0):
                    iter_node = ast.BinOp(left=iter_node, op=ast.Add(), right=e.lower)
                new_elts.append(iter_node)
            elif isinstance(e, ast.Constant) and e.value is None:
                axis_pos += 1
            else:
                new_elts.append(e)
                src_axis += 1
        if not new_elts:
            return ast.Name(id=node.value.id, ctx=ast.Load())
        new_slot = new_elts[0] if len(new_elts) == 1 else ast.Tuple(elts=new_elts, ctx=ast.Load())
        return ast.Subscript(value=node.value, slice=new_slot, ctx=ast.Load())

    def partial_scalar_index(self, node: ast.Subscript) -> ast.AST | NotImplementedType:
        """A partial scalar index on a HIGHER-rank array: numpy ``Ham[n]`` == ``Ham[n, :, :]``, so the
        trailing axes take the right-aligned iters (otherwise the read stays rank-1: a Fortran rank
        mismatch, silently wrong numerics in C). A subscript that already FULLY indexes the array
        with scalars (``w_dist[-1]``) is a scalar element read, returned with negatives resolved
        rather than letting the default visit subscript the base Name. Index-array axes keep the
        generic gather path."""
        sl = node.slice
        lead = list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]
        if not (
            lead
            and not any(isinstance(e, ast.Slice) for e in lead)
            and not any(isinstance(e, ast.Constant) and e.value is None for e in lead)
            and not any(self.reads_index_array(e) for e in lead)
        ):
            return UNHANDLED
        shape = self.shape_table.get(node.value.id)
        if shape is None:
            return UNHANDLED
        n_trailing = len(shape) - len(lead)
        # C / Fortran have no negative indexing.
        res_lead = [resolve_neg_index(e, const_or_name(shape[ax])) for ax, e in enumerate(lead)]
        if 0 < n_trailing <= len(self.iters):
            offset = len(self.iters) - n_trailing
            new_elts = list(res_lead) + [ast.Name(id=self.iters[offset + j], ctx=ast.Load()) for j in range(n_trailing)]
            return ast.Subscript(value=node.value, slice=ast.Tuple(elts=new_elts, ctx=ast.Load()), ctx=ast.Load())
        if n_trailing == 0:
            new_slot = res_lead[0] if len(res_lead) == 1 else ast.Tuple(elts=res_lead, ctx=ast.Load())
            return ast.Subscript(value=node.value, slice=new_slot, ctx=ast.Load())
        return UNHANDLED

    def reads_index_array(self, e: ast.expr) -> bool:
        """An advanced-index axis references an index ARRAY (a Name whose own shape is known)."""
        return any(isinstance(s, ast.Name) and self.shape_table.get(s.id) for s in ast.walk(e))
