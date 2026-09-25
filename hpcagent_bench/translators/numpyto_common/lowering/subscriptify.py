"""Bare array names inside loops rewritten to element subscripts."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import iter_extent_of_, extent_is_scalar
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import slice_step_any, step_is_negative, step_node
from hpcagent_bench.translators.numpyto_common.lowering.indexing import advanced_runs
from hpcagent_bench.translators.numpyto_common.lowering.shape_reads import is_newaxis, token_to_ast
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
          indices as-is. Required for the gmres
          ``y -= H[j, k] * Q[:, j]`` shape so Q lands as ``Q[__w0, j]``.
        * ``arr[:-1, :, k]`` (bounded slice + slices + concrete): each
          slice element (whether ``:`` or bounded like ``:-1`` /
          ``1:``) is substituted with the next iter -- the iter loop
          bounds already enforce the slice range, so the subscript
          just needs the iter variable. Concrete indices stay.
          Required for vadv's ``u_stage[:-1, :, k]`` form.
        """
        # Subscript on a NON-Name value (a BinOp / Call result) whose slice is
        # a pure broadcast-reshape -- only ``:`` and ``np.newaxis``:
        # ``(q_nb[:, None, :] * fs)[:, :, :, None]`` (lavamd force term).
        # Recursively scalarise the inner expression at the iters mapped to the
        # ``:`` axes; each newaxis adds a result axis (and consumes an iter) but
        # contributes no source axis. Handled here because the inner value is
        # not a Name, so the Name paths below don't apply.
        if not isinstance(node.value, ast.Name) and isinstance(node.ctx, ast.Load):
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
        if isinstance(node.value, ast.Name) and isinstance(node.ctx, ast.Load):
            sl = node.slice
            # Fancy-index gather: ``arr[idx]`` where ``idx`` is an INDEX ARRAY
            # (it has its own shape) -> ``arr[idx[iter...]]``. The gathered
            # value indexes ``arr``; ``arr`` itself is NOT subscripted by the
            # iter (the bug otherwise: generic_visit would emit
            # ``arr[iter][idx[iter]]``). The index array's rank consumes the
            # right-aligned iters. edge_laplacian's ``x[src]`` / ``x[dst]``.
            # The index may be a bare Name OR any array-valued EXPRESSION over one
            # (``coulomb_table_f[ri + 1]``). Only the Name form was recognised, so an
            # offset gather fell through to generic_visit, which subscripted the BASE and
            # emitted ``coulomb_table_f[iter][ri[iters] + 1]``.
            idx_shape = None
            if isinstance(sl, ast.Name) and self.shape_table.get(sl.id):
                idx_shape = self.shape_table[sl.id]
            elif not isinstance(sl, (ast.Slice, ast.Tuple)) and not is_newaxis(sl):
                idx_ext = iter_extent_of_(sl, self.shape_table)
                if idx_ext is not None and not extent_is_scalar(idx_ext):
                    idx_shape = tuple(ast.unparse(e) for e in idx_ext)
            if idx_shape is not None:
                src_shape = self.shape_table.get(node.value.id)
                # numpy basic fancy indexing ``arr[idx]`` with a single index
                # array ``idx`` (rank r) gathers along ``arr``'s LEADING axis:
                # result shape = idx.shape + arr.shape[1:]. So the index array
                # consumes the first ``r`` result axes and the source's
                # remaining trailing axes (arr.shape[1:]) consume the rest --
                # ``momentum[nb]`` on (ncells, 3) -> ``momentum[nb[i], j]``,
                # not the (1-D-only) ``momentum[nb[j]]``. cfd / lavamd.
                r = len(idx_shape)
                n_trailing = (len(src_shape) - 1) if src_shape else 0
                result_rank = r + n_trailing
                if result_rank <= len(self.iters):
                    offset = len(self.iters) - result_rank
                    idx_iters = [self.iters[offset + i] for i in range(r)]
                    trail_iters = [ast.Name(id=self.iters[offset + r + i], ctx=ast.Load()) for i in range(n_trailing)]
                    # Scalarise the index EXPRESSION at its own iters; for a bare Name this is
                    # exactly the ``sl[idx_iters]`` the Name-only path built.
                    gathered = SubscriptifyNames(self.shape_table, idx_iters).visit(copy.deepcopy(sl))
                    full = [gathered] + trail_iters
                    slot = full[0] if len(full) == 1 else ast.Tuple(elts=full, ctx=ast.Load())
                    return ast.Subscript(value=node.value, slice=slot, ctx=ast.Load())
            if isinstance(sl, ast.Slice):
                if sl.lower is None and sl.upper is None and sl.step is None:
                    return self.visit_Name(node.value)
                step = slice_step_any(sl)
                if step is not None and step != 1 and self.iters:
                    # Strided / reverse lone slice ``arr[::k]`` / ``arr[lo::k]``: source
                    # index = start + iter*k, where start is ``lower``, or 0 (positive step)
                    # / axis_len-1 (negative step -- ``arr[::-1]``) when omitted.
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
                    # A negative step with an UNRESOLVED start (axis length not tracked)
                    # cannot emit the reverse index ``(len-1) - iter*|step|``. Falling through
                    # to the bounded path below would emit a FORWARD ``arr[iter]`` -- a silent
                    # un-reversed copy. Refuse loudly instead (mirrors _SliceToScalarRewriter,
                    # which raises for the identical untracked-shape reverse). Positive strided
                    # slices (start None, step > 0) are fine: idx = iter*step is forward.
                    if step_is_negative(step) and start is None:
                        raise NotImplementedError(
                            f"reverse slice of {node.value.id!r} needs a known axis length (shape untracked)"
                        )
                    iterv: ast.expr = ast.Name(id=self.iters[-1], ctx=ast.Load())
                    scaled: ast.expr = ast.BinOp(left=iterv, op=ast.Mult(), right=step_node(step))
                    idx = scaled if start is None else ast.BinOp(left=scaled, op=ast.Add(), right=start)
                    return ast.Subscript(value=node.value, slice=idx, ctx=ast.Load())
                # Bounded lone slice ``arr[:k]`` / ``arr[a:b]`` / ``arr[1:]``
                # on a 1-D array: the iter loop bound already enforces the
                # slice range, so replace the slice with the (right-aligned)
                # next iter, adding any ``lower`` offset. Required for
                # durbin's ``__cb[:] = r[:k]`` copy. (Negative ``lower``
                # like ``arr[-K:]`` is uncommon and not handled here.)
                if self.iters:
                    iter_node: ast.expr = ast.Name(id=self.iters[-1], ctx=ast.Load())
                    if sl.lower is not None and not (isinstance(sl.lower, ast.Constant) and sl.lower.value == 0):
                        iter_node = ast.BinOp(left=iter_node, op=ast.Add(), right=sl.lower)
                    return ast.Subscript(value=node.value, slice=iter_node, ctx=ast.Load())
            elif isinstance(sl, ast.Tuple) and sl.elts:
                # All-slice fast path -- restored bare-Name behaviour.
                all_slice = all(
                    isinstance(e, ast.Slice) and e.lower is None and e.upper is None and e.step is None for e in sl.elts
                )
                if all_slice:
                    return self.visit_Name(node.value)

                # Slice-or-index form: every element is either a Slice
                # (full ``:`` OR bounded ``:-1`` / ``a:b``) or a
                # non-Slice concrete index. Substitute each Slice with
                # the next iter (in axis order, right-aligned).
                # Advanced indices (index arrays AND plain scalars -- numpy counts
                # a bare integer as "advanced" too when it sits next to an index
                # array) that are ADJACENT to each other BROADCAST into one shared
                # block of result axes; a Slice keeps its own axis. Handles a
                # single array on any axis (lulesh ``x1[:, _VOLU_PERM]`` with
                # _VOLU_PERM (8,6) -> ``x1[w0, _VOLU_PERM[w1, w2]]``), several
                # adjacent arrays broadcasting together (icon_gather's
                # ``A[idx, lev, blk]`` -> rank 3, not the sum 9), and no Slice at
                # all (the whole subscript is one adjacent group).
                def idx_rank(e):
                    if isinstance(e, ast.Name):
                        return len(self.shape_table.get(e.id) or ())
                    # An index EXPRESSION over an index array (``dxa[ib - 1, :, :]``) is advanced
                    # indexing exactly as the bare ``ib`` is -- the lone-index gather above already
                    # reads it that way. Recognised only here, it fell through to the slice path,
                    # which copied the expression through untouched; the NEXT scalarising pass then
                    # saw a bare rank-1 ``ib`` under a rank-3 nest and right-aligned it onto the
                    # LAST iter. fv3_xppm's edge columns read ``ib[k]`` over the vertical extent --
                    # a wrong answer, and out of bounds as soon as nk exceeds the index array.
                    if isinstance(e, (ast.Slice, ast.Constant)) or is_newaxis(e):
                        return 0
                    ext = iter_extent_of_(e, self.shape_table)
                    return len(ext) if ext is not None and not extent_is_scalar(ext) else 0

                def is_index_array(e):
                    return idx_rank(e) >= 1

                if any(is_index_array(e) for e in sl.elts):
                    runs = advanced_runs(sl.elts)
                    if len(runs) > 1:
                        raise NotImplementedError(
                            f"advanced indices of {node.value.id!r} separated by a slice/newaxis "
                            f"({ast.unparse(node)!r}) -- broadcast-to-front placement is not implemented"
                        )
                    run = set(runs[0])
                    run_rank = max((idx_rank(sl.elts[i]) for i in run), default=0)
                    n_other = len(sl.elts) - len(run)
                    result_axis_count = n_other + run_rank
                    if result_axis_count <= len(self.iters):
                        offset = len(self.iters) - result_axis_count
                        pos = 0
                        giters: list[str] | None = None
                        new_elts = []
                        for i, e in enumerate(sl.elts):
                            if i in run:
                                if giters is None:
                                    giters = self.iters[offset + pos : offset + pos + run_rank]
                                    pos += run_rank
                                # Each operand in the run right-aligns against the SAME
                                # shared iters -- a lower-rank (or scalar) operand reads
                                # only its own trailing slice of them; a size-1 own axis
                                # pins to 0 (visit_Name's existing broadcast rule, reused
                                # here since a fresh sub-rewriter just delegates to it).
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
                partial_or_bounded = all(isinstance(e, ast.Slice) or not isinstance(e, ast.Slice) for e in sl.elts)
                if partial_or_bounded and any(isinstance(e, ast.Slice) for e in sl.elts):
                    n_slices = sum(1 for e in sl.elts if isinstance(e, ast.Slice))
                    # A ``None`` (np.newaxis) inserts a length-1 RESULT axis: it
                    # consumes an output axis (and thus an iter) but contributes
                    # no source index. The right-alignment must count it, else a
                    # lone slice in ``V[:, None]`` mis-binds to the trailing iter
                    # (``V[__w1]``) instead of the leading one (``V[__w0]``).
                    n_newaxis = sum(1 for e in sl.elts if isinstance(e, ast.Constant) and e.value is None)
                    result_rank = n_slices + n_newaxis
                    if result_rank <= len(self.iters):
                        axis_pos = len(self.iters) - result_rank
                        new_elts: list[ast.expr] = []
                        # ``src_axis`` tracks the SOURCE axis each element reads: a slice or a
                        # concrete index consumes one, a newaxis consumes none.
                        src_shape = self.shape_table.get(node.value.id)
                        src_axis = 0
                        for e in sl.elts:
                            if isinstance(e, ast.Slice):
                                # A source axis the array declares as 1 BROADCASTS along the result
                                # axis it lands on -- every result position reads the same element --
                                # so it pins to 0 instead of consuming the (larger) iter.
                                # ``visit_Name`` applies this rule to a bare Name; a slice spelling
                                # of the same operand has to agree. cfd's ``pressure[..., None]`` is
                                # an ``(ncells, 1)`` array under an ``(ncells, 4, 3)`` nest: taking
                                # the extent-4 iter reads the next cell's row, and runs past the
                                # allocation at the last cell.
                                if src_shape and src_axis < len(src_shape) and str(src_shape[src_axis]).strip() == "1":
                                    new_elts.append(ast.Constant(value=0))
                                    axis_pos += 1
                                    src_axis += 1
                                    continue
                                iter_name = self.iters[axis_pos]
                                axis_pos += 1
                                src_axis += 1
                                # Add the slice's ``lower`` bound to
                                # the iter so ``arr[1:, j]`` lowers as
                                # ``arr(iter + 1, j)`` instead of
                                # ``arr(iter, j)``. Negative ``lower``
                                # (e.g. ``arr[-K:]``) is uncommon and
                                # not handled here.
                                iter_node = ast.Name(id=iter_name, ctx=ast.Load())
                                if e.lower is not None:
                                    # Constant 0 means no offset.
                                    if not (isinstance(e.lower, ast.Constant) and e.lower.value == 0):
                                        iter_node = ast.BinOp(left=iter_node, op=ast.Add(), right=e.lower)
                                new_elts.append(iter_node)
                            elif isinstance(e, ast.Constant) and e.value is None:
                                # ``None`` (np.newaxis): consume a result axis
                                # (and its iter) but add no source index.
                                axis_pos += 1
                            else:
                                new_elts.append(e)
                                src_axis += 1
                        if not new_elts:
                            return ast.Name(id=node.value.id, ctx=ast.Load())
                        new_slot = new_elts[0] if len(new_elts) == 1 else ast.Tuple(elts=new_elts, ctx=ast.Load())
                        return ast.Subscript(value=node.value, slice=new_slot, ctx=ast.Load())
            # Partial scalar index on a HIGHER-rank array: ``Ham[n]``
            # where ``Ham`` is rank-3 indexes only the leading axis and
            # the remaining axes form a slice broadcast against the loop
            # nest. numpy ``Ham[n]`` == ``Ham[n, :, :]``. Pad the
            # trailing axes with the right-aligned iter vars so the
            # scalarised subscript spans ALL of the array's axes
            # (contour_integral's ``Tz += zz * Ham[n]``). Without this
            # the read stays rank-1 (``Ham(n)`` in Fortran -> rank
            # mismatch; ``Ham[n]`` in C -> silently wrong numerics).
            lead = list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]

            def has_idx_array(e):
                # An advanced-index axis references an index ARRAY (a Name whose
                # own shape is known) -- ``arr[idx - 1, jk, blk - 1]`` (velocity /
                # icon gathers). Those must keep the generic-visit gather path.
                return any(isinstance(s, ast.Name) and self.shape_table.get(s.id) for s in ast.walk(e))

            if (
                lead
                and not any(isinstance(e, ast.Slice) for e in lead)
                and not any(isinstance(e, ast.Constant) and e.value is None for e in lead)
                and not any(has_idx_array(e) for e in lead)
            ):
                shape = self.shape_table.get(node.value.id)
                if shape is not None:
                    rank = len(shape)
                    n_trailing = rank - len(lead)
                    # Resolve any negative scalar index (``arr[-1]``) against its
                    # axis length -- C / Fortran have no negative indexing.
                    res_lead = [resolve_neg_index(e, token_to_ast(shape[ax])) for ax, e in enumerate(lead)]
                    if 0 < n_trailing <= len(self.iters):
                        offset = len(self.iters) - n_trailing
                        new_elts = list(res_lead) + [
                            ast.Name(id=self.iters[offset + j], ctx=ast.Load()) for j in range(n_trailing)
                        ]
                        new_slot = ast.Tuple(elts=new_elts, ctx=ast.Load())
                        return ast.Subscript(value=node.value, slice=new_slot, ctx=ast.Load())
                    if n_trailing == 0:
                        # The subscript already FULLY indexes the array with
                        # concrete scalar indices (``w_dist[-1]``, ``w[r - 1]``):
                        # it is a scalar element read. Returning it (with negatives
                        # resolved) stops the default generic_visit from
                        # subscriptifying the base Name and emitting
                        # ``w_dist[__w2][-1]`` (the stencils' last-weight read).
                        new_slot = res_lead[0] if len(res_lead) == 1 else ast.Tuple(elts=res_lead, ctx=ast.Load())
                        return ast.Subscript(value=node.value, slice=new_slot, ctx=ast.Load())
        self.generic_visit(node)
        return node
