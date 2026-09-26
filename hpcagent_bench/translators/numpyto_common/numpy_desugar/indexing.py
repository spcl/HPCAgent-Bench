"""Grids, fancy gathers, masked and ``np.ix_`` stores, slice-object and roll rewrites."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.subscripts import is_ellipsis, is_newaxis
from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import AUG_OP_SRC, RewritePass, np_attr
from hpcagent_bench.translators.numpyto_common.numpy_desugar.hoist import HoistForm, ValueHoist
from hpcagent_bench.translators.numpyto_common.numpy_desugar.kinds import dtype_kind
from hpcagent_bench.translators.numpyto_common.numpy_desugar.ranks import newaxis_singletons, expr_rank

__all__ = [
    "FANCY_GATHER_HOIST",
    "DecomposeRollSlice",
    "FancySliceStoreToLoop",
    "IxWriteToLoop",
    "MaskedAssignToLoop",
    "MgridInline",
    "ScalarizeMask",
    "SubstituteName",
    "fancy_gather_lines",
    "hoist_fancy_gather",
    "ix_unpack_scatters",
    "ix_vectors",
    "mgrid_inline_stmts",
]


def mgrid_inline_stmts(tnames: list[str], slices: list[ast.AST]) -> list[ast.stmt] | None:
    """``i, j = np.mgrid[a0:b0, a1:b1]`` -> per-axis ``arange`` reshaped onto its axis and
    broadcast-added to full-shape int zeros (numba and pythran lack ``np.mgrid``).
    ``None`` when a slice has a step or an open upper bound."""
    k = len(slices)
    if len(tnames) != k:
        return None
    los, his = [], []
    for sl in slices:
        if not isinstance(sl, ast.Slice) or sl.step is not None or sl.upper is None:
            return None
        los.append("0" if sl.lower is None else f"({ast.unparse(sl.lower)})")
        his.append(f"({ast.unparse(sl.upper)})")
    exts = [f"({his[m]} - {los[m]})" for m in range(k)]
    full = ", ".join(exts)
    lines = []
    for m in range(k):
        rshape = ", ".join(exts[mm] if mm == m else "1" for mm in range(k))
        # int64 is numpy's mgrid dtype for integer bounds; the zeros must not change it.
        lines.append(f"{tnames[m]} = np.arange({los[m]}, {his[m]}).reshape({rshape}) + np.zeros(({full},), np.int64)")
    return ast.parse("\n".join(lines)).body


class MgridInline(RewritePass):
    """Replace ``i, j = np.mgrid[s0, s1]`` with explicit ``arange`` broadcasts."""

    __slots__ = ("changed",)

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        val = node.value
        if not (
            isinstance(val, ast.Subscript)
            and isinstance(val.value, ast.Attribute)
            and val.value.attr == "mgrid"
            and isinstance(val.value.value, ast.Name)
            and val.value.value.id in ("np", "numpy")
        ):
            return node
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Tuple):
            return node
        elts = node.targets[0].elts
        if not all(isinstance(e, ast.Name) for e in elts):
            return node
        slices = val.slice.elts if isinstance(val.slice, ast.Tuple) else [val.slice]
        stmts = mgrid_inline_stmts([e.id for e in elts], slices)
        if stmts is None:
            return node
        self.changed = True
        return stmts


def fancy_gather_lines(
    arr: str, elts: list[ast.expr], elt_ranks: list[int | None], driver_rank: int, p: str
) -> list[str]:
    """Source lines gathering ``arr[elts]`` point-wise into ``<p>_o``, one loop per driver axis."""
    iters = [f"{p}_i{k}" for k in range(driver_rank)]
    it = ", ".join(iters)
    # An axis an entry pins to extent 1 is read at 0 in that entry, not at the iterator.
    idx_j = [j for j, r in enumerate(elt_ranks) if (r or 0) >= 1]
    singles = {j: newaxis_singletons(elts[j], driver_rank) for j in idx_j}
    pre: list[str] = []
    idx_exprs: list[str] = []
    for j, e in enumerate(elts):
        if j not in singles:
            idx_exprs.append(ast.unparse(e))
            continue
        t = f"{p}_x{j}"
        pre.append(f"{t} = {ast.unparse(e)}")
        idx_exprs.append(f"{t}[{', '.join('0' if k in singles[j] else iters[k] for k in range(driver_rank))}]")
    # Shape by broadcasting the entries, not by naming extents: a symbolic-shape backend
    # cannot prove a re-spelled extent equal to the one the statement already carries.
    driver = f"{p}_x{idx_j[0]}"
    if any(singles.values()):
        driver = f"{p}_b"
        pre.append(f"{driver} = " + " + ".join(f"{p}_x{j} * 0" for j in idx_j))
    extents = [f"{driver}.shape[{k}]" for k in range(driver_rank)]
    temp = f"{p}_o"
    lines = pre + [f"{temp} = np.empty({driver}.shape, {arr}.dtype)"]
    deepen = ""
    for k in range(driver_rank):
        lines.append(f"{deepen}for {iters[k]} in range({extents[k]}):")
        deepen += "    "
    lines.append(f"{deepen}{temp}[{it}] = {arr}[{', '.join(idx_exprs)}]")
    return lines


def hoist_fancy_gather(node: ast.AST, hoist: ValueHoist) -> ast.expr | None:
    """A point-wise fancy gather ``A[e0, e1, ...]`` (one entry per axis, >=1 index array) -> the
    temp its gather loop fills. numba supports a single advanced index but not this multi-index form.

    Array entries are hoisted to temps and must share one rank; scalar entries ride each
    iteration. Entries broadcast against each other: an axis a ``None`` pins to extent 1 in one
    entry is read at 0 there (see :func:`newaxis_singletons`)."""
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and isinstance(node.slice, ast.Tuple)
    ):
        return None
    ranks = hoist.tables.ranks
    arr = node.value.id
    arank = ranks.get(arr)
    elts = node.slice.elts
    if not arank or len(elts) != arank:
        return None
    if any(isinstance(e, ast.Slice) or is_newaxis(e) or is_ellipsis(e) for e in elts):
        # Point-wise only: a ``:`` axis survives into the result, which the temp cannot hold.
        return None
    elt_ranks = [expr_rank(e, ranks) for e in elts]
    arrs = [r for r in elt_ranks if r and r >= 1]
    if not arrs or any(r != arrs[0] for r in arrs):
        return None  # need >=1 index array; all arrays share the driver rank
    p = f"__gather{hoist.ctr}"
    hoist.ctr += 1
    hoist.queue(fancy_gather_lines(arr, elts, elt_ranks, arrs[0], p))
    return ast.Name(id=f"{p}_o", ctx=ast.Load())


FANCY_GATHER_HOIST = HoistForm(frozenset(), (ast.Tuple,), hoist_fancy_gather)


class ScalarizeMask(ast.NodeTransformer):
    """Index every full-rank array reference by the loop iterators: ``X[<mask>]`` -> ``X[i, j]``
    and a bare full-rank ``Z`` -> ``Z[i, j]``. Lower-rank operands and scalars broadcast as is."""

    def __init__(self, maskdump: str, idx_slice: ast.AST, arank: int, ranks: dict[str, int]) -> None:
        self.maskdump = maskdump
        self.idx_slice = idx_slice
        self.arank = arank
        self.ranks = ranks

    def sub_(self, value_node: ast.AST) -> ast.Subscript:
        return ast.Subscript(value=value_node, slice=copy.deepcopy(self.idx_slice), ctx=ast.Load())

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.ctx, ast.Load) and ast.dump(node.slice) == self.maskdump:
            return self.sub_(node.value)  # X[mask] -> X[idx]; do not recurse into it
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and self.ranks.get(node.id) == self.arank:
            return self.sub_(node)
        return node


class MaskedAssignToLoop(ast.NodeTransformer):
    """``T[mask] = rhs`` -> ``for i, j: if mask[i, j]: T[i, j] = rhs[i, j]``; numba rejects
    multi-dimensional boolean-mask indexing.

    A loop, not ``np.where``: the masked form evaluates ``rhs`` only on selected elements, so
    guarded overflow or division by zero stays out of the result.

    The mask is >=2-D and of the target's rank: a non-numeric-kind Name, or an inline
    Compare / ``& | ^ ~`` expression. An integer index Name is a fancy index and stays verbatim."""

    def __init__(self, ranks: dict[str, int], dtypes: dict[str, str]) -> None:
        self.ranks = ranks
        self.dtypes = dtypes
        self.changed = False
        self._ctr = 0

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        tgt = node.targets[0]
        if not (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name) and isinstance(tgt.ctx, ast.Store)):
            return node
        idx = tgt.slice
        arank = self.ranks.get(tgt.value.id)
        if not arank or arank < 2 or isinstance(idx, (ast.Tuple, ast.Slice)):
            return node
        struct_mask = (
            isinstance(idx, (ast.Compare, ast.BoolOp))
            or (isinstance(idx, ast.BinOp) and isinstance(idx.op, (ast.BitAnd, ast.BitOr, ast.BitXor)))
            or (isinstance(idx, ast.UnaryOp) and isinstance(idx.op, ast.Invert))
        )
        if isinstance(idx, ast.Name):
            # A numeric-kind index array is not a mask (an integer one is a fancy index).
            if self.ranks.get(idx.id) != arank or dtype_kind(idx, self.dtypes) in ("int", "float", "complex"):
                return node
        elif struct_mask:
            if expr_rank(idx, self.ranks) != arank:
                return node
        else:
            return node
        T = tgt.value.id
        p = f"__mi{self._ctr}"
        self._ctr += 1
        idx_vars = [f"{p}_{k}" for k in range(arank)]
        idx_slice = ast.parse(f"_x[{', '.join(idx_vars)}]", mode="eval").body.slice
        scal = ScalarizeMask(ast.dump(idx), idx_slice, arank, self.ranks)
        mask_s = ast.unparse(scal.visit(copy.deepcopy(idx)))
        rhs_s = ast.unparse(ScalarizeMask(ast.dump(idx), idx_slice, arank, self.ranks).visit(copy.deepcopy(node.value)))
        lines, deepen = [], ""
        for k in range(arank):
            lines.append(f"{deepen}for {idx_vars[k]} in range({T}.shape[{k}]):")
            deepen += "    "
        lines.append(f"{deepen}if {mask_s}:")
        lines.append(f"{deepen}    {T}[{', '.join(idx_vars)}] = {rhs_s}")
        self.changed = True
        return [ast.copy_location(s, node) for s in ast.parse("\n".join(lines)).body]


class DecomposeRollSlice(ast.NodeTransformer):
    """``T = np.roll(O, shift, axis)`` with a subscripted ``O`` or ``T`` -> bare-name temps, since
    the native ``expand_roll`` needs bare Names. A sliced self-roll then reads a snapshot, so the
    in-place write is safe. Native-only: numpy and the Python backends roll a slice verbatim."""

    def __init__(self) -> None:
        self.changed = False
        self._n = 0

    def fresh_(self) -> str:
        self._n += 1
        return f"__roll_{self._n}"

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        v = node.value
        # ``expand_roll`` reads a positional shift from args[1]; a keyword ``shift=`` roll is
        # not lowerable, so leave it unchanged to fail loudly.
        if not (
            isinstance(v, ast.Call)
            and isinstance(v.func, ast.Attribute)
            and v.func.attr == "roll"
            and isinstance(v.func.value, ast.Name)
            and v.func.value.id in ("np", "numpy")
            and len(v.args) >= 2
            and len(node.targets) == 1
        ):
            return node
        target = node.targets[0]
        op_bare = isinstance(v.args[0], ast.Name)
        tgt_bare = isinstance(target, ast.Name)
        if op_bare and tgt_bare:
            return node  # expand_roll handles the bare-Name form directly
        out: list[ast.stmt] = []
        if not op_bare:  # snapshot a sliced operand into a bare-name temp
            src = self.fresh_()
            out.append(ast.Assign(targets=[ast.Name(id=src, ctx=ast.Store())], value=v.args[0]))
            v.args[0] = ast.Name(id=src, ctx=ast.Load())
        if tgt_bare:
            out.append(node)  # target bare -> roll writes it directly
        else:  # roll into a bare temp, then copy back to the sliced target
            dst = self.fresh_()
            out.append(ast.Assign(targets=[ast.Name(id=dst, ctx=ast.Store())], value=v))
            out.append(ast.Assign(targets=[target], value=ast.Name(id=dst, ctx=ast.Load())))
        for s in out:
            ast.copy_location(s, node)
        self.changed = True
        return out


def ix_vectors(node: ast.AST) -> list[ast.expr] | None:
    """``np.ix_(i, j, k)`` call -> its index vectors, else None."""
    if np_attr(node) == "ix_" and node.args and not node.keywords:
        return list(node.args)
    return None


def ix_unpack_scatters(fn: ast.AST) -> dict[int, list[ast.expr]]:
    """``id`` of each store target ``A[g0, g1, ..]`` whose indices are, in order, the names one
    ``g0, g1, .. = np.ix_(v0, v1, ..)`` bound earlier in the same block -> that call's vectors.

    The unpacked grids select the same open mesh as ``A[np.ix_(v0, v1, ..)]``, and dace refuses a
    store through rank-3 index arrays. The vectors stay that selection only until a grid or a name
    a vector reads is rebound, so the first statement storing one ends the search.
    """
    found: dict[int, list[ast.expr]] = {}
    for parent in ast.walk(fn):
        for field in ("body", "orelse", "finalbody"):
            block = vars(parent).get(field)
            if not isinstance(block, list):
                continue
            for index, stmt in enumerate(block):
                if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
                    continue
                unpack = stmt.targets[0]
                vecs = ix_vectors(stmt.value)
                if vecs is None or not isinstance(unpack, ast.Tuple) or len(unpack.elts) != len(vecs):
                    continue
                grids = [e.id for e in unpack.elts if isinstance(e, ast.Name)]
                if len(grids) != len(vecs):
                    continue
                watched = set(grids) | {n.id for v in vecs for n in ast.walk(v) if isinstance(n, ast.Name)}
                for later in block[index + 1 :]:
                    if any(
                        isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id in watched
                        for n in ast.walk(later)
                    ):
                        break
                    for node in ast.walk(later):
                        stores = node.targets if isinstance(node, ast.Assign) else []
                        stores = [node.target] if isinstance(node, ast.AugAssign) else stores
                        for target in stores:
                            if (
                                isinstance(target, ast.Subscript)
                                and isinstance(target.slice, ast.Tuple)
                                and [e.id if isinstance(e, ast.Name) else None for e in target.slice.elts] == grids
                            ):
                                found[id(target)] = vecs
    return found


class FancySliceStoreToLoop(ast.NodeTransformer):
    """``A[idx, :, :] (op)= rhs`` (one index array, the other axes sliced) -> a loop over ``idx``.

    pythran silently compiles this store to the wrong elements; its read form is correct, so only
    the store is lowered.

    A lone advanced index keeps its axis position (``q[:, ja, :nk]`` is ``[dim0, len(ja), nk]``),
    so the hoisted right-hand side is indexed at the carrier's axis, with a full slice for every
    axis before it.
    """

    def __init__(self, ranks: dict[str, int], dtypes: dict[str, str]) -> None:
        self.ranks = ranks
        self.dtypes = dtypes
        self.changed = False
        self._ctr = 0

    def carrier(self, lead: list[ast.expr]) -> int | None:
        """Index of the one lead position holding a rank-1 index array, if the shape fits."""
        if not any(isinstance(e, ast.Slice) for e in lead):
            return None
        found = None
        for k, e in enumerate(lead):
            if isinstance(e, ast.Slice):
                continue
            names = [n.id for n in ast.walk(e) if isinstance(n, ast.Name)]
            arrs = [n for n in names if self.ranks.get(n) == 1 and dtype_kind(ast.Name(id=n), self.dtypes) != "bool"]
            if not arrs:
                continue
            if found is not None or len(set(arrs)) != 1:
                return None
            found = k
        return found

    def lower_(self, node: ast.stmt, target: ast.expr, value: ast.expr, op: str) -> ast.AST:
        if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)):
            return node
        lead = list(target.slice.elts) if isinstance(target.slice, ast.Tuple) else [target.slice]
        k = self.carrier(lead)
        if k is None:
            return node
        p = f"__fs{self._ctr}"
        self._ctr += 1
        it = f"{p}_i"
        idx_name = next(n.id for n in ast.walk(lead[k]) if isinstance(n, ast.Name) and self.ranks.get(n.id) == 1)
        at_iter = SubstituteName(idx_name, f"{idx_name}[{it}]").visit(copy.deepcopy(lead[k]))
        new_lead = [ast.unparse(e) if j != k else ast.unparse(at_iter) for j, e in enumerate(lead)]
        lines = [
            f"{p}_v = {ast.unparse(value)}",
            f"for {it} in range({idx_name}.shape[0]):",
            f"    {target.value.id}[{', '.join(new_lead)}] {op} {p}_v[{', '.join([':'] * k + [it])}]",
        ]
        self.changed = True
        return [ast.copy_location(st, node) for st in ast.parse("\n".join(lines)).body]

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        return self.lower_(node, node.targets[0], node.value, "=")

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        self.generic_visit(node)
        op = AUG_OP_SRC.get(type(node.op))
        return node if op is None else self.lower_(node, node.target, node.value, op)


class SubstituteName(ast.NodeTransformer):
    """Replace bare ``name`` with the parsed ``text``."""

    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self.repl = ast.parse(text, mode="eval").body

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return copy.deepcopy(self.repl) if node.id == self.name else node


class IxWriteToLoop(ast.NodeTransformer):
    """``A[np.ix_(i, j, k)] (op)= rhs`` -> one loop per index vector, each read by its own
    iterator: ``np.ix_`` selects the outer product ``A[i[p], j[q], k[r]]``. DaCe lowers ``np.ix_``
    to an opaque callback; numba and pythran lack it.

    Only the write form with one vector per axis is lowered. Left verbatim: a partial ``np.ix_``,
    a boolean vector (its extent is not its length) and a read-position ``np.ix_``. Soundness
    assumes no duplicates within a vector: a repeated value accumulates here where numpy's
    ``+=`` applies the update once."""

    def __init__(self, ranks: dict[str, int], dtypes: dict[str, str], fn: ast.AST) -> None:
        self.ranks = ranks
        self.dtypes = dtypes
        self.changed = False
        self._ctr = 0
        self.fn = fn
        self.unpacked: dict[int, list[ast.expr]] | None = None

    def lower_(self, node: ast.stmt, target: ast.expr, op: str) -> ast.AST:
        if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)):
            return node
        # Built lazily, after every earlier pass has rewritten the whole body.
        if self.unpacked is None:
            self.unpacked = ix_unpack_scatters(self.fn)
        vecs = ix_vectors(target.slice) or self.unpacked.get(id(target))
        if vecs is None or self.ranks.get(target.value.id) != len(vecs):
            return node
        if any(dtype_kind(v, self.dtypes) == "bool" for v in vecs):
            return node
        p = f"__ix{self._ctr}"
        self._ctr += 1
        lines: list[str] = []

        def hoist(e: ast.expr, tmp: str) -> str:
            """Bind ``e`` to ``tmp`` before the nest (numpy evaluates it once, before the store)."""
            if isinstance(e, ast.Name):
                return e.id
            lines.append(f"{tmp} = {ast.unparse(e)}")
            return tmp

        xs = [hoist(v, f"{p}_x{k}") for k, v in enumerate(vecs)]
        val = hoist(node.value, f"{p}_v")
        iters = [f"{p}_i{k}" for k in range(len(vecs))]
        indent = ""
        for k in range(len(vecs)):
            lines.append(f"{indent}for {iters[k]} in range({xs[k]}.shape[0]):")
            indent += "    "
        # Rank-0 rhs is broadcast; any other rhs is indexed per grid point without trusting its
        # rank further (expr_rank over-counts np.einsum), so a broadcasting rhs fails loudly.
        rhs = val if expr_rank(node.value, self.ranks) == 0 else f"{val}[{', '.join(iters)}]"
        idx = ", ".join(f"{xs[k]}[{iters[k]}]" for k in range(len(vecs)))
        lines.append(f"{indent}{target.value.id}[{idx}] {op} {rhs}")
        self.changed = True
        return [ast.copy_location(s, node) for s in ast.parse("\n".join(lines)).body]

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        return self.lower_(node, node.targets[0], "=")

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        self.generic_visit(node)
        op = AUG_OP_SRC.get(type(node.op))
        return node if op is None else self.lower_(node, node.target, op)
