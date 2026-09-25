"""Batched, integer and reshape-wrapped matmul lowered to loops."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import DesugarError, np_attr
from hpcagent_bench.translators.numpyto_common.numpy_desugar.hoist import HoistForm, ValueHoist
from hpcagent_bench.translators.numpyto_common.numpy_desugar.kinds import dtype_kind
from hpcagent_bench.translators.numpyto_common.numpy_desugar.ranks import expr_rank


def matmul_pairs(node: ast.AST) -> list[ast.AST]:
    """Every matmul (``@`` BinOp or ``np.matmul`` call) under ``node``."""
    out: list[ast.AST] = []
    for n in ast.walk(node):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult):
            out.append(n)
        elif np_attr(n) == "matmul" and len(vars(n).get("args") or []) == 2:
            out.append(n)
    return out


def matmul_operands(mm: ast.AST):
    if isinstance(mm, ast.BinOp):
        return mm.left, mm.right
    return mm.args[0], mm.args[1]


class IndexLeadingAxis(ast.NodeTransformer):
    """Subscript every rank > 2 ``Name`` by ``[bv]`` (its leading/batch axis),
    dropping it to a 2-D operand. Names of rank <= 2 are left untouched (a
    shared 2-D right operand broadcasts across the batch)."""

    def __init__(self, bv: str, ranks: dict[str, int]) -> None:
        self.bv = bv
        self.ranks = ranks

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and (self.ranks.get(node.id, 0) or 0) > 2:
            return ast.copy_location(
                ast.Subscript(
                    value=ast.Name(id=node.id, ctx=ast.Load()),
                    slice=ast.Name(id=self.bv, ctx=ast.Load()),
                    ctx=ast.Load(),
                ),
                node,
            )
        return node


class BatchedMatmulToLoop(ast.NodeTransformer):
    """``Q[:] = Q + I @ star`` (I rank-3) -> a loop over the batch axis doing a
    2-D GEMM per element. numba / pythran support 2-D ``@`` but not the stacked
    (>=3-D) form -- and Fortran has no matmul-broadcast either, so this is the
    universal "batched GEMM = for-loop over GEMMs" lowering."""

    def __init__(self, ranks: dict[str, int]) -> None:
        self.ranks = ranks
        self._ctr = 0
        self.changed = False

    def batch_source(self, value: ast.AST) -> ast.Name | None:
        """First rank > 2 Name feeding a batched matmul -- its leading axis is
        the batch extent."""
        for mm in matmul_pairs(value):
            for op in matmul_operands(mm):
                for n in ast.walk(op):
                    if isinstance(n, ast.Name) and (self.ranks.get(n.id, 0) or 0) > 2:
                        return n
        return None

    def is_batched(self, value: ast.AST) -> bool:
        """True iff the statement carries a CLEANLY batched matmul: at least one
        operand has rank > 2 AND every operand is a bare ``Name``. The bare-Name
        guard is load-bearing -- a ``reshape`` / ``transpose`` wrapping the
        operand restructures axes, so indexing its leading axis (doitgen's
        ``np.reshape(A, (NR, NQ, 1, NP)) @ C4``) would be a miscompile, not a
        batched GEMM. Those stay verbatim (and skip on numba/pythran)."""
        for mm in matmul_pairs(value):
            a, b = matmul_operands(mm)
            if not (isinstance(a, ast.Name) and isinstance(b, ast.Name)):
                return False
            ra, rb = expr_rank(a, self.ranks), expr_rank(b, self.ranks)
            if (ra or 0) > 2 or (rb or 0) > 2:
                return True
        return False

    def index_target(self, target: ast.AST, bv: str) -> ast.AST | None:
        """``T[:]`` or bare rank > 2 ``T`` -> ``T[bv]``."""
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and isinstance(target.slice, ast.Slice)
            and target.slice.lower is None
            and target.slice.upper is None
        ):
            return ast.Subscript(
                value=ast.Name(id=target.value.id, ctx=ast.Load()),
                slice=ast.Name(id=bv, ctx=ast.Load()),
                ctx=ast.Store(),
            )
        if isinstance(target, ast.Name) and (self.ranks.get(target.id, 0) or 0) > 2:
            return ast.Subscript(
                value=ast.Name(id=target.id, ctx=ast.Load()), slice=ast.Name(id=bv, ctx=ast.Load()), ctx=ast.Store()
            )
        return None

    def allocation(self, name: str, value: ast.AST) -> ast.stmt | None:
        """The ``np.empty`` that BINDS a bare-Name target, or None when its extents are not exact:
        the loop form only WRITES the target, so nothing would bind it (dace: "ctx used before
        definition"). Exact means equal-rank operands -- ``A``'s extents with ``B``'s last."""
        pairs = matmul_pairs(value)
        if len(pairs) != 1 or pairs[0] is not value:
            return None  # the matmul is nested in a larger expression: its result extents are not this one's
        a, b = matmul_operands(value)
        rank = expr_rank(a, self.ranks)
        if not (isinstance(a, ast.Name) and isinstance(b, ast.Name)) or rank != expr_rank(b, self.ranks):
            return None
        if rank is None or rank < 3:
            return None
        dims = [f"{a.id}.shape[{k}]" for k in range(rank - 1)] + [f"{b.id}.shape[{rank - 1}]"]
        return ast.parse(f"{name} = np.empty(({', '.join(dims)}), {a.id}.dtype)").body[0]

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if len(node.targets) != 1 or not self.is_batched(node.value):
            return node
        bsrc = self.batch_source(node.value)
        if bsrc is None:
            return node
        new_target = self.index_target(node.targets[0], "")  # probe form first
        if new_target is None:
            return node  # target not a recognised batched whole-array write
        # A bare-Name target is a BINDING, not a write into an existing array: it needs its own
        # allocation, and without an exact shape for it the statement stays verbatim.
        alloc = None
        if isinstance(node.targets[0], ast.Name):
            alloc = self.allocation(node.targets[0].id, node.value)
            if alloc is None:
                return node
        bv = f"__bm{self._ctr}"
        self._ctr += 1
        self.changed = True
        new_target = self.index_target(node.targets[0], bv)
        new_value = IndexLeadingAxis(bv, self.ranks).visit(copy.deepcopy(node.value))
        extent = ast.Subscript(
            value=ast.Attribute(value=ast.Name(id=bsrc.id, ctx=ast.Load()), attr="shape", ctx=ast.Load()),
            slice=ast.Constant(value=0),
            ctx=ast.Load(),
        )
        loop = ast.For(
            target=ast.Name(id=bv, ctx=ast.Store()),
            iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[extent], keywords=[]),
            body=[ast.Assign(targets=[new_target], value=new_value)],
            orelse=[],
        )
        ast.copy_location(loop, node)
        return [ast.copy_location(alloc, node), loop] if alloc is not None else loop


def int_matmul_acc_dtype(aid: str, bid: str, ka: str, kb: str) -> str:
    """Accumulator dtype for an integer ``a @ b``, as source the emitted program evaluates.

    numpy's ``@`` returns ``result_type(a, b)``, so the accumulator must too. A hardcoded
    ``np.int64`` stored every int32 port's result through a NARROWING copy, which DaCe
    cannot emit at all (comet_int4_gemm died on ``dace::CopyNDDynamic<int, 1, 0, 2>::
    Dynamic::Copy(int64_t*, int*, ...)``). ``np.result_type``/``np.promote_types`` do not
    survive the DaCe frontend, and this pass knows operand KINDS but never widths, so the
    promotion is spelled as an operand's ``.dtype`` -- the form the lowerings here already
    emit. A bool operand never decides it: ``bool @ int32`` is int32 in numpy."""
    return f"{bid}.dtype" if (ka == "bool" and kb != "bool") else f"{aid}.dtype"


def int_matmul_stmts(temp: str, a: str, b: str, ra: int, rb: int, ctr: int, acc: str) -> list[str]:
    """Source lines for an INTEGER matmul (``a @ b``) as an explicit loop, accumulating in
    ``acc`` (:func:`int_matmul_acc_dtype`). numba's BLAS-backed ``@`` is float-only, so the
    loop is the lowering. Raises for ranks numba could not express even after batching."""
    p = f"__mm{ctr}"
    if ra == 1 and rb == 1:  # dot -> scalar
        return [f"{temp} = 0", f"for {p}_k in range({a}.shape[0]):", f"    {temp} += {a}[{p}_k] * {b}[{p}_k]"]
    if ra == 1 and rb == 2:  # (K,) @ (K, N) -> (N,)
        return [
            f"{temp} = np.zeros({b}.shape[1], {acc})",
            f"for {p}_j in range({b}.shape[1]):",
            f"    for {p}_k in range({a}.shape[0]):",
            f"        {temp}[{p}_j] += {a}[{p}_k] * {b}[{p}_k, {p}_j]",
        ]
    if ra == 2 and rb == 1:  # (M, K) @ (K,) -> (M,)
        return [
            f"{temp} = np.zeros({a}.shape[0], {acc})",
            f"for {p}_i in range({a}.shape[0]):",
            f"    for {p}_k in range({a}.shape[1]):",
            f"        {temp}[{p}_i] += {a}[{p}_i, {p}_k] * {b}[{p}_k]",
        ]
    if ra == 2 and rb == 2:  # (M, K) @ (K, N) -> (M, N)
        return [
            f"{temp} = np.zeros(({a}.shape[0], {b}.shape[1]), {acc})",
            f"for {p}_i in range({a}.shape[0]):",
            f"    for {p}_j in range({b}.shape[1]):",
            f"        for {p}_k in range({a}.shape[1]):",
            f"            {temp}[{p}_i, {p}_j] += {a}[{p}_i, {p}_k] * {b}[{p}_k, {p}_j]",
        ]
    raise DesugarError(
        f"integer matmul of ranks {ra}x{rb} is unsupported (numba has no integer @; only <=2-D operands are lowered)"
    )


def int_matmul_temp(a: ast.expr, b: ast.expr, hoist: ValueHoist) -> ast.expr | None:
    """The temp an INTEGER ``a @ b`` loop fills, or None when an operand is not (definitely) integer/bool."""
    dtypes, ranks = hoist.tables.dtypes, hoist.tables.ranks
    ka, kb = dtype_kind(a, dtypes), dtype_kind(b, dtypes)
    if ka not in ("int", "bool") or kb not in ("int", "bool"):
        return None  # not (definitely) an integer matmul -> leave for numba's float @
    ra, rb = expr_rank(a, ranks), expr_rank(b, ranks)
    if ra is None or rb is None:
        return None  # can't determine the shape -> leave verbatim (a clean skip),
        # NOT a raise: an unknown rank is an inference gap, not a known-unsupported shape.
    p = f"__mmi{hoist.ctr}"
    pre = []
    aid = a.id if isinstance(a, ast.Name) else f"{p}_a"
    bid = b.id if isinstance(b, ast.Name) else f"{p}_b"
    if not isinstance(a, ast.Name):
        pre.append(f"{aid} = {ast.unparse(a)}")
    if not isinstance(b, ast.Name):
        pre.append(f"{bid} = {ast.unparse(b)}")
    temp = f"{p}_o"
    acc = int_matmul_acc_dtype(aid, bid, ka, kb)
    hoist.queue(pre + int_matmul_stmts(temp, aid, bid, ra, rb, hoist.ctr, acc))
    hoist.ctr += 1
    return ast.Name(id=temp, ctx=ast.Load())


def hoist_int_matmul(node: ast.AST, hoist: ValueHoist) -> ast.expr | None:
    """An INTEGER ``a @ b`` / ``np.matmul`` / ``np.dot`` (both operands integer/bool kind) -> the temp its explicit
    loop fills (bfs's ``reach = frontier @ graph``). numba's ``@`` is BLAS-backed and float-only, so int matmul fails
    to type; float matmul is LEFT for numba's fast path. Owned-but-unhandled shapes (>2-D) raise DesugarError."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
        return int_matmul_temp(node.left, node.right, hoist)
    if isinstance(node, ast.Call) and np_attr(node) in ("matmul", "dot") and len(node.args) == 2:
        return int_matmul_temp(node.args[0], node.args[1], hoist)
    return None


INT_MATMUL_HOIST = HoistForm(frozenset({"matmul", "dot"}), (ast.MatMult,), hoist_int_matmul)


def is_transpose_expr(v: ast.AST) -> bool:
    """``np.transpose(x, ...)`` / ``x.transpose(...)`` / ``x.T`` -- these produce a
    non-contiguous view."""
    return (
        np_attr(v) == "transpose"
        or (isinstance(v, ast.Attribute) and v.attr == "T")
        or (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "transpose")
    )


def noncontig_names(tree: ast.AST) -> set:
    """Names bound to a non-contiguous view (a transpose, or a transpose chained
    through another such name) -- to a fixpoint."""
    nc: set = set()
    for unused in range(6):
        grew = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                v = node.value
                bad = is_transpose_expr(v) or (isinstance(v, ast.Name) and v.id in nc)
                if bad and node.targets[0].id not in nc:
                    nc.add(node.targets[0].id)
                    grew = True
        if not grew:
            break
    return nc


class ReshapeContiguousInline(ast.NodeTransformer):
    """Wrap a reshape's array operand in ``np.ascontiguousarray`` when it is
    non-contiguous (a transpose or a transpose-derived name) -- numba's reshape
    requires a contiguous array (stockham's ``np.reshape(tmp_perm, (N,))`` where
    ``tmp_perm = np.transpose(yv, ...)``). A no-op for already-contiguous inputs."""

    def __init__(self, noncontig: set) -> None:
        self.noncontig = noncontig
        self.changed = False

    def noncontig_(self, x: ast.AST) -> bool:
        return (isinstance(x, ast.Name) and x.id in self.noncontig) or is_transpose_expr(x)

    def wrap(self, x: ast.AST) -> ast.Call:
        self.changed = True
        acont = ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="ascontiguousarray", ctx=ast.Load())
        return ast.Call(func=acont, args=[x], keywords=[])

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if np_attr(node) == "reshape" and node.args and self.noncontig_(node.args[0]):
            node.args[0] = self.wrap(node.args[0])
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "reshape" and self.noncontig_(node.func.value):
            node.func.value = self.wrap(node.func.value)
        return node


def as_matmul(node: ast.AST):
    """``a @ b`` / ``np.matmul(a, b)`` / ``np.dot(a, b)`` -> ``(a, b)`` else None."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
        return node.left, node.right
    if np_attr(node) in ("matmul", "dot") and len(vars(node).get("args") or []) == 2:
        return node.args[0], node.args[1]
    return None


def as_reshape(node: ast.AST):
    """``np.reshape(x, shape)`` / ``x.reshape(shape)`` -> ``(x, shape_node)`` else None."""
    if np_attr(node) == "reshape" and len(node.args) >= 2:
        return node.args[0], node.args[1]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "reshape"
        and node.args
    ):
        shape = node.args[0] if len(node.args) == 1 else ast.Tuple(elts=list(node.args), ctx=ast.Load())
        return node.func.value, shape
    return None


class ReshapeMatmulInline(ast.NodeTransformer):
    """``T[:] = np.reshape(np.reshape(X, (*batch, 1, K)) @ Y, (*batch, N))`` -> a
    contraction loop ``r[*b, n] = sum_k X[*b, k] * Y[k, n]`` into a fresh full
    temp (so the ``A = f(A)`` WAR in doitgen is safe). numba cannot type the
    reshape-wrapped batched ``@`` (the existing batched-matmul pass deliberately
    refuses reshape-wrapped operands as a miscompile risk). Fires ONLY on the
    unit-dim-insertion form (``mid[-2] == 1``, ``len(mid) == X.ndim + 1``, Y 2-D);
    a genuinely different reshape is left verbatim. A matched-but-inconsistent
    shape (Y not 2-D) raises DesugarError rather than miscompiling."""

    def __init__(self, ranks: dict[str, int]) -> None:
        self.ranks = ranks
        self.changed = False
        self._ctr = 0

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        tgt = node.targets[0]
        if not (
            (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name) and isinstance(tgt.slice, ast.Slice))
            or isinstance(tgt, ast.Name)
        ):
            return node
        outer = as_reshape(node.value)
        if not outer:
            return node
        mm = as_matmul(outer[0])
        if not mm:
            return node
        inner = as_reshape(mm[0])
        if not inner or not isinstance(inner[0], ast.Name) or not isinstance(mm[1], ast.Name):
            return node
        X, Y, mid = inner[0], mm[1], inner[1]
        rX = expr_rank(X, self.ranks)
        # Fire only on the unit-dim-insertion batched form; other reshapes -> verbatim.
        if not (
            isinstance(mid, ast.Tuple)
            and rX
            and len(mid.elts) == rX + 1
            and isinstance(mid.elts[-2], ast.Constant)
            and mid.elts[-2].value == 1
        ):
            return node
        if expr_rank(Y, self.ranks) != 2 or rX < 2:
            raise DesugarError(
                f"reshape-batched matmul: unit-dim form needs a 2-D right operand and a >=2-D "
                f"left operand (got left ndim {rX}, right ndim {expr_rank(Y, self.ranks)})"
            )
        p = f"__dg{self._ctr}"
        self._ctr += 1
        batch = list(range(rX - 1))
        bi = [f"{p}_b{i}" for i in batch]
        bidx = ", ".join(bi)
        oshape = "".join(f"{X.id}.shape[{i}], " for i in batch) + f"{Y.id}.shape[1]"
        temp = f"{p}_o"
        lines = [f"{temp} = np.zeros(({oshape},), {X.id}.dtype)"]
        deep = ""
        for i in batch:
            lines.append(f"{deep}for {bi[i]} in range({X.id}.shape[{i}]):")
            deep += "    "
        lines.append(f"{deep}for {p}_n in range({Y.id}.shape[1]):")
        lines.append(f"{deep}    for {p}_k in range({X.id}.shape[{rX - 1}]):")
        lines.append(f"{deep}        {temp}[{bidx}, {p}_n] += {X.id}[{bidx}, {p}_k] * {Y.id}[{p}_k, {p}_n]")
        node.value = ast.Name(id=temp, ctx=ast.Load())
        self.changed = True
        return [ast.copy_location(s, node) for s in ast.parse("\n".join(lines)).body] + [node]
