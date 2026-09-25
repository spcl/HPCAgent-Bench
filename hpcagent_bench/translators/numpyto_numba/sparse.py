"""Sparse ``@`` lowered onto the unpacked buffer ABI the manifest declares."""

import ast

from hpcagent_bench.translators.numpyto_common.ir import KernelIR, SparseArrayDesc

#: A CSR matrix's buffers ARE its transpose's CSC buffers, and back -- so ``A.T @ x`` needs the dual
#: format name over the SAME buffers, never a second loop nest. Same relabelling the C path applies
#: in ``lib_nodes.matmul_hoist.transpose_sparse_desc``.
TRANSPOSE_DUAL_FORMAT = {"csr": "csc", "csc": "csr"}


#: Calls that BUILD a rank >= 2 array. One in the body means a local could be a matrix, and the
#: matvec lowering below -- which assumes a dense VECTOR operand -- is then not proven.
RANK_RAISING_CALLS = frozenset(
    {"reshape", "outer", "eye", "identity", "tile", "stack", "vstack", "hstack", "meshgrid", "diag", "atleast_2d"}
)


#: Same, but only on a multi-axis shape: ``np.zeros(n)`` is a vector, ``np.zeros((n, m))`` is not.
SHAPED_CTOR_CALLS = frozenset({"zeros", "ones", "empty", "full"})


class SparseLoweringRefused(Exception):
    """One sparse op this emitter cannot lower, which abandons the WHOLE rewrite.

    A half-lowered body still takes the logical scipy matrix and dies in numba typing exactly as it
    does today; a guessed loop nest would instead validate as wrong numbers. Refusing leaves the
    kernel where it was, which is the only honest outcome of the two.
    """


def call_leaf(func: ast.AST) -> str:
    """The bare / attribute-leaf name of a call target (``np.zeros`` -> ``zeros``)."""
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def sparse_operand(node: ast.AST, sparse: dict[str, SparseArrayDesc]) -> tuple[SparseArrayDesc, bool] | None:
    """``(descriptor, is_transpose)`` when ``node`` names a sparse array or its ``.T``, else ``None``."""
    if isinstance(node, ast.Name) and node.id in sparse:
        return sparse[node.id], False
    if isinstance(node, ast.Attribute) and node.attr == "T" and isinstance(node.value, ast.Name):
        return (sparse[node.value.id], True) if node.value.id in sparse else None
    return None


def transposed_desc(desc: SparseArrayDesc) -> SparseArrayDesc:
    """``desc`` for ``A.T``: the dual format over the same buffers, logical shape reversed."""
    dual = TRANSPOSE_DUAL_FORMAT.get(desc.format)
    if dual is None:
        raise SparseLoweringRefused(f"format {desc.format!r} has no dual descriptor for a transpose")
    return SparseArrayDesc(desc.name, dual, tuple(reversed(desc.logical_shape)), dict(desc.buffers))


def symbol_readbacks(kir: KernelIR, params: set[str]) -> dict[str, str]:
    """``{shape token: python expression}`` -- how the kernel reads each shape symbol back from its OWN
    parameters. The unpacked ABI passes no ``N``, so a loop bound comes from a peer array's shape."""
    exprs = {s.name: s.name for s in kir.symbols if s.name in params}
    for arr in kir.arrays:
        if arr.name not in params:
            continue
        for axis, token in enumerate(arr.shape):
            if str(token).isidentifier():
                exprs.setdefault(str(token), f"{arr.name}.shape[{axis}]")
    return exprs


def dense_operands_are_vectors(fn: ast.FunctionDef, kir: KernelIR, buffers: set[str]) -> bool:
    """True when no local in ``fn`` can be a matrix: every dense array PARAMETER is rank 1 and the body
    builds nothing wider. That is the proof the matvec lowering needs about its dense operand."""
    for arr in kir.arrays:
        if arr.name not in buffers and len(arr.shape) != 1:
            return False
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        leaf = call_leaf(node.func)
        if leaf in RANK_RAISING_CALLS:
            return False
        if leaf in SHAPED_CTOR_CALLS and node.args and isinstance(node.args[0], (ast.Tuple, ast.List)):
            return False
    return True


def prange_row_loop(loop: ast.For) -> None:
    """Hand a ROW-PARTITIONED loop to numba's parfor.

    Iteration ``i`` writes only row ``i`` of a freshly allocated temp, so independence holds by
    construction. Done here because :func:`parallelize_one_range_loop` picks ONE loop in source
    order, and the sparse matvec -- which is where these kernels spend their time -- is not it.
    """
    call = loop.iter
    if isinstance(call, ast.Call):
        call.func = ast.Attribute(value=ast.Name(id="nb", ctx=ast.Load()), attr="prange", ctx=ast.Load())


class SparseMatmulRewriter(ast.NodeTransformer):
    """Replace every sparse ``@`` with a dense temp plus the loop nest ``sparse_emit`` lowers it to.

    The nest is spliced in front of the statement that used the matmul, inside whatever block that
    statement lives in: a matvec inside the Krylov loop must be recomputed every iteration, never
    hoisted above it.
    """

    def __init__(self, sparse: dict[str, SparseArrayDesc], symbol_exprs: dict[str, str], vectors_only: bool) -> None:
        self.sparse = sparse
        self.symbol_exprs = symbol_exprs
        self.vectors_only = vectors_only
        self.counter = 0
        self.bounds: dict[str, str] = {}
        self.pre: list[ast.stmt] = []

    def block(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        out = []
        for stmt in stmts:
            outer, self.pre = self.pre, []
            visited = self.visit(stmt)
            out.extend(self.pre)
            self.pre = outer
            out.append(visited)
        return out

    def generic_visit(self, node: ast.AST) -> ast.AST:
        for field, value in ast.iter_fields(node):
            if isinstance(value, list) and any(isinstance(v, ast.stmt) for v in value):
                setattr(node, field, self.block(value))
            elif isinstance(value, list):
                setattr(node, field, [self.visit(v) if isinstance(v, ast.AST) else v for v in value])
            elif isinstance(value, ast.AST):
                setattr(node, field, self.visit(value))
        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        # A while TEST is re-evaluated per iteration; splicing its matvec above the loop would
        # compute it once and then spin on a stale value.
        for sub in ast.walk(node.test):
            if not (isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.MatMult)):
                continue
            if sparse_operand(sub.left, self.sparse) or sparse_operand(sub.right, self.sparse):
                raise SparseLoweringRefused("sparse matmul in a while condition")
        return self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.op, ast.MatMult):
            return node
        operand = sparse_operand(node.left, self.sparse)
        if operand is None:
            if sparse_operand(node.right, self.sparse) is not None:
                raise SparseLoweringRefused("dense @ sparse has no lowering here")
            return node
        desc, transposed = operand
        if transposed:
            desc = transposed_desc(desc)
        self.counter += 1
        temp = f"__sp{self.counter}"
        self.pre.extend(self.lower(temp, desc, node.right))
        return ast.Name(id=temp, ctx=ast.Load())

    def bound(self, token: object) -> str:
        """Name of the function-top local holding shape symbol ``token``."""
        expr = self.symbol_exprs.get(str(token))
        if expr is None:
            raise SparseLoweringRefused(f"shape symbol {token!r} is not readable from any parameter")
        name = f"__sp_{token}"
        self.bounds[name] = expr
        return name

    def alloc(self, temp: str, data_buffer: str, extents: tuple[str, ...]) -> ast.stmt:
        shape = extents[0] if len(extents) == 1 else "({})".format(", ".join(extents))
        return ast.parse(f"{temp} = np.zeros({shape}, dtype={data_buffer}.dtype)").body[0]

    def lower(self, temp: str, desc: SparseArrayDesc, rhs: ast.AST) -> list[ast.stmt]:
        from hpcagent_bench.translators.numpyto_common.sparse_emit import SPARSE_MATMUL_DISPATCH

        if len(desc.logical_shape) != 2:
            raise SparseLoweringRefused(f"sparse operand {desc.name!r} is not 2-D")
        rows = self.bound(desc.logical_shape[0])
        rhs_sparse = sparse_operand(rhs, self.sparse)
        if rhs_sparse is not None:
            rhs_desc, rhs_transposed = rhs_sparse
            if rhs_transposed:
                rhs_desc = transposed_desc(rhs_desc)
            expand = SPARSE_MATMUL_DISPATCH.get((desc.format, rhs_desc.format, "matmul_dense"))
            if expand is None:
                raise SparseLoweringRefused(f"no sparse x sparse lowering for {desc.format}/{rhs_desc.format}")
            out_cols = self.bound(rhs_desc.logical_shape[1])
            stmts = expand(temp, desc.buffers, rhs_desc.buffers, rows, out_cols)
            for loop in stmts:
                prange_row_loop(loop)
            return [self.alloc(temp, desc.buffers["data"], (rows, out_cols)), *stmts]
        if not isinstance(rhs, ast.Name):
            raise SparseLoweringRefused("sparse @ <expression>: the dense operand must be a name")
        if not self.vectors_only:
            raise SparseLoweringRefused("dense operand is not proven to be a vector")
        target = ast.Name(id=temp, ctx=ast.Load())
        if desc.format == "csr":
            stmts = SPARSE_MATMUL_DISPATCH[("csr", "dense", "matmul_vec")](target, desc.buffers, rhs.id, rows)
            prange_row_loop(stmts[0])
        elif desc.format == "csc":
            # CSC scatter-adds into y[indices[k]], a data-dependent row, so this one stays serial.
            cols = self.bound(desc.logical_shape[1])
            stmts = SPARSE_MATMUL_DISPATCH[("csc", "dense", "matmul_vec")](target, desc.buffers, rhs.id, rows, cols)
        else:
            raise SparseLoweringRefused(f"no matvec lowering wired for format {desc.format!r}")
        return [self.alloc(temp, desc.buffers["data"], (rows,)), *stmts]


def rewrite_sparse_matmuls(numpy_source: str, kir: KernelIR) -> str | None:
    """Lower the kernel onto the UNPACKED sparse ABI, or return ``None`` to leave it verbatim.

    The manifest declares a sparse ``A`` as physical buffers (``A_indptr`` / ``A_indices`` /
    ``A_data``) and the harness binds those, but the numpy reference still writes the logical
    ``A @ x`` -- which numba cannot type, because at run time it is a live ``scipy.sparse`` object.
    So the signature becomes ``kir.input_args``, the expanded ABI the C and dace backends already
    compile against, and every sparse ``@`` becomes the per-format loop nest
    :mod:`numpyto_common.sparse_emit` owns.

    ``None`` whenever the kernel has no sparse operand, or has one this path declines to guess at.
    """
    if not kir.sparse:
        return None
    tree = ast.parse(numpy_source)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == kir.kernel_name), None)
    if fn is None:
        return None
    params = [a.arg for a in fn.args.args]
    if not any(p in kir.sparse for p in params):
        return None
    args = fn.args
    if args.defaults or args.kw_defaults or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg:
        return None
    expanded: list[str] = []
    for p in params:
        desc = kir.sparse.get(p)
        expanded.extend(desc.buffers.values() if desc is not None else [p])
    # The IR's own ABI order is the contract every other backend compiles to. A mismatch means this
    # kernel expands some other way, and the rewrite would bind arguments to the wrong slots.
    if expanded != list(kir.input_args):
        return None
    buffers = {name for desc in kir.sparse.values() for name in desc.buffers.values()}
    rewriter = SparseMatmulRewriter(
        kir.sparse, symbol_readbacks(kir, set(expanded)), dense_operands_are_vectors(fn, kir, buffers)
    )
    try:
        rewriter.visit(fn)
    except SparseLoweringRefused:
        return None
    if any(isinstance(n, ast.Name) and n.id in kir.sparse for n in ast.walk(fn)):
        return None  # a logical sparse name survived; it would still reach numba as a scipy object
    lead = 1 if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant) else 0
    prologue = [ast.parse(f"{name} = {expr}").body[0] for name, expr in rewriter.bounds.items()]
    fn.body = fn.body[:lead] + prologue + fn.body[lead:]
    fn.args.args = [ast.arg(arg=name) for name in expanded]
    return ast.unparse(ast.fix_missing_locations(tree))
