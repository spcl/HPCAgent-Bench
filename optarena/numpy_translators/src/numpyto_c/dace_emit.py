"""Emit a DaCe ``@dc.program`` from the canonical numpy reference.

The Foundation-track numpy kernels are already dace-shaped (they were
ported FROM ``@dace.program`` bodies), and they now carry every size
symbol as an explicit scalar argument plus a per-array shape. That is
exactly the metadata a numpy->dace converter needs:

* size SYMBOLS (``LEN_1D``, ``K``, ...) become module-level
  ``dc.symbol(...)`` declarations and are DROPPED from the program
  signature (dace passes them implicitly via the array shapes);
* ARRAY arguments are typed ``<dctype>[shape]`` -- ``dc_float`` for the
  precision-driven floats, ``dc.int32`` / ``dc.int64`` for index arrays
  (the dtype the original ``dace.int*`` annotation carried, recovered
  from bench_info ``init.dtypes``);
* kernel SCALARS (``alpha``, s174's ``M``, ...) stay typed scalar args.

The body is the numpy reference verbatim. Reuses :func:`parse_kernel`
so the array/symbol/scalar classification is the single source of
truth shared with the C/Fortran emitters.
"""

import ast
import copy
import re
from typing import Dict, List

from numpyto_c.ir import KernelIR
from numpyto_common.numpy_desugar import desugar_for_python_backend

_IDENT_RE = re.compile(r"[A-Za-z_]\w*")


class _ShapeToSymbol(ast.NodeTransformer):
    """dace has no runtime ``.shape``: an array carries a SYMBOLIC shape (the
    ``a: dc_float[M + 1]`` annotation), so ``a.shape[k]`` IS that symbolic dimension.
    Replace each ``<array>.shape[<const k>]`` with the array's k-th declared shape
    token (``A_indptr.shape[0]`` -> ``M + 1``)."""

    def __init__(self, arr_shapes: Dict[str, List[str]]):
        self.arr_shapes = arr_shapes

    def visit_Subscript(self, node: ast.Subscript):
        self.generic_visit(node)
        v = node.value
        if (isinstance(v, ast.Attribute) and v.attr == "shape" and isinstance(v.value, ast.Name)
                and v.value.id in self.arr_shapes and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, int)):
            toks = self.arr_shapes[v.value.id]
            if 0 <= node.slice.value < len(toks):
                return ast.copy_location(ast.parse(toks[node.slice.value], mode="eval").body, node)
        return node


class _DropSymbolAssign(ast.NodeTransformer):
    """Drop ``<sym> = ...`` where ``<sym>`` is a declared size symbol. A dc.symbol is a
    compile-time constant the harness supplies via the array shapes, so recomputing it
    in the body (``M = A_indptr.shape[0] - 1``) is both redundant and illegal in dace."""

    def __init__(self, symbols):
        self.symbols = set(symbols)

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in self.symbols:
            return None
        return node


class _ResolveZeros(ast.NodeTransformer):
    """A LOWERED kir marks each allocation-init intermediate with ``<name> =
    __optarena_zeros__()`` (a placeholder the C emitter turns into malloc + fill). For
    dace, resolve it to ``<name> = np.zeros/np.ones((<shape>,), dtype=<dctype>)`` from
    the kir's ``zeros_locals`` shape map and ``zeros_fills`` kind map (np/dace need an
    explicit shape + dtype). The marker is shared by every allocator alias, so pick the
    constructor from the recorded fill kind: ``ones``/``ones_like`` -> ``np.ones``,
    everything else (``zeros``/``empty``/``ndarray``) -> ``np.zeros`` (a safe defined
    value for the uninitialised ``empty`` too). Used when a logical sparse ``A @ x`` was
    lowered to CSR loops over a fixed-shape accumulator (the Krylov solvers, spmm).

    Fill semantics match the C emit's ``is_reassign`` branch, detected the same way it is
    there -- the FIRST marker arg is the ``'__reassign__'`` string constant, not merely
    "the call carries an arg":
    * a ``'__reassign__'`` marker is a whole-array assignment (the lowered ``r = r - alpha
      * Ap``). Once a buffer of that shape exists, a re-marked reassign is an in-place
      reuse whose following loop may READ the old values (``r`` on both sides), so DROP it
      -- re-zeroing would clear the buffer before that read and corrupt it. The FIRST
      marker for a name still allocates: a self-referential reassign can never BE the first
      occurrence (Python must define the name before reading it, so its allocating write
      -- e.g. ``r = b - A@x`` -- came earlier), so a first-seen reassign is always a full
      overwrite (``Ap = A@p``) and zeroing the fresh buffer is correct even inside a loop;
    * a genuine reset marker (no sentinel, e.g. a fresh ``__mm`` matmul accumulator) is
      ALWAYS emitted, so one sitting inside a loop re-fills every iteration.
    The drop is keyed on the SHAPE, not just the name: a same-name local re-bound to a
    different shape (a reshape/transpose transient) is a real re-allocation, and dace
    rebinds the transient, so it re-emits rather than keeping the stale first shape.
    A marker on a name the lowering did NOT register as a zeros-local is a reassignment
    of an EXISTING buffer -- an output param, e.g. spmm's C in ``C[:] = alpha*(A@B) +
    beta*C`` -- and is dropped (never allocated, so a live input like ``beta*C`` is not
    clobbered)."""

    def __init__(self, zeros_locals: Dict[str, tuple], zeros_fills: Dict[str, str], local_dtypes: Dict[str, str],
                 default_dtype: str):
        self.zeros_locals = zeros_locals
        self.zeros_fills = zeros_fills
        self.local_dtypes = local_dtypes
        self.default_dtype = default_dtype
        self.allocated: Dict[str, tuple] = {}  # name -> the shape it was last allocated with

    def visit_Assign(self, node: ast.Assign):
        if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "__optarena_zeros__"):
            return node
        name = node.targets[0].id
        if name not in self.zeros_locals:
            return None  # a reassigned param (spmm's output C): update in place, never allocate
        # Detect the self-referential sentinel exactly as the C/Fortran emitters do (first
        # arg is the ``"__reassign__"`` constant), so the three backends agree.
        is_reassign = any(isinstance(a, ast.Constant) and a.value == "__reassign__" for a in node.value.args)
        shape = self.zeros_locals[name] or ("1", )
        prev_shape = self.allocated.get(name)
        # An in-place reuse (same buffer, same shape) whose following loop reads the OLD
        # values -> drop the re-zero. A shape change (or first sight) still allocates.
        if is_reassign and prev_shape == shape:
            return None
        self.allocated[name] = shape
        ctor = "np.ones" if self.zeros_fills.get(name) in ("ones", "ones_like") else "np.zeros"
        dtype = _dace_dtype(self.local_dtypes.get(name) or self.default_dtype)
        elts = ", ".join(str(s) for s in shape) + ("," if len(shape) == 1 else "")
        return ast.copy_location(ast.parse(f"{name} = {ctor}(({elts}), dtype={dtype})").body[0], node)


#: numpy dtype tag -> dace type expression. Floats route through the
#: precision-driven ``dc_float`` / ``dc_complex_float`` globals the dace
#: framework rebinds per run; integers carry a fixed width.
_DTYPE_TO_DACE = {
    "float64": "dc_float",
    "float32": "dc_float",
    "complex128": "dc_complex_float",
    "complex64": "dc_complex_float",
    "int64": "dc.int64",
    "int32": "dc.int32",
    "int16": "dc.int16",
    "int8": "dc.int8",
    "uint64": "dc.uint64",
    "uint32": "dc.uint32",
    "uint16": "dc.uint16",
    "uint8": "dc.uint8",
    "int": "dc.int64",
    "bool": "dc.bool",
}


def _dace_dtype(tag: str) -> str:
    return _DTYPE_TO_DACE.get(tag, "dc_float")


def _array_annotation(arr) -> str:
    """``a`` of shape ``(LEN_1D,)`` float64 -> ``dc_float[LEN_1D]``."""
    shape = ", ".join(str(s) for s in arr.shape) if arr.shape else "1"
    return f"{_dace_dtype(arr.dtype)}[{shape}]"


class _DesugarTernary(ast.NodeTransformer):
    """dace's frontend rejects a conditional expression as an assignment RHS
    (``t = A if C else B`` -> "the rhs may only be data, numerical/boolean constants
    and symbols"). Rewrite it to an ``if/else`` statement, which dace traces. Covers the
    divide-by-zero guards the lowered sparse solves emit -- gmres's least-squares
    back-substitution: ``factor = H[r,p] / H[p,p] if H[p,p] != 0.0 else 0.0``."""

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        if isinstance(node.value, ast.IfExp) and len(node.targets) == 1:
            tgt = node.targets[0]
            return ast.copy_location(
                ast.If(test=node.value.test,
                       body=[ast.Assign(targets=[copy.deepcopy(tgt)], value=node.value.body)],
                       orelse=[ast.Assign(targets=[copy.deepcopy(tgt)], value=node.value.orelse)]), node)
        return node


class _DesugarOuter(ast.NodeTransformer):
    """dace's frontend has no ``np.outer`` -- it treats the call as an untyped
    Python callback (``Trying to operate on a callback return value with an
    undefined type``). For 1-D operands ``np.outer(a, b)`` is exactly the
    broadcast product, which dace lowers, so rewrite it to ``a[:, None] * b[None,
    :]`` (gemver's rank-1 updates). The ufunc form ``np.add.outer`` is already
    handled upstream by the shared python-backend desugar; this covers the bare
    ``np.outer`` it leaves untouched."""

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "outer"
                and isinstance(node.func.value, ast.Name) and node.func.value.id in ("np", "numpy")
                and len(node.args) == 2 and not node.keywords):
            a, b = ast.unparse(node.args[0]), ast.unparse(node.args[1])
            new = ast.parse(f"({a})[:, None] * ({b})[None, :]", mode="eval").body
            return ast.copy_location(new, node)
        return node


class _DesugarReverseSlice(ast.NodeTransformer):
    """dace rejects a negative-stride subscript (``Negative strides are not
    supported in subscripts. Please use a Map scope``). A full reverse
    ``x[::-1]`` is exactly ``np.flip(x)``, which dace lowers (even on a dynamic
    slice, ``np.flip(r[:k])``), so rewrite it. durbin's ``r[:k][::-1]`` /
    ``y[:k][::-1]``. Runs AFTER the shared python-backend desugar, which lowers
    ``np.flip`` back to ``x[::-1]`` for pythran -- so this re-lifts every reverse
    (original or desugar-introduced) to the form dace accepts."""

    @staticmethod
    def _is_neg_one(node: ast.AST) -> bool:
        # ``-1`` parses to ``UnaryOp(USub, Constant(1))``, not ``Constant(-1)``.
        if isinstance(node, ast.Constant):
            return node.value == -1
        return (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
                and isinstance(node.operand, ast.Constant) and node.operand.value == 1)

    def visit_Subscript(self, node: ast.Subscript):
        self.generic_visit(node)
        sl = node.slice
        if isinstance(sl, ast.Slice) and sl.lower is None and sl.upper is None and self._is_neg_one(sl.step):
            flip = ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="flip", ctx=ast.Load()),
                            args=[node.value],
                            keywords=[])
            return ast.copy_location(flip, node)
        return node


class _DesugarBroadcastAugAssign(ast.NodeTransformer):
    """An in-place augmented assign that BROADCASTS a lower-rank operand into a
    whole array (``data -= mean``: ``data`` is ``[N, M]``, ``mean`` is ``[M]``)
    parses, but dace builds an invalid SDFG -- the reduction edge subset ``[0:M]``
    cannot map onto ``data[0:N, 0:M]`` (``Dimensionality mismatch between src/dst
    subsets``). The equivalent out-of-place binop written back through a full-array
    store lowers cleanly (covariance2's ``centered = data - mean`` builds), so
    rewrite ``A <op>= b`` -> ``A[:] = A <op> b`` for a whole-array (bare Name)
    target. Semantically identical for a same-rank operand too, so it is safe to
    apply to every array-target augassign (covariance / correlation ``data -=
    mean``)."""

    def __init__(self, array_names: set):
        self.array_names = set(array_names)

    def visit_AugAssign(self, node: ast.AugAssign):
        self.generic_visit(node)
        if not (isinstance(node.target, ast.Name) and node.target.id in self.array_names):
            return node
        load = ast.Name(id=node.target.id, ctx=ast.Load())
        binop = ast.BinOp(left=load, op=node.op, right=node.value)
        store = ast.Subscript(value=ast.Name(id=node.target.id, ctx=ast.Load()),
                              slice=ast.Slice(lower=None, upper=None, step=None),
                              ctx=ast.Store())
        return ast.copy_location(ast.Assign(targets=[store], value=binop), node)


class _DesugarChainedAssign(ast.NodeTransformer):
    """dace cannot codegen a multi-target assignment whose targets are slices
    (``cov[i:M, i] = cov[i, i:M] = rhs`` -> ``Write slicing not implemented``).
    Split it into a single evaluation of ``rhs`` into a temp followed by one
    assignment per target (covariance / correlation's symmetric fill). Semantics
    are preserved: Python evaluates the chained RHS once and assigns left-to-right,
    which is exactly ``t = rhs; a = t; b = t`` -- and on any overlap the last
    target still wins."""

    def __init__(self):
        self.ctr = 0

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        if len(node.targets) <= 1:
            return node
        tmp = f"__optarena_chain{self.ctr}"
        self.ctr += 1
        stmts: List[ast.stmt] = [ast.Assign(targets=[ast.Name(id=tmp, ctx=ast.Store())], value=node.value)]
        for tgt in node.targets:
            stmts.append(ast.Assign(targets=[tgt], value=ast.Name(id=tmp, ctx=ast.Load())))
        for s in stmts:
            ast.copy_location(s, node)
        return stmts


class _SubstituteNames(ast.NodeTransformer):
    """Replace every load of a name in ``mapping`` with a copy of its expression."""

    def __init__(self, mapping: Dict[str, ast.AST]):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id in self.mapping:
            return ast.copy_location(copy.deepcopy(self.mapping[node.id]), node)
        return node


class _DropAliasAssign(ast.NodeTransformer):
    """Drop ``<name> = ...`` for each inlined alias name (its uses are substituted)."""

    def __init__(self, names):
        self.names = set(names)

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in self.names:
            return None
        return node


#: numpy allocators whose FIRST argument is a shape tuple (so the identifiers inside it
#: are array DIMENSIONS that dace requires to be symbolic, not runtime scalars).
_ALLOC_FUNCS = frozenset({"zeros", "empty", "ones"})


def _is_symbol_expr(node: ast.AST, allowed: set) -> bool:
    """True iff ``node`` is a shape expression dace can evaluate as a symbol: names drawn
    from ``allowed``, integer constants, ``+ - * // %`` / unary sign, and ``min``/``max``
    (the closed-form the caller can re-evaluate to bind the promoted symbol)."""
    if isinstance(node, ast.Name):
        return node.id in allowed
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
        return _is_symbol_expr(node.left, allowed) and _is_symbol_expr(node.right, allowed)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_symbol_expr(node.operand, allowed)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("min", "max"):
        return bool(node.args) and all(_is_symbol_expr(a, allowed) for a in node.args)
    return False


def _shape_ident_candidates(fn_ast: ast.AST, known: set) -> set:
    """Identifiers used inside an ``np.zeros/empty/ones`` shape argument that are not
    already an array / scalar / symbol -- body-computed scalars that feed a transient's
    shape, which dace forbids (shapes must be symbolic). These are promoted to symbols."""
    names = set()
    for node in ast.walk(fn_ast):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _ALLOC_FUNCS
                and node.args):
            for sub in ast.walk(node.args[0]):
                if isinstance(sub, ast.Name) and sub.id not in known:
                    names.add(sub.id)
    return names


def _scan_size_assigns(fn_ast: ast.AST, targets: set):
    """For each name in ``targets`` assigned in the body: its FIRST (defining) RHS, the
    first-def order, and which names are assigned more than once. ``ast.walk`` is
    breadth-first, so a top-level def is always seen before a loop-nested reassignment."""
    first_rhs, order, counts = {}, [], {}
    for node in ast.walk(fn_ast):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            nm = node.targets[0].id
            if nm in targets:
                counts[nm] = counts.get(nm, 0) + 1
                if nm not in first_rhs:
                    first_rhs[nm] = node.value
                    order.append(nm)
    reassigned = {nm for nm, c in counts.items() if c > 1}
    return first_rhs, order, reassigned


def _inline_symbol_aliases(fn_ast: ast.AST, symbols: set, known: set) -> ast.AST:
    """Inline a shape scalar that is defined as a pure SYMBOLIC expression over the
    already-declared dc.symbols, instead of promoting it to a fresh symbol.

    A column reduction lowers ``data.mean(axis=0)`` to an accumulator sized by a
    fresh dim ``__rd0_d1``, with a body def ``__rd0_d1 = M`` (after
    :class:`_ShapeToSymbol` resolves ``data.shape[1]``). If that name is promoted
    to its OWN ``dc.symbol``, the frontend cannot prove ``__rd0_d1 == M``, so the
    later ``data -= mean`` (``mean`` shaped ``[__rd0_d1]``) fails to broadcast into
    ``[N, M]`` (covariance / covariance2 / correlation). Substituting the RHS makes
    the transient ``np.empty((M,))`` -- expressed in the arrays' OWN symbols -- and
    the broadcast is provable. Only single-assignment shape idents whose def is a
    pure symbol expression qualify; a runtime-derived size (gmres
    ``m = min(max_iter, n)`` with a scalar ``max_iter``) is not pure-symbolic, so it
    is left for :func:`_plan_size_promotion` to bind as a symbol."""
    shape_idents = _shape_ident_candidates(fn_ast, known)
    if not shape_idents:
        return fn_ast
    first_rhs, order, reassigned = _scan_size_assigns(fn_ast, shape_idents)
    alias: Dict[str, ast.AST] = {}
    for nm in order:
        if nm in reassigned:
            continue
        if _is_symbol_expr(first_rhs[nm], symbols | set(alias)):
            alias[nm] = _SubstituteNames(alias).visit(copy.deepcopy(first_rhs[nm]))
    if not alias:
        return fn_ast
    fn_ast = _SubstituteNames(alias).visit(fn_ast)
    fn_ast = _DropAliasAssign(alias).visit(fn_ast)
    ast.fix_missing_locations(fn_ast)
    return fn_ast


def _plan_size_promotion(fn_ast: ast.AST, known: set):
    """Plan the promotion of body-computed size scalars to dace symbols.

    Returns ``(order, symbol_defs, reassigned)``: ``order`` the names to declare as
    ``dc.symbol`` (dependency order); ``symbol_defs`` ``[(name, rhs_src)]`` so the caller
    can re-evaluate each dimension to bind it (an ABI value, not a program argument);
    ``reassigned`` the subset the body also mutates (a runtime iteration count that must
    be split off the allocation symbol). Empty when nothing needs promotion, or when a
    def is not a pure symbol expression (then the kernel is left unchanged / unbuildable
    rather than silently mis-lowered)."""
    cand = _shape_ident_candidates(fn_ast, known)
    if not cand:
        return [], [], set()
    body_assigned = {
        a.targets[0].id
        for a in ast.walk(fn_ast)
        if isinstance(a, ast.Assign) and len(a.targets) == 1 and isinstance(a.targets[0], ast.Name)
    }
    # Transitive closure: a promoted def's own operands must be symbols too, so pull in
    # any body-assigned dependency (m = min(max_iter, n) drags in n).
    first_rhs, order, reassigned = _scan_size_assigns(fn_ast, cand)
    changed = True
    while changed:
        changed = False
        for nm in list(order):
            for sub in ast.walk(first_rhs[nm]):
                if isinstance(sub, ast.Name) and sub.id not in known and sub.id not in cand and sub.id in body_assigned:
                    cand.add(sub.id)
                    changed = True
        if changed:
            first_rhs, order, reassigned = _scan_size_assigns(fn_ast, cand)
    allowed = known | cand
    symbol_defs = []
    for nm in order:
        if not _is_symbol_expr(first_rhs[nm], allowed):
            return [], [], set()  # non-symbolic size -> not safely promotable
        symbol_defs.append((nm, ast.unparse(first_rhs[nm])))
    # Every candidate must have a def we can bind; a shape ident with no body assignment
    # would be an unbound symbol, so refuse the whole promotion (leave the kernel as-is).
    if set(order) != cand:
        return [], [], set()
    return order, symbol_defs, reassigned


class _SplitReassignedSize(ast.NodeTransformer):
    """A promoted size symbol the body also REASSIGNS (gmres's ``m = k + 1`` on early
    convergence) can't remain a single symbol: dace symbols are immutable, and that
    reassignment is the *runtime* iteration count, not the allocation size. Keep the
    symbol for ALLOCATION shapes (dace needs a symbol there) and route every other use --
    loop bounds, indices -- through a runtime scalar ``<name>_iter`` seeded from the
    symbol and updated by the reassignment. The workspace is allocated to the symbolic
    upper bound; the tail past ``<name>_iter`` stays zero and is never read into the
    result, so this matches the reference exactly. The defining assignment (the first,
    e.g. ``m = min(max_iter, n)``) is dropped -- the caller binds the symbol's value."""

    def __init__(self, names):
        self.names = set(names)
        self._defined = set()  # first assignment per name = the (dropped) def
        self._in_alloc_shape = False

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in self.names:
            nm = node.targets[0].id
            if nm not in self._defined:
                self._defined.add(nm)
                return None  # drop the defining assignment; the symbol value is caller-bound
        self.generic_visit(node)  # a reassignment: target + rhs uses rename to <name>_iter
        return node

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr in _ALLOC_FUNCS and node.args:
            prev, self._in_alloc_shape = self._in_alloc_shape, True
            node.args[0] = self.visit(node.args[0])  # shape arg: leave the symbol in place
            self._in_alloc_shape = prev
            node.args[1:] = [self.visit(a) for a in node.args[1:]]
            node.keywords = [self.visit(k) for k in node.keywords]
            return node
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name):
        if node.id in self.names and not self._in_alloc_shape:
            node.id = f"{node.id}_iter"
        return node


def emit_dace(kir: KernelIR, fn_name: str | None = None) -> str:
    """Return the source of a ``<short>_dace.py`` module for ``kir``."""
    name = fn_name or kir.kernel_name
    arrays = {a.name: a for a in kir.arrays}
    scalars = {s.name: s for s in kir.scalars}
    symbol_names = [s.name for s in kir.symbols]
    # Sparse kirs carry their size symbols (nnz, M, N) ONLY in the array shapes, not
    # kir.symbols -- collect every free identifier in a shape that is not itself an
    # array or scalar, so it is declared as a dc.symbol below (otherwise the shape
    # annotation ``A_indptr: dc.int64[M + 1]`` references an undefined name).
    arr_shapes = {a.name: [str(s) for s in a.shape] for a in kir.arrays}
    _known = set(arrays) | set(scalars)
    for _toks in arr_shapes.values():
        for _tok in _toks:
            for _ident in _IDENT_RE.findall(_tok):
                if _ident not in _known and _ident not in symbol_names:
                    symbol_names.append(_ident)

    # Program signature: arrays + scalars in their original input_args
    # order; symbols are module-level, not parameters.
    params: List[str] = []
    for arg in kir.input_args:
        if arg in arrays:
            params.append(f"{arg}: {_array_annotation(arrays[arg])}")
        elif arg in scalars:
            params.append(f"{arg}: {_dace_dtype(scalars[arg].dtype)}")
        # symbols: skip (declared at module scope below)

    needs_complex = any(_dace_dtype(a.dtype) == "dc_complex_float"
                        for a in kir.arrays) or any(_dace_dtype(s.dtype) == "dc_complex_float" for s in kir.scalars)

    # Body: the numpy reference desugared for the verbatim-body backends -- the
    # SAME pass numba/pythran use, so dace gains feature parity (np.fft, fancy
    # multi-index gather, np.add.at scatter, axis reductions, boolean-mask
    # assignment, ... lower to the plain loops a @dc.program traces). The
    # function is renamed to kir.kernel_name first so the desugar seeds its rank/
    # dtype tables from the kir arrays; falls back to the verbatim body if the
    # desugar cannot parse (it never should for a valid kernel).
    fn_ast = copy.deepcopy(kir.tree)
    fn_ast.name = kir.kernel_name
    try:
        desugared = desugar_for_python_backend(ast.unparse(fn_ast), kir, backend="dace")
        fn_ast = next(n for n in ast.parse(desugared).body if isinstance(n, ast.FunctionDef))
    except Exception:  # noqa: BLE001 -- keep the verbatim body if desugar fails
        fn_ast = kir.tree
    # dace's frontend has no conditional-expression RHS: lower ``t = A if C else B`` to
    # an if/else statement (the divide-by-zero guards in the lowered solves).
    fn_ast = _DesugarTernary().visit(fn_ast)
    # dace has no ``np.outer`` (untyped callback) and rejects a negative-stride
    # subscript. Rewrite ``np.outer(a, b)`` -> ``a[:, None] * b[None, :]`` (gemver) and
    # ``x[::-1]`` -> ``np.flip(x)`` (durbin). Runs after the shared python-backend
    # desugar, which lowers np.flip BACK to ``x[::-1]`` for pythran.
    fn_ast = _DesugarOuter().visit(fn_ast)
    fn_ast = _DesugarReverseSlice().visit(fn_ast)
    # dace cannot codegen a chained slice assignment (``a[s1] = a[s2] = rhs``):
    # evaluate rhs into a temp, then assign each target (covariance symmetric fill).
    fn_ast = _DesugarChainedAssign().visit(fn_ast)
    # A broadcasting in-place augassign into a whole array (``data -= mean``) builds an
    # invalid SDFG; rewrite it to an explicit write-back binop (covariance/correlation).
    fn_ast = _DesugarBroadcastAugAssign(set(arrays)).visit(fn_ast)
    ast.fix_missing_locations(fn_ast)
    # A logical sparse ``A @ x`` lowered to CSR loops leaves ``<acc> =
    # __optarena_zeros__()`` markers for its fixed-shape accumulators -- turn them into
    # ``np.zeros`` / ``np.ones`` (per the recorded fill kind) so the @dc.program
    # allocates the transient with its declared initial value.
    zeros_locals = kir.zeros_locals
    zeros_fills = kir.zeros_fills
    local_dtypes = kir.local_dtypes
    default_dtype = kir.float_precision or "float64"
    fn_ast = _ResolveZeros(zeros_locals, zeros_fills, local_dtypes, default_dtype).visit(fn_ast)
    # dace has no runtime ``.shape`` and its symbols are immutable: rewrite
    # ``arr.shape[k]`` to the symbolic dimension and drop any recompute of a size
    # symbol (``M = A_indptr.shape[0] - 1``), which is redundant + illegal in dace.
    fn_ast = _ShapeToSymbol(arr_shapes).visit(fn_ast)
    # A shape scalar that is just a pure symbolic alias of an existing dc.symbol (a
    # reduction's ``__rd0_d1 = M``) is INLINED to that symbol, so the transient it sizes
    # is expressed in the arrays' own symbols and the frontend can prove the broadcast
    # (covariance ``data -= mean``) -- rather than promoting it to a fresh symbol dace
    # cannot prove equal to ``M``.
    fn_ast = _inline_symbol_aliases(fn_ast, set(symbol_names), set(arrays) | set(scalars) | set(symbol_names))
    # dace forbids a data-dependent (runtime-scalar) array shape, but the lowered Krylov
    # workspaces have body-computed dims (gmres ``Q = zeros((n, m + 1))`` with
    # ``m = min(max_iter, n)``). Promote those size scalars to dc.symbols the caller
    # binds; split off a runtime iteration count for a size the body also reassigns.
    promoted, symbol_defs, reassigned = _plan_size_promotion(fn_ast, set(arrays) | set(scalars) | set(symbol_names))
    for nm in promoted:
        if nm not in symbol_names:
            symbol_names.append(nm)
    if reassigned:
        fn_ast = _SplitReassignedSize(reassigned).visit(fn_ast)
        ast.fix_missing_locations(fn_ast)
        fn_ast.body[0:0] = [ast.parse(f"{nm}_iter = {nm}").body[0] for nm in reassigned]
    fn_ast = _DropSymbolAssign(symbol_names).visit(fn_ast)
    ast.fix_missing_locations(fn_ast)
    body = list(fn_ast.body)
    if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]

    out: List[str] = []
    out.append('"""DaCe program auto-generated from the numpy reference '
               'by numpyto_c.dace_emit."""')
    out.append("import numpy as np")
    out.append("import dace as dc")
    imp = "dc_float, dc_complex_float" if needs_complex else "dc_float"
    out.append(f"from optarena.infrastructure.dace_framework import {imp}")
    out.append("from math import sin, cos, log, exp, pow, sqrt")
    out.append("")
    if symbol_names:
        names = ", ".join(symbol_names)
        srcs = ", ".join(f"'{s}'" for s in symbol_names)
        if len(symbol_names) == 1:
            out.append(f"{names} = dc.symbol({srcs}, dtype=dc.int64)")
        else:
            out.append(f"{names} = (dc.symbol(s, dtype=dc.int64) "
                       f"for s in ({srcs}))")
        out.append("")
    if symbol_defs:
        # Per-dimension binding recipe for the harness: each promoted symbol is not a
        # program argument, so the caller evaluates these (in order) over the known
        # symbol values to supply them at call time. See sparse_oracle._run_dace.
        out.append(f"__optarena_symbol_defs__ = {symbol_defs!r}")
        out.append("")
    out.append("")
    out.append("@dc.program")
    out.append(f"def {name}({', '.join(params)}):")
    if not body:
        out.append("    pass")
    else:
        for stmt in body:
            for line in ast.unparse(stmt).splitlines():
                out.append("    " + line)
    return "\n".join(out) + "\n"
