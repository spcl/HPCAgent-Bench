"""AST rewrites that translate the numpy-numeric subset into plain loops.

Three kinds of rewrites:

1. **Identifier renames** for ``math.*`` and ``np.*`` numeric
   primitives that map to a single C function:
   ``math.exp(x)`` -> ``exp(x)``, ``math.sqrt(x)`` -> ``sqrt(x)``,
   ``np.fabs(x)`` -> ``fabs(x)``, etc.

2. **Library-node expansions** that turn a single Python call into a
   small loop nest:

   * ``np.zeros((N, K))`` -- introduce a local array declaration of
     the appropriate shape (the assignment LHS provides the name).
   * ``A @ B`` -- naive triple-loop GEMM into the assignment LHS.
   * ``np.dot(a, b)`` -- accumulator loop.
   * ``np.sum(A)`` -- accumulator loop.

3. **Slice fusion** -- when an assignment's LHS is a slice and its
   RHS contains other slices, emit ONE loop nest that scalarises every
   slice. Concretely::

       A[1:N-1] = (B[:N-2] + B[2:] + B[1:N-1]) / 3.0

   lowers to a single ``for i in range(1, N-1)`` whose body reads
   ``B[i-1]`` / ``B[i+1]`` / ``B[i]`` -- not four separate temporary
   arrays. Multi-dim slices nest accordingly. See :class:`SliceFusion`.

The Foundation corpus exercises only ``math.exp``, ``math.sqrt`` and
``np.zeros``; the other rules are declared but inert until a kernel uses them.
"""

import ast
import copy
import os
import re
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import sympy

from numpyto_common.ir import _COMPLEX_FOR_FLOAT, KernelIR
from numpyto_common.numpy_desugar import _np_linalg_attr
from numpyto_common.lib_nodes import (MESHGRID_AXIS_KW, _iter_extent_of,
                                      _scalarize_at_iters, expand_meshgrid)

#: One-to-one rewrites: ``<module>.<name>`` -> bare C function name.
#: All targets resolve through ``<math.h>``.
#: Trigonometric / algebraic / transcendental intrinsics.
_TRIG = ("sin", "cos", "tan", "tanh",
         "asin", "acos", "atan", "atan2",
         "sinh", "cosh", "asinh", "acosh", "atanh",
         "hypot")
_ALG_TRANS = ("exp", "exp2", "expm1",
              "log", "log2", "log10", "log1p",
              "sqrt", "cbrt", "pow",
              "fabs", "floor", "ceil", "round", "rint", "trunc",
              "fmod", "fmax", "fmin", "copysign",
              "erf", "erfc",
              "tgamma", "lgamma")
#: Math intrinsics whose C name collides with common kernel variable
#: names (Bessel functions ``j0``/``j1``/``y0``/``y1``). When the user
#: writes ``np.j0(x)`` we rename to ``bessel_j0(x)`` and emit a static
#: forwarder that delegates to the libm intrinsic; the local variable
#: shadowing risk is sidestepped.
_BESSEL_INTRINSICS: Dict[str, str] = {
    "j0": "bessel_j0", "j1": "bessel_j1",
    "y0": "bessel_y0", "y1": "bessel_y1",
}

#: Map ``("math", x) -> x`` for every entry in ``_TRIG`` + ``_ALG_TRANS``.
MATH_BUILTINS: Dict[Tuple[str, str], str] = {
    ("math", n): n for n in (*_TRIG, *_ALG_TRANS)
}
#: ``np.<intrinsic>(scalar)`` rename. The elementwise expander catches
#: array args BEFORE this rename fires (LibNodeRewriter runs first
#: through ``NP_CALL_EXPANDERS``); the rename only succeeds for scalar
#: arg forms like ``np.tanh(a[i, i])``.
MATH_BUILTINS.update({("np", n): n for n in (*_TRIG, *_ALG_TRANS)})
#: numpy aliases that don't share their C name.
MATH_BUILTINS[("np", "arctan2")] = "atan2"
MATH_BUILTINS[("np", "arcsin")] = "asin"
MATH_BUILTINS[("np", "arccos")] = "acos"
MATH_BUILTINS[("np", "arctan")] = "atan"
MATH_BUILTINS[("np", "arcsinh")] = "asinh"
MATH_BUILTINS[("np", "arccosh")] = "acosh"
MATH_BUILTINS[("np", "arctanh")] = "atanh"
MATH_BUILTINS[("np", "abs")] = "fabs"
MATH_BUILTINS[("np", "absolute")] = "fabs"
MATH_BUILTINS[("np", "power")] = "pow"
MATH_BUILTINS[("np", "maximum")] = "fmax"  # 2-arg scalar form falls here
MATH_BUILTINS[("np", "minimum")] = "fmin"
# Bessel functions renamed to avoid kernel-name collisions.
for _orig, _renamed in _BESSEL_INTRINSICS.items():
    MATH_BUILTINS[("math", _orig)] = _renamed
    MATH_BUILTINS[("np", _orig)] = _renamed
    MATH_BUILTINS[("scipy.special", _orig)] = _renamed
#: Identifiers the parameter-promotion pass must NOT lift to int
#: parameters (they resolve to C / Fortran intrinsics post-emit).
_MATH_INTRINSIC_NAMES: Set[str] = (
    set(_TRIG) | set(_ALG_TRANS) | set(_BESSEL_INTRINSICS.values())
    | {"__npb_sign"})   # np.sign marker; specialised per-backend in emit


#: Method-call form -> free-function rewrite. The rewriter dynamically
#: replaces the method invocation with the ``np.X(arr, ...)`` form so
#: downstream lowering never sees the method syntax.
#: Only reductions (max / min / sum / mean / prod / std) and ``copy``
#: are supported -- they have no kwargs the call-hoister can't handle
#: in their bare form. Reshape / transpose method forms are rejected
#: (they take shape / perm tuples that complicate scalar broadcast).
_METHOD_TO_NP: Dict[str, str] = {
    "copy": "copy",
    "max": "max",
    "min": "min",
    "sum": "sum",
    "mean": "mean",
    "prod": "prod",
    "std": "std",
    "any": "any",
    "all": "all",
    "argmax": "argmax",
    "argmin": "argmin",
}


class _StmtHoister(ast.NodeTransformer):
    """Base for rewriters that must lift a sub-expression into a fresh temp
    assignment emitted immediately before the statement that contains it.

    A subclass calls :meth:`_spill` from its expression visitor to swap an
    inline sub-expression for a fresh Name and stage ``<temp> = <expr>`` in
    :attr:`pre_stmts`; this base flushes the staged assignments into the
    enclosing block right before the current statement. Mirrors the
    ``pre_stmts`` lift ``_ScalarTimesMatmulRewriter`` uses -- generalised so the
    splice happens inline (``NodeTransformer`` flattens a returned statement
    list into the parent body) rather than at the top-level driver, so a spill
    inside a loop / branch body lands in that same body at any nesting depth.

    The per-statement save/restore of :attr:`pre_stmts` keeps a spill from a
    compound statement's header (``if`` test / ``for`` iter) separate from
    spills produced by its body statements: the header's temps flush before the
    compound statement, each body statement's temps flush before that body
    statement.
    """

    def __init__(self):
        #: Temp assignments staged for the statement currently being flushed.
        self.pre_stmts: List[ast.stmt] = []
        #: Monotonic id for unique hoist-temp names across the whole body.
        self._hoist_ctr: List[int] = [0]

    def _spill(self, expr: ast.expr, prefix: str) -> ast.Name:
        """Stage ``<prefix><n> = <expr>`` and return a Load Name for the temp."""
        self._hoist_ctr[0] += 1
        name = f"{prefix}{self._hoist_ctr[0]}"
        self.pre_stmts.append(
            ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=expr))
        return ast.Name(id=name, ctx=ast.Load())

    def _flush(self, node: ast.stmt):
        saved = self.pre_stmts
        self.pre_stmts = []
        self.generic_visit(node)
        pre = self.pre_stmts
        self.pre_stmts = saved
        if not pre:
            return node
        for s in pre:
            ast.copy_location(s, node)
            ast.fix_missing_locations(s)
        return pre + [node]

    visit_Assign = _flush
    visit_AugAssign = _flush
    visit_Expr = _flush
    visit_Return = _flush
    visit_If = _flush
    visit_While = _flush
    visit_For = _flush


class _MethodCallRewriter(_StmtHoister):
    """Translate ``a.copy()``, ``A.max()`` etc. into their ``np.``
    counterparts so the LibNodeRewriter picks them up uniformly.

    Fires when the receiver is a bare Name (a parameter or declared local) or a
    Subscript of one -- neither a module-like identifier (``np`` / ``numpy`` /
    ``math`` / ``scipy``), else ``np.max(x)`` would wrongly become
    ``np.max(np, x)``. A Call receiver (``np.abs(rho_in - rho_out).sum()``) is
    hoisted to a fresh temp first (:class:`_StmtHoister`), so the method operates
    on a bare Name -- the reduction expanders and backends never accept an
    inline sub-expression receiver.
    """

    _MODULE_NAMES = frozenset({"np", "numpy", "math", "scipy"})

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        func = node.func
        # ``a.ravel()`` / ``a.flatten()`` -> ``np.reshape(a, (-1,))``: a 1-D view /
        # copy the reshape expander already lowers (``.ravel() @ .ravel()`` is the
        # flattened-dot idiom in the CG kernels).
        if (isinstance(func, ast.Attribute) and func.attr in ("ravel", "flatten")
                and not node.args
                and isinstance(func.value, ast.Name)
                and func.value.id not in self._MODULE_NAMES):
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="reshape", ctx=ast.Load()),
                args=[func.value, ast.Tuple(elts=[ast.Constant(value=-1)], ctx=ast.Load())],
                keywords=[])
        if not (isinstance(func, ast.Attribute) and func.attr in _METHOD_TO_NP):
            return node
        recv = func.value
        # Receiver may be a bare Name (a parameter / declared local) or a
        # Subscript of one (``grid[0].copy()`` -- a row materialised into a
        # fresh local). Both lower through the same ``np.<fn>`` expanders, which
        # scalarize Subscript operands. A module identifier (``np.max(x)``) is
        # never a receiver here -- that ``x`` is the argument, not the receiver.
        if isinstance(recv, ast.Call):
            # Call receiver (``np.abs(rho_in - rho_out).sum()``): materialise the
            # inner Call into a fresh temp emitted before this statement, then
            # reduce over the bare Name. The reduction expanders / backends only
            # accept a Name or Subscript receiver, never an inline Call.
            recv = self._spill(recv, "__mc")
        elif not ((isinstance(recv, ast.Name) and recv.id not in self._MODULE_NAMES)
                  or (isinstance(recv, ast.Subscript)
                      and isinstance(recv.value, ast.Name)
                      and recv.value.id not in self._MODULE_NAMES)):
            return node
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr=_METHOD_TO_NP[func.attr], ctx=ast.Load()),
            args=[recv] + list(node.args),
            keywords=node.keywords)


class _ComputedIndexCallHoister(_StmtHoister):
    """Hoist a Call used as a subscript index into a fresh temp Name assignment
    emitted before the statement, so the index is a bare Name the backends emit.

    ``U[np.argmax(absU[:, j]), j]`` / ``v[np.argmax(np.abs(v))]`` -- a library
    Call sitting in index position -- becomes ``__ix = np.argmax(...)`` staged
    before the statement and the subscript indexes with ``__ix``. Structural, not
    argmax-specific: any Call index (single-index or ANY position of a tuple
    subscript) is spilled EXCEPT the scalar builtins the emitters already render
    inline as an index expression (``int`` / ``abs`` / ``min`` / ``max`` /
    ``len`` / ``round``), which need no pre-statement.

    ``argmax`` / ``argmin`` index calls need one extra step: their expander
    requires a bare-Name operand, but the LibNode call-hoister only materialises a
    non-Name first arg for the VALUE reductions (max / min / sum / ...), never for
    argmax / argmin. So when the hoisted index call is an argmax / argmin over a
    non-Name operand -- a slice (``absU[:, j]``) or a nested Call
    (``np.abs(w)``) -- that operand is materialised into its own temp Name first
    (a whole-array copy the later lift lowers), so the reduction reaches its
    expander with a Name operand.
    """

    #: Scalar builtins each backend renders inline in index position -- left in
    #: place so a plain ``hist[int(x)]`` does not gain a needless spill temp.
    _INLINE_INDEX_BUILTINS = frozenset({"int", "abs", "min", "max", "len", "round"})

    #: ``np`` arg-reductions whose expander needs a bare-Name operand (the
    #: LibNode call-hoister leaves their non-Name first arg unmaterialised).
    _ARG_REDUCTIONS = frozenset({"argmax", "argmin"})

    def _should_hoist(self, e: ast.expr) -> bool:
        if not isinstance(e, ast.Call):
            return False
        f = e.func
        if isinstance(f, ast.Name) and f.id in self._INLINE_INDEX_BUILTINS:
            return False
        return True

    def _is_arg_reduction(self, call: ast.Call) -> bool:
        f = call.func
        return (isinstance(f, ast.Attribute) and f.attr in self._ARG_REDUCTIONS
                and isinstance(f.value, ast.Name) and f.value.id in ("np", "numpy"))

    def _hoist_index(self, e: ast.Call) -> ast.Name:
        """Spill index Call ``e`` to a fresh Name. For an argmax / argmin over a
        non-Name operand, materialise that operand into its own temp Name first so
        the reduction expander (which needs a Name operand) can lower it."""
        if (self._is_arg_reduction(e) and e.args
                and not isinstance(e.args[0], ast.Name)):
            e.args[0] = self._spill(e.args[0], "__ixa")
        return self._spill(e, "__ix")

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        sl = node.slice
        is_tuple = isinstance(sl, ast.Tuple)
        elts = list(sl.elts) if is_tuple else [sl]
        new_elts = [self._hoist_index(e) if self._should_hoist(e) else e
                    for e in elts]
        if new_elts != elts:
            node.slice = (ast.Tuple(elts=new_elts, ctx=ast.Load()) if is_tuple
                          else new_elts[0])
        return node


class _AstypeRewriter(ast.NodeTransformer):
    """Lower ``<expr>.astype(<dtype>)`` on ANY receiver (not just a bare
    Name, which ``_MethodCallRewriter`` is limited to) into the cast form the
    emitters already handle.

    * ``x.astype(np.int32)`` / ``x.astype(np.float64)`` / ``x.astype("int64")``
      -> ``np.int32(x)`` -- a per-element cast that the emitters render as a
      C/Fortran cast (integer targets truncate, matching numpy).
    * ``x.astype(other.dtype)`` -> ``x`` -- a cast to *another array's* dtype is
      realised by the destination's declared dtype on store, so it drops to the
      receiver (the frontend already read the ``.astype`` for dtype inference).
      This is what makes ``(labels[:, None] == ids).astype(X.dtype)`` (kmeans)
      and ``(level == d).astype(np.int64)`` (bfs) lowerable.
    """

    def __init__(self, array_dtypes: Optional[Dict[str, str]] = None):
        #: ``{array_name: dtype}`` so ``(cmp).astype(X.dtype)`` can resolve
        #: ``X.dtype`` to a concrete cast when the receiver is logical.
        self.array_dtypes = array_dtypes or {}

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "astype" and node.args):
            return node
        recv, dt = f.value, node.args[0]
        name = None
        if (isinstance(dt, ast.Attribute) and isinstance(dt.value, ast.Name)
                and dt.value.id in ("np", "numpy")):
            name = dt.attr                       # np.<dtype>
        elif isinstance(dt, ast.Name) and dt.id in ("int", "float", "bool"):
            name = {"int": "int64", "float": "float64", "bool": "bool_"}[dt.id]
        elif isinstance(dt, ast.Constant) and isinstance(dt.value, str):
            name = dt.value                      # "float64" etc.
        if name is None:
            # ``other.dtype``: normally the destination's declared dtype
            # realises the cast, so we drop to the receiver. BUT a comparison /
            # boolean receiver leaves a LOGICAL value -- Fortran cannot store it
            # into / sum it as the REAL destination (kmeans' one-hot
            # ``(labels == ids).astype(X.dtype)``). Resolve the source array's
            # dtype and emit the concrete cast so the merge(1, 0, cond) path
            # fires and the destination is declared REAL.
            if (isinstance(recv, (ast.Compare, ast.BoolOp))
                    and isinstance(dt, ast.Attribute) and dt.attr == "dtype"
                    and isinstance(dt.value, ast.Name)):
                name = self.array_dtypes.get(dt.value.id)
            if name is None:
                return recv
        return ast.copy_location(ast.Call(
            func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()),
                               attr=name, ctx=ast.Load()),
            args=[recv], keywords=[]), node)


def _match_reshape(node: ast.AST):
    """If ``node`` is a reshape call (method ``X.reshape(shape...)`` OR func
    ``np.reshape(X, shape)``), return ``(base_expr, shape_elts)`` -- the array
    being reshaped and the list of shape AST elements. Else ``None``.

    Both the single-tuple (``X.reshape((a, b))``) and the varargs
    (``X.reshape(a, b)``) method spellings are accepted."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "reshape"):
        return None
    recv = node.func.value
    if isinstance(recv, ast.Name) and recv.id in ("np", "numpy"):
        if len(node.args) < 2:
            return None
        shape = node.args[1]
        elts = list(shape.elts) if isinstance(shape, (ast.Tuple, ast.List)) else [shape]
        return node.args[0], elts
    # method form: receiver is the array
    if len(node.args) == 1 and isinstance(node.args[0], (ast.Tuple, ast.List)):
        return recv, list(node.args[0].elts)
    return recv, list(node.args)


class _ReshapeMethodRewriter(ast.NodeTransformer):
    """Normalize the method form ``X.reshape(a, b)`` / ``X.reshape((a, b))`` to
    the function form ``np.reshape(X, (a, b))`` so the single ``expand_reshape``
    path handles every spelling (lulesh uses the varargs method form)."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "reshape"):
            return node
        recv = node.func.value
        if isinstance(recv, ast.Name) and recv.id in ("np", "numpy"):
            return node            # already the function form
        matched = _match_reshape(node)
        if matched is None:
            return node
        base, elts = matched
        # Preserve ``order=`` (C/F) so expand_reshape can honour a column-major
        # reshape; every other kwarg is dropped (the method form has none else).
        keep = [kw for kw in node.keywords if kw.arg == "order"]
        return ast.copy_location(ast.Call(
            func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="reshape", ctx=ast.Load()),
            args=[base, ast.Tuple(elts=elts, ctx=ast.Load())], keywords=keep), node)


_FFT_FNS = {"fftn": (False, True), "ifftn": (True, True),
            "fft": (False, False), "ifft": (True, False)}


def _match_fft(node: ast.AST):
    """If ``node`` is ``np.fft.{fftn,ifftn,fft,ifft}(arg, ...)``, return
    ``(fn_name, arg, keywords)``; else ``None``."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in _FFT_FNS):
        return None
    f = node.func
    if (isinstance(f.value, ast.Attribute) and f.value.attr == "fft"
            and isinstance(f.value.value, ast.Name) and f.value.value.id in ("np", "numpy")
            and node.args):
        return f.attr, node.args[0], node.keywords
    return None


class _FftGridReshapeRewriter(ast.NodeTransformer):
    """Lower the QE flat-grid FFT idiom into materialised reshape + fft temps.

    ``np.fft.ifftn(X.reshape((d1, d2, d3, -1)), axes=(0, 1, 2)).reshape(M, -1)``
    (optionally followed by ``[:, 0]``) is a 3-D DFT applied column-wise to an
    ``(M, C)`` array reinterpreted as a ``d1 x d2 x d3`` grid (vexx's
    ``invfft``/``fwfft``). ``X`` has a known 2-D shape ``(M, C)`` so the
    reshape ``-1`` is exactly ``C`` -- no general ``-1`` inference, no reliance
    on the unprovable ``d1*d2*d3 == M`` identity in symbolic form. Rewrite to::

        __g = np.reshape(X, (d1, d2, d3, C))     # split leading axis, keep C
        __f = np.fft.ifftn(__g, axes=(0, 1, 2))  # _expand_dftn batches axis 3
        __o = np.reshape(__f, (M, C))            # flatten back
        <lhs> = __o            # or __o (1-D, length M) when the chain ends [:,0]

    Each temp's shape is registered so LibNodeRewriter's ``expand_reshape`` /
    ``_expand_dftn`` (both C-order, matching numpy) expand them into loops.
    Runs before LibNodeRewriter."""

    def __init__(self, shape_table: Dict[str, Tuple[str, ...]],
                 local_dtypes: Dict[str, str], counter: List[int]):
        self.shape_table = shape_table
        self.local_dtypes = local_dtypes
        self.counter = counter

    def _src_MC(self, src: ast.AST):
        """Resolve the FFT input expression to ``(name, M, C)`` -- a bare Name
        to reshape and its leading/trailing extents. Handles a 2-D Name, a 1-D
        Name (C=1), and ``X[:, None]`` of a 1-D Name (C=1). Else ``None``."""
        if isinstance(src, ast.Name):
            shp = self.shape_table.get(src.id)
            if not shp:
                return None
            return src.id, shp[0], (shp[1] if len(shp) > 1 else "1")
        if (isinstance(src, ast.Subscript) and isinstance(src.value, ast.Name)
                and isinstance(src.slice, ast.Tuple) and len(src.slice.elts) == 2
                and isinstance(src.slice.elts[0], ast.Slice)
                and isinstance(src.slice.elts[1], ast.Constant)
                and src.slice.elts[1].value is None):
            shp = self.shape_table.get(src.value.id)
            if not shp or len(shp) != 1:
                return None
            return src.value.id, shp[0], "1"
        return None

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        rhs = node.value
        # Optional trailing ``[:, k]`` column select.
        col_k = None
        if (isinstance(rhs, ast.Subscript) and isinstance(rhs.slice, ast.Tuple)
                and len(rhs.slice.elts) == 2
                and isinstance(rhs.slice.elts[0], ast.Slice)
                and isinstance(rhs.slice.elts[1], ast.Constant)):
            col_k = rhs.slice.elts[1].value
            chain = rhs.value
        else:
            chain = rhs
        outer = _match_reshape(chain)
        if outer is None:
            return node
        fft_node, out_elts = outer
        fft = _match_fft(fft_node)
        if fft is None:
            return node
        fn_name, inner_call, fft_kw = fft
        inner = _match_reshape(inner_call)
        if inner is None:
            return node
        src_expr, grid_elts = inner
        # grid shape = leading dims + trailing ``-1``; need >=2 dims and -1 last.
        if len(grid_elts) < 2 or ast.unparse(grid_elts[-1]).strip() != "-1":
            return node
        mc = self._src_MC(src_expr)
        if mc is None:
            return node
        src_name, M, C = mc
        grid_dims = [ast.unparse(e) for e in grid_elts[:-1]]
        grid_shape = tuple(grid_dims) + (C,)
        n = self.counter[0]
        self.counter[0] += 3
        g, f, o = f"__fg{n}", f"__ff{n}", f"__fo{n}"

        def _tok(t):
            return ast.parse(t, mode="eval").body

        def _tuple(toks):
            return ast.Tuple(elts=[_tok(t) for t in toks], ctx=ast.Load())

        reshape_g = ast.Assign(
            targets=[ast.Name(id=g, ctx=ast.Store())],
            value=ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()),
                                              attr="reshape", ctx=ast.Load()),
                           args=[ast.Name(id=src_name, ctx=ast.Load()), _tuple(grid_shape)],
                           keywords=[]))
        fft_call = ast.Assign(
            targets=[ast.Name(id=f, ctx=ast.Store())],
            value=ast.Call(func=ast.Attribute(
                value=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()),
                                    attr="fft", ctx=ast.Load()),
                attr=fn_name, ctx=ast.Load()),
                args=[ast.Name(id=g, ctx=ast.Load())], keywords=fft_kw))
        # Output reshape: drop the singleton column when the chain ends ``[:, 0]``
        # (only valid for C == 1, the single-column case); else keep (M, C).
        if col_k is not None:
            if str(C) != "1" or col_k != 0:
                return node
            out_shape = (M,)
        else:
            out_shape = (M, C)
        reshape_o = ast.Assign(
            targets=[ast.Name(id=o, ctx=ast.Store())],
            value=ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()),
                                              attr="reshape", ctx=ast.Load()),
                           args=[ast.Name(id=f, ctx=ast.Load()), _tuple(out_shape)],
                           keywords=[]))
        for nm, shp in ((g, grid_shape), (f, grid_shape), (o, out_shape)):
            self.shape_table[nm] = shp
            self.local_dtypes[nm] = "complex128"
        node.value = ast.Name(id=o, ctx=ast.Load())
        for s in (reshape_g, fft_call, reshape_o, node):
            ast.copy_location(s, node) if isinstance(s, ast.stmt) else None
        ast.fix_missing_locations(reshape_g)
        ast.fix_missing_locations(fft_call)
        ast.fix_missing_locations(reshape_o)
        return [reshape_g, fft_call, reshape_o, node]


#: numpy free-function aliases that are exact synonyms of a name the
#: translator already lowers. Normalising them at the AST level means the whole
#: downstream machinery (expander, shape derivation, hoister) is reused with no
#: duplication. ``permute_dims`` is the array-API spelling of ``transpose``
#: (both take ``(a, axes)``); ``amax``/``amin`` are the long names of max/min.
_NP_FUNC_ALIASES: Dict[str, str] = {
    "permute_dims": "transpose",
    "permute": "transpose",
    "amax": "max",
    "amin": "min",
}


class _ScatterAtRewriter(ast.NodeTransformer):
    """Lower ``np.<op>.at(target, idx, vals)`` (unbuffered scatter) into an
    explicit indexed loop -- the only correct sequential form, since repeated
    indices must accumulate (plain ``target[idx] += vals`` would not).

        np.add.at(Lx, src, flux)      ->  for __k in range(E): Lx[src[__k]] += flux[__k]
        np.subtract.at(Lx, dst, flux) ->  for __k in range(E): Lx[dst[__k]] -= flux[__k]
        np.maximum.at(M, idx, v)      ->  for __k in range(E): M[idx[__k]] = max(M[idx[__k]], v[__k])

    Every binary ufunc exposes ``.at``; we cover the realistic scatter ops:
    arithmetic (add/subtract/multiply/divide -> compound assign) and
    maximum/minimum (no compound operator -> ``t[i] = max(t[i], v)``). ``idx``
    is a 1-D index array (its first extent gives the trip count). ``vals`` is an
    array Name (subscripted per element) or its unary negation; anything else is
    refused rather than mis-lowered. Used by edge_laplacian.
    """

    #: arithmetic ufuncs -> the compound-assign operator (``t[i] op= v``).
    _AUG = {"add": ast.Add, "subtract": ast.Sub, "multiply": ast.Mult,
            "divide": ast.Div, "true_divide": ast.Div}
    #: max/min ufuncs -> a builtin folded into ``t[i] = fn(t[i], v)``.
    _FOLD = {"maximum": "max", "minimum": "min"}

    def __init__(self, shapes: Dict[str, List[str]]):
        self.shapes = shapes
        self._n = 0

    @staticmethod
    def _is_ufunc_at(func: ast.AST):
        # Attribute chain ``np.<op>.at`` -> Attribute(Attribute(Name('np'), op), 'at')
        if (isinstance(func, ast.Attribute) and func.attr == "at"
                and isinstance(func.value, ast.Attribute)
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id in ("np", "numpy")):
            return func.value.attr
        return None

    @staticmethod
    def _index_of(iters: List[str]) -> ast.expr:
        """A scalar subscript index over ``iters`` -- a single Name (1 axis) or a
        Tuple of Names (multi-axis ``arr[k0, k1, ...]``)."""
        if len(iters) == 1:
            return ast.Name(id=iters[0], ctx=ast.Load())
        return ast.Tuple(elts=[ast.Name(id=it, ctx=ast.Load()) for it in iters], ctx=ast.Load())

    def _val_at(self, vals: ast.expr, iters: List[str]) -> ast.expr:
        if isinstance(vals, ast.Name):
            return ast.Subscript(value=ast.Name(id=vals.id, ctx=ast.Load()),
                                 slice=self._index_of(iters), ctx=ast.Load())
        if isinstance(vals, ast.UnaryOp) and isinstance(vals.op, ast.USub) \
                and isinstance(vals.operand, ast.Name):
            return ast.UnaryOp(op=ast.USub(), operand=self._val_at(vals.operand, iters))
        raise NotImplementedError("np.<op>.at value must be an array name or its negation")

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        call = node.value
        if not isinstance(call, ast.Call):
            return node
        op = self._is_ufunc_at(call.func)
        if op is None:
            return node
        if (op not in self._AUG and op not in self._FOLD) or len(call.args) != 3:
            raise NotImplementedError(f"unsupported np.{op}.at form")
        target, idx, vals = call.args
        if not isinstance(target, ast.Name):
            raise NotImplementedError("np.<op>.at needs a Name target")
        # MULTI-index scatter -- the unstructured / semi-structured ICON form
        # ``np.add.at(out, (idx2d - 1, jk, blk2d - 1), val[:, jk, :])``: the
        # index is a TUPLE of mixed indirect-array / scalar axes. Lower to an
        # accumulation loop nest over the (broadcast) value plane.
        if isinstance(idx, ast.Tuple):
            return self._multi_index_scatter(node, op, target, idx, vals)
        if not isinstance(idx, ast.Name):
            raise NotImplementedError("np.<op>.at needs Name target and index array")
        bound = self.shapes.get(idx.id)
        if not bound:
            raise NotImplementedError(f"np.<op>.at: unknown extent for index '{idx.id}'")
        self._n += 1
        # Iterate EVERY axis of the index array (lulesh's nodelist is 2-D
        # ``(numelem, 8)``), so the scatter is a scalar ``target[idx[k0,k1]] op=
        # vals[k0,k1]`` -- not a leading-axis-only loop that leaves the trailing
        # axes as unlowered slices. ``vals`` is indexed with the same iters
        # (it broadcasts to the index shape for a 1-D target).
        # 1-D index keeps the flat ``__sat{n}`` name (the common edge_laplacian
        # case); a multi-D index (lulesh nodelist) suffixes one iter per axis.
        iters = ([f"__sat{self._n}"] if len(bound) == 1
                 else [f"__sat{self._n}_{d}" for d in range(len(bound))])
        idx_k = ast.Subscript(value=ast.Name(id=idx.id, ctx=ast.Load()),
                              slice=self._index_of(iters), ctx=ast.Load())
        val_k = self._val_at(vals, iters)
        if op in self._AUG:
            lhs = ast.Subscript(value=ast.Name(id=target.id, ctx=ast.Load()),
                                slice=idx_k, ctx=ast.Store())
            stmt: ast.stmt = ast.AugAssign(target=lhs, op=self._AUG[op](), value=val_k)
        else:                                   # maximum / minimum -> t[i] = fn(t[i], v)
            lhs = ast.Subscript(value=ast.Name(id=target.id, ctx=ast.Load()),
                                slice=idx_k, ctx=ast.Store())
            cur = ast.Subscript(value=ast.Name(id=target.id, ctx=ast.Load()),
                                slice=idx_k, ctx=ast.Load())
            stmt = ast.Assign(targets=[lhs], value=ast.Call(
                func=ast.Name(id=self._FOLD[op], ctx=ast.Load()),
                args=[cur, val_k], keywords=[]))
        body: List[ast.stmt] = [stmt]
        for it, ext in zip(reversed(iters), reversed(bound)):  # nest deepest-last
            body = [ast.For(
                target=ast.Name(id=it, ctx=ast.Store()),
                iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                              args=[_const_or_name_token(ext)], keywords=[]),
                body=body, orelse=[])]
        return ast.copy_location(body[0], node)

    def _multi_index_scatter(self, node, op, target: ast.Name,
                             idx_tuple: ast.Tuple, vals: ast.expr) -> ast.AST:
        """Lower a TUPLE-index ``np.<op>.at(out, (i0, i1, ...), vals)`` scatter.

        Each tuple component is an INDIRECT axis (a 2-D index array slice such
        as ``nbr_idx[:, :, n] - 1``) or a STRUCTURED axis (a scalar loop var /
        constant). The value ``vals`` (e.g. ``val[:, jk, :]``) defines the
        broadcast plane; we loop over that plane and, at each point, scalarize
        every index component and the value via :func:`_scalarize_at_iters`
        (Slice axes consume an iter; scalar axes pass through), then accumulate
        ``out[idx0, idx1, ...] op= val`` -- the only sequentially-correct form
        when distinct neighbours hit the same target (duplicate-index sum)."""
        from numpyto_common.lib_nodes import _iter_extent_of, _scalarize_at_iters
        # The iteration plane: the value's broadcast extent (fall back to the
        # first array-valued index component if the value has no slice extent).
        ext = _iter_extent_of(vals, self.shapes)
        if ext is None:
            for comp in idx_tuple.elts:
                ext = _iter_extent_of(comp, self.shapes)
                if ext is not None:
                    break
        if ext is None:
            raise NotImplementedError(
                "multi-index np.<op>.at: cannot determine scatter extent")
        self._n += 1
        iters = [f"__sat{self._n}_{d}" for d in range(len(ext))]
        iter_nodes = [ast.Name(id=i, ctx=ast.Load()) for i in iters]
        idx_scalars = [_scalarize_at_iters(c, iter_nodes, self.shapes)
                       for c in idx_tuple.elts]
        val_s = _scalarize_at_iters(vals, iter_nodes, self.shapes)
        slot = ast.Tuple(elts=idx_scalars, ctx=ast.Load())
        lhs = ast.Subscript(value=ast.Name(id=target.id, ctx=ast.Load()),
                            slice=slot, ctx=ast.Store())
        if op in self._AUG:
            stmt: ast.stmt = ast.AugAssign(target=lhs, op=self._AUG[op](), value=val_s)
        else:                                   # maximum / minimum -> t[i] = fn(t[i], v)
            cur = ast.Subscript(value=ast.Name(id=target.id, ctx=ast.Load()),
                                slice=ast.Tuple(elts=list(idx_scalars), ctx=ast.Load()),
                                ctx=ast.Load())
            stmt = ast.Assign(targets=[lhs], value=ast.Call(
                func=ast.Name(id=self._FOLD[op], ctx=ast.Load()),
                args=[cur, val_s], keywords=[]))
        body: List[ast.stmt] = [stmt]
        for d in reversed(range(len(ext))):
            # ``_iter_extent_of`` already returns each extent as an AST node
            # (a Name like ``nproma`` or a computed length), so it is the
            # loop's ``range`` bound directly.
            body = [ast.For(
                target=ast.Name(id=iters[d], ctx=ast.Store()),
                iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                              args=[copy.deepcopy(ext[d])], keywords=[]),
                body=body, orelse=[])]
        return ast.copy_location(body[0], node)


def _const_or_name_token(tok: str) -> ast.expr:
    """A shape-table token (``"E"`` / ``"12"``) -> a Name or int Constant node."""
    s = str(tok)
    if s.lstrip("-").isdigit():
        return ast.Constant(value=int(s))
    try:
        return ast.parse(s, mode="eval").body
    except SyntaxError:
        return ast.Name(id=s, ctx=ast.Load())


class _NpAliasRewriter(ast.NodeTransformer):
    """Rename ``np.<alias>(...)`` to its canonical ``np.<name>(...)`` form so a
    single lowering path serves every spelling (e.g. ``np.permute_dims`` ->
    ``np.transpose``)."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        f = node.func
        if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and f.value.id in ("np", "numpy") and f.attr in _NP_FUNC_ALIASES):
            f.attr = _NP_FUNC_ALIASES[f.attr]
        return node


class _ConditionalNoneAllocRewriter(ast.NodeTransformer):
    """``X = <expr> if cond else None`` (and the mirror ``X = None if cond else <expr>``)
    -> ``X = <expr>``.

    An array that is a value in one branch and ``None`` in the other is a
    CONDITIONALLY-ALLOCATED buffer (QE vexx_k's ``deexx``; an ML optional
    residual/bias accumulator). The backends have no ``None``, and a *valid* kernel
    only reads ``X`` where it was allocated -- reading it on the ``None`` branch would be a
    ``None``-index error -- so unconditionally taking the allocated branch is sound: the
    extra buffer is written/read only under the same guard, and is otherwise never
    observed. Left untouched when ``X`` is later tested with ``is None`` / ``is not None``
    (there its None-ness is observable, so allocating unconditionally would flip the
    guard); that case is the separate is-None allocation-check handling."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Names whose None-ness is observed (``x is None`` / ``x is not None``): a
        # conditional alloc into one of these must NOT be forced, so record them first.
        checked = set()
        for cmp in ast.walk(node):
            if (isinstance(cmp, ast.Compare) and isinstance(cmp.left, ast.Name)
                    and any(isinstance(op, (ast.Is, ast.IsNot)) for op in cmp.ops)
                    and any(isinstance(c, ast.Constant) and c.value is None for c in cmp.comparators)):
                checked.add(cmp.left.id)
        self._none_checked = checked
        self.generic_visit(node)
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.IfExp)
                and node.targets[0].id not in getattr(self, "_none_checked", set())):
            return node
        ifexp = node.value
        body_none = isinstance(ifexp.body, ast.Constant) and ifexp.body.value is None
        orelse_none = isinstance(ifexp.orelse, ast.Constant) and ifexp.orelse.value is None
        if orelse_none and not body_none:
            node.value = ifexp.body  # X = A if cond else None -> X = A
        elif body_none and not orelse_none:
            node.value = ifexp.orelse  # X = None if cond else A -> X = A
        return node


class _MatmulCallRewriter(ast.NodeTransformer):
    """Normalize ``np.matmul(a, b)`` to the ``a @ b`` BinOp so the call reuses
    the existing matmul machinery (the ``_MatmulHoister`` loop lowering and the
    Fortran ``MATMUL`` / ``DOT_PRODUCT`` emit path) -- no parallel detector."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        f = node.func
        if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and f.value.id in ("np", "numpy") and f.attr == "matmul"
                and len(node.args) == 2 and not node.keywords):
            return ast.copy_location(
                ast.BinOp(left=node.args[0], op=ast.MatMult(), right=node.args[1]), node)
        return node


class _ScalarTimesMatmulRewriter(ast.NodeTransformer):
    """Restructure ``alpha * A @ B`` so the matmul hoister can fire.

    Python parses ``alpha * A @ B`` left-to-right as ``(alpha * A) @ B``;
    the matmul's left operand is a BinOp not a Name, so the hoister
    rejects it. We rewrite the assignment to lift ``alpha * A`` into a
    per-element scaled-copy temp first; the next pass then matmuls the
    temp with B.

    Triggered only when the assignment LHS is a slice / Subscript --
    the typical gemm / k2mm shape. The IR's shape table tells us
    A's shape so we can declare the temp.
    """

    def __init__(self, shape_table: Dict[str, List[str]],
                 temps: Dict[str, Tuple[str, ...]],
                 counter):
        self.shape_table = shape_table
        self.temps = temps
        self.counter = counter
        self.pre_stmts: List[ast.stmt] = []

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        # Pattern: ``(scalar * Name) @ B`` -- left is BinOp(Mult), right is anything.
        if (isinstance(node.op, ast.MatMult)
                and isinstance(node.left, ast.BinOp)
                and isinstance(node.left.op, ast.Mult)):
            inner = node.left
            scaled_name = None
            scalar = None
            if isinstance(inner.left, ast.Name) and inner.left.id in self.shape_table:
                scaled_name = inner.left
                scalar = inner.right
            elif isinstance(inner.right, ast.Name) and inner.right.id in self.shape_table:
                scaled_name = inner.right
                scalar = inner.left
            if scaled_name is not None and scalar is not None:
                shape = self.shape_table.get(scaled_name.id)
                if shape:
                    self.counter[0] += 1
                    temp = f"__sm{self.counter[0]}"
                    self.temps[temp] = tuple(shape)
                    self.shape_table[temp] = shape
                    iters = [f"__si{i}" for i in range(len(shape))]
                    idx = (ast.Name(id=iters[0], ctx=ast.Load()) if len(iters) == 1
                           else ast.Tuple(elts=[ast.Name(id=i, ctx=ast.Load()) for i in iters], ctx=ast.Load()))
                    body = [ast.Assign(
                        targets=[ast.Subscript(value=ast.Name(id=temp, ctx=ast.Load()),
                                               slice=idx, ctx=ast.Store())],
                        value=ast.BinOp(
                            left=scalar,
                            op=ast.Mult(),
                            right=ast.Subscript(value=ast.Name(id=scaled_name.id, ctx=ast.Load()),
                                                slice=idx, ctx=ast.Load())))]
                    out = body
                    for v, b in zip(reversed(iters), reversed(shape)):
                        out = [ast.For(
                            target=ast.Name(id=v, ctx=ast.Store()),
                            iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                                          args=[ast.Name(id=b, ctx=ast.Load())
                                                if not b.isdigit() else
                                                ast.Constant(value=int(b))],
                                          keywords=[]),
                            body=out, orelse=[])]
                    self.pre_stmts.extend(out)
                    # Replace ``alpha * A`` in this MatMult with the temp.
                    node.left = ast.Name(id=temp, ctx=ast.Load())
        return node


class _ArrayIterRewriter(ast.NodeTransformer):
    """Rewrite ``for x in arr:`` into ``for __i in range(N): x = arr[__i]``.

    Numpy / Python permits direct iteration over an array (each iteration
    yields one element of the leading axis). C / Fortran have no
    equivalent, so we desugar to a counted loop with a leading-axis
    subscript. ``arr`` must be a bare Name whose shape is known --
    otherwise the loop falls through unchanged for downstream emit to
    flag.

    Records each loop-var's source array name in :attr:`var_to_array`
    so the emitter can inherit the element dtype (``for b in data:``
    where ``data`` is ``uint8`` should declare ``b`` as ``uint8_t``).
    """

    def __init__(self, shape_table):
        self.shape_table = shape_table
        self._counter = [0]
        #: Mapping from synthesised loop-var name to the source array.
        self.var_to_array: Dict[str, str] = {}

    def visit_For(self, node: ast.For) -> ast.AST:
        self.generic_visit(node)
        if (isinstance(node.iter, ast.Name)
                and isinstance(node.target, ast.Name)):
            shape = self.shape_table.get(node.iter.id)
            if shape:
                self._counter[0] += 1
                iv = f"__ai{self._counter[0]}"
                self.var_to_array[node.target.id] = node.iter.id
                # Build the per-iteration assignment ``x = arr[__i]``.
                preamble = ast.Assign(
                    targets=[ast.Name(id=node.target.id, ctx=ast.Store())],
                    value=ast.Subscript(
                        value=ast.Name(id=node.iter.id, ctx=ast.Load()),
                        slice=ast.Name(id=iv, ctx=ast.Load()),
                        ctx=ast.Load()))
                bound = (ast.Constant(value=int(shape[0]))
                         if shape[0].isdigit() else
                         ast.Name(id=shape[0], ctx=ast.Load()))
                return ast.For(
                    target=ast.Name(id=iv, ctx=ast.Store()),
                    iter=ast.Call(
                        func=ast.Name(id="range", ctx=ast.Load()),
                        args=[bound], keywords=[]),
                    body=[preamble] + node.body,
                    orelse=node.orelse)
        return node


class _EnumerateZipRewriter(ast.NodeTransformer):
    """Desugar ``for x in enumerate(arr):`` and ``for x in zip(a, b):``
    to plain ``for __i in range(N):`` with the per-iteration assignments
    inlined as the first statement of the loop body.
    """

    def __init__(self, shape_table):
        self.shape_table = shape_table

    @staticmethod
    def _enumerate_start(call: ast.Call) -> ast.expr:
        """The ``start=`` of ``enumerate(seq, start=s)`` (positional or kw), else 0."""
        for kw in call.keywords:
            if kw.arg == "start":
                return kw.value
        if len(call.args) >= 2:
            return call.args[1]
        return ast.Constant(value=0)

    def visit_For(self, node: ast.For) -> ast.AST:
        self.generic_visit(node)
        it = node.iter
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name):
            # ``for m, w in enumerate((a, b, c), start=s):`` over a LITERAL/const
            # sequence (the finite-difference-stencil idiom ``enumerate(_CW)``):
            # unroll to straight-line ``m = s+i; w = <elt i>; <body>`` blocks so the
            # element values are compile-time constants (an axis/shift a roll needs).
            if (it.func.id == "enumerate" and it.args
                    and isinstance(it.args[0], (ast.Tuple, ast.List))
                    and isinstance(node.target, ast.Tuple) and len(node.target.elts) == 2):
                idx_name, val_name = node.target.elts[0], node.target.elts[1]
                start = self._enumerate_start(it)
                out: List[ast.stmt] = []
                for i, elt in enumerate(it.args[0].elts):
                    out.append(ast.Assign(targets=[ast.Name(id=idx_name.id, ctx=ast.Store())],
                                          value=ast.BinOp(left=copy.deepcopy(start), op=ast.Add(),
                                                          right=ast.Constant(value=i))))
                    out.append(ast.Assign(targets=[ast.Name(id=val_name.id, ctx=ast.Store())],
                                          value=copy.deepcopy(elt)))
                    out.extend(copy.deepcopy(stmt) for stmt in node.body)
                return out
            if it.func.id == "enumerate" and it.args and isinstance(it.args[0], ast.Name):
                arr = it.args[0]
                shape = self.shape_table.get(arr.id)
                if shape and isinstance(node.target, ast.Tuple) and len(node.target.elts) == 2:
                    idx_name, val_name = node.target.elts[0], node.target.elts[1]
                    start = self._enumerate_start(it)
                    ei = "__ei"
                    # idx = start + __ei ; val = arr[__ei]
                    idx_assign = ast.Assign(
                        targets=[ast.Name(id=idx_name.id, ctx=ast.Store())],
                        value=ast.BinOp(left=copy.deepcopy(start), op=ast.Add(), right=ast.Name(id=ei, ctx=ast.Load())))
                    val_assign = ast.Assign(
                        targets=[ast.Name(id=val_name.id, ctx=ast.Store())],
                        value=ast.Subscript(value=ast.Name(id=arr.id, ctx=ast.Load()),
                                            slice=ast.Name(id=ei, ctx=ast.Load()), ctx=ast.Load()))
                    new_for = ast.For(
                        target=ast.Name(id=ei, ctx=ast.Store()),
                        iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                                      args=[ast.Name(id=shape[0], ctx=ast.Load())], keywords=[]),
                        body=[idx_assign, val_assign] + node.body,
                        orelse=node.orelse)
                    return new_for
            if it.func.id == "zip" and len(it.args) == 2 and all(isinstance(a, ast.Name) for a in it.args):
                a, b = it.args
                shape = self.shape_table.get(a.id)
                if shape and isinstance(node.target, ast.Tuple) and len(node.target.elts) == 2:
                    x_name, y_name = node.target.elts[0], node.target.elts[1]
                    new_for = ast.For(
                        target=ast.Name(id="__zi", ctx=ast.Store()),
                        iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                                      args=[ast.Name(id=shape[0], ctx=ast.Load())], keywords=[]),
                        body=[
                            ast.Assign(
                                targets=[ast.Name(id=x_name.id, ctx=ast.Store())],
                                value=ast.Subscript(value=ast.Name(id=a.id, ctx=ast.Load()),
                                                    slice=ast.Name(id="__zi", ctx=ast.Load()),
                                                    ctx=ast.Load())),
                            ast.Assign(
                                targets=[ast.Name(id=y_name.id, ctx=ast.Store())],
                                value=ast.Subscript(value=ast.Name(id=b.id, ctx=ast.Load()),
                                                    slice=ast.Name(id="__zi", ctx=ast.Load()),
                                                    ctx=ast.Load())),
                        ] + node.body,
                        orelse=node.orelse)
                    return new_for
        return node


class _TransposeRewriter(ast.NodeTransformer):
    """Normalize both transpose spellings to the ``np.transpose(A[, axes])``
    function form so the single ``expand_transpose`` path serves every spelling:

    * the property ``A.T`` -> ``np.transpose(A)``;
    * the method ``A.transpose()`` / ``A.transpose(axes)`` / ``A.transpose(1, 0)``
      -> ``np.transpose(A[, (axes)])`` (the varargs ints are packed into a tuple).
    """

    def __init__(self, sparse_names=None):
        #: Logical sparse matrices whose ``A.T`` / ``A.transpose()`` must stay a
        #: transpose ATTRIBUTE/method -- the sparse matmul hoister turns ``A.T @
        #: x`` into a transpose SpMV on A's own buffers (CSR<->CSC). Densifying it
        #: via np.transpose would index the sparse buffers as a dense 2-D matrix
        #: (wrong + uncompilable).
        self.sparse_names = set(sparse_names or ())

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if node.attr != "T":
            return node
        base = node.value
        # Bare Name -- the original path (a sparse buffer keeps its ``.T`` for the
        # sparse matmul hoister, which lowers ``A.T @ x`` on A's own CSR/CSC buffers).
        if isinstance(base, ast.Name):
            if base.id in self.sparse_names:
                return node
        elif isinstance(base, ast.Subscript):
            # A subscript of a sparse buffer likewise keeps its transpose.
            if isinstance(base.value, ast.Name) and base.value.id in self.sparse_names:
                return node
        elif not isinstance(base, (ast.BinOp, ast.Call)):
            # ``.T`` on a matmul result (``(Yf.T @ Wf).T`` -- the LS3DF generalized
            # Rayleigh-Ritz symmetriser) or another array-valued call. Anything else
            # (a scalar attribute chain) is left intact.
            return node
        # ``<array-expr>.T`` -> ``np.transpose(<array-expr>)``. The call hoister
        # materialises a non-Name argument (the matmul) into a temp before
        # ``expand_transpose`` lowers it, so the transpose never survives as an
        # attribute the per-element scalarizer would misapply.
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="transpose", ctx=ast.Load()),
            args=[base],
            keywords=[])

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "transpose"):
            return node
        if isinstance(f.value, ast.Name) and f.value.id in ("np", "numpy"):
            return node  # already the ``np.transpose(...)`` function form
        if isinstance(f.value, ast.Name) and f.value.id in self.sparse_names:
            return node  # sparse transpose stays a method on its own buffers
        base = f.value
        if len(node.args) == 1 and isinstance(node.args[0], (ast.Tuple, ast.List)):
            args = [base, node.args[0]]            # x.transpose((1, 0))
        elif node.args:
            args = [base, ast.Tuple(elts=list(node.args), ctx=ast.Load())]  # x.transpose(1, 0)
        else:
            args = [base]                          # x.transpose() -- full reverse
        return ast.copy_location(
            ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="transpose", ctx=ast.Load()),
                     args=args, keywords=[]), node)


def _const_int_index(node: ast.AST) -> Optional[int]:
    """Return the integer value of a constant subscript index (``arr[3]`` /
    ``arr[-1]``), or ``None`` for a non-constant one. Numpy spells a negative
    literal index as ``UnaryOp(USub, Constant)``, not a signed ``Constant``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int)):
        return -node.operand.value
    return None


class _ShapeMidExpressionRewriter(ast.NodeTransformer):
    """Replace ``arr.shape[k]`` (and bare ``arr.shape``) anywhere in
    the body with the matching shape symbol from the IR's shape table.

    Legacy OptArena kernels read array extents inline -- e.g.
    ``for i in range(A.shape[0]):`` or ``a = np.zeros(A.shape)`` --
    which the C / Fortran emitter cannot lower directly. Resolve them
    at the AST level to the names declared on the array's shape tuple
    (parsed from ``bench_info.init.shapes`` or recovered via
    ``_shapes_from_initialize``).

    A ``.shape`` / ``.shape[k]`` read whose base is a rank-shifting
    SUBSCRIPT rather than a bare Name (``v[..., None].shape[-1]`` in an
    inlined ``X.reshape(-1, X.shape[-1])``) resolves via
    :func:`_iter_extent_of`, which is Ellipsis / newaxis-aware -- so a
    broadcast subscript's static extent folds the same way a declared
    array's does. The Name-base path is unchanged.
    """

    def __init__(self, arrays_shapes):
        self.arrays_shapes = arrays_shapes

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        # Check the ``arr.shape[k]`` pattern BEFORE descending into
        # the children -- otherwise ``visit_Attribute`` would rewrite
        # the ``arr.shape`` inner node to a Tuple and the pattern
        # match below would miss.
        if (isinstance(node.value, ast.Attribute)
                and node.value.attr == "shape"
                and isinstance(node.value.value, ast.Name)
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, int)):
            shape = self.arrays_shapes.get(node.value.value.id)
            if shape and 0 <= node.slice.value < len(shape):
                return _token_to_ast(shape[node.slice.value])
        # ``<array-expr>.shape[k]`` on a Subscript base (``v[..., None]``):
        # resolve the base's static extent (newaxis / Ellipsis aware) and pick
        # axis ``k`` (negative indices allowed). Unresolvable -> left intact.
        if (isinstance(node.value, ast.Attribute)
                and node.value.attr == "shape"
                and isinstance(node.value.value, ast.Subscript)):
            k = _const_int_index(node.slice)
            if k is not None:
                ext = _iter_extent_of(node.value.value, self.arrays_shapes)
                if ext is not None and -len(ext) <= k < len(ext):
                    return copy.deepcopy(ext[k])
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        # ``len(arr)`` -> the array's FIRST-dim size symbol (numpy ``len`` is
        # ``shape[0]``). C / C++ have no array ``len`` and Fortran's ``len`` is
        # the CHARACTER-length intrinsic, so the literal call fails to compile;
        # the python backends (numba / pythran / jax) run the body verbatim and
        # keep the builtin, so they never reach this native-only rewriter.
        if (isinstance(node.func, ast.Name) and node.func.id == "len" and len(node.args) == 1 and not node.keywords
                and isinstance(node.args[0], ast.Name)):
            shape = self.arrays_shapes.get(node.args[0].id)
            if shape:
                return _token_to_ast(shape[0])
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.value, ast.Name):
            # Bare ``<array-expr>.shape`` on a Subscript base (``Y.shape`` where
            # ``Y`` inlined to ``psi_frag[f]``, or ``v[..., None].shape``) -> the
            # tuple of static extents from :func:`_iter_extent_of`. Downstream
            # tuple-subscript / reshape folding then consumes the literal tuple.
            if node.attr == "shape" and isinstance(node.value, ast.Subscript):
                ext = _iter_extent_of(node.value, self.arrays_shapes)
                if ext is not None:
                    return ast.Tuple(elts=[copy.deepcopy(e) for e in ext], ctx=ast.Load())
            return node
        shape = self.arrays_shapes.get(node.value.id)
        if not shape:
            return node
        if node.attr == "shape":
            # ``_token_to_ast`` (not a bare ``Name(id=token)``) so a COMPOUND shape
            # token -- a slice extent like ``"__inl2_na - 1"`` (LS3DF's Lanczos
            # ``off = betas[:na - 1]``) -- re-parses into a real BinOp rather than a
            # malformed Name whose id is source text (which the int-context / logical
            # analyses then misclassify).
            return ast.Tuple(elts=[_token_to_ast(s) for s in shape], ctx=ast.Load())
        if node.attr == "size":
            # ``arr.size`` -> product of shape symbols (each token re-parsed).
            if len(shape) == 1:
                return _token_to_ast(shape[0])
            expr = _token_to_ast(shape[0])
            for s in shape[1:]:
                expr = ast.BinOp(left=expr, op=ast.Mult(), right=_token_to_ast(s))
            return expr
        if node.attr == "ndim":
            return ast.Constant(value=len(shape))
        # ``arr.dtype`` -- leave intact; downstream emit drops the dtype
        # kwarg via the builtin-cast / math rewriters as appropriate.
        return node


class _BuiltinCastRewriter(ast.NodeTransformer):
    """Drop Python's ``float(x)`` cast on the kernel body.

    In C / Fortran the surrounding operation promotes int -> double
    automatically, so the ``float(x)`` Python uses for division
    semantics is a genuine no-op (and the emitter would otherwise
    produce ``float(x)`` literally, which is not valid C).

    ``int(x)`` is NOT dropped: it is a value-changing TRUNCATION, not a
    no-op. Dropping it relied on the target being int-declared so the
    assignment truncated implicitly -- but that is fragile (``y =
    int(x) + 0.5`` with a double ``y`` would silently keep the fraction)
    and, worse, it erases the barrier that keeps int-ness from
    propagating BACKWARD into a float source: after ``ri = int(rs)``
    became ``ri = rs``, the used-as-int analysis walked from the
    index ``ri`` into ``rs`` and mistyped the whole GROMACS distance
    chain (``rsq``/``rinv``/``dx``) as int, truncating every force to
    zero. Every native emitter already renders a bare ``int(x)`` (C/C++
    ``(int)(x)``, Fortran ``INT(x, kind)``), so leaving it in place is
    both correct and faithful.
    """

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        # True-division barrier: a ``float(x)`` operand of ``/`` must be
        # PRESERVED, not dropped -- numpy ``/`` is true division, so an int/int
        # ``float(a) / b`` must stay floating. Rewrite the ``float(x)`` cast to a
        # real ``np.float64(x)`` cast (which the C / Fortran emitters render)
        # BEFORE ``generic_visit`` reaches the inner ``float`` call and drops it.
        if isinstance(node.op, ast.Div):
            node.left = self._keep_float_as_cast(node.left)
            node.right = self._keep_float_as_cast(node.right)
        self.generic_visit(node)
        return node

    @staticmethod
    def _keep_float_as_cast(operand: ast.AST) -> ast.AST:
        if (isinstance(operand, ast.Call) and isinstance(operand.func, ast.Name)
                and operand.func.id == "float" and len(operand.args) == 1):
            return ast.copy_location(
                ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()),
                                            attr="float64", ctx=ast.Load()),
                         args=list(operand.args), keywords=[]), operand)
        return operand

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (isinstance(node.func, ast.Name)
                and node.func.id == "float"
                and len(node.args) == 1):
            return node.args[0]
        return node


class _ScalarFloatTagger(ast.NodeVisitor):
    """Tag a local scalar float when its DEFINING expression is provably non-integer.

    :func:`_is_integer_expr` reads an untagged non-array Name as INTEGER -- right for the
    symbols (``N`` / ``k`` / ``m``) it exists to classify, wrong for a float scalar, which
    reaches it untagged: ``local_dtypes`` carries the arrays, and a scalar derived in the
    body (``e = 0.5 * (b - a)``) is in no table at all. Reading those as integer makes
    :class:`_TrueDivisionPromoter` fire on a float ``/`` and bake in an fp64 cast.

    Visits in source order so a tag is available to the statements that follow it, and only
    ever ADDS float tags it can prove (an integer expression stays untagged and keeps the
    old reading). So this can only turn a false promotion OFF -- never a new one on."""

    def __init__(self, tags: Dict[str, str], array_names: Set[str]):
        self.tags = tags
        self.array_names = array_names

    def visit_Assign(self, node: ast.Assign) -> None:
        from numpyto_common.lib_nodes import _is_integer_expr
        self.generic_visit(node)
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return
        name = node.targets[0].id
        if name in self.tags or name in self.array_names:
            return
        if not _is_integer_expr(node.value, self.tags, self.array_names):
            self.tags[name] = "float64"


class _TrueDivisionPromoter(ast.NodeTransformer):
    """numpy ``/`` is TRUE division: int / int -> float64. C ``/`` and Fortran
    ``/`` do INTEGER division on integer operands, so wrap the left operand of an
    all-integer division in an ``np.float64(...)`` cast (which both emitters
    render as ``(double)(x)`` / ``REAL(x, kind=c_double)``) to force a floating
    divide -- matching numpy. Float / complex operands are left untouched (the
    surrounding arithmetic already promotes); ``//`` (FloorDiv) is a distinct op
    handled by the emitters' integer floor macro and is NOT touched here.

    ``np.float64`` is deliberate and stays fp64 even on an fp32 emit: numpy's int/int
    IS float64 regardless of the kernel's float precision, so this cast is faithful. It
    is only correct while the operands really are integers, though -- hence the dtype
    table this is handed must be complete (see :class:`_ScalarFloatTagger`). Firing on a
    float divide silently promotes the surrounding expression to double, which fp64
    cannot reveal because there double IS the precision."""

    def __init__(self, local_dtypes, array_names):
        self.local_dtypes = local_dtypes or {}
        self.array_names = array_names or set()

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        from numpyto_common.lib_nodes import _is_integer_expr
        self.generic_visit(node)
        if (isinstance(node.op, ast.Div)
                and _is_integer_expr(node.left, self.local_dtypes, self.array_names)
                and _is_integer_expr(node.right, self.local_dtypes, self.array_names)):
            node.left = ast.copy_location(
                ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()),
                                            attr="float64", ctx=ast.Load()),
                         args=[node.left], keywords=[]), node.left)
        return node


class _MathRewriter(ast.NodeTransformer):
    """Convert ``math.exp(x)`` / ``np.exp(x)`` into ``exp(x)``.

    Shape-aware: when the first arg is a Name that resolves to a
    known array, leaves the call untouched so the LibNode-side
    elementwise expander catches it (which writes a per-element
    loop). Scalar args fall through to the renamed math intrinsic.
    """

    def __init__(self, array_names=None):
        self.array_names = array_names or set()

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and \
                isinstance(node.func.value, ast.Name):
            mod = node.func.value.id
            name = node.func.attr
            new_name = MATH_BUILTINS.get((mod, name))
            if new_name is not None:
                # Skip the rename when the first arg involves an array
                # reference (Name or Subscript with slice) -- the LibNode
                # expander emits the per-element form. Pure-scalar
                # arguments fall through to the math intrinsic rename.
                if node.args and self._refers_to_array(node.args[0]):
                    return node
                node.func = ast.Name(id=new_name, ctx=ast.Load())
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """Lower ``np.pi`` / ``np.e`` to a numeric literal and ``np.inf``
        / ``np.nan`` to the C99 constants ``INFINITY`` / ``NAN``.

        ``pi`` / ``e`` are finite mathematical constants, so we resolve
        them to their value via :mod:`sympy` (a single source of truth)
        rather than emitting a C-only ``M_PI`` / ``M_E`` macro -- the
        plain literal renders uniformly in EVERY backend (C, C++,
        Fortran, ...), each adding its own kind suffix, with no
        per-language constant table or Fortran ``acos(-1)`` substitute.
        ``inf`` / ``nan`` have no finite literal form, so they keep the
        ``<math.h>`` constants (also valid in C++).
        """
        self.generic_visit(node)
        if (isinstance(node.value, ast.Name) and node.value.id == "np"):
            const_values = {"pi": sympy.pi, "e": sympy.E}
            if node.attr in const_values:
                return ast.Constant(value=float(const_values[node.attr]))
            mapping = {
                "inf": "INFINITY",
                "nan": "NAN",
                "newaxis": None,  # handled by _NewaxisToNone
            }
            replacement = mapping.get(node.attr)
            if replacement is not None:
                return ast.Name(id=replacement, ctx=ast.Load())
        return node

    def _refers_to_array(self, expr: ast.expr) -> bool:
        """Return True if the expression reads any declared array as
        an array value (i.e. without a complete scalar subscript).

        ``X`` (Name of an array) -> True.
        ``X - tmp_max`` (BinOp with an array Name child) -> True.
        ``X[i, j]`` (Subscript with all-scalar index) -> False (it's a
        scalar element).
        ``X[1:N-1]`` (Subscript with at least one Slice axis) -> True.
        """
        if isinstance(expr, ast.Name):
            return expr.id in self.array_names
        if isinstance(expr, ast.Subscript):
            # Scalar Subscript -- the result is a scalar, regardless
            # of whether the base is an array.
            slc = expr.slice
            if isinstance(slc, ast.Slice):
                return True
            if isinstance(slc, ast.Tuple):
                if any(isinstance(e, ast.Slice) for e in slc.elts):
                    return True
                return False
            return False
        if isinstance(expr, (ast.BinOp, ast.UnaryOp)):
            children = ([expr.left, expr.right] if isinstance(expr, ast.BinOp)
                        else [expr.operand])
            return any(self._refers_to_array(c) for c in children)
        if isinstance(expr, ast.Call):
            return any(self._refers_to_array(a) for a in expr.args)
        if isinstance(expr, ast.IfExp):
            return any(self._refers_to_array(c) for c in (expr.test, expr.body, expr.orelse))
        return False


_NP_ELEMENTWISE: Set[str] = {
    "maximum", "minimum", "add", "subtract", "multiply", "divide",
    "power", "mod", "floor_divide", "true_divide",
    "exp", "log", "sqrt", "sin", "cos", "tan", "tanh",
    "abs", "absolute", "negative", "positive",
    "less", "less_equal", "greater", "greater_equal",
    "equal", "not_equal", "logical_and", "logical_or", "logical_not",
}


def _resolve_shape_token(node: ast.AST,
                         shape_table: Dict[str, Tuple[str, ...]]) -> str:
    """Stringify a shape-tuple element, resolving ``arr.shape[i]``
    references against the known shape of ``arr``.

    * ``arr.shape[i]`` -> the ``i``-th token of ``shape_table[arr]``
      (constant ``i``) so a helper-inlined ``np.empty([x.shape[0],
      x.shape[1] // 2, ...])`` becomes a concrete shape tuple.
    * ``arr.shape[i] // K`` -> the resolved token wrapped in the
      ``//`` BinOp (still printable, still a valid C extent).
    * Anything else -> ``ast.unparse(node)`` (existing behaviour).
    """
    resolved = _resolve_arr_shape_subscript(node, shape_table)
    if resolved is not None:
        return resolved
    if isinstance(node, ast.BinOp):
        left = _resolve_shape_token(node.left, shape_table)
        right = _resolve_shape_token(node.right, shape_table)
        op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*",
              ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%"}.get(
                  type(node.op))
        if op is not None:
            return f"({left} {op} {right})"
    return ast.unparse(node)


def _resolve_arr_shape_subscript(node: ast.AST,
                                 shape_table: Dict[str, Tuple[str, ...]]
                                 ) -> Optional[str]:
    """Return the resolved shape token for ``arr.shape[i]``, or None
    if the form does not match or the source array is unknown."""
    if not (isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "shape"
            and isinstance(node.value.value, ast.Name)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)):
        return None
    src = shape_table.get(node.value.value.id)
    if src is None or node.slice.value >= len(src):
        return None
    return src[node.slice.value]


def _ssa_rename_reassigned(tree: ast.AST,
                            arrays_shapes: Dict[str, List[str]]) -> None:
    """SSA-style rename for Names reassigned with different broadcast
    extents.

    Walks every function body's statement list in source order. For
    each ``Name = expr`` whose RHS has a derivable iteration extent,
    tracks the current active extent per Name. When a reassignment
    yields a different extent, mints ``<name>__v<n>`` and rewrites
    forward Load-context references to ``<name>`` to the new version
    until the next reassignment.

    The first occurrence keeps the original name. Recurses into
    ``For`` / ``If`` / ``While`` bodies but treats each as an
    independent scope -- nested writes do not poison the outer
    version map (the outer scope's name stays bound to its outer
    extent across the nested block).

    Unblocks the canonical hdiff / vadv kernels where ``res`` is
    reassigned twice with different shapes; without renaming the
    ``_LiftFreshArrayFromSlices`` lifter bails on the shape mismatch.
    """
    from numpyto_common.lib_nodes import _iter_extent_of

    shapes: Dict[str, Tuple[str, ...]] = {
        name: tuple(shape) for name, shape in arrays_shapes.items()
    }

    def _maybe_register_alloc(target_id: str, rhs: ast.AST) -> None:
        """Register the shape of ``np.zeros((...))`` / ``np.empty((...))``
        style allocators so subsequent reads see the allocated extent
        when the SSA pass computes broadcast extents inside loop
        bodies. Conservative -- only handles the Tuple-shape form."""
        if not isinstance(rhs, ast.Call):
            return
        func = rhs.func
        attr = None
        if isinstance(func, ast.Attribute):
            attr = func.attr
        elif isinstance(func, ast.Name):
            attr = func.id
        if attr not in {"zeros", "empty", "ones", "ndarray", "zeros_like",
                         "empty_like", "ones_like"}:
            return
        if attr.endswith("_like") and rhs.args and isinstance(rhs.args[0], ast.Name):
            src = shapes.get(rhs.args[0].id)
            if src:
                shapes[target_id] = src
            return
        if not rhs.args:
            return
        sh = rhs.args[0]
        if isinstance(sh, (ast.Tuple, ast.List)):
            shapes[target_id] = tuple(ast.unparse(e) for e in sh.elts)
        elif isinstance(sh, ast.Constant) and isinstance(sh.value, int):
            shapes[target_id] = (str(sh.value),)
        elif isinstance(sh, ast.Name):
            shapes[target_id] = (sh.id,)

    def _apply_renames(node: ast.AST, rename_map: Dict[str, str]) -> None:
        if not rename_map:
            return
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Name)
                    and isinstance(sub.ctx, ast.Load)
                    and sub.id in rename_map):
                sub.id = rename_map[sub.id]

    def _walk(stmts: List[ast.stmt],
              rename_map: Dict[str, str],
              last_shape: Dict[str, Tuple[str, ...]],
              version: Dict[str, int]) -> None:
        # Single function-scope rename_map / shape map -- Python does
        # not have block scope for assignments, so a ``bcol = ...``
        # inside sibling for-loops at function scope is the SAME local
        # being reassigned. Sharing the state across nested scopes lets
        # the pre-pass mint a fresh version for each shape change even
        # when the assignments live in different loop bodies.
        for stmt in stmts:
            # Rewrite Load-context Names on the RHS / iter / test BEFORE
            # the version-mint decision (the assignment's RHS reads the
            # old version's storage).
            if isinstance(stmt, (ast.Assign, ast.AugAssign)):
                _apply_renames(stmt.value, rename_map)
                if isinstance(stmt, ast.AugAssign):
                    _apply_renames(stmt.target, rename_map)
            elif isinstance(stmt, (ast.If, ast.While)):
                _apply_renames(stmt.test, rename_map)
            elif isinstance(stmt, ast.For):
                _apply_renames(stmt.iter, rename_map)
            elif isinstance(stmt, ast.Expr):
                _apply_renames(stmt.value, rename_map)
            elif isinstance(stmt, ast.Return):
                if stmt.value is not None:
                    _apply_renames(stmt.value, rename_map)

            # Decide on rename for a plain ``Name = expr`` LHS.
            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)):
                orig = stmt.targets[0].id
                # Allocator-style RHS -- register the allocated shape
                # so later reads inside loop bodies resolve their
                # extents. Allocations themselves don't trigger a rename.
                _maybe_register_alloc(orig, stmt.value)
                ext = _iter_extent_of(stmt.value, shapes)
                if ext is not None:
                    shape_toks = tuple(ast.unparse(e) for e in ext)
                    # ``version[orig]`` is a dict ``{shape_tuple ->
                    # active_name}``. Each distinct shape gets its own
                    # active name; once minted, every later assignment
                    # with that shape reuses it so sibling-loop
                    # reassignments stay on the same buffer.
                    if not isinstance(version.get(orig), dict):
                        version[orig] = {}
                    name_for_shape = version[orig].get(shape_toks)
                    if name_for_shape is None:
                        if not version[orig]:
                            # First occurrence -- keep the original name.
                            name_for_shape = orig
                        else:
                            n = len(version[orig])
                            name_for_shape = f"{orig}__v{n}"
                        version[orig][shape_toks] = name_for_shape
                    if name_for_shape != orig:
                        stmt.targets[0].id = name_for_shape
                        rename_map[orig] = name_for_shape
                    else:
                        rename_map.pop(orig, None)
                    last_shape[orig] = shape_toks
                    shapes[name_for_shape] = shape_toks
            # Recurse into nested control flow with a fresh scope so
            # inner reassignments don't leak the rename outward. Use
            # the outer ``shapes`` so the inner scope sees the current
            # extent table.
            if isinstance(stmt, ast.For):
                _walk(stmt.body, rename_map, last_shape, version)
                _walk(stmt.orelse, rename_map, last_shape, version)
            elif isinstance(stmt, ast.If):
                _walk(stmt.body, rename_map, last_shape, version)
                _walk(stmt.orelse, rename_map, last_shape, version)
            elif isinstance(stmt, ast.While):
                _walk(stmt.body, rename_map, last_shape, version)
                _walk(stmt.orelse, rename_map, last_shape, version)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            _walk(node.body, {}, {}, {})


def _ctor_shape_arg(call: ast.Call) -> Optional[ast.expr]:
    """Return the shape argument of an array constructor call
    (``np.zeros/empty/ones/ndarray(...)``): the first positional arg, or the
    ``shape=`` keyword when there is none (``np.ndarray(shape=(nlev, klon))`` --
    cloudsc's local declarations)."""
    if call.args:
        return call.args[0]
    for kw in call.keywords:
        if kw.arg == "shape":
            return kw.value
    return None


def _harvest_local_shapes(tree: ast.AST,
                          shape_table: Dict[str, Tuple[str, ...]],
                          dtype_table: Optional[Dict[str, str]] = None) -> None:
    """Pre-scan the body for ``name = np.<alloc>(...)`` and seed the
    shape table with the inferred output shapes.

    Recognises ``zeros / empty / ones / zeros_like / empty_like`` and
    ``np.copy / transpose / triu / flip / linalg.cholesky / outer`` --
    any registered allocator-style call that has a deterministic
    output shape from its args. Run before LibNodeRewriter so the
    downstream call-hoister / scalarizer see ``Q``'s shape when
    visiting ``Q[:, k]``.

    Also populates ``dtype_table`` (if provided) with the dtype hint
    from the constructor's ``dtype=`` kwarg, so the emitter can
    declare ``X = np.zeros((N,), dtype=np.complex128)`` as
    ``double _Complex X[N]``.
    """
    from numpyto_common.lib_nodes import NP_ZEROS_ALIASES, _iter_extent_of
    from numpyto_common.frontend import _dtype_from_constructor
    for stmt in ast.walk(tree):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            continue
        rhs = stmt.value
        # ``X = np.zeros(.., dtype=..) if cond else None`` -- vexx_k's ``deexx``,
        # allocated only in the ultrasoft / PAW branch. The shape + dtype live in
        # the constructor branch of the ternary; unwrap to it so the local is
        # typed / sized like a direct ``X = np.zeros(..)``. Without this the local
        # falls through untyped and defaults to real though it is np.complex128 --
        # C narrows the complex accumulation silently (safe only while the branch
        # is dead), but C++ rejects the complex->real assignment at compile.
        if isinstance(rhs, ast.IfExp):
            ctor = [b for b in (rhs.body, rhs.orelse) if isinstance(b, ast.Call)]
            none_br = [b for b in (rhs.body, rhs.orelse)
                       if isinstance(b, ast.Constant) and b.value is None]
            if len(ctor) == 1 and len(none_br) == 1:
                rhs = ctor[0]
        # Name = Name alias -- inherit shape and dtype from the source.
        if isinstance(rhs, ast.Name):
            src_shape = shape_table.get(rhs.id)
            if src_shape and target.id not in shape_table:
                shape_table[target.id] = tuple(src_shape)
            if dtype_table is not None:
                src_dt = dtype_table.get(rhs.id)
                if src_dt is not None and target.id not in dtype_table:
                    dtype_table[target.id] = src_dt
            continue
        # ``np.linalg.<op>`` is a TWO-level attribute, so the single-level
        # ``np.<attr>`` gate below never matches it and the last-ditch extent
        # guess mirrors the FIRST operand instead -- sizing ``x = np.linalg.
        # solve(A, b)`` like the SQUARE A rather than like b. A 1-D b then has
        # its reads padded to a phantom second dim (``x[i]`` -> ``x[i, :]``).
        # Register what the solve / inv / cholesky expanders actually write.
        _lin_op = _np_linalg_attr(rhs)
        if _lin_op in ("solve", "inv", "cholesky"):
            # ``solve`` returns x with b's shape; ``inv`` / ``cholesky`` are
            # shape-preserving in their single operand.
            _src_arg = rhs.args[1] if _lin_op == "solve" and len(rhs.args) >= 2 else (rhs.args[0]
                                                                                      if rhs.args else None)
            if isinstance(_src_arg, ast.Name):
                _src_shape = shape_table.get(_src_arg.id)
                if _src_shape:
                    shape_table[target.id] = tuple(_src_shape)
            continue
        if not (isinstance(rhs, ast.Call)
                and isinstance(rhs.func, ast.Attribute)
                and isinstance(rhs.func.value, ast.Name)
                and rhs.func.value.id == "np"):
            # Last-ditch: a BinOp / UnaryOp / Compare / BoolOp / Subscript
            # whose operands have known shapes -- mirror the (broadcast /
            # slice / gather) extent. Lets the harvest see ``x = a + b``, a
            # boolean mask ``in_range = (rsq < c) & (rsq > 0)`` (force_lj),
            # or a slice/gather local ``nb = neigh[:, j]`` (cfd / lavamd
            # unstructured-grid neighbor gather) as a new shape entry so the
            # downstream slice-fusion / boolean-mask rewriter / gather
            # scalarizer can resolve ``arr[nb]`` to ``arr[nb[i]]`` instead of
            # treating the index array ``nb`` as a bare scalar.
            # ``ast.Call`` covers a method-form shape op the pre-normalise harvest
            # sees before it becomes ``np.<fn>`` -- notably ``X = (Yf @ C).reshape(
            # shp)`` (LS3DF Rayleigh-Ritz), whose target must be sized here or every
            # downstream local derived from ``X`` inherits a wrong extent.
            if isinstance(rhs, (ast.BinOp, ast.UnaryOp, ast.Compare,
                                ast.BoolOp, ast.Subscript, ast.Call)) \
                    and target.id not in shape_table:
                from numpyto_common.lib_nodes import extent_is_scalar
                ext = _iter_extent_of(rhs, shape_table)
                # An all-size-1 broadcast (``t = a[i] > x`` with ``x`` shape ``(1,)``) is a SCALAR local:
                # numpyto reads size-1 arrays as ``x[0]``, so sizing ``t`` as ``T t[1]`` would desync its
                # scalar declaration from the array-style writes the extent drives (see extent_is_scalar).
                if ext is not None and not extent_is_scalar(ext):
                    shape_table[target.id] = tuple(
                        ast.unparse(e) for e in ext)
            continue
        if dtype_table is not None:
            dt = _dtype_from_constructor(rhs)
            if dt is not None:
                dtype_table[target.id] = dt
        attr = rhs.func.attr
        if attr in NP_ZEROS_ALIASES:
            # ``np.zeros_like(other)`` -> other's shape.
            if attr.endswith("_like") and rhs.args and isinstance(rhs.args[0], ast.Name):
                src_shape = shape_table.get(rhs.args[0].id)
                if src_shape:
                    shape_table[target.id] = tuple(src_shape)
                continue
            # ``np.zeros((N, M))`` / ``np.ndarray(shape=(N, M))`` -- the shape
            # is the first positional arg OR the ``shape=`` keyword (cloudsc's
            # ``ztp1 = np.ndarray(shape=(nlev, klon))`` locals).
            shape_arg = _ctor_shape_arg(rhs)
            if shape_arg is not None:
                if isinstance(shape_arg, (ast.Tuple, ast.List)):
                    parts = [_resolve_shape_token(e, shape_table)
                             for e in shape_arg.elts]
                    shape_table[target.id] = tuple(parts)
                elif isinstance(shape_arg, ast.Name):
                    shape_table[target.id] = (shape_arg.id,)
                elif isinstance(shape_arg, ast.Constant) and isinstance(shape_arg.value, int):
                    shape_table[target.id] = (str(shape_arg.value),)
                elif isinstance(shape_arg, ast.Attribute) and shape_arg.attr == "shape":
                    # ``np.zeros(x.shape, ...)`` -- mirror x's shape.
                    if isinstance(shape_arg.value, ast.Name):
                        src = shape_table.get(shape_arg.value.id)
                        if src is not None:
                            shape_table[target.id] = tuple(src)
                elif isinstance(shape_arg, (ast.Subscript, ast.BinOp)):
                    # A scalar ``arr.shape[i]`` (or arithmetic over it) 1-D extent --
                    # ``np.zeros(M.shape[0], np.float64)``, the eigh eigenvalue vector
                    # over a LOCAL operand. Register it (resolving the token the same
                    # way a shape-TUPLE element is) so the 1-D temp lands in the table
                    # like its 2-D siblings instead of being dropped; without it the
                    # ``w = __eigh0_wa`` alias never learns ``w`` is an array.
                    shape_table[target.id] = (_resolve_shape_token(shape_arg, shape_table),)
        # ``np.eye(M)`` -> ``(M, M)``; ``np.eye(M, N)`` -> ``(M, N)``.
        elif attr == "eye" and rhs.args:
            first = rhs.args[0]
            first_tok = (str(first.value) if isinstance(first, ast.Constant)
                         and isinstance(first.value, int) else
                         first.id if isinstance(first, ast.Name) else
                         ast.unparse(first))
            second_tok = first_tok
            if len(rhs.args) >= 2:
                second = rhs.args[1]
                second_tok = (str(second.value) if isinstance(second, ast.Constant)
                              and isinstance(second.value, int) else
                              second.id if isinstance(second, ast.Name) else
                              ast.unparse(second))
            shape_table[target.id] = (first_tok, second_tok)
        # ``np.linspace(start, stop, n)`` -> ``(n,)``. The third
        # positional arg is the sample count; numpy default 50 if
        # omitted but the in-tree expander rejects that.
        elif attr == "linspace" and len(rhs.args) >= 3:
            count = rhs.args[2]
            tok = (str(count.value) if isinstance(count, ast.Constant)
                   and isinstance(count.value, int) else
                   count.id if isinstance(count, ast.Name) else
                   ast.unparse(count))
            shape_table[target.id] = (tok,)
        # ``np.arange(stop)`` -> ``(stop,)``; ``np.arange(start, stop)`` ->
        # ``(stop - start,)``.
        elif attr == "arange" and rhs.args:
            if len(rhs.args) == 1:
                stop = rhs.args[0]
                tok = (str(stop.value) if isinstance(stop, ast.Constant)
                       and isinstance(stop.value, int) else
                       stop.id if isinstance(stop, ast.Name) else
                       ast.unparse(stop))
                shape_table[target.id] = (tok,)
        # ``np.identity(n)`` -> ``(n, n)``.
        elif attr == "identity" and rhs.args:
            first = rhs.args[0]
            tok = (str(first.value) if isinstance(first, ast.Constant)
                   and isinstance(first.value, int) else
                   first.id if isinstance(first, ast.Name) else
                   ast.unparse(first))
            shape_table[target.id] = (tok, tok)
        # Elementwise broadcast ops: np.maximum / minimum / add / etc.
        # The result shape is the broadcast of the args' shapes; defer
        # to _iter_extent_of which already knows the rules.
        elif attr in _NP_ELEMENTWISE and rhs.args:
            ext = _iter_extent_of(rhs.args[0], shape_table)
            for arg in rhs.args[1:]:
                a_ext = _iter_extent_of(arg, shape_table)
                if a_ext is not None and ext is not None:
                    from numpyto_common.lib_nodes import _broadcast_extents
                    ext = _broadcast_extents(ext, a_ext)
                elif a_ext is not None:
                    ext = a_ext
            if ext is not None:
                shape_table[target.id] = tuple(ast.unparse(e) for e in ext)
        # ``np.copy(other)`` (function form) / np.transpose / np.triu /
        # np.flip / np.asarray / np.ascontiguousarray / np.linalg.* -> share shape.
        elif attr in {"copy", "asarray", "ascontiguousarray", "triu", "flip"} and rhs.args and isinstance(
                rhs.args[0], ast.Name):
            src_shape = shape_table.get(rhs.args[0].id)
            if src_shape:
                shape_table[target.id] = tuple(src_shape)
        elif attr == "transpose" and rhs.args and isinstance(rhs.args[0], ast.Name):
            src_shape = shape_table.get(rhs.args[0].id)
            if src_shape:
                if len(rhs.args) >= 2 and isinstance(rhs.args[1], ast.Tuple):
                    perm = [e.value for e in rhs.args[1].elts
                            if isinstance(e, ast.Constant) and isinstance(e.value, int)]
                    if len(perm) == len(src_shape):
                        shape_table[target.id] = tuple(src_shape[p] for p in perm)
                else:
                    shape_table[target.id] = tuple(reversed(src_shape))
        # Fallback for any other ``np.<func>(...)`` whose result shape
        # ``_iter_extent_of`` can derive: axis-aware reductions
        # (``rsq = np.sum(dpos * dpos, axis=2)`` -> ``(N, N)``) and elementwise
        # math wrapping one (gem's ``r = np.sqrt(np.sum(d * d, axis=2))``).
        # Registering the shape lets a downstream local (``r2inv =
        # np.zeros_like(rsq)``), the boolean-mask rewriter, and the array
        # declaration all resolve the reduction chain.
        elif target.id not in shape_table:
            ext = _iter_extent_of(rhs, shape_table)
            if ext is not None:
                shape_table[target.id] = tuple(ast.unparse(e) for e in ext)


class _FullLikeRewriter(ast.NodeTransformer):
    """``X = np.full_like(src, val)`` -> ``X = np.empty_like(src); X[:] = val`` and
    ``X = np.full(shape, val)`` -> ``X = np.empty(shape); X[:] = val``.

    The existing empty-alias shape harvest declares X (shape from src / the shape
    arg) and the whole-array scalar-broadcast assign fills it -- so no dedicated
    full/full_like emitter path is needed (lulesh ``pbvc = np.full_like(bvc, c1s)``)."""

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        v = node.value
        if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(v, ast.Call)
                and isinstance(v.func, ast.Attribute) and v.func.attr in ("full_like", "full")
                and isinstance(v.func.value, ast.Name) and v.func.value.id in ("np", "numpy") and len(v.args) >= 2):
            return node
        tgt = node.targets[0]
        alloc_attr = "empty_like" if v.func.attr == "full_like" else "empty"
        dtype_kw = [kw for kw in v.keywords if kw.arg == "dtype"]
        alloc = ast.Assign(
            targets=[ast.Name(id=tgt.id, ctx=ast.Store())],
            value=ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=alloc_attr, ctx=ast.Load()),
                           args=[v.args[0]], keywords=dtype_kw))
        fill = ast.Assign(
            targets=[ast.Subscript(value=ast.Name(id=tgt.id, ctx=ast.Load()),
                                   slice=ast.Slice(lower=None, upper=None, step=None), ctx=ast.Store())],
            value=v.args[1])
        for s in (alloc, fill):
            ast.copy_location(s, node)
        ast.fix_missing_locations(alloc)
        ast.fix_missing_locations(fill)
        return [alloc, fill]


class _EyeCallHoister(_StmtHoister):
    """Materialise a nested ``np.eye(...)`` / ``np.identity(...)`` call into its own
    ``__eye<k> = <call>`` statement, so the direct-assign :class:`_EyeToZerosDiagonal`
    can lower it to a zeros + diagonal fill.

    LS3DF's generalized Rayleigh-Ritz adds an identity jitter inline --
    ``s_sub = 0.5 * (...) + 1.0e-12 * np.eye(k)`` -- where ``np.eye`` is buried in an
    expression, not a standalone assignment. A call that is already the direct RHS of
    an assignment is left in place for the diagonal rewriter to consume."""

    @staticmethod
    def _is_eye_call(v: ast.AST) -> bool:
        return (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)
                and v.func.attr in ("eye", "identity") and isinstance(v.func.value, ast.Name)
                and v.func.value.id in ("np", "numpy") and bool(v.args))

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if self._is_eye_call(node):
            return self._spill(node, "__eye")
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and self._is_eye_call(node.value)):
            return node
        return self._flush(node)


class _EyeToZerosDiagonal(ast.NodeTransformer):
    """``X = np.eye(n)`` / ``np.eye(m, n)`` / ``np.identity(n)`` -> a zeros
    allocation plus an explicit diagonal fill::

        X = np.zeros((n, n))            # or (m, n)
        for __eye<k> in range(n):       # range(min(m, n)) when rectangular
            X[__eye<k>, __eye<k>] = 1.0

    Built from primitives every backend already lowers (``np.zeros`` + a loop +
    a scalar store), so no per-emitter identity path is needed. The zeros
    harvest then declares X and picks up the ``(n, n)`` shape as usual. Native
    lowering only -- the python backends keep the builtin ``np.eye``.
    """

    def __init__(self):
        self._n = 0

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        v = node.value
        if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(v, ast.Call)
                and isinstance(v.func, ast.Attribute) and v.func.attr in ("eye", "identity")
                and isinstance(v.func.value, ast.Name) and v.func.value.id in ("np", "numpy") and v.args):
            return node
        tgt = node.targets[0].id
        rows = v.args[0]
        # ``eye(m, n)`` with a real second extent is rectangular (diagonal =
        # min(m, n)); ``eye(n)`` / ``identity(n)`` (or ``eye(n, None)``) is square.
        rectangular = (v.func.attr == "eye" and len(v.args) >= 2
                       and not (isinstance(v.args[1], ast.Constant) and v.args[1].value is None))
        if rectangular:
            cols = v.args[1]
            diag = ast.Call(func=ast.Name(id="min", ctx=ast.Load()),
                            args=[copy.deepcopy(rows), copy.deepcopy(cols)], keywords=[])
        else:
            cols = copy.deepcopy(rows)
            diag = copy.deepcopy(rows)
        dtype_kw = [kw for kw in v.keywords if kw.arg == "dtype"]
        zeros = ast.Assign(
            targets=[ast.Name(id=tgt, ctx=ast.Store())],
            value=ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="zeros", ctx=ast.Load()),
                           args=[ast.Tuple(elts=[copy.deepcopy(rows), cols], ctx=ast.Load())], keywords=dtype_kw))
        it = f"__diag{self._n}"
        self._n += 1
        loop = ast.For(
            target=ast.Name(id=it, ctx=ast.Store()),
            iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[diag], keywords=[]),
            body=[ast.Assign(
                targets=[ast.Subscript(
                    value=ast.Name(id=tgt, ctx=ast.Load()),
                    slice=ast.Tuple(elts=[ast.Name(id=it, ctx=ast.Load()),
                                          ast.Name(id=it, ctx=ast.Load())], ctx=ast.Load()),
                    ctx=ast.Store())],
                value=ast.Constant(value=1.0))],
            orelse=[])
        for s in (zeros, loop):
            ast.copy_location(s, node)
        ast.fix_missing_locations(zeros)
        ast.fix_missing_locations(loop)
        return [zeros, loop]


class _ZerosRewriter(ast.NodeTransformer):
    """Turn ``x = np.zeros((N, K))`` (and family) into a side-table entry.

    Recognises every member of :data:`numpyto_common.lib_nodes.NP_ZEROS_ALIASES`
    (``np.zeros`` / ``np.empty`` / ``np.ones`` / ``np.zeros_like`` /
    ``np.empty_like``). For the ``_like`` forms the LHS shape comes
    from the named array (looked up in :attr:`shape_table`) instead
    of from the call's explicit shape argument.
    """

    def __init__(self, shape_table: Optional[Dict[str, Tuple[str, ...]]] = None):
        from numpyto_common.lib_nodes import NP_ZEROS_ALIASES
        self.zeros: Dict[str, Tuple[str, ...]] = {}
        # Fill kind per harvested local, keyed by name: the constructor
        # attr (``zeros`` / ``ones`` / ``empty`` / ``zeros_like`` / ...).
        # Lets the emitter pick the right initialiser when a constructor
        # aliases an OUTPUT parameter (zeros -> memset 0, ones -> fill 1,
        # empty -> nothing) instead of declaring a shadowing local.
        self.fills: Dict[str, str] = {}
        self.aliases = set(NP_ZEROS_ALIASES)
        self.shape_table = shape_table or {}

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        if (len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "np"
                and node.value.func.attr in self.aliases):
            name = node.targets[0].id
            attr = node.value.func.attr
            shape: Optional[Tuple[str, ...]] = None
            if attr.endswith("_like"):
                # ``np.zeros_like(a)`` -> share ``a``'s shape.
                if node.value.args and isinstance(node.value.args[0], ast.Name):
                    other = node.value.args[0].id
                    shape = self.shape_table.get(other)
            else:
                shape_arg = _ctor_shape_arg(node.value)
                shape = _shape_from_ast(shape_arg, self.shape_table)
            if shape is not None:
                self.zeros[name] = shape
                self.fills[name] = attr
                # Replace the call with a marker the emitter recognises.
                node.value = ast.Call(
                    func=ast.Name(id="__optarena_zeros__", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                )
        return node


def _shape_from_ast(node, shape_table=None) -> Tuple[str, ...]:
    """Return the source-level shape ``(N, K)`` from an AST tuple / list / int.

    ``np.empty([x.shape[0], x.shape[1] // 2, ...])`` resolves the per-axis
    ``arr.shape[i]`` references against the optional ``shape_table``.
    ``np.zeros(C.shape, ...)`` mirrors C's shape from the table.
    """
    if node is None:
        return ()
    if isinstance(node, (ast.Tuple, ast.List)):
        if shape_table is not None:
            return tuple(_resolve_shape_token(e, shape_table)
                         for e in node.elts)
        return tuple(ast.unparse(e) for e in node.elts)
    # ``np.zeros(C.shape, ...)`` -- single-arg whole-shape mirror.
    if (isinstance(node, ast.Attribute) and node.attr == "shape"
            and isinstance(node.value, ast.Name)
            and shape_table is not None):
        src = shape_table.get(node.value.id)
        if src is not None:
            return tuple(src)
    # A single scalar shape arg (``np.zeros(M.shape[0], np.float64)`` -- the eigh
    # eigenvalue vector's 1-D extent over a LOCAL operand): resolve an
    # ``arr.shape[i]`` token the same way a shape-TUPLE element already is, so the
    # local's ``.shape[0]`` folds to its dimension symbol instead of surviving as an
    # unlowerable ``M.shape[0]`` malloc / allocate extent.
    if shape_table is not None:
        return (_resolve_shape_token(node, shape_table),)
    return (ast.unparse(node),)


#: When a slice's ``start`` or ``stop`` is omitted in numpy
#: (``A[:K]`` / ``A[K:]`` / ``A[:]``), we substitute the array's
#: declared length symbol. These are the per-axis defaults applied
#: by :class:`SliceFusion` when it can resolve the array shape from
#: the IR.
_DEFAULT_SLICE_START = "0"


def _slice_dims(node: ast.Subscript) -> List[ast.AST]:
    """Return per-axis slice entries (either ``Slice`` or non-slice index)."""
    sl = node.slice
    if isinstance(sl, ast.Tuple):
        return list(sl.elts)
    return [sl]


def _has_any_slice(node: ast.AST) -> bool:
    """``True`` iff ``node`` is a subscript whose dim list contains a ``Slice``."""
    if not isinstance(node, ast.Subscript):
        return False
    return any(isinstance(d, ast.Slice) for d in _slice_dims(node))


def _is_full_slice(node: ast.AST) -> bool:
    """``True`` for a bare ``:`` slice (no lower / upper / step)."""
    return (isinstance(node, ast.Slice) and node.lower is None
            and node.upper is None and node.step is None)


class _CollapseChainedSubscripts(ast.NodeTransformer):
    """Collapse a chained subscript ``A[i][j]`` into a single ``A[i, j]``.

    Helper inlining leaves chained indexing where the source sliced a view then
    indexed it -- vexx_k's ``tabxx_qr[ia][:, ijtoh[ih, jh]]`` (a real-space
    Q-table column) and ``becxx[:, jbnd, ikq][ikb]`` (a beta-projection row).
    numpy basic indexing associates: ``A[i][rest] == A[i, rest]`` for a scalar
    ``i``, and an index applied to a full-slice axis selects that axis
    (``A[:, j, k][m] == A[m, j, k]``). Flattening to a SINGLE subscript up front
    lets every downstream shape harvest / scalarizer / scatter path treat the
    access uniformly, instead of mis-mapping a loop iterator onto the inner ``:``
    (which corrupts the fancy-scatter store and the dot-product operand).

    Conservative: only collapses when the base is a known-shape array, every inner
    index is a scalar or a FULL ``:`` slice, and the outer indices fit the
    surviving (slice + trailing) axes -- any partial slice, strided slice, gather,
    or ``newaxis`` in the inner subscript is left untouched.
    """

    def __init__(self, shape_table: Dict[str, Tuple[str, ...]]):
        self.shape_table = shape_table

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)  # collapse inner chains first (bottom-up)
        inner = node.value
        if not isinstance(inner, ast.Subscript) or not isinstance(inner.value, ast.Name):
            return node
        base_shape = self.shape_table.get(inner.value.id)
        if not base_shape:
            return node
        inner_idx = _slice_dims(inner)
        outer_idx = _slice_dims(node)
        # A newaxis or ellipsis in either subscript shifts the axis alignment by an
        # unknown number of axes -- basic-index associativity no longer holds, bail.
        if any(isinstance(x, ast.Constant) and (x.value is None or x.value is Ellipsis)
               for x in inner_idx + outer_idx):
            return node
        # Map inner indices onto base axes: a full ``:`` survives as a result axis,
        # a scalar consumes its axis. Anything else (partial / strided slice, or a
        # fancy index-array) bails.
        new_idx: List[ast.AST] = []
        result_axes: List[int] = []
        for ix in inner_idx:
            if _is_full_slice(ix):
                result_axes.append(len(new_idx))
                new_idx.append(ix)
            elif isinstance(ix, ast.Slice):
                return node
            elif isinstance(ix, ast.Name) and ix.id in self.shape_table:
                # An inner index that is itself an ARRAY is a fancy GATHER, not a
                # scalar axis-consume: ``A[idx][j] == A[idx[j]]`` (a gathered row),
                # NOT the ``A[idx, j]`` a collapse would emit. numpy basic-index
                # associativity does not hold for an advanced index, so leave the
                # chain untouched (the docstring's gather exclusion).
                return node
            else:
                new_idx.append(ix)
        # Base axes the inner subscript did not name are trailing result axes.
        trailing = len(base_shape) - len(new_idx)
        if trailing < 0:
            return node
        for _ in range(trailing):
            result_axes.append(len(new_idx))
            new_idx.append(ast.Slice())
        # The outer indices apply positionally to the surviving result axes.
        if len(outer_idx) > len(result_axes):
            return node
        for oi, ax in zip(outer_idx, result_axes):
            new_idx[ax] = oi
        slot = new_idx[0] if len(new_idx) == 1 else ast.Tuple(elts=new_idx, ctx=ast.Load())
        return ast.copy_location(ast.Subscript(value=inner.value, slice=slot, ctx=node.ctx), node)


def _name_of_subscript(node: ast.Subscript) -> Optional[str]:
    return node.value.id if isinstance(node.value, ast.Name) else None


def _iter_var_name(axis: int) -> str:
    """Stable iter-var generator: ``si0``, ``si1``, ``si2``, ..."""
    return f"si{axis}"


def _const(value: int) -> ast.Constant:
    return ast.Constant(value=value)


def _binop(left: ast.AST, op, right: ast.AST) -> ast.BinOp:
    return ast.BinOp(left=left, op=op, right=right)


class _ChainedSubscriptFlattener(ast.NodeTransformer):
    """Flatten a chained subscript ``A[i0, i1, ...][rest]`` into the single
    combined subscript ``A[i0, i1, ..., rest]`` -- but ONLY when the inner
    index is entirely SCALAR (int / Name), so each inner index consumes a
    leading source axis and the outer index continues on the remaining axes
    (exactly numpy combined basic indexing). ``psi_frag[f][..., 0]`` ->
    ``psi_frag[f, ..., 0]``. A ``Slice``/``Ellipsis``/``newaxis`` inner index is
    NOT flattened -- ``A[1:3][0]`` != ``A[1:3, 0]`` -- so those are left intact.
    Runs before the ellipsis/scalarize passes so they only ever see a subscript
    whose base is a Name."""

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)  # collapse nested chains bottom-up first
        inner = node.value
        if not isinstance(inner, ast.Subscript):
            return node
        inner_elts = list(inner.slice.elts) if isinstance(inner.slice, ast.Tuple) else [inner.slice]
        if not all(_is_scalar_index(e) for e in inner_elts):
            return node
        outer_elts = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        combined = inner_elts + outer_elts
        return ast.copy_location(
            ast.Subscript(value=inner.value, slice=ast.Tuple(elts=combined, ctx=ast.Load()), ctx=node.ctx), node)


def _is_scalar_index(elt: ast.expr) -> bool:
    """A subscript element that selects (consumes) a single source axis: an int
    Constant or a bare Name (loop iter / symbol) -- NOT a Slice, Ellipsis, or
    newaxis (``None``)."""
    if isinstance(elt, ast.Constant):
        return elt.value is not Ellipsis and elt.value is not None
    return isinstance(elt, ast.Name)


class _EllipsisExpander(ast.NodeTransformer):
    """Replace ``...`` (Ellipsis) in a subscript with the explicit full slices
    it stands for, using the array's rank: ``a[..., 0]`` on a 3-D array ->
    ``a[:, :, 0]``. Only fires on a subscript of a known-shape Name (chained
    subscripts are flattened to this form first by _ChainedSubscriptFlattener)."""

    def __init__(self, array_shapes: Dict[str, List[str]]):
        self.array_shapes = array_shapes

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.value, ast.Name):
            return node
        shape = self.array_shapes.get(node.value.id)
        if not shape:
            return node
        sl = node.slice
        elts = list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]
        ell = [k for k, e in enumerate(elts) if isinstance(e, ast.Constant) and e.value is Ellipsis]
        if len(ell) != 1:
            return node
        rank = len(shape)
        # Source-axis-consuming entries (exclude the Ellipsis and any newaxis).
        consumed = sum(1 for e in elts
                       if not (isinstance(e, ast.Constant) and (e.value is Ellipsis or e.value is None)))
        pad = max(rank - consumed, 0)
        pos = ell[0]
        new_elts = (elts[:pos]
                    + [ast.Slice(lower=None, upper=None, step=None) for _ in range(pad)]
                    + elts[pos + 1:])
        node.slice = (new_elts[0] if len(new_elts) == 1
                      else ast.Tuple(elts=new_elts, ctx=ast.Load()))
        return ast.copy_location(node, node)


class _PadImplicitTrailingSlices(ast.NodeTransformer):
    """Make numpy's implicit trailing axes explicit on basic-indexed subscripts.

    ``A[i, j]`` on an n-D array (n > 2) means ``A[i, j, :, ...]`` -- the unlisted
    trailing axes are full slices. The slice / scalar lowering keys off the
    number of index positions, so a 3-D stencil written ``TN[:, 1:] = T[:, :-1]``
    (hotspot_3d) would otherwise iterate only 2 axes and drop the innermost,
    emitting invalid nested ``[][]`` on a flat buffer. Pad each such subscript
    with explicit full ``Slice()`` entries up to the array's rank.

    Only BASIC indexing is padded -- every existing index must be a Slice, a
    scalar int Constant, or a Name that is NOT itself an array (a loop iter /
    symbol). Advanced indexing (``x[src]`` with ``src`` an index array, the
    fancy-gather path) is left untouched so it is not mis-expanded."""

    def __init__(self, array_shapes: Dict[str, List[str]]):
        self.array_shapes = array_shapes

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.value, ast.Name):
            return node
        shape = self.array_shapes.get(node.value.id)
        if not shape:
            return node
        rank = len(shape)
        sl = node.slice
        elts = list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]
        # A newaxis (``None``) inserts a RESULT axis but consumes NO source
        # axis, so it must not count against the array rank -- ``weights[None,
        # :, :, :]`` on a 4-D array still leaves one trailing source axis
        # implicit (conv2d's ``weights[np.newaxis, :, :, :]`` -> 5-D result
        # over a 4-D operand). Count only source-axis-consuming positions.
        def _is_newaxis(e):
            return isinstance(e, ast.Constant) and e.value is None
        n_index = sum(1 for e in elts if not _is_newaxis(e))
        if n_index >= rank:
            return node
        # Basic-indexing gate: a Slice, newaxis, int Constant, or non-array Name.
        for e in elts:
            if isinstance(e, ast.Slice) or _is_newaxis(e):
                continue
            if isinstance(e, ast.Constant) and isinstance(e.value, int):
                continue
            if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub) \
                    and isinstance(e.operand, ast.Constant):
                continue
            if isinstance(e, ast.Name) and e.id not in self.array_shapes:
                continue
            return node                       # advanced / unknown index -> skip
        pad = rank - n_index
        new_elts = elts + [ast.Slice(lower=None, upper=None, step=None)
                           for _ in range(pad)]
        node.slice = ast.Tuple(elts=new_elts, ctx=ast.Load())
        return ast.copy_location(node, node)


def _fold_subarray_aliases(tree: ast.AST, array_shapes: Dict[str, List[str]]) -> None:
    """Fold a partial / trailing-slice sub-array alias into ONE flat multi-dim index.

    ``low = A[i, j]`` (or ``A[i, j, :]``) on a 3-D array is a sub-array; each use
    ``low[k]`` becomes ``A[i, j, k]`` -- a single subscript the emitter lowers to a
    flat offset -- instead of the chained ``A[i][j]`` a partial index otherwise emits
    on a flat C pointer (xsbench's ``low`` / ``high`` five-channel reads). Fires only
    when the alias is a basic-index sub-array of a known array, is assigned exactly
    once, and EVERY use is a further subscript (a bare whole-array use would need the
    row materialised, so it is left alone)."""

    def _is_full_slice(e):
        return isinstance(e, ast.Slice) and e.lower is None and e.upper is None and e.step is None

    aliases: Dict[str, tuple] = {}
    for stmt in ast.walk(tree):
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
            continue
        val = stmt.value
        if not (isinstance(val, ast.Subscript) and isinstance(val.value, ast.Name)):
            continue
        shape = array_shapes.get(val.value.id)
        if not shape:
            continue
        elts = list(val.slice.elts) if isinstance(val.slice, ast.Tuple) else [val.slice]
        while elts and _is_full_slice(elts[-1]):
            elts.pop()  # trailing ``:`` axes are exactly what ``local[k]`` will fill
        # remaining index axes must be plain scalars (no slice / newaxis) and leave at
        # least one trailing source axis (a genuine sub-array, not a full element index).
        if any(isinstance(e, ast.Slice) or (isinstance(e, ast.Constant) and e.value is None) for e in elts):
            continue
        if len(elts) >= len(shape):
            continue
        aliases[stmt.targets[0].id] = (val.value.id, elts)
    if not aliases:
        return

    assigns: Dict[str, int] = {}
    sub_value_ids: set = set()
    load_ids: Dict[str, List[int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in aliases:
                    assigns[t.id] = assigns.get(t.id, 0) + 1
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            sub_value_ids.add(id(node.value))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in aliases:
            load_ids.setdefault(node.id, []).append(id(node))
    # Base-index stability: reject an alias whose base index name is reassigned in a
    # statement that can execute AFTER it (its block-tail, recursively) -- else the
    # folded ``A[i, j, k]`` at the use site would read the NEW i/j, not the value the
    # alias captured. (Reassignment BEFORE the alias is fine.)
    def _child_blocks(s):
        if isinstance(s, (ast.For, ast.While, ast.If)):
            yield s.body
            yield s.orelse
        elif isinstance(s, ast.Try):
            yield s.body
            yield s.orelse
            yield s.finalbody
            for h in s.handlers:
                yield h.body

    def _stores_in(stmts):
        out: set = set()
        for s in stmts:
            for n in ast.walk(s):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    out.add(n.id)
                elif isinstance(n, ast.For) and isinstance(n.target, ast.Name):
                    out.add(n.target.id)
        return out

    unsafe: set = set()

    def _scan(stmts):
        for i, s in enumerate(stmts):
            if (isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)
                    and s.targets[0].id in aliases):
                _, base = aliases[s.targets[0].id]
                base_names = {n.id for b in base for n in ast.walk(b) if isinstance(n, ast.Name)}
                if base_names & _stores_in(stmts[i + 1:]):
                    unsafe.add(s.targets[0].id)
            for cb in _child_blocks(s):
                _scan(cb)

    _scan(tree.body if isinstance(tree, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)) else [tree])
    good = {name: aliases[name] for name in aliases
            if name not in unsafe and assigns.get(name, 0) == 1
            and all(i in sub_value_ids for i in load_ids.get(name, []))}
    if not good:
        return

    class _Fold(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id in good:
                aname, base = good[node.value.id]
                more = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
                new_idx = [copy.deepcopy(b) for b in base] + more
                sl = ast.Tuple(elts=new_idx, ctx=ast.Load()) if len(new_idx) > 1 else new_idx[0]
                return ast.copy_location(
                    ast.Subscript(value=ast.Name(id=aname, ctx=ast.Load()), slice=sl, ctx=node.ctx), node)
            return node

        def visit_Assign(self, node: ast.Assign) -> Optional[ast.AST]:
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in good:
                return None  # drop the now-unused alias assignment
            self.generic_visit(node)
            return node

    _Fold().visit(tree)
    ast.fix_missing_locations(tree)


class _FlattenChainedSubscripts(ast.NodeTransformer):
    """Flatten a chained subscript ``B[inner][outer]`` into ONE combined subscript
    ``B[combined]`` -- the outer index addresses the axes the inner FULL-slices, in
    order. ``deexx[:, ii][ikb]`` -> ``deexx[ikb, ii]``; ``tabxx_qr[ia, :, :][:, k]``
    -> ``tabxx_qr[ia, :, k]``. Unlike :func:`_fold_subarray_aliases` (scalar-prefix
    aliases), this handles slices interleaved in the inner index and applies to any
    chained subscript, not just single-assign aliases -- the QE ultrasoft
    augmentation reads ``tabxx_qr[ia][:, ijtoh[ih, jh]]`` / ``deexx[:, ii][ikb]``.
    A non-full inner slice (``a[1:5][k]``) carries an offset the flat combine can't
    express, so it is left untouched."""

    def __init__(self, shapes: Dict[str, List[str]]):
        self.shapes = shapes

    @staticmethod
    def _is_full_slice(e) -> bool:
        return isinstance(e, ast.Slice) and e.lower is None and e.upper is None and e.step is None

    @staticmethod
    def _is_special(e) -> bool:  # np.newaxis (``None``) / Ellipsis -- rank-shifting
        return isinstance(e, ast.Constant) and (e.value is None or e.value is Ellipsis)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)  # flatten inner chains first (bottom-up)
        inner = node.value
        if not (isinstance(inner, ast.Subscript) and isinstance(inner.value, ast.Name)):
            return node
        shape = self.shapes.get(inner.value.id)
        if not shape:
            return node
        rank = len(shape)
        inner_idx = list(inner.slice.elts) if isinstance(inner.slice, ast.Tuple) else [inner.slice]
        if len(inner_idx) > rank or any(self._is_special(e) for e in inner_idx):
            return node
        inner_idx = inner_idx + [ast.Slice() for _ in range(rank - len(inner_idx))]  # pad trailing ``:``
        if any(isinstance(e, ast.Slice) and not self._is_full_slice(e) for e in inner_idx):
            return node
        outer_idx = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        n_kept = sum(1 for e in inner_idx if isinstance(e, ast.Slice))
        if len(outer_idx) != n_kept or any(self._is_special(e) for e in outer_idx):
            return node
        combined: List[ast.expr] = []
        oi = 0
        for e in inner_idx:
            if isinstance(e, ast.Slice):
                combined.append(copy.deepcopy(outer_idx[oi]))
                oi += 1
            else:
                combined.append(copy.deepcopy(e))
        sl = ast.Tuple(elts=combined, ctx=ast.Load()) if len(combined) > 1 else combined[0]
        return ast.copy_location(
            ast.Subscript(value=ast.Name(id=inner.value.id, ctx=ast.Load()), slice=sl, ctx=node.ctx), node)


class SliceFusion(ast.NodeTransformer):
    """Rewrite slice-bearing assignments into a single fused loop.

    Handles the canonical jacobi-style pattern::

        A[a0:b0, a1:b1] = expr

    where ``expr`` may contain any number of nested ``B[c0:d0, c1:d1]``
    references that share the same logical shape as the LHS. The
    rewriter picks one iteration variable per axis and replaces every
    slice with a scalar subscript indexed by the iter var (plus the
    offset between the slice's start and the LHS slice's start).

    Limitations -- raised as :class:`NotImplementedError`:

    * ``step != 1`` on any slice,
    * slices whose ``stop`` is omitted on an array whose shape we
      cannot resolve (the IR only carries shape symbols for declared
      parameters; for local arrays declared via ``np.zeros`` we have
      shape info too).
    """

    def __init__(self, array_shapes: Dict[str, List[str]]):
        self.array_shapes = array_shapes

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        return self._rewrite(node.targets[0], node.value, aug_op=None) or node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        self.generic_visit(node)
        return self._rewrite(node.target, node.value, aug_op=node.op) or node

    def _rewrite(self, target: ast.AST, value: ast.expr,
                 aug_op: Optional[ast.AST]) -> Optional[ast.AST]:
        """Common slice-fusion path for both Assign and AugAssign.

        ``aug_op`` is ``None`` for plain Assign or the augmented operator
        (``ast.Add``, ``ast.Mult`` etc.) when invoked from AugAssign --
        the rewritten body becomes ``LHS = LHS op RHS`` per element so
        AugAssign semantics survive the per-element expansion.
        """
        if not _has_any_slice(target):
            return None
        if not isinstance(target, ast.Subscript):
            return None
        lhs_name = _name_of_subscript(target)
        if lhs_name is None:
            return None
        lhs_dims = _slice_dims(target)
        # Compute the per-axis iteration range = LHS slice bounds.
        # Negative-index slice bounds ``A[1:-1]`` (or any int < 0) are
        # numpy-style ``axis_length + K``; resolve here so downstream
        # passes see fully concrete bounds.
        ranges: List[Tuple[ast.AST, ast.AST]] = []
        for axis, d in enumerate(lhs_dims):
            if not isinstance(d, ast.Slice):
                ranges.append((d, d))
                continue
            if d.step is not None:
                raise NotImplementedError("slice step != 1 not supported")
            start = self._resolve_bound(d.lower, lhs_name, axis, default=_const(0))
            stop = self._resolve_bound(d.upper, lhs_name, axis,
                                       default=lambda: self._axis_length(lhs_name, axis))
            ranges.append((start, stop))
        # Build the per-axis scalarisation: iter var ``i_axis`` ranging
        # ``[start, stop)``; every RHS subscript gets the iter var
        # offset by the LHS slice's start.
        iter_vars: List[ast.Name] = []
        for axis, (lo, hi) in enumerate(ranges):
            if not isinstance(lhs_dims[axis], ast.Slice):
                iter_vars.append(None)  # type: ignore[arg-type]
                continue
            iter_vars.append(ast.Name(id=_iter_var_name(axis), ctx=ast.Load()))

        # 1) Rewrite the LHS so every slice dim becomes the iter var (plus
        #    the LHS slice's own start when we choose to index relative
        #    to zero; here we index absolute so the LHS subscript is
        #    just the iter var).
        new_lhs = ast.Subscript(
            value=target.value,
            slice=self._scalar_slice(lhs_dims, iter_vars, ranges, lhs_name),
            ctx=ast.Store(),
        )

        # 2) Rewrite the RHS in-place: every Subscript with a Slice gets
        #    replaced by a Subscript whose index is the iter var + offset.
        rhs_rewriter = _SliceToScalarRewriter(self.array_shapes,
                                              iter_vars, ranges, lhs_name, lhs_dims)
        new_rhs = rhs_rewriter.visit(copy.deepcopy(value))
        # Top-level RHS Name (e.g. ``corr[i+1:M, i] = __mm4`` where
        # __mm4 is a 1-D temp matching the LHS slice extents) -- the
        # NodeTransformer doesn't dispatch into a bare Name unless we
        # ask it to, so do the subscriptify on the new top-level node.
        new_rhs = rhs_rewriter._maybe_subscriptify(new_rhs)

        # 3) Emit Assign or AugAssign per element.
        if aug_op is None:
            inner: ast.stmt = ast.Assign(targets=[new_lhs], value=new_rhs)
        else:
            inner = ast.AugAssign(target=new_lhs, op=aug_op, value=new_rhs)

        # 4) Wrap with one ``for`` per slice axis (skip the scalar dims).
        body: List[ast.stmt] = [inner]
        for axis in reversed(range(len(lhs_dims))):
            if not isinstance(lhs_dims[axis], ast.Slice):
                continue
            lo, hi = ranges[axis]
            ivar = iter_vars[axis]
            body = [ast.For(
                target=ast.Name(id=ivar.id, ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[lo, hi], keywords=[]),
                body=body,
                orelse=[],
            )]
        return body[0] if len(body) == 1 else body

    def _axis_length(self, array_name: str, axis: int) -> ast.AST:
        shape = self.array_shapes.get(array_name)
        if shape is None or axis >= len(shape):
            raise NotImplementedError(
                f"slice with omitted stop on {array_name!r} axis {axis}: "
                f"shape unknown to NumpyToC")
        return _token_to_ast(shape[axis])

    def _resolve_bound(self, bound: Optional[ast.AST], array_name: str,
                       axis: int, default) -> ast.AST:
        """Resolve a slice bound, expanding numpy's negative-index form.

        A bound of ``None`` -> ``default`` (typically 0 for start,
        axis length for stop). A bound of ``-K`` (integer constant or
        ``UnaryOp(USub, Constant)``) -> ``axis_length - K``. All other
        bounds pass through unchanged so symbolic expressions like
        ``N-1`` survive.

        ``default`` may be a plain AST node or a zero-arg callable
        producing one; the stop bound's default calls ``_axis_length``,
        which can raise ``NotImplementedError`` on an array with unknown
        shape, so it must only be evaluated when the bound is actually
        omitted -- not on every explicit-stop slice (e.g. ``a[:n]``).
        """
        if bound is None:
            return default() if callable(default) else default
        if isinstance(bound, ast.Constant) and isinstance(bound.value, int) and bound.value < 0:
            return _binop(self._axis_length(array_name, axis),
                          ast.Sub(), _const(-bound.value))
        if (isinstance(bound, ast.UnaryOp) and isinstance(bound.op, ast.USub)
                and isinstance(bound.operand, ast.Constant)
                and isinstance(bound.operand.value, int)):
            return _binop(self._axis_length(array_name, axis),
                          ast.Sub(), _const(bound.operand.value))
        return bound

    def _scalar_slice(self, lhs_dims, iter_vars, ranges, name) -> ast.AST:
        """Build the LHS scalar subscript: iter vars per slice dim,
        original (negative-resolved) index per non-slice dim."""
        idx_nodes: List[ast.AST] = []
        for axis, d in enumerate(lhs_dims):
            if isinstance(d, ast.Slice):
                idx_nodes.append(ast.Name(id=iter_vars[axis].id, ctx=ast.Load()))
            else:
                idx_nodes.append(self._resolve_scalar_index(d, name, axis))
        if len(idx_nodes) == 1:
            return idx_nodes[0]
        return ast.Tuple(elts=idx_nodes, ctx=ast.Load())

    def _resolve_scalar_index(self, idx: ast.AST, name: str,
                              axis: int) -> ast.AST:
        """A negative constant scalar index ``-K`` (e.g. ``y[:, -1]``)
        wraps to ``axis_length - K`` -- numpy semantics. C has no
        wrap-around, so leaving it literal indexes ``arr[... + (-1)]``
        out of bounds (the deriche heap corruption). Other indices pass
        through unchanged so ``N - 1`` etc. survive."""
        if (isinstance(idx, ast.Constant) and isinstance(idx.value, int)
                and idx.value < 0):
            return _binop(self._axis_length(name, axis), ast.Sub(),
                          _const(-idx.value))
        if (isinstance(idx, ast.UnaryOp) and isinstance(idx.op, ast.USub)
                and isinstance(idx.operand, ast.Constant)
                and isinstance(idx.operand.value, int)):
            return _binop(self._axis_length(name, axis), ast.Sub(),
                          _const(idx.operand.value))
        return idx


class _SliceToScalarRewriter(ast.NodeTransformer):
    """Replace each slice-bearing Subscript in an expression with the
    equivalent scalar subscript indexed off the iteration variables.

    The offset between the RHS slice's start and the LHS slice's start
    flows through: at iter var ``i``, an RHS slice ``X[c:d]`` is read
    as ``X[i + (c - lhs_start)]``.
    """

    def __init__(self, array_shapes, iter_vars, lhs_ranges, lhs_name, lhs_dims):
        self.array_shapes = array_shapes
        self.iter_vars = iter_vars
        self.lhs_ranges = lhs_ranges
        self.lhs_name = lhs_name
        self.lhs_dims = lhs_dims
        # Iter vars for slice axes only, in order.
        self._slice_iter_names = [
            iv.id for iv, dim in zip(iter_vars, lhs_dims)
            if isinstance(dim, ast.Slice) and iv is not None
        ]

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node.left = self._maybe_subscriptify(self.visit(node.left))
        node.right = self._maybe_subscriptify(self.visit(node.right))
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        # A bare array Name nested in a unary op (``-ft_e``) must be
        # subscriptified too -- otherwise it stays a whole-array operand inside
        # the per-element store (ICON ddt_vn_cor's ``clin * (-ft_e)``).
        node.operand = self._maybe_subscriptify(self.visit(node.operand))
        return node

    def _maybe_subscriptify(self, node: ast.AST) -> ast.AST:
        """If ``node`` is a bare Name(arr) whose shape rank fits the
        LHS iteration nest, return ``arr[iter_vars]``.

        Two cases are supported:

        * ``rank == len(slice_iter_names)`` -- straight per-axis mapping.
        * ``rank < len(slice_iter_names)`` -- numpy broadcasting: a
          lower-rank array reads from the trailing iter vars (the
          ``b + A`` shape with b:(M,) and A:(N, M) -> b[j], A[i, j]).

        Conservative: only fires from inside ``visit_BinOp`` so we
        don't accidentally subscript names that are receivers of an
        outer Subscript (e.g. ``B[i, j]`` where the rewriter already
        turned the outer slice into a scalar subscript).
        """
        if not isinstance(node, ast.Name):
            return node
        if not isinstance(node.ctx, ast.Load):
            return node
        shape = self.array_shapes.get(node.id)
        if not shape:
            return node
        if len(shape) > len(self._slice_iter_names):
            return node
        # A bare Name reaching here is a 0-based operand (typically a
        # hoisted matmul/dot temp whose logical index 0 aligns with the
        # LHS slice start). Read it at ``iter - lhs_start`` so a slice
        # assignment into a non-zero-start destination
        # (``cov[i:M, i] = data[:, i] @ data[:, i:M] / ...``) pulls the
        # temp's element 0 into destination row ``i``, not row ``2*i``.
        # ``visit_Subscript`` applies the same correction to real sliced
        # operands; this is its bare-Name counterpart.
        lhs_slice_starts = [rng[0] for iv, dim, rng in
                            zip(self.iter_vars, self.lhs_dims, self.lhs_ranges)
                            if isinstance(dim, ast.Slice) and iv is not None]
        iters = self._slice_iter_names[-len(shape):]
        starts = lhs_slice_starts[-len(shape):]
        elts: List[ast.AST] = []
        for dim, iv, start in zip(shape, iters, starts):
            # A size-1 axis broadcasts: pin it to index 0 rather than consuming
            # the (larger) result-axis iter. ``w.reshape(1, -1)`` multiplied
            # against an (N, N) array is (1, N) -- dim 0 must read row 0, not the
            # row iter (which would run off the single-row temp -> OOB).
            if str(dim).strip() == "1":
                elts.append(_const(0))
                continue
            ivar = ast.Name(id=iv, ctx=ast.Load())
            if isinstance(start, ast.Constant) and start.value == 0:
                elts.append(ivar)
            else:
                # Copy the shared ``start`` node (also the loop-header bound) so the
                # bare-operand offset does not alias it into two tree positions.
                elts.append(_binop(ivar, ast.Sub(), copy.deepcopy(start)))
        slot = elts[0] if len(elts) == 1 else ast.Tuple(elts=elts, ctx=ast.Load())
        return ast.Subscript(value=node, slice=slot, ctx=ast.Load())

    @staticmethod
    def _iter_minus_start(iter_name: ast.Name, start: ast.AST) -> ast.AST:
        """The LOCAL result position ``iter - lhs_start`` (or just ``iter`` when the
        LHS slice starts at 0). A gather-index array / trailing source axis reads at
        its 0-based position within the slice, not the absolute destination index."""
        iv = ast.Name(id=iter_name.id, ctx=ast.Load())
        if isinstance(start, ast.Constant) and start.value == 0:
            return iv
        # Copy ``start``: it is the SAME node object as the loop-header ``range``
        # lower bound, so embedding it directly would alias one mutable subtree into
        # two live tree positions (a later in-place rewrite of one corrupts both).
        return _binop(iv, ast.Sub(), copy.deepcopy(start))

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        # Pure broadcast-reshape on a NON-Name value (a BinOp / Call result):
        # ``(q_nb[:, None, :] * fs)[:, :, :, None]`` (lavamd). The slice is only
        # ``:`` and ``np.newaxis``; recursively scalarise the inner expression
        # against just the iters mapped to the ``:`` axes (each newaxis adds a
        # result axis with no source axis). Handled BEFORE generic_visit so the
        # inner expression is scalarised with the correct (newaxis-aware) iter
        # mapping rather than the raw right-aligned one.
        if not isinstance(node.value, ast.Name):
            sl0 = node.slice
            elts0 = list(sl0.elts) if isinstance(sl0, ast.Tuple) else [sl0]
            _full = lambda e: (isinstance(e, ast.Slice) and e.lower is None
                               and e.upper is None and e.step is None)
            _newax = lambda e: isinstance(e, ast.Constant) and e.value is None
            if (elts0 and all(_full(e) or _newax(e) for e in elts0)
                    and any(_full(e) for e in elts0)):
                lhs_slice_iters = [(iv, rng[0]) for iv, dim, rng in
                                   zip(self.iter_vars, self.lhs_dims, self.lhs_ranges)
                                   if isinstance(dim, ast.Slice) and iv is not None]
                align = max(0, len(lhs_slice_iters) - len(elts0))
                if len(elts0) <= len(lhs_slice_iters):
                    sub_iters = [lhs_slice_iters[align + pos]
                                 for pos, e in enumerate(elts0) if _full(e)]
                    sub = _SliceToScalarRewriter(
                        self.array_shapes,
                        [iv for iv, _ in sub_iters],
                        [(lo, lo) for _, lo in sub_iters],
                        None,
                        [ast.Slice(lower=None, upper=None, step=None)
                         for _ in sub_iters])
                    return sub.visit(copy.deepcopy(node.value))
        self.generic_visit(node)
        dims = _slice_dims(node)
        if not any(isinstance(d, ast.Slice) for d in dims):
            # No explicit ``:`` slice, but a PARTIAL scalar index on a
            # higher-rank array (``dH[a, b, j]`` on rank-5 dH) leaves the
            # trailing axes implicit: numpy reads ``dH[a, b, j, :, :]``.
            # Pad those residual axes with the trailing LHS slice iters so
            # the read spans every source axis
            # (scattering_self_energies' ``dHD[si0,si1] = dH[a,b,j]*D``).
            # A full index (num indices == rank) needs no padding.
            name = _name_of_subscript(node)
            source_shape = self.array_shapes.get(name) if name else None
            # Fancy gather: a dim that is an index-array Name (its own shape is
            # in the table) gathers along that source axis. ``momentum[nb]`` on
            # (ncells, 3) -> ``momentum[nb[i], j]`` (cfd / lavamd). The index
            # array(s) consume their (broadcast) rank of LEADING result axes;
            # the source's remaining trailing axes consume the rest.
            if (source_shape is not None
                    and any(isinstance(d, ast.Name) and self.array_shapes.get(d.id)
                            for d in dims)):
                lhs_pairs = [(iv, rng[0]) for iv, dim, rng in
                             zip(self.iter_vars, self.lhs_dims, self.lhs_ranges)
                             if isinstance(dim, ast.Slice) and iv is not None]
                lhs_iters = [iv for iv, _ in lhs_pairs]
                lhs_starts = [st for _, st in lhs_pairs]
                n_trailing = len(source_shape) - len(dims)
                result_rank = sum(
                    (len(self.array_shapes[d.id])
                     if isinstance(d, ast.Name) and self.array_shapes.get(d.id)
                     else 0)
                    for d in dims) + max(0, n_trailing)
                if result_rank <= len(lhs_iters):
                    pos = len(lhs_iters) - result_rank
                    new_elts: List[ast.AST] = []
                    for axis, d in enumerate(dims):
                        if isinstance(d, ast.Name) and self.array_shapes.get(d.id):
                            r = len(self.array_shapes[d.id])
                            # The gather INDEX is the LOCAL result position, so read
                            # it at ``iter - lhs_start`` -- a slice assignment into a
                            # non-zero-start destination (vexx_k noncolin
                            # ``big_result[ip*n:ip*n+n] -= rg[nlg]``, ip=1) must read
                            # ``nlg[si0 - ip*n]``, not ``nlg[si0]`` (which runs off
                            # the length-n index array).
                            giters = [self._iter_minus_start(lhs_iters[pos + k], lhs_starts[pos + k])
                                      for k in range(r)]
                            pos += r
                            gslot = (giters[0] if r == 1
                                     else ast.Tuple(elts=giters, ctx=ast.Load()))
                            new_elts.append(ast.Subscript(value=d, slice=gslot,
                                                          ctx=ast.Load()))
                        else:
                            new_elts.append(self._resolve_scalar_index(d, name, axis))
                    for _ in range(max(0, n_trailing)):
                        new_elts.append(self._iter_minus_start(lhs_iters[pos], lhs_starts[pos]))
                        pos += 1
                    slot = (new_elts[0] if len(new_elts) == 1
                            else ast.Tuple(elts=new_elts, ctx=ast.Load()))
                    return ast.Subscript(value=node.value, slice=slot, ctx=node.ctx)
            if (source_shape is not None
                    and len(dims) < len(source_shape)
                    and not any(isinstance(d, ast.Constant) and d.value is None
                                for d in dims)):
                lhs_pairs = [(iv, rng[0]) for iv, dim, rng in
                             zip(self.iter_vars, self.lhs_dims, self.lhs_ranges)
                             if isinstance(dim, ast.Slice) and iv is not None]
                n_trailing = len(source_shape) - len(dims)
                if 0 < n_trailing <= len(lhs_pairs):
                    # The implicit trailing source axes read at the LOCAL slice
                    # position ``iter - lhs_start`` -- a partial-scalar read
                    # (``out[k:k+m] = dH[a, b]``) into a NON-zero-start destination
                    # must span the source's length-``m`` trailing axis from 0, not
                    # from ``k`` (the sibling gather branch applies the same offset).
                    pad = [self._iter_minus_start(iv, st)
                           for iv, st in lhs_pairs[-n_trailing:]]
                    new_slice = ast.Tuple(elts=list(dims) + pad, ctx=ast.Load())
                    return ast.Subscript(value=node.value, slice=new_slice,
                                         ctx=node.ctx)
            # A FULLY scalar-indexed read (``w_dist[-1]``) is a scalar element:
            # resolve any negative index against the axis length (C / Fortran
            # have no negative indexing) and keep it -- the stencil_*_vc
            # last-weight read inside a slice-fused statement.
            if source_shape is not None and len(dims) == len(source_shape):
                resolved = [self._resolve_scalar_index(d, name, axis)
                            for axis, d in enumerate(dims)]
                if any(r is not d for r, d in zip(resolved, dims)):
                    slot = (resolved[0] if len(resolved) == 1
                            else ast.Tuple(elts=resolved, ctx=ast.Load()))
                    return ast.Subscript(value=node.value, slice=slot, ctx=node.ctx)
            return node
        rhs_name = _name_of_subscript(node)
        # The LHS has N slice axes -- collect the iter vars + LHS lo
        # for those in order. RHS slice axes (which may live on
        # different positions) consume that sequence in order.
        # ``C[i, :i+1] += A[:i+1, k]`` -> LHS slice axis 1, RHS slice
        # axis 0; both use iter var ``si0``.
        lhs_slice_iters = [(iv, rng[0]) for iv, dim, rng in
                           zip(self.iter_vars, self.lhs_dims, self.lhs_ranges)
                           if isinstance(dim, ast.Slice) and iv is not None]
        # numpy broadcasting aligns operand axes from the RIGHT. The RHS's
        # *result* axes (each Slice or newaxis contributes one; a scalar index
        # removes an axis) map onto the LHS slice iters right-aligned -- so a
        # row vector ``A[k, k:]`` (one result axis, the columns) inside a
        # 2-slice-axis LHS ``A[k+1:, k:]`` reads from the COLUMN iter ``si1``,
        # not the row iter ``si0`` (gaussian's rank-1 update). ``align`` shifts
        # the per-axis consumption by the rank difference.
        # A Slice / newaxis contributes one result axis; an ADVANCED index (a Name
        # whose own shape is in the table, e.g. lulesh ``x1[:, _VOLU_PERM]`` with
        # _VOLU_PERM (8,6)) contributes its RANK; a scalar index contributes none.
        rhs_result_axes = sum(
            (len(self.array_shapes[d.id]) if isinstance(d, ast.Name) and self.array_shapes.get(d.id) else
             1 if (isinstance(d, ast.Slice) or (isinstance(d, ast.Constant) and d.value is None)) else 0)
            for d in dims)
        align = max(0, len(lhs_slice_iters) - rhs_result_axes)
        idx_nodes: List[ast.AST] = []
        rhs_slice_idx = 0
        for axis, d in enumerate(dims):
            if isinstance(d, ast.Constant) and d.value is None:
                # numpy newaxis -- result-axis inserter; consume one
                # LHS slice iter but emit no source-axis index. The
                # broadcast pulls the source through the size-1 axis.
                rhs_slice_idx += 1
                continue
            # Advanced index mixed with slices: a rank-r index array consumes r
            # result axes and reads ``IDX[(those iters)]`` along this source axis
            # (``x1[:, _VOLU_PERM]`` -> ``x1[w0, _VOLU_PERM[w1, w2]]``).
            if isinstance(d, ast.Name) and self.array_shapes.get(d.id):
                r = len(self.array_shapes[d.id])
                if align + rhs_slice_idx + r <= len(lhs_slice_iters):
                    # Gather index reads at the LOCAL result position (iter - start),
                    # so a non-zero-start LHS slice indexes the length-matched index
                    # array within bounds.
                    giters = [self._iter_minus_start(lhs_slice_iters[align + rhs_slice_idx + k][0],
                                                     lhs_slice_iters[align + rhs_slice_idx + k][1])
                              for k in range(r)]
                    rhs_slice_idx += r
                    gslot = giters[0] if r == 1 else ast.Tuple(elts=giters, ctx=ast.Load())
                    idx_nodes.append(ast.Subscript(value=ast.Name(id=d.id, ctx=ast.Load()),
                                                   slice=gslot, ctx=ast.Load()))
                    continue
            if not isinstance(d, ast.Slice):
                idx_nodes.append(self._resolve_scalar_index(d, rhs_name, axis))
                continue
            from numpyto_common.lib_nodes import _slice_step_const
            step = _slice_step_const(d)
            if align + rhs_slice_idx >= len(lhs_slice_iters):
                # More RHS slices than LHS slice axes -- keep the slice
                # for downstream emission to flag.
                idx_nodes.append(d)
                continue
            ivar_node, lhs_start = lhs_slice_iters[align + rhs_slice_idx]
            rhs_slice_idx += 1
            rhs_start = self._resolve_bound(d.lower, rhs_name, axis,
                                            default=_const(0))
            ivar = ast.Name(id=ivar_node.id, ctx=ast.Load())
            if step is not None and step != 1:
                # Strided RHS slice ``a[lo:hi:k]``: the source index for the
                # result position ``pos = ivar - lhs_start`` is ``lo + pos*k``.
                # dwt2d Haar ``b[:, 0::2]`` with a full-slice LHS (lhs_start 0)
                # -> ``b[i, 2*j]``.
                # A NEGATIVE step with the start omitted (``a[::-1]`` / ``a[:hi:-1]``)
                # begins at the LAST index ``axis_len - 1``, not 0 (numpy reverse), so
                # ``a[::-1]`` reads ``a[(N - 1) - pos]`` rather than the wrong ``a[-pos]``.
                if step < 0 and d.lower is None:
                    _ss = self.array_shapes.get(rhs_name)
                    if _ss and axis < len(_ss):
                        _al = (_const(int(_ss[axis])) if str(_ss[axis]).isdigit()
                               else ast.Name(id=str(_ss[axis]), ctx=ast.Load()))
                        rhs_start = _binop(_al, ast.Sub(), _const(1))
                    else:
                        # Without the axis length we cannot seed the reverse start at
                        # ``axis_len - 1``; emitting ``pos * -1`` would be a negative,
                        # out-of-bounds read. Refuse rather than miscompile (a loud,
                        # rare skip -- untracked-shape reverse slice).
                        raise NotImplementedError(
                            f"reverse slice of {rhs_name!r} needs a known axis length")
                pos: ast.expr = ivar
                if not (isinstance(lhs_start, ast.Constant) and lhs_start.value == 0):
                    pos = _binop(ivar, ast.Sub(), lhs_start)
                scaled = _binop(pos, ast.Mult(), _const(step))
                if isinstance(rhs_start, ast.Constant) and rhs_start.value == 0:
                    idx_nodes.append(scaled)
                else:
                    idx_nodes.append(_binop(scaled, ast.Add(), rhs_start))
                continue
            offset = _fold_offset(rhs_start, lhs_start)
            if offset is None:
                idx_nodes.append(_binop(
                    ivar, ast.Add(),
                    _binop(rhs_start, ast.Sub(), lhs_start)))
            elif offset == 0:
                idx_nodes.append(ivar)
            elif offset > 0:
                idx_nodes.append(_binop(ivar, ast.Add(), _const(offset)))
            else:
                idx_nodes.append(_binop(ivar, ast.Sub(), _const(-offset)))
        # Implicit trailing axes: ``conv1[np.newaxis, :, :, :]`` on a
        # 4-D conv1 has only 4 dim elements (1 newaxis + 3 slices) but
        # the source array has 4 axes -- the 4th axis is implicit (all
        # of it). Pad ``idx_nodes`` with the remaining LHS iters so the
        # emitted Subscript covers every source axis.
        source_axes_consumed = sum(
            1 for d in dims
            if not (isinstance(d, ast.Constant) and d.value is None))
        source_shape = self.array_shapes.get(rhs_name)
        if source_shape is not None:
            while (source_axes_consumed < len(source_shape)
                   and rhs_slice_idx < len(lhs_slice_iters)):
                ivar_node, _ = lhs_slice_iters[rhs_slice_idx]
                rhs_slice_idx += 1
                source_axes_consumed += 1
                idx_nodes.append(ast.Name(id=ivar_node.id, ctx=ast.Load()))
        new_slice = idx_nodes[0] if len(idx_nodes) == 1 else \
            ast.Tuple(elts=idx_nodes, ctx=ast.Load())
        return ast.Subscript(value=node.value, slice=new_slice, ctx=node.ctx)

    def _resolve_scalar_index(self, idx: ast.AST, array_name: Optional[str],
                              axis: int) -> ast.AST:
        """A negative constant scalar index ``-K`` on a non-slice axis
        (``imgIn[:, -1]``) wraps to ``axis_length - K`` -- numpy
        semantics. Mirrors :meth:`SliceFusion._resolve_scalar_index` but
        reads the operand shape from ``self.array_shapes`` (RHS side)."""
        shape = self.array_shapes.get(array_name) if array_name else None
        val = None
        if (isinstance(idx, ast.Constant) and isinstance(idx.value, int)
                and idx.value < 0):
            val = -idx.value
        elif (isinstance(idx, ast.UnaryOp) and isinstance(idx.op, ast.USub)
              and isinstance(idx.operand, ast.Constant)
              and isinstance(idx.operand.value, int)):
            val = idx.operand.value
        if val is not None and shape and axis < len(shape):
            axis_len = (_const(int(shape[axis])) if str(shape[axis]).isdigit()
                        else ast.Name(id=str(shape[axis]), ctx=ast.Load()))
            return _binop(axis_len, ast.Sub(), _const(val))
        return idx

    def _resolve_bound(self, bound: Optional[ast.AST], array_name: Optional[str],
                       axis: int, default: ast.AST) -> ast.AST:
        """Mirror :meth:`SliceFusion._resolve_bound` for the RHS scalarizer.

        Resolves negative-index bounds against the operand array's shape
        (not the LHS shape). Required so a stencil read
        ``A[1:-1, 1:-1, 1:-1]`` on a 3-D ``A`` rewrites to
        ``A[i, j, k]`` with iter vars whose upper bound is ``N - 1``
        rather than the literal ``-1`` (which would generate an empty loop).
        """
        if bound is None:
            return default
        shape = self.array_shapes.get(array_name) if array_name else None
        if isinstance(bound, ast.Constant) and isinstance(bound.value, int) and bound.value < 0:
            if shape and axis < len(shape):
                axis_len = _const(int(shape[axis])) if shape[axis].isdigit() \
                    else ast.Name(id=shape[axis], ctx=ast.Load())
                return _binop(axis_len, ast.Sub(), _const(-bound.value))
        if (isinstance(bound, ast.UnaryOp) and isinstance(bound.op, ast.USub)
                and isinstance(bound.operand, ast.Constant)
                and isinstance(bound.operand.value, int)
                and shape and axis < len(shape)):
            axis_len = _const(int(shape[axis])) if shape[axis].isdigit() \
                else ast.Name(id=shape[axis], ctx=ast.Load())
            return _binop(axis_len, ast.Sub(), _const(bound.operand.value))
        return bound


def _fold_offset(rhs_start: ast.AST, lhs_start: ast.AST) -> Optional[int]:
    """Return the integer offset ``rhs_start - lhs_start`` when both
    sides are integer constants; ``None`` otherwise.

    Used by :class:`_SliceToScalarRewriter` so the emitted body reads
    ``A[i-1]`` / ``A[i+1]`` / ``A[i]`` instead of ``A[i+(0-1)]`` /
    ``A[i+(2-1)]`` / ``A[i+(1-1)]`` -- the C compiler folds these
    anyway, but the human-readable form is the whole point of slice
    fusion.
    """
    if (isinstance(rhs_start, ast.Constant)
            and isinstance(lhs_start, ast.Constant)
            and isinstance(rhs_start.value, int)
            and isinstance(lhs_start.value, int)):
        return rhs_start.value - lhs_start.value
    return None


class _DaceMapRewriter(ast.NodeTransformer):
    """Rewrite ``for i, in dace.map[lo:hi:step]:`` to ``for i in range(lo, hi, step):``.

    The Foundation corpus uses ``dace.map`` for some kernels' outer
    loops; semantically it's a parallel range, but for the emitter
    we only care that the iteration shape matches a Python ``range``.
    """

    def visit_For(self, node: ast.For) -> ast.AST:
        self.generic_visit(node)
        # Detect ``for i, in dace.map[a:b:c]:`` (single-element tuple target,
        # subscript of attribute ``dace.map``).
        target = node.target
        if (isinstance(target, ast.Tuple)
                and len(target.elts) == 1
                and isinstance(target.elts[0], ast.Name)):
            target = target.elts[0]  # type: ignore[assignment]
            node.target = target
        if (isinstance(node.iter, ast.Subscript)
                and isinstance(node.iter.value, ast.Attribute)
                and isinstance(node.iter.value.value, ast.Name)
                and node.iter.value.value.id == "dace"
                and node.iter.value.attr == "map"):
            sl = node.iter.slice
            if isinstance(sl, ast.Slice):
                args: List[ast.AST] = [
                    sl.lower if sl.lower is not None else ast.Constant(value=0),
                    sl.upper,
                ]
                if sl.step is not None:
                    args.append(sl.step)
                node.iter = ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[a for a in args if a is not None],
                    keywords=[])
        return node


class _BooleanMaskRewriter(ast.NodeTransformer):
    """Lower ``arr[mask_expr] = value`` / ``arr[mask_expr] op= value``
    into a per-element loop with a conditional guard.

    Three shapes recognised on the LHS index expression:

    * A ``Compare`` whose left side is the LHS array (or any operand
      with the LHS array's shape) -- ``stddev[stddev <= 0.1] = 1.0``.
    * A ``BoolOp`` over per-element comparisons.
    * A bare ``Name`` referencing a previously-computed boolean array
      of the LHS array's shape (mandelbrot ``Z[I] = ...`` where ``I``
      came from ``np.less(abs(Z), horizon)``).

    The rewritten form is a per-element loop nest over the LHS array's
    shape; the ``if`` body holds the original assignment with the LHS
    array indexed at the iter vars and the RHS scalarised at the same
    iters (so ``Z[I] = Z[I]**2 + C[I]`` becomes
    ``for i: if I[i]: Z[i] = Z[i]**2 + C[i]``).
    """

    def __init__(self, shape_table):
        self.shape_table = shape_table

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        return self._rewrite(node.targets[0] if len(node.targets) == 1 else None,
                              node.value, aug_op=None) or node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        self.generic_visit(node)
        return self._rewrite(node.target, node.value, aug_op=node.op) or node

    def _rewrite(self, target, value, aug_op):
        if (not isinstance(target, ast.Subscript)
                or not isinstance(target.value, ast.Name)):
            return None
        arr_name = target.value.id
        shape = self.shape_table.get(arr_name)
        if not shape:
            return None
        mask_expr = target.slice
        if not self._is_mask_expr(mask_expr, shape, arr_name):
            return None
        iters = [f"__bm{i}" for i in range(len(shape))]
        idx = (ast.Name(id=iters[0], ctx=ast.Load()) if len(iters) == 1 else
               ast.Tuple(elts=[ast.Name(id=i, ctx=ast.Load()) for i in iters],
                         ctx=ast.Load()))
        # Scalarise the mask at iter indices.
        mask_scalar = _SubscriptifyNames(self.shape_table, iters).visit(
            copy.deepcopy(mask_expr))
        # Strip out ``arr[mask_name]`` boolean-mask reads on the RHS:
        # the read produces a bool-masked slice in numpy, but inside
        # the guarded per-element body the access reduces to
        # ``arr[iters]``. Keep the original ``mask_name`` only on the
        # mask check itself.
        rhs_clean = _strip_mask_subscripts(
            copy.deepcopy(value),
            mask_names=_mask_names(mask_expr),
            mask_expr=mask_expr)
        # Scalarise the RHS at iter indices.
        rhs_scalar = _SubscriptifyNames(self.shape_table, iters).visit(
            rhs_clean)
        # Build LHS subscript at iter vars.
        lhs_sub = ast.Subscript(
            value=ast.Name(id=arr_name, ctx=ast.Load()),
            slice=idx, ctx=ast.Store())
        if aug_op is None:
            inner = ast.Assign(targets=[lhs_sub], value=rhs_scalar)
        else:
            inner = ast.AugAssign(target=lhs_sub, op=aug_op, value=rhs_scalar)
        guarded = ast.If(test=mask_scalar, body=[inner], orelse=[])
        # Wrap in loops.
        out: List[ast.stmt] = [guarded]
        for var, bound in zip(reversed(iters), reversed(list(shape))):
            out = [ast.For(
                target=ast.Name(id=var, ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[_token_to_ast(bound)],
                    keywords=[]),
                body=out, orelse=[])]
        return out

    def _is_mask_expr(self, expr, lhs_shape, lhs_name):
        """Return True when ``expr`` evaluates to a boolean array of
        ``lhs_shape``. Conservative: only the recognised shapes."""
        from numpyto_common.lib_nodes import _iter_extent_of

        def _array_shaped(e):
            # A bare Name fast-path, then any array-valued EXPRESSION whose
            # iteration extent has the LHS rank: ``abs(Z) < horizon`` (the
            # operand is a Call wrapping the array, not a bare Name) is a valid
            # mask, as is ``N_out == 0`` (mandelbrot2 ``&``-combined masks).
            if isinstance(e, ast.Name):
                shape = self.shape_table.get(e.id)
                return bool(shape) and tuple(shape) == tuple(lhs_shape)
            ext = _iter_extent_of(e, self.shape_table)
            return ext is not None and len(ext) == len(lhs_shape)
        if isinstance(expr, ast.Compare):
            return any(_array_shaped(op)
                       for op in [expr.left, *expr.comparators])
        if isinstance(expr, ast.BoolOp):
            return all(self._is_mask_expr(v, lhs_shape, lhs_name) for v in expr.values)
        # ``&`` / ``|`` on boolean arrays are elementwise BitAnd / BitOr (numpy
        # spells logical array-ops this way): mandelbrot2's
        # ``N_out[(abs(Z) > horizon) & (N_out == 0)] = i + 1``.
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.BitAnd, ast.BitOr)):
            return (self._is_mask_expr(expr.left, lhs_shape, lhs_name)
                    and self._is_mask_expr(expr.right, lhs_shape, lhs_name))
        if isinstance(expr, ast.Name):
            shape = self.shape_table.get(expr.id)
            if shape and tuple(shape) == tuple(lhs_shape):
                return True
            return False
        return False


def _token_to_ast(tok: str) -> ast.expr:
    """Render a shape token as the appropriate AST node.

    Plain integer / identifier shortcuts; compound expressions
    (``"H - K + 1"`` / ``"x.shape[3]"`` / ``"(H_out // 2)"``) re-parse
    via :func:`ast.parse` so downstream AST walkers see real Subscript
    / BinOp nodes rather than ``Name(id="<literal source text>")``.
    """
    try:
        return ast.Constant(value=int(tok))
    except (TypeError, ValueError):
        pass
    if isinstance(tok, str) and tok.isidentifier():
        return ast.Name(id=tok, ctx=ast.Load())
    try:
        return ast.parse(str(tok), mode="eval").body
    except (SyntaxError, ValueError):
        return ast.Name(id=str(tok), ctx=ast.Load())


class _ResolveArrShape(ast.NodeTransformer):
    """Replace ``arr.shape[i]`` (where ``i`` is a constant int and
    ``arr`` is in the shape table) with the corresponding token.

    Walks the function body in source order so a reassigned local
    (``x = relu(...); x = maxpool2d(x); ...``) has the THEN-current
    shape used at each reference point. Cross-statement state is
    tracked via :attr:`current` which is forked at branches.

    The token may be a plain identifier (``N``), an integer literal
    (``5``), or a compound source expression (``H - K + 1``). Plain
    identifiers and ints parse back to the appropriate AST; compound
    forms are re-parsed via :func:`ast.parse` so the result is a real
    expression node, not an unparsable string.
    """

    def __init__(self, shapes: Dict[str, List[str]],
                 param_shapes: Optional[Dict[str, Tuple[str, ...]]] = None,
                 zeros_locals: Optional[Dict[str, Tuple[str, ...]]] = None,
                 reassign_shapes: Optional[Dict[str, List[Tuple[str, ...]]]] = None,
                 ) -> None:
        # ``shapes`` is the harvest's final-state table (used as a
        # fallback for purely-static lookups). ``current`` is the
        # WORKING table -- it is seeded ONLY with bench-info
        # parameter shapes (and any harvested locals that are never
        # reassigned) and gets updated as we walk statements in
        # source order. This way ``x.shape[i]`` at line K resolves
        # against the value of ``x`` AT line K, not the final value.
        self.shapes = shapes
        # ``zeros_locals`` carries the harvested shapes of every
        # ``Name = __optarena_zeros__()`` marker so the resolver can
        # populate ``current`` when it hits one of those markers
        # without having to look at the np.zeros original call.
        self.zeros_locals = zeros_locals or {}
        # ``reassign_shapes`` is a per-name FIFO of shapes recorded
        # by ``_WholeArrayAssignRewriter`` for every reassignment.
        # When we hit the Nth marker for a given name we pop the Nth
        # shape from this list (a name reassigned 3 times will have
        # 3 entries here, consumed in source order).
        self._reassign_shapes: Dict[str, List[Tuple[str, ...]]] = {
            k: list(v) for k, v in (reassign_shapes or {}).items()}
        if param_shapes is not None:
            self.current: Dict[str, Tuple[str, ...]] = {
                k: tuple(v) for k, v in param_shapes.items()}
        else:
            self.current = {k: tuple(v) for k, v in shapes.items()}

    def _reresolve_token(self, tok: str) -> str:
        """Re-resolve any ``arr.shape[i]`` references inside ``tok``
        against the live ``self.current`` shape table. Tokens are
        re-parsed and substituted axis-wise; the returned string is
        always re-emittable (passes back through
        :func:`_token_to_ast` correctly)."""
        try:
            tree = ast.parse(str(tok), mode="eval").body
        except (SyntaxError, ValueError):
            return tok

        class _Sub(ast.NodeTransformer):

            def __init__(self_inner, current):
                self_inner.current = current

            def visit_Subscript(self_inner, node):
                self_inner.generic_visit(node)
                if not (isinstance(node.value, ast.Attribute)
                        and node.value.attr == "shape"
                        and isinstance(node.value.value, ast.Name)
                        and isinstance(node.slice, ast.Constant)
                        and isinstance(node.slice.value, int)):
                    return node
                src = self_inner.current.get(node.value.value.id)
                if not src or node.slice.value >= len(src):
                    return node
                return _token_to_ast(src[node.slice.value])

        tree = _Sub(self.current).visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.body = self._visit_stmt_list(node.body)
        # After walking the body, write the resolver's updated
        # zeros_locals back to the tree-attribute the emitter reads.
        if hasattr(node, "zeros_locals"):
            node.zeros_locals.update(self.zeros_locals)  # type: ignore[attr-defined]
        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        node.iter = self.visit(node.iter)
        node.body = self._visit_stmt_list(node.body)
        node.orelse = self._visit_stmt_list(node.orelse)
        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        node.test = self.visit(node.test)
        node.body = self._visit_stmt_list(node.body)
        node.orelse = self._visit_stmt_list(node.orelse)
        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        node.test = self.visit(node.test)
        node.body = self._visit_stmt_list(node.body)
        node.orelse = self._visit_stmt_list(node.orelse)
        return node

    def _visit_stmt_list(self, stmts: List[ast.stmt]) -> List[ast.stmt]:
        out: List[ast.stmt] = []
        for stmt in stmts:
            new_stmt = self.visit(stmt)
            self._update_shape_for(stmt)
            if isinstance(new_stmt, list):
                out.extend(new_stmt)
            else:
                out.append(new_stmt)
        return out

    def _update_shape_for(self, stmt: ast.stmt) -> None:
        """Update ``self.current`` to reflect the shape of an Assign
        target. Recognises:

        * ``Name = Name`` -- alias, inherit source shape.
        * ``Name = np.zeros((N, M), ...)`` / ``np.empty([...])`` /
          ``np.empty_like(other)`` -- shape from the constructor.
        * ``Name = BinOp/UnaryOp/IfExp`` -- shape from broadcast via
          :func:`_iter_extent_of`.
        * ``Name = Call(...)`` -- if the call is an elementwise math
          intrinsic, propagate the first array operand's shape.
        * Anything else -- leave the existing entry alone (or unset
          a non-broadcast result).
        """
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)):
            return
        target = stmt.targets[0].id
        rhs = stmt.value
        # ``Name = __optarena_zeros__()`` marker -- could be from the
        # ZerosRewriter (single shape per name in ``zeros_locals``)
        # OR from ``_WholeArrayAssignRewriter`` (one marker per
        # reassignment, shape FIFO in ``_reassign_shapes``).
        if (isinstance(rhs, ast.Call)
                and isinstance(rhs.func, ast.Name)
                and rhs.func.id == "__optarena_zeros__"):
            if target in self._reassign_shapes and self._reassign_shapes[target]:
                self.current[target] = self._reassign_shapes[target].pop(0)
                return
            if target in self.zeros_locals:
                # Re-resolve any ``arr.shape[i]`` references inside the
                # stored shape tokens against ``self.current`` so a
                # reassigned source array (lenet's ``x``) contributes
                # the correct THEN-current axis lengths. The
                # ``self.zeros_locals`` entry is also updated so the
                # emitter's decl block uses the fresh tokens.
                fresh = tuple(self._reresolve_token(t)
                              for t in self.zeros_locals[target])
                self.current[target] = fresh
                self.zeros_locals[target] = fresh
                return
        # Bare Name alias.
        if isinstance(rhs, ast.Name):
            src = self.current.get(rhs.id)
            if src is not None:
                self.current[target] = src
            return
        # np.zeros / empty / ones / linspace / arange.
        from numpyto_common.lib_nodes import NP_ZEROS_ALIASES, _iter_extent_of
        if (isinstance(rhs, ast.Call)
                and isinstance(rhs.func, ast.Attribute)
                and isinstance(rhs.func.value, ast.Name)
                and rhs.func.value.id == "np"):
            attr = rhs.func.attr
            if attr in NP_ZEROS_ALIASES and rhs.args:
                if attr.endswith("_like") and isinstance(rhs.args[0], ast.Name):
                    src = self.current.get(rhs.args[0].id)
                    if src is not None:
                        self.current[target] = src
                    return
                shape_arg = rhs.args[0]
                if isinstance(shape_arg, (ast.Tuple, ast.List)):
                    self.current[target] = tuple(
                        _resolve_shape_token(e, self.current)
                        for e in shape_arg.elts)
                    return
                if (isinstance(shape_arg, ast.Attribute)
                        and shape_arg.attr == "shape"
                        and isinstance(shape_arg.value, ast.Name)):
                    src = self.current.get(shape_arg.value.id)
                    if src is not None:
                        self.current[target] = tuple(src)
                    return
            if attr == "linspace" and len(rhs.args) >= 3:
                tok = ast.unparse(rhs.args[2])
                self.current[target] = (tok,)
                return
        # BinOp / UnaryOp / IfExp -- broadcast. An all-size-1 result is a scalar (see extent_is_scalar).
        from numpyto_common.lib_nodes import extent_is_scalar
        if isinstance(rhs, (ast.BinOp, ast.UnaryOp, ast.IfExp)):
            ext = _iter_extent_of(rhs, self.current)
            if ext is not None and not extent_is_scalar(ext):
                self.current[target] = tuple(
                    ast.unparse(e) for e in ext)
            return
        # Elementwise call (np.maximum, np.add, etc.) -- propagate first
        # array operand's broadcast extent.
        if isinstance(rhs, ast.Call):
            ext = _iter_extent_of(rhs, self.current)
            if ext is not None and not extent_is_scalar(ext):
                self.current[target] = tuple(
                    ast.unparse(e) for e in ext)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if not (isinstance(node.value, ast.Attribute)
                and node.value.attr == "shape"
                and isinstance(node.value.value, ast.Name)
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, int)):
            return node
        src = self.current.get(node.value.value.id)
        if not src or node.slice.value >= len(src):
            return node
        tok = src[node.slice.value]
        # Plain literal int / identifier shortcut.
        try:
            return ast.Constant(value=int(tok))
        except (TypeError, ValueError):
            pass
        # Try parsing as a pure expression -- ``H - K + 1`` / ``N`` /
        # ``(N + 1)``. Strip any surrounding parens for cleanliness.
        try:
            parsed = ast.parse(str(tok), mode="eval").body
            return parsed
        except (SyntaxError, ValueError):
            return ast.Name(id=str(tok), ctx=ast.Load())


def _mask_names(mask_expr: ast.AST) -> Set[str]:
    """Return the bare Name references inside a boolean mask
    expression -- the candidates whose ``arr[name]`` reads should be
    treated as boolean-mask reductions in the RHS-cleanup pass."""
    out: Set[str] = set()
    if isinstance(mask_expr, ast.Name):
        out.add(mask_expr.id)
    else:
        for sub in ast.walk(mask_expr):
            if isinstance(sub, ast.Name):
                out.add(sub.id)
    return out


def _strip_mask_subscripts(expr: ast.AST, mask_names: Set[str],
                            mask_expr: Optional[ast.AST] = None) -> ast.AST:
    """Recursively replace ``arr[name]`` (where ``name`` is one of
    ``mask_names``) with the bare ``arr`` so the surrounding scalariser
    can subscript ``arr`` at the per-element iters. The mask itself is
    pulled out and applied as an ``if`` guard upstream.

    Also strips ``arr[<mask_expr>]`` reads on the RHS where the slice is
    the same Compare / BoolOp as the LHS-side mask (the
    ``inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0]**(-1.5)`` self-mask form).
    """
    mask_src = ast.unparse(mask_expr) if mask_expr is not None else None

    class _Strip(ast.NodeTransformer):

        def visit_Subscript(self_inner, node: ast.Subscript) -> ast.AST:
            self_inner.generic_visit(node)
            if (isinstance(node.slice, ast.Name)
                    and node.slice.id in mask_names):
                return node.value
            if (mask_src is not None
                    and isinstance(node.slice, (ast.Compare, ast.BoolOp, ast.BinOp))
                    and ast.unparse(node.slice) == mask_src):
                return node.value
            return node

    out = _Strip().visit(expr)
    ast.fix_missing_locations(out)
    return out


class _LiftFreshArrayFromSlices(ast.NodeTransformer):
    """Convert ``lap_field = expr_with_slice_subscripts`` into

        lap_field = np.zeros((extent,));   # registered as a local
        lap_field[:] = expr

    so the existing :class:`SliceFusion` lowers the per-element form.

    Triggered when the LHS is a bare Name without a shape entry and
    the RHS contains at least one Subscript with a Slice axis whose
    iteration extent is derivable.

    The new ``Name = np.zeros(...)`` is a marker -- we don't emit any
    initializer; the LHS storage is declared by the emitter from
    ``zeros_locals``. The marker is dropped by stamping the RHS as
    ``__optarena_zeros__()`` (which the emitter already swallows).
    """

    def __init__(self, shapes: Dict[str, List[str]],
                 local_dtypes: Optional[Dict[str, str]] = None) -> None:
        self.shapes: Dict[str, List[str]] = dict(shapes)
        self.new_locals: Dict[str, Tuple[str, ...]] = {}
        # Side-effect: when the RHS contains a complex literal like
        # ``1j``, infer that the fresh local should be declared as
        # complex128 (mandelbrot ``C = X + Y[:, None] * 1j``).
        self.local_dtypes: Dict[str, str] = (
            local_dtypes if local_dtypes is not None else {})

    def run(self, tree: ast.AST) -> Dict[str, Tuple[str, ...]]:
        """Mutate ``tree`` in place and return the new-local shape map."""
        self.visit(tree)
        return self.new_locals

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            return node
        if not (self._has_slice_subscript(node.value)
                or self._is_array_binop(node.value)):
            return node
        from numpyto_common.lib_nodes import _iter_extent_of
        ext = _iter_extent_of(node.value, self.shapes)
        if ext is None:
            return node
        shape_toks: Tuple[str, ...] = tuple(self._tokenise(e) for e in ext)
        existing = self.shapes.get(target.id)
        # If the target already has a shape that matches the derived
        # extent, lift unconditionally (this is the
        # ``C = X + Y[:, None] * 1j`` case where an earlier rewriter
        # already deduced C's shape via broadcasting). Otherwise the
        # target must be a fresh local.
        if existing is not None:
            if tuple(existing) != shape_toks:
                return node
        else:
            self.new_locals[target.id] = shape_toks
        self.shapes[target.id] = list(shape_toks)
        # Infer complex dtype when the RHS contains any complex literal
        # ``1j`` (or operates on an already-complex array). C99
        # ``_Complex`` is assignment-compatible with real ``double`` in
        # C (with a warning) but the C++ emit path is a hard type error,
        # so we must tag the LHS as complex when the value is complex.
        if target.id not in self.local_dtypes:
            inferred = _infer_complex_dtype(node.value, self.local_dtypes)
            if inferred is not None:
                self.local_dtypes[target.id] = inferred
        marker = ast.Assign(
            targets=[ast.Name(id=target.id, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id="__optarena_zeros__", ctx=ast.Load()),
                args=[], keywords=[]))
        # ``C[:]`` only iterates the first axis; for multi-D targets
        # we need ``C[:, :]`` so slice fusion emits a per-element loop
        # nest covering every axis (mandelbrot's
        # ``C = X + Y[:, None] * 1j`` is (yn, xn) and requires 2 loops).
        rank = len(shape_toks)
        if rank == 1:
            slice_form: ast.expr = ast.Slice(lower=None, upper=None, step=None)
        else:
            slice_form = ast.Tuple(
                elts=[ast.Slice(lower=None, upper=None, step=None)
                      for _ in range(rank)],
                ctx=ast.Load())
        slice_lhs = ast.Subscript(
            value=ast.Name(id=target.id, ctx=ast.Load()),
            slice=slice_form,
            ctx=ast.Store())
        slice_assign = ast.Assign(targets=[slice_lhs], value=node.value)
        return [marker, slice_assign]

    @staticmethod
    def _tokenise(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return str(node.value)
        if isinstance(node, ast.Name):
            return node.id
        return ast.unparse(node)

    @staticmethod
    def _has_slice_subscript(expr):
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Subscript):
                sl = sub.slice
                if isinstance(sl, ast.Slice):
                    return True
                if isinstance(sl, ast.Tuple) and any(
                        isinstance(e, ast.Slice) for e in sl.elts):
                    return True
        return False

    def _is_array_binop(self, expr):
        """``True`` for a BinOp / UnaryOp whose tree contains at least
        one bare-Name reference to an array of rank >= 1 in
        ``self.shapes``. Recognises forms like ``__cb1 = mass * vel``
        that the call-hoister synthesises -- the lifter then registers
        a fresh local with the broadcast extent so the per-element copy
        loop materialises the buffer.
        """
        if not isinstance(expr, (ast.BinOp, ast.UnaryOp)):
            return False
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Name):
                s = self.shapes.get(sub.id)
                if s:
                    return True
        return False


#: numpy functions / accessors that return a REAL value even from a COMPLEX
#: operand. The complex-detection walk must NOT descend into their arguments --
#: else ``np.abs(z)`` / ``np.real(z)`` / ``z.real`` read as complex and mis-drive
#: both the lifted-temp dtype (declaring a real magnitude ``complex128``) and the
#: intrinsic router (``csqrt`` on a real, wrong for a negative operand).
_REAL_FROM_COMPLEX: FrozenSet[str] = frozenset({"real", "imag", "abs", "absolute", "angle", "hypot", "sign"})


def _walk_complex(node: ast.AST, name_dtype: "Callable[[str], Optional[str]]") -> Optional[str]:
    """Return a complex dtype string if ``node`` produces a complex value, else
    ``None``. Call-AWARE: a ``np.<real-returning>(...)`` call (:data:`_REAL_FROM_COMPLEX`)
    or a ``.real`` / ``.imag`` accessor is REAL regardless of complex operands (its
    arguments are NOT walked); complex-preserving ops (``exp``/``sqrt``/``conj``/
    arithmetic) are complex iff an operand is. ``name_dtype(id)`` resolves a Name's
    element dtype. This is the single complex predicate for the lowering + emitters."""
    if isinstance(node, ast.Constant):
        return "complex128" if isinstance(node.value, complex) else None
    if isinstance(node, ast.Name):
        dt = name_dtype(node.id)
        return dt if dt and dt.startswith("complex") else None
    if isinstance(node, ast.Attribute):
        return None if node.attr in ("real", "imag") else _walk_complex(node.value, name_dtype)
    if isinstance(node, ast.Subscript):
        return _walk_complex(node.value, name_dtype)
    if isinstance(node, (ast.Compare, ast.BoolOp)):
        return None
    if isinstance(node, ast.BinOp):
        return _walk_complex(node.left, name_dtype) or _walk_complex(node.right, name_dtype)
    if isinstance(node, ast.UnaryOp):
        return _walk_complex(node.operand, name_dtype)
    if isinstance(node, ast.IfExp):
        return _walk_complex(node.body, name_dtype) or _walk_complex(node.orelse, name_dtype)
    if isinstance(node, ast.Call):
        fn = (node.func.attr if isinstance(node.func, ast.Attribute)
              else node.func.id if isinstance(node.func, ast.Name) else None)
        if fn in _REAL_FROM_COMPLEX:
            return None
        # An explicit complex ``dtype=`` (``np.zeros((n,), dtype=np.complex128)``)
        # or ``.astype(np.complex128)`` PRODUCES a complex value even when no
        # operand is complex -- the dtype lives in a KEYWORD, not node.args, so
        # inspect it directly. Without this a complex array whose only visible
        # write is its zero-init (vexx_k's ``deexx``) reads as real and the
        # complex->real narrowing pass unsoundly demotes it (compiles in C by
        # dropping the imaginary part, but C++ rejects the assignment).
        from numpyto_common.frontend import _dtype_from_constructor
        ctor_dt = _dtype_from_constructor(node)
        if ctor_dt is not None and ctor_dt.startswith("complex"):
            return ctor_dt
        for a in node.args:
            r = _walk_complex(a, name_dtype)
            if r:
                return r
        return None
    # Unhandled node type -- fall back to a conservative whole-subtree scan.
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, complex):
            return "complex128"
        if isinstance(sub, ast.Name):
            dt = name_dtype(sub.id)
            if dt and dt.startswith("complex"):
                return dt
    return None


def _infer_complex_dtype(expr: ast.AST,
                         local_dtypes: Dict[str, str]) -> Optional[str]:
    """Return a complex dtype string if ``expr`` produces a complex value, else
    ``None``. Delegates to the call-aware :func:`_walk_complex` so a real-returning
    ufunc / accessor of a complex operand is correctly REAL."""
    return _walk_complex(expr, local_dtypes.get)


#: The real element type underlying each complex width -- the inverse of the IR's
#: real->complex precision map, so a ``.real`` / ``.imag`` / ``abs`` / ``hypot``
#: scalar temp derived from a complex array is retagged to the matching real
#: width (never hardcoded: derived from ``ir._COMPLEX_FOR_FLOAT``, first real per
#: complex, so complex128->float64, complex64->float32, complex256->float128).
_REAL_FOR_COMPLEX: Dict[str, str] = {}
for _flt, _cplx in _COMPLEX_FOR_FLOAT.items():
    _REAL_FOR_COMPLEX.setdefault(_cplx, _flt)


def _is_conj_call(node: ast.AST) -> Optional[ast.expr]:
    """If ``node`` is a conjugation -- ``np.conj(x)`` / ``np.conjugate(x)`` (free
    function) or ``x.conjugate()`` (method) -- return its single operand ``x``;
    else ``None``. The operand is the value whose conjugate is taken."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return None
    f = node.func
    if (f.attr in ("conj", "conjugate") and isinstance(f.value, ast.Name)
            and f.value.id in ("np", "numpy") and len(node.args) == 1):
        return node.args[0]
    if f.attr == "conjugate" and not node.args:
        return f.value
    return None


class _RealConjDropper(ast.NodeTransformer):
    """Drop a conjugation applied to a provably-REAL operand: ``conj(x) -> x``
    when :func:`_walk_complex` classifies ``x`` real.

    numpy ``conj`` of a real is the identity, but Fortran ``CONJG`` requires a
    COMPLEX argument, so ``CONJG(<real>)`` is a compile error (the eigh /
    eigvalsh cyclic-Jacobi's ``ephi`` is real -- ``np.float64(apq) / m`` -- yet is
    wrapped in ``np.conj`` for the general Hermitian form). Removing the no-op
    conjugation on a real operand keeps both native backends valid; a genuinely
    complex operand keeps its conjugation."""

    def __init__(self, local_dtypes: Dict[str, str]):
        self.local_dtypes = local_dtypes

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        operand = _is_conj_call(node)
        if operand is not None and _walk_complex(operand, self.local_dtypes.get) is None:
            return operand
        return node


def _ctor_complex_tag(call: ast.Call, local_dtypes: Dict[str, str]) -> Optional[str]:
    """``np.zeros/ones/empty/eye(shape, <dtype>)`` -> a ``complexNN`` tag when the
    constructor's dtype arg is a complex array's ``Y.dtype`` or a bare
    ``np.complexNN`` (else None). The eigh reduction allocates its complex work
    matrices this way; without this they default to real."""
    if not (isinstance(call.func, ast.Attribute) and call.func.attr in ("zeros", "ones", "empty", "eye")):
        return None
    kw = {k.arg: k.value for k in call.keywords}
    da = kw.get("dtype")
    if da is None and call.func.attr != "eye" and len(call.args) > 1:
        da = call.args[1]
    if isinstance(da, ast.Attribute) and da.attr == "dtype" and isinstance(da.value, ast.Name):
        dt = local_dtypes.get(da.value.id)
        return dt if dt and dt.startswith("complex") else None
    if isinstance(da, ast.Attribute) and da.attr in ("complex128", "complex64"):
        return da.attr
    return None


def _scalar_expr_complex(expr: ast.AST, local_dtypes: Dict[str, str]) -> bool:
    """True iff a SCALAR arithmetic ``expr`` is complex, by its DIRECT operands
    (recursing only through BinOp/UnaryOp). It deliberately does NOT descend into
    ``.real``/``.imag`` accessors or calls (``hypot``/``abs`` produce a real from a
    complex), so ``tau = (aqq - app) / (2 * m)`` over real parts stays real while
    ``ephi = apq / m`` / ``acc = L[i, i] - L[i, i]`` over complex values is complex."""
    if isinstance(expr, ast.Constant):
        return isinstance(expr.value, complex)
    if isinstance(expr, ast.Name):
        return (local_dtypes.get(expr.id) or "").startswith("complex")
    if isinstance(expr, ast.Subscript):
        # Bottom out a subscript CHAIN (``deexx[:, ii][ikb]``) at its base Name --
        # the element dtype is the base array's, whatever indexing follows.
        base = expr.value
        while isinstance(base, ast.Subscript):
            base = base.value
        if isinstance(base, ast.Name):
            return (local_dtypes.get(base.id) or "").startswith("complex")
    if isinstance(expr, ast.BinOp):
        return _scalar_expr_complex(expr.left, local_dtypes) or _scalar_expr_complex(expr.right, local_dtypes)
    if isinstance(expr, ast.UnaryOp):
        return _scalar_expr_complex(expr.operand, local_dtypes)
    return False


def _seed_complex_work_dtypes(tree: ast.AST, local_dtypes: Dict[str, str]) -> None:
    """Seed ``local_dtypes`` for the complex work-array temps and their directly
    derived scalar reads, EARLY -- before the ``promote-true-division`` and
    ``libnode-expand`` phases consume those dtypes.

    The eigh / eigvalsh cyclic-Jacobi lowering allocates its complex work matrices
    (``L`` / ``Li`` / ``Tm`` / ``Cm`` / ``X`` / ``jv``) with ``np.zeros((n, n),
    b.dtype)`` and derives scalars off them (``apq = Cm[i, j]``, ``m =
    hypot(apq.real, apq.imag)``, ``ephi = apq / m``). Those dtypes are otherwise
    only recorded at the whole-array phase (:class:`_WholeArrayAssignRewriter`),
    which runs AFTER two phases that already consume them:

    * ``promote-true-division`` reads an untagged ``apq`` as an INTEGER (a non-array
      Name defaults integer), so ``apq / m`` is wrongly promoted with a
      ``np.float64(apq)`` cast -- truncating the complex Jacobi phase (and that cast
      is what C++ rejects as ``(double)(complex)``);
    * ``libnode-expand``'s :class:`_RealConjDropper` classifies the still-untyped
      complex work arrays as REAL via :func:`_walk_complex` and DROPS every
      ``np.conj`` on them, so the reduction computes the wrong eigenvalues.

    Seeding here (phase 4) makes both later phases see the true dtypes. It reuses
    the SAME predicates the whole-array rewriter uses (:func:`_ctor_complex_tag` /
    :func:`_scalar_expr_complex` / :func:`_walk_complex`), iterated to a fixpoint
    because a derived scalar's dtype depends on the temp it reads
    (``m`` / ``ephi`` <- ``apq`` <- ``Cm``'s constructor)."""
    assigns = [
        s for s in ast.walk(tree)
        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)
    ]

    def dtype_for(value: ast.expr) -> Optional[str]:
        if isinstance(value, ast.Call):
            # X = np.zeros/ones/empty/eye(shape, Y.dtype | np.complexNN)
            ctag = _ctor_complex_tag(value, local_dtypes)
            if ctag is not None:
                return ctag
            # X = Y.copy() / np.copy(Y) / np.ascontiguousarray(Y) -- inherit the complex source
            if isinstance(value.func, ast.Attribute):
                f = value.func
                src = (f.value.id if f.attr == "copy" and isinstance(f.value, ast.Name) else
                       value.args[0].id if f.attr in ("copy", "ascontiguousarray", "asarray", "array") and value.args
                       and isinstance(value.args[0], ast.Name) else None)
                sdt = local_dtypes.get(src) if src else None
                if sdt and sdt.startswith("complex"):
                    return sdt
            # m = np.hypot/abs/real/imag(<complex ...>) -- a real-returning magnitude of a
            # complex operand types to the MATCHING REAL width (so ``m`` is real, not complex).
            fn = (value.func.attr if isinstance(value.func, ast.Attribute) else
                  value.func.id if isinstance(value.func, ast.Name) else None)
            if fn in _REAL_FROM_COMPLEX:
                for sub in ast.walk(value):
                    if isinstance(sub, ast.Name):
                        bdt = local_dtypes.get(sub.id)
                        if bdt and bdt.startswith("complex"):
                            return _REAL_FOR_COMPLEX.get(bdt, "float64")
            return None
        # z = A[scalar-index] -- inherit a complex array's element dtype
        if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
            sdt = local_dtypes.get(value.value.id)
            return sdt if sdt and sdt.startswith("complex") else None
        # ephi = apq / m -- a scalar BinOp/UnaryOp over complex operands
        if isinstance(value, (ast.BinOp, ast.UnaryOp)) and _scalar_expr_complex(value, local_dtypes):
            return "complex128"
        return None

    changed = True
    while changed:
        changed = False
        for s in assigns:
            name = s.targets[0].id
            if name in local_dtypes:
                continue
            dt = dtype_for(s.value)
            if dt is not None:
                local_dtypes[name] = dt
                changed = True


class _PromoteMixedComplexIfExp(ast.NodeTransformer):
    """Make a mixed real/complex conditional's two branches the SAME type.

    ``d = z.real if gamma_only else z`` pairs a REAL branch (``.real`` strips the
    imaginary part) with a COMPLEX one. C promotes the real branch implicitly, but
    Fortran ``merge`` -- and the numba/pythran/jax type unifiers -- are strict and
    reject a real-vs-complex pair. Promote the real branch to complex with a
    cast-free ``+ 0j`` (a complex-literal add, NOT a C-style cast), so every
    backend sees a uniform-type select. Numerically identical: the promoted branch
    carries a zero imaginary part. (QE vexx ``_add_nlxx_pot`` gamma_only path.)"""

    def __init__(self, local_dtypes: Dict[str, str]):
        self.local_dtypes = local_dtypes

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.generic_visit(node)
        body_cplx = _scalar_expr_complex(node.body, self.local_dtypes)
        else_cplx = _scalar_expr_complex(node.orelse, self.local_dtypes)
        if body_cplx and not else_cplx:
            node.orelse = self._to_complex(node.orelse)
        elif else_cplx and not body_cplx:
            node.body = self._to_complex(node.body)
        return node

    @staticmethod
    def _to_complex(e: ast.expr) -> ast.expr:
        return ast.copy_location(
            ast.BinOp(left=e, op=ast.Add(), right=ast.Constant(value=0j)), e)


class _TupleSubscriptFolder(ast.NodeTransformer):
    """Fold ``(t1, t2, ..., tn)[K]`` to ``tk`` at lowering time so
    downstream passes don't see Tuple subscripts. Comes up when
    ``D.shape[-2]`` resolves to ``(Nqz, Nw, NA, NB, N3D, N3D)[-2]``
    after the shape harvest -- the Tuple is a constant-folded shape
    expression and the index picks one element."""

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Tuple):
            elts = node.value.elts
            idx_node = node.slice
            if (isinstance(idx_node, ast.Constant)
                    and isinstance(idx_node.value, int)):
                idx = idx_node.value
                if idx < 0:
                    idx += len(elts)
                if 0 <= idx < len(elts):
                    return elts[idx]
            if (isinstance(idx_node, ast.UnaryOp)
                    and isinstance(idx_node.op, ast.USub)
                    and isinstance(idx_node.operand, ast.Constant)
                    and isinstance(idx_node.operand.value, int)):
                idx = -idx_node.operand.value + len(elts)
                if 0 <= idx < len(elts):
                    return elts[idx]
        return node


_BLOCK_STMT_TYPES = (ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith,
                     ast.FunctionDef, ast.AsyncFunctionDef)


def _fill_empty_blocks(tree: ast.AST) -> None:
    """Back-fill a ``pass`` into any compound-statement ``body`` emptied by
    statement removal -- an empty ``for`` / ``while`` / ``if`` body is invalid
    Python and fails the next parse / compile. An empty ``orelse`` (no ``else``
    clause) is left as-is; only the primary body must be non-empty."""
    for node in ast.walk(tree):
        if isinstance(node, _BLOCK_STMT_TYPES):
            body = vars(node).get("body")
            if isinstance(body, list) and not body:
                filler = ast.Pass()
                ast.copy_location(filler, node)
                node.body = [filler]


class _TupleLocalPropagator(ast.NodeTransformer):
    """Forward-substitute a local bound exactly once to a Tuple literal into its
    uses, then drop the now-dead assignment.

    A native backend has no runtime tuple: a ``shp = Y.shape`` that
    :class:`_ShapeMidExpressionRewriter` folded to ``shp = (Lb, Lb, Lb, nstate)`` is
    a shape descriptor, and left as a bare Tuple assignment the emitter cannot lower
    it (and a ``np.reshape(x, shp)`` reading the bare Name would size ``x`` as a
    spurious 1-D ``(shp,)``). Inlining the tuple into ``shp[-1]`` and
    ``np.reshape(x, shp)`` -- which the tuple-subscript folder and reshape expander
    already lower -- resolves both.

    The substitution REPLAYS the element expressions (not a captured value) at each
    use site, so it is sound only when every name they read is stable. Three guards
    enforce that: (1) the tuple's own target is assigned exactly once; (2) every
    element is an integer-shape expression -- a Name, an ``int`` literal, or ``+ - *
    //`` arithmetic over those (a float / str constant or a Call is a genuine runtime
    value, not a shape, and is left alone); (3) every Name the elements read is itself
    assigned at most once in the scope (single-static-assign, hence one fixed value --
    a reassigned dim would make the replay pick up the wrong value). Dropping the dead
    assign never leaves an empty block: :func:`_fill_empty_blocks` back-fills a
    ``pass`` if the tuple assign was a block's sole statement.
    """

    def __init__(self):
        self.tuples: Dict[str, ast.Tuple] = {}

    @classmethod
    def _is_dim(cls, elt: ast.expr) -> bool:
        if isinstance(elt, ast.Name):
            return True
        if isinstance(elt, ast.Constant):
            # A bool is an int subclass but not a dimension; exclude it.
            return isinstance(elt.value, int) and not isinstance(elt.value, bool)
        if isinstance(elt, ast.UnaryOp):
            return cls._is_dim(elt.operand)
        if isinstance(elt, ast.BinOp):
            return cls._is_dim(elt.left) and cls._is_dim(elt.right)
        return False

    def run(self, tree: ast.AST) -> "_TupleLocalPropagator":
        store_counts: Dict[str, int] = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                store_counts[n.id] = store_counts.get(n.id, 0) + 1
        for stmt in ast.walk(tree):
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Tuple)
                    and store_counts.get(stmt.targets[0].id) == 1
                    and stmt.value.elts and all(self._is_dim(e) for e in stmt.value.elts)):
                continue
            target = stmt.targets[0].id
            reads = {n.id for e in stmt.value.elts for n in ast.walk(e) if isinstance(n, ast.Name)}
            # A reassigned element name is unstable; a self-referential tuple would
            # inline a Name whose defining assign we are about to drop.
            if target in reads or any(store_counts.get(name, 0) > 1 for name in reads):
                continue
            self.tuples[target] = stmt.value
        if self.tuples:
            self.visit(tree)
            _fill_empty_blocks(tree)
        return self

    def visit_Assign(self, node: ast.Assign) -> Optional[ast.AST]:
        self.generic_visit(node)
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in self.tuples):
            return None
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.tuples:
            return copy.deepcopy(self.tuples[node.id])
        return node


def _collect_bool_names(tree: ast.AST, arrays) -> Set[str]:
    """Names known to hold a boolean array.

    The conservative criterion that separates ``arr[bool_mask]`` (a masked
    select) from ``arr[int_idx]`` (an integer gather): only unambiguously
    boolean producers count, so an integer index array is never misread as a
    mask. A single forward pass over the body suffices because a mask is
    defined before it is used (``m = a > c``; later ``m2 = m & other``)."""
    bn: Set[str] = {a.name for a in arrays if a.dtype in ("bool", "bool_")}

    def _is_bool(e: ast.AST) -> bool:
        if isinstance(e, (ast.Compare, ast.BoolOp)):
            return True
        if isinstance(e, ast.Name):
            return e.id in bn
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Invert):
            return _is_bool(e.operand)
        if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
            return _is_bool(e.left) and _is_bool(e.right)
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) and isinstance(e.func.value, ast.Name) \
                and e.func.value.id == "np":
            if e.func.attr in ("logical_and", "logical_or", "logical_not", "logical_xor",
                               "isnan", "isinf", "isfinite", "greater", "greater_equal",
                               "less", "less_equal", "equal", "not_equal"):
                return True
            if e.func.attr in ("zeros", "ones", "empty", "full", "zeros_like", "ones_like"):
                for kw in e.keywords:
                    dv = kw.value
                    if kw.arg == "dtype" and ((isinstance(dv, ast.Attribute) and dv.attr in ("bool_", "bool"))
                                              or (isinstance(dv, ast.Name) and dv.id == "bool")):
                        return True
        return False

    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and _is_bool(node.value)):
            bn.add(node.targets[0].id)
    return bn


class _BooleanMaskReductionRewriter(ast.NodeTransformer):
    def __init__(self, shape_table=None, bool_names=None):
        self.shape_table = shape_table or {}
        self.bool_names = bool_names or set()

    """Peephole: rewrite ``tmp = arr[mask]; X = np.<reduction>(tmp)``
    into a single masked-iteration form that skips materialising the
    compacted view.

    Recognises ``np.mean`` / ``np.sum`` / ``np.max`` / ``np.min``.
    For ``mean`` the masked iteration tracks both sum and count;
    for ``sum`` only the sum is needed; for ``max``/``min`` the
    accumulator tracks the running extreme.

    Required because boolean fancy indexing (``arr[bool_mask]``)
    produces a dynamic-length compacted view that NumpyToC has no
    materialised representation for. By fusing the consumer into the
    same loop we avoid the dynamic shape.
    """

    def _walk_body(self, stmts):
        out = []
        i = 0
        while i < len(stmts):
            stmt = stmts[i]
            # Inline single-statement form ``X = np.<reduction>(arr[mask])``
            # (X a Name or Subscript LHS): the masked select is nested directly
            # in the reduction call rather than bound to a temp. Gate on a
            # KNOWN-boolean mask so an integer gather ``np.sum(a[idx])`` is left
            # for the gather materialiser instead of becoming a masked loop.
            inline = self._inline_masked_reduction(stmt)
            if inline is not None:
                arr, mask, op, tgt = inline
                if isinstance(tgt, ast.Name):
                    replacement = self._emit_masked(tgt.id, arr, mask, op)
                else:
                    scratch = f"__msk_res_{i}"
                    replacement = self._emit_masked(scratch, arr, mask, op)
                    if replacement is not None:
                        replacement = list(replacement) + [ast.Assign(
                            targets=[tgt], value=ast.Name(id=scratch, ctx=ast.Load()))]
                        ast.fix_missing_locations(replacement[-1])
                if replacement is not None:
                    out.extend(replacement)
                    i += 1
                    continue
            # ``Name = Subscript(Name(arr), Name(mask))`` followed by
            # ``Name2 = np.<reduction>(Name)``. Gated on a KNOWN-boolean mask
            # (like the inline form) so an integer-index gather ``t = a[idx];
            # s = t.sum()`` is left for the gather materialiser, not mis-lowered
            # into a mask-guarded accumulate loop.
            if (isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Subscript)
                    and isinstance(stmt.value.value, ast.Name)
                    and isinstance(stmt.value.slice, ast.Name)
                    and stmt.value.slice.id in self.bool_names
                    and i + 1 < len(stmts)):
                tmp_name = stmt.targets[0].id
                arr = stmt.value.value.id
                mask = stmt.value.slice.id
                nxt = stmts[i + 1]
                op = self._consumer_op(nxt, tmp_name)
                if op is not None:
                    # res_name can be a bare Name LHS or a Subscript
                    # LHS (e.g. ``res[i] = values_r12.mean()``).
                    tgt = nxt.targets[0]
                    if isinstance(tgt, ast.Name):
                        replacement = self._emit_masked(
                            tgt.id, arr, mask, op)
                        if replacement is not None:
                            out.extend(replacement)
                            i += 2
                            continue
                    elif isinstance(tgt, ast.Subscript):
                        # Emit the masked compute into a scratch and
                        # then assign to the Subscript LHS.
                        scratch = f"__msk_res_{i}"
                        replacement = self._emit_masked(
                            scratch, arr, mask, op)
                        if replacement is not None:
                            out.extend(replacement)
                            out.append(ast.Assign(
                                targets=[tgt],
                                value=ast.Name(id=scratch, ctx=ast.Load())))
                            ast.fix_missing_locations(out[-1])
                            i += 2
                            continue
            # Recurse into nested compound bodies.
            for attr in ("body", "orelse"):
                if hasattr(stmt, attr) and isinstance(getattr(stmt, attr), list):
                    setattr(stmt, attr, self._walk_body(getattr(stmt, attr)))
            out.append(stmt)
            i += 1
        return out

    def _inline_masked_reduction(self, stmt):
        """Detect ``X = np.<reduction>(arr[mask])`` / ``X = arr[mask].<reduction>()``
        as a single statement with a KNOWN-boolean ``mask``.

        Returns ``(arr, mask, op, target)`` or ``None``. ``target`` is the LHS
        (a Name or Subscript). The masked select is the sole reduction argument
        (``np.sum``) or the call receiver (``.sum()``)."""
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            return None
        tgt = stmt.targets[0]
        if not isinstance(tgt, (ast.Name, ast.Subscript)):
            return None
        call = stmt.value
        if not isinstance(call, ast.Call):
            return None
        func = call.func
        sel = None
        op = None
        # Form ``np.<op>(arr[mask])``.
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id == "np" and func.attr in {"mean", "sum", "max", "min"}
                and len(call.args) == 1 and not call.keywords):
            sel, op = call.args[0], func.attr
        # Form ``arr[mask].<op>()``.
        elif (isinstance(func, ast.Attribute) and func.attr in {"mean", "sum", "max", "min"}
                and not call.args and not call.keywords):
            sel, op = func.value, func.attr
        if (isinstance(sel, ast.Subscript) and isinstance(sel.value, ast.Name)
                and isinstance(sel.slice, ast.Name) and sel.slice.id in self.bool_names):
            return sel.value.id, sel.slice.id, op, tgt
        return None

    def _consumer_op(self, stmt, expected_name):
        """Detect a reduction consumer of ``expected_name`` in ``stmt``.

        Recognises three forms:
        1. Bare ``Name = np.<reduction>(Name(expected_name))``
        2. Bare ``Name = Name(expected_name).<reduction>()``
        3. ``Subscript = ...`` of either form above (treat as Name LHS
           when the Subscript is on a scalar / bare-Name target).

        Returns the reduction op string ("mean" / "sum" / "max" /
        "min") and the resolved result Name; else None.
        """
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            return None
        if not isinstance(stmt.value, ast.Call):
            return None
        call = stmt.value
        if not (isinstance(call.args, list) and len(call.args) == 0
                or (len(call.args) == 1
                    and isinstance(call.args[0], ast.Name)
                    and call.args[0].id == expected_name)):
            return None
        func = call.func
        # Form 1: np.<op>(...)
        if (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "np"
                and func.attr in {"mean", "sum", "max", "min"}
                and len(call.args) == 1):
            return func.attr
        # Form 2: arr.<op>()
        if (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == expected_name
                and func.attr in {"mean", "sum", "max", "min"}
                and len(call.args) == 0):
            return func.attr
        return None

    def _emit_masked(self, res_name, arr, mask, op):
        i_name = f"__msk_i_{res_name}"
        sum_name = f"__msk_acc_{res_name}"
        cnt_name = f"__msk_cnt_{res_name}"
        # Resolve ``arr``'s first-dim extent via the shape table if
        # available; fall back to ``len(arr)`` which the C emit
        # may translate via a length local.
        arr_shape = self.shape_table.get(arr) or self.shape_table.get(mask)
        if arr_shape:
            n_expr = self._tok_to_ast(arr_shape[0])
        else:
            n_expr = ast.Call(
                func=ast.Name(id="len", ctx=ast.Load()),
                args=[ast.Name(id=arr, ctx=ast.Load())], keywords=[])
        mask_load = ast.Subscript(
            value=ast.Name(id=mask, ctx=ast.Load()),
            slice=ast.Name(id=i_name, ctx=ast.Load()), ctx=ast.Load())
        arr_load = ast.Subscript(
            value=ast.Name(id=arr, ctx=ast.Load()),
            slice=ast.Name(id=i_name, ctx=ast.Load()), ctx=ast.Load())
        out: List[ast.stmt] = []
        if op == "mean":
            out.append(ast.Assign(
                targets=[ast.Name(id=sum_name, ctx=ast.Store())],
                value=ast.Constant(value=0.0)))
            out.append(ast.Assign(
                targets=[ast.Name(id=cnt_name, ctx=ast.Store())],
                value=ast.Constant(value=0)))
            body = [
                ast.AugAssign(target=ast.Name(id=sum_name, ctx=ast.Store()),
                               op=ast.Add(), value=arr_load),
                ast.AugAssign(target=ast.Name(id=cnt_name, ctx=ast.Store()),
                               op=ast.Add(), value=ast.Constant(value=1)),
            ]
            out.append(ast.For(
                target=ast.Name(id=i_name, ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[n_expr], keywords=[]),
                body=[ast.If(test=mask_load, body=body, orelse=[])],
                orelse=[]))
            out.append(ast.Assign(
                targets=[ast.Name(id=res_name, ctx=ast.Store())],
                value=ast.BinOp(
                    left=ast.Name(id=sum_name, ctx=ast.Load()),
                    op=ast.Div(),
                    right=ast.Name(id=cnt_name, ctx=ast.Load()))))
        elif op == "sum":
            out.append(ast.Assign(
                targets=[ast.Name(id=res_name, ctx=ast.Store())],
                value=ast.Constant(value=0.0)))
            body = [ast.AugAssign(
                target=ast.Name(id=res_name, ctx=ast.Store()),
                op=ast.Add(), value=arr_load)]
            out.append(ast.For(
                target=ast.Name(id=i_name, ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[n_expr], keywords=[]),
                body=[ast.If(test=mask_load, body=body, orelse=[])],
                orelse=[]))
        elif op in {"max", "min"}:
            # The FIRST masked hit seeds the accumulator; subsequent hits
            # compare and update. Seeding from ``arr[0]`` unconditionally would
            # be wrong when index 0 is masked out and more extreme than every
            # masked value -- so a ``seen`` flag guards the seed instead.
            cmp = ast.Gt() if op == "max" else ast.Lt()
            out.append(ast.Assign(
                targets=[ast.Name(id=res_name, ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Name(id=arr, ctx=ast.Load()),
                    slice=ast.Constant(value=0), ctx=ast.Load())))
            out.append(ast.Assign(
                targets=[ast.Name(id=cnt_name, ctx=ast.Store())],
                value=ast.Constant(value=0)))
            update = ast.If(
                test=ast.BoolOp(op=ast.Or(), values=[
                    ast.Compare(left=ast.Name(id=cnt_name, ctx=ast.Load()),
                                ops=[ast.Eq()], comparators=[ast.Constant(value=0)]),
                    ast.Compare(left=arr_load, ops=[cmp],
                                comparators=[ast.Name(id=res_name, ctx=ast.Load())])]),
                body=[ast.Assign(targets=[ast.Name(id=res_name, ctx=ast.Store())], value=arr_load)],
                orelse=[])
            body = [update, ast.Assign(
                targets=[ast.Name(id=cnt_name, ctx=ast.Store())], value=ast.Constant(value=1))]
            out.append(ast.For(
                target=ast.Name(id=i_name, ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[n_expr], keywords=[]),
                body=[ast.If(test=mask_load, body=body, orelse=[])],
                orelse=[]))
        else:
            return None
        for s in out:
            ast.fix_missing_locations(s)
        return out

    def _tok_to_ast(self, tok):
        """Parse a shape token like ``'N'`` or ``'N - 1'`` to an AST."""
        try:
            return ast.parse(str(tok), mode="eval").body
        except SyntaxError:
            return ast.Name(id=str(tok), ctx=ast.Load())

    def visit_FunctionDef(self, node):
        node.body = self._walk_body(node.body)
        return node


class _ShapeAttrToReshape(ast.NodeTransformer):
    """Rewrite in-place shape mutation ``x.shape = expr`` to
    ``x = np.reshape(x, expr)``.

    Numpy allows setting ``arr.shape = (N,)`` as an in-place view-
    reshape that returns the same data with a new shape. NumpyToC has
    no in-place attribute writes; the existing ``expand_reshape`` path
    already handles ``x = np.reshape(x, ...)`` so we just rewrite the
    LHS form into the function-call form.

    Also handles the chained form ``Xi.shape = Yi.shape = expr`` by
    splitting into per-target reshapes (mandelbrot2 canonical pattern).
    """

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        # Detect ``x.shape = expr`` -- one or more LHS targets that are
        # all ``Attribute(Name(x), 'shape')``.
        if not node.targets:
            return node
        shape_targets: List[ast.Name] = []
        for tgt in node.targets:
            if (isinstance(tgt, ast.Attribute)
                    and tgt.attr == "shape"
                    and isinstance(tgt.value, ast.Name)):
                shape_targets.append(tgt.value)
            else:
                return node
        if not shape_targets:
            return node
        # Normalise the new-shape expression: a bare integer ``N`` is
        # treated as ``(N,)`` (numpy quirk: arr.shape = N is a valid
        # 1-D reshape). A tuple stays as-is.
        new_shape = node.value
        if not isinstance(new_shape, ast.Tuple):
            new_shape = ast.Tuple(elts=[new_shape], ctx=ast.Load())
        # Emit one Assign per target ``x = np.reshape(x, new_shape)``.
        out: List[ast.stmt] = []
        for name in shape_targets:
            call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr="reshape", ctx=ast.Load()),
                args=[ast.Name(id=name.id, ctx=ast.Load()),
                      copy.deepcopy(new_shape)],
                keywords=[])
            out.append(ast.Assign(
                targets=[ast.Name(id=name.id, ctx=ast.Store())],
                value=call))
        for s in out:
            ast.fix_missing_locations(s)
        return out


class _MgridLowering(ast.NodeTransformer):
    """Lower ``X0, X1, ... = np.mgrid[a0:b0, a1:b1, ...]`` to a
    sequence of ``Xk = np.zeros(shape, dtype=np.int64)`` markers
    plus one per-element init loop per axis.

    ``np.mgrid[0:R, 0:S]`` returns two 2-D arrays of shape (R, S)::

        I[i, j] = i
        J[i, j] = j

    NumpyToC has no shape-mutating ``mgrid`` object; we expand it
    eagerly into the per-element initialisers and emit `np.empty`
    declarations whose shape harvest then picks them up like any
    other local array.
    """

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if not (len(node.targets) == 1
                and isinstance(node.targets[0], ast.Tuple)):
            return node
        rhs = node.value
        if not (isinstance(rhs, ast.Subscript)
                and isinstance(rhs.value, ast.Attribute)
                and isinstance(rhs.value.value, ast.Name)
                and rhs.value.value.id == "np"
                and rhs.value.attr == "mgrid"):
            return node
        # Collect target Names and slice axes.
        targets = node.targets[0].elts
        if not all(isinstance(t, ast.Name) for t in targets):
            return node
        sl = rhs.slice
        if isinstance(sl, ast.Tuple):
            axes = list(sl.elts)
        else:
            axes = [sl]
        if len(axes) != len(targets) or not all(
                isinstance(a, ast.Slice) for a in axes):
            return node
        # Compute shape tuple: (b0 - a0, b1 - a1, ...).
        shape_elts: List[ast.expr] = []
        for ax in axes:
            lo = ax.lower if ax.lower is not None else ast.Constant(value=0)
            hi = ax.upper
            if hi is None:
                return node
            shape_elts.append(ast.BinOp(left=hi, op=ast.Sub(), right=lo))
        shape_tuple = ast.Tuple(elts=shape_elts, ctx=ast.Load())
        # For each target, emit:
        #   Tk = np.empty(shape, dtype=np.int64)
        #   for i0 in range(b0 - a0):
        #       ...
        #         Tk[i0, i1, ...] = ik + ak
        out: List[ast.stmt] = []
        iters = [ast.Name(id=f"__mg{k}", ctx=ast.Load())
                 for k in range(len(targets))]
        for k, tgt in enumerate(targets):
            # Allocator: ``Tk = np.empty(shape, dtype=np.int64)``.
            out.append(ast.Assign(
                targets=[ast.Name(id=tgt.id, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()),
                                       attr="empty", ctx=ast.Load()),
                    args=[shape_tuple],
                    keywords=[ast.keyword(
                        arg="dtype",
                        value=ast.Attribute(
                            value=ast.Name(id="np", ctx=ast.Load()),
                            attr="int64", ctx=ast.Load()))])))
            # Init body: ``Tk[iters] = iter[k] + axis[k].lower``.
            lo_k = axes[k].lower if axes[k].lower is not None \
                else ast.Constant(value=0)
            idx_expr: ast.expr = ast.Name(id=iters[k].id, ctx=ast.Load())
            if not (isinstance(lo_k, ast.Constant) and lo_k.value == 0):
                idx_expr = ast.BinOp(left=idx_expr, op=ast.Add(), right=lo_k)
            slice_form = (iters[0] if len(iters) == 1
                          else ast.Tuple(
                              elts=[ast.Name(id=it.id, ctx=ast.Load())
                                    for it in iters], ctx=ast.Load()))
            body = [ast.Assign(
                targets=[ast.Subscript(
                    value=ast.Name(id=tgt.id, ctx=ast.Load()),
                    slice=slice_form, ctx=ast.Store())],
                value=idx_expr)]
            # Wrap the body in nested loops, deepest first.
            stmt: List[ast.stmt] = body
            for it, ax in zip(reversed(iters), reversed(axes)):
                ax_lo = ax.lower if ax.lower is not None else ast.Constant(value=0)
                ax_hi = ax.upper
                bound = (ax_hi if isinstance(ax_lo, ast.Constant)
                         and ax_lo.value == 0
                         else ast.BinOp(left=ax_hi, op=ast.Sub(), right=ax_lo))
                stmt = [ast.For(
                    target=ast.Name(id=it.id, ctx=ast.Store()),
                    iter=ast.Call(
                        func=ast.Name(id="range", ctx=ast.Load()),
                        args=[bound], keywords=[]),
                    body=stmt, orelse=[])]
            out.extend(stmt)
        return out


def _is_constructor_call(node: ast.Call) -> bool:
    """``True`` for ``np.zeros / np.empty / np.ones / np.full /
    np.zeros_like / np.empty_like`` -- the constructors whose
    semantics is allocation, NOT a shape-preserving elementwise op.
    Used by the whole-array rewriter to refuse expanding these
    forms into per-element loops."""
    if not (isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"):
        return False
    return node.func.attr in {
        "zeros", "empty", "ones", "full", "ndarray",
        "zeros_like", "empty_like", "ones_like", "full_like",
        "linspace", "arange", "eye", "identity", "mgrid",
        # Shape-changing ops: NOT elementwise. The dedicated
        # ``expand_reshape / expand_repeat / expand_transpose``
        # paths handle these via the LibNodeRewriter.
        "reshape", "repeat", "transpose",
    }


def _normalise_shape(shape) -> Tuple[str, ...]:
    """Normalise shape tokens for structural comparison.

    Tokens like ``"H + 2"`` and ``"(H + 2)"`` denote the same extent
    but mismatch on a plain ``==``. Re-parse compound tokens via
    :func:`ast.parse` (mode='eval') and unparse back -- the result
    is canonical Python syntax that compares correctly regardless of
    the original wrapper parens.
    """
    out = []
    for tok in shape:
        try:
            if tok and not str(tok).isdigit() and not str(tok).isidentifier():
                parsed = ast.parse(str(tok), mode="eval").body
                out.append(ast.unparse(parsed))
                continue
        except (SyntaxError, ValueError):
            pass
        out.append(str(tok))
    return tuple(out)


def _is_bool_expr(node: ast.AST, local_dtypes: Dict[str, str]) -> bool:
    """True when ``node`` evaluates to a BOOLEAN (array): a comparison, a
    boolean connective (``and``/``or``/``not``), a bitwise ``& | ^`` of boolean
    operands, or a reference to a boolean-typed array. Used to type a derived
    local array (``mask = cfl_clip & owner``) as bool on every backend."""
    if isinstance(node, (ast.Compare, ast.BoolOp)):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
        return _is_bool_expr(node.left, local_dtypes) and _is_bool_expr(node.right, local_dtypes)
    # ``~x`` inverts a boolean MASK (logical negation) when x is boolean; on an
    # integer operand it is bitwise NOT and stays integer.
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return _is_bool_expr(node.operand, local_dtypes)
    if isinstance(node, ast.Name):
        return local_dtypes.get(node.id) in ("bool", "bool_")
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        return local_dtypes.get(node.value.id) in ("bool", "bool_")
    return False


def _has_index_array(node: ast.Subscript, shape_table) -> bool:
    """True when a Subscript uses at least one integer index-ARRAY index
    (advanced indexing) -- ``u2[q, r, s]`` with q/r/s in the shape table, or
    ``A[B[i]]``. Distinguishes a fancy gather (whole-array-expandable) from a
    plain slice / scalar read."""
    if not isinstance(node, ast.Subscript):
        return False
    sl = node.slice
    elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
    for e in elts:
        if isinstance(e, ast.Name) and e.id in shape_table:
            return True
        if isinstance(e, ast.Subscript) and isinstance(e.value, ast.Name) \
                and e.value.id in shape_table:
            return True
    return False


def _np_func_call(value: ast.AST, name: str) -> Optional[ast.Call]:
    """Return ``value`` when it is ``np.<name>(...)`` / ``numpy.<name>(...)``,
    else ``None`` -- used to recognise ``np.ix_`` / ``np.meshgrid`` calls."""
    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
            and value.func.attr == name and isinstance(value.func.value, ast.Name)
            and value.func.value.id in ("np", "numpy")):
        return value
    return None


def _ix_call_args(value: ast.AST) -> Optional[List[ast.expr]]:
    """The open-mesh index arrays of an ``np.ix_(a, b, c)`` call, else ``None``."""
    call = _np_func_call(value, "ix_")
    return list(call.args) if call is not None else None


class _WholeArrayAssignRewriter(ast.NodeTransformer):
    """Turn whole-array Assign / AugAssign between named arrays into
    per-element loops.

    Examples (where ``shape_table[name]`` returns the array shape)::

        x1 += __mm1                ->  for i in range(N): x1[i] += __mm1[i]
        out = a                    ->  for i in range(N): out[i] = a[i]

    Without this, the emitter would render ``x1 += __mm1`` as pointer
    arithmetic in C and as undefined Fortran.
    """

    def __init__(self, shape_table, real_arrays=None, local_dtypes=None):
        # We mutate ``shape_table`` to track Name aliases per Assign in
        # source order. Use a local copy so the caller's table is not
        # repeatedly clobbered when an alias gets reassigned.
        self.shape_table = dict(shape_table)
        #: Keys present in the caller's table at entry -- anything the pass adds
        #: beyond these (a meshgrid output, a ``gsq = gx**2 + ...`` broadcast local)
        #: is a genuinely NEW local whose shape the later slice-fusion pass needs;
        #: see :attr:`discovered_shapes`.
        self._input_keys = set(shape_table)
        #: Shared dtype tag table. Alias / BinOp expansions propagate
        #: dtype here so the emitter sees the right C type for a
        #: complex-RHS local that was never directly declared.
        self.local_dtypes: Dict[str, str] = (
            local_dtypes if local_dtypes is not None else {})
        #: New local array names introduced by alias propagation
        #: (``x = __cb2`` where ``x`` wasn't previously an array) --
        #: emitter must declare them as stack arrays.
        self.alias_locals: Dict[str, Tuple[str, ...]] = {}
        #: Monotonic id for the buffered fancy ``A[idx] += rhs`` snapshot temps.
        self._scatter_ctr = 0
        #: Per-name list of shapes recorded in source order, one entry
        #: per ``Name = expr`` reassignment that the rewriter
        #: expanded to a per-element loop nest. Consumed by the
        #: source-order shape resolver so reassigned locals (lenet's
        #: ``x = relu(...); x = maxpool2d(x); ...``) carry the
        #: THEN-current shape at each ``arr.shape[i]`` reference.
        self._reassign_shapes: Dict[str, List[Tuple[str, ...]]] = {}
        # Names that existed as declared kernel arrays before any
        # alias / matmul-hoist temps were synthesised. When the LHS of
        # a whole-array alias is not in this set we register it as a
        # fresh local for the emitter to declare.
        self._known_arrays = set(real_arrays) if real_arrays else set(shape_table.keys())
        #: ``grid = np.ix_(a, b, c)`` bindings: grid name -> the open-mesh index
        #: arrays, resolved at each ``A[grid]`` gather / ``A[grid] (+)= rhs``
        #: scatter use site (numpy advanced open-mesh indexing).
        self._ix_grids: Dict[str, List[ast.expr]] = {}
        #: Monotonic id for open-mesh (``np.ix_``) gather / scatter loop iters.
        self._ix_ctr = 0

    @property
    def discovered_shapes(self) -> Dict[str, Tuple[str, ...]]:
        """Shapes the pass inferred for locals the caller's table did not yet know
        (meshgrid outputs, broadcast-BinOp locals such as ``gsq``). Merged back into
        the shared table so the downstream slice-fusion scalarizer sizes them, rather
        than treating a shapeless denominator (``rho_g / gsq``) as a bare pointer."""
        return {k: tuple(v) for k, v in self.shape_table.items()
                if k not in self._input_keys and v}

    def _expand(self, target: ast.Name, value: ast.expr,
                op: Optional[ast.AST] = None,
                ) -> List[ast.stmt]:
        shape = self.shape_table.get(target.id)
        if not shape:
            return []
        iters = [f"__w{i}" for i in range(len(shape))]
        idx = (ast.Name(id=iters[0], ctx=ast.Load()) if len(iters) == 1 else
               ast.Tuple(elts=[ast.Name(id=i, ctx=ast.Load()) for i in iters],
                         ctx=ast.Load()))
        lhs_sub = ast.Subscript(value=ast.Name(id=target.id, ctx=ast.Load()),
                                slice=idx, ctx=ast.Store())
        # Replace any Name(value) whose shape matches with a per-element
        # subscript; scalars pass through.
        rhs = _SubscriptifyNames(self.shape_table, iters).visit(copy.deepcopy(value))
        if op is None:
            body = [ast.Assign(targets=[lhs_sub], value=rhs)]
        else:
            body = [ast.AugAssign(target=lhs_sub, op=op, value=rhs)]
        out = body
        for var, bound in zip(reversed(iters), reversed(shape)):
            out = [ast.For(
                target=ast.Name(id=var, ctx=ast.Store()),
                iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                              args=[_token_to_ast(bound)],
                              keywords=[]),
                body=out, orelse=[])]
        # Prepend a ``Name = __optarena_zeros__("__reassign__")`` marker so
        # the source-order shape resolver (``_ResolveArrShape``) can pick
        # up the THEN-current shape of the LHS. The marker is a no-op at
        # emit-time -- the emitter declares the LHS once at the function
        # start. The ``"__reassign__"`` sentinel distinguishes it from a
        # genuine ``np.zeros(...)`` reset: a reassignment is immediately
        # followed by a per-element loop that FULLY overwrites the LHS, so
        # the emitter must NOT re-zero (memset) the buffer first -- doing
        # so corrupts a self-referential reassignment like bicgstab's
        # ``p = r + beta * (p - omega * v)`` (the loop reads the old p).
        if op is None:
            marker = ast.Assign(
                targets=[ast.Name(id=target.id, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="__optarena_zeros__", ctx=ast.Load()),
                    args=[ast.Constant(value="__reassign__")], keywords=[]))
            self._reassign_shapes.setdefault(target.id, []).append(tuple(shape))
            out = [marker] + out
        return out

    def _expand_partial(self, target: ast.Subscript,
                        value: ast.Name) -> List[ast.stmt]:
        """Lower ``A[i] = B`` where ``A[i]`` is a PARTIAL subscript (a row /
        sub-array, fewer integer indices than ``A``'s rank) and ``B`` is a
        whole array of the remaining shape, into a per-element copy loop over
        the trailing dims. This is the ``back[t] = np.argmax(scores, axis=0)``
        pattern after the reduction is hoisted to a temp ``B`` -- without the
        expansion the emitter renders ``A[i] = B`` as a pointer store.
        """
        name = target.value.id
        shape = self.shape_table.get(name)
        if not shape:
            return []
        sl = target.slice
        given = list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]
        if any(isinstance(g, ast.Slice) for g in given):
            return []   # a Slice index is whole-array, handled elsewhere
        k = len(given)
        if k >= len(shape):
            return []   # fully indexed -> a scalar store, not a row copy
        remaining = shape[k:]
        rhs_shape = self.shape_table.get(value.id)
        if not rhs_shape or len(rhs_shape) != len(remaining):
            return []
        iters = [f"__w{i}" for i in range(len(remaining))]
        full_idx = list(given) + [ast.Name(id=i, ctx=ast.Load()) for i in iters]
        lhs_sub = ast.Subscript(
            value=ast.Name(id=name, ctx=ast.Load()),
            slice=ast.Tuple(elts=full_idx, ctx=ast.Load()), ctx=ast.Store())
        rhs = _SubscriptifyNames(self.shape_table, iters).visit(
            copy.deepcopy(value))
        out: List[ast.stmt] = [ast.Assign(targets=[lhs_sub], value=rhs)]
        for var, bound in zip(reversed(iters), reversed(remaining)):
            out = [ast.For(
                target=ast.Name(id=var, ctx=ast.Store()),
                iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                              args=[_token_to_ast(bound)], keywords=[]),
                body=out, orelse=[])]
        return out

    def _expand_fancy_scatter_store(self, target: ast.Subscript,
                                    value: ast.expr, op) -> List[ast.stmt]:
        """Lower a fancy-index scatter store ``A[idx, c] (op)= rhs`` where one
        index component is an INDEX ARRAY (``idx``) and the rest are scalars,
        into a per-element loop ``for k: A[idx[k], c] (op)= rhs[k]``.

        The C/Fortran emitter has no notion of array-valued subscripts, so a
        raw ``facb[nl] = v`` / ``tg[nl, 0] = psi[:, i]`` would emit an invalid
        ``arr[ptr] = ...``. ``idx`` (the lone index-array component) gives the
        trip count; the RHS is scalarised at the loop iter."""
        if not isinstance(target.value, ast.Name):
            return []
        name = target.value.id
        if name not in self.shape_table:
            return []
        lead = (list(target.slice.elts) if isinstance(target.slice, ast.Tuple)
                else [target.slice])
        if any(isinstance(e, ast.Slice) for e in lead):
            return []
        arr_pos = [k for k, e in enumerate(lead)
                   if isinstance(e, ast.Name) and e.id != name
                   and len(self.shape_table.get(e.id, ())) == 1]
        if len(arr_pos) != 1:
            return []
        p = arr_pos[0]
        idx_name = lead[p].id
        extent = self.shape_table[idx_name][0]
        it = "__sc0"
        new_lead = list(lead)
        new_lead[p] = ast.Subscript(value=ast.Name(id=idx_name, ctx=ast.Load()),
                                    slice=ast.Name(id=it, ctx=ast.Load()),
                                    ctx=ast.Load())
        lhs_slice = (new_lead[0] if len(new_lead) == 1
                     else ast.Tuple(elts=new_lead, ctx=ast.Load()))
        lhs = ast.Subscript(value=ast.Name(id=name, ctx=ast.Load()),
                            slice=lhs_slice, ctx=ast.Store())
        rhs = _SubscriptifyNames(self.shape_table, [it]).visit(copy.deepcopy(value))

        def _loop(body_stmt: ast.stmt) -> ast.For:
            f = ast.For(target=ast.Name(id=it, ctx=ast.Store()),
                        iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                                      args=[_token_to_ast(extent)], keywords=[]),
                        body=[body_stmt], orelse=[])
            return f

        if op is None:
            # Plain fancy store ``A[idx, c] = rhs`` -- a single per-element loop.
            # Sequential last-write-wins on a repeated index equals numpy's
            # buffered fancy assignment, so no snapshot is needed.
            out: List[ast.stmt] = [_loop(ast.Assign(targets=[lhs], value=rhs))]
        else:
            # numpy fancy ``A[idx] += rhs`` is BUFFERED: it reads the OLD A[idx],
            # applies the op against rhs, and scatters back with LAST-WRITE-WINS for
            # a repeated index -- it does NOT accumulate (that is ``np.add.at``,
            # routed elsewhere). Snapshot the gathered old values into a temp, then
            # store, so a duplicate index matches numpy (a single in-place ``+=``
            # loop would over-count). The snapshot is a rank-1 local of A's dtype.
            gname = f"__scg{self._scatter_ctr}"
            self._scatter_ctr += 1
            self.alias_locals[gname] = (extent, )
            if name in self.local_dtypes:
                self.local_dtypes[gname] = self.local_dtypes[name]
            g_store = ast.Subscript(value=ast.Name(id=gname, ctx=ast.Load()),
                                    slice=ast.Name(id=it, ctx=ast.Load()), ctx=ast.Store())
            g_load = ast.Subscript(value=ast.Name(id=gname, ctx=ast.Load()),
                                   slice=ast.Name(id=it, ctx=ast.Load()), ctx=ast.Load())
            a_load = ast.Subscript(value=ast.Name(id=name, ctx=ast.Load()),
                                   slice=copy.deepcopy(lhs_slice), ctx=ast.Load())
            gather = _loop(ast.Assign(targets=[g_store], value=a_load))
            store = _loop(ast.Assign(targets=[copy.deepcopy(lhs)],
                                     value=ast.BinOp(left=g_load, op=op, right=rhs)))
            out = [gather, store]
        for s in out:
            ast.fix_missing_locations(s)
        return out

    def _resolve_ix_operands(self, sub: ast.Subscript) -> Optional[List[ast.expr]]:
        """The open-mesh index arrays of a ``A[grid]`` / ``A[np.ix_(...)]``
        subscript -- resolving a bound ``grid = np.ix_(...)`` (recorded in
        :attr:`_ix_grids`) or an inline call -- else ``None``."""
        idx = sub.slice
        if isinstance(idx, ast.Name) and idx.id in self._ix_grids:
            return self._ix_grids[idx.id]
        return _ix_call_args(idx)

    def _ix_dims(self, ops: List[ast.expr]) -> Optional[List[ast.expr]]:
        """Per-operand length for a list of 1-D index arrays (the open-mesh /
        meshgrid axis extents), or ``None`` when any operand's shape is unknown
        or not rank-1."""
        dims: List[ast.expr] = []
        for op in ops:
            ext = _iter_extent_of(op, self.shape_table)
            if ext is None or len(ext) != 1:
                return None
            dims.append(ext[0])
        return dims

    def _empty_alloc(self, name: str, dims: List[ast.expr]) -> ast.Assign:
        """``name = np.empty((d0, d1, ...))`` marker so the zeros harvest declares
        the fresh gather / meshgrid output; its element type rides on
        :attr:`local_dtypes`."""
        shape_tuple = ast.Tuple(elts=[copy.deepcopy(d) for d in dims], ctx=ast.Load())
        return ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()),
                                   attr="empty", ctx=ast.Load()),
                args=[shape_tuple], keywords=[]))

    def _ix_iters(self, prefix: str, k: int) -> List[str]:
        self._ix_ctr += 1
        return [f"{prefix}{self._ix_ctr}_{d}" for d in range(k)]

    def _expand_ix_gather(self, target: ast.Name, arr: ast.Name,
                          ops: List[ast.expr]) -> Optional[List[ast.stmt]]:
        """``vloc = A[np.ix_(a, b, c)]`` -> ``vloc[i,j,k] = A[a[i], b[j], c[k]]``.

        The result has shape ``(len(a), len(b), len(c))`` (open-mesh gather);
        ``A`` is read at the Cartesian product of the index arrays."""
        dims = self._ix_dims(ops)
        if dims is None:
            return None
        k = len(ops)
        iters = self._ix_iters("__ixg", k)
        read = [_scalarize_at_iters(copy.deepcopy(op), [ast.Name(id=it, ctx=ast.Load())],
                                    self.shape_table)
                for op, it in zip(ops, iters)]
        read_slot = read[0] if k == 1 else ast.Tuple(elts=read, ctx=ast.Load())
        src = ast.Subscript(value=ast.Name(id=arr.id, ctx=ast.Load()),
                            slice=read_slot, ctx=ast.Load())
        out_slot = (ast.Name(id=iters[0], ctx=ast.Load()) if k == 1
                    else ast.Tuple(elts=[ast.Name(id=it, ctx=ast.Load()) for it in iters],
                                   ctx=ast.Load()))
        store = ast.Subscript(value=ast.Name(id=target.id, ctx=ast.Load()),
                              slice=out_slot, ctx=ast.Store())
        body: List[ast.stmt] = [ast.Assign(targets=[store], value=src)]
        for it, dim in zip(reversed(iters), reversed(dims)):
            body = [ast.For(
                target=ast.Name(id=it, ctx=ast.Store()),
                iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                              args=[copy.deepcopy(dim)], keywords=[]),
                body=body, orelse=[])]
        self.shape_table[target.id] = tuple(ast.unparse(d) for d in dims)
        dt = self.local_dtypes.get(arr.id)
        if dt is not None:
            self.local_dtypes[target.id] = dt
        out: List[ast.stmt] = [self._empty_alloc(target.id, dims)] + body
        for s in out:
            ast.fix_missing_locations(s)
        return out

    def _expand_ix_scatter(self, arr: ast.Name, ops: List[ast.expr],
                           value: ast.expr, op: Optional[ast.AST]) -> Optional[List[ast.stmt]]:
        """``A[np.ix_(a, b, c)] (op)= rhs`` -> a nested loop
        ``A[a[i], b[j], c[k]] (op)= rhs[i, j, k]`` over the Cartesian product of
        the index arrays. The index arrays are distinct per axis (the LS3DF
        periodic fragment-box placement), so every scattered cell is unique and a
        plain accumulate matches numpy's buffered ``A[ix_] += rhs`` bit-for-bit."""
        dims = self._ix_dims(ops)
        if dims is None:
            return None
        k = len(ops)
        iters = self._ix_iters("__ixs", k)
        lhs_idx = [_scalarize_at_iters(copy.deepcopy(o), [ast.Name(id=it, ctx=ast.Load())],
                                       self.shape_table)
                   for o, it in zip(ops, iters)]
        lhs_slot = lhs_idx[0] if k == 1 else ast.Tuple(elts=lhs_idx, ctx=ast.Load())
        lhs = ast.Subscript(value=ast.Name(id=arr.id, ctx=ast.Load()),
                            slice=lhs_slot, ctx=ast.Store())
        rhs = _SubscriptifyNames(self.shape_table, iters).visit(copy.deepcopy(value))
        stmt: ast.stmt = (ast.Assign(targets=[lhs], value=rhs) if op is None
                          else ast.AugAssign(target=lhs, op=op, value=rhs))
        body: List[ast.stmt] = [stmt]
        for it, dim in zip(reversed(iters), reversed(dims)):
            body = [ast.For(
                target=ast.Name(id=it, ctx=ast.Store()),
                iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                              args=[copy.deepcopy(dim)], keywords=[]),
                body=body, orelse=[])]
        for s in body:
            ast.fix_missing_locations(s)
        return body

    def _expand_meshgrid_unpack(self, target: ast.Tuple,
                                value: ast.expr) -> Optional[List[ast.stmt]]:
        """``g0, ..., g_{k-1} = np.meshgrid(a0, ..., a_{k-1}, indexing=...)`` ->
        per-output allocator + broadcast-copy loop nest (via
        :func:`expand_meshgrid`). Each output is a fresh local of its input's
        dtype whose shape follows the ``ij`` / ``xy`` convention."""
        call = _np_func_call(value, "meshgrid")
        if call is None:
            return None
        names = [e.id for e in target.elts if isinstance(e, ast.Name)]
        if len(names) != len(target.elts):
            return None
        args = list(call.args)
        if not args or len(args) != len(names):
            return None
        in_dims = self._ix_dims(args)
        if in_dims is None:
            return None
        indexing = "xy"
        for kw in call.keywords:
            if kw.arg == "indexing" and isinstance(kw.value, ast.Constant):
                indexing = kw.value.value
        if indexing not in ("ij", "xy"):
            return None
        k = len(args)
        # perm maps an output axis to the input axis whose length it takes: a
        # single 0<->1 swap for 'xy', identity for 'ij'.
        perm = list(range(k))
        if indexing == "xy" and k >= 2:
            perm[0], perm[1] = 1, 0
        out_dims = [in_dims[perm[p]] for p in range(k)]
        out_shape_tokens = tuple(ast.unparse(d) for d in out_dims)
        out_stmts: List[ast.stmt] = []
        for d, gname in enumerate(names):
            # Output d carries input d's element type (numpy meshgrid preserves
            # per-input dtype); shape follows the indexing convention.
            if isinstance(args[d], ast.Name):
                dt = self.local_dtypes.get(args[d].id)
                if dt is not None:
                    self.local_dtypes[gname] = dt
            self.shape_table[gname] = out_shape_tokens
            kwargs = [ast.keyword(arg="indexing", value=ast.Constant(value=indexing)),
                      ast.keyword(arg=MESHGRID_AXIS_KW, value=ast.Constant(value=d))]
            loops = expand_meshgrid(ast.Name(id=gname, ctx=ast.Store()),
                                    [copy.deepcopy(a) for a in args],
                                    self.shape_table, kwargs=kwargs)
            out_stmts.append(self._empty_alloc(gname, out_dims))
            out_stmts.extend(loops)
        for s in out_stmts:
            ast.fix_missing_locations(s)
        return out_stmts

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        target = node.targets[0]
        # ``g0, g1, ... = np.meshgrid(a0, a1, ..., indexing=...)`` multi-output
        # tuple unpack -> one broadcast-copy loop nest per output.
        if isinstance(target, ast.Tuple):
            meshed = self._expand_meshgrid_unpack(target, node.value)
            if meshed is not None:
                return meshed
            return node
        # ``grid = np.ix_(a, b, c)`` open-mesh index binding: record the operands
        # and drop the statement (resolved at each ``A[grid]`` use site below).
        if isinstance(target, ast.Name):
            ix_ops = _ix_call_args(node.value)
            if ix_ops is not None:
                self._ix_grids[target.id] = ix_ops
                return None
        # ``vloc = A[grid]`` (or ``A[np.ix_(...)]``) open-mesh GATHER.
        if (isinstance(target, ast.Name)
                and isinstance(node.value, ast.Subscript)
                and isinstance(node.value.value, ast.Name)):
            ops = self._resolve_ix_operands(node.value)
            if ops is not None:
                gathered = self._expand_ix_gather(target, node.value.value, ops)
                if gathered is not None:
                    return gathered
        # ``A[grid] = rhs`` open-mesh scatter store (plain, no accumulate).
        if (isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)):
            ops = self._resolve_ix_operands(target)
            if ops is not None:
                scattered = self._expand_ix_scatter(target.value, ops, node.value, None)
                if scattered is not None:
                    return scattered
        # Fancy-index scatter store ``A[idx, c] = rhs`` (idx an index array):
        # a per-element loop. Runs first so the sliced-RHS form the emitter
        # rejects (vexx ``tg[nl, 0] = psi[:, i]``) is lowered here.
        if (isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)):
            scattered = self._expand_fancy_scatter_store(target, node.value, None)
            if scattered:
                return scattered
        # Per-statement shape table update for reassigned locals.
        # Without this, resnet's ``x = (padded - mean) / sqrt(std + eps)``
        # (after batchnorm inlining) sees ``x`` with its harvest-time
        # final shape and skips the whole-array expansion.
        if (isinstance(target, ast.Name)
                and isinstance(node.value, (ast.BinOp, ast.UnaryOp, ast.IfExp,
                                            ast.Call))
                and not (isinstance(node.value, ast.Call)
                         and isinstance(node.value.func, ast.Name)
                         and node.value.func.id == "__optarena_zeros__")):
            from numpyto_common.lib_nodes import _iter_extent_of, extent_is_scalar
            ext = _iter_extent_of(node.value, self.shape_table)
            # All-size-1 broadcast -> a scalar local, not a ``T x[1]`` array (see extent_is_scalar).
            if ext is not None and not extent_is_scalar(ext):
                self.shape_table[target.id] = tuple(
                    ast.unparse(e) for e in ext)
        # ``C[:] = expr`` on a multi-D array means whole-array elementwise
        # assignment in numpy; lower to a per-element loop that walks the
        # full extent and subscripts every Name expression on the RHS.
        if (isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id in self.shape_table
                and isinstance(target.slice, ast.Slice)
                and target.slice.lower is None and target.slice.upper is None):
            # Skip constructor-style calls (np.repeat / reshape /
            # transpose / mgrid / zeros etc.) -- they are NOT
            # shape-preserving elementwise ops and lowering them
            # per-element produces nonsense like
            # ``D[w0, w1, w2] = np.repeat(arr[w0, w1, 0], K, axis=2)``.
            if (isinstance(node.value, ast.Call)
                    and _is_constructor_call(node.value)):
                return node
            name = target.value.id
            expanded = self._expand(ast.Name(id=name, ctx=ast.Store()),
                                    node.value, None)
            if expanded:
                return expanded
        # ``A[i] = B`` -- partial-subscript LHS (a row / sub-array) assigned a
        # whole array of the remaining shape (the reduction-into-a-row pattern
        # ``back[t] = np.argmax(scores, axis=0)`` once the RHS is hoisted to a
        # temp Name). Expand to a per-element copy over the trailing dims.
        if (isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id in self.shape_table
                and not isinstance(target.slice, ast.Slice)
                and isinstance(node.value, ast.Name)
                and node.value.id in self.shape_table):
            expanded = self._expand_partial(target, node.value)
            if expanded:
                return expanded
        # ``A[b] = <Nd slice/BinOp expression>`` -- a partial-subscript LHS (a
        # sub-array, plain SCALAR leading index) assigned a whole arithmetic
        # expression of the remaining shape (stencil_4d's
        # ``out_grid[b] = w_dist[-1]*padded[...]``). The trailing residual axes
        # loop element-by-element, mirroring the AugAssign partial-subscript path.
        # Guarded narrowly so gather stores / scatter (index-array lead, bare
        # Subscript gather RHS) keep their own handling: the lead must be plain
        # scalars (no index arrays) and the RHS broadcast extent must match the
        # residual rank.
        if (isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id in self.shape_table
                and not isinstance(target.slice, ast.Slice)
                and isinstance(node.value, (ast.BinOp, ast.UnaryOp, ast.IfExp))):
            shape = self.shape_table.get(target.value.id)
            lead = (list(target.slice.elts) if isinstance(target.slice, ast.Tuple)
                    else [target.slice])
            from numpyto_common.lib_nodes import _iter_extent_of
            if (shape
                    and not any(isinstance(e, ast.Slice) for e in lead)
                    and not any(isinstance(e, ast.Name)
                                and self.shape_table.get(e.id) for e in lead)):
                n_trailing = len(shape) - len(lead)
                rhs_ext = _iter_extent_of(node.value, self.shape_table)
                if (n_trailing > 0 and rhs_ext is not None
                        and len(rhs_ext) == n_trailing):
                    expanded = self._expand_partial_subscript(target, node.value, None)
                    if expanded:
                        return expanded
        # Track Name = Name aliases in source order so a reassigned ``x``
        # gets the shape of whichever RHS preceded each use. If the LHS
        # is a fresh local (not already an array), record it so the
        # emitter declares it.
        if (isinstance(target, ast.Name)
                and isinstance(node.value, ast.Name)
                and node.value.id in self.shape_table):
            rhs_shape = self.shape_table[node.value.id]
            self.shape_table[target.id] = rhs_shape
            if target.id not in self._known_arrays:
                self.alias_locals[target.id] = tuple(rhs_shape)
            # Carry the RHS's dtype tag (notably ``complex128``) to
            # the LHS so a ``tmp = __cb3`` alias of a complex temp
            # keeps the complex tag for the C/C++ declaration.
            rhs_dt = self.local_dtypes.get(node.value.id)
            if rhs_dt and target.id not in self.local_dtypes:
                self.local_dtypes[target.id] = rhs_dt
        # ``z = arr[...]`` -- scalar local taking its dtype from a
        # known-dtype array. Lets ``abs(z)`` on a complex-array element
        # route through the ``cabs`` complex-intrinsic path.
        if (isinstance(target, ast.Name)
                and isinstance(node.value, ast.Subscript)
                and isinstance(node.value.value, ast.Name)
                and target.id not in self.local_dtypes):
            src_dt = self.local_dtypes.get(node.value.value.id)
            if src_dt is not None:
                self.local_dtypes[target.id] = src_dt
        # ``X = np.zeros/ones/empty/eye(shape, Y.dtype | np.complexNN)`` -- a fresh
        # complex work array (the eigh reduction's L / Li / C / V). The shape is
        # tracked elsewhere; here we tag the complex element type.
        if (isinstance(target, ast.Name) and target.id not in self.local_dtypes
                and isinstance(node.value, ast.Call)):
            ctag = _ctor_complex_tag(node.value, self.local_dtypes)
            if ctag is not None:
                self.local_dtypes[target.id] = ctag
        # ``X = Y.copy()`` / ``np.copy(Y)`` / ``np.ascontiguousarray(Y)`` -- inherit
        # the source's (complex) dtype (the Jacobi copies its working matrix).
        if (isinstance(target, ast.Name) and target.id not in self.local_dtypes
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)):
            f = node.value.func
            src = (f.value.id if f.attr == "copy" and isinstance(f.value, ast.Name) else
                   node.value.args[0].id if f.attr in ("copy", "ascontiguousarray", "asarray", "array")
                   and node.value.args and isinstance(node.value.args[0], ast.Name) else None)
            dt = self.local_dtypes.get(src) if src else None
            if dt and dt.startswith("complex"):
                self.local_dtypes[target.id] = dt
        # ``X = <scalar complex arithmetic>`` (``ephi = apq / m``) -- a scalar
        # BinOp/UnaryOp over complex operands. The array-BinOp branch below only
        # fires when the value is a whole-array expr (``_iter_extent_of`` non-None);
        # a scalar complex temp needs its own tag or the emit declares it real.
        if (isinstance(target, ast.Name) and target.id not in self.local_dtypes
                and isinstance(node.value, (ast.BinOp, ast.UnaryOp))
                and _scalar_expr_complex(node.value, self.local_dtypes)):
            self.local_dtypes[target.id] = "complex128"
        # ``x = BinOp(array, array)`` where ``x`` is a Name: infer
        # x's shape from the broadcast extent of the RHS and treat as
        # whole-array assignment. ``_iter_extent_of`` returns ``None``
        # for purely-scalar RHS expressions like
        # ``__inl_H_out = x.shape[1] - K + 1`` so they are not
        # misclassified as arrays.
        if (isinstance(target, ast.Name)
                and isinstance(node.value, (ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Compare, ast.BoolOp))):
            from numpyto_common.lib_nodes import _iter_extent_of, extent_is_scalar
            ext = _iter_extent_of(node.value, self.shape_table)
            # An all-size-1 broadcast (``t = (a[i] > x)`` with ``x`` shape ``(1,)``) is a SCALAR, not a
            # ``T t[1]`` array: numpyto reads size-1 arrays element-wise as ``x[0]``, so registering ``t`` as
            # an array here desyncs its scalar declaration (from ``t = 0`` / ``if t`` / ``out[0] = t``) from
            # the array-style ``t[__w0] = ...`` writes the extent drives below (a mix that will not compile).
            if ext is not None and not extent_is_scalar(ext):
                rhs_widest = tuple(ast.unparse(e) for e in ext)
                # Propagate inferred shape to the LHS so downstream
                # uses pick it up; declare as a fresh local if new.
                self.shape_table[target.id] = rhs_widest
                if target.id not in self._known_arrays:
                    self.alias_locals[target.id] = tuple(rhs_widest)
                # A whole-array boolean expression -- a Compare / BoolOp
                # (``owner = mask != 0``), a ``not``, or a bitwise ``& | ^`` of
                # boolean operands (``mask = cfl_clip & owner``) -- yields a
                # BOOLEAN array. Type it so both backends declare it bool
                # (Fortran ``logical``, C ``bool``) rather than real.
                if (_is_bool_expr(node.value, self.local_dtypes)
                        and target.id not in self.local_dtypes):
                    self.local_dtypes[target.id] = "bool_"
                # Complex-dtype propagation for BinOp / UnaryOp RHS:
                # a subtree carrying a complex literal or a complex-
                # tagged Name promotes the LHS to ``complex128`` so
                # the emit declares the right C dtype.
                if target.id not in self.local_dtypes:
                    for sub in ast.walk(node.value):
                        if (isinstance(sub, ast.Constant)
                                and isinstance(sub.value, complex)):
                            self.local_dtypes[target.id] = "complex128"
                            break
                        if isinstance(sub, ast.Name):
                            dt = self.local_dtypes.get(sub.id)
                            if dt and dt.startswith("complex"):
                                self.local_dtypes[target.id] = "complex128"
                                break
                    else:
                        # Integer-typed whole-array result (``q = j % nx`` where
                        # j is int64) stays integer -- so an index array derived
                        # from arange keeps its int dtype through the % / * chain
                        # (fft_3d's q/r/s gather indices).
                        from numpyto_common.lib_nodes import _is_integer_expr
                        if (isinstance(node.value, ast.BinOp)
                                and _is_integer_expr(node.value, self.local_dtypes,
                                                     set(self.shape_table))):
                            self.local_dtypes[target.id] = "int64"
        if (isinstance(target, ast.Name)
                and target.id in self.shape_table):
            if (isinstance(node.value, ast.Name)
                    and self.shape_table.get(node.value.id)
                    == self.shape_table[target.id]):
                expanded = self._expand(target, node.value, None)
                if expanded:
                    return expanded
            # Whole-array BinOp / UnaryOp / IfExp / Call on the RHS:
            # lower to per-element loop. The _SubscriptifyNames walker
            # rewrites every array reference inside the expression to
            # its subscripted form. ``Call`` covers cases like
            # ``x = fmax(x, 0)`` (relu post math-rename) or
            # ``x = sqrt(arr)`` where every array operand has the same
            # shape as the LHS.
            elif isinstance(node.value, (ast.BinOp, ast.UnaryOp, ast.IfExp,
                                          ast.Call, ast.Subscript, ast.Compare, ast.BoolOp)):
                # Skip constructor calls (np.zeros / np.empty etc.) --
                # those are NOT shape-preserving Calls, and lowering
                # them per-element would emit garbage like
                # ``N[w0, w1] = np.zeros(...)``.
                if isinstance(node.value, ast.Call) \
                        and _is_constructor_call(node.value):
                    return node
                # A bare-Subscript RHS is handled here ONLY when it is a fancy
                # gather (>=1 integer index-ARRAY index) -- ``__cb = u2[q, r, s]``
                # whose result extent equals the LHS (fft_3d's checksum gather).
                # A plain slice/scalar read (``x = a[:, i]``) must NOT be
                # whole-array-expanded here; it has its own handling and doing so
                # corrupts dense kernels (resnet / softmax / mlp).
                if isinstance(node.value, ast.Subscript):
                    if not _has_index_array(node.value, self.shape_table):
                        return node
                    # Fancy gather ``pos_nb = pos[nb]`` (cfd / lavamd): the
                    # result extent already equals the LHS shape (registered by
                    # the harvest), so expand DIRECTLY. ``_rhs_is_whole_array``
                    # would mis-reject it -- it treats the index array ``nb``
                    # (shape ``(nboxes,)``) as a value operand that must
                    # broadcast against the (nboxes, npart, 3) result.
                    expanded = self._expand(target, node.value, None)
                    if expanded:
                        return expanded
                    return node
                # Only attempt if every array-valued subexpression has
                # the same shape as the LHS.
                if self._rhs_is_whole_array(node.value, target.id):
                    expanded = self._expand(target, node.value, None)
                    if expanded:
                        return expanded
        return node

    def _rhs_is_whole_array(self, expr: ast.AST, lhs_name: str) -> bool:
        """Check that every array Name referenced in ``expr`` has a
        shape compatible with ``lhs_name``: equal, or broadcastable
        (rank <= LHS rank with each axis either equal to LHS or
        equal to 1). Names that appear as the ``.value`` of a
        Subscript are SKIPPED -- the Subscript itself yields a
        possibly lower-rank value, so the bare Name's declared
        rank does not constrain whole-array compatibility.
        """
        target_shape = self.shape_table.get(lhs_name)
        if target_shape is None:
            return False
        target_norm = _normalise_shape(target_shape)
        # Collect Names that appear as a Subscript value -- those
        # are accessed in lower-rank form via the Subscript, so the
        # bare Name's full-rank shape is not the relevant constraint.
        subscript_targets: Set[str] = set()
        for sub in ast.walk(expr):
            if (isinstance(sub, ast.Subscript)
                    and isinstance(sub.value, ast.Name)):
                subscript_targets.add(sub.value.id)
        has_array = False
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Name):
                if sub.id in subscript_targets:
                    continue
                shape = self.shape_table.get(sub.id)
                if shape is None:
                    continue
                shape_norm = _normalise_shape(shape)
                if shape_norm == target_norm:
                    has_array = True
                    continue
                if self._broadcastable_to(shape_norm, target_norm):
                    has_array = True
                    continue
                return False
        return has_array

    @staticmethod
    def _broadcastable_to(shape, target_shape):
        """Numpy broadcast rule: align shapes right-to-left; each
        dim must match the target or be 1; missing dims are treated
        as 1."""
        if len(shape) > len(target_shape):
            return False
        # Right-align by padding the shorter (``shape``) with implicit 1s.
        offset = len(target_shape) - len(shape)
        for i, s in enumerate(shape):
            if s == target_shape[offset + i]:
                continue
            if s == "1":
                continue
            return False
        return True

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        self.generic_visit(node)
        if (isinstance(node.target, ast.Name)
                and node.target.id in self.shape_table):
            expanded = self._expand(node.target, node.value, node.op)
            if expanded:
                return expanded
        # ``A[grid] += rhs`` open-mesh SCATTER-ADD (grid = np.ix_(a, b, c), or an
        # inline ``A[np.ix_(...)] += rhs``): a nested accumulate loop over the
        # Cartesian product of the (distinct-per-axis) index arrays.
        if (isinstance(node.target, ast.Subscript)
                and isinstance(node.target.value, ast.Name)):
            ops = self._resolve_ix_operands(node.target)
            if ops is not None:
                scattered = self._expand_ix_scatter(
                    node.target.value, ops, node.value, node.op)
                if scattered is not None:
                    return scattered
        # Fancy-index scatter ``A[idx, c] += rhs`` (idx an index array): lowered by
        # ``_expand_fancy_scatter_store`` to a snapshot-gather + store pair, matching
        # numpy's BUFFERED fancy ``+=`` (old value, last-write-wins on a duplicate
        # index -- NOT an accumulate) (the QE ultrasoft ``rhoc[nl] += aux2 * sf`` /
        # ``rhoc[box] += ...``).
        if (isinstance(node.target, ast.Subscript)
                and isinstance(node.target.value, ast.Name)):
            scattered = self._expand_fancy_scatter_store(
                node.target, node.value, node.op)
            if scattered:
                return scattered
        # Partial-index subscript target: ``Sigma[k, E, a] += __mm2`` on a
        # rank-5 Sigma indexes only 3 axes, leaving a (Norb, Norb) residual
        # slice. Loop the residual axes so the matrix accumulate lands
        # element-by-element (scattering_self_energies).
        if isinstance(node.target, ast.Subscript):
            expanded = self._expand_partial_subscript(
                node.target, node.value, node.op)
            if expanded:
                return expanded
        return node

    def _expand_partial_subscript(self, target: ast.Subscript,
                                  value: ast.expr,
                                  op: ast.AST) -> List[ast.stmt]:
        """Expand ``arr[lead] (op)= rhs`` where ``arr[lead]`` indexes only
        the leading axes of a higher-rank ``arr`` -- the trailing axes form
        a residual slice looped element-by-element. Returns [] when the
        target is not a partial scalar index."""
        if not isinstance(target.value, ast.Name):
            return []
        name = target.value.id
        shape = self.shape_table.get(name)
        if not shape:
            return []
        lead = (list(target.slice.elts) if isinstance(target.slice, ast.Tuple)
                else [target.slice])
        if any(isinstance(e, ast.Slice) for e in lead):
            return []
        if any(isinstance(e, ast.Constant) and e.value is None for e in lead):
            return []
        n_trailing = len(shape) - len(lead)
        if n_trailing <= 0:
            return []  # full index -> scalar; nothing to loop
        iters = [f"__w{i}" for i in range(n_trailing)]
        trailing = shape[-n_trailing:]
        lhs_idx = ast.Tuple(
            elts=list(lead) + [ast.Name(id=i, ctx=ast.Load()) for i in iters],
            ctx=ast.Load())
        lhs_sub = ast.Subscript(value=ast.Name(id=name, ctx=ast.Load()),
                                slice=lhs_idx, ctx=ast.Store())
        rhs = _SubscriptifyNames(self.shape_table, iters).visit(
            copy.deepcopy(value))
        # ``op is None`` -> a plain ``arr[lead] = rhs`` store (stencil_4d's
        # ``out_grid[b] = w_dist[-1] * padded[...]`` slice-expression RHS);
        # otherwise the augmented accumulate (``out_grid[b] += ...``).
        leaf: ast.stmt = (ast.Assign(targets=[lhs_sub], value=rhs) if op is None
                          else ast.AugAssign(target=lhs_sub, op=op, value=rhs))
        out: List[ast.stmt] = [leaf]
        for var, bound in zip(reversed(iters), reversed(trailing)):
            out = [ast.For(
                target=ast.Name(id=var, ctx=ast.Store()),
                iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                              args=[_token_to_ast(bound)], keywords=[]),
                body=out, orelse=[])]
        return out


def _resolve_neg_index(idx: ast.expr, axis_len: ast.expr) -> ast.expr:
    """Resolve a negative constant array index to ``axis_len - K``.

    numpy ``arr[-1]`` reads the last element; C / Fortran have no negative
    indexing, so a literal ``-K`` must become ``dim - K`` (the stencils'
    ``w_dist[-1]`` last-weight read). Both spellings -- ``Constant(-K)`` and
    ``UnaryOp(USub, Constant(K))`` -- are handled; everything else passes
    through unchanged."""
    k = None
    if isinstance(idx, ast.Constant) and isinstance(idx.value, int) and idx.value < 0:
        k = -idx.value
    elif (isinstance(idx, ast.UnaryOp) and isinstance(idx.op, ast.USub)
          and isinstance(idx.operand, ast.Constant)
          and isinstance(idx.operand.value, int)):
        k = idx.operand.value
    if k is None:
        return idx
    return ast.BinOp(left=copy.deepcopy(axis_len), op=ast.Sub(), right=ast.Constant(value=k))


class _SubscriptifyNames(ast.NodeTransformer):
    """Rewrite ``Name(arr)`` references whose shape matches the loop
    nest's bounds into ``Subscript(arr, idx)``."""

    def __init__(self, shape_table, iters):
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
        if (not isinstance(node.value, ast.Name)
                and isinstance(node.ctx, ast.Load)):
            sl0 = node.slice
            elts0 = list(sl0.elts) if isinstance(sl0, ast.Tuple) else [sl0]

            def _is_full(e):
                return (isinstance(e, ast.Slice) and e.lower is None
                        and e.upper is None and e.step is None)

            def _is_newaxis(e):
                return isinstance(e, ast.Constant) and e.value is None
            if (elts0 and all(_is_full(e) or _is_newaxis(e) for e in elts0)
                    and any(_is_full(e) for e in elts0)
                    and len(elts0) <= len(self.iters)):
                offset = len(self.iters) - len(elts0)
                sub_iters = [self.iters[offset + k]
                             for k, e in enumerate(elts0) if _is_full(e)]
                return _SubscriptifyNames(self.shape_table, sub_iters).visit(
                    copy.deepcopy(node.value))
        if (isinstance(node.value, ast.Name)
                and isinstance(node.ctx, ast.Load)):
            sl = node.slice
            # Fancy-index gather: ``arr[idx]`` where ``idx`` is an INDEX ARRAY
            # (it has its own shape) -> ``arr[idx[iter...]]``. The gathered
            # value indexes ``arr``; ``arr`` itself is NOT subscripted by the
            # iter (the bug otherwise: generic_visit would emit
            # ``arr[iter][idx[iter]]``). The index array's rank consumes the
            # right-aligned iters. edge_laplacian's ``x[src]`` / ``x[dst]``.
            if isinstance(sl, ast.Name) and self.shape_table.get(sl.id):
                idx_shape = self.shape_table[sl.id]
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
                    idx_iters = [ast.Name(id=self.iters[offset + i], ctx=ast.Load())
                                 for i in range(r)]
                    trail_iters = [ast.Name(id=self.iters[offset + r + i],
                                            ctx=ast.Load())
                                   for i in range(n_trailing)]
                    gslot = (idx_iters[0] if len(idx_iters) == 1
                             else ast.Tuple(elts=idx_iters, ctx=ast.Load()))
                    gathered = ast.Subscript(value=sl, slice=gslot, ctx=ast.Load())
                    full = [gathered] + trail_iters
                    slot = (full[0] if len(full) == 1
                            else ast.Tuple(elts=full, ctx=ast.Load()))
                    return ast.Subscript(value=node.value, slice=slot,
                                         ctx=ast.Load())
            if isinstance(sl, ast.Slice):
                if (sl.lower is None and sl.upper is None
                        and sl.step is None):
                    return self.visit_Name(node.value)
                from numpyto_common.lib_nodes import _slice_step_const
                step = _slice_step_const(sl)
                if step is not None and step != 1 and self.iters:
                    # Strided / reverse lone slice ``arr[::k]`` / ``arr[lo::k]``: source
                    # index = start + iter*k, where start is ``lower``, or 0 (positive step)
                    # / axis_len-1 (negative step -- ``arr[::-1]``) when omitted.
                    start: Optional[ast.expr] = sl.lower
                    if start is None and step < 0:
                        sh = self.shape_table.get(node.value.id)
                        if sh:
                            al = (ast.Constant(value=int(sh[0])) if str(sh[0]).isdigit()
                                  else ast.Name(id=str(sh[0]), ctx=ast.Load()))
                            start = ast.BinOp(left=al, op=ast.Sub(), right=ast.Constant(value=1))
                    # A negative step with an UNRESOLVED start (axis length not tracked)
                    # cannot emit the reverse index ``(len-1) - iter*|step|``. Falling through
                    # to the bounded path below would emit a FORWARD ``arr[iter]`` -- a silent
                    # un-reversed copy. Refuse loudly instead (mirrors _SliceToScalarRewriter,
                    # which raises for the identical untracked-shape reverse). Positive strided
                    # slices (start None, step > 0) are fine: idx = iter*step is forward.
                    if step < 0 and start is None:
                        raise NotImplementedError(
                            f"reverse slice of {node.value.id!r} needs a known axis length (shape untracked)")
                    iterv: ast.expr = ast.Name(id=self.iters[-1], ctx=ast.Load())
                    scaled: ast.expr = ast.BinOp(left=iterv, op=ast.Mult(), right=ast.Constant(value=step))
                    idx = scaled if start is None else ast.BinOp(left=scaled, op=ast.Add(), right=start)
                    return ast.Subscript(value=node.value, slice=idx, ctx=ast.Load())
                # Bounded lone slice ``arr[:k]`` / ``arr[a:b]`` / ``arr[1:]``
                # on a 1-D array: the iter loop bound already enforces the
                # slice range, so replace the slice with the (right-aligned)
                # next iter, adding any ``lower`` offset. Required for
                # durbin's ``__cb[:] = r[:k]`` copy. (Negative ``lower``
                # like ``arr[-K:]`` is uncommon and not handled here.)
                if self.iters:
                    iter_node: ast.expr = ast.Name(id=self.iters[-1],
                                                   ctx=ast.Load())
                    if sl.lower is not None and not (
                            isinstance(sl.lower, ast.Constant)
                            and sl.lower.value == 0):
                        iter_node = ast.BinOp(left=iter_node, op=ast.Add(),
                                              right=sl.lower)
                    return ast.Subscript(value=node.value, slice=iter_node,
                                         ctx=ast.Load())
            elif isinstance(sl, ast.Tuple) and sl.elts:
                # All-slice fast path -- restored bare-Name behaviour.
                all_slice = all(
                    isinstance(e, ast.Slice)
                    and e.lower is None and e.upper is None and e.step is None
                    for e in sl.elts)
                if all_slice:
                    return self.visit_Name(node.value)
                # Slice-or-index form: every element is either a Slice
                # (full ``:`` OR bounded ``:-1`` / ``a:b``) or a
                # non-Slice concrete index. Substitute each Slice with
                # the next iter (in axis order, right-aligned).
                # Mixed slice + index-array form ``xe[:, idx]`` (lulesh): a ``:``
                # axis consumes one result axis (subscripts its iter); an index
                # array of rank r consumes r result axes and becomes
                # ``idx[(those r iters)]``; concrete indices stay. Right-aligned.
                # Handles a rank>1 index array (lulesh ``x1[:, _VOLU_PERM]`` with
                # _VOLU_PERM (8,6) -> ``x1[w0, _VOLU_PERM[w1, w2]]``) and the index
                # array on any axis, not just leading.
                def _idx_rank(e):
                    return (len(self.shape_table[e.id])
                            if isinstance(e, ast.Name) and self.shape_table.get(e.id) else 0)

                def _is_index_array(e):
                    return _idx_rank(e) >= 1
                if (any(isinstance(e, ast.Slice) for e in sl.elts)
                        and any(_is_index_array(e) for e in sl.elts)):
                    result_axis_count = sum(
                        1 if isinstance(e, ast.Slice) else (_idx_rank(e) if _is_index_array(e) else 0)
                        for e in sl.elts)
                    if result_axis_count <= len(self.iters):
                        offset = len(self.iters) - result_axis_count
                        pos = 0
                        new_elts = []
                        for e in sl.elts:
                            if isinstance(e, ast.Slice):
                                it = ast.Name(id=self.iters[offset + pos], ctx=ast.Load())
                                pos += 1
                                if e.lower is not None and not (
                                        isinstance(e.lower, ast.Constant) and e.lower.value == 0):
                                    it = ast.BinOp(left=it, op=ast.Add(), right=e.lower)
                                new_elts.append(it)
                            elif _is_index_array(e):
                                r = _idx_rank(e)
                                giters = [ast.Name(id=self.iters[offset + pos + k], ctx=ast.Load())
                                          for k in range(r)]
                                pos += r
                                gslot = (giters[0] if r == 1 else ast.Tuple(elts=giters, ctx=ast.Load()))
                                new_elts.append(ast.Subscript(value=e, slice=gslot, ctx=ast.Load()))
                            else:
                                new_elts.append(e)
                        slot = (new_elts[0] if len(new_elts) == 1
                                else ast.Tuple(elts=new_elts, ctx=ast.Load()))
                        return ast.Subscript(value=node.value, slice=slot, ctx=ast.Load())
                partial_or_bounded = all(
                    isinstance(e, ast.Slice) or not isinstance(e, ast.Slice)
                    for e in sl.elts)
                if partial_or_bounded and any(isinstance(e, ast.Slice) for e in sl.elts):
                    n_slices = sum(1 for e in sl.elts if isinstance(e, ast.Slice))
                    # A ``None`` (np.newaxis) inserts a length-1 RESULT axis: it
                    # consumes an output axis (and thus an iter) but contributes
                    # no source index. The right-alignment must count it, else a
                    # lone slice in ``V[:, None]`` mis-binds to the trailing iter
                    # (``V[__w1]``) instead of the leading one (``V[__w0]``).
                    n_newaxis = sum(1 for e in sl.elts
                                    if isinstance(e, ast.Constant) and e.value is None)
                    result_rank = n_slices + n_newaxis
                    if result_rank <= len(self.iters):
                        axis_pos = len(self.iters) - result_rank
                        new_elts: List[ast.expr] = []
                        for e in sl.elts:
                            if isinstance(e, ast.Slice):
                                iter_name = self.iters[axis_pos]
                                axis_pos += 1
                                # Add the slice's ``lower`` bound to
                                # the iter so ``arr[1:, j]`` lowers as
                                # ``arr(iter + 1, j)`` instead of
                                # ``arr(iter, j)``. Negative ``lower``
                                # (e.g. ``arr[-K:]``) is uncommon and
                                # not handled here.
                                iter_node = ast.Name(id=iter_name,
                                                       ctx=ast.Load())
                                if e.lower is not None:
                                    # Constant 0 means no offset.
                                    if not (isinstance(e.lower, ast.Constant)
                                            and e.lower.value == 0):
                                        iter_node = ast.BinOp(
                                            left=iter_node,
                                            op=ast.Add(),
                                            right=e.lower)
                                new_elts.append(iter_node)
                            elif (isinstance(e, ast.Constant)
                                    and e.value is None):
                                # ``None`` (np.newaxis): consume a result axis
                                # (and its iter) but add no source index.
                                axis_pos += 1
                            else:
                                new_elts.append(e)
                        if not new_elts:
                            return ast.Name(id=node.value.id, ctx=ast.Load())
                        new_slot = (new_elts[0] if len(new_elts) == 1
                                    else ast.Tuple(elts=new_elts, ctx=ast.Load()))
                        return ast.Subscript(value=node.value,
                                             slice=new_slot, ctx=ast.Load())
            # Partial scalar index on a HIGHER-rank array: ``Ham[n]``
            # where ``Ham`` is rank-3 indexes only the leading axis and
            # the remaining axes form a slice broadcast against the loop
            # nest. numpy ``Ham[n]`` == ``Ham[n, :, :]``. Pad the
            # trailing axes with the right-aligned iter vars so the
            # scalarised subscript spans ALL of the array's axes
            # (contour_integral's ``Tz += zz * Ham[n]``). Without this
            # the read stays rank-1 (``Ham(n)`` in Fortran -> rank
            # mismatch; ``Ham[n]`` in C -> silently wrong numerics).
            lead = (list(sl.elts) if isinstance(sl, ast.Tuple) else [sl])

            def _has_idx_array(e):
                # An advanced-index axis references an index ARRAY (a Name whose
                # own shape is known) -- ``arr[idx - 1, jk, blk - 1]`` (velocity /
                # icon gathers). Those must keep the generic-visit gather path.
                return any(isinstance(s, ast.Name) and self.shape_table.get(s.id)
                           for s in ast.walk(e))
            if (lead
                    and not any(isinstance(e, ast.Slice) for e in lead)
                    and not any(isinstance(e, ast.Constant) and e.value is None
                                for e in lead)
                    and not any(_has_idx_array(e) for e in lead)):
                shape = self.shape_table.get(node.value.id)
                if shape is not None:
                    rank = len(shape)
                    n_trailing = rank - len(lead)
                    # Resolve any negative scalar index (``arr[-1]``) against its
                    # axis length -- C / Fortran have no negative indexing.
                    res_lead = [_resolve_neg_index(e, _token_to_ast(shape[ax]))
                                for ax, e in enumerate(lead)]
                    if 0 < n_trailing <= len(self.iters):
                        offset = len(self.iters) - n_trailing
                        new_elts = list(res_lead) + [
                            ast.Name(id=self.iters[offset + j], ctx=ast.Load())
                            for j in range(n_trailing)]
                        new_slot = ast.Tuple(elts=new_elts, ctx=ast.Load())
                        return ast.Subscript(value=node.value,
                                             slice=new_slot, ctx=ast.Load())
                    if n_trailing == 0:
                        # The subscript already FULLY indexes the array with
                        # concrete scalar indices (``w_dist[-1]``, ``w[r - 1]``):
                        # it is a scalar element read. Returning it (with negatives
                        # resolved) stops the default generic_visit from
                        # subscriptifying the base Name and emitting
                        # ``w_dist[__w2][-1]`` (the stencils' last-weight read).
                        new_slot = (res_lead[0] if len(res_lead) == 1
                                    else ast.Tuple(elts=res_lead, ctx=ast.Load()))
                        return ast.Subscript(value=node.value,
                                             slice=new_slot, ctx=ast.Load())
        self.generic_visit(node)
        return node


class _ChainedAssignRewriter(ast.NodeTransformer):
    """Rewrite ``s0 = s1 = s2 = 0.0`` into three separate assignments."""

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if len(node.targets) <= 1:
            return node
        # ``a = b = c = X`` -> [a=X, b=X, c=X] in source order.
        text = "\n".join(
            f"{ast.unparse(t)} = {ast.unparse(node.value)}"
            for t in node.targets)
        return ast.parse(text).body


def _target_base_name(node: ast.AST) -> Optional[str]:
    """Root name of an assignment target (``out[i][j]`` -> ``out``, a bare ``x``
    -> ``x``), or ``None`` when the target is not name-rooted."""
    while isinstance(node, ast.Subscript):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


class _TupleAssignRewriter(ast.NodeTransformer):
    """Expand tuple-LHS assignments into per-element statements.

    Two source shapes covered:

    * ``a, b, c = X, Y, Z`` -> three assignments (jacobi_2d_tile_4lvlsilly).
    * ``n, k = arr.shape`` -> substitute the shape symbols if known
      (thomas_solve, vertical_flux_prefix_scan).

    Integer local names produced by either path are collected in
    :attr:`int_locals` so the emitter can emit ``int n;`` declarations
    before first use.

    A tuple assign binds every target from the OLD values simultaneously
    (``a, b = b, a + b``). A plain sequential split reads an already-updated
    target, so when any RHS element reads a target name this pass evaluates
    each RHS into a fresh temp first and binds the targets from the temps --
    restoring numpy/python simultaneity. Temps are dtyped by the later harvest
    phase from their RHS (this rewriter runs in ``normalize-calls``, before
    ``seed-dtypes-and-harvest``), so no explicit declaration hook is needed.
    """

    def __init__(self, arrays_shapes):
        self.arrays_shapes = arrays_shapes  # dict[name, list[symbol_name]]
        #: Names introduced as integer scalar locals (collected for the
        #: emitter to declare at the top of the function body).
        self.int_locals: List[str] = []
        #: Monotonic counter for unique swap-temp names across the body.
        self._ctr: List[int] = [0]

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        if not (len(node.targets) == 1
                and isinstance(node.targets[0], ast.Tuple)):
            return node
        names = [e.id for e in node.targets[0].elts if isinstance(e, ast.Name)]
        if len(names) != len(node.targets[0].elts):
            # Mixed Subscript / Name targets, e.g. ``KE[0], PE[0] = (a, b)``
            # from helper-return tuple unpacking. Split into per-element
            # Assigns when the RHS is a Tuple literal of matching length so
            # the emit walker sees plain Subscript-assigns. Each element's
            # store target reused as-is; the RHS expressions are unparsed
            # back into source so ast.parse rebuilds them in the new
            # context.
            tgt_elts = node.targets[0].elts
            if (isinstance(node.value, ast.Tuple)
                    and len(node.value.elts) == len(tgt_elts)):
                val_elts = node.value.elts
                # Simultaneous bind (numpy): every target reads the OLD values.
                # A subscript target written here must not be observed already
                # updated by another element's RHS -- ``out[i], out[j] = out[j],
                # out[i]`` split sequentially double-reads the overwritten slot.
                # Stage each RHS into a fresh temp when a written base array is
                # read by any element; else a plain split preserves the order.
                written_bases = {b for b in (_target_base_name(t) for t in tgt_elts) if b}
                read_names = {nd.id for v in val_elts for nd in ast.walk(v) if isinstance(nd, ast.Name)}
                stmts: List[ast.stmt] = []
                if written_bases & read_names:
                    self._ctr[0] += 1
                    pfx = f"__swap{self._ctr[0]}_"
                    for i, val in enumerate(val_elts):
                        stmts.extend(ast.parse(f"{pfx}{i} = {ast.unparse(val)}").body)
                    for i, tgt in enumerate(tgt_elts):
                        stmts.extend(ast.parse(f"{ast.unparse(tgt)} = {pfx}{i}").body)
                    return stmts
                for tgt, val in zip(tgt_elts, val_elts):
                    # Preserve the per-element store context (Subscript store /
                    # Name store) by parsing the unparsed form.
                    stmts.extend(ast.parse(f"{ast.unparse(tgt)} = {ast.unparse(val)}").body)
                return stmts
            return node

        # arr.shape RHS -> shape-symbol substitution.
        if (isinstance(node.value, ast.Attribute)
                and node.value.attr == "shape"
                and isinstance(node.value.value, ast.Name)):
            arr_name = node.value.value.id
            shape = self.arrays_shapes.get(arr_name)
            if shape is None or len(shape) != len(names):
                return node
            self.int_locals.extend(names)
            text = "\n".join(f"{n} = {sym}" for n, sym in zip(names, shape))
            return ast.parse(text).body

        # Tuple RHS -> per-element assignment.
        if isinstance(node.value, ast.Tuple) and len(node.value.elts) == len(names):
            elts = node.value.elts
            if all(isinstance(v, ast.Constant) and isinstance(v.value, int)
                   for v in elts):
                # Pure int-constant tuple (grid dims): no read-after-write
                # hazard is possible, so emit direct int locals.
                self.int_locals.extend(names)
                text = "\n".join(f"{n} = {ast.unparse(v)}" for n, v in zip(names, elts))
                return ast.parse(text).body
            # Positional self-copies (``x = x``) never race, and drop out of the
            # hazard test. They are common after shape-symbol resolution rewrites
            # ``n, m = arr.shape`` into ``n, m = n, m``: those must stay plain
            # ``n = n`` splits so the promote-params pass keeps seeing n / m as
            # scalar parameters (temping them would demote them to locals).
            changed = [i for i, v in enumerate(elts)
                       if not (isinstance(v, ast.Name) and v.id == names[i])]
            written = {names[i] for i in changed}
            read = {nd.id for i in changed for nd in ast.walk(elts[i]) if isinstance(nd, ast.Name)}
            if written & read:
                # Read-after-write hazard: a reassigned target is read by another
                # element (``a, b = b, a + b``). Stage each changed RHS into a
                # fresh temp, bind the targets from the temps, and keep any
                # self-copies as plain no-op splits.
                self._ctr[0] += 1
                pfx = f"__swap{self._ctr[0]}_"
                lines = [f"{pfx}{i} = {ast.unparse(elts[i])}" for i in changed]
                lines += [f"{names[i]} = {pfx}{i}" for i in changed]
                lines += [f"{names[i]} = {ast.unparse(elts[i])}"
                          for i in range(len(names)) if i not in set(changed)]
                return ast.parse("\n".join(lines)).body
            # No hazard: plain per-element split.
            text = "\n".join(f"{n} = {ast.unparse(v)}" for n, v in zip(names, elts))
            return ast.parse(text).body

        return node


#: Matches a residual inlined-scalar token (``__inl3_N``) or an unresolved
#: ``arr.shape[`` attribute access -- the never-worse guard in the inl resolver
#: keeps the original token whenever expansion would leave one of these behind.
_INL_RE = re.compile(r"__inl\w*|\w+\.shape\[")


class LoweringContext:
    """Mutable state threaded across the ordered lowering phases in :func:`lower`.

    Each ``_lp_*`` phase reads and writes fields here instead of the long list of
    loose locals the monolithic ``lower()`` used to carry. The finalised
    side-tables (``local_dtypes`` / ``zeros_locals`` / ``zeros_fills`` /
    ``reassign_shapes`` / ``int_locals`` / ``scalar_call_temps``) are written
    straight onto :attr:`kir` -- typed :class:`KernelIR` fields the emitter reads
    directly, not attributes monkey-patched onto ``tree.__dict__``.
    """

    def __init__(self, original_kir: KernelIR, lowered: KernelIR) -> None:
        #: The un-lowered input IR -- source of ``.sparse`` and ``.helpers``.
        self.original_kir = original_kir
        #: The working (lowered) IR -- what :func:`lower` returns.
        self.kir = lowered
        #: Shortcut to the function-body AST every pass rewrites in place.
        self.tree = lowered.tree
        # Shape / dtype tables built up across phases and consumed downstream.
        self.arrays_shapes: Dict[str, List[str]] = {}
        self.lib_shape_table: Dict[str, object] = {}
        self.local_dtypes: Dict[str, str] = {}
        self.zeros_locals: Dict[str, Tuple[str, ...]] = {}
        self.shapes: Dict[str, List[str]] = {}
        self.scalar_temps: Dict[str, Tuple[str, ...]] = {}
        self.inl_defs: Dict[str, object] = {}
        self.param_seed: Dict[str, Tuple[str, ...]] = {}
        #: Bound ``_resolve_inl_table`` closure, set in the resolve-inl phase and
        #: re-used by the slice-normalise phase (both resolve ``__inl`` tokens).
        self.resolve_inl_table: Optional[Callable[[Dict], None]] = None
        # Rewriter handles whose post-visit state a later phase consumes.
        self.iter_rewriter: Optional[_ArrayIterRewriter] = None
        self.wa_rewriter: Optional[_WholeArrayAssignRewriter] = None
        self.lib_rewriter: object = None
        self.zeros: Optional[_ZerosRewriter] = None
        self.lifter: Optional[_LiftFreshArrayFromSlices] = None


def _lp_seed_shape_table(ctx: LoweringContext) -> None:
    """Seed the array-shape table, then resolve shape-mid-expressions / ellipses.

    Shape-mid-expression first -- legacy kernels use ``A.shape[0]`` inside loops /
    array constructors; everything downstream is easier if those are resolved to
    bare symbol names.
    """
    lowered = ctx.kir
    ctx.arrays_shapes = {a.name: list(a.shape) for a in lowered.arrays}
    # Sparse arrays carry CSR/etc. buffers, not a dense ArrayDesc, so they
    # are absent from ``lowered.arrays`` -- but the body still reads
    # ``A.shape[i]`` (cg/bicgstab/minres' ``n = A.shape[0]``). Seed the
    # shape table from each sparse desc's logical_shape so the resolver
    # maps ``A.shape[0]`` -> the logical dim symbol.
    for _sname, _sd in (lowered.sparse or {}).items():
        if _sd.logical_shape:
            ctx.arrays_shapes.setdefault(_sname, list(_sd.logical_shape))
    _ShapeMidExpressionRewriter(ctx.arrays_shapes).visit(ctx.tree)
    # Index-access normalisation (chained-flatten / ellipsis-expand / trailing-slice
    # pad) is deferred to the single ``normalize-index-access`` phase, which runs
    # AFTER the harvest/inlined-shape resolution so post-inline locals' ranks are
    # known (breaking the harvest<->ellipsis circular dependency for ls3df_scf).


def _lp_normalize_calls(ctx: LoweringContext) -> None:
    """Canonicalise numpy call / method forms (alloc, matmul, transpose, math,
    casts, iteration, tuple-unpack) into the loop-lowerable subset."""
    tree = ctx.tree
    ash = ctx.arrays_shapes
    _NpAliasRewriter().visit(tree)
    # ``X = alloc(...) if cond else None`` -> ``X = alloc(...)`` before the zeros
    # harvester runs, so the conditionally-allocated buffer is seen as a plain local
    # (the backends have no ``None``; reads are guarded by the same ``cond``).
    _ConditionalNoneAllocRewriter().visit(tree)
    _FullLikeRewriter().visit(tree)
    # ``np.eye`` / ``np.identity`` -> zeros + diagonal fill, BEFORE the zeros
    # harvest so the resulting ``np.zeros((n, n))`` is picked up normally. A nested
    # ``... + 1e-12 * np.eye(k)`` (LS3DF's RR jitter) is spilled to a temp first so
    # the direct-assign diagonal rewriter sees it.
    _EyeCallHoister().visit(tree)
    _EyeToZerosDiagonal().visit(tree)
    _MatmulCallRewriter().visit(tree)
    _ScatterAtRewriter(ash).visit(tree)
    _TransposeRewriter(set(ctx.original_kir.sparse or {})).visit(tree)
    _AstypeRewriter({a.name: a.dtype for a in ctx.kir.arrays if a.dtype}).visit(tree)
    _MethodCallRewriter().visit(tree)
    # A Call in subscript-index position (``v[np.argmax(np.abs(v))]``) is hoisted
    # to a fresh temp so the index is a bare Name the backends emit; the spilled
    # ``__ix = np.argmax(...)`` is expanded by the later LibNode reduction pass.
    _ComputedIndexCallHoister().visit(tree)
    ctx.iter_rewriter = _ArrayIterRewriter(ash)
    ctx.iter_rewriter.visit(tree)
    _EnumerateZipRewriter(ash).visit(tree)
    _BuiltinCastRewriter().visit(tree)
    _MathRewriter(set(ash.keys())).visit(tree)
    _DaceMapRewriter().visit(tree)
    _ChainedAssignRewriter().visit(tree)
    tuple_rewriter = _TupleAssignRewriter(ash)
    tuple_rewriter.visit(tree)
    # Stash the int-locals so the emitter can declare them.
    ctx.kir.int_locals = tuple_rewriter.int_locals


def _lp_promote_params(ctx: LoweringContext) -> None:
    """Promote shape symbols / free names to params and flag output+index arrays.

    After tuple-unpack expansion, the kernel body may reference shape symbols that
    the JSON's input_args did not declare (a numpy kernel commonly reads
    ``n, k = a.shape`` then iterates over ``range(n)``). Promote those symbols to
    first-class kernel parameters so the emitted C signature carries them.
    """
    lowered = ctx.kir
    _fold_shape_aliases(lowered)
    _promote_shape_symbols_to_params(lowered)
    # Anything still referenced in the body that isn't a declared parameter,
    # builtin, or assigned local becomes an ``int`` parameter (symbolic strides /
    # chunk sizes in TSVC-2.5 kernels).
    _promote_free_names_to_params(lowered)
    # Body-driven: detect writes (force is_output) and index-array usage (force
    # int64 dtype) so the emitter picks the right pointer qualifier / element type.
    _detect_output_and_index_arrays(lowered)


def _lp_pre_libnode_normalize(ctx: LoweringContext) -> None:
    """Pre-LibNode normalisation: mgrid, ``x.shape =`` reshape, tuple-subscript
    fold, boolean-mask reduction fusion. Also seeds the LibNode shape table."""
    tree = ctx.tree
    ctx.lib_shape_table = dict(ctx.arrays_shapes)
    # Pre-pass: collapse chained subscripts ``A[i][j]`` -> ``A[i, j]`` (vexx_k's
    # ``tabxx_qr[ia][:, ijtoh[ih, jh]]`` / ``becxx[:, jbnd, ikq][ikb]``) so the
    # harvest, scalarizers and fancy-scatter store all see a single-level access.
    _CollapseChainedSubscripts(ctx.arrays_shapes).visit(tree)
    ast.fix_missing_locations(tree)
    # Pre-pass: lower ``Xi, Yi = np.mgrid[a:b, c:d]`` tuple-unpack assignments to a
    # pair of per-element init loops -- before the main harvest so the resulting
    # fresh arrays get their shape registered like any other local.
    _MgridLowering().visit(tree)
    ast.fix_missing_locations(tree)
    # Pre-pass: rewrite ``x.shape = expr`` -> ``x = np.reshape(x, expr)``. Handles
    # chained ``Xi.shape = Yi.shape = expr`` too. Mandelbrot2 canonical uses this.
    _ShapeAttrToReshape().visit(tree)
    ast.fix_missing_locations(tree)
    # Forward-substitute a shape-tuple local (``shp = (Lb, Lb, Lb, nstate)`` --
    # the seed-time fold of a declared-array-based ``shp = Y.shape``) into its uses
    # BEFORE the harvest, so ``np.reshape(x, shp)`` is sized from the concrete tuple
    # (not a spurious 1-D ``(shp,)``) when the harvest records the reshape target.
    _TupleLocalPropagator().run(tree)
    ast.fix_missing_locations(tree)
    # Fold ``(a, b, c)[K]`` Tuple subscripts -- comes from ``arr.shape[-2]`` when
    # the shape is a tuple literal.
    _TupleSubscriptFolder().visit(tree)
    ast.fix_missing_locations(tree)
    # Peephole: fuse ``tmp = arr[mask]; X = np.<reduction>(tmp)`` into a single
    # masked iteration so we avoid materialising the dynamic-length compacted view
    # from boolean fancy indexing. Seeded with the kernel-array shapes so the loop
    # bound is the right symbol.
    _BooleanMaskReductionRewriter(
        ctx.arrays_shapes, _collect_bool_names(tree, ctx.kir.arrays)).visit(tree)
    ast.fix_missing_locations(tree)


def _lp_seed_dtypes_and_harvest(ctx: LoweringContext) -> None:
    """Seed local dtypes (signature + boolean constructors), unify mixed-complex
    selects, SSA-rename reassigned locals, then harvest local-array shapes."""
    tree = ctx.tree
    # Seed with signature-array dtypes so downstream passes (call hoister
    # _infer_complex, BinOp dtype propagation, emit-time decl) consistently treat
    # declared inputs/outputs the same way as locals. Every array goes in -- the
    # table is keyed by name so there is no cost to a uniform copy of all dtypes.
    ctx.local_dtypes = {}
    # Bind the finalised table onto the IR now; the phases below mutate it in
    # place, so the emitter reads the fully-populated dict after ``lower`` returns.
    ctx.kir.local_dtypes = ctx.local_dtypes
    for arr in ctx.kir.arrays:
        if arr.dtype:
            ctx.local_dtypes[arr.name] = arr.dtype
    # Seed boolean-typed locals from explicit ``np.zeros/empty/ones(..,
    # dtype=np.bool_)`` constructors (ICON cfl_clip / levmask) so a derived
    # ``mask = cfl_clip & owner`` is recognised as boolean (and declared bool /
    # logical) before the whole-array rewriter runs.
    for _s in ast.walk(tree):
        if (isinstance(_s, ast.Assign) and len(_s.targets) == 1
                and isinstance(_s.targets[0], ast.Name) and isinstance(_s.value, ast.Call)):
            for _kw in _s.value.keywords:
                if _kw.arg != "dtype":
                    continue
                _dv = _kw.value
                if ((isinstance(_dv, ast.Attribute) and _dv.attr in ("bool_", "bool"))
                        or (isinstance(_dv, ast.Name) and _dv.id == "bool")):
                    ctx.local_dtypes.setdefault(_s.targets[0].id, "bool_")
    # Seed the complex work-array temps (and their directly-derived scalar reads)
    # that the eigh / eigvalsh cyclic-Jacobi lowering allocates from a complex
    # signature array's ``.dtype`` -- BEFORE the true-division and libnode-expand
    # phases, which otherwise consume those still-untyped temps and mis-lower the
    # complex divide (``apq / m``) and the ``np.conj`` on the reduction matrices.
    _seed_complex_work_dtypes(tree, ctx.local_dtypes)
    # SSA-style rename for Names reassigned with different broadcast extents (hdiff
    # / vadv ``res = ...; res = ...`` with two distinct shapes). Runs BEFORE harvest
    # so each version registers under its own name and downstream passes (harvest /
    # LibNodeRewriter / lifter) see unambiguous shapes per local.
    _ssa_rename_reassigned(tree, ctx.arrays_shapes)
    _harvest_local_shapes(tree, ctx.lib_shape_table, ctx.local_dtypes)
    # Unify a mixed real/complex conditional's branches (``d = z.real if flag else
    # z``) so Fortran ``merge`` (strict same-type) and the JIT type unifiers see a
    # uniform complex select instead of a real-vs-complex pair (QE vexx gamma_only
    # path). Runs AFTER the harvest so a complex LOCAL branch (``deexx`` typed from
    # its ``np.zeros(.., complex128)`` constructor) is already known complex.
    _PromoteMixedComplexIfExp(ctx.local_dtypes).visit(tree)


def _lp_resolve_inlined_shapes(ctx: LoweringContext) -> None:
    """Resolve inlined-scalar dim tokens in the harvest table, inherit loop-var
    dtypes, and pre-lift ``alpha * A`` so the matmul hoister sees a bare Name."""
    tree = ctx.tree
    # Inlined-helper locals (conv2d's ``__inl1_output``) get their shape from
    # ``__inl<k>_`` scalar-dim locals (``__inl1_N`` ...) that are *assigned later
    # in the body* -- so an allocation sized from them at function top reads garbage,
    # and the tokens never bind. Build a resolver that substitutes each ``__inl<k>_``
    # dim-local away (fixpoint) and concretises the resulting ``param.shape[i]``
    # against the real param shapes; applied later to the declaration / malloc sink
    # (``zeros_locals`` / ``shapes``).
    from numpyto_common.frontend import (_collect_inlined_scalar_defs,
                                    _substitute_inlined_scalar_defs,
                                    _resolve_shape_attr_tokens)
    ctx.inl_defs = _collect_inlined_scalar_defs(tree)
    ctx.param_seed = {n: tuple(s) for n, s in ctx.arrays_shapes.items()}

    def _resolve_inl(shape):
        """Substitute ``__inl<k>_`` dim-locals away then resolve
        ``param.shape[i]`` -> a pure-param shape tuple.

        Best-effort and *never-worse*: a token is only rewritten when the
        result fully resolves to real parameters. If expansion would leave
        a residual ``__inl`` name or a ``.shape`` on a non-parameter local
        (the chained-inline case -- ``__inl3_N = x__v1.shape[0]`` where
        ``x__v1`` is itself a local), the ORIGINAL token is kept so the
        downstream source-order ``_ResolveArrShape`` pass still handles it
        exactly as it did before this fix existed."""
        if not ctx.inl_defs:
            return tuple(shape)
        subbed = _substitute_inlined_scalar_defs(tuple(shape), ctx.inl_defs)
        resolved = _resolve_shape_attr_tokens(subbed, ctx.param_seed)
        return tuple(new if not _INL_RE.search(new) else str(orig)
                     for orig, new in zip(shape, resolved))

    def _resolve_inl_table(table):
        for nm in list(table):
            table[nm] = list(_resolve_inl(table[nm])) \
                if isinstance(table[nm], list) else _resolve_inl(table[nm])

    ctx.resolve_inl_table = _resolve_inl_table
    # Resolve the harvest table now so the passes that consume it (notably
    # ``_WholeArrayAssignRewriter``, which decides whether ``__hcall1 + bias``
    # is a same-rank broadcast to expand into a loop vs. a raw pointer add)
    # see ``__inl1_output``'s real shape. The never-worse guard leaves the
    # chained-inline locals (whose dims reference another local's ``.shape``)
    # untouched for the source-order ``_ResolveArrShape`` pass downstream.
    if ctx.inl_defs:
        _resolve_inl_table(ctx.lib_shape_table)
    # Loop-var dtype inheritance: ``for b in data:`` (where ``data`` is
    # uint8) declares ``b`` as the element dtype of ``data``.
    array_dtypes_by_name = {a.name: a.dtype for a in ctx.kir.arrays}
    for loop_var, source_arr in ctx.iter_rewriter.var_to_array.items():
        src_dt = array_dtypes_by_name.get(source_arr) or ctx.local_dtypes.get(source_arr)
        if src_dt is not None:
            ctx.local_dtypes[loop_var] = src_dt
    # Pre-matmul: lift ``alpha * A`` -> temp so the matmul hoister sees a bare Name
    # on the left of ``A @ B``. The pre-lift runs per Assign.
    ctx.scalar_temps = {}
    scalar_counter = [0]
    for stmt in list(tree.body):
        if isinstance(stmt, (ast.Assign, ast.AugAssign)):
            sm = _ScalarTimesMatmulRewriter(ctx.lib_shape_table, ctx.scalar_temps, scalar_counter)
            stmt.value = sm.visit(stmt.value)
            # Prepend pre_stmts before the original statement.
            if sm.pre_stmts:
                idx = tree.body.index(stmt)
                tree.body[idx:idx] = sm.pre_stmts


def _lp_normalize_index_access(ctx: LoweringContext) -> None:
    """Consolidated index-access normalisation, run once AFTER every array shape is
    known (post ``resolve-inlined-shapes``).

    Three structural rewrites, in order:

    1. :class:`_ChainedSubscriptFlattener` -- collapse a scalar-chained subscript
       ``A[f][..., 0]`` -> ``A[f, ..., 0]`` so the base is always a Name.
    2. :class:`_EllipsisExpander` -- replace ``...`` with the explicit full slices
       its array's rank implies (``a[..., 0]`` on a 3-D array -> ``a[:, :, 0]``).
    3. :class:`_PadImplicitTrailingSlices` -- make numpy's implicit trailing full
       slices explicit (``A[i, j]`` on a 3-D array -> ``A[i, j, :]``).

    Placing this after the harvest / inlined-shape resolution breaks the
    harvest<->ellipsis circular dependency: post-inline locals (``hx``, ``vloc``,
    ``psi_frag[f][..., 0]``) only acquire a shape at the harvest, and the ellipsis /
    trailing-slice rewrites need that shape to know each array's rank -- so both must
    run after it, not at parse-shape time. The shape source is
    :attr:`ctx.lib_shape_table`, the harvest / inlined-resolve table that carries
    BOTH the signature arrays and every derived local (the raw
    :attr:`ctx.arrays_shapes` still holds only the declared arrays at this point,
    so an ellipsis on a post-inline local would not resolve against it)."""
    tree = ctx.tree
    shapes = ctx.lib_shape_table
    _ChainedSubscriptFlattener().visit(tree)
    _EllipsisExpander(shapes).visit(tree)
    _PadImplicitTrailingSlices(shapes).visit(tree)
    # Re-fold ``<array-expr>.shape`` / ``.shape[k]`` now that every post-inline
    # local's shape is harvested: the seed-time pass only had the DECLARED-array
    # shapes, so a ``.shape`` read on an inlined local (``v[..., None].shape[-1]``
    # in ``_hpsi``'s ``X.reshape(-1, X.shape[-1])``, or the eigh helper's
    # ``a.shape[0]``) could not resolve then. With the full table the newaxis /
    # subscript base folds to concrete dims BEFORE the reshape / LibNode expander
    # bakes the (otherwise unresolved) token into a loop bound.
    _ShapeMidExpressionRewriter(shapes).visit(tree)
    _TupleLocalPropagator().run(tree)
    _TupleSubscriptFolder().visit(tree)
    ast.fix_missing_locations(tree)
    # Re-resolve inlined-scalar dims now that the folds above concretised the
    # reshape-shape locals (``__inl5_k = shp[-1]`` -> ``= nstate``). The earlier
    # resolve-inlined-shapes phase ran BEFORE that fold, so a size local defined only
    # in a LATER inlined solve (``w`` / ``X`` reassigned across the two Rayleigh-Ritz
    # inlines) kept its ``__inl<k>_`` token and drove a use-before-def allocation
    # (garbage size). Recollect and reapply against the now-fuller table.
    from numpyto_common.frontend import _collect_inlined_scalar_defs
    ctx.inl_defs = _collect_inlined_scalar_defs(tree)
    if ctx.inl_defs and ctx.resolve_inl_table is not None:
        ctx.resolve_inl_table(ctx.lib_shape_table)


def _lp_libnode_expand(ctx: LoweringContext) -> None:
    """FFT-grid + reshape normalisation, then the LibNode expander (reductions /
    matmul / linalg); a second free-name promotion for structural scalars."""
    tree = ctx.tree
    # Flat-grid FFT idiom (vexx invfft/fwfft: reshape-to-grid -> fftn over the
    # grid axes -> reshape-back) -> materialised reshape + fft temps, so the
    # reshape/DFT expanders below see bare Names with known shapes.
    _FftGridReshapeRewriter(ctx.lib_shape_table, ctx.local_dtypes, [0]).visit(tree)
    ast.fix_missing_locations(tree)
    # Normalize ``X.reshape(a, b)`` method form to ``np.reshape(X, (a, b))``
    # AFTER the FFT idiom match (which consumes its own reshape chains) so the
    # single expand_reshape path serves lulesh's varargs spelling.
    _ReshapeMethodRewriter().visit(tree)
    ast.fix_missing_locations(tree)
    # Seed the INTEGER/UINT element dtype of every kernel-parameter array into
    # ``local_dtypes`` so the library-node hoister can see, e.g., that ``idx`` in
    # ``int(np.max(idx))`` is int64 and tag the max-reduction temp int64 rather
    # than the float default. Without this a value-preserving reduction over an
    # int array declares a real accumulator, and the Fortran emit's ``merge(int,
    # real)`` update is a kind mismatch (gfortran rejects it). Only int/uint tags
    # are seeded: complex is handled by the dedicated complex-propagation pass, and
    # a float tag would flip untagged-float-default code paths. Param arrays are
    # declared from the ABI signature (not ``local_dtypes``), so tagging them here
    # never redeclares them.
    for arr in ctx.kir.arrays:
        if arr.dtype.startswith(("int", "uint")):
            ctx.local_dtypes.setdefault(arr.name, arr.dtype)
    # Library-node expansion -- reductions, matmul, etc. -- runs before
    # ``_ZerosRewriter`` so any matmul temps the rewriter introduces are picked up
    # by the zeros pass as local arrays.
    from numpyto_common.lib_nodes import LibNodeRewriter
    ctx.lib_rewriter = LibNodeRewriter(ctx.lib_shape_table,
                                       known_arrays=set(ctx.arrays_shapes.keys()),
                                       local_dtypes=ctx.local_dtypes,
                                       sparse=ctx.original_kir.sparse)
    ctx.lib_rewriter.visit(tree)
    # Second math rename: an intrinsic whose argument only becomes a SCALAR once the library
    # nodes expand. ``np.sqrt(w @ (cov @ w))`` (portfolio_optimization) defers the rename in
    # ``_lp_normalize_calls`` -- the arg reads arrays, so the elementwise expander gets first
    # refusal -- but a 1-D x 1-D matmul lowers to a scalar dot temp, leaving ``np.sqrt(__mm2)``
    # that no later pass renames and the backends reject as an unsupported call. Re-running the
    # SAME rewriter (rather than special-casing the emitter) with the temps now known keeps the
    # rule intact: an argument that is still array-valued -- a vector matmul temp, which IS in
    # ``lib_shape_table`` -- keeps deferring; a scalar one renames to the math intrinsic.
    _MathRewriter(set(ctx.arrays_shapes.keys()) | set(ctx.lib_shape_table.keys())).visit(tree)
    # Second free-name promotion: sparse matvec dispatchers introduce structural
    # scalar symbols that don't exist until the hoister runs (SELL-C-sigma's slice
    # height ``C``), so the first promotion at the top of lower() can't see them.
    # Re-run now -- matmul temps and the JDS scratch are subscript-store targets, so
    # they're treated as locals and excluded; only genuinely free Load names get
    # promoted.
    _promote_free_names_to_params(ctx.kir)
    _fix_real_scalar_dtypes(ctx)


def _fix_real_scalar_dtypes(ctx: LoweringContext) -> None:
    """Re-derive the dtype of a LOCAL (scalar OR array temp) that LibNodeRewriter
    tagged COMPLEX from a complex source but whose value is actually REAL.

    ``LibNodeRewriter`` propagates a complex source array's element type onto
    every derived local, but a ``.real`` / ``.imag`` accessor, an ``abs`` /
    ``hypot`` magnitude, or a ``np.float64(...)`` cast produces a REAL value --
    the eigh / eigvalsh cyclic-Jacobi's ``app = A[p, p].real`` / ``aqq`` / ``m =
    hypot(...)`` / ``tau = (aqq - app) / (2 * m)``; equally the LS3DF GENPOT
    ``v = ifftn(v_g).real + V_ion + xc`` array (``.real`` result plus real
    operands) and its ``v.mean()`` temp. Left complex, the emitter declares them
    complex and ``tau >= 0.0`` (a complex comparison), ``conjg(<real>)``, or a
    real ``V_tot[...] = v[...] - v.mean()`` narrowing all fail to compile under
    C++ (C silently drops the zero imaginary part). Route each through the file's
    existing :func:`_walk_complex` predicate (which already classifies those forms
    real) and, when it disagrees with a complex tag, retag to the matching real
    width -- for an array this flips its C/C++ declaration from ``double _Complex
    *`` to ``double *``, so downstream reads, ``.mean()`` and the ``V_tot`` write
    are all real.

    A fixpoint resolves the cascade: ``tau`` is real only once ``app`` / ``aqq`` /
    ``m`` are; the GENPOT ``v`` array is real only once the ``.real`` poisson
    result is, and its mean temp only once ``v`` is. An accumulator's own
    self-reference (``acc = acc + real[...]``) carries the running value and is
    neutral -- resolved REAL -- so a mean/sum of a real array narrows, while a
    genuinely COMPLEX injected value (``acc = acc + a * b`` on complex ``a`` /
    ``b``) keeps it complex. Kernel parameter arrays are never candidates (their
    declaration is the ABI contract), but their dtypes DO resolve names on a
    candidate's RHS so a complex input cannot mis-read as real and unsoundly
    narrow. Then drop any conjugation left on a now-real operand
    (:class:`_RealConjDropper`), so ``conj(ephi)`` on the real ``ephi`` is not
    emitted as ``CONJG(<real>)``. Same class as the ``abs(complex) -> double``
    root fix -- a real-returning op of a complex operand is REAL."""
    tree = ctx.tree
    ld = ctx.local_dtypes
    # Kernel-parameter array dtypes live outside ``local_dtypes``; resolve names
    # through both so a complex INPUT read on a candidate's RHS is seen as complex
    # (else the walk would call it real and unsoundly narrow the candidate).
    array_dtypes = {a.name: a.dtype for a in ctx.kir.arrays}

    def name_dtype(nm: str) -> Optional[str]:
        dt = ld.get(nm)
        return dt if dt is not None else array_dtypes.get(nm)

    # Every complex-tagged LOCAL of known real width is a candidate -- scalars and
    # local temp arrays alike (an array's ``lib_shape_table`` shape is orthogonal
    # to its element dtype). Kernel input/output arrays are excluded: narrowing
    # their declaration would break the marshalled ABI.
    candidates = {n for n, dt in ld.items()
                  if dt in _REAL_FOR_COMPLEX and n not in array_dtypes}
    # Every value WRITTEN to a candidate -- a whole-name ``x = e`` / ``x += e`` or
    # a per-element ``x[i] = e`` / ``x[i] += e`` (an array is written elementwise).
    # A candidate is real only if EVERY write is real.
    writes: Dict[str, List[ast.expr]] = {}
    for stmt in ast.walk(tree):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            tgt: ast.AST = stmt.targets[0]
        elif isinstance(stmt, ast.AugAssign):
            tgt = stmt.target
        else:
            continue
        base = (tgt.id if isinstance(tgt, ast.Name)
                else tgt.value.id if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name) else None)
        if base in candidates:
            writes.setdefault(base, []).append(stmt.value)

    def all_writes_real(name: str, vals: List[ast.expr]) -> bool:
        # The candidate's own self-reference is neutral (resolved real): it carries
        # the running accumulator value, so a mean / sum of a real array is real,
        # yet every OTHER operand must resolve real for the write to be real.
        def resolve(nm: str) -> Optional[str]:
            return None if nm == name else name_dtype(nm)

        return all(_walk_complex(v, resolve) is None for v in vals)

    changed = True
    while changed and candidates:
        changed = False
        for name in list(candidates):
            vals = writes.get(name)
            # Fixpoint: ``tau = (aqq - app) / (2 * m)`` -- or the GENPOT ``v`` array
            # and its ``v.mean()`` temp -- read real only once the locals they
            # derive from have themselves been retagged real (this pass).
            if vals and all_writes_real(name, vals):
                ld[name] = _REAL_FOR_COMPLEX[ld[name]]
                candidates.discard(name)
                changed = True
    _RealConjDropper(ld).visit(tree)
    ast.fix_missing_locations(tree)


def _lp_whole_array_and_zeros(ctx: LoweringContext) -> None:
    """Whole-array assignment expansion, the zeros harvest, and the merged
    local-array declaration tables (``zeros_locals`` / ``zeros_fills`` /
    ``scalar_call_temps`` / ``shapes``)."""
    tree = ctx.tree
    # Whole-array Augmented / plain assignment between same-shape arrays (``x1 +=
    # temp`` or ``out = a``) is numpy's elementwise form; expand to a loop nest so
    # the C/Fortran emitter does not see pointer arithmetic. Pass the set of REAL
    # kernel arrays (not aliases that LibNodeRewriter added) so the whole-array
    # rewriter knows which aliases are fresh locals needing declaration.
    real_arrays = set(ctx.arrays_shapes.keys())
    # Boolean masking: ``arr[mask_expr] = value`` -> per-element loop with an ``if
    # mask_expr[i]:`` guard. Runs before the whole-array rewriter so the LHS is a
    # plain scalar subscript downstream.
    _BooleanMaskRewriter(ctx.lib_shape_table).visit(tree)
    ctx.wa_rewriter = _WholeArrayAssignRewriter(ctx.lib_shape_table, real_arrays,
                                                local_dtypes=ctx.local_dtypes)
    ctx.wa_rewriter.visit(tree)
    # Fold the shapes the whole-array pass inferred for genuinely-new locals
    # (meshgrid ``gx``/``gy``/``gz``, the broadcast ``gsq``) back into the shared
    # table, so the zeros harvest and slice-fusion scalarizer below size them --
    # otherwise a shapeless ``gsq`` denominator stays a bare pointer in ``v_g =
    # rho_g / gsq`` and the C division is ``complex / double *``.
    for _nm, _shp in ctx.wa_rewriter.discovered_shapes.items():
        ctx.lib_shape_table.setdefault(_nm, _shp)
    ctx.zeros = _ZerosRewriter(ctx.lib_shape_table)
    ctx.zeros.visit(tree)
    # Merge matmul-hoisted temps with the np.zeros locals -- both become C stack
    # arrays / Fortran locals in the prelude.
    zeros_locals = dict(ctx.zeros.zeros)
    zeros_locals.update(ctx.lib_rewriter.matmul_temps)
    zeros_locals.update(ctx.lib_rewriter.fresh_local_allocs)
    zeros_locals.update(ctx.scalar_temps)
    zeros_locals.update(ctx.wa_rewriter.alias_locals)
    # Pre-pass harvested local arrays (corr = np.eye(M, ...), imgOut = np.copy(...),
    # etc.) that the LibNode expanders didn't rewrite. They must still be declared
    # so the emitter sees them.
    for name, shape in ctx.lib_shape_table.items():
        if name not in zeros_locals and name not in ctx.arrays_shapes:
            zeros_locals[name] = tuple(shape) if shape else ("1",)
    ctx.zeros_locals = zeros_locals
    ctx.kir.zeros_locals = zeros_locals
    # Fill kind per local (zeros / ones / empty / ...). Only the explicit
    # ``np.<ctor>`` path (``_ZerosRewriter``) carries a meaningful kind; every other
    # source (matmul temps, slice-fusion lifts, alias locals) is a write-before-read
    # temp, so it defaults to ``empty``. The emitter consults this only when a local
    # name aliases an OUTPUT parameter -- to initialise the caller's buffer correctly
    # without a shadowing declaration.
    ctx.kir.zeros_fills = dict(ctx.zeros.fills)
    # Scalar call-hoist temps: declared as plain double locals by the emit walker
    # via its implicit-local logic (they appear as a bare Name on the LHS of an
    # Assign whose RHS is a Call).
    ctx.kir.scalar_call_temps = list(ctx.lib_rewriter.scalar_call_temps)
    # Re-collect the shape table -- np.zeros locals are included for slice fusion.
    shapes: Dict[str, List[str]] = dict(ctx.arrays_shapes)
    for name, shape in ctx.zeros.zeros.items():
        shapes[name] = list(shape) if shape else ["1"]
    # Also include any shapes harvested by the pre-pass (np.eye / np.copy /
    # np.transpose / np.linalg.* etc) that aren't in arrays_shapes / zeros locals --
    # needed when slice fusion encounters omitted-stop slices on such temps.
    for name, shape in ctx.lib_shape_table.items():
        if name not in shapes:
            shapes[name] = list(shape)
    ctx.shapes = shapes


def _lp_slice_normalize_and_lift(ctx: LoweringContext) -> None:
    """Normalise subscript forms, lift array-valued slice RHS to fresh locals, and
    re-resolve ``.size`` / ``.shape`` / ``__inl`` tokens over the new locals."""
    tree = ctx.tree
    shapes = ctx.shapes
    # Normalise subscript forms BEFORE the slice lifter so a row/column read
    # ``box = tabxx_box[ia]`` / ``qr = tabxx_qr[ia][:, k]`` (QE ultrasoft
    # augmentation) is materialised into a rank-1 local the fancy scatter / gather
    # can index. Flatten chained subscripts ``B[inner][outer]`` into one combined
    # index, then fold scalar-prefix sub-array aliases (``low = A[i, j]; low[k]`` ->
    # ``A[i, j, k]``, xsbench). Trailing-slice padding is done once, earlier, in the
    # ``normalize-index-access`` phase.
    _FlattenChainedSubscripts(shapes).visit(tree)
    _fold_subarray_aliases(tree, shapes)
    # Lift array-valued RHS (slice-bearing BinOp / Call / etc) on a bare-Name LHS to
    # a ``Name = np.zeros(extent); Name[:] = expr`` pair so slice fusion can lower
    # the per-element loop. Computes the shape from the iteration extent of the RHS,
    # registers the new local in both ``shapes`` and ``zeros_locals``.
    ctx.lifter = _LiftFreshArrayFromSlices(shapes, local_dtypes=ctx.local_dtypes)
    new_locals = ctx.lifter.run(tree)
    if new_locals:
        for name, shape in new_locals.items():
            shapes[name] = list(shape)
            ctx.zeros_locals[name] = tuple(shape)
    # ``zeros_locals`` / ``local_dtypes`` are the same objects the IR already holds
    # (bound in earlier phases), so the lifter's in-place additions -- and the
    # complex dtypes it inferred -- are already visible to the emitter.
    # Re-resolve ``.size`` / ``.shape`` / ``len(..)`` over the NOW-materialised
    # locals (``box = tabxx_box[ia, :]`` -> a rank-1 local). The early pass at
    # parse-shape time saw only params, so ``box.size == 0`` (the QE ultrasoft empty-
    # box guard) survived unresolved; with ``box`` in ``shapes`` it folds to its
    # extent. Already-resolved references are bare Names now, so the other branches
    # are no-ops.
    _ShapeMidExpressionRewriter(shapes).visit(tree)
    # ``_ZerosRewriter`` re-derives the ``np.empty`` shape straight from the AST
    # tuple, so ``__inl<k>_`` tokens reappear in ``zeros_locals`` / ``shapes`` even
    # after the early ``lib_shape_table`` resolve. These are the declaration / malloc
    # / subscript-stride sink the emitter reads -- resolve them to pure params here
    # so the top-of-function malloc is sized from real parameters (not yet-unassigned
    # ``__inl1_N`` locals).
    if ctx.inl_defs:
        ctx.resolve_inl_table(ctx.zeros_locals)
        ctx.resolve_inl_table(shapes)


def _fold_local_shape_attr_tokens(tuple_tables: List[Dict[str, object]],
                                  reassign_shapes: Optional[Dict[str, List]]) -> None:
    """Fold surviving ``arr.shape[i]`` STRING tokens in the finalised shape tables
    against the now-resolved shapes of the LOCAL arrays they name.

    The inl resolver (:meth:`LoweringContext.resolve_inl_table`) only resolves a
    ``.shape[i]`` token against PARAMETER shapes, so a token naming a LOCAL survives
    it. That happens when a temp is derived from a local operand whose shape is
    itself resolved LATE -- the eigh eigenvector temps aliased off
    ``M = Linv @ h_sub @ Linv.T``: the ``M.shape[0]`` token is recorded before the
    chained-matmul shape of ``M`` is known, and the AST ``_ResolveArrShape`` pass
    rewrites body Attributes, not the malloc / row-major-stride TABLES the emitter
    reads. By this final phase every local's shape is known, so a token-level fold
    turns ``M.shape[0]`` into ``k`` in both the allocation extents and the subscript
    strides. Reuses the frontend's token substituter (it operates on shape tokens,
    never on source). Iterated to a bounded fixpoint so a temp whose dimension names
    ANOTHER temp still converges; never-worse (a token whose base is unknown is
    left untouched)."""
    from numpyto_common.frontend import _resolve_shape_attr_tokens
    tables = [t for t in tuple_tables if t is not None]
    for _ in range(4):
        seed: Dict[str, Tuple[str, ...]] = {}
        for t in tables:
            for nm, shp in t.items():
                seed[nm] = tuple(shp)
        changed = False
        for t in tables:
            for nm in list(t):
                new = _resolve_shape_attr_tokens(tuple(t[nm]), seed)
                if new != tuple(t[nm]):
                    t[nm] = list(new) if isinstance(t[nm], list) else new
                    changed = True
        if reassign_shapes:
            for nm in list(reassign_shapes):
                new_list = [tuple(_resolve_shape_attr_tokens(tuple(s), seed))
                            for s in reassign_shapes[nm]]
                if new_list != [tuple(s) for s in reassign_shapes[nm]]:
                    reassign_shapes[nm] = new_list
                    changed = True
        if not changed:
            break


def _lp_slice_fusion_and_resolve(ctx: LoweringContext) -> None:
    """Slice fusion, source-order ``arr.shape[i]`` resolution, and forcing
    index-array locals to int64."""
    tree = ctx.tree
    shapes = ctx.shapes
    SliceFusion(shapes).visit(tree)
    # Final pass: resolve any surviving ``arr.shape[i]`` references to the concrete
    # shape token from the harvested table. These survive whenever a harvested helper
    # variable's shape was a string-form ``arr.shape[i]`` (e.g. inlined maxpool
    # ``np.empty([x.shape[0], x.shape[1] // 2, ...])`` -- the harvest stage resolves
    # what it can, but the body's range / loop bounds still reference the attribute
    # expression). The emit walker has no idea what to do with an Attribute, so we
    # substitute here.
    #
    # Seed the source-order resolver with bench-info parameter shapes only (not the
    # harvest's final-state shapes for reassigned locals). The resolver then builds
    # the current shape table per statement as it walks, so ``x.shape[i]`` at line K
    # resolves against the shape ``x`` had AT line K -- not after the kernel's final
    # reassignment.
    #
    # Stash a fresh copy of the reassign FIFO on the IR -- the emit walker consumes
    # it in source order to thread per-statement shape into multi-D subscript
    # flattening. The resolver below consumes its OWN copy (a fresh dict).
    ctx.kir.reassign_shapes = {
        k: list(v) for k, v in ctx.wa_rewriter._reassign_shapes.items()}
    _ResolveArrShape(
        shapes,
        param_shapes={k: tuple(v) for k, v in ctx.arrays_shapes.items()},
        zeros_locals={k: tuple(v) for k, v in ctx.zeros_locals.items()},
        reassign_shapes={
            k: list(v) for k, v in ctx.wa_rewriter._reassign_shapes.items()},
    ).visit(tree)
    ast.fix_missing_locations(tree)
    # Force index-array LOCALS to int64. A local whose VALUES index another array
    # (``delv[neigh_safe[w0]]`` -- neigh_safe = np.clip(lxim, ..) is a local, so the
    # param-only _detect_output_and_index_arrays misses it) must be integer;
    # C/Fortran reject a float subscript. A name used as a subscript index is always
    # integral, so this is sound.
    _idx_locals: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            sl = node.slice
            elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            for e in elts:
                if isinstance(e, ast.Subscript) and isinstance(e.value, ast.Name):
                    _idx_locals.add(e.value.id)        # A[B[i]] -> B is an index array
                elif isinstance(e, ast.Name):
                    _idx_locals.add(e.id)              # A[B] (whole-array gather) -> B
    for _nm in _idx_locals:
        if (_nm in shapes or _nm in ctx.local_dtypes):
            _dt = ctx.local_dtypes.get(_nm)
            if not (_dt and _dt.startswith(("int", "uint"))):
                ctx.local_dtypes[_nm] = "int64"
    # Resolve any ``arr.shape[i]`` token left in the malloc / stride tables against
    # the now-fully-known LOCAL shapes (M => (k, k)), so eigh temps aliased off a
    # late-resolved chained-matmul operand allocate + index with concrete extents.
    _fold_local_shape_attr_tokens([ctx.arrays_shapes, shapes, ctx.zeros_locals],
                                  ctx.kir.reassign_shapes)


def _lp_promote_true_division(ctx: LoweringContext) -> None:
    """Promote all-integer ``/`` to a floating divide (numpy true division).

    Runs EARLY -- right after dtype seeding + harvest, BEFORE the LibNode /
    slice / reshape passes synthesize their own integer index arithmetic
    (row-major ``idx / stride`` decompositions that MUST stay integer). It
    therefore only rewrites divisions the numpy SOURCE wrote, never internally
    generated index math. Array names come from the harvested shape tables so a
    float array element is not mistaken for an integer; a later scalarization of
    a whole-array ``np.float64(a) / b`` recurses into the cast operand
    unchanged.

    The dtype table is ``local_dtypes`` (arrays) WIDENED with the declared scalar params
    and the body's provable float scalars: neither is in ``local_dtypes``, and
    ``_is_integer_expr`` reads an untagged non-array Name as integer, so without them a
    float divide (chebyshev_filter_subspace's ``sigma1 / e``, over declared float64
    params) is promoted as if it were int/int -- baking an fp64 cast into an otherwise
    fp32 kernel."""
    array_names = set(ctx.lib_shape_table) | set(ctx.arrays_shapes)
    tags: Dict[str, str] = {s.name: s.dtype for s in ctx.kir.scalars}
    tags.update(ctx.local_dtypes)  # the harvested tags win over the declaration
    _ScalarFloatTagger(tags, array_names).visit(ctx.tree)
    _TrueDivisionPromoter(tags, array_names).visit(ctx.tree)
    ast.fix_missing_locations(ctx.tree)


def _lp_lower_helpers(ctx: LoweringContext) -> None:
    """Lower each non-inlinable helper the same way -- it is a self-contained
    sub-kernel (own params + body). Its early ``return`` survives lowering (the
    return-extraction is a parse_kernel step, not a lowering pass)."""
    ctx.kir.helpers = [lower(h) for h in ctx.original_kir.helpers]


#: The lowering pipeline as data: an ordered list of ``(name, phase)`` pairs run
#: over a shared :class:`LoweringContext`. The ORDER is load-bearing -- a slice
#: pass consults the array-shape table, so ``np.zeros`` locals must be registered
#: before it; see each phase's docstring / inline comments for the rationale.
_LOWER_PHASES: List[Tuple[str, Callable[["LoweringContext"], None]]] = [
    ("seed-shape-table", _lp_seed_shape_table),
    ("normalize-calls", _lp_normalize_calls),
    ("promote-params", _lp_promote_params),
    ("pre-libnode-normalize", _lp_pre_libnode_normalize),
    ("seed-dtypes-and-harvest", _lp_seed_dtypes_and_harvest),
    ("promote-true-division", _lp_promote_true_division),
    ("resolve-inlined-shapes", _lp_resolve_inlined_shapes),
    ("normalize-index-access", _lp_normalize_index_access),
    ("libnode-expand", _lp_libnode_expand),
    ("whole-array-and-zeros", _lp_whole_array_and_zeros),
    ("slice-normalize-and-lift", _lp_slice_normalize_and_lift),
    ("slice-fusion-and-resolve", _lp_slice_fusion_and_resolve),
    ("lower-helpers", _lp_lower_helpers),
]

#: Environment flag that turns on the between-phase invariant checks below.
#: Off by default (production emit pays nothing); flip it on to localise a
#: pipeline regression to the exact phase that corrupted the shared state.
_INVARIANT_ENV = "OPTARENA_LOWER_INVARIANTS"


def _assert_lowering_invariants(phase_name: str, ctx: LoweringContext) -> None:
    """Check the cross-phase invariants that must hold after ``phase_name``.

    Debug-only (gated by :data:`_INVARIANT_ENV`). Every failure names the phase
    that broke it, so a side-table assigned the wrong container type or an AST
    the context stopped tracking is caught at the phase boundary that introduced
    it -- not later, as an inscrutable emit-time ``KeyError`` or ``AttributeError``.
    """
    kir = ctx.kir
    # The context's tree handle must remain the kir's tree: a phase that rebuilds
    # the AST has to write it back to both, or later phases rewrite an orphan.
    if ctx.tree is not kir.tree:
        raise AssertionError(f"lowering invariant after '{phase_name}': ctx.tree no "
                             "longer aliases ctx.kir.tree (a phase rebuilt the AST "
                             "without writing it back to both handles)")
    if not isinstance(kir.tree, ast.FunctionDef):
        raise AssertionError(f"lowering invariant after '{phase_name}': kir.tree is "
                             f"{type(kir.tree).__name__}, expected ast.FunctionDef")
    # The typed side-tables keep their declared container type -- a phase that
    # assigns the wrong shape surfaces here, not at the emitter reader.
    for _fld, _typ in (("int_locals", list), ("local_dtypes", dict), ("zeros_locals", dict), ("zeros_fills", dict),
                       ("scalar_call_temps", list), ("reassign_shapes", dict)):
        _val = vars(kir)[_fld]
        if not isinstance(_val, _typ):
            raise AssertionError(f"lowering invariant after '{phase_name}': kir.{_fld} "
                                 f"is {type(_val).__name__}, expected {_typ.__name__}")
    # The AST stays structurally well-formed: a rewriter that leaves a bad field
    # (a raw string where a node belongs, a Call missing args) fails to unparse.
    # Unparse a fixed-up copy -- synthetic nodes legitimately lack ``lineno``
    # mid-lowering, so filling locations on a throwaway keeps the check about
    # structure (and leaves the real tree untouched).
    try:
        ast.unparse(ast.fix_missing_locations(copy.deepcopy(kir.tree)))
    except Exception as exc:
        raise AssertionError(f"lowering invariant after '{phase_name}': kir.tree does "
                             f"not round-trip through ast.unparse ({exc})") from exc


def lower(kir: KernelIR) -> KernelIR:
    """Return a lowered copy of ``kir`` ready for backend emission.

    The body is a fixed sequence of named phases (:data:`_LOWER_PHASES`), each
    mutating a shared :class:`LoweringContext`. Pipeline shape: math rename ->
    ``np.zeros`` -> slice fusion. Order matters: the slice rewriter consults the
    array-shape table, and ``np.zeros`` locals must be registered first so their
    shapes are visible to it.

    Matmul (``A @ B`` / ``np.matmul`` -- normalised to ``@`` by
    :class:`_MatmulCallRewriter`) is loop-lowered uniformly for every target;
    the Fortran ``MATMUL`` intrinsic is reserved for the rare unresolved-shape
    case the loop hoister cannot lower (handled in the Fortran emitter).

    Set :data:`_INVARIANT_ENV` in the environment to run
    :func:`_assert_lowering_invariants` after every phase.
    """
    check = _assert_lowering_invariants if _INVARIANT_ENV in os.environ else None
    ctx = LoweringContext(kir, copy.deepcopy(kir))
    for _name, _phase in _LOWER_PHASES:
        _phase(ctx)
        if check is not None:
            check(_name, ctx)
    return ctx.kir


#: Python builtins / harness identifiers that may appear in the body
#: but are not parameter candidates.
_BUILTIN_NAMES: Set[str] = {
    "range", "len", "min", "max", "abs", "int", "float", "bool",
    "True", "False", "None", "enumerate", "zip", "round",
    "__optarena_zeros__",
    # ``np`` is the numpy module alias and ``math`` the math module --
    # they appear as free Names in stockham_fft's ``np.mgrid`` /
    # ``np.pi`` etc. after partial lowering. ``numpy`` covers the
    # alternate ``import numpy`` (no alias) form. ``cp`` / ``cupy``
    # for GPU-flavoured kernel source. Resolving them to module
    # macros happens elsewhere; they must NOT be promoted to scalar
    # function parameters.
    "np", "math", "numpy", "cp", "cupy",
    # C math constants emitted by ``_MathRewriter.visit_Attribute``
    # (np.pi / np.e / np.inf / np.nan). They look like bare Name
    # references in the lowered tree but resolve to <math.h> macros
    # at C emit time, so they should NOT be promoted to scalar
    # function parameters.
    "M_PI", "M_E", "INFINITY", "NAN",
    # ``optarena.frameworks.framework`` dtype aliases the legacy mandelbrot
    # kernels import (``from ... import np_float, np_complex``) and pass as
    # ``dtype=np_float``. The dtype harvest reads them for the local's element
    # type and the zeros/linspace expander then consumes the kwarg, so they
    # must NOT be promoted to scalar parameters in the meantime.
    "np_float", "np_complex", "np_int",
} | _MATH_INTRINSIC_NAMES


def _detect_output_and_index_arrays(kir: KernelIR) -> None:
    """Two body-driven adjustments to :class:`ArrayDesc`:

    * If a parameter array is written to (appears as the base of a
      ``Subscript`` on the LHS of an assignment) and was not already
      marked ``is_output``, flip the flag so the C signature uses
      a non-const pointer.
    * If a parameter array is used as a subscript index (``A[B[i]]``
      where ``B`` is a parameter array), force its dtype to
      ``int64`` so the emitter picks ``int64_t *`` instead of
      ``double *`` -- C rejects ``double`` subscripts.
    """
    name_to_arr = {a.name: a for a in kir.arrays}
    written: Set[str] = set()
    index_arrays: Set[str] = set()
    # Indirect-index tracking: ``k = ip[i]`` records ip as the source of
    # scalar ``k``; if ``k`` is later used inside any subscript index,
    # ip is an index array (its values index another array), even though
    # the use is one hop removed from the ``A[B[i]]`` direct form.
    scalar_src: Dict[str, str] = {}            # scalar name -> source array
    scalars_used_as_index: Set[str] = set()

    def _walk(node):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                    if t.value.id in name_to_arr:
                        written.add(t.value.id)
                # ``data -= mean`` or ``A[:] = ...`` -- whole-array writes
                # on a bare Name target also count as output.
                elif isinstance(t, ast.Name) and t.id in name_to_arr:
                    written.add(t.id)
            # ``k = arr[...]`` -- scalar takes its value from a param array.
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Subscript)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id in name_to_arr):
                scalar_src[node.targets[0].id] = node.value.value.id
        if isinstance(node, ast.Subscript):
            sl = node.slice
            elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            for e in elts:
                # Subscript-as-index: ``A[B[i]]`` -> B is an index array.
                if isinstance(e, ast.Subscript) and isinstance(e.value, ast.Name):
                    if e.value.id in name_to_arr:
                        index_arrays.add(e.value.id)
                # Direct array index: ``u2[q, r, s]`` where q/r/s are ARRAY
                # Names used as integer-index arrays (fft_3d fancy gather) ->
                # each is an index array (must be int, not the float default).
                # A boolean-mask Name goes through the mask rewriter earlier, so
                # any array Name still appearing as a bare index here is integer.
                if (isinstance(e, ast.Name) and e.id in name_to_arr):
                    index_arrays.add(e.id)
                # Any bare Name appearing in the index expression (e.g.
                # ``c[LEN_1D - k - 1]``) is a scalar used as an index.
                for sub in ast.walk(e):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                        scalars_used_as_index.add(sub.id)
        # ``np.take(a, idx[, axis])`` -- ``idx`` (a param array) holds gather indices, so
        # it is an index array (must be int), even though ``take`` is not yet expanded into
        # the ``a[idx[..]]`` subscript form the direct detection above keys on.
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id in ("np", "numpy")
                and node.func.attr == "take" and len(node.args) >= 2
                and isinstance(node.args[1], ast.Name) and node.args[1].id in name_to_arr):
            index_arrays.add(node.args[1].id)
        # ``np.ix_(a, b, c)`` -- each operand is an open-mesh index array (used
        # to index another array), so a param operand must be integer, even
        # though the ``A[ix_]`` gather / scatter is not expanded yet.
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id in ("np", "numpy")
                and node.func.attr == "ix_"):
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id in name_to_arr:
                    index_arrays.add(arg.id)
        for child in ast.iter_child_nodes(node):
            _walk(child)

    _walk(kir.tree)

    # Promote any array feeding a scalar that is itself used as an index.
    for scalar in scalars_used_as_index:
        src = scalar_src.get(scalar)
        if src is not None:
            index_arrays.add(src)

    for name in written:
        name_to_arr[name].is_output = True  # type: ignore[misc]
    for name in index_arrays:
        a = name_to_arr[name]
        # Respect an explicit integer dtype (declared via bench_info
        # ``init.dtypes`` -- the authoritative source ported from the
        # original ``dace.int32`` annotation). Only auto-promote arrays
        # still at a float default, so the heuristic stays a safety net
        # for undeclared kernels without overriding a declared width.
        if str(getattr(a, "dtype", "") or "").startswith(("int", "uint")):
            continue
        a.dtype = "int64"  # type: ignore[misc]


def _promote_free_names_to_params(kir: KernelIR) -> None:
    """Add free names referenced in the body to ``input_args`` as ``int``.

    Many ported kernels (TSVC-2.5 in particular) carry a symbolic
    stride ``K`` / chunk size ``T`` / divisor ``M`` referenced in the
    body but never declared in the function signature. Treat each such
    name as a scalar integer parameter; the bench_info layer then
    binds it to the preset's ``parameters`` dict (or to a synthesised
    default of 1 if no preset declares it).
    """
    declared: Set[str] = set(kir.input_args)
    declared.update(_BUILTIN_NAMES)
    # Logical sparse arrays (``A`` / ``B`` in ``A @ B``) are expanded
    # into physical buffer params by the frontend; their bare names must
    # never be promoted to scalar int params even if a residual
    # reference survives lowering. The matmul hoister consumes them.
    declared.update(getattr(kir, "sparse", {}) or {})
    # Non-inlinable helpers emitted as their own native functions: their names
    # appear as CALL funcs (``classify(x[i])``), never as scalar parameters.
    declared.update(h.kernel_name for h in getattr(kir, "helpers", []) or [])
    # A Name used as a call function is never a scalar parameter either.
    for node in ast.walk(kir.tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            declared.add(node.func.id)
    # Names assigned anywhere in the body are local variables, not params.
    def _names_in_target(tgt: ast.AST) -> List[str]:
        """Collect every bare Name id appearing as an assignment target,
        including inside a Tuple / Starred unpack
        (``i_coord, j_coord = np.mgrid[...]``) and inside a Subscript
        base (``a[i] = ...``). Without this, mgrid-style tuple unpacks
        leak the locals into the free-name promotion as undeclared
        parameters."""
        if isinstance(tgt, ast.Name):
            return [tgt.id]
        if isinstance(tgt, ast.Tuple):
            out = []
            for e in tgt.elts:
                out.extend(_names_in_target(e))
            return out
        if isinstance(tgt, ast.Starred):
            return _names_in_target(tgt.value)
        if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name):
            return [tgt.value.id]
        return []

    for node in ast.walk(kir.tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                for nm in _names_in_target(tgt):
                    declared.add(nm)
        elif isinstance(node, ast.AugAssign):
            for nm in _names_in_target(node.target):
                declared.add(nm)
        elif isinstance(node, ast.For):
            for nm in _names_in_target(node.target):
                declared.add(nm)

    free: List[str] = []
    seen: Set[str] = set()
    for node in ast.walk(kir.tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            n = node.id
            if n in declared or n in seen:
                continue
            # Sparse dispatchers pass size EXPRESSIONS as synthetic Name
            # ids (e.g. ``"(NBR + 1) - 1"`` for a block-row count). These
            # render fine inline but are not real parameters -- a genuine
            # free parameter is always a valid identifier.
            if not n.isidentifier():
                continue
            seen.add(n)
            free.append(n)
    from numpyto_common.ir import SymbolDesc
    for name in free:
        kir.symbols.append(SymbolDesc(name=name))
        if name not in kir.input_args:
            kir.input_args.append(name)


def _fold_shape_aliases(kir: KernelIR) -> None:
    """Eliminate body-defined dimension aliases by substituting them inline.

    A kernel that opens with ``M = a.shape[0]`` (already resolved to ``M = N``
    by the shape-mid-expression pass) and then allocates an output ``H`` of
    shape ``(M + 1, N + 1)`` must not carry ``M`` anywhere: the output-zeroing
    ``memset`` is injected at the TOP of the body, before the ``M = N`` local
    assignment runs, so any use of ``M`` (in ``H``'s ``np.zeros`` shape, or its
    descriptor) reads it uninitialised. We replace each such alias with its
    defining expression throughout the body AST and the array descriptors, then
    drop the now-dead defining assignment -- so ``H`` becomes ``(N + 1, N + 1)``
    and ``M`` disappears.

    Only a name assigned EXACTLY ONCE, to an expression built entirely from
    in-scope symbols / params (not itself, not another local), is folded -- a
    reassigned counter is left alone, so no semantics change."""
    import re as _re
    _IDENT = _re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    in_scope = ({s.name for s in kir.symbols}
                | {s.name for s in kir.scalars}
                | {a.name for a in kir.arrays})
    # Dimension symbols (``N`` in ``a: (N,)``) are not yet promoted to the
    # symbol list when this runs, but they ARE valid scope for an alias RHS --
    # ``M = N`` is foldable because ``N`` names ``a``/``b``'s extent. Only
    # INPUT arrays contribute genuine dim symbols; an OUTPUT shape may itself
    # carry the alias (``H: (M + 1, N + 1)``) -- scanning it would re-add ``M``
    # to scope and veto its own folding.
    for arr in kir.arrays:
        if getattr(arr, "is_output", False):
            continue
        for tok in arr.shape:
            in_scope.update(_IDENT.findall(str(tok)))
    # Only names that actually appear in an array's shape tokens are dimension
    # aliases worth folding -- this is what makes ``M`` (in ``H: (M+1, N+1)``)
    # a candidate while excluding array-valued temps like edge_laplacian's
    # ``flux = w * (x[src] - x[dst])`` (never a shape token), which must NOT be
    # inlined into the body.
    shape_idents: Set[str] = set()
    for arr in kir.arrays:
        for tok in arr.shape:
            shape_idents.update(_IDENT.findall(str(tok)))
    # Count assignments per name so only single-definition aliases qualify.
    assign_count: Dict[str, int] = {}
    for node in ast.walk(kir.tree):
        tgts = (node.targets if isinstance(node, ast.Assign)
                else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign))
                else [])
        for t in tgts:
            for nm in (n.id for n in ast.walk(t) if isinstance(n, ast.Name)):
                assign_count[nm] = assign_count.get(nm, 0) + 1
    aliases: Dict[str, ast.expr] = {}
    for node in ast.walk(kir.tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if (not isinstance(tgt, ast.Name) or tgt.id in in_scope
                or tgt.id not in shape_idents
                or assign_count.get(tgt.id, 0) != 1):
            continue
        rhs_names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
        if tgt.id in rhs_names or not rhs_names <= in_scope:
            continue
        aliases[tgt.id] = node.value
    if not aliases:
        return

    class _Fold(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in aliases):
                return None                # drop the now-dead defining stmt
            self.generic_visit(node)
            return node

        def visit_Name(self, node: ast.Name):
            if isinstance(node.ctx, ast.Load) and node.id in aliases:
                return copy.deepcopy(aliases[node.id])
            return node

    _Fold().visit(kir.tree)
    ast.fix_missing_locations(kir.tree)

    def _sub(tok: str) -> str:
        for name, expr in aliases.items():
            tok = _re.sub(rf"\b{_re.escape(name)}\b", f"({ast.unparse(expr)})", tok)
        return tok

    for arr in kir.arrays:
        new_shape = tuple(_sub(str(t)) for t in arr.shape)
        if new_shape != tuple(arr.shape):
            arr.shape = new_shape  # type: ignore[misc]


def _body_defined_locals(tree: ast.AST) -> Set[str]:
    """Names the kernel body DEFINES as scalar locals: a plain ``X = <expr>``
    whose target ``X`` does not itself appear in ``<expr>``.

    A self-referential assignment (``N = N``, the residue of ``N = b.shape[0]``
    when ``b`` is declared ``(N,)``) is excluded -- it merely re-states an input
    dimension and ``N`` must remain a promotable shape symbol. AugAssign
    (``X += ...``) and subscript targets (``A[i] = ...``) are likewise not
    definitions. Used to keep computed dimension aliases (smith_waterman's
    ``M = a.shape[0]``) out of the parameter list."""
    defined: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name):
            continue
        rhs_names = {n.id for n in ast.walk(node.value)
                     if isinstance(n, ast.Name)}
        if tgt.id not in rhs_names:
            defined.add(tgt.id)
    return defined


def _promote_shape_symbols_to_params(kir: KernelIR) -> None:
    """Add every array-shape symbol to ``input_args`` if not declared.

    Each array's declared shape contains the symbol names the C/Fortran
    backends need in scope to render the parameter type. Promote ALL of
    them -- otherwise a kernel signature like ``s174(a, b, M)`` that
    declares ``a(LEN_1D)`` would refer to an undeclared ``LEN_1D``.
    The order preserves declaration order of the arrays so the param
    list looks natural (``LEN_1D, M, a, b``).
    """
    import re as _re
    _IDENT = _re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    declared = {s.name for s in kir.symbols}
    # Names that are scalars / arrays already in scope must NOT be
    # re-promoted as shape symbols (e.g. a buffer whose own name appears
    # in another shape token would be wrong, but more importantly the
    # logical sparse names and existing scalars are already declared).
    in_scope = (declared
                | {s.name for s in kir.scalars}
                | {a.name for a in kir.arrays}
                # A name DEFINED in the body (``M = a.shape[0]`` -> ``M = N``)
                # is a computed local, not a free input symbol -- even when it
                # appears in an output array's shape (smith_waterman's H is
                # ``(M+1, N+1)``). Promoting it would add a redundant scalar
                # param the caller cannot resolve. A self-assignment ``N = N``
                # (from ``N = b.shape[0]`` where ``b`` is ``(N,)``) is NOT a
                # definition -- it just re-states the real dimension param, so
                # such names stay promotable.
                | _body_defined_locals(kir.tree))
    shape_syms: List[str] = []
    seen: Set[str] = set()
    for arr in kir.arrays:
        for tok in arr.shape:
            # A shape token may be a bare identifier (``N``) or a
            # compound expression (``NK + 1`` / ``nnz_A``). Extract every
            # identifier so symbols inside arithmetic (the CSR
            # ``indptr`` bound ``NK + 1``) are promoted -- C tolerates
            # an undeclared bound via flat pointers, but Fortran renders
            # the explicit-shape array and needs the symbol in scope.
            ident_iter = ([tok] if tok.isidentifier()
                          else _IDENT.findall(str(tok)))
            for sym in ident_iter:
                if sym in in_scope or sym in seen:
                    continue
                shape_syms.append(sym)
                seen.add(sym)
    from numpyto_common.ir import SymbolDesc
    for sym in shape_syms:
        kir.symbols.insert(0, SymbolDesc(name=sym))
        if sym not in kir.input_args:
            kir.input_args.insert(0, sym)
