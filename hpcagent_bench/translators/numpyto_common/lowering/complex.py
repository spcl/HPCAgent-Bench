"""Complex dtype inference and seeding for locals and temporaries."""

import ast
from collections.abc import Callable

from hpcagent_bench.translators.numpyto_common.frontend import dtype_from_constructor
from hpcagent_bench.translators.numpyto_common.ir import COMPLEX_FOR_FLOAT
from hpcagent_bench.translators.numpyto_common.lib_nodes.array_methods import ARRAY_METHOD_SHAPE_OPS

#: numpy functions / accessors that return a REAL value even from a COMPLEX
#: operand. The complex-detection walk must NOT descend into their arguments --
#: else ``np.abs(z)`` / ``np.real(z)`` / ``z.real`` read as complex and mis-drive
#: both the lifted-temp dtype (declaring a real magnitude ``complex128``) and the
#: intrinsic router (``csqrt`` on a real, wrong for a negative operand).
REAL_FROM_COMPLEX: frozenset[str] = frozenset({"real", "imag", "abs", "absolute", "angle", "hypot", "sign"})

#: Array METHODS that carry their operand's element type through: the value being typed is the
#: RECEIVER, which is nowhere in ``node.args``. ``ARRAY_METHOD_SHAPE_OPS`` is the shared list of
#: methods whose numpy twin means the same thing; the reshaping trio is not in it (each has its own
#: emit branch) and is just as dtype-preserving.
DTYPE_PRESERVING_METHODS: frozenset[str] = ARRAY_METHOD_SHAPE_OPS | frozenset({"reshape", "ravel", "flatten"})

#: The same rearrange-don't-convert ops in their ``np.<name>(x)`` spelling, where the value being
#: typed is the FIRST ARGUMENT instead of the receiver. The ``*_like`` allocators belong here for
#: the same reason -- they take their element type from the PROTOTYPE -- and they matter because
#: the whole-array pass rewrites ``Y.copy()`` into ``np.empty_like(Y)`` plus a copy loop before
#: any of this runs, so ``copy`` alone never sees the source again.
DTYPE_PRESERVING_FUNCS: frozenset[str] = frozenset(
    {
        "copy",
        "ascontiguousarray",
        "asarray",
        "array",
        "transpose",
        "conj",
        "conjugate",
        "empty_like",
        "zeros_like",
        "ones_like",
        "full_like",
    }
)


def dtype_carrying_operands(call: ast.Call) -> tuple[str, ...]:
    """The array NAMES whose element dtype ``call`` could carry, best first.

    The METHOD spelling holds the value in its RECEIVER (``Y.copy()``, ``Y[:, j].reshape(p, q)``)
    and the FUNCTION spelling in its first argument (``np.copy(Y)``, ``np.empty_like(Y)``); a
    SLICED receiver still has Y's element type. Both are offered because ``copy`` is spelled both
    ways -- ``np.copy(Y)`` resolves the ``np`` receiver to nothing and falls through to ``Y``.
    Empty for a call that CONVERTS (an explicit ``dtype=``, ``astype``) or is neither.
    """
    func = call.func
    if not isinstance(func, ast.Attribute) or any(kw.arg == "dtype" for kw in call.keywords):
        return ()
    out: list[str] = []
    for node in ([func.value] if func.attr in DTYPE_PRESERVING_METHODS else []) + (
        [call.args[0]] if func.attr in DTYPE_PRESERVING_FUNCS and call.args else []
    ):
        while isinstance(node, ast.Subscript):
            node = node.value
        if isinstance(node, ast.Name):
            out.append(node.id)
    return tuple(out)


def walk_complex(node: ast.AST, name_dtype: "Callable[[str], str | None]") -> str | None:
    """Return a complex dtype string if ``node`` produces a complex value, else
    ``None``. Call-AWARE: a ``np.<real-returning>(...)`` call (:data:`REAL_FROM_COMPLEX`)
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
        return None if node.attr in ("real", "imag") else walk_complex(node.value, name_dtype)
    if isinstance(node, ast.Subscript):
        return walk_complex(node.value, name_dtype)
    if isinstance(node, (ast.Compare, ast.BoolOp)):
        return None
    if isinstance(node, ast.BinOp):
        return walk_complex(node.left, name_dtype) or walk_complex(node.right, name_dtype)
    if isinstance(node, ast.UnaryOp):
        return walk_complex(node.operand, name_dtype)
    if isinstance(node, ast.IfExp):
        return walk_complex(node.body, name_dtype) or walk_complex(node.orelse, name_dtype)
    if isinstance(node, ast.Call):
        return call_complex(node, name_dtype)
    # Unhandled node type -- fall back to a conservative whole-subtree scan.
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, complex):
            return "complex128"
        if isinstance(sub, ast.Name):
            dt = name_dtype(sub.id)
            if dt and dt.startswith("complex"):
                return dt
    return None


def call_complex(node: ast.Call, name_dtype: "Callable[[str], str | None]") -> str | None:
    """:func:`walk_complex` of a call. A real-returning function is real. An explicit ``dtype=`` /
    ``.astype(...)`` DECIDES, both ways: it produces a complex value even when no operand is complex
    (a complex array whose only visible write is its zero-init), and ``z.astype(np.float64)`` is real
    however complex ``z`` is. Otherwise complex iff an argument is -- or, for a dtype-preserving
    METHOD (``x.copy()``, ``w[:, j].reshape(p, q)``), iff the receiver is."""
    fn = (
        node.func.attr
        if isinstance(node.func, ast.Attribute)
        else node.func.id
        if isinstance(node.func, ast.Name)
        else None
    )
    if fn in REAL_FROM_COMPLEX:
        return None
    ctor_dt = dtype_from_constructor(node)
    if ctor_dt is not None:
        return ctor_dt if ctor_dt.startswith("complex") else None
    for a in node.args:
        r = walk_complex(a, name_dtype)
        if r:
            return r
    if isinstance(node.func, ast.Attribute) and fn in DTYPE_PRESERVING_METHODS:
        return walk_complex(node.func.value, name_dtype)
    return None


def infer_complex_dtype(expr: ast.AST, local_dtypes: dict[str, str]) -> str | None:
    """Return a complex dtype string if ``expr`` produces a complex value, else
    ``None``. Delegates to the call-aware :func:`walk_complex` so a real-returning
    ufunc / accessor of a complex operand is correctly REAL."""
    return walk_complex(expr, local_dtypes.get)


#: The real element type underlying each complex width -- the inverse of the IR's
#: real->complex precision map, so a ``.real`` / ``.imag`` / ``abs`` / ``hypot``
#: scalar temp derived from a complex array is retagged to the matching real
#: width (never hardcoded: derived from ``ir.COMPLEX_FOR_FLOAT``, first real per
#: complex, so complex128->float64, complex64->float32, complex256->float128).
REAL_FOR_COMPLEX: dict[str, str] = {}
for flt, cplx in COMPLEX_FOR_FLOAT.items():
    REAL_FOR_COMPLEX.setdefault(cplx, flt)


def is_conj_call(node: ast.AST) -> ast.expr | None:
    """If ``node`` is a conjugation -- ``np.conj(x)`` / ``np.conjugate(x)`` (free
    function) or ``x.conjugate()`` (method) -- return its single operand ``x``;
    else ``None``. The operand is the value whose conjugate is taken."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return None
    f = node.func
    if (
        f.attr in ("conj", "conjugate")
        and isinstance(f.value, ast.Name)
        and f.value.id in ("np", "numpy")
        and len(node.args) == 1
    ):
        return node.args[0]
    if f.attr == "conjugate" and not node.args:
        return f.value
    return None


class RealConjDropper(ast.NodeTransformer):
    """Drop a conjugation applied to a provably-REAL operand: ``conj(x) -> x``
    when :func:`walk_complex` classifies ``x`` real.

    numpy ``conj`` of a real is the identity, but Fortran ``CONJG`` requires a
    COMPLEX argument, so ``CONJG(<real>)`` is a compile error (the eigh /
    eigvalsh cyclic-Jacobi's ``ephi`` is real -- ``np.float64(apq) / m`` -- yet is
    wrapped in ``np.conj`` for the general Hermitian form). Removing the no-op
    conjugation on a real operand keeps both native backends valid; a genuinely
    complex operand keeps its conjugation."""

    def __init__(self, local_dtypes: dict[str, str]) -> None:
        self.local_dtypes = local_dtypes

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        operand = is_conj_call(node)
        if operand is not None and walk_complex(operand, self.local_dtypes.get) is None:
            return operand
        return node


def ctor_complex_tag(call: ast.Call, local_dtypes: dict[str, str]) -> str | None:
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


def scalar_expr_complex(expr: ast.AST, local_dtypes: dict[str, str]) -> bool:
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
        return scalar_expr_complex(expr.left, local_dtypes) or scalar_expr_complex(expr.right, local_dtypes)
    if isinstance(expr, ast.UnaryOp):
        return scalar_expr_complex(expr.operand, local_dtypes)
    if isinstance(expr, ast.Call):
        fn = (
            expr.func.attr
            if isinstance(expr.func, ast.Attribute)
            else (expr.func.id if isinstance(expr.func, ast.Name) else "")
        )
        if fn in REAL_FROM_COMPLEX:
            return False
        # Anything else (exp, sqrt, conj, a reshape, an unknown helper) is assumed to CARRY the
        # element type of its arguments: assuming real instead would silently drop an imaginary
        # part, which is the worse failure of the two.
        return any(scalar_expr_complex(a, local_dtypes) for a in expr.args)
    return False


def seed_complex_work_dtypes(
    tree: ast.AST, local_dtypes: dict[str, str], array_dtypes: dict[str, str] | None = None
) -> None:
    """Seed ``local_dtypes`` for complex work-array temps and their directly
    derived scalar reads, before ``promote-true-division`` and ``libnode-expand``
    consume those dtypes.

    The eigh / eigvalsh cyclic-Jacobi lowering allocates complex work matrices
    (``L`` / ``Li`` / ``Tm`` / ``Cm`` / ``X`` / ``jv``) via ``np.zeros((n, n),
    b.dtype)`` and derives scalars off them (``apq = Cm[i, j]``, ``m =
    hypot(apq.real, apq.imag)``, ``ephi = apq / m``). Those dtypes are otherwise
    only recorded at the whole-array phase (:class:`WholeArrayAssignRewriter`),
    which runs after two phases that already consume them:

    * ``promote-true-division`` reads an untagged ``apq`` as integer, wrongly
      promoting ``apq / m`` with a ``np.float64(apq)`` cast -- truncating the
      complex Jacobi phase (and C++ rejects the cast as ``(double)(complex)``);
    * ``libnode-expand``'s :class:`RealConjDropper` classifies the still-untyped
      complex arrays as REAL via :func:`walk_complex` and drops every
      ``np.conj`` on them, computing the wrong eigenvalues.

    Reuses the whole-array rewriter's own predicates (:func:`ctor_complex_tag` /
    :func:`scalar_expr_complex` / :func:`walk_complex`), iterated to a fixpoint
    since a derived scalar's dtype depends on the temp it reads (``m`` / ``ephi``
    <- ``apq`` <- ``Cm``'s constructor)."""
    assigns = [
        s
        for s in ast.walk(tree)
        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)
    ]
    seed = ComplexWorkSeed(local_dtypes, array_dtypes or {})
    changed = True
    while changed:
        changed = False
        for s in assigns:
            name = s.targets[0].id
            if name in local_dtypes:
                continue
            dt = seed.dtype_for(s.value)
            if dt is not None:
                local_dtypes[name] = dt
                changed = True


class ComplexWorkSeed:
    """The complex (or matching real) dtype :func:`seed_complex_work_dtypes` gives a local from the
    value it is bound to."""

    def __init__(self, local_dtypes: dict[str, str], array_dtypes: dict[str, str]) -> None:
        self.local_dtypes = local_dtypes
        self.array_dtypes = array_dtypes

    def known(self, name: str | None) -> str | None:
        """The recorded element dtype of ``name`` -- a local, else a declared KERNEL array (a
        parameter's dtype lives outside ``local_dtypes``, and an untyped local reads as REAL to
        :class:`RealConjDropper`)."""
        if name is None:
            return None
        dt = self.local_dtypes.get(name)
        return dt if dt is not None else self.array_dtypes.get(name)

    def known_complex(self, name: str | None) -> str | None:
        dt = self.known(name)
        return dt if dt and dt.startswith("complex") else None

    def dtype_for(self, value: ast.expr) -> str | None:
        if isinstance(value, ast.Call):
            return self.call_dtype(value)
        # z = A[scalar-index] -- inherit a complex array's element dtype
        if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
            return self.known_complex(value.value.id)
        # ephi = apq / m -- a scalar BinOp/UnaryOp over complex operands
        if isinstance(value, (ast.BinOp, ast.UnaryOp)) and scalar_expr_complex(value, self.local_dtypes):
            return "complex128"
        return None

    def call_dtype(self, value: ast.Call) -> str | None:
        """``np.zeros/ones/empty/eye(shape, Y.dtype | np.complexNN)``; a call that rearranges or selects
        values without changing what a value IS (``copy`` / ``transpose`` / ``conj`` / ``where``,
        either arm deciding a ``where``) inherits a complex operand's dtype; a real-returning magnitude
        (``hypot`` / ``abs`` / ``.real`` / ``.imag``) of a complex operand takes the MATCHING REAL
        width."""
        ctag = ctor_complex_tag(value, self.local_dtypes)
        if ctag is not None:
            return ctag
        if isinstance(value.func, ast.Attribute):
            for src in dtype_carrying_operands(value):
                sdt = self.known_complex(src)
                if sdt:
                    return sdt
            if value.func.attr == "where" and len(value.args) == 3:
                for arm in value.args[1:]:
                    adt = self.known_complex(arm.id) if isinstance(arm, ast.Name) else None
                    if adt:
                        return adt
        fn = (
            value.func.attr
            if isinstance(value.func, ast.Attribute)
            else value.func.id
            if isinstance(value.func, ast.Name)
            else None
        )
        if fn in REAL_FROM_COMPLEX:
            for sub in ast.walk(value):
                if isinstance(sub, ast.Name):
                    bdt = self.known_complex(sub.id)
                    if bdt:
                        return REAL_FOR_COMPLEX.get(bdt, "float64")
        return None


class PromoteMixedComplexIfExp(ast.NodeTransformer):
    """Make a mixed real/complex conditional's two branches the SAME type.

    ``d = z.real if gamma_only else z`` pairs a REAL branch (``.real`` strips the
    imaginary part) with a COMPLEX one. C promotes the real branch implicitly, but
    Fortran ``merge`` -- and the numba/pythran/jax type unifiers -- are strict and
    reject a real-vs-complex pair. Promote the real branch to complex with a
    cast-free ``+ 0j`` (a complex-literal add, NOT a C-style cast), so every
    backend sees a uniform-type select. Numerically identical: the promoted branch
    carries a zero imaginary part. (QE vexx ``_add_nlxx_pot`` gamma_only path.)"""

    def __init__(self, local_dtypes: dict[str, str]) -> None:
        self.local_dtypes = local_dtypes

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.generic_visit(node)
        body_cplx = scalar_expr_complex(node.body, self.local_dtypes)
        else_cplx = scalar_expr_complex(node.orelse, self.local_dtypes)
        if body_cplx and not else_cplx:
            node.orelse = self.to_complex(node.orelse)
        elif else_cplx and not body_cplx:
            node.body = self.to_complex(node.body)
        return node

    @staticmethod
    def to_complex(e: ast.expr) -> ast.expr:
        return ast.copy_location(ast.BinOp(left=e, op=ast.Add(), right=ast.Constant(value=0j)), e)
