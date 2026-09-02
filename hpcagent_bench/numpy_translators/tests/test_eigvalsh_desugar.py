"""Validation of the ``np.linalg.eigvalsh`` (eigenvalues-only symmetric
eigensolve) lowering.

``np.linalg.eigvalsh`` reuses the same self-contained cyclic-Jacobi sweep as
``np.linalg.eigh`` (``numpyto_common.numpy_desugar._eigh_c_stmts``), but binds
only the ascending eigenvalue vector into a SINGLE Name target -- the eigenvector
back-transform / ``U`` output is dropped. The kernel ``ls3df_scf`` uses it as
``theta_max = np.linalg.eigvalsh(T).max()``; a standalone ``w = np.linalg.eigvalsh(A)``
is the plain form exercised here.

The first two tests exec the desugared loop nest as numpy and compare against
``numpy.linalg.eigvalsh`` -- the same exec-the-desugar validation the existing
eigh tests use (``test_translator_feature_fixes.test_eigh_generalized_subset_matches_scipy``).
The third drives the full C/Fortran compile+run oracle.
"""

import ast

import numpy as np

from _op_oracle import run_op
from numpyto_common.numpy_desugar import _EighLoopRewriter, _eigh_alias_names

_EIGVALSH_SRC = "def f(A):\n    w = np.linalg.eigvalsh(A)\n"


def _sym(n: int, seed: int) -> np.ndarray:
    """A real symmetric ``n``-by-``n`` matrix (distinct eigenvalues)."""
    m = np.random.default_rng(seed).random((n, n))
    return m + m.T


def _desugar_body(src: str, dtypes: dict | None = None) -> list:
    """Run the C/Fortran-frontend eigh/eigvalsh rewriter over ``src`` and return
    the rewritten body of its single function -- the exact loop nest the native
    backends receive (``frontend.parse_kernel`` runs the same pass). ``dtypes``
    is the declared-dtype KIND table (``{"A": "float"}``, ...); empty means every
    operand's dtype is unknown, so the rewriter keeps the (always-safe) complex
    Jacobi form."""
    tree = ast.parse(src)
    _EighLoopRewriter(_eigh_alias_names(tree), dtypes or {}).visit(tree)
    ast.fix_missing_locations(tree)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    return fn.body


def _exec_desugared(src: str, scope: dict, dtypes: dict | None = None) -> dict:
    mod = ast.Module(body=_desugar_body(src, dtypes), type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, "<eigvalsh>", "exec"), {"np": np, "range": range, "abs": abs}, scope)
    return scope


def test_eigvalsh_lowers_to_eigenvalues_only_nest():
    """``w = np.linalg.eigvalsh(A)`` rewrites to the shared cyclic-Jacobi sweep
    bound to a single Name -- no ``L^-H`` back-transform, no eigenvector output."""
    txt = ast.unparse(ast.Module(body=_desugar_body(_EIGVALSH_SRC), type_ignores=[]))
    assert "np.hypot" in txt  # the Jacobi sweep is emitted
    assert "_X" not in txt  # no back-transform x = L^-H y
    assert txt.rstrip().endswith("w = __eigh0_wa")  # binds ONLY the eigenvalue vector


def test_eigvalsh_desugar_matches_numpy():
    """The rewritten eigenvalues-only loop nest, executed as numpy, reproduces
    ``numpy.linalg.eigvalsh`` (ascending) on a real symmetric matrix."""
    n = 5
    A = _sym(n, 0)
    w = np.asarray(_exec_desugared(_EIGVALSH_SRC, {"A": A.copy()})["w"])
    ref = np.linalg.eigvalsh(A)
    assert np.allclose(w, ref, rtol=1e-6, atol=1e-6)
    assert np.all(np.diff(w) >= -1e-9)  # ascending, like numpy


def test_eigvalsh_real_operand_drops_real_imag_accessors():
    """A ``A`` DECLARED real (dtype kind ``"float"``) must desugar with no ``.real``/
    ``.imag`` accessor left in the Jacobi sweep: DaCe lowers those to an UNQUALIFIED
    ``real()``/``imag()`` C++ call ADL cannot reach for a non-complex operand
    (``'real' was not declared in this scope; did you mean 'std::real'?``, dace
    issue 08-unqualified-real-imag) -- the desugar must not emit the call at all
    when the operand is provably real, since ``np.real(x)`` on a real ``x`` is
    ``x`` and ``np.imag(x)`` is exactly ``0.0``. An UNKNOWN-dtype operand (the
    default ``{}`` table other tests here use) keeps emitting them -- this is the
    dtype-gated case only."""
    txt = ast.unparse(ast.Module(body=_desugar_body(_EIGVALSH_SRC, dtypes={"A": "float"}), type_ignores=[]))
    assert ".real" not in txt
    assert ".imag" not in txt
    assert "np.hypot" in txt  # the Jacobi sweep still runs, just without the accessors

    n = 5
    A = _sym(n, 2)
    w = np.asarray(_exec_desugared(_EIGVALSH_SRC, {"A": A.copy()}, dtypes={"A": "float"})["w"])
    ref = np.linalg.eigvalsh(A)
    assert np.allclose(w, ref, rtol=1e-6, atol=1e-6)


def test_eigvalsh_native_c_fortran_matches_numpy():
    """Full C + Fortran compile+run of ``w[:] = np.linalg.eigvalsh(A)`` vs numpy."""
    n = 5
    A = _sym(n, 1)
    res = run_op(
        "import numpy as np\ndef f(A, w):\n tmp = np.linalg.eigvalsh(A)\n w[:] = tmp\n",
        "f",
        {"A": A},
        {"w": (n,)},
        {"N": n},
        shapes={"A": "(N, N)", "w": "(N,)"},
        rtol=1e-6,
        atol=1e-6,
        backends=("c", "fortran"),
    )
    for b in ("c", "fortran"):
        assert res[b] == "ok", f"native {b} did not validate: {res}"


_DERIVED_OPERAND_SRC = (
    "def f(X, W):\n"
    "    h_sub = X.T @ W\n"
    "    s_sub = X.T @ X\n"
    "    L = np.linalg.cholesky(s_sub)\n"
    "    Linv = np.linalg.inv(L)\n"
    "    M = Linv @ h_sub @ Linv.T\n"
    "    w = np.linalg.eigvalsh(M)\n"
)


def test_a_real_operand_stays_real_through_transpose_and_factorisations():
    """rayleigh_ritz_rotation's shape: the operand is a LOCAL, not a declared array.

    ``M = Linv @ h_sub @ Linv.T`` is three assignments and two factorisations away from anything
    bench_info declares. While the rewriter consulted only the declared table, ``M`` read as
    unknown -- never provably real -- so the real branch was unreachable for exactly the kernels
    whose operand is built rather than passed, and the emitted ``.real``/``.imag`` became an
    unqualified C++ ``real()``/``imag()`` that does not compile."""
    txt = ast.unparse(
        ast.Module(body=_desugar_body(_DERIVED_OPERAND_SRC, dtypes={"X": "float", "W": "float"}), type_ignores=[])
    )
    assert ".real" not in txt, f"a real operand built through .T / cholesky / inv still emits .real:\n{txt}"
    assert ".imag" not in txt, f"a real operand built through .T / cholesky / inv still emits .imag:\n{txt}"


def test_an_undeclared_operand_keeps_the_complex_form():
    """The other direction, and the one that must never regress: with nothing declared, the
    operand is unknown, and unknown must take the COMPLEX branch. Guessing real here would drop
    an imaginary part -- a wrong answer, not a missed optimisation."""
    txt = ast.unparse(ast.Module(body=_desugar_body(_DERIVED_OPERAND_SRC, dtypes={}), type_ignores=[]))
    assert ".real" in txt, f"an unknown-dtype operand must keep the complex Jacobi form:\n{txt}"
