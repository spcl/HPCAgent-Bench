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
The third (xfail) drives the full C/Fortran compile+run oracle; it is blocked by a
PRE-EXISTING emitter type-inference gap (documented on the marker) that breaks the
baseline ``eigh`` identically, and lives in ``numpyto_common/lowering.py`` -- not in
the desugar. It flips to XPASS once that emitter gap is fixed.
"""
import ast

import numpy as np
import pytest

from _op_oracle import run_op
from numpyto_common.numpy_desugar import _EighLoopRewriter, _eigh_alias_names

_EIGVALSH_SRC = "def f(A):\n    w = np.linalg.eigvalsh(A)\n"


def _sym(n: int, seed: int) -> np.ndarray:
    """A real symmetric ``n``-by-``n`` matrix (distinct eigenvalues)."""
    m = np.random.default_rng(seed).random((n, n))
    return m + m.T


def _desugar_body(src: str) -> list:
    """Run the C/Fortran-frontend eigh/eigvalsh rewriter over ``src`` and return
    the rewritten body of its single function -- the exact loop nest the native
    backends receive (``frontend.parse_kernel`` runs the same pass)."""
    tree = ast.parse(src)
    _EighLoopRewriter(_eigh_alias_names(tree)).visit(tree)
    ast.fix_missing_locations(tree)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    return fn.body


def _exec_desugared(src: str, scope: dict) -> dict:
    mod = ast.Module(body=_desugar_body(src), type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, "<eigvalsh>", "exec"), {"np": np, "range": range, "abs": abs}, scope)
    return scope


def test_eigvalsh_lowers_to_eigenvalues_only_nest():
    """``w = np.linalg.eigvalsh(A)`` rewrites to the shared cyclic-Jacobi sweep
    bound to a single Name -- no ``L⁻ᴴ`` back-transform, no eigenvector output."""
    txt = ast.unparse(ast.Module(body=_desugar_body(_EIGVALSH_SRC), type_ignores=[]))
    assert "np.hypot" in txt                        # the Jacobi sweep is emitted
    assert "_X" not in txt                          # no back-transform x = L⁻ᴴ y
    assert txt.rstrip().endswith("w = __eigh0_wa")  # binds ONLY the eigenvalue vector


def test_eigvalsh_desugar_matches_numpy():
    """The rewritten eigenvalues-only loop nest, executed as numpy, reproduces
    ``numpy.linalg.eigvalsh`` (ascending) on a real symmetric matrix."""
    n = 5
    A = _sym(n, 0)
    w = np.asarray(_exec_desugared(_EIGVALSH_SRC, {"A": A.copy()})["w"])
    ref = np.linalg.eigvalsh(A)
    assert np.allclose(w, ref, rtol=1e-6, atol=1e-6)
    assert np.all(np.diff(w) >= -1e-9)              # ascending, like numpy


@pytest.mark.xfail(
    reason="PRE-EXISTING emitter gap (not the desugar): the C/Fortran type inference in "
    "numpyto_common/lowering.py declares the Jacobi's real scalar temps derived from "
    "`.real`/`.imag` (e.g. tau, app, aqq) as complex, so `tau >= 0.0` / `conjg(real)` "
    "fail to compile. Breaks baseline `eigh` identically; XPASSes once that is fixed.",
    strict=False,
)
def test_eigvalsh_native_c_fortran_matches_numpy():
    """Full C + Fortran compile+run of ``w[:] = np.linalg.eigvalsh(A)`` vs numpy."""
    n = 5
    A = _sym(n, 1)
    res = run_op("import numpy as np\ndef f(A, w):\n tmp = np.linalg.eigvalsh(A)\n w[:] = tmp\n",
                 "f", {"A": A}, {"w": (n, )}, {"N": n},
                 shapes={"A": "(N, N)", "w": "(N,)"},
                 rtol=1e-6, atol=1e-6,
                 backends=("c", "fortran"))
    for b in ("c", "fortran"):
        assert res[b] == "ok", f"native {b} did not validate: {res}"
