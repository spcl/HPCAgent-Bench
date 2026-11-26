# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Validation of the ``np.linalg.eigvalsh`` (eigenvalues-only symmetric
eigensolve) lowering.

``np.linalg.eigvalsh`` reuses the same self-contained cyclic-Jacobi sweep as
``np.linalg.eigh`` (``numpyto_common.numpy_desugar.eigh.eigh_c_stmts``), but binds
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
import pytest

from hpcagent_bench.translators.numpyto_common.numpy_desugar import EighLoopRewriter, eigh_alias_names
from tests.translators.source_module import run_source

EIGVALSH_SRC = "def f(A):\n    w = np.linalg.eigvalsh(A)\n"


def sym(n: int, seed: int) -> np.ndarray:
    """A real symmetric ``n``-by-``n`` matrix (distinct eigenvalues)."""
    m = np.random.default_rng(seed).random((n, n))
    return m + m.T


def desugar_body(src: str, dtypes: dict | None = None) -> list:
    """Run the C/Fortran-frontend eigh/eigvalsh rewriter over ``src`` and return
    the rewritten body of its single function -- the exact loop nest the native
    backends receive (``frontend.parse_kernel`` runs the same pass). ``dtypes``
    is the declared-dtype KIND table (``{"A": "float"}``, ...); empty means every
    operand's dtype is unknown, so the rewriter keeps the (always-safe) complex
    Jacobi form."""
    tree = ast.parse(src)
    EighLoopRewriter(eigh_alias_names(tree), dtypes or {}).visit(tree)
    ast.fix_missing_locations(tree)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    return fn.body


def exec_desugared(src: str, scope: dict, dtypes: dict | None = None) -> dict:
    mod = ast.Module(body=desugar_body(src, dtypes), type_ignores=[])
    ast.fix_missing_locations(mod)
    scope.update({"np": np, "range": range, "abs": abs})
    run_source(mod, scope, "<eigvalsh>")
    return scope


def test_eigvalsh_desugar_matches_numpy() -> None:
    """The rewritten eigenvalues-only loop nest, executed as numpy, reproduces
    ``numpy.linalg.eigvalsh`` (ascending) on a real symmetric matrix."""
    n = 5
    A = sym(n, 0)
    w = np.asarray(exec_desugared(EIGVALSH_SRC, {"A": A.copy()})["w"])
    ref = np.linalg.eigvalsh(A)
    assert np.allclose(w, ref, rtol=1e-6, atol=1e-6)
    assert np.all(np.diff(w) >= -1e-9)  # ascending, like numpy


DERIVED_OPERAND_SRC = (
    "def f(X, W):\n"
    "    h_sub = X.T @ W\n"
    "    s_sub = X.T @ X\n"
    "    L = np.linalg.cholesky(s_sub)\n"
    "    Linv = np.linalg.inv(L)\n"
    "    M = Linv @ h_sub @ Linv.T\n"
    "    w = np.linalg.eigvalsh(M)\n"
)


@pytest.mark.parametrize(
    ("dtypes", "complex_form"), [({"X": "float", "W": "float"}, False), ({}, True)], ids=["real-operand", "undeclared"]
)
def test_a_real_operand_stays_real_through_transpose_and_factorisations(
    dtypes: dict[str, str], complex_form: bool
) -> None:
    """rayleigh_ritz_rotation's shape: the operand is a LOCAL, not a declared array.

    ``M = Linv @ h_sub @ Linv.T`` is three assignments and two factorisations away from anything
    bench_info declares. While the rewriter consulted only the declared table, ``M`` read as
    unknown -- never provably real -- so the real branch was unreachable for exactly the kernels
    whose operand is built rather than passed, and the emitted ``.real``/``.imag`` became an
    unqualified C++ ``real()``/``imag()`` that does not compile. The other direction must never
    regress: with nothing declared, the operand is unknown, and unknown must take the COMPLEX branch.
    Guessing real there would drop an imaginary part -- a wrong answer, not a missed optimisation."""
    txt = ast.unparse(ast.Module(body=desugar_body(DERIVED_OPERAND_SRC, dtypes=dtypes), type_ignores=[]))
    if complex_form:
        assert ".real" in txt, f"an unknown-dtype operand must keep the complex Jacobi form:\n{txt}"
    else:
        assert ".real" not in txt, f"a real operand built through .T / cholesky / inv still emits .real:\n{txt}"
        assert ".imag" not in txt, f"a real operand built through .T / cholesky / inv still emits .imag:\n{txt}"
