# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The lowered generalized Hermitian ``eigh(a, b)`` agrees with scipy under numba.

The desugar reduces ``a x = w b x`` by Cholesky to a standard Hermitian problem and solves that by
inline cyclic Jacobi (:func:`numpyto_common.numpy_desugar._eigh_stmts`); no LAPACK eigensolver reaches
the emitted module. Eigenvectors are unique only up to a phase, so they are checked through what defines
them -- ``a v = b v diag(w)`` and ``v^H b v = I`` -- elementwise, beside the eigenvalues themselves.
"""

import ast
import importlib.util
from types import ModuleType

import numpy as np
import pytest
import scipy.linalg
from numpyto_common.ir import ArrayDesc, KernelIR
from numpyto_numba.emit import emit_numba

PENCIL_SRC = """import numpy as np
from scipy.linalg import eigh


def solve_pencil(a, b, w, v):
    ww, vv = eigh(a, b)
    w[:] = ww
    v[:, :] = vv
"""

#: Operand kind -> the dtype ``a`` and ``b`` are declared with.
OPERAND_DTYPES = {"real": "float64", "complex": "complex128"}


def pencil_ir(operand_dtype: str) -> KernelIR:
    """KernelIR for :data:`PENCIL_SRC` with ``a``/``b`` of ``operand_dtype``."""
    tree = next(n for n in ast.parse(PENCIL_SRC).body if isinstance(n, ast.FunctionDef))
    return KernelIR(
        tree=tree,
        kernel_name="solve_pencil",
        input_args=["a", "b", "w", "v"],
        arrays=[
            ArrayDesc(name="a", dtype=operand_dtype, shape=("N", "N")),
            ArrayDesc(name="b", dtype=operand_dtype, shape=("N", "N")),
            ArrayDesc(name="w", dtype="float64", shape=("N",)),
            ArrayDesc(name="v", dtype="complex128", shape=("N", "N")),
        ],
    )


@pytest.fixture(scope="module")
def numba_solvers(tmp_path_factory: pytest.TempPathFactory) -> dict[str, ModuleType]:
    """The pencil kernel emitted and imported once per operand kind."""
    solvers: dict[str, ModuleType] = {}
    for kind, dtype in OPERAND_DTYPES.items():
        path = tmp_path_factory.mktemp(f"numba_{kind}") / "solve_pencil_numba_np.py"
        emitted = emit_numba(PENCIL_SRC, kir=pencil_ir(dtype))
        assert "eigh(" not in emitted.split("def solve_pencil", 1)[1]
        path.write_text(emitted)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        solvers[kind] = module
    return solvers


def pencil(n: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """A Hermitian ``a`` and a Hermitian positive definite ``b`` of order ``n``."""
    rng = np.random.default_rng(n)
    m = rng.standard_normal((n, n))
    q = rng.standard_normal((n, n))
    if kind == "complex":
        m = m + 1j * rng.standard_normal((n, n))
        q = q + 1j * rng.standard_normal((n, n))
    return m + m.conj().T, q @ q.conj().T + n * np.eye(n)


@pytest.mark.parametrize("kind", sorted(OPERAND_DTYPES))
@pytest.mark.parametrize("n", [3, 5, 8])
def test_numba_generalized_eigh_matches_scipy_eigenpairs(
    numba_solvers: dict[str, ModuleType], n: int, kind: str
) -> None:
    """Ascending eigenvalues equal scipy's, and every returned column is a ``b``-orthonormal eigenvector."""
    a, b = pencil(n, kind)
    w = np.zeros(n)
    v = np.zeros((n, n), dtype=np.complex128)
    numba_solvers[kind].solve_pencil(a.copy(), b.copy(), w, v)
    np.testing.assert_allclose(w, scipy.linalg.eigh(a, b, eigvals_only=True), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(a @ v, (b @ v) * w[None, :], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(v.conj().T @ b @ v, np.eye(n), rtol=1e-8, atol=1e-8)
