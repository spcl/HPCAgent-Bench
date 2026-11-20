# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dtype kinds carried across helper calls, and the real/complex eigh Jacobi they decide.

The frontend lowers every ``eigh`` before helpers inline, so an operand built in a helper is typed only
through :func:`module_kind_tables`. A kind proven wrongly real drops an imaginary part; a real operand left
unproven gets the complex Jacobi, whose ``.real`` on a double C++ refuses (ls3df_scf's CPF forms).
"""

import ast

import pytest

from hpcagent_bench.translators.numpyto_common.numpy_desugar import (
    EighCallHoister,
    EighLoopRewriter,
    dtype_kind,
    eigh_alias_names,
    module_kind_tables,
)

ROTATE = """
import numpy as np

def rotate(Y):
    M = Y.conj().T @ Y
    w, U = np.linalg.eigh(M)
    return Y @ U, w
"""

COMPLEX_ACCESSORS = (".real", ".imag", "np.real(", "np.imag(", "np.conj")


def rotate_is_real(kernel: str, seed: dict[str, str]) -> bool:
    """Whether ``rotate``'s eigh lowers to the real Jacobi, run the way the frontend runs it."""
    tree = ast.parse(ROTATE + kernel)
    aliases = eigh_alias_names(tree)
    EighCallHoister(aliases).visit(tree)
    ast.fix_missing_locations(tree)
    EighLoopRewriter(aliases, seed, module_kind_tables(tree, "kernel", seed)).visit(tree)
    ast.fix_missing_locations(tree)
    rotate = next(fn for fn in tree.body if isinstance(fn, ast.FunctionDef) and fn.name == "rotate")
    text = ast.unparse(rotate)
    assert "__eigh" in text, text
    return not any(token in text for token in COMPLEX_ACCESSORS)


def test_a_helper_fed_float_at_one_site_and_complex_at_another_keeps_the_complex_jacobi() -> None:
    kernel = "def kernel(A, B):\n    X, w = rotate(A)\n    Z, v = rotate(B)\n"
    assert not rotate_is_real(kernel, {"A": "float", "B": "complex"})


def test_a_parameter_fed_by_another_helpers_unpacked_return_resolves() -> None:
    kernel = "def split(A):\n    return A * 2.0, A[0]\n\ndef kernel(A):\n    X, w = split(A)\n    Z, v = rotate(X)\n"
    assert rotate_is_real(kernel, {"A": "float"})


@pytest.mark.parametrize(
    ("expr", "dtypes", "kind"),
    [
        ("max(a, b)", {"a": "float", "b": "float"}, "float"),
        ("min(a, i)", {"a": "float", "i": "int"}, None),
        ("float(z)", {}, "float"),
        ("int(x)", {}, "int"),
        ("np.tensordot(a, z, axes=1)", {"a": "float", "z": "complex"}, "complex"),
        ("np.moveaxis(z, 0, 1)", {"z": "complex"}, "complex"),
        ("np.eye(n)", {}, "float"),
        ("np.eye(n, dtype=z.dtype)", {"z": "complex"}, "complex"),
        ("np.sign(z)", {"z": "complex"}, "complex"),
        ("np.diag(z, 1)", {"z": "complex"}, "complex"),
        ("np.linalg.norm(z)", {"z": "complex"}, "float"),
        ("np.linalg.eigvalsh(z)", {"z": "complex"}, "float"),
        ("np.zeros(n, dtype=dt)", {"dt": "complex"}, "complex"),
    ],
)
def test_numpy_and_builtin_calls_have_their_numpy_kind(expr: str, dtypes: dict[str, str], kind: str | None) -> None:
    got = dtype_kind(ast.parse(expr, mode="eval").body, dtypes)
    assert got == kind, got
