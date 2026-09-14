"""Dtype kinds carried across helper calls, and the real/complex eigh Jacobi they decide.

The frontend lowers every ``eigh`` before helpers inline, so an operand built in a helper is typed only
through :func:`module_kind_tables`. A kind proven wrongly real drops an imaginary part; a real operand left
unproven gets the complex Jacobi, whose ``.real`` on a double C++ refuses (ls3df_scf's CPF forms).
"""

import ast

import pytest

from numpyto_common.numpy_desugar import (
    CallKinds,
    NO_CALLS,
    _EighCallHoister,
    _EighLoopRewriter,
    _dtype_kind,
    _dtype_table,
    _eigh_alias_names,
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
    aliases = _eigh_alias_names(tree)
    _EighCallHoister(aliases).visit(tree)
    ast.fix_missing_locations(tree)
    _EighLoopRewriter(aliases, seed, module_kind_tables(tree, "kernel", seed)).visit(tree)
    ast.fix_missing_locations(tree)
    rotate = next(fn for fn in tree.body if isinstance(fn, ast.FunctionDef) and fn.name == "rotate")
    text = ast.unparse(rotate)
    assert "__eigh" in text, text
    return not any(token in text for token in COMPLEX_ACCESSORS)


def test_a_helper_every_call_site_feeds_a_float_array_gets_the_real_jacobi() -> None:
    kernel = "def kernel(A, B):\n    X, w = rotate(A)\n    Z, v = rotate(Y=B)\n"
    assert rotate_is_real(kernel, {"A": "float", "B": "float"})


def test_a_helper_fed_float_at_one_site_and_complex_at_another_keeps_the_complex_jacobi() -> None:
    kernel = "def kernel(A, B):\n    X, w = rotate(A)\n    Z, v = rotate(B)\n"
    assert not rotate_is_real(kernel, {"A": "float", "B": "complex"})


def test_a_helper_called_with_an_unknown_argument_keeps_the_complex_jacobi() -> None:
    kernel = "def kernel(A, S):\n    X, w = rotate(A)\n    Z, v = rotate(S)\n"
    assert not rotate_is_real(kernel, {"A": "float"})


def test_a_parameter_fed_by_another_helpers_unpacked_return_resolves() -> None:
    kernel = "def split(A):\n    return A * 2.0, A[0]\n\ndef kernel(A):\n    X, w = split(A)\n    Z, v = rotate(X)\n"
    assert rotate_is_real(kernel, {"A": "float"})


def test_a_cycle_through_a_real_helper_resolves_from_its_one_known_site() -> None:
    """ls3df's shape: rayleigh_ritz reads the block its own result was filtered into."""
    kernel = (
        "def smooth(X):\n    return 0.5 * X\n\n"
        "def kernel(A, n):\n    X, w = rotate(A)\n    for _ in range(n):\n"
        "        Y = smooth(X)\n        X, w = rotate(Y)\n"
    )
    assert rotate_is_real(kernel, {"A": "float", "n": "int"})


def test_a_cycle_with_a_complex_leg_keeps_the_complex_jacobi() -> None:
    kernel = (
        "def twist(X):\n    return X * 1j\n\n"
        "def kernel(A, n):\n    X, w = rotate(A)\n    for _ in range(n):\n"
        "        Y = twist(X)\n        X, w = rotate(Y)\n"
    )
    assert not rotate_is_real(kernel, {"A": "float", "n": "int"})


def test_a_cycle_leg_the_kinds_cannot_read_retracts_the_assumption() -> None:
    """The first site is float, so the float assumption is made; the unreadable leg must take it back."""
    kernel = (
        "def kernel(A, n):\n    X, w = rotate(A)\n    for _ in range(n):\n"
        "        Y = X.unknown_method()\n        X, w = rotate(Y)\n"
    )
    assert not rotate_is_real(kernel, {"A": "float", "n": "int"})


def test_a_helper_passed_as_a_value_keeps_the_complex_jacobi() -> None:
    """Its other callers are not visible, so no site proves anything."""
    kernel = "def kernel(A):\n    X, w = rotate(A)\n    pairs = list(map(rotate, [A]))\n"
    assert not rotate_is_real(kernel, {"A": "float"})


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
    got = _dtype_kind(ast.parse(expr, mode="eval").body, dtypes)
    assert got == kind, got


def test_a_helper_call_inside_an_expression_takes_the_helpers_return_kind() -> None:
    """ls3df's ``Wf = hpsi(...).reshape(-1, k)``."""
    calls = CallKinds({"hpsi": "complex"}, {})
    assert _dtype_kind(ast.parse("hpsi(Y).reshape(-1, k)", mode="eval").body, {}, calls) == "complex"


@pytest.mark.parametrize(
    ("source", "seed", "table"),
    [
        ("a, b = x, z", {"x": "float", "z": "complex"}, {"a": "float", "b": "complex"}),
        ("w, U = np.linalg.eigh(z)", {"z": "complex"}, {"w": "float", "U": "complex"}),
        ("w, U = np.linalg.eigh(i)", {"i": "int"}, {"w": "float", "U": "float"}),
        ("w, U = np.linalg.eigh(s)", {}, {"w": "float"}),
    ],
)
def test_a_tuple_target_takes_each_elements_kind(source: str, seed: dict[str, str], table: dict[str, str]) -> None:
    got = _dtype_table(ast.parse(source), seed, NO_CALLS)
    assert {name: got[name] for name in got if name not in seed} == table, got


def test_a_parameter_holding_a_dtype_types_the_array_built_from_it() -> None:
    """ls3df's ``stencil_matrix(X.shape[0], X.dtype)`` builds ``np.zeros(..., dtype=dtype)``."""
    tree = ast.parse(
        "def build(n, dtype):\n    return np.zeros((n, n), dtype=dtype)\n\n"
        "def kernel(z, n):\n    M = build(n, z.dtype)\n"
    )
    tables = module_kind_tables(tree, "kernel", {"z": "complex", "n": "int"})
    assert tables["kernel"].get("M") == "complex", tables
