"""Recognising references to the numpy module in a kernel AST."""

import ast

__all__ = [
    "CONJ_ATTRS",
    "NUMPY_MODULE_NAMES",
    "REAL_IMAG_ATTRS",
    "is_numpy_module",
    "numpy_call_attr",
    "numpy_func_attr",
    "numpy_submodule_attr",
]

#: The names a kernel binds numpy to.
NUMPY_MODULE_NAMES = ("np", "numpy")

#: ``np.conj`` / ``np.conjugate``.
CONJ_ATTRS = frozenset({"conj", "conjugate"})

#: ``np.real`` / ``np.imag``.
REAL_IMAG_ATTRS = frozenset({"real", "imag"})


def is_numpy_module(node: ast.AST) -> bool:
    """``np`` or ``numpy`` as a bare name."""
    return isinstance(node, ast.Name) and node.id in NUMPY_MODULE_NAMES


def numpy_func_attr(func: ast.AST) -> str | None:
    """``np.<attr>`` / ``numpy.<attr>`` -> ``attr``, anything else -> ``None``."""
    if isinstance(func, ast.Attribute) and is_numpy_module(func.value):
        return func.attr
    return None


def numpy_call_attr(node: ast.AST) -> str | None:
    """The ``attr`` of an ``np.<attr>(...)`` / ``numpy.<attr>(...)`` call, else ``None``."""
    return numpy_func_attr(node.func) if isinstance(node, ast.Call) else None


def numpy_submodule_attr(node: ast.AST, submodule: str) -> str | None:
    """``np.<submodule>.<attr>(...)`` call (``np.fft.fft``, ``np.linalg.solve``) -> ``attr``, else ``None``."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == submodule
        and is_numpy_module(node.func.value.value)
    ):
        return node.func.attr
    return None
