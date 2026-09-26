"""Recognising references to the numpy module in a kernel AST."""

import ast

__all__ = ["CONJ_ATTRS", "NUMPY_MODULE_NAMES", "REAL_IMAG_ATTRS", "is_numpy_module"]

#: The names a kernel binds numpy to.
NUMPY_MODULE_NAMES = ("np", "numpy")

#: ``np.conj`` / ``np.conjugate``.
CONJ_ATTRS = frozenset({"conj", "conjugate"})

#: ``np.real`` / ``np.imag``.
REAL_IMAG_ATTRS = frozenset({"real", "imag"})


def is_numpy_module(node: ast.AST) -> bool:
    """``np`` or ``numpy`` as a bare name."""
    return isinstance(node, ast.Name) and node.id in NUMPY_MODULE_NAMES
