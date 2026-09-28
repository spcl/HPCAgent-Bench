"""Leaf AST builders shared by every numpyto_common pass and backend (imports nothing of the package)."""

import ast

__all__ = [
    "expr_of",
    "numpy_attribute",
    "numpy_call",
]


def expr_of(src: str) -> ast.expr:
    """``src`` parsed as one expression."""
    return ast.parse(src, mode="eval").body


def numpy_attribute(attr: str) -> ast.Attribute:
    """Build ``np.<attr>``."""
    return ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=attr, ctx=ast.Load())


def numpy_call(fn: str, args: list[ast.expr]) -> ast.Call:
    """Build ``np.<fn>(*args)``."""
    return ast.Call(func=numpy_attribute(fn), args=args, keywords=[])
