"""Leaf AST builders, literal readers and name rewriters shared by every numpyto_common pass and backend
(imports nothing of the package)."""

import ast
import copy
from collections.abc import Mapping

__all__ = [
    "RenameNames",
    "SubstituteLoads",
    "const_int",
    "expr_of",
    "literal_loads",
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


def const_int(node: ast.AST | None) -> int | None:
    """``node`` as a Python int, or ``None``. Accepts a signed literal (``-1`` parses as a UnaryOp)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = const_int(node.operand)
        if inner is not None:
            return -inner if isinstance(node.op, ast.USub) else inner
    return None


class SubstituteLoads(ast.NodeTransformer):
    """Replace every LOAD of a name in ``values`` with a fresh copy of its expression."""

    __slots__ = ("values",)

    def __init__(self, values: Mapping[str, ast.expr]) -> None:
        self.values = values

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.values:
            return ast.copy_location(copy.deepcopy(self.values[node.id]), node)
        return node


def literal_loads(values: Mapping[str, object]) -> SubstituteLoads:
    """A :class:`SubstituteLoads` that reads each name in ``values`` as its literal."""
    return SubstituteLoads({name: ast.Constant(value=value) for name, value in values.items()})


class RenameNames(ast.NodeTransformer):
    """Rename every ``Name`` (load and store) and every parameter per ``renames``: a consistent
    alpha-renaming, so a lambda parameter keeps matching the names its body reads."""

    __slots__ = ("renames",)

    def __init__(self, renames: Mapping[str, str]) -> None:
        self.renames = renames

    def visit_Name(self, node: ast.Name) -> ast.AST:
        new = self.renames.get(node.id)
        return node if new is None else ast.copy_location(ast.Name(id=new, ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self.renames.get(node.arg, node.arg)
        return node
