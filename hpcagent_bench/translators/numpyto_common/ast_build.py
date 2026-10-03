"""Leaf AST builders, literal readers and name rewriters shared by every numpyto_common pass and backend
(imports nothing of the package)."""

import ast
import copy
from collections.abc import Callable, Mapping, Sequence
from types import EllipsisType

__all__ = [
    "ALL_BLOCK_FIELDS",
    "NESTED_BLOCK_FIELDS",
    "ConstantValue",
    "RenameNames",
    "SubstituteLoads",
    "callee_attribute",
    "const_int",
    "expr_of",
    "literal_loads",
    "map_blocks",
    "map_statement_lists",
    "name_",
    "name_ids",
    "nested_blocks",
    "numpy_attribute",
    "numpy_call",
    "range_for",
    "store_",
]


def name_(name: str) -> ast.Name:
    """Build a load of ``name``."""
    return ast.Name(id=name, ctx=ast.Load())


def store_(name: str) -> ast.Name:
    """Build a store to ``name``."""
    return ast.Name(id=name, ctx=ast.Store())


def expr_of(src: str) -> ast.expr:
    """``src`` parsed as one expression."""
    return ast.parse(src, mode="eval").body


def numpy_attribute(attr: str) -> ast.Attribute:
    """Build ``np.<attr>``."""
    return ast.Attribute(value=name_("np"), attr=attr, ctx=ast.Load())


def numpy_call(fn: str, args: list[ast.expr]) -> ast.Call:
    """Build ``np.<fn>(*args)``."""
    return ast.Call(func=numpy_attribute(fn), args=args, keywords=[])


def callee_attribute(call: ast.Call) -> ast.Attribute:
    """The ``Attribute`` ``call`` dispatches on (``np.sum(x)`` -> ``np.sum``; ``x.sum()`` -> ``x.sum``); the
    dispatchers only route such calls here, so any other callee is a bug in the caller."""
    func = call.func
    if not isinstance(func, ast.Attribute):
        raise TypeError(f"expected an attribute call, got {ast.unparse(call)}")
    return func


def name_ids(elts: Sequence[ast.expr]) -> tuple[str, ...] | None:
    """The names of ``elts`` when every one is a bare Name, else ``None``."""
    ids = tuple(e.id for e in elts if isinstance(e, ast.Name))
    return ids if len(ids) == len(elts) else None


def const_int(node: ast.expr | None) -> int | None:
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


#: What an ``ast.Constant`` holds.
type ConstantValue = str | bytes | bool | int | float | complex | EllipsisType | None


def literal_loads(values: Mapping[str, ConstantValue]) -> SubstituteLoads:
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


def range_for(var: str, bounds: list[ast.expr], body: list[ast.stmt]) -> ast.For:
    """Build ``for <var> in range(*bounds): <body>``."""
    return ast.For(
        target=store_(var),
        iter=ast.Call(func=name_("range"), args=bounds, keywords=[]),
        body=body,
        orelse=[],
    )


#: The fields a compound statement keeps its nested statement lists in (``try`` adds ``finalbody``).
NESTED_BLOCK_FIELDS = ("body", "orelse")

#: Every field that holds a statement list directly on a statement.
ALL_BLOCK_FIELDS = (*NESTED_BLOCK_FIELDS, "finalbody")


def nested_blocks(node: ast.AST, fields: tuple[str, ...] = ALL_BLOCK_FIELDS) -> list[list[ast.stmt]]:
    """The statement lists ``node`` holds directly under ``fields``; none for a simple statement."""
    return [value for field, value in ast.iter_fields(node) if field in fields and isinstance(value, list)]


def map_blocks(
    stmt: ast.AST, rewrite: Callable[[list[ast.stmt]], list[ast.stmt]], fields: tuple[str, ...] = NESTED_BLOCK_FIELDS
) -> None:
    """Replace each statement list ``stmt`` holds under ``fields`` by ``rewrite`` of it, in place."""
    for field, value in ast.iter_fields(stmt):
        if field in fields and isinstance(value, list):
            setattr(stmt, field, rewrite(value))


def map_statement_lists(root: ast.AST, rewrite: Callable[[list[ast.stmt]], list[ast.stmt]]) -> None:
    """Replace every statement list anywhere under ``root`` (bodies, branches, handlers) by ``rewrite`` of it."""
    for node in ast.walk(root):
        for field, value in ast.iter_fields(node):
            if isinstance(value, list) and any(isinstance(v, ast.stmt) for v in value):
                setattr(node, field, rewrite(value))
