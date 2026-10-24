"""Leaf AST builders, literal readers and name rewriters shared by every numpyto_common pass and backend
(imports nothing of the package)."""

import ast
import copy
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import EllipsisType

__all__ = [
    "ALL_BLOCK_FIELDS",
    "attach",
    "attached",
    "NESTED_BLOCK_FIELDS",
    "ConstantValue",
    "InlineDefinitions",
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
    "line_of",
    "range_for",
    "set_field",
    "store_",
    "walk_nodes",
]


def walk_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """:func:`ast.walk`, same nodes in the same order (children queued before the parent is yielded), reading
    ``_fields`` straight from the node instead of through ``ast.iter_child_nodes``."""
    todo: deque[ast.AST] = deque((node,))
    while todo:
        node = todo.popleft()
        fields = vars(node)
        for name in node._fields:
            value = fields.get(name)
            if isinstance(value, ast.AST):
                todo.append(value)
            elif isinstance(value, list):
                todo.extend(item for item in value if isinstance(item, ast.AST))
        yield node


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


class InlineDefinitions(SubstituteLoads):
    """:class:`SubstituteLoads` that also drops each name's single-target ``name = ...`` statement,
    dead once every load reads the expression instead."""

    __slots__ = ()

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in self.values:
            return None
        self.generic_visit(node)
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


def line_of(node: ast.AST) -> int | str:
    """``node``'s source line for an error message; ``"?"`` for a synthesized node that carries none."""
    if not isinstance(node, ast.expr | ast.stmt):
        return "?"
    try:
        return node.lineno
    except AttributeError:  # built by a pass and never given a location
        return "?"


def attach(node: ast.AST, key: str, value: str) -> None:
    """Attach metadata ``key`` to ``node``, the way the ``ast`` module keeps a node's location: it rides on the
    node, so every ``copy.deepcopy`` a later pass makes keeps it (a side table keyed by node would lose it)."""
    setattr(node, key, value)


def attached(node: ast.AST, key: str) -> str | None:
    """The metadata :func:`attach` put on ``node`` under ``key``, or None."""
    value = vars(node).get(key)
    return value if isinstance(value, str) else None


def set_field(node: ast.AST, field: str, value: object) -> None:
    """Replace ``node``'s field ``field`` by ``value``: the one place a pass writes an AST field it found by name
    (walking ``ast.iter_fields``). KeyError for a name that is not one of the node's fields."""
    if field not in node._fields:
        raise KeyError(f"{type(node).__name__} has no field {field!r}")
    setattr(node, field, value)


def map_blocks(
    stmt: ast.AST, rewrite: Callable[[list[ast.stmt]], list[ast.stmt]], fields: tuple[str, ...] = NESTED_BLOCK_FIELDS
) -> None:
    """Replace each statement list ``stmt`` holds under ``fields`` by ``rewrite`` of it, in place."""
    for field, value in ast.iter_fields(stmt):
        if field in fields and isinstance(value, list):
            set_field(stmt, field, rewrite(value))


def map_statement_lists(root: ast.AST, rewrite: Callable[[list[ast.stmt]], list[ast.stmt]]) -> None:
    """Replace every statement list anywhere under ``root`` (bodies, branches, handlers) by ``rewrite`` of it."""
    for node in walk_nodes(root):
        for field, value in ast.iter_fields(node):
            if isinstance(value, list) and any(isinstance(v, ast.stmt) for v in value):
                set_field(node, field, rewrite(value))
