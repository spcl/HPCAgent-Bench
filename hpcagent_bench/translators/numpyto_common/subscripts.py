"""Predicates over numpy subscript entries, shared by the frontend, the lowering and the backends."""

import ast


def is_full_slice(e: ast.AST) -> bool:
    """A bare ``:`` entry: a whole-axis selection, the same as omitting the axis."""
    return isinstance(e, ast.Slice) and e.lower is None and e.upper is None and e.step is None


def base_name(node: ast.AST) -> str | None:
    """The variable an assignment target or a read names: ``x`` for ``x`` and for ``x[i, j]``."""
    while isinstance(node, ast.Subscript):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def index_slot(entries: list[ast.expr]) -> ast.expr:
    """The ``slice`` field for a subscript with ``entries``."""
    return entries[0] if len(entries) == 1 else ast.Tuple(elts=entries, ctx=ast.Load())


def is_ellipsis(e: ast.AST) -> bool:
    """A ``...`` entry (``ast.Constant(Ellipsis)``). It expands to full slices over every otherwise-unindexed
    axis, so it drops no axis."""
    return isinstance(e, ast.Constant) and e.value is Ellipsis


def is_newaxis(e: ast.AST) -> bool:
    """Every spelling of a subscript newaxis: ``None``, ``np.newaxis`` and a bare ``newaxis``."""
    return (
        (isinstance(e, ast.Constant) and e.value is None)
        or (isinstance(e, ast.Attribute) and e.attr == "newaxis")
        or (isinstance(e, ast.Name) and e.id == "newaxis")
    )


def has_slice_subscript(expr: ast.AST) -> bool:
    """True when ``expr`` contains a Subscript with a literal ``ast.Slice`` axis."""
    for sub in ast.walk(expr):
        if isinstance(sub, ast.Subscript):
            sl = sub.slice
            if isinstance(sl, ast.Slice):
                return True
            if isinstance(sl, ast.Tuple) and any(isinstance(e, ast.Slice) for e in sl.elts):
                return True
    return False
