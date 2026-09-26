"""Names the kernel uses in integer-only positions."""

import ast

__all__ = [
    "BITWISE_OPS",
    "INT_TRANSPARENT",
    "IntUses",
    "index_slots",
    "integer_positions",
    "names_used_as_int",
    "pure_int_arith",
    "shape_arguments",
]


def pure_int_arith(n: ast.AST) -> bool:
    """True when ``n`` is a value-preserving integer computation over Names
    and int literals: ``+ - * // %``, unary ``+ -``, and ``min``/``max``/
    ``abs`` (int in -> int out). Bounds the backward int-ness closure in
    :func:`names_used_as_int` so it never crosses a float divide, a
    transcendental call, or -- critically -- an ``int(...)`` truncation.

    ``int(...)`` is a barrier: an integer result says nothing about the type of its argument.
    """
    if isinstance(n, ast.Name):
        return True
    if isinstance(n, ast.Constant):
        return isinstance(n.value, int) and not isinstance(n.value, bool)
    if isinstance(n, ast.BinOp):
        return (
            isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod))
            and pure_int_arith(n.left)
            and pure_int_arith(n.right)
        )
    if isinstance(n, ast.UnaryOp):
        return isinstance(n.op, (ast.USub, ast.UAdd)) and pure_int_arith(n.operand)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("min", "max", "abs"):
        return all(pure_int_arith(a) for a in n.args)
    return False


#: Int-in/int-out calls, so int context flows backward through them.
INT_TRANSPARENT = frozenset({"min", "max", "abs"})

BITWISE_OPS = (ast.BitOr, ast.BitAnd, ast.BitXor, ast.LShift, ast.RShift)


def index_slots(node: ast.Subscript) -> list[ast.expr]:
    sl = node.slice
    return sl.elts if isinstance(sl, ast.Tuple) else [sl]


class IntUses:
    """Collects the names an expression feeds into an integer position, through arithmetic."""

    __slots__ = ("names",)

    def __init__(self) -> None:
        self.names: set[str] = set()

    def collect(self, node: ast.expr | None) -> None:
        if node is None:
            return
        if isinstance(node, ast.Name):
            self.names.add(node.id)
        elif isinstance(node, ast.BinOp):
            self.collect(node.left)
            self.collect(node.right)
        elif isinstance(node, ast.UnaryOp):
            self.collect(node.operand)
        elif isinstance(node, ast.Slice):
            # Every slice part is an integer position, the step included.
            self.collect(node.lower)
            self.collect(node.upper)
            self.collect(node.step)
        elif isinstance(node, ast.Subscript):
            # ``A[B[i]]``: the inner subscript is an int, so its base and index promote.
            self.collect(node.value)
            for e in index_slots(node):
                self.collect(e)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in INT_TRANSPARENT:
            # Only int-in/int-out calls: ``int(x)`` converts FROM a float.
            for arg in node.args:
                self.collect(arg)


def shape_arguments(node: ast.Call) -> list[ast.expr]:
    """Array-dimension expressions of a constructor or reshape call (each tuple element separately)."""
    if not isinstance(node.func, ast.Attribute):
        return []
    attr = node.func.attr
    shape_args: list[ast.expr] = []
    if attr in ("zeros", "empty", "ones", "full", "ndarray") and node.args:
        shape_args = [node.args[0]]
    elif attr == "reshape":
        base = node.func.value
        if isinstance(base, ast.Name) and base.id in ("np", "numpy"):
            if len(node.args) >= 2:  # np.reshape(a, newshape)
                shape_args = [node.args[1]]
        else:  # a.reshape(N, M)
            shape_args = list(node.args)
    shape_args += [kw.value for kw in node.keywords if kw.arg in ("shape", "newshape")]
    return [e for sh in shape_args for e in (sh.elts if isinstance(sh, (ast.Tuple, ast.List)) else [sh])]


def integer_positions(node: ast.AST) -> list[ast.expr]:
    """The expressions ``node`` places in an integer-only position: subscript indices, ``range``
    arguments, array dimensions, bitwise operands, ``//`` and ``%`` operands."""
    out: list[ast.expr] = []
    if isinstance(node, ast.Subscript):
        out += index_slots(node)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "range":
            out += node.args
        out += shape_arguments(node)
    if isinstance(node, ast.BinOp) and isinstance(node.op, BITWISE_OPS):
        out += [node.left, node.right]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        out.append(node.operand)
    if isinstance(node, ast.AugAssign) and isinstance(node.op, BITWISE_OPS):
        out += ([node.target] if isinstance(node.target, ast.Name) else []) + [node.value]
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.FloorDiv, ast.Mod)):
        out += [node.left, node.right]
    return out


def names_used_as_int(tree: ast.AST) -> set[str]:
    """Names that flow into an integer-only position, so implicit locals are typed ``int``.

    Closed backward over assignments whose right-hand side is pure integer arithmetic
    (:func:`pure_int_arith`), so ``b[n - k]`` promotes ``n`` and ``k`` and whatever they are
    computed from, but never across a float divide or an ``int(...)`` truncation.
    """
    uses = IntUses()
    for node in ast.walk(tree):
        for e in integer_positions(node):
            uses.collect(e)
    assigns = [
        (node.targets[0].id, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
    ]
    changed = True
    while changed:
        changed = False
        for name, rhs in assigns:
            if name in uses.names and pure_int_arith(rhs):
                before = len(uses.names)
                uses.collect(rhs)
                if len(uses.names) > before:
                    changed = True
    return uses.names
