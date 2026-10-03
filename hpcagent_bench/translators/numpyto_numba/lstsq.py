"""``np.linalg.lstsq`` with numpy's default cutoff, spelled so numba can type it."""

import ast

__all__ = [
    "CUTOFF",
    "LstsqRcond",
    "is_lstsq",
    "is_none",
    "rewrite_lstsq_rcond",
]

#: numpy's default ``rcond=None`` cutoff is ``eps * max(M, N)``; numba types only a float ``rcond``
#: (its ``-1.0`` default means plain ``eps``, a different truncation), so the cutoff is spelled out.
CUTOFF = "np.finfo({a}.dtype).eps * max({a}.shape[0], {a}.shape[1])"


def is_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def is_lstsq(func: ast.AST) -> bool:
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "lstsq"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "linalg"
    )


class LstsqRcond(ast.NodeTransformer):
    __slots__ = ()

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not is_lstsq(node.func) or not 2 <= len(node.args) <= 3:
            return node
        rcond = next((k for k in node.keywords if k.arg == "rcond"), None)
        if len(node.args) == 3:
            default = is_none(node.args[2])
        else:
            default = rcond is None or is_none(rcond.value)
        if not default:
            return node
        cutoff = ast.parse(CUTOFF.format(a=ast.unparse(node.args[0])), mode="eval").body
        node.args = node.args[:2]
        node.keywords = [k for k in node.keywords if k.arg != "rcond"] + [ast.keyword(arg="rcond", value=cutoff)]
        return node


def rewrite_lstsq_rcond(source: str) -> str:
    """``source`` with every default-cutoff ``np.linalg.lstsq`` given an explicit numpy-equal ``rcond``."""
    if "lstsq" not in source:
        return source
    return ast.unparse(ast.fix_missing_locations(LstsqRcond().visit(ast.parse(source))))
