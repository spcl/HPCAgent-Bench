"""Array-method spellings (``X.sum()``) routed to their ``np.<name>(X)`` function form."""

import ast
from collections.abc import Iterable

#: Array methods whose numpy function twin takes the array first and means the same thing, so the
#: method spelling can be routed to it instead of growing a second branch. ``reshape`` is absent on
#: purpose: it has a method branch of its own that resolves a ``-1`` against the receiver.
ARRAY_METHOD_SHAPE_OPS: frozenset[str] = frozenset(
    {
        "copy",
        "transpose",
        "squeeze",
        "clip",
        "round",
        "conj",
        "conjugate",
        "cumsum",
        "cumprod",
        "take",
        "repeat",
        "diagonal",
        "swapaxes",
    }
)

#: Array REDUCTION methods with the same property. Kept beside the shape ops rather than merged
#: into them: a reduction CHANGES the rank and a shape op does not, and the sizer routes the two
#: differently. ``sort`` is absent on purpose -- the METHOD sorts in place and ``np.sort`` returns
#: a copy, so they are not the same call.
ARRAY_METHOD_REDUCTIONS: frozenset[str] = frozenset(
    {"sum", "prod", "mean", "max", "min", "argmax", "argmin", "std", "var", "any", "all", "ptp"}
)


class ArrayMethodRewriter(ast.NodeTransformer):
    """Normalize ``X.<m>(...)`` to ``np.<m>(X, ...)`` for every array method whose numpy function
    twin takes the array first (:data:`lib_nodes.ARRAY_METHOD_REDUCTIONS`).

    The method spelling only ever reached the expanders when the RECEIVER was a bare Name; an
    expression receiver walked straight through to the emitter, where correlation's
    ``((data - mean) ** 2).sum(axis=0)`` and nbody's ``(mass[i, 0] * vel[i, j]).sum()`` both died
    as "call to <expr>.sum not supported". One rewrite here serves every backend and every
    receiver shape, and the reduction expanders keep their single function-form path.

    A LOGICAL SPARSE receiver keeps its method: its buffers are not a dense array, and
    ``np.sum(A)`` over them would index a CSR triple as a 2-D matrix.
    """

    def __init__(self, sparse_names: Iterable[str] | None = None) -> None:
        self.sparse_names = set(sparse_names or ())

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in ARRAY_METHOD_REDUCTIONS):
            return node
        recv = func.value
        if isinstance(recv, ast.Name) and (recv.id in ("np", "numpy") or recv.id in self.sparse_names):
            return node
        if isinstance(recv, ast.Subscript) and isinstance(recv.value, ast.Name) and recv.value.id in self.sparse_names:
            return node
        return ast.copy_location(
            ast.Call(
                func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=func.attr, ctx=ast.Load()),
                args=[recv] + list(node.args),
                keywords=list(node.keywords),
            ),
            node,
        )
