"""``x.shape = ...`` rewritten to ``np.reshape``."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.ast_build import name_, numpy_attribute, store_

__all__ = ["ShapeAttrToReshape"]


class ShapeAttrToReshape(ast.NodeTransformer):
    """Rewrite in-place shape mutation ``x.shape = expr`` to
    ``x = np.reshape(x, expr)``.

    Numpy allows setting ``arr.shape = (N,)`` as an in-place view-
    reshape that returns the same data with a new shape. NumpyToC has
    no in-place attribute writes; the existing ``expand_reshape`` path
    already handles ``x = np.reshape(x, ...)`` so we just rewrite the
    LHS form into the function-call form.

    Also handles the chained form ``Xi.shape = Yi.shape = expr`` by
    splitting into per-target reshapes (mandelbrot2 canonical pattern).
    """

    __slots__ = ()

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        self.generic_visit(node)
        # Detect ``x.shape = expr`` -- one or more LHS targets that are
        # all ``Attribute(Name(x), 'shape')``.
        if not node.targets:
            return node
        shape_targets: list[ast.Name] = []
        for tgt in node.targets:
            if isinstance(tgt, ast.Attribute) and tgt.attr == "shape" and isinstance(tgt.value, ast.Name):
                shape_targets.append(tgt.value)
            else:
                return node
        if not shape_targets:
            return node
        # Normalise the new-shape expression: a bare integer ``N`` is
        # treated as ``(N,)`` (numpy quirk: arr.shape = N is a valid
        # 1-D reshape). A tuple stays as-is.
        new_shape = node.value
        if not isinstance(new_shape, ast.Tuple):
            new_shape = ast.Tuple(elts=[new_shape], ctx=ast.Load())
        # Emit one Assign per target ``x = np.reshape(x, new_shape)``.
        out: list[ast.stmt] = []
        for name in shape_targets:
            call = ast.Call(
                func=numpy_attribute("reshape"),
                args=[name_(name.id), copy.deepcopy(new_shape)],
                keywords=[],
            )
            out.append(ast.Assign(targets=[store_(name.id)], value=call))
        for s in out:
            ast.fix_missing_locations(s)
        return out
