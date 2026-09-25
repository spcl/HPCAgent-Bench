"""Tuple literals: subscript folding, local propagation, tuple-unpack splitting."""

import ast
import copy
from collections.abc import Mapping, Sequence

from hpcagent_bench.translators.numpyto_common.statement_desugar import Spelled, SplitTupleUnpack
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import const_or_name


class TupleSubscriptFolder(ast.NodeTransformer):
    """Fold ``(t1, t2, ..., tn)[K]`` to ``tk`` at lowering time so
    downstream passes don't see Tuple subscripts. Comes up when
    ``D.shape[-2]`` resolves to ``(Nqz, Nw, NA, NB, N3D, N3D)[-2]``
    after the shape harvest -- the Tuple is a constant-folded shape
    expression and the index picks one element."""

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Tuple):
            elts = node.value.elts
            idx_node = node.slice
            if isinstance(idx_node, ast.Constant) and isinstance(idx_node.value, int):
                idx = idx_node.value
                if idx < 0:
                    idx += len(elts)
                if 0 <= idx < len(elts):
                    return elts[idx]
            if (
                isinstance(idx_node, ast.UnaryOp)
                and isinstance(idx_node.op, ast.USub)
                and isinstance(idx_node.operand, ast.Constant)
                and isinstance(idx_node.operand.value, int)
            ):
                idx = -idx_node.operand.value + len(elts)
                if 0 <= idx < len(elts):
                    return elts[idx]
        return node


BLOCK_STMT_TYPES = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.If,
    ast.With,
    ast.AsyncWith,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
)


def fill_empty_blocks(tree: ast.AST) -> None:
    """Back-fill a ``pass`` into any compound-statement ``body`` emptied by
    statement removal -- an empty ``for`` / ``while`` / ``if`` body is invalid
    Python and fails the next parse / compile. An empty ``orelse`` (no ``else``
    clause) is left as-is; only the primary body must be non-empty."""
    for node in ast.walk(tree):
        if isinstance(node, BLOCK_STMT_TYPES):
            body = vars(node).get("body")
            if isinstance(body, list) and not body:
                filler = ast.Pass()
                ast.copy_location(filler, node)
                node.body = [filler]


class TupleLocalPropagator(ast.NodeTransformer):
    """Forward-substitute a local bound exactly once to a Tuple literal into its
    uses, then drop the now-dead assignment.

    A native backend has no runtime tuple: a ``shp = Y.shape`` that
    :class:`ShapeMidExpressionRewriter` folded to ``shp = (Lb, Lb, Lb, nstate)`` is
    a shape descriptor, and left as a bare Tuple assignment the emitter cannot lower
    it (and a ``np.reshape(x, shp)`` reading the bare Name would size ``x`` as a
    spurious 1-D ``(shp,)``). Inlining the tuple into ``shp[-1]`` and
    ``np.reshape(x, shp)`` -- which the tuple-subscript folder and reshape expander
    already lower -- resolves both.

    The substitution REPLAYS the element expressions (not a captured value) at each
    use site, so it is sound only when every name they read is stable. Three guards
    enforce that: (1) the tuple's own target is assigned exactly once; (2) every
    element is an integer-shape expression -- a Name, an ``int`` literal, or ``+ - *
    //`` arithmetic over those (a float / str constant or a Call is a genuine runtime
    value, not a shape, and is left alone); (3) every Name the elements read is itself
    assigned at most once in the scope (single-static-assign, hence one fixed value --
    a reassigned dim would make the replay pick up the wrong value). Dropping the dead
    assign never leaves an empty block: :func:`fill_empty_blocks` back-fills a
    ``pass`` if the tuple assign was a block's sole statement.
    """

    def __init__(self) -> None:
        self.tuples: dict[str, ast.Tuple] = {}

    @classmethod
    def is_dim(cls, elt: ast.expr) -> bool:
        if isinstance(elt, ast.Name):
            return True
        if isinstance(elt, ast.Constant):
            # A bool is an int subclass but not a dimension; exclude it.
            return isinstance(elt.value, int) and not isinstance(elt.value, bool)
        if isinstance(elt, ast.UnaryOp):
            return cls.is_dim(elt.operand)
        if isinstance(elt, ast.BinOp):
            return cls.is_dim(elt.left) and cls.is_dim(elt.right)
        return False

    def run(self, tree: ast.AST) -> "TupleLocalPropagator":
        store_counts: dict[str, int] = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                store_counts[n.id] = store_counts.get(n.id, 0) + 1
        for stmt in ast.walk(tree):
            if not (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Tuple)
                and store_counts.get(stmt.targets[0].id) == 1
                and stmt.value.elts
                and all(self.is_dim(e) for e in stmt.value.elts)
            ):
                continue
            target = stmt.targets[0].id
            reads = {n.id for e in stmt.value.elts for n in ast.walk(e) if isinstance(n, ast.Name)}
            # A reassigned element name is unstable; a self-referential tuple would
            # inline a Name whose defining assign we are about to drop.
            if target in reads or any(store_counts.get(name, 0) > 1 for name in reads):
                continue
            self.tuples[target] = stmt.value
        if self.tuples:
            self.visit(tree)
            fill_empty_blocks(tree)
        return self

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in self.tuples:
            return None
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.tuples:
            return copy.deepcopy(self.tuples[node.id])
        return node


class ShapeTableTupleSplit(SplitTupleUnpack):
    """Native lowering's tuple split: subscript targets too, and ``n, k = arr.shape`` from the shape table.

    * ``a, b, c = X, Y, Z`` -> three assignments (jacobi_2d_tile_4lvlsilly); ``KE[0], PE[0] = a, b``
      from a helper's tuple return stores each element into its array.
    * ``n, k = arr.shape`` -> the shape symbols, when ``arr``'s shape is known (thomas_solve,
      vertical_flux_prefix_scan).

    Integer names bound either way are collected in :attr:`int_locals` so the emitter declares them
    ``int`` before first use. A racing element (``a, b = b, a + b``, ``out[i], out[j] = out[j],
    out[i]``) goes through ``__swap<k>_<position>`` temps, dtyped by the later harvest phase from their
    right side: this split runs in ``normalize-calls``, before ``seed-dtypes-and-harvest``. A self-copy
    (``n, m = n, m`` after shape resolution) stays a plain binding, which keeps promote-params seeing
    ``n`` / ``m`` as scalar parameters; :class:`SelfAssignDropper` deletes it at the end of the phase.
    """

    TARGETS = (ast.Name, ast.Subscript)

    def __init__(self, arrays_shapes: Mapping[str, Sequence[str]]) -> None:
        super().__init__()
        self.arrays_shapes = arrays_shapes
        #: Names introduced as integer scalar locals, for the emitter to declare.
        self.int_locals: list[str] = []

    def values(self, targets: list[ast.expr], value: ast.expr) -> Spelled | None:
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        if len(names) != len(targets):
            return super().values(targets, value)
        if isinstance(value, ast.Attribute) and value.attr == "shape" and isinstance(value.value, ast.Name):
            return self.shape_values(names, value.value.id)
        if isinstance(value, ast.Tuple) and len(value.elts) == len(names) and all(map(is_int_constant, value.elts)):
            self.int_locals.extend(names)
        return super().values(targets, value)

    def shape_values(self, names: list[str], array: str) -> Spelled | None:
        """The shape symbols of ``array`` for ``names``, or ``None`` when its shape is unknown or of another rank."""
        shape = self.arrays_shapes.get(array)
        if shape is None or len(shape) != len(names):
            return None
        # A self-copy (``H = H``) is promoted to a shape PARAMETER, and ``integer_valued_locals`` pins
        # every ``kir.symbols`` name int anyway. Declaring it would shadow the parameter with an
        # uninitialized local.
        self.int_locals.extend(name for name, token in zip(names, shape) if name != token)
        return [], [const_or_name(token) for token in shape]

    def temp_name(self, position: int) -> str:
        return f"__swap{self.racing}_{position}"


def is_int_constant(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, int)
