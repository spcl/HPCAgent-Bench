"""Self-assignment removal and forward substitution of loop-invariant scalars."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.lowering.mathfuncs import MATH_INTRINSIC_NAMES
from hpcagent_bench.translators.numpyto_common.lowering.ssa import live_on_loop_reentry, read_in
from hpcagent_bench.translators.numpyto_common.lowering.tuples import fill_empty_blocks
from hpcagent_bench.translators.numpyto_common.statement_desugar import written_name


class SelfAssignDropper(ast.NodeTransformer):
    """Delete a tautological ``X = X`` scalar statement.

    Shape resolution mints these: ``ShapeMidExpressionRewriter`` rewrites the RHS of
    ``N = rank.shape[0]`` to the shape symbol ``N``, and the tuple split turns
    ``H, W = image.shape`` into ``H = H; W = W``. They compute nothing, and in the C
    pluto input they WRITE a signature parameter inside ``#pragma scop`` -- pet models
    that as a data-dependent condition and aborts polycc on an isl assert (POLYCC-003
    in :mod:`hpcagent_bench.pluto_affine`). Every other backend just carries dead text.

    Two names are never dropped. A kernel ARRAY (``a = a``) is a whole-array write that
    ``detect_output_and_index_arrays`` reads to flip ``is_output``, so deleting it would
    change the emitted signature. A name with a recorded shape reassignment is an SSA
    version marker the shape FIFO counts off. Only a bare ``Name = Name`` pair qualifies:
    ``A[i] = A[i]`` is a real store and drives the same output detection.
    """

    def __init__(self, keep: set[str]) -> None:
        self.keep = keep

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        tgt = node.targets[0]
        if (
            isinstance(tgt, ast.Name)
            and isinstance(node.value, ast.Name)
            and tgt.id == node.value.id
            and tgt.id not in self.keep
        ):
            return None
        return node


#: Expression nodes a substituted RHS may use; ``Slice`` is absent (that is an array).
FWD_SUBST_NODES: tuple[type, ...] = (
    ast.Constant,
    ast.Name,
    ast.Subscript,
    ast.Tuple,
    ast.BinOp,
    ast.UnaryOp,
    ast.IfExp,
    ast.Compare,
    ast.BoolOp,
)

#: Callees that recompute the same value wherever they are replayed.
FWD_SUBST_PURE_CALLS: set[str] = MATH_INTRINSIC_NAMES | {
    "min",
    "max",
    "abs",
    "int",
    "float",
    "int_floor",
    "python_mod",
}

#: Caps on the already-grown expression, so they bound a chain (conv_2d needs 1 / 9).
FWD_SUBST_MAX_SUBSCRIPTS = 3
FWD_SUBST_MAX_NODES = 20


def stmt_exprs_by_depth(stmts: list[ast.stmt], depth: int):
    """Yield ``(expr, loop_depth)`` per expression; a loop header sits OUTSIDE its own body."""
    for stmt in stmts:
        if isinstance(stmt, (ast.For, ast.While)):
            yield (stmt.iter if isinstance(stmt, ast.For) else stmt.test), depth
            yield from stmt_exprs_by_depth(stmt.body, depth + 1)
            yield from stmt_exprs_by_depth(stmt.orelse, depth)
        elif isinstance(stmt, ast.If):
            yield stmt.test, depth
            yield from stmt_exprs_by_depth(stmt.body, depth)
            yield from stmt_exprs_by_depth(stmt.orelse, depth)
        else:
            for child in ast.iter_child_nodes(stmt):
                if isinstance(child, ast.expr):
                    yield child, depth


def fwd_subst_is_pure(expr: ast.expr) -> bool:
    """True when ``expr`` recomputes the same value wherever it is replayed."""
    for node in ast.walk(expr):
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in FWD_SUBST_PURE_CALLS and not node.keywords):
                return False
        elif isinstance(node, ast.expr) and not isinstance(node, FWD_SUBST_NODES):
            return False
    return True


class ForwardSubstituteInvariantScalars(ast.NodeTransformer):
    """Replay a loop-invariant scalar's RHS at its deeper use sites and delete the assign.

    pet drops the assign and its deeper consumers then read an uninitialised value: conv_2d's
    ``w = w_box[di + R, dj + R]`` (POLYCC-001 in :mod:`hpcagent_bench.pluto_affine`). Two things
    fall out -- laundered indirection becomes literal, so the detector declines lavamd
    (POLYCC-006), and the OpenMP column loses a scalar ``--parallel`` never privatises (C-001).
    An AST copy keeps the grouping the assign had, so floating point stays bit-identical.
    ``qualifies`` carries the guards; one name is substituted per round against the CURRENT
    tree, so a chain (``first_i`` -> ``ai`` -> ``rv[...]``) resolves with no read left dangling.
    """

    def __init__(self, array_names: set[str], params: set[str]) -> None:
        self.array_names = array_names
        self.params = params
        #: Substituted name -> the replayed expression, in substitution order.
        self.substituted: dict[str, str] = {}
        self.stmt_: ast.Assign | None = None
        self.name_: str = ""
        self.expr_: ast.expr | None = None

    def run(self, fn: ast.FunctionDef) -> "ForwardSubstituteInvariantScalars":
        """Substitute every qualifying scalar in ``fn``; one assign leaves per round."""
        for unused in range(sum(1 for n in ast.walk(fn) if isinstance(n, ast.Assign))):
            found = self.first_candidate(fn)
            if found is None:
                break
            self.stmt_, self.name_, self.expr_ = found
            self.substituted[self.name_] = ast.unparse(self.expr_)
            self.visit(fn)
            fill_empty_blocks(fn)
        self.stmt_, self.name_, self.expr_ = None, "", None
        ast.fix_missing_locations(fn)
        return self

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        if node is self.stmt_:
            return None
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if self.expr_ is not None and node.id == self.name_ and isinstance(node.ctx, ast.Load):
            return copy.deepcopy(self.expr_)
        return node

    def first_candidate(self, fn: ast.FunctionDef) -> tuple[ast.Assign, str, ast.expr] | None:
        """The first ``(assign, name, rhs-copy)`` in source order that meets every condition."""
        store_counts: dict[str, int] = {}
        for_targets: set[str] = set()
        written: set[str] = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                store_counts[node.id] = store_counts.get(node.id, 0) + 1
                written.add(node.id)
            elif isinstance(node, ast.For):
                for tgt in ast.walk(node.target):
                    if isinstance(tgt, ast.Name):
                        for_targets.add(tgt.id)
            elif isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for tgt in targets:
                    base = written_name(tgt)
                    if base is not None:
                        written.add(base)
        # One pass, so the depth test is O(1) and gates the costly liveness walk.
        deepest: dict[str, int] = {}
        for expr, depth in stmt_exprs_by_depth(fn.body, 0):
            for node in ast.walk(expr):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and depth > deepest.get(node.id, -1):
                    deepest[node.id] = depth
        found: list[tuple[ast.Assign, str, ast.expr]] = []

        def qualifies(name: str, expr: ast.expr, depth: int, loop_vars: frozenset[str]) -> bool:
            # Scalar, in a loop, single-assigned, and read deeper than it is written. A
            # function-level assign is a whole-kernel constant, not the measured defect --
            # replaying deriche's exp() coefficients only pushes work down a nest.
            if not depth or name in self.array_names or name in self.params or name in for_targets:
                return False
            if store_counts.get(name, 0) != 1 or deepest.get(name, -1) <= depth:
                return False
            nodes = [n for n in ast.walk(expr) if isinstance(n, ast.expr)]
            subscripts = [n for n in nodes if isinstance(n, ast.Subscript)]
            if len(subscripts) > FWD_SUBST_MAX_SUBSCRIPTS or len(nodes) > FWD_SUBST_MAX_NODES:
                return False
            if not fwd_subst_is_pure(expr):
                return False
            for sub in subscripts:  # aliasing: a replayed read must not cross its own store
                base = written_name(sub)
                if base is None or base in written:
                    return False
            for node in ast.walk(expr):  # every operand stable between the assign and the reads
                if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
                    continue
                count = store_counts.get(node.id, 0)
                if count == 0 or node.id in loop_vars:
                    continue
                if not (count == 1 and node.id not in for_targets):
                    return False
            return True

        def scan(
            stmts: list[ast.stmt],
            depth: int,
            after: tuple[list[ast.stmt], ...],
            reentry: tuple[tuple[list[ast.stmt], int], ...],
            loop_vars: frozenset[str],
        ) -> None:
            for i, stmt in enumerate(stmts):
                if found:
                    return
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    name = stmt.targets[0].id
                    if qualifies(name, stmt.value, depth, loop_vars):
                        # Condition 6 last: it walks every block that runs after this one.
                        outside = after
                        for blk, idx in reentry:
                            outside = outside + live_on_loop_reentry(blk, idx, name)
                        if depth:
                            outside = outside + live_on_loop_reentry(stmts, i, name)
                        if not read_in(name, outside):
                            found.append((stmt, name, copy.deepcopy(stmt.value)))
                            return
                if not isinstance(stmt, (ast.For, ast.While, ast.If)):
                    continue
                tail = (stmts[i + 1 :],) + after
                # A block nested in a loop re-runs its own prefix on the next iteration.
                inner_reentry = (reentry + ((stmts, i),)) if depth else reentry
                if isinstance(stmt, ast.If):
                    # Counting the sibling branch as "after" only declines more.
                    scan(stmt.body, depth, (stmt.orelse,) + tail, inner_reentry, loop_vars)
                    scan(stmt.orelse, depth, (stmt.body,) + tail, inner_reentry, loop_vars)
                    continue
                body_vars = loop_vars
                if isinstance(stmt, ast.For):
                    body_vars = loop_vars | frozenset(n.id for n in ast.walk(stmt.target) if isinstance(n, ast.Name))
                scan(stmt.body, depth + 1, ((stmt.orelse,) + tail) if stmt.orelse else tail, inner_reentry, body_vars)
                scan(stmt.orelse, depth, tail, inner_reentry, loop_vars)

        scan(fn.body, 0, (), (), frozenset())
        return found[0] if found else None
