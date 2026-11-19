# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``rows = np.nonzero(mask)[0]`` and the row-wise statements after it, as a guarded loop over the mask.

The selection has a data-dependent length, which no static extent can size. When every later statement of the
block works row by row on ``rows`` -- it indexes arrays with ``rows`` on their first axis, combines the rows
elementwise or multiplies them by a matrix on the right -- the block is the same computation done one selected row at
a time:

    rows = np.nonzero(m)[0]                     __nzm1 = m
    h = x[rows] @ w.T + b              ->       for __nzr1 in range(__nzm1.shape[0]):
    out[rows] += p[rows, e][:, None] * h            if __nzm1[__nzr1]:
                                                        h = x[__nzr1] @ w.T + b
                                                        out[__nzr1] += p[__nzr1, e] * h

``nonzero`` yields each selected row once, in increasing order, so a fancy-index store (``out[rows] += v``) and the
per-row store agree. Anything the rewrite cannot prove row-wise -- a reduction over the rows, ``rows`` read in any
other position, a row-local read after the block, an operand whose rank could broadcast against the row axis -- is
refused by name, so the kernel fails as loudly as before instead of computing something else.
"""

import ast
import copy
from collections.abc import Iterator

import numpy as np

from hpcagent_bench import modules
from hpcagent_bench.translators.numpyto_common.ast_build import name_, set_field, store_
from hpcagent_bench.translators.numpyto_common.emit_helpers.numpy_names import numpy_call_attr
from hpcagent_bench.translators.numpyto_common.lib_nodes.registry import ELEMENTWISE_SHAPE_OPS
from hpcagent_bench.translators.numpyto_common.numpy_desugar.ranks import expr_rank, rank_table

__all__ = ["NonzeroRowsToGuardedLoop", "selection_mask"]


def selection_mask(value: ast.expr) -> ast.expr | None:
    """``m`` of ``np.nonzero(m)[0]``, ``np.where(m)[0]`` or ``np.flatnonzero(m)``; None for anything else."""
    if isinstance(value, ast.Call) and numpy_call_attr(value) == "flatnonzero" and len(value.args) == 1:
        return value.args[0]
    if not (isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Constant) and value.slice.value == 0):
        return None
    call = value.value
    if isinstance(call, ast.Call) and numpy_call_attr(call) in {"nonzero", "where"} and len(call.args) == 1:
        return call.args[0]
    return None


def loads(node: ast.AST) -> Iterator[str]:
    """Every name ``node`` reads."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            yield sub.id


class RowRefusal(NotImplementedError):
    """The block is not provably row-wise; the message says which part."""


class RowRewriter:
    """Rewrites one block's statements for a single selected row ``row``, tracking which values carry the row axis."""

    __slots__ = ("locals_", "ranks", "row", "rows")

    def __init__(self, rows: str, row: str, ranks: dict[str, int]) -> None:
        self.rows = rows
        self.row = row
        self.ranks = ranks
        #: Row-local name -> its rank WITH the row axis (what numpy saw).
        self.locals_: dict[str, int] = {}

    def statement(self, stmt: ast.stmt) -> ast.stmt:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            value, rank = self.expr(stmt.value)
            if rank is None:
                raise RowRefusal(f"{stmt.targets[0].id} = ... does not depend on the selected rows")
            self.locals_[stmt.targets[0].id] = rank
            return ast.Assign(targets=[store_(stmt.targets[0].id)], value=value)
        if isinstance(stmt, (ast.Assign, ast.AugAssign)):
            target = stmt.targets[0] if isinstance(stmt, ast.Assign) else stmt.target
            if isinstance(stmt, ast.Assign) and len(stmt.targets) != 1:
                raise RowRefusal("a chained assignment")
            new_target, target_rank = self.expr(target)
            if target_rank is None or not isinstance(new_target, ast.Subscript):
                raise RowRefusal(f"a store to {ast.unparse(target)}, which is not indexed by the selected rows")
            new_target.ctx = ast.Store()
            value, rank = self.expr(stmt.value)
            if rank is not None and rank > target_rank:
                raise RowRefusal(f"{ast.unparse(stmt)} stores a value of more axes than its target")
            if isinstance(stmt, ast.Assign):
                return ast.Assign(targets=[new_target], value=value)
            return ast.AugAssign(target=new_target, op=stmt.op, value=value)
        raise RowRefusal(f"statement {type(stmt).__name__} after the selection")

    def expr(self, node: ast.expr) -> tuple[ast.expr, int | None]:
        """``node`` for one row, and its rank with the row axis if it carries the row axis (None: it does not)."""
        if isinstance(node, ast.Name):
            if node.id == self.rows:
                raise RowRefusal(f"{self.rows} read outside a first-axis index")
            if node.id in self.locals_:
                return name_(node.id), self.locals_[node.id]
            return copy.deepcopy(node), None
        if isinstance(node, ast.Constant):
            return copy.deepcopy(node), None
        if isinstance(node, ast.Subscript):
            return self.subscript(node)
        if isinstance(node, ast.Attribute):
            value, rank = self.expr(node.value)
            if rank is not None:
                raise RowRefusal(f"attribute .{node.attr} of a value over the selected rows")
            return ast.Attribute(value=value, attr=node.attr, ctx=ast.Load()), None
        if isinstance(node, ast.UnaryOp):
            operand, rank = self.expr(node.operand)
            return ast.UnaryOp(op=node.op, operand=operand), rank
        if isinstance(node, ast.BinOp):
            return self.binop(node)
        if isinstance(node, ast.Compare) and len(node.comparators) == 1:
            left, left_rank = self.expr(node.left)
            right, right_rank = self.expr(node.comparators[0])
            rank = self.elementwise(node, [(node.left, left_rank), (node.comparators[0], right_rank)])
            return ast.Compare(left=left, ops=node.ops, comparators=[right]), rank
        if isinstance(node, ast.Call):
            return self.call(node)
        raise RowRefusal(f"expression {type(node).__name__} over the selected rows")

    def subscript(self, node: ast.Subscript) -> tuple[ast.expr, int | None]:
        index = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if isinstance(index[0], ast.Name) and index[0].id == self.rows:
            base, base_rank = self.expr(node.value)
            full = expr_rank(node.value, self.ranks)
            if base_rank is not None or full is None:
                raise RowRefusal(f"{ast.unparse(node)}: the indexed array's rank is unknown")
            rest = [self.invariant(i) for i in index[1:]]
            elts: list[ast.expr] = [name_(self.row), *rest]
            if any(isinstance(i, ast.Slice) or self.is_newaxis(i) for i in rest):
                raise RowRefusal(f"{ast.unparse(node)}: a slice or newaxis beside the selected rows")
            rank = full - len(rest)
            return ast.Subscript(
                value=base, slice=elts[0] if len(elts) == 1 else ast.Tuple(elts, ast.Load()), ctx=ast.Load()
            ), rank
        value, inner = self.expr(node.value)
        if inner is None:
            return ast.Subscript(value=value, slice=self.invariant(node.slice), ctx=ast.Load()), None
        # ``v[:, None]`` of a row-carrying 1-D value: the ``:`` is the row axis, so one row is the scalar itself,
        # which broadcasts as the (1,) column did.
        if inner == 1 and len(index) == 2 and self.is_full_slice(index[0]) and self.is_newaxis(index[1]):
            return value, 2
        raise RowRefusal(f"{ast.unparse(node)}: an index into a value over the selected rows")

    def binop(self, node: ast.BinOp) -> tuple[ast.expr, int | None]:
        left, left_rank = self.expr(node.left)
        right, right_rank = self.expr(node.right)
        if isinstance(node.op, ast.MatMult):
            if right_rank is not None:
                raise RowRefusal(f"{ast.unparse(node)} contracts over the selected rows")
            if left_rank is None:
                return ast.BinOp(left=left, op=node.op, right=right), None
            if left_rank != 2:
                raise RowRefusal(f"{ast.unparse(node)}: a matrix product of a rank-{left_rank} row value")
            return ast.BinOp(left=left, op=node.op, right=right), self.matmul_rank(node.right)
        rank = self.elementwise(node, [(node.left, left_rank), (node.right, right_rank)])
        return ast.BinOp(left=left, op=node.op, right=right), rank

    def matmul_rank(self, right: ast.expr) -> int:
        """Rank of ``rows_matrix @ right``: a row of the product is a vector when ``right`` is a matrix."""
        rank = expr_rank(right, self.ranks)
        if rank != 2:
            raise RowRefusal(f"@ {ast.unparse(right)}: the right operand must be a matrix")
        return 2

    def elementwise(self, node: ast.expr, operands: list[tuple[ast.expr, int | None]]) -> int | None:
        """Rank of an elementwise combination; refused when an operand without the row axis could align with it."""
        carried = [pair[1] for pair in operands if pair[1] is not None]
        if not carried:
            return None
        top = max(carried)
        for operand, rank in operands:
            if rank is None and (expr_rank(operand, self.ranks) or 0) >= top:
                raise RowRefusal(f"{ast.unparse(node)}: {ast.unparse(operand)} could broadcast against the rows")
        return top

    def call(self, node: ast.Call) -> tuple[ast.expr, int | None]:
        pairs = [self.expr(a) for a in node.args]
        ranks = [pair[1] for pair in pairs]
        if node.keywords:
            if any(rank is not None for rank in ranks):
                raise RowRefusal(f"{ast.unparse(node)}: a call with keywords over the selected rows")
            return copy.deepcopy(node), None
        call = ast.Call(func=copy.deepcopy(node.func), args=[pair[0] for pair in pairs], keywords=[])
        if all(rank is None for rank in ranks):
            return call, None
        name = numpy_call_attr(node) or ""
        # a numpy ufunc is elementwise by definition (np.abs, np.exp, ...)
        if name not in ELEMENTWISE_SHAPE_OPS and not isinstance(modules.members(np).get(name), np.ufunc):
            raise RowRefusal(f"{ast.unparse(node)}: not an elementwise call over the selected rows")
        return call, self.elementwise(node, list(zip(node.args, ranks, strict=True)))

    def invariant(self, node: ast.expr) -> ast.expr:
        new, rank = self.expr(node) if not isinstance(node, ast.Slice) else (copy.deepcopy(node), None)
        if rank is not None:
            raise RowRefusal(f"{ast.unparse(node)}: an index that depends on the selected rows")
        return new

    @staticmethod
    def is_full_slice(node: ast.expr) -> bool:
        return isinstance(node, ast.Slice) and node.lower is None and node.upper is None and node.step is None

    @staticmethod
    def is_newaxis(node: ast.expr) -> bool:
        return (isinstance(node, ast.Constant) and node.value is None) or (
            isinstance(node, ast.Attribute) and node.attr == "newaxis"
        )


class NonzeroRowsToGuardedLoop:
    """Rewrite every ``rows = np.nonzero(m)[0]`` whose block's remaining statements are row-wise (module docstring).

    :param seed_ranks: the parameters' and scalars' ranks; locals are propagated from them.
    :raises NotImplementedError: a selection whose block is not row-wise, naming why.
    """

    __slots__ = ("counter", "ranks", "tree")

    def __init__(self, tree: ast.AST, seed_ranks: dict[str, int]) -> None:
        self.tree = tree
        self.ranks = rank_table(tree, seed_ranks)
        self.counter = 0

    def run(self) -> None:
        for node in list(ast.walk(self.tree)):
            for field, block in ast.iter_fields(node):
                if field not in ("body", "orelse"):
                    continue
                if isinstance(block, list) and block and all(isinstance(s, ast.stmt) for s in block):
                    set_field(node, field, self.block(block))

    def block(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        for at, stmt in enumerate(stmts):
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
                continue
            mask = selection_mask(stmt.value)
            if mask is None:
                continue
            if expr_rank(mask, self.ranks) != 1:
                raise NotImplementedError(f"{ast.unparse(stmt)}: only a 1-D mask's selection becomes a row loop")
            return [*stmts[:at], *self.loop(stmt.targets[0].id, mask, stmts[at + 1 :])]
        return stmts

    def loop(self, rows: str, mask: ast.expr, after: list[ast.stmt]) -> list[ast.stmt]:
        self.counter += 1
        held, row = f"__nzm{self.counter}", f"__nzr{self.counter}"
        rewriter = RowRewriter(rows, row, self.ranks)
        try:
            body = [rewriter.statement(s) for s in after]
        except RowRefusal as exc:
            raise NotImplementedError(f"np.nonzero selection {rows}: {exc}") from exc
        after_ids = {id(n) for s in after for n in ast.walk(s)}
        outside = {
            n.id
            for n in ast.walk(self.tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and id(n) not in after_ids
        }
        if leaked := sorted(outside & (set(rewriter.locals_) | {rows})):
            raise NotImplementedError(f"np.nonzero selection {rows}: {leaked} read outside the selection's block")
        written: set[str] = set()
        for stmt in after:
            # a row-local read before the block writes it would carry one row's value into the next
            if early := sorted((set(loads(stmt)) & set(rewriter.locals_)) - written):
                raise NotImplementedError(f"np.nonzero selection {rows}: {early} read before it is written")
            if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
                written.add(stmt.targets[0].id)
        guard = ast.If(test=ast.Subscript(value=name_(held), slice=name_(row), ctx=ast.Load()), body=body, orelse=[])
        extent = ast.Subscript(
            value=ast.Attribute(value=name_(held), attr="shape", ctx=ast.Load()), slice=ast.Constant(0), ctx=ast.Load()
        )
        return [
            ast.Assign(targets=[store_(held)], value=copy.deepcopy(mask)),
            ast.For(
                target=store_(row),
                iter=ast.Call(func=name_("range"), args=[extent], keywords=[]),
                body=[guard],
                orelse=[],
            ),
        ]
