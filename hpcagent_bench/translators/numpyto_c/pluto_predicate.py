# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""If-conversion for the Pluto emission: a value-dependent ``if`` inside a loop becomes predicated
assignments, ``t = c ? e : t``, so the loop keeps an affine domain and stays inside a scop.

A scop may not contain an ``if`` whose condition reads an array element or a float: clan cannot
put it in the iteration domain, so the whole enclosing loop (an argmax, a conditional sum) would
fall outside the scop. A ternary is part of the
statement's expression, which Pluto treats as an opaque read, so the domain stays affine.

Two forms, both exact:

* INLINE. When no test of the ``if`` reads anything its branches write, every assignment is guarded
  by the conjunction of its tests, evaluated in place, in source order. No new variable, so the
  loop's parallelism is untouched.
* FLAGGED. Otherwise re-evaluating a test after an earlier branch assignment could read the new
  value (an argmax tests ``a[i] > x`` and then writes ``x``). Each test is evaluated ONCE, where the
  ``if`` stood, into an integer flag, and every assignment is guarded by flags alone. The flags are
  scalars written every iteration, so Pluto sees their dependence and keeps that loop in order.

An ``if`` whose branches hold anything but assignments and nested ``if``s (a ``break``, a loop, a
call statement) is left as it is, and stays outside the scop.
"""

import ast
import copy
import itertools
from collections.abc import Callable, Iterable

#: Prefix of the flag locals the FLAGGED form introduces; a counter makes each name unique.
FLAG_PREFIX = "pluto_pred"


def base_name(node: ast.AST) -> str | None:
    """The variable an assignment target or a read names: ``x`` for ``x`` and for ``x[i, j]``."""
    while isinstance(node, ast.Subscript):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def names_read(expr: ast.AST) -> set[str]:
    """Every variable ``expr`` reads, arrays by name."""
    return {node.id for node in ast.walk(expr) if isinstance(node, ast.Name)}


def written(stmts: Iterable[ast.stmt]) -> set[str]:
    """Every variable the statements assign, arrays by name, through nested ``if``s."""
    out: set[str] = set()
    for stmt in stmts:
        if isinstance(stmt, ast.Assign):
            out |= {name for name in (base_name(t) for t in stmt.targets) if name}
        elif isinstance(stmt, ast.AugAssign):
            name = base_name(stmt.target)
            if name:
                out.add(name)
        elif isinstance(stmt, ast.If):
            out |= written(stmt.body) | written(stmt.orelse)
    return out


def tests_read(stmts: Iterable[ast.stmt]) -> set[str]:
    """Every variable a test of an ``if`` among the statements reads, through nested ``if``s."""
    out: set[str] = set()
    for stmt in stmts:
        if isinstance(stmt, ast.If):
            out |= names_read(stmt.test) | tests_read(stmt.body) | tests_read(stmt.orelse)
    return out


def convertible(stmts: Iterable[ast.stmt]) -> bool:
    """True when every statement is a single-target assignment, an augmented one, a ``pass``, or a
    nested ``if`` of the same kind."""
    for stmt in stmts:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or base_name(stmt.targets[0]) is None:
                return False
        elif isinstance(stmt, ast.AugAssign):
            if base_name(stmt.target) is None:
                return False
        elif isinstance(stmt, ast.If):
            if not (convertible(stmt.body) and convertible(stmt.orelse)):
                return False
        elif not isinstance(stmt, ast.Pass):
            return False
    return True


def load(target: ast.expr) -> ast.expr:
    """``target`` as a read: the value an unselected predicated assignment keeps."""
    node = copy.deepcopy(target)
    for sub in ast.walk(node):
        if hasattr(sub, "ctx"):
            sub.ctx = ast.Load()
    return node


def conj(guard: ast.expr | None, test: ast.expr) -> ast.expr:
    """``guard and test``, or ``test`` alone at the top."""
    return test if guard is None else ast.BoolOp(op=ast.And(), values=[copy.deepcopy(guard), copy.deepcopy(test)])


def negate(test: ast.expr) -> ast.expr:
    return ast.UnaryOp(op=ast.Not(), operand=copy.deepcopy(test))


def predicated(stmt: ast.stmt, guard: ast.expr) -> ast.Assign:
    """``stmt`` (an assignment) as ``target = guard ? value : target``."""
    if isinstance(stmt, ast.AugAssign):
        target, value = stmt.target, ast.BinOp(left=load(stmt.target), op=stmt.op, right=stmt.value)
    else:
        target, value = stmt.targets[0], stmt.value
    return ast.Assign(
        targets=[copy.deepcopy(target)], value=ast.IfExp(test=copy.deepcopy(guard), body=value, orelse=load(target))
    )


def inline(stmts: Iterable[ast.stmt], guard: ast.expr) -> list[ast.stmt]:
    """INLINE form: each assignment guarded by the conjunction of its tests, in source order."""
    out: list[ast.stmt] = []
    for stmt in stmts:
        if isinstance(stmt, ast.If):
            out += inline(stmt.body, conj(guard, stmt.test)) + inline(stmt.orelse, conj(guard, negate(stmt.test)))
        elif not isinstance(stmt, ast.Pass):
            out.append(predicated(stmt, guard))
    return out


class Flagger:
    """FLAGGED form: one integer flag per test, set where its ``if`` stood, guarding by flags alone."""

    def __init__(self, fresh: Callable[[], str]) -> None:
        self.fresh = fresh
        self.flags: list[str] = []

    def branch(self, stmts: Iterable[ast.stmt], guard: ast.expr | None) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for stmt in stmts:
            if isinstance(stmt, ast.If):
                out += self.split(stmt, guard)
            elif not isinstance(stmt, ast.Pass):
                out.append(predicated(stmt, guard) if guard is not None else stmt)
        return out

    def split(self, node: ast.If, guard: ast.expr | None) -> list[ast.stmt]:
        flag = self.fresh()
        self.flags.append(flag)
        bit = ast.IfExp(test=copy.deepcopy(node.test), body=ast.Constant(1), orelse=ast.Constant(0))
        value = bit if guard is None else ast.IfExp(test=copy.deepcopy(guard), body=bit, orelse=ast.Constant(0))
        taken = ast.Name(id=flag, ctx=ast.Load())
        others = negate(taken) if guard is None else conj(guard, negate(taken))
        return (
            [ast.Assign(targets=[ast.Name(id=flag, ctx=ast.Store())], value=value)]
            + self.branch(node.body, taken)
            + self.branch(node.orelse, others)
        )


def convert(node: ast.If, fresh: Callable[[], str], flags: list[str]) -> list[ast.stmt]:
    """``node`` as predicated assignments: INLINE when no test reads a branch's write, else FLAGGED."""
    if not (tests_read([node]) & written([node])):
        return inline(node.body, node.test) + inline(node.orelse, negate(node.test))
    flagger = Flagger(fresh)
    out = flagger.split(node, None)
    flags += flagger.flags
    return out


class IfConverter(ast.NodeTransformer):
    """Rewrites every convertible value-dependent ``if`` that sits inside a ``for`` loop."""

    def __init__(self, value_dependent: Callable[[ast.expr], bool], taken: set[str]) -> None:
        self.value_dependent = value_dependent
        self.depth = 0
        self.flags: list[str] = []
        counter = itertools.count()
        self.fresh = lambda: next(f"{FLAG_PREFIX}{n}" for n in counter if f"{FLAG_PREFIX}{n}" not in taken)

    def visit_For(self, node: ast.For) -> ast.For:
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1
        return node

    def visit_If(self, node: ast.If) -> object:
        if self.depth and self.value_dependent(node.test) and convertible(node.body) and convertible(node.orelse):
            return convert(node, self.fresh, self.flags)
        self.generic_visit(node)
        return node


def if_convert(tree: ast.AST, value_dependent: Callable[[ast.expr], bool]) -> list[str]:
    """Rewrite ``tree`` in place; returns the flag locals introduced, each an integer."""
    taken = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    converter = IfConverter(value_dependent, taken)
    converter.visit(tree)
    ast.fix_missing_locations(tree)
    return converter.flags
