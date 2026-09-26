# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Statement desugars every backend runs: element iteration over an array, chained assignment, tuple unpack.

dace, native lowering and jax each carried a copy of the first two; dace, native lowering and the C /
Fortran emitters each carried a tuple-unpack split. The copies differed in where an array's leading
extent comes from, in how a chained value reaches its targets, and in which right sides an unpack
spells per target; those are the parameters and hooks.

Entry points: :class:`DesugarArrayIteration`, :class:`SplitChainedAssign` and :class:`SplitTupleUnpack`.
"""

import ast
import copy
from collections.abc import Callable, Collection, Mapping

from hpcagent_bench.translators.numpyto_common.numpy_desugar import expr_rank, rank_table
from hpcagent_bench.translators.numpyto_common.ordered import OrderedSet

__all__ = [
    "STATEMENT_FIELDS",
    "STATEMENT_LISTS",
    "DesugarArrayIteration",
    "Spelled",
    "SplitChainedAssign",
    "SplitTupleUnpack",
    "StatementTransformer",
    "assign",
    "bind",
    "binding_names",
    "element_read",
    "indexed_loop",
    "is_plain_rebinding",
    "is_scalar_literal",
    "is_self_copy",
    "leaves_block",
    "load",
    "pair_names",
    "placed",
    "races",
    "read_name",
    "rebound_names",
    "rename_names",
    "statement_blocks",
    "written_name",
]


def is_scalar_literal(node: ast.AST) -> bool:
    """True iff the expression is numeric literals only -- provably a scalar, and folded by dace's frontend."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (bool, int, float, complex))
    if isinstance(node, ast.UnaryOp):
        return is_scalar_literal(node.operand)
    if isinstance(node, ast.BinOp):
        return is_scalar_literal(node.left) and is_scalar_literal(node.right)
    return False


def load(name: str) -> ast.Name:
    return ast.Name(id=name, ctx=ast.Load())


def bind(name: str, value: ast.expr) -> ast.Assign:
    """``name = value``."""
    return ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value)


def element_read(array: str, index: str) -> ast.Subscript:
    """``array[index]``."""
    return ast.Subscript(value=load(array), slice=load(index), ctx=ast.Load())


def pair_names(target: ast.expr) -> tuple[str, str] | None:
    """The two names of a ``x, y`` loop target, else ``None``."""
    if isinstance(target, ast.Tuple) and len(target.elts) == 2:
        first, second = target.elts
        if isinstance(first, ast.Name) and isinstance(second, ast.Name):
            return first.id, second.id
    return None


def indexed_loop(node: ast.For, index: str, extent: ast.expr, binds: list[ast.stmt]) -> ast.For:
    """``node`` respelled ``for index in range(extent): *binds; *node.body``, its ``else`` kept."""
    loop = ast.For(
        target=ast.Name(id=index, ctx=ast.Store()),
        iter=ast.Call(func=load("range"), args=[extent], keywords=[]),
        body=[*binds, *node.body],
        orelse=node.orelse,
    )
    return ast.fix_missing_locations(ast.copy_location(loop, node))


#: The fields that hold statements, in ``_fields`` order.
STATEMENT_LISTS = ("body", "orelse", "finalbody")
#: Those, and the handlers / match cases that hold more statements, in ``_fields`` order.
STATEMENT_FIELDS = ("body", "handlers", "orelse", "finalbody", "cases")


def statement_blocks(node: ast.AST) -> list[list[ast.AST]]:
    fields = vars(node)
    return [block for name in STATEMENT_FIELDS if isinstance(block := fields.get(name), list)]


class StatementTransformer(ast.NodeTransformer):
    """A transformer that walks statements only: no statement sits inside an expression, so skip them all."""

    def generic_visit(self, node: ast.AST) -> ast.AST:
        for block in statement_blocks(node):
            block[:] = [self.visit(child) for child in block]
        return node


class DesugarArrayIteration(StatementTransformer):
    """``for x in arr`` -> ``for i in range(<leading extent>): x = arr[i]``.

    Neither dace, C, Fortran nor a traced jax loop walks an array by element. ``extent_of`` gives an
    array's leading extent, or ``None`` to leave the loop alone; ``index_name`` names the index from
    the loop target and the number of loops rewritten before it.
    """

    __slots__ = ("extent_of", "index_name", "rewritten", "var_to_array")

    def __init__(self, extent_of: Callable[[str], ast.expr | None], index_name: Callable[[str, int], str]) -> None:
        self.extent_of = extent_of
        self.index_name = index_name
        self.rewritten = 0
        #: Loop variable -> the array it walks, so native lowering can give it the element dtype.
        self.var_to_array: dict[str, str] = {}

    def visit_For(self, node: ast.For) -> ast.For:
        self.generic_visit(node)
        if not (isinstance(node.iter, ast.Name) and isinstance(node.target, ast.Name)):
            return node
        extent = self.extent_of(node.iter.id)
        if extent is None:
            return node
        index = self.index_name(node.target.id, self.rewritten)
        self.rewritten += 1
        self.var_to_array[node.target.id] = node.iter.id
        return indexed_loop(node, index, extent, [bind(node.target.id, element_read(node.iter.id, index))])


def placed(origin: ast.stmt, stmts: list[ast.stmt]) -> list[ast.stmt]:
    return [ast.fix_missing_locations(ast.copy_location(stmt, origin)) for stmt in stmts]


def rename_names(node: ast.AST, names: Collection[str], holder: str, keep: Collection[int] = ()) -> None:
    """Respell every ``Name`` in ``node`` found in ``names`` as ``holder``, except the node ids in ``keep``."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in names and id(sub) not in keep:
            sub.id = holder


def binding_names(target: ast.expr) -> list[ast.Name]:
    """The names an assignment target binds: itself, or the elements of a tuple / list / starred target."""
    if isinstance(target, ast.Name):
        return [target]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for element in target.elts for name in binding_names(element)]
    if isinstance(target, ast.Starred):
        return binding_names(target.value)
    return []


def rebound_names(stmt: ast.stmt, watched: Collection[str], in_place: bool) -> OrderedSet[str]:
    """Names in ``watched`` that ``stmt`` binds anew anywhere inside it.

    ``x += v`` on an array writes in place, so it rebinds nothing when ``in_place``.
    """
    augmented: OrderedSet[int] = OrderedSet()
    rebound: OrderedSet[str] = OrderedSet()
    for sub in ast.walk(stmt):  # breadth-first: an AugAssign is met before its target
        if in_place and isinstance(sub, ast.AugAssign):
            augmented.add(id(sub.target))
        elif isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
            if sub.id in watched and id(sub) not in augmented:
                rebound.add(sub.id)
        elif isinstance(sub, ast.arg) and sub.arg in watched:
            rebound.add(sub.arg)
        elif isinstance(sub, (ast.FunctionDef, ast.ClassDef)) and sub.name in watched:
            rebound.add(sub.name)
    return rebound


def read_name(node: ast.AST, names: Collection[str]) -> str | None:
    """The name in ``names`` that ``node`` reads: a loaded ``Name``, or the ``Name`` an ``x += v`` updates."""
    if isinstance(node, ast.Name) and node.id in names and isinstance(node.ctx, ast.Load):
        return node.id
    if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id in names:
        return node.target.id
    return None


def is_plain_rebinding(stmt: ast.stmt, rebound: Collection[str]) -> bool:
    """``stmt`` is an assignment whose targets bind every name in ``rebound``, and nothing else in it does."""
    if not isinstance(stmt, ast.Assign):
        return False
    bound = [name for target in stmt.targets for name in binding_names(target) if name.id in rebound]
    stores = [
        sub
        for sub in ast.walk(stmt)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)) and sub.id in rebound
    ]
    return len(stores) == len(bound) and all(any(name.id == wanted for name in bound) for wanted in rebound)


def leaves_block(node: ast.AST) -> bool:
    """``node`` holds a ``break`` / ``continue`` that exits the block ``node`` sits in."""
    if isinstance(node, (ast.Break, ast.Continue)):
        return True
    if isinstance(node, (ast.expr, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return False
    if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        return any(leaves_block(stmt) for stmt in node.orelse)
    return any(leaves_block(child) for child in ast.iter_child_nodes(node))


class SplitChainedAssign(StatementTransformer):
    """``a = b = v`` -> one binding per target, ``v`` evaluated once, numpy's aliasing kept.

    * ``repeat_literals``: a numeric literal repeats at each target, which is the same value. dace
      needs it: issue 05 makes ``s0 = tmp`` a second name for ``tmp``'s container, so ``s0 = s1 = 0.0``
      through a temp collapses every accumulator onto one cell and over-counts by the unroll factor.
    * A rank-0 value goes through a temp named ``temp_name(k)``: ``t = v; a = t; b = t``.
    * Any other value may be an array, which numpy binds to every name as ONE buffer, and no emitter
      has a second name for a buffer. The first name target takes ``v`` and the later uses of the
      other names are renamed to it, up to a rebinding of any of them; a name still read past that
      point is bound to the shared name right there.

    ``seed_ranks`` seeds the rank table that tells a scalar from an array.
    """

    __slots__ = ("ranks", "repeat_literals", "scope", "seed_ranks", "temp_name", "temps")

    def __init__(
        self,
        temp_name: Callable[[int], str],
        repeat_literals: bool = False,
        seed_ranks: Mapping[str, int] | None = None,
    ) -> None:
        self.temp_name = temp_name
        self.repeat_literals = repeat_literals
        self.seed_ranks: dict[str, int] = dict(seed_ranks) if seed_ranks is not None else {}
        #: Temps minted so far.
        self.temps = 0
        #: Function (or module) whose names a rename may touch, and its rank table once needed.
        self.scope: ast.Module | ast.FunctionDef | None = None
        self.ranks: dict[str, int] | None = None

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.enter(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.enter(node)
        return node

    def enter(self, node: ast.Module | ast.FunctionDef) -> None:
        outer_scope, outer_ranks = self.scope, self.ranks
        self.scope, self.ranks = node, None
        self.generic_visit(node)
        self.scope, self.ranks = outer_scope, outer_ranks

    def generic_visit(self, node: ast.AST) -> ast.AST:
        fields = vars(node)
        for name in STATEMENT_LISTS:
            block = fields.get(name)
            if isinstance(block, list):
                self.split_block(block, node is self.scope)
        return super().generic_visit(node)

    def rank_of(self, value: ast.expr) -> int | None:
        if isinstance(value, ast.Constant):
            return 0
        if self.ranks is None:
            self.ranks = rank_table(self.scope, self.seed_ranks) if self.scope is not None else dict(self.seed_ranks)
        return expr_rank(value, self.ranks)

    def split_block(self, block: list[ast.stmt], scope_body: bool) -> None:
        position = 0
        while position < len(block):
            stmt = block[position]
            if isinstance(stmt, ast.Assign) and len(stmt.targets) > 1:
                position = self.split(block, position, stmt, scope_body)
            else:
                position += 1

    def split(self, block: list[ast.stmt], position: int, node: ast.Assign, scope_body: bool) -> int:
        """Replace the chained ``node`` at ``block[position]``; the position after its replacement."""
        if self.repeat_literals and is_scalar_literal(node.value):
            block[position : position + 1] = placed(
                node, [ast.Assign(targets=[target], value=copy.deepcopy(node.value)) for target in node.targets]
            )
            return position + len(node.targets)
        rank = self.rank_of(node.value)
        first = node.targets[0]
        if rank != 0 and isinstance(first, ast.Name):
            holder, targets, head = first.id, node.targets[1:], [ast.Assign(targets=[first], value=node.value)]
        else:
            holder = self.temp_name(self.temps)
            self.temps += 1
            targets, head = node.targets, [bind(holder, node.value)]
        aliases: list[str] = []
        for target in targets:
            if rank != 0 and isinstance(target, ast.Name):
                if target.id != holder and target.id not in aliases:
                    aliases.append(target.id)
                continue
            rename_names(target, aliases, holder)  # a later target sees the names bound before it
            head.append(ast.Assign(targets=[target], value=load(holder)))
        head = placed(node, head)
        block[position : position + 1] = head
        position += len(head)
        if aliases:
            skipped = OrderedSet(id(stmt) for stmt in head)
            self.rename_aliases(block, position, holder, aliases, rank is not None, scope_body, skipped)
        return position

    def rename_aliases(
        self,
        block: list[ast.stmt],
        position: int,
        holder: str,
        aliases: list[str],
        in_place: bool,
        scope_body: bool,
        skipped: Collection[int],
    ) -> None:
        """Respell ``aliases`` as ``holder`` from ``block[position]`` on, while they all still name one value."""
        live = list(aliases)
        while live and position < len(block):
            stmt = block[position]
            rebound = rebound_names(stmt, [holder, *live], in_place)
            if not rebound and (scope_body or not leaves_block(stmt)):
                rename_names(stmt, live, holder)
                position += 1
                continue
            if not (rebound and is_plain_rebinding(stmt, rebound)):
                block[position:position] = self.stitches(holder, live, skipped, stmt)
                return
            keep = OrderedSet(id(name) for target in stmt.targets for name in binding_names(target))
            rename_names(stmt, live, holder, keep)
            live = [name for name in live if name not in rebound]
            if holder in rebound:
                live = [name for name in live if name in self.reads(live, skipped)]
                if live:
                    block[position:position] = placed(stmt, [bind(live[0], load(holder))])
                    position += 1
                    holder, live = live[0], live[1:]
            position += 1
        if live and not scope_body:
            block[position:position] = self.stitches(holder, live, skipped, block[position - 1])

    def stitches(self, holder: str, live: list[str], skipped: Collection[int], origin: ast.stmt) -> list[ast.stmt]:
        """``name = holder`` for every alias still read where the rename did not reach."""
        read = self.reads(live, skipped)
        return placed(origin, [bind(name, load(holder)) for name in live if name in read])

    def reads(self, names: Collection[str], skipped: Collection[int]) -> OrderedSet[str]:
        """The ``names`` read anywhere in the scope outside the node ids in ``skipped``."""
        if self.scope is None:
            return OrderedSet(names)
        found: OrderedSet[str] = OrderedSet()
        pending: list[ast.AST] = [self.scope]
        wanted = len(OrderedSet(names))
        while pending and len(found) < wanted:  # every name found: nothing left to learn
            node = pending.pop()
            if id(node) in skipped:
                continue
            name = read_name(node, names)
            if name is not None:
                found.add(name)
            pending.extend(ast.iter_child_nodes(node))
        return found


#: Statements that run before the split bindings, and one value per target.
Spelled = tuple[list[ast.stmt], list[ast.expr]]


def is_self_copy(target: ast.expr, value: ast.expr) -> bool:
    """``n = n``: the binding writes back the value it reads."""
    return isinstance(target, ast.Name) and isinstance(value, ast.Name) and value.id == target.id


def written_name(target: ast.expr) -> str | None:
    """The name a target writes: itself, or the array a subscript stores into."""
    while isinstance(target, ast.Subscript):
        target = target.value
    return target.id if isinstance(target, ast.Name) else None


def races(targets: list[ast.expr], values: list[ast.expr], changed: list[int]) -> bool:
    """A changed value reads a name a changed target writes, so a sequential split could read it updated."""
    written = OrderedSet(written_name(targets[position]) for position in changed)
    return any(
        isinstance(sub, ast.Name) and sub.id in written for position in changed for sub in ast.walk(values[position])
    )


class SplitTupleUnpack(StatementTransformer):
    """``a, b = x, y`` -> ``a = x; b = y``, python's simultaneous bind kept.

    Python evaluates the whole right side before it binds any target. When a value reads a name a
    target writes, every changed value is latched in a temp first and the targets bind from the temps;
    a positional self-copy (``n = n``) writes back what it reads, so it binds plainly after them. A right
    side with no per-target spelling -- a call returning a tuple, a starred element -- stays whole.

    Per-backend hooks: :attr:`TARGETS`, :meth:`values` and :meth:`temp_name`.
    """

    __slots__ = ("racing", "temps")

    #: Target node types a split may bind; any other target leaves the statement whole.
    TARGETS: tuple[type[ast.expr], ...] = (ast.Name,)

    def __init__(self) -> None:
        #: Temps minted so far.
        self.temps = 0
        #: Racing statements met so far.
        self.racing = 0

    def values(self, targets: list[ast.expr], value: ast.expr) -> Spelled | None:
        """The per-target spelling of ``value``, or ``None`` when it has none."""
        return ([], value.elts) if isinstance(value, ast.Tuple) else None

    def temp_name(self, position: int) -> str | None:
        """The temp that latches the value at ``position``, or ``None`` to leave a racing statement whole."""
        return None

    def generic_visit(self, node: ast.AST) -> ast.AST:
        for block in statement_blocks(node):
            spliced: list[ast.AST] = []
            for stmt in block:
                if isinstance(stmt, ast.Assign):
                    spliced.extend(self.split(stmt))
                else:
                    spliced.append(self.generic_visit(stmt))
            block[:] = spliced
        return node

    def unpacked(self, node: ast.Assign) -> tuple[list[ast.expr], Spelled] | None:
        """The targets of the unpack ``node`` and one value per target, or ``None`` when it stays whole."""
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Tuple):
            return None
        targets = node.targets[0].elts
        if not all(isinstance(target, self.TARGETS) for target in targets):
            return None
        spelled = self.values(targets, node.value)
        if spelled is None or len(spelled[1]) != len(targets):
            return None
        return None if any(isinstance(value, ast.Starred) for value in spelled[1]) else (targets, spelled)

    def split(self, node: ast.Assign) -> list[ast.stmt]:
        """The bindings ``node`` splits into, or ``[node]`` when it stays whole."""
        unpacked = self.unpacked(node)
        if unpacked is None:
            return [node]
        targets, (prelude, values) = unpacked
        changed = [position for position, value in enumerate(values) if not is_self_copy(targets[position], value)]
        if not races(targets, values, changed):
            return placed(node, [*prelude, *map(assign, targets, values)])
        latched = self.latched(targets, values, changed)
        return [node] if latched is None else placed(node, [*prelude, *latched])

    def latched(self, targets: list[ast.expr], values: list[ast.expr], changed: list[int]) -> list[ast.stmt] | None:
        """Temps for the changed values, the changed targets bound from them, then the self-copies."""
        self.racing += 1
        holders: list[str] = []
        for position in changed:
            holder = self.temp_name(position)
            if holder is None:
                return None
            self.temps += 1
            holders.append(holder)
        changing = OrderedSet(changed)
        kept = [position for position in range(len(targets)) if position not in changing]
        return [
            *(bind(holder, values[position]) for holder, position in zip(holders, changed)),
            *(assign(targets[position], load(holder)) for holder, position in zip(holders, changed)),
            *(assign(targets[position], values[position]) for position in kept),
        ]


def assign(target: ast.expr, value: ast.expr) -> ast.Assign:
    """``target = value``."""
    return ast.Assign(targets=[target], value=value)
