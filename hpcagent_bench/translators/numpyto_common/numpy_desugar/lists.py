"""Python lists grown by ``append`` folded into sized arrays."""

import ast
import copy
from dataclasses import dataclass

from hpcagent_bench.translators.numpyto_common.ordered import OrderedSet
from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import const_int

__all__ = [
    "ListSegment",
    "SubstituteLen",
    "appended_elts",
    "bound_fits",
    "cut_target",
    "fold_list_accumulators",
    "growth_loop",
    "integer_expression",
    "is_len_of",
    "list_build_statements",
    "list_display_elts",
    "mutation_count",
    "name_count",
    "offset_add",
    "plan_list_build",
    "range_growth",
    "statement_blocks",
]


def list_display_elts(node: ast.AST) -> list[ast.expr] | None:
    """A 1-D ``[e0, e1, ...]`` display -> its elements; a nested display or a star is refused."""
    if not isinstance(node, ast.List) or any(isinstance(e, (ast.List, ast.Tuple, ast.Starred)) for e in node.elts):
        return None
    return list(node.elts)


def is_len_of(node: ast.AST, name: str) -> bool:
    """``len(name)`` -- the list's running length."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == name
    )


class SubstituteLen(ast.NodeTransformer):
    """``len(name)`` -> the fill index: the element landing at ``i`` was appended at length ``i``."""

    def __init__(self, name: str, index: str) -> None:
        self.name = name
        self.index = index

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        return ast.Name(id=self.index, ctx=ast.Load()) if is_len_of(node, self.name) else node


def appended_elts(stmt: ast.stmt, name: str) -> list[ast.expr] | None:
    """``name.append(e)``, ``name += [e, ...]`` or ``name = name + [e, ...]`` -> the appended elements."""
    if (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "append"
        and isinstance(stmt.value.func.value, ast.Name)
        and stmt.value.func.value.id == name
        and len(stmt.value.args) == 1
        and not stmt.value.keywords
    ):
        return [stmt.value.args[0]]
    if (
        isinstance(stmt, ast.AugAssign)
        and isinstance(stmt.op, ast.Add)
        and isinstance(stmt.target, ast.Name)
        and stmt.target.id == name
    ):
        return list_display_elts(stmt.value)
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id == name
        and isinstance(stmt.value, ast.BinOp)
        and isinstance(stmt.value.op, ast.Add)
        and isinstance(stmt.value.left, ast.Name)
        and stmt.value.left.id == name
    ):
        return list_display_elts(stmt.value.right)
    return None


def range_growth(stmt: ast.stmt, name: str) -> tuple[str, ast.expr, list[ast.expr]] | None:
    """``for v in range(E): <grows of name>`` -> ``(v, E, elements per trip)``."""
    if not (isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name) and not stmt.orelse):
        return None
    trips = stmt.iter
    if not (
        isinstance(trips, ast.Call)
        and isinstance(trips.func, ast.Name)
        and trips.func.id == "range"
        and len(trips.args) == 1
        and not trips.keywords
    ):
        return None
    per: list[ast.expr] = []
    for sub in stmt.body:
        grown = appended_elts(sub, name)
        if grown is None:
            return None
        per.extend(grown)
    return (stmt.target.id, trips.args[0], per) if per else None


def growth_loop(stmt: ast.stmt, name: str) -> tuple[ast.expr, ast.expr] | None:
    """``while len(name) < bound: <one-element grow>`` -> ``(bound, element)``."""
    if not (isinstance(stmt, ast.While) and not stmt.orelse and len(stmt.body) == 1):
        return None
    test = stmt.test
    if not (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Lt)
        and is_len_of(test.left, name)
    ):
        return None
    grown = appended_elts(stmt.body[0], name)
    if grown is None or len(grown) != 1:
        return None
    return test.comparators[0], grown[0]


def cut_target(stmt: ast.stmt | None, name: str, bound: ast.expr) -> str | None:
    """``target = name[:bound]`` -> ``target``: a cut to the length the growth loop was bounded by."""
    if not (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(stmt.value, ast.Subscript)
        and isinstance(stmt.value.value, ast.Name)
        and stmt.value.value.id == name
    ):
        return None
    cut = stmt.value.slice
    if (
        isinstance(cut, ast.Slice)
        and cut.lower is None
        and cut.step is None
        and cut.upper is not None
        and ast.dump(cut.upper) == ast.dump(bound)
    ):
        return stmt.targets[0].id
    return None


def mutation_count(node: ast.AST, name: str) -> int:
    """Stores to ``name`` plus ``name.append`` calls -- every mutation the fold has to account for."""
    total = 0
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)) and sub.id == name:
            total += 1
        elif (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "append"
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id == name
        ):
            total += 1
    return total


def name_count(node: ast.AST, name: str) -> int:
    return sum(1 for sub in ast.walk(node) if isinstance(sub, ast.Name) and sub.id == name)


def offset_add(offset: str, delta: str) -> str:
    """``offset + delta`` as source, folding literal + literal so a constant prefix stays one number."""
    if delta == "0":
        return offset
    if offset == "0":
        return delta
    if offset.isdigit() and delta.isdigit():
        return str(int(offset) + int(delta))
    return f"{offset} + {delta}"


def bound_fits(offset: str, bound: ast.expr) -> bool:
    """``offset <= bound`` is known, so ``while len(name) < bound`` leaves exactly ``bound`` elements.

    Python leaves ``max(offset, bound)``. An extent is never negative, so only an empty prefix fits
    a symbolic bound.
    """
    limit = const_int(bound)
    if limit is None:
        return offset == "0"
    return offset.isdigit() and int(offset) <= limit


def integer_expression(node: ast.expr, counters: frozenset[str]) -> bool:
    """Built from int literals and loop counters by integer-closed operators."""
    if isinstance(node, ast.Constant):
        return type(node.value) is int
    if isinstance(node, ast.Name):
        return node.id in counters
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.UAdd, ast.USub)) and integer_expression(node.operand, counters)
    if isinstance(node, ast.BinOp):
        return (
            isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod))
            and integer_expression(node.left, counters)
            and integer_expression(node.right, counters)
        )
    return False


@dataclass(frozen=True, slots=True)
class ListSegment:
    """Elements a build appends from offset ``base``.

    ``kind`` is ``"lit"`` (``elements`` in order), ``"for"`` (``elements`` per ``counter`` trip over
    ``range(bound)``) or ``"while"`` (``elements[0]`` at every index from ``base`` to ``bound``).
    """

    kind: str
    base: str
    elements: list[ast.expr]
    counter: str = ""
    bound: ast.expr | None = None


def plan_list_build(
    block: list[ast.stmt], start: int, fn: ast.FunctionDef, vectors: frozenset[str], index: str
) -> tuple[list[ast.stmt], int] | None:
    """The fold of the list bound at ``block[start]`` -> ``(replacement, end)``, or None to leave it.

    Recognized, in execution order::

        name = [e, ...]                       # seed display (possibly empty)
        name.append(e) / name += [e, ...]     # straight growth
        for v in range(E): name += [e, ...]   # a fixed stride per trip
        while len(name) < E: name.append(<expr of len(name)>)
        name = name[:E]                       # or ``other = name[:E]``, then ``name`` is never read

    Every length is an expression in the kernel's symbols, never data-dependent: a ``while`` with no
    cut leaves ``max(offset, E)`` elements and is refused unless ``offset <= E`` is known.
    """
    stmt = block[start]
    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
        return None
    name = stmt.targets[0].id
    seed = list_display_elts(stmt.value)
    if seed is None:
        return None
    segments = [ListSegment("lit", "0", seed)]
    offset = str(len(seed))
    length: str | None = None
    end = start + 1
    while end < len(block) and length is None:
        grown = appended_elts(block[end], name)
        if grown is not None:
            segments.append(ListSegment("lit", offset, grown))
            offset = offset_add(offset, str(len(grown)))
            end += 1
            continue
        trip = range_growth(block[end], name)
        if trip is not None:
            counter, trips, per = trip
            segments.append(ListSegment("for", offset, per, counter, trips))
            offset = offset_add(offset, f"{len(per)} * ({ast.unparse(trips)})")
            end += 1
            continue
        fill = growth_loop(block[end], name)
        if fill is None:
            break
        bound, step = fill
        segments.append(
            ListSegment("while", offset, [SubstituteLen(name, index).visit(copy.deepcopy(step))], index, bound)
        )
        cut = cut_target(block[end + 1] if end + 1 < len(block) else None, name, bound)
        if cut == name:
            length, end = f"({ast.unparse(bound)})", end + 2
        elif cut is not None and name_count(fn, name) == sum(name_count(s, name) for s in block[start : end + 2]):
            # The cut into a fresh name is the list's only reader: its length past ``bound`` is unobservable.
            length, end = f"({ast.unparse(bound)})", end + 1
        elif bound_fits(offset, bound):
            offset, end = f"({ast.unparse(bound)})", end + 1
        else:
            return None
    pinned = length is not None
    if length is None:
        length = offset
    if length == "0" or (len(segments) == 1 and name not in vectors):
        return None
    if mutation_count(fn, name) != sum(mutation_count(s, name) for s in block[start:end]):
        return None
    reads = [e for segment in segments for e in segment.elements]
    reads.extend(segment.bound for segment in segments if segment.bound is not None)
    if any(name_count(e, name) for e in reads):
        return None  # an element or bound reading the list would read the preallocated buffer instead
    return list_build_statements(name, segments, length, pinned), end


def list_build_statements(name: str, segments: list[ListSegment], length: str, pinned: bool) -> list[ast.stmt]:
    """The planned build as an allocation plus stores; a cut length guards every store it may drop."""
    integral = all(integer_expression(e, frozenset({s.counter})) for s in segments for e in s.elements)
    lines = [f"{name} = np.zeros({length}, dtype=np.{'int64' if integral else 'float64'})"]
    fill = segments[-1]
    prefix = [e for s in segments[:-1] for e in s.elements]
    if (
        fill.kind == "while"
        and all(s.kind == "lit" for s in segments[:-1])
        and all(isinstance(e, ast.Constant) and isinstance(e.value, (int, float)) for e in prefix)
    ):
        # One loop over the whole length: literal slots pick their constant, the rest the growth rule.
        value = ast.unparse(fill.elements[0])
        for pos in range(len(prefix) - 1, -1, -1):
            value = f"({ast.unparse(prefix[pos])}) if {fill.counter} == {pos} else ({value})"
        lines.append(f"for {fill.counter} in range({length}):\n    {name}[{fill.counter}] = {value}")
        return ast.parse("\n".join(lines)).body
    for segment in segments:
        if segment.kind == "while":
            store = f"{name}[{segment.counter}] = {ast.unparse(segment.elements[0])}"
            lines.append(
                f"for {segment.counter} in range({segment.base}, ({ast.unparse(segment.bound)})):\n    {store}"
            )
            continue
        indent = "    " if segment.kind == "for" else ""
        if segment.kind == "for":
            lines.append(f"for {segment.counter} in range({ast.unparse(segment.bound)}):")
        stride = len(segment.elements)
        for k, element in enumerate(segment.elements):
            slot = offset_add(f"{stride} * {segment.counter}", str(k)) if segment.kind == "for" else str(k)
            slot = offset_add(segment.base, slot)
            store = f"{name}[{slot}] = {ast.unparse(element)}"
            lines.append(f"{indent}if {slot} < {length}:\n{indent}    {store}" if pinned else f"{indent}{store}")
    return ast.parse("\n".join(lines)).body


def statement_blocks(node: ast.AST) -> list[list[ast.stmt]]:
    """Every statement list under ``node``, so a list built inside a loop or a branch folds too."""
    blocks: list[list[ast.stmt]] = []
    for sub in ast.walk(node):
        for value in vars(sub).values():
            if isinstance(value, list) and any(isinstance(v, ast.stmt) for v in value):
                blocks.append(value)
    return blocks


def fold_list_accumulators(fn: ast.FunctionDef, vectors: frozenset[str] = frozenset()) -> None:
    """Rewrite a Python list built by growth into an array plus indexed stores, in every block of ``fn``.

    No emitter has a list type, and ``len`` of one is not a symbol, so a list left standing lowers
    with wrong extents. See :func:`plan_list_build` for the recognized builds. A name mutated anywhere
    the build does not account for is left alone, and so is a bare display unless the caller names it
    in ``vectors`` (``curve_fit``'s ``p0``, which the LM lowering indexes).
    """
    taken = OrderedSet(sub.id for sub in ast.walk(fn) if isinstance(sub, ast.Name))
    for block in statement_blocks(fn):
        start = 0
        while start < len(block):
            stmt = block[start]
            if not (isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.List)):
                start += 1
                continue
            # Numbered per function: two builds in sibling blocks must not share a fill index.
            number = 1
            while f"__la{number}" in taken:
                number += 1
            plan = plan_list_build(block, start, fn, vectors, f"__la{number}")
            if plan is None:
                start += 1
                continue
            replacement, end = plan
            taken.add(f"__la{number}")
            block[start:end] = replacement
            start += len(replacement)
    ast.fix_missing_locations(fn)
