"""Overlap between the cells a fused loop nest stores and the cells its right-hand side reads.

numpy evaluates the whole right-hand side of ``a[1:] = a[:-1] + 1.0`` before it stores anything. The
fused nest :class:`~numpyto_common.lowering.slice_fusion.SliceFusion` and
:class:`~numpyto_common.lowering.whole_array.WholeArrayAssignRewriter` build stores one element per
iteration instead, so a read of a cell an EARLIER iteration already stored sees the new value: the
shift above turns into a running sum. :func:`reads_a_stored_cell` says when that can happen, and
:func:`staged_through_temp` splits such a statement into two nests through a temp, which is what
numpy's own evaluation order amounts to.
"""

import ast
import copy
from collections.abc import Sequence

from hpcagent_bench.translators.numpyto_common.ast_build import name_, range_for, store_, walk_nodes
from hpcagent_bench.translators.numpyto_common.lib_nodes.dims import constant_difference
from hpcagent_bench.translators.numpyto_common.lowering.indexing import binop
from hpcagent_bench.translators.numpyto_common.parallelism import index_exprs

__all__ = ["OVERLAP_TEMP_PREFIX", "reads_a_stored_cell", "staged_through_temp", "temp_extents"]

#: Prefix of the temps :func:`staged_through_temp` evaluates an overlapping right-hand side into.
OVERLAP_TEMP_PREFIX = "__sfov"


def mentions(expr: ast.expr, names: set[str]) -> bool:
    return any(isinstance(n, ast.Name) and n.id in names for n in walk_nodes(expr))


def index_distance(read: ast.expr, write: ast.expr) -> int | None:
    """``read - write`` as an integer, or ``None`` when it is not one constant for every iteration."""
    if ast.dump(read) == ast.dump(write):
        return 0
    return constant_difference(ast.unparse(read), ast.unparse(write))


def reads_a_stored_cell(
    array: str, write_index: list[ast.expr], rhs: ast.expr, loops: Sequence[tuple[str, ast.expr, ast.expr]]
) -> bool:
    """True when some read of ``array`` in ``rhs`` may load a cell an earlier iteration stored.

    ``write_index`` is the per-axis index of the store, and ``loops`` the nest's ``(var, lo, hi)``,
    outermost first. Each axis index of the store must increase with its loop variable, as the
    slice-fusion and whole-array nests build it.

    A read is safe when it provably never looks back in iteration order:

    * on an axis the store holds fixed (no loop variable), a read index that is also fixed either
      names the same cells or disjoint ones: a constant nonzero distance proves them disjoint, and
      otherwise the axis decides nothing; a read index that moves with a loop variable is a
      transposed or gathered read and is not decided here;
    * on an axis a loop variable walks, a read index that does not move is disjoint from the store
      when it lies a constant distance outside the range the store walks (gaussian elimination reads
      pivot row ``k`` while it writes rows ``k + 1:``);
    * over the axes the loop variables walk, outermost first, the first nonzero distance must be
      positive -- the read is of a cell a LATER iteration stores, still unchanged -- and every
      distance before it must be zero. A distance that is not one integer (a reversed, strided or
      gathered read) is not decided here.

    Anything not decided is reported as a possible overlap: staging it costs a temp, while a wrong
    "safe" is a silent miscompile.
    """
    subscript_bases = {id(n.value) for n in walk_nodes(rhs) if isinstance(n, ast.Subscript)}
    for node in walk_nodes(rhs):
        if isinstance(node, ast.Name) and node.id == array and id(node) not in subscript_bases:
            # A bare whole-array read inside an element expression: nothing to compare.
            return True
        if not (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == array):
            continue
        read_index = index_exprs(node)
        if len(read_index) != len(write_index):
            return True
        if any(isinstance(d, ast.Slice) for d in read_index):
            return True
        if not read_is_safe(read_index, write_index, loops):
            return True
    return False


def read_is_safe(
    read_index: list[ast.expr], write_index: list[ast.expr], loops: Sequence[tuple[str, ast.expr, ast.expr]]
) -> bool:
    """One read of :func:`reads_a_stored_cell`, against the store it shares an array with."""
    moving = {var for var, unused_lo, unused_hi in loops}
    walked: list[tuple[ast.expr, ast.expr]] = []
    for r, w in zip(read_index, write_index, strict=True):
        if mentions(w, moving):
            if not mentions(r, moving) and outside_walked_range(r, w, loops):
                return True  # a fixed row the store never reaches
            walked.append((r, w))
        elif mentions(r, moving):
            return False
        elif index_distance(r, w) not in (0, None):
            return True  # a different fixed row: no cell in common
    for r, w in walked:
        d = index_distance(r, w)
        if d is None or d < 0:
            return False
        if d > 0:
            return True
    return True


def outside_walked_range(read: ast.expr, write: ast.expr, loops: Sequence[tuple[str, ast.expr, ast.expr]]) -> bool:
    """True when the fixed index ``read`` lies a constant distance below the store's first index on
    this axis or above its last one. ``write`` must move with exactly one loop variable."""
    walking = [(var, lo, hi) for var, lo, hi in loops if mentions(write, {var})]
    if len(walking) != 1:
        return False
    var, lo, hi = walking[0]
    first = substituted(write, var, lo)
    last = substituted(write, var, binop(copy.deepcopy(hi), ast.Sub(), ast.Constant(value=1)))
    below = constant_difference(ast.unparse(read), ast.unparse(first))
    above = constant_difference(ast.unparse(read), ast.unparse(last))
    return (below is not None and below < 0) or (above is not None and above > 0)


def substituted(expr: ast.expr, var: str, value: ast.expr) -> ast.expr:
    """``expr`` with every read of ``var`` replaced by ``value``."""

    class Substitute(ast.NodeTransformer):
        __slots__ = ()

        def visit_Name(self, node: ast.Name) -> ast.expr:
            return copy.deepcopy(value) if node.id == var else node

    return Substitute().visit(copy.deepcopy(expr))


def staged_through_temp(
    store: ast.Subscript,
    rhs: ast.expr,
    aug_op: ast.operator | None,
    loops: Sequence[tuple[str, ast.expr, ast.expr]],
    temp: str,
) -> list[ast.stmt]:
    """Two nests over ``loops`` (``(var, lo, hi)``, outermost first): the first evaluates ``rhs``
    into ``temp``, the second stores ``temp`` (or combines it under ``aug_op``) into ``store``.

    ``temp`` is indexed by each loop variable's position ``var - lo``, so its extents are
    ``hi - lo`` -- :func:`temp_extents` -- whatever the store's own offsets and strides are."""
    pos: list[ast.expr] = []
    for var, lo, unused in loops:
        is_zero = isinstance(lo, ast.Constant) and lo.value == 0
        pos.append(name_(var) if is_zero else binop(name_(var), ast.Sub(), copy.deepcopy(lo)))
    slot: ast.expr = pos[0] if len(pos) == 1 else ast.Tuple(elts=pos, ctx=ast.Load())
    fill: list[ast.stmt] = [
        ast.Assign(targets=[ast.Subscript(value=store_(temp), slice=slot, ctx=ast.Store())], value=rhs)
    ]
    read_back = ast.Subscript(value=name_(temp), slice=copy.deepcopy(slot), ctx=ast.Load())
    put: list[ast.stmt] = [
        ast.Assign(targets=[store], value=read_back)
        if aug_op is None
        else ast.AugAssign(target=store, op=aug_op, value=read_back)
    ]
    for var, lo, hi in reversed(loops):
        fill = [range_for(var, [copy.deepcopy(lo), copy.deepcopy(hi)], fill)]
        put = [range_for(var, [copy.deepcopy(lo), copy.deepcopy(hi)], put)]
    return [*fill, *put]


def temp_extents(loops: Sequence[tuple[str, ast.expr, ast.expr]]) -> tuple[str, ...]:
    """Shape tokens of the temp :func:`staged_through_temp` indexes: one ``hi - lo`` per loop."""
    out: list[str] = []
    for unused, lo, hi in loops:
        is_zero = isinstance(lo, ast.Constant) and lo.value == 0
        out.append(ast.unparse(hi if is_zero else binop(hi, ast.Sub(), lo)))
    return tuple(out)
