"""Chained subscripts ``A[i][j]`` flattened into one subscript."""

import ast
import copy
from collections.abc import Mapping, Sequence

from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import iter_extent_of_
from hpcagent_bench.translators.numpyto_common.lowering.indexing import (
    compose_kept_axis,
    is_scalar_index,
    rebases_onto_view_axis,
    slice_dims,
)
from hpcagent_bench.translators.numpyto_common.lowering.shape_reads import is_newaxis, negative_literal_offset
from hpcagent_bench.translators.numpyto_common.ordered import OrderedSet
from hpcagent_bench.translators.numpyto_common.subscripts import is_ellipsis, is_full_slice

#: Base name a chained view's own axes are scalarized under before they compose onto the real base.
CHAINED_VIEW = "__chained_view__"


def counts_from_end(bound: ast.expr | None) -> bool:
    """A negated symbol, which numpy counts from an axis END and no offset arithmetic resolves."""
    return isinstance(bound, ast.UnaryOp) and isinstance(bound.op, ast.USub) and negative_literal_offset(bound) is None


#: Where one result axis of a subscript comes from, so two spellings compare axis for axis:
#: ``("axis", i)`` inner slice i, ``("trail", t)`` the t-th base axis after the inner's entries,
#: ``("rest", 0)`` every base axis after those, ``("adv", k)`` / ``("outer", k)`` the k-th broadcast
#: axis of the inner / outer index arrays, ``("new", i)`` the newaxis at outer entry i.
AxisLabel = tuple[str, int]
#: One subscript entry as numpy lays out its result: a ``"slice"`` or ``"newaxis"`` keeps its one
#: labelled axis, a ``"scalar"`` keeps none, an ``"array"`` keeps its right-aligned broadcast axes.
IndexEntry = tuple[str, tuple[AxisLabel, ...]]


def result_axes(entries: Sequence[IndexEntry]) -> list[AxisLabel] | None:
    """numpy's result-axis order for one subscript, or ``None`` when its index arrays do not line up.

    Slices and newaxes keep their axes in order. The broadcast axes of the index arrays sit where the
    advanced entries (arrays AND scalars) are when those form one unbroken run, and move to the FRONT
    once a slice or newaxis separates them.
    """
    kept = [axes[0] for kind, axes in entries if kind in ("slice", "newaxis")]
    arrays = [axes for kind, axes in entries if kind == "array"]
    if not arrays:
        return kept
    broadcast: list[AxisLabel] = []
    for offset in range(max(len(axes) for axes in arrays), 0, -1):
        labels = OrderedSet(axes[-offset] for axes in arrays if len(axes) >= offset)
        if len(labels) != 1:
            return None
        broadcast.extend(labels)
    advanced = [kind in ("scalar", "array") for kind, axes in entries]
    first = advanced.index(True)
    last = len(advanced) - 1 - advanced[::-1].index(True)
    if not all(advanced[first : last + 1]):
        return broadcast + kept
    before = sum(1 for kind, axes in entries[:first] if kind in ("slice", "newaxis"))
    return kept[:before] + broadcast + kept[before:]


def index_rank(elt: ast.expr, shape_table: Mapping[str, Sequence[str]]) -> int | None:
    """Rank of what a subscript entry selects with: 0 for one position, ``k`` for an index array of
    rank ``k``, ``None`` when a sized array feeds the entry but its result cannot be sized."""
    if not any(isinstance(n, ast.Name) and shape_table.get(n.id) for n in ast.walk(elt)):
        return 0
    extent = iter_extent_of_(elt, shape_table)
    if extent:
        return len(extent)
    if isinstance(elt, ast.Subscript) and isinstance(elt.value, ast.Name):
        dims = slice_dims(elt)
        if len(dims) == len(shape_table.get(elt.value.id) or ()) and all(
            not isinstance(d, ast.Slice)
            and not is_newaxis(d)
            and not is_ellipsis(d)
            and index_rank(d, shape_table) == 0
            for d in dims
        ):
            return 0  # every axis picked by one position: ``ijtoh[ih, jh]``
    return None


def reads_a_mask(elt: ast.expr, bool_names: frozenset[str] | set[str]) -> bool:
    """Whether an index entry may be a boolean mask, which selects by value rather than position."""
    return any(
        isinstance(n, (ast.Compare, ast.BoolOp)) or (isinstance(n, ast.Name) and n.id in bool_names)
        for n in ast.walk(elt)
    )


def compose_onto_view(view: ast.Slice, use: ast.expr, use_is_array: bool) -> ast.expr | None:
    """The entry for the base axis ``view`` kept, once ``use`` indexes that kept axis, or ``None``
    when a single entry cannot say it."""
    if is_full_slice(view):
        return use
    if is_full_slice(use):
        return view
    if isinstance(use, ast.Slice):
        return compose_kept_axis(view, use) if rebases_onto_view_axis(view, use) else None
    from_end = (isinstance(use, ast.UnaryOp) and isinstance(use.op, ast.USub)) or (
        isinstance(use, ast.Constant) and isinstance(use.value, int) and use.value < 0
    )
    if view.step is not None or from_end or not (use_is_array or is_scalar_index(use)):
        return None  # a stride, or a position counted from the view's END, is not ``lower + use``
    return use if view.lower is None else ast.BinOp(left=copy.deepcopy(view.lower), op=ast.Add(), right=use)


def entry_model(elt: ast.expr, rank: int, label: AxisLabel, broadcast_rank: int, family: str) -> IndexEntry:
    """The layout of one entry: a slice or newaxis keeps ``label``, an index array its right-aligned
    axes of the ``family`` broadcast."""
    if is_newaxis(elt):
        return ("newaxis", (label,))
    if isinstance(elt, ast.Slice):
        return ("slice", (label,))
    if rank == 0:
        return ("scalar", ())
    return ("array", tuple((family, broadcast_rank - rank + axis) for axis in range(rank)))


def index_slot(entries: list[ast.expr]) -> ast.expr:
    """The ``slice`` field for a subscript with ``entries``."""
    return entries[0] if len(entries) == 1 else ast.Tuple(elts=entries, ctx=ast.Load())


#: Field values that can hold no subscript, so a chain scan never descends into them.
LEAF_TYPES = frozenset(
    {type(None), str, int, float, complex, bool, bytes, type(Ellipsis), ast.Name, ast.Constant}
    | {
        leaf
        for family in (ast.expr_context, ast.operator, ast.unaryop, ast.cmpop, ast.boolop)
        for leaf in family.__subclasses__()
    }
)


def outermost_chains(tree: ast.AST) -> list[tuple[ast.AST, str, int | None, ast.Subscript]]:
    """``(parent, field, position, chain)`` for every ``A[i][j]`` under ``tree`` that no other chain
    contains; ``position`` is ``None`` for a single-node field."""
    found: list[tuple[ast.AST, str, int | None, ast.Subscript]] = []
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        values = vars(node)
        for field in node._fields:
            value = values.get(field)
            kind = type(value)
            if kind in LEAF_TYPES:
                continue
            for pos, item in enumerate(value) if kind is list else ((None, value),):
                if type(item) in LEAF_TYPES:
                    continue
                if type(item) is ast.Subscript and type(item.value) is ast.Subscript:
                    found.append((node, field, pos, item))
                elif isinstance(item, ast.AST):
                    stack.append(item)
    return found


class ChainedSubscriptFlattener(ast.NodeTransformer):
    """Rewrite a chained subscript ``A[inner][outer]`` into one that selects the same elements in the
    same axis order.

    Helper inlining leaves these where the source sliced a view and indexed it -- vexx_k's
    ``tabxx_qr[ia][:, ijtoh[ih, jh]]`` and ``becxx[:, jbnd, ikq][ikb]``. The outer entries index the
    inner's RESULT axes in order, and each composes with what made its axis: a bare ``:`` takes it
    as-is, a partial slice rebases it (``a[1:3][0]`` -> ``a[1 + 0]``), an index array's axis indexes
    INTO the array (``A[idx][j]`` -> ``A[idx[j]]``), and an axis after the inner's entries takes it
    appended (``psi[f][..., 0]`` -> ``psi[f, ..., 0]``).

    numpy moves the broadcast axes of advanced indices to the FRONT once a slice separates them, so
    ``A[2][:3, idx]`` (3, P) and ``A[2, :3, idx]`` (P, 3) differ. The flat form is emitted only when
    :func:`result_axes` lays both out the same. Otherwise the chain stays two-step with its outer
    slices folded inward, ``A[2, :3][:, idx]``, which numpy lays out like the original.

    ``explicit_trailing_axes`` spells out every base axis a known rank names (``A[i][j]`` on rank 3 ->
    ``A[i, j, :]``), for phases whose consumers look for an explicit ``Slice``. ``bool_names`` are
    masks: ``A[mask][j]`` is not ``A[mask[j]]``.
    """

    def __init__(
        self,
        shape_table: Mapping[str, Sequence[str]],
        *,
        bool_names: frozenset[str] | set[str] = frozenset(),
        explicit_trailing_axes: bool = False,
    ) -> None:
        self.shape_table = shape_table
        self.bool_names = bool_names
        self.explicit_trailing_axes = explicit_trailing_axes

    def visit(self, node: ast.AST) -> ast.AST:
        """Rewrite every chain under ``node``; only a chain's own subtree pays the transformer walk."""
        if type(node) is ast.Subscript and type(node.value) is ast.Subscript:
            return self.visit_Subscript(node)
        for parent, field, pos, chain in outermost_chains(node):
            rewritten = self.visit_Subscript(chain)
            if pos is None:
                vars(parent)[field] = rewritten
            else:
                vars(parent)[field][pos] = rewritten
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)  # bottom-up: a longer chain arrives with its inner already rewritten
        if not isinstance(node.value, ast.Subscript):
            return node
        rewritten = self.rewrite_chain(node, node.value)
        return node if rewritten is None else ast.copy_location(rewritten, node)

    def entry_ranks(self, elts: list[ast.expr]) -> list[int] | None:
        """:func:`index_rank` per entry, or ``None`` when one is unsized or may be a mask."""
        ranks: list[int] = []
        for elt in elts:
            if isinstance(elt, ast.Slice):
                ranks.append(0)
                continue
            # A mask local has no shape-table entry, so index_rank reports it as one position.
            rank = None if reads_a_mask(elt, self.bool_names) else index_rank(elt, self.shape_table)
            if rank is None:
                return None
            ranks.append(rank)
        return ranks

    def rewrite_chain(self, node: ast.Subscript, inner: ast.Subscript) -> ast.expr | None:
        """The single or folded two-step form of ``inner[outer]``, or ``None`` to keep the chain."""
        base = inner.value
        shape = self.shape_table.get(base.id) if isinstance(base, ast.Name) else None
        rank = len(shape) if shape else None
        inner_elts = slice_dims(inner)
        ellipses = [pos for pos, elt in enumerate(inner_elts) if is_ellipsis(elt)]
        if ellipses:
            if len(ellipses) > 1 or rank is None or rank < len(inner_elts) - 1:
                return None
            width = rank - len(inner_elts) + 1
            inner_elts[ellipses[0] : ellipses[0] + 1] = [ast.Slice() for axis in range(width)]
        if any(is_newaxis(elt) for elt in inner_elts) or (rank is not None and len(inner_elts) > rank):
            return None
        if self.explicit_trailing_axes and rank is not None:
            inner_elts.extend(ast.Slice() for axis in range(rank - len(inner_elts)))
        outer_elts = slice_dims(node)
        inner_ranks = self.entry_ranks(inner_elts)
        outer_ranks = self.entry_ranks(outer_elts)
        if inner_ranks is None or outer_ranks is None:
            return None
        if any(is_ellipsis(elt) for elt in outer_elts):
            # An Ellipsis spans an uncounted number of axes: only appending after a scalar inner is exact.
            if any(inner_ranks) or any(outer_ranks) or any(isinstance(elt, ast.Slice) for elt in inner_elts):
                return None
            return ast.Subscript(value=base, slice=index_slot([*inner_elts, *outer_elts]), ctx=node.ctx)

        adv_rank = max(inner_ranks, default=0)
        inner_model = [
            entry_model(elt, elt_rank, ("axis", pos), adv_rank, "adv")
            for pos, (elt, elt_rank) in enumerate(zip(inner_elts, inner_ranks))
        ]
        consuming = sum(1 for elt in outer_elts if not is_newaxis(elt))
        trail_count = rank - len(inner_elts) if rank is not None else consuming
        tail_labels: list[AxisLabel] = [("trail", t) for t in range(trail_count)]
        if rank is None:
            tail_labels.append(("rest", 0))
        inner_axes = result_axes([*inner_model, *(("slice", (label,)) for label in tail_labels)])
        if inner_axes is None or consuming > sum(1 for label in inner_axes if label[0] != "rest"):
            return None

        slots: list[ast.expr] = list(inner_elts)
        slot_models: list[IndexEntry] = list(inner_model)
        slot_newaxes: list[list[int]] = [[] for elt in inner_elts]
        tail: list[tuple[ast.expr, IndexEntry]] = []
        adv_uses: dict[int, tuple[ast.expr, IndexEntry]] = {}
        adv_newaxes: dict[int, list[int]] = {}
        folded_inner: list[ast.expr] = list(inner_elts)
        folded_tail: list[ast.expr] = []
        folded_outer: list[ast.expr] = list(outer_elts)
        folded = False
        flat_ok = True
        outer_rank = max(outer_ranks, default=0)
        outer_model: list[IndexEntry] = []
        pending: list[int] = []
        consumed = 0
        for i, (elt, elt_rank) in enumerate(zip(outer_elts, outer_ranks)):
            if is_newaxis(elt):
                outer_model.append(("newaxis", (("new", i),)))
                pending.append(i)
                continue
            label = inner_axes[consumed]
            consumed += 1
            self.attach_newaxes(pending, label, slot_newaxes, adv_newaxes, tail)
            pending.clear()
            model = entry_model(elt, elt_rank, label, outer_rank, "outer")
            outer_model.append(model)
            kind, index = label
            if kind == "adv":
                adv_uses[index] = (elt, model)
                continue
            if kind == "axis":
                composed = compose_onto_view(inner_elts[index], elt, elt_rank > 0)
                if composed is None:
                    flat_ok = False
                    continue
                slots[index] = composed
                slot_models[index] = model
                if isinstance(elt, ast.Slice) and not is_full_slice(elt):
                    folded_inner[index] = composed
            else:
                tail.append((elt, model))
                if isinstance(elt, ast.Slice) and not is_full_slice(elt):
                    folded_tail.extend(ast.Slice() for gap in range(index - len(folded_tail)))
                    folded_tail.append(elt)
            if isinstance(elt, ast.Slice) and not is_full_slice(elt):
                folded_outer[i] = ast.Slice()
                folded = True
        if pending:
            label = inner_axes[consumed] if consumed < len(inner_axes) else ("rest", 0)
            self.attach_newaxes(pending, label, slot_newaxes, adv_newaxes, tail)
        outer_model.extend(("slice", (label,)) for label in inner_axes[consumed:])
        expected = result_axes(outer_model)

        arrays = [pos for pos, elt_rank in enumerate(inner_ranks) if elt_rank > 0]
        if len(arrays) > 1 and any(not is_full_slice(use) for use, model in adv_uses.values()):
            flat_ok = flat_ok and self.same_broadcast_extents([inner_elts[pos] for pos in arrays])
        for pos in arrays if flat_ok else ():
            indexed = self.index_into_array(inner_elts[pos], inner_ranks[pos], adv_rank, adv_uses, adv_newaxes)
            if indexed is None:
                flat_ok = False
                break
            slots[pos], slot_models[pos] = indexed
        if flat_ok and expected is not None:
            entries: list[ast.expr] = []
            layout: list[IndexEntry] = []
            for pos, slot in enumerate(slots):
                entries.extend(ast.Constant(value=None) for i in slot_newaxes[pos])
                layout.extend(("newaxis", (("new", i),)) for i in slot_newaxes[pos])
                entries.append(slot)
                layout.append(slot_models[pos])
            entries.extend(entry for entry, model in tail)
            layout.extend(model for entry, model in tail)
            used_trails = sum(1 for label in inner_axes[:consumed] if label[0] == "trail")
            layout.extend(("slice", (label,)) for label in tail_labels[used_trails:])
            if result_axes(layout) == expected:
                return ast.Subscript(value=base, slice=index_slot(entries), ctx=node.ctx)
        if not folded:
            return None
        folded_base = ast.Subscript(value=base, slice=index_slot([*folded_inner, *folded_tail]), ctx=ast.Load())
        return ast.Subscript(value=folded_base, slice=index_slot(folded_outer), ctx=node.ctx)

    @staticmethod
    def attach_newaxes(
        pending: list[int],
        label: AxisLabel,
        slot_newaxes: list[list[int]],
        adv_newaxes: dict[int, list[int]],
        tail: list[tuple[ast.expr, IndexEntry]],
    ) -> None:
        """Place the outer newaxes that sit just before result axis ``label``."""
        kind, index = label
        if kind == "axis":
            slot_newaxes[index].extend(pending)
        elif kind == "adv":
            adv_newaxes.setdefault(index, []).extend(pending)
        else:
            tail.extend((ast.Constant(value=None), ("newaxis", (("new", i),))) for i in pending)

    def same_broadcast_extents(self, arrays: list[ast.expr]) -> bool:
        """Whether index arrays share every broadcast axis at one extent, so indexing one axis of each
        never indexes a length-1 axis that was only broadcast."""
        extents: list[tuple[ast.expr, ...]] = []
        for array in arrays:
            extent = iter_extent_of_(array, self.shape_table)
            if extent is None:
                return False
            extents.append(extent)
        for offset in range(1, max(len(extent) for extent in extents) + 1):
            if len(OrderedSet(ast.unparse(extent[-offset]) for extent in extents if len(extent) >= offset)) > 1:
                return False
        return True

    def index_into_array(
        self,
        array: ast.expr,
        array_rank: int,
        broadcast_rank: int,
        uses: dict[int, tuple[ast.expr, IndexEntry]],
        newaxes: dict[int, list[int]],
    ) -> tuple[ast.expr, IndexEntry] | None:
        """``array`` indexed by the outer entries that landed on its broadcast axes, with its layout."""
        entries: list[ast.expr] = []
        model: list[IndexEntry] = []
        for axis in range(array_rank):
            k = broadcast_rank - array_rank + axis
            entries.extend(ast.Constant(value=None) for i in newaxes.get(k, []))
            model.extend(("newaxis", (("new", i),)) for i in newaxes.get(k, []))
            use = uses.get(k)
            entries.append(ast.Slice() if use is None else copy.deepcopy(use[0]))
            model.append(("slice", (("adv", k),)) if use is None else use[1])
        labels = result_axes(model)
        if labels is None:
            return None
        while entries and is_full_slice(entries[-1]):
            entries.pop()
        layout: IndexEntry = ("array", tuple(labels)) if labels else ("scalar", ())
        if not entries:
            return array, layout
        indexed = ast.Subscript(value=array, slice=index_slot(entries), ctx=ast.Load())
        return (self.visit_Subscript(indexed) if isinstance(array, ast.Subscript) else indexed), layout
