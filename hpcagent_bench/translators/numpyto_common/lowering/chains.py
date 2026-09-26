"""Chained subscripts ``A[i][j]`` flattened into one subscript."""

import ast
import copy
import dataclasses
from collections.abc import Mapping, Sequence

from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import iter_extent_of
from hpcagent_bench.translators.numpyto_common.lowering.indexing import (
    compose_kept_axis,
    is_scalar_index,
    rebases_onto_view_axis,
    slice_dims,
)
from hpcagent_bench.translators.numpyto_common.lowering.shape_reads import is_newaxis, negative_literal_offset
from hpcagent_bench.translators.numpyto_common.ordered import OrderedSet
from hpcagent_bench.translators.numpyto_common.subscripts import index_slot, is_ellipsis, is_full_slice

__all__ = [
    "CHAINED_VIEW",
    "LEAF_TYPES",
    "AxisLabel",
    "ChainFold",
    "ChainedSubscriptFlattener",
    "IndexEntry",
    "compose_onto_view",
    "counts_from_end",
    "entry_model",
    "flat_entries",
    "index_rank",
    "outermost_chains",
    "reads_a_mask",
    "result_axes",
]

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
    extent = iter_extent_of(elt, shape_table)
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
        inner_elts = self.inner_entries(inner, rank)
        if inner_elts is None:
            return None
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

        fold = ChainFold(
            inner_elts=inner_elts,
            slots=list(inner_elts),
            slot_models=list(inner_model),
            slot_newaxes=[[] for elt in inner_elts],
            folded_inner=list(inner_elts),
            folded_outer=list(outer_elts),
        )
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
            self.attach_newaxes(pending, label, fold.slot_newaxes, fold.adv_newaxes, fold.tail)
            pending.clear()
            model = entry_model(elt, elt_rank, label, outer_rank, "outer")
            outer_model.append(model)
            fold.place(i, elt, elt_rank, label, model)
        if pending:
            label = inner_axes[consumed] if consumed < len(inner_axes) else ("rest", 0)
            self.attach_newaxes(pending, label, fold.slot_newaxes, fold.adv_newaxes, fold.tail)
        outer_model.extend(("slice", (label,)) for label in inner_axes[consumed:])
        expected = result_axes(outer_model)
        self.index_inner_arrays(fold, inner_ranks, adv_rank)
        if fold.flat_ok and expected is not None:
            used_trails = sum(1 for label in inner_axes[:consumed] if label[0] == "trail")
            entries = flat_entries(
                fold.slots, fold.slot_models, fold.slot_newaxes, fold.tail, tail_labels[used_trails:], expected
            )
            if entries is not None:
                return ast.Subscript(value=base, slice=index_slot(entries), ctx=node.ctx)
        if not fold.folded:
            return None
        folded_base = ast.Subscript(
            value=base, slice=index_slot([*fold.folded_inner, *fold.folded_tail]), ctx=ast.Load()
        )
        return ast.Subscript(value=folded_base, slice=index_slot(fold.folded_outer), ctx=node.ctx)

    def index_inner_arrays(self, fold: "ChainFold", inner_ranks: list[int], adv_rank: int) -> None:
        """Index each inner index array by the outer entries that land on its broadcast axes (several
        arrays only when they broadcast to the same extents); a failure keeps the two-step form."""
        arrays = [pos for pos, elt_rank in enumerate(inner_ranks) if elt_rank > 0]
        if len(arrays) > 1 and any(not is_full_slice(use) for use, model in fold.adv_uses.values()):
            fold.flat_ok = fold.flat_ok and self.same_broadcast_extents([fold.inner_elts[pos] for pos in arrays])
        for pos in arrays if fold.flat_ok else ():
            indexed = self.index_into_array(
                fold.inner_elts[pos], inner_ranks[pos], adv_rank, fold.adv_uses, fold.adv_newaxes
            )
            if indexed is None:
                fold.flat_ok = False
                break
            fold.slots[pos], fold.slot_models[pos] = indexed

    def inner_entries(self, inner: ast.Subscript, rank: int | None) -> list[ast.expr] | None:
        """The inner subscript's entries with a single Ellipsis expanded to full slices (and, under
        ``explicit_trailing_axes``, the base's trailing axes spelled out); None when an Ellipsis or
        newaxis leaves the axis count unknowable or the entries exceed the base rank."""
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
        return inner_elts

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
            extent = iter_extent_of(array, self.shape_table)
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


def flat_entries(
    slots: list[ast.expr],
    slot_models: list["IndexEntry"],
    slot_newaxes: list[list[int]],
    tail: list[tuple[ast.expr, "IndexEntry"]],
    unused_trails: list["AxisLabel"],
    expected: list["AxisLabel"],
) -> list[ast.expr] | None:
    """The single-subscript entries of a folded chain -- each slot preceded by the newaxes attached to
    it, then the tail -- or None when their result axes differ from the chain's ``expected`` ones."""
    entries: list[ast.expr] = []
    layout: list[IndexEntry] = []
    for pos, slot in enumerate(slots):
        entries.extend(ast.Constant(value=None) for i in slot_newaxes[pos])
        layout.extend(("newaxis", (("new", i),)) for i in slot_newaxes[pos])
        entries.append(slot)
        layout.append(slot_models[pos])
    entries.extend(entry for entry, model in tail)
    layout.extend(model for entry, model in tail)
    layout.extend(("slice", (label,)) for label in unused_trails)
    return entries if result_axes(layout) == expected else None


@dataclasses.dataclass(slots=True)
class ChainFold:
    """A chain ``inner[outer]`` being folded: the single-subscript slots (and their result-axis
    models and attached newaxes), the outer entries that land past the inner entries (``tail``) or on
    an inner index array's axes (``adv_uses``), and the two-step fallback in which only the bounded
    outer slices are folded into the inner subscript."""

    inner_elts: list[ast.expr]
    slots: list[ast.expr]
    slot_models: list[IndexEntry]
    slot_newaxes: list[list[int]]
    folded_inner: list[ast.expr]
    folded_outer: list[ast.expr]
    tail: list[tuple[ast.expr, IndexEntry]] = dataclasses.field(default_factory=list)
    adv_uses: dict[int, tuple[ast.expr, IndexEntry]] = dataclasses.field(default_factory=dict)
    adv_newaxes: dict[int, list[int]] = dataclasses.field(default_factory=dict)
    folded_tail: list[ast.expr] = dataclasses.field(default_factory=list)
    folded: bool = False
    flat_ok: bool = True

    def place(self, i: int, elt: ast.expr, elt_rank: int, label: AxisLabel, model: IndexEntry) -> None:
        """Land outer entry ``i`` on the inner result axis ``label``: composed onto an inner slice,
        recorded against an inner index array, or appended past the inner entries."""
        kind, index = label
        if kind == "adv":
            self.adv_uses[index] = (elt, model)
            return
        bounded = isinstance(elt, ast.Slice) and not is_full_slice(elt)
        if kind == "axis":
            composed = compose_onto_view(self.inner_elts[index], elt, elt_rank > 0)
            if composed is None:
                self.flat_ok = False
                return
            self.slots[index] = composed
            self.slot_models[index] = model
            if bounded:
                self.folded_inner[index] = composed
        else:
            self.tail.append((elt, model))
            if bounded:
                self.folded_tail.extend(ast.Slice() for gap in range(index - len(self.folded_tail)))
                self.folded_tail.append(elt)
        if bounded:
            self.folded_outer[i] = ast.Slice()
            self.folded = True
