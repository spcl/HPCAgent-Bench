"""Ellipsis expansion, implicit trailing slices, and view-alias folding."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import iter_extent_of
from hpcagent_bench.translators.numpyto_common.lowering.indexing import (
    compose_kept_axis,
    has_negative_step,
    is_fancy_dim,
    is_scalar_index,
    slice_dims,
)
from hpcagent_bench.translators.numpyto_common.lowering.shape_reads import is_newaxis
from hpcagent_bench.translators.numpyto_common.ordered import OrderedSet
from hpcagent_bench.translators.numpyto_common.subscripts import is_full_slice

__all__ = [
    "AliasFold",
    "EllipsisExpander",
    "PadImplicitTrailingSlices",
    "SliceViewFold",
    "SubarrayAliasFold",
    "aliases_with_rebound_base",
    "child_blocks_of",
    "composable_view_aliases",
    "flatten_view_chains",
    "fold_slice_view_aliases",
    "fold_subarray_aliases",
    "is_rank_preserving_slice_view",
    "names_stored_in",
    "names_written_in",
    "refuse_scalarising_a_contraction",
    "reject_view_writes_between_bind_and_use",
    "subarray_alias_candidates",
    "view_alias_candidates",
]


class EllipsisExpander(ast.NodeTransformer):
    """Replace ``...`` (Ellipsis) in a subscript with the explicit full slices
    it stands for, using the array's rank: ``a[..., 0]`` on a 3-D array ->
    ``a[:, :, 0]``. Chained subscripts are flattened to a Name base first by
    ChainedSubscriptFlattener; a base that is an EXPRESSION is sized through
    :func:`iter_extent_of`, which is all the rank costs."""

    def __init__(self, array_shapes: dict[str, list[str]]) -> None:
        self.array_shapes = array_shapes

    def base_rank(self, base: ast.expr) -> int | None:
        """Axis count of a subscript base, or None when nothing says what it is.

        cfd subscripts an arithmetic expression -- ``(__cb4 / density_i)[..., None]`` -- whose
        operands are sized even though the expression itself has no name to look up.
        """
        if isinstance(base, ast.Name):
            shape = self.array_shapes.get(base.id)
            return len(shape) if shape else None
        ext = iter_extent_of(base, self.array_shapes)
        return len(ext) if ext else None

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        rank = self.base_rank(node.value)
        if rank is None:
            return node
        sl = node.slice
        elts = list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]
        ell = [k for k, e in enumerate(elts) if isinstance(e, ast.Constant) and e.value is Ellipsis]
        if len(ell) != 1:
            return node
        # Source-axis-consuming entries (exclude the Ellipsis and any newaxis).
        consumed = sum(
            1 for e in elts if not (isinstance(e, ast.Constant) and (e.value is Ellipsis or e.value is None))
        )
        pad = max(rank - consumed, 0)
        pos = ell[0]
        new_elts = (
            elts[:pos] + [ast.Slice(lower=None, upper=None, step=None) for unused in range(pad)] + elts[pos + 1 :]
        )
        node.slice = new_elts[0] if len(new_elts) == 1 else ast.Tuple(elts=new_elts, ctx=ast.Load())
        return ast.copy_location(node, node)


class PadImplicitTrailingSlices(ast.NodeTransformer):
    """Make numpy's implicit trailing axes explicit on basic-indexed subscripts.

    ``A[i, j]`` on an n-D array (n > 2) means ``A[i, j, :, ...]`` -- the unlisted
    trailing axes are full slices. The slice / scalar lowering keys off the
    number of index positions, so a 3-D stencil written ``TN[:, 1:] = T[:, :-1]``
    (hotspot_3d) would otherwise iterate only 2 axes and drop the innermost,
    emitting invalid nested ``[][]`` on a flat buffer. Pad each such subscript
    with explicit full ``Slice()`` entries up to the array's rank.

    Only BASIC indexing is padded -- every existing index must be a Slice, a
    scalar int Constant, or a Name that is NOT itself an array (a loop iter /
    symbol). Advanced indexing (``x[src]`` with ``src`` an index array, the
    fancy-gather path) is left untouched so it is not mis-expanded."""

    def __init__(self, array_shapes: dict[str, list[str]]) -> None:
        self.array_shapes = array_shapes

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.value, ast.Name):
            return node
        shape = self.array_shapes.get(node.value.id)
        if not shape:
            return node
        rank = len(shape)
        sl = node.slice
        elts = list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]

        # A newaxis (``None``) inserts a RESULT axis but consumes NO source
        # axis, so it must not count against the array rank -- ``weights[None,
        # :, :, :]`` on a 4-D array still leaves one trailing source axis
        # implicit (conv2d's ``weights[np.newaxis, :, :, :]`` -> 5-D result
        # over a 4-D operand). Count only source-axis-consuming positions.
        n_index = sum(1 for e in elts if not is_newaxis(e))
        if n_index >= rank:
            return node
        # Basic-indexing gate: a Slice, a newaxis, or integer arithmetic over int Constants and
        # non-array Names -- ``hn[2 * l]`` selects one axis exactly as ``hn[l]`` does. Only an
        # index ARRAY makes a position advanced.
        for e in elts:
            if isinstance(e, ast.Slice) or is_newaxis(e):
                continue
            if is_scalar_index(e) and not any(
                isinstance(n, ast.Name) and n.id in self.array_shapes for n in ast.walk(e)
            ):
                continue
            return node  # advanced / unknown index -> skip
        pad = rank - n_index
        new_elts = elts + [ast.Slice(lower=None, upper=None, step=None) for unused in range(pad)]
        node.slice = ast.Tuple(elts=new_elts, ctx=ast.Load())
        return ast.copy_location(node, node)


def fold_subarray_aliases(tree: ast.AST, array_shapes: dict[str, list[str]]) -> None:
    """Fold a partial / trailing-slice sub-array alias into ONE flat multi-dim index.

    ``low = A[i, j]`` (or ``A[i, j, :]``) on a 3-D array is a sub-array; each use
    ``low[k]`` becomes ``A[i, j, k]`` -- a single subscript the emitter lowers to a
    flat offset -- instead of the chained ``A[i][j]`` a partial index otherwise emits
    on a flat C pointer (xsbench's ``low`` / ``high`` five-channel reads). Fires only
    when the alias is a basic-index sub-array of a known array, is assigned exactly
    once, its base indices are not rebound after it, and EVERY use is a further
    subscript (a bare whole-array use would need the row materialised, so it is left
    alone)."""
    aliases = subarray_alias_candidates(tree, array_shapes)
    if not aliases:
        return
    assigns: dict[str, int] = {}
    sub_value_ids: set = set()
    load_ids: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in aliases:
                    assigns[t.id] = assigns.get(t.id, 0) + 1
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            sub_value_ids.add(id(node.value))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in aliases:
            load_ids.setdefault(node.id, []).append(id(node))
    unsafe = aliases_with_rebound_base(tree, aliases)
    good = {
        name: aliases[name]
        for name in aliases
        if name not in unsafe and assigns.get(name, 0) == 1 and all(i in sub_value_ids for i in load_ids.get(name, []))
    }
    if not good:
        return
    SubarrayAliasFold(good).visit(tree)
    ast.fix_missing_locations(tree)


def subarray_alias_candidates(tree: ast.AST, array_shapes: dict[str, list[str]]) -> dict[str, tuple]:
    """``{alias: (array, lead indices)}`` for each ``alias = A[i, j(, :...)]``: plain scalar leading
    indices (trailing ``:`` axes dropped -- they are what ``alias[k]`` fills) that leave at least one
    trailing source axis (a genuine sub-array, not a full element index)."""
    aliases: dict[str, tuple] = {}
    for stmt in ast.walk(tree):
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
            continue
        val = stmt.value
        if not (isinstance(val, ast.Subscript) and isinstance(val.value, ast.Name)):
            continue
        shape = array_shapes.get(val.value.id)
        if not shape:
            continue
        elts = list(val.slice.elts) if isinstance(val.slice, ast.Tuple) else [val.slice]
        while elts and is_full_slice(elts[-1]):
            elts.pop()
        if any(isinstance(e, ast.Slice) or (isinstance(e, ast.Constant) and e.value is None) for e in elts):
            continue
        if len(elts) >= len(shape):
            continue
        aliases[stmt.targets[0].id] = (val.value.id, elts)
    return aliases


def aliases_with_rebound_base(tree: ast.AST, aliases: dict[str, tuple]) -> set[str]:
    """Aliases whose base index name is reassigned in a statement that can execute AFTER the alias
    (its block-tail, recursively) -- the folded ``A[i, j, k]`` would read the NEW i/j, not the value
    the alias captured. (Reassignment BEFORE the alias is fine.)"""
    unsafe: set = set()

    def scan(stmts) -> None:
        for i, s in enumerate(stmts):
            if (
                isinstance(s, ast.Assign)
                and len(s.targets) == 1
                and isinstance(s.targets[0], ast.Name)
                and s.targets[0].id in aliases
            ):
                unused, base = aliases[s.targets[0].id]
                base_names = {n.id for b in base for n in ast.walk(b) if isinstance(n, ast.Name)}
                if base_names & names_stored_in(stmts[i + 1 :]):
                    unsafe.add(s.targets[0].id)
            for cb in child_blocks_of(s):
                scan(cb)

    scan(tree.body if isinstance(tree, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)) else [tree])
    return unsafe


def names_stored_in(stmts: list[ast.stmt]) -> set[str]:
    """Names rebound anywhere in ``stmts``: Store-context Names and for-loop targets."""
    out: set = set()
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                out.add(n.id)
            elif isinstance(n, ast.For) and isinstance(n.target, ast.Name):
                out.add(n.target.id)
    return out


class AliasFold(ast.NodeTransformer):
    """Fold each alias in ``good`` into its uses (``visit_Subscript``, per subclass) and drop the
    now-unused alias assignment."""

    def __init__(self, good: dict[str, tuple]) -> None:
        self.good = good

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in self.good:
            return None
        self.generic_visit(node)
        return node


class SubarrayAliasFold(AliasFold):
    """``alias[k]`` -> ``A[i, j, k]``."""

    __slots__ = ()

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in self.good:
            aname, base = self.good[node.value.id]
            more = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
            new_idx = [copy.deepcopy(b) for b in base] + more
            sl = ast.Tuple(elts=new_idx, ctx=ast.Load()) if len(new_idx) > 1 else new_idx[0]
            return ast.copy_location(
                ast.Subscript(value=ast.Name(id=aname, ctx=ast.Load()), slice=sl, ctx=node.ctx), node
            )
        return node


def child_blocks_of(stmt: ast.stmt):
    if isinstance(stmt, (ast.For, ast.While, ast.If)):
        yield stmt.body
        yield stmt.orelse
    elif isinstance(stmt, ast.Try):
        yield stmt.body
        yield stmt.orelse
        yield stmt.finalbody
        for h in stmt.handlers:
            yield h.body


def names_written_in(stmts: list[ast.stmt]) -> OrderedSet:
    """Names written in ``stmts``: a rebind, a loop target, or the base of a
    subscript STORE (``x[...] = ...`` writes THROUGH ``x``, which a plain
    Name-rebind scan misses)."""
    out: OrderedSet = OrderedSet()
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                out.add(n.id)
            elif isinstance(n, ast.For) and isinstance(n.target, ast.Name):
                out.add(n.target.id)
            elif isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name):
                out.add(n.value.id)
    return out


def reject_view_writes_between_bind_and_use(
    tree: ast.AST, candidates: dict[str, tuple[str, list[ast.expr]]]
) -> OrderedSet:
    """Names among ``candidates`` whose base array, or a name their captured bounds
    read, is written in a statement able to run AFTER the alias's own ``Assign``
    (same block, recursively) -- folding such an alias would read the value AFTER
    that write at the use site, not the one the view captured at bind time."""
    free_names: dict[str, OrderedSet] = {}
    for name, (base_name, elts) in candidates.items():
        names = OrderedSet((base_name,))
        for e in elts:
            for n in ast.walk(e):
                if isinstance(n, ast.Name):
                    names.add(n.id)
        free_names[name] = names

    unsafe: OrderedSet = OrderedSet()

    def scan(stmts: list[ast.stmt]) -> None:
        for i, s in enumerate(stmts):
            if (
                isinstance(s, ast.Assign)
                and len(s.targets) == 1
                and isinstance(s.targets[0], ast.Name)
                and s.targets[0].id in candidates
            ):
                name = s.targets[0].id
                written = names_written_in(stmts[i + 1 :])
                if any(n in written for n in free_names[name]):
                    unsafe.add(name)
            for cb in child_blocks_of(s):
                scan(cb)

    scan(tree.body if isinstance(tree, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)) else [tree])
    return unsafe


def flatten_view_chains(good: dict[str, tuple[str, list[ast.expr]]]) -> dict[str, tuple[str, list[ast.expr]]]:
    """Compose a VIEW-OF-A-VIEW chain down to its root array in one hop.

    ``window = x_g[:, :, iy0:iy0+span_h:stride[0], ...]`` where ``x_g`` is itself a
    folding view (``x_g = padded[:, g*ipg:(g+1)*ipg]``) binds ``window`` against
    ``x_g``, not against ``padded``. Composing straight through to ``padded`` here
    -- instead of leaving ``window`` pointing at ``x_g`` -- matters because ``x_g``'s
    own ``Assign`` is about to be dropped by the SAME fold pass: an unresolved
    chain would leave ``window`` referencing a name that no longer exists. Each
    step reuses :func:`compose_kept_axis`, exactly the math a further SUBSCRIPT of
    ``x_g`` composes with; a view's own defining indices are algebraically no
    different from a later use's. ``good`` already proved every entry safe to fold
    against its OWN direct base, so composing two safe hops is still safe.
    """
    resolved: dict[str, tuple[str, list[ast.expr]]] = {}

    def resolve_(name: str, seen: OrderedSet) -> tuple[str, list[ast.expr]]:
        if name in resolved:
            return resolved[name]
        base_name, elts = good[name]
        if base_name in good and base_name not in seen:
            deeper = OrderedSet(seen)
            deeper.add(name)
            root_base, root_elts = resolve_(base_name, deeper)
            kept_positions = [i for i, e in enumerate(root_elts) if isinstance(e, ast.Slice)]
            padded = elts + [
                ast.Slice(lower=None, upper=None, step=None) for unused in range(len(kept_positions) - len(elts))
            ]
            composed = [copy.deepcopy(e) for e in root_elts]
            for pos, u in zip(kept_positions, padded):
                composed[pos] = compose_kept_axis(root_elts[pos], u)
            result = (root_base, composed)
        else:
            result = (base_name, elts)
        resolved[name] = result
        return result

    for name in good:
        resolve_(name, OrderedSet())
    return resolved


def is_rank_preserving_slice_view(node: ast.Subscript, array_shapes: dict[str, list[str]], target_rank: int) -> bool:
    """Whether ``node`` is a basic slice of a known array that KEEPS every axis.

    ``canvas[:, :, p:p + oh, p:p + ow]`` bound to a name and then used bare has no fold to resolve
    it -- :func:`fold_slice_view_aliases` only rewrites subscripted uses -- so the crop has to be
    materialised into the target instead. Restricted to the rank-preserving case: a dropped axis
    (``x = a[:, i]``) would map the copy nest's iterators onto the wrong right-hand-side positions.
    """
    if not isinstance(node.value, ast.Name):
        return False
    shape = array_shapes.get(node.value.id)
    if not shape or len(shape) != target_rank:
        return False
    elts = slice_dims(node)
    if len(elts) > len(shape) or has_negative_step(elts):
        return False
    return all(isinstance(e, ast.Slice) and not is_fancy_dim(e, array_shapes) for e in elts)


def fold_slice_view_aliases(tree: ast.AST, array_shapes: dict[str, list[str]]) -> OrderedSet:
    """Fold a name bound to a partial/strided VIEW of an array into every subscripted use.

    ``x_g = padded[:, g*in_per_group:(g+1)*in_per_group]`` on a 4-D ``padded`` binds a
    view whose axis 1 is offset by ``g*in_per_group``, other axes passing through
    untouched; a further subscript ``x_g[i0, i1, i2, i3]`` composes to
    ``padded[i0, g*in_per_group + i1, i2, i3]`` (grouped conv's per-group input
    slab -- conv2d_batch_norm_scaling and the rest of the "expression Slice"
    machine_learning refusals). Unlike :func:`fold_subarray_aliases` (a scalar
    index PREFIX plus dropped trailing full slices), this handles a Slice with real
    bounds/step at ANY axis position, composing offsets/strides against both a
    further scalar index (``start + step*j``) and a further slice
    (``start+step*a : start+step*b : step*c``, :func:`compose_kept_axis`). numpy
    squeeze semantics pick which base axis a kept view axis is: an INTEGER view
    index drops the axis (it never appears at a use site again), a Slice view
    index -- even a length-1 one -- keeps it as one of the view's own axes, in
    the order it appears.

    A store THROUGH the alias DECLINES the fold outright. ``view[0, 0] = 5`` is what makes the name
    a genuine alias rather than a private copy the emitters may materialise, and the fold's job is
    to make the name disappear; redirecting the store onto the base at a composed offset is a
    rewrite of the kernel's aliasing, not of its indexing, and it is not this pass's to make. The
    refusal costs nothing measured: every transposed-conv kernel in the corpus (the accumulation
    canvas this once fired on) emits byte-identical C with the store folded or declined.

    Fires only when it is provably sound: the alias is assigned exactly once, every
    use of it is a further BASIC-indexed subscript READ (never passed
    around bare, never gathered through an index array, never written through), and neither the
    source array nor a name the view's bounds read is written before every use
    (:func:`reject_view_writes_between_bind_and_use`). Any alias failing these
    checks is left alone -- the existing "expression Slice" refusal stands rather
    than risk a silently wrong shape or offset.
    """
    aliases = view_alias_candidates(tree, array_shapes)
    if not aliases:
        return OrderedSet()
    candidates = composable_view_aliases(tree, aliases, array_shapes)
    if not candidates:
        return OrderedSet()
    unsafe = reject_view_writes_between_bind_and_use(tree, candidates)
    good = {name: v for name, v in candidates.items() if name not in unsafe}
    if not good:
        return OrderedSet()
    good = flatten_view_chains(good)
    SliceViewFold(good).visit(tree)
    ast.fix_missing_locations(tree)
    live = OrderedSet(n.id for n in ast.walk(tree) if isinstance(n, ast.Name))
    return OrderedSet(name for name in good if name not in live)


def refuse_scalarising_a_contraction(value: ast.expr) -> None:
    """Raise if ``value`` still holds an array-level ``@``.

    Scalarising a contraction changes what it means: ``C[:] = A @ B`` becomes ``C[i, j] = A[i, j] *
    B[i, j]``, which drops the sum over k entirely and reads both operands at the OUTPUT's extents.
    It compiles, it runs, and it returns wrong numbers -- netvlad's
    ``np.swapaxes(assignment, 1, 2) @ x`` did exactly that.

    The emitter has a guard for a surviving ``@``, but it cannot catch this one: by the time it runs
    the rewrite has already replaced both operands with scalar subscripts, so the guard's
    "are the operands scalar" test passes and ``*`` is emitted. The only place the difference is
    still visible is here, BEFORE the rewrite.

    Reaching this means the matmul hoister declined -- normally a shape it could not resolve. That is
    a gap to fix, and a refusal names it; the silent product does not.
    """
    for sub in ast.walk(value):
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.MatMult):
            raise NotImplementedError(
                f"matmul '{ast.unparse(sub)}' was not lowered before slice fusion; "
                f"scalarising it would drop the contraction and silently "
                f"compute an elementwise product"
            )


class SliceViewFold(AliasFold):
    """``view[use]`` -> ``base[composed]``: each kept view axis composed with the use's entry
    (:func:`compose_kept_axis`), dropped view axes passing through."""

    __slots__ = ()

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if not (isinstance(node.value, ast.Name) and node.value.id in self.good):
            return node
        base_name, view_elts = self.good[node.value.id]
        kept_positions = [i for i, e in enumerate(view_elts) if isinstance(e, ast.Slice)]
        use_elts = slice_dims(node)
        use_elts = use_elts + [
            ast.Slice(lower=None, upper=None, step=None) for unused in range(len(kept_positions) - len(use_elts))
        ]
        composed = [copy.deepcopy(e) for e in view_elts]
        for pos, u in zip(kept_positions, use_elts):
            composed[pos] = compose_kept_axis(view_elts[pos], u)
        sl = ast.Tuple(elts=composed, ctx=ast.Load()) if len(composed) > 1 else composed[0]
        return ast.copy_location(
            ast.Subscript(value=ast.Name(id=base_name, ctx=ast.Load()), slice=sl, ctx=node.ctx), node
        )


def view_alias_candidates(tree: ast.AST, array_shapes: dict[str, list[str]]) -> dict[str, tuple[str, list[ast.expr]]]:
    """``{alias: (base, view entries padded to the base rank)}`` for each ``alias = base[...]`` that
    is a basic-indexed VIEW: at least one Slice (a fully scalar index is an element read), no index
    array, no negative step (the offset algebra assumes a positive stride)."""
    aliases: dict[str, tuple[str, list[ast.expr]]] = {}
    for stmt in ast.walk(tree):
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
            continue
        val = stmt.value
        if not (isinstance(val, ast.Subscript) and isinstance(val.value, ast.Name)):
            continue
        base_name = val.value.id
        shape = array_shapes.get(base_name)
        if not shape:
            continue
        elts = slice_dims(val)
        if len(elts) > len(shape) or any(is_fancy_dim(e, array_shapes) for e in elts):
            continue
        elts = elts + [ast.Slice(lower=None, upper=None, step=None) for unused in range(len(shape) - len(elts))]
        if not any(isinstance(e, ast.Slice) for e in elts) or has_negative_step(elts):
            continue
        aliases[stmt.targets[0].id] = (base_name, elts)
    return aliases


def composable_view_aliases(
    tree: ast.AST, aliases: dict[str, tuple[str, list[ast.expr]]], array_shapes: dict[str, list[str]]
) -> dict[str, tuple[str, list[ast.expr]]]:
    """The aliases assigned exactly once whose every use is a BASIC-indexed subscript READ within the
    view's kept rank (never bare, never gathered through an index array, never written through)."""
    assigns: dict[str, int] = {}
    uses_composable: dict[str, bool] = {}
    sub_value_ids: OrderedSet = OrderedSet()
    load_ids: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in aliases:
                    assigns[t.id] = assigns.get(t.id, 0) + 1
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            name = node.value.id
            sub_value_ids.add(id(node.value))
            kept = sum(1 for e in aliases[name][1] if isinstance(e, ast.Slice))
            use_elts = slice_dims(node)
            ok = (
                len(use_elts) <= kept
                and not any(is_fancy_dim(e, array_shapes) for e in use_elts)
                and not has_negative_step(use_elts)
                and not isinstance(node.ctx, ast.Store)
            )
            uses_composable[name] = uses_composable.get(name, True) and ok
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in aliases:
            load_ids.setdefault(node.id, []).append(id(node))
    return {
        name: aliases[name]
        for name in aliases
        if assigns.get(name, 0) == 1
        and uses_composable.get(name, True)
        and all(i in sub_value_ids for i in load_ids.get(name, []))
    }
