"""Axis handling: axis-reshaping calls to indexing, structural constants, runtime-axis dispatch."""

import ast
import copy
from collections.abc import Callable, Mapping, Sequence

from hpcagent_bench.translators.numpyto_common.lib_nodes import slice_axes
from hpcagent_bench.translators.numpyto_common.ordered import OrderedSet
from hpcagent_bench.translators.numpyto_common.subscripts import index_slot, is_full_slice, is_newaxis
from hpcagent_bench.translators.numpyto_common.numpy_desugar import REDUCE_FNS, expr_rank
from hpcagent_bench.translators.numpyto_common.frontend.manifest import preset_constant_symbols
from hpcagent_bench.translators.numpyto_common.frontend.shape_arith import const_int


class AxisReshapeToIndexing(ast.NodeTransformer):
    """``np.expand_dims`` / ``np.swapaxes`` -> indexing forms the pipeline already lowers.

    Both need the operand's rank: ``expand_dims`` to place the newaxis (a negative axis counts from
    the RESULT rank), ``swapaxes`` to spell the full permutation. Unknown rank leaves the call
    alone, which surfaces as an unsupported-call error rather than a wrong axis.
    """

    def __init__(self, ranks: dict[str, int], scalars: frozenset[str] = frozenset()) -> None:
        self.ranks = ranks
        #: Declared scalar parameters. ``expr_rank`` only tracks arrays, so without these a
        #: ``np.array(constant_value)`` on a scalar knob reads as "rank unknown".
        self.scalars = scalars

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        name = np_attr_name(node) if isinstance(node.func, ast.Attribute) else None
        if name in AXIS_STRUCTURAL_FNS or name == "norm":
            self.drop_noop_keepdims(node)
        if name == "array" and len(node.args) == 1 and not isinstance(node.args[0], (ast.List, ast.Tuple)):
            # ``np.array(0.0)`` is a 0-d array: the scalar itself. Only when the operand is already
            # a scalar -- ``np.array(some_array)`` is a COPY, and dropping it would alias.
            return node.args[0] if self.is_scalar(node.args[0]) else node
        if name == "norm" and self.is_linalg(node.func) and node.args:
            return self.axis_norm(node)
        if name not in ("expand_dims", "swapaxes", "squeeze", "moveaxis") or not node.args:
            return node
        rank = expr_rank(node.args[0], self.ranks)
        axes = self.literal_axes(node)
        if rank is None or axes is None:
            return node
        if name in ("moveaxis", "swapaxes") and isinstance(node.args[0], ast.Call):
            # These two build a PERMUTATION out of the rank alone, and a rank read off a nested
            # call is a guess. ls3df_scf's ``np.moveaxis(np.tensordot(row, X, axes=([1], [1])),
            # 0, 1)`` is rank 4; a mis-read rank builds a perm of the wrong LENGTH, which
            # expand_transpose refuses ("perm size != ndim") -- and that refusal is swallowed, so
            # the kernel fails much later as "call to np.transpose not supported". Lowering knows
            # the real shape (the call hoister spills the operand to a sized temp first), so leave
            # the call for it. A Name/Subscript operand keeps the rewrite: that is the helper-
            # parameter case this pass exists for.
            return node
        if rank == 0 and name != "expand_dims":
            # Every rewrite below normalises its axis with ``% rank``, which a 0-d operand has no
            # meaning for -- and an operand whose rank the table could not resolve arrives here as
            # 0, not None. Declining leaves the call for the unsupported-call error rather than
            # dividing by zero or inventing a permutation. ``expand_dims`` is exempt: it counts
            # against ``rank + 1``, and wrapping a scalar into a 1-element axis is well defined.
            return node
        if name == "expand_dims":
            axis = axes[0] % (rank + 1)
            return self.index_(
                node.args[0], [ast.Constant(value=None) if d == axis else ast.Slice() for d in range(rank + 1)], node
            )
        if name == "squeeze":
            axis = axes[0] % rank
            return self.index_(
                node.args[0], [ast.Constant(value=0) if d == axis else ast.Slice() for d in range(rank)], node
            )
        if name == "moveaxis":
            # A pure index rewrite like swapaxes, but the axis MOVES rather than trades places:
            # pull it out of the identity order and re-insert it at the destination.
            source, destination = (a % rank for a in axes[:2])
            perm = [d for d in range(rank) if d != source]
            perm.insert(destination, source)
            return self.rewrite_(f"np.transpose({ast.unparse(node.args[0])}, ({', '.join(map(str, perm))},))", node)
        i, j = (a % rank for a in axes[:2])
        perm = list(range(rank))
        perm[i], perm[j] = perm[j], perm[i]
        # An identity ``perm`` is emitted as a transpose like any other, never dropped: it is built
        # from ``rank``, so it means either a genuine no-op or a rank this pass read wrong, and the
        # two are indistinguishable from here. Dropping it on that reading turned
        # conv_transpose3d_leaky_relu_multiply_leaky_relu_max's rank-5 ``moveaxis(tap, -1, 1)`` into
        # an identity copy while its consumer went on indexing the permuted layout.
        return self.rewrite_(f"np.transpose({ast.unparse(node.args[0])}, ({', '.join(map(str, perm))},))", node)

    @staticmethod
    def drop_noop_keepdims(node: ast.Call) -> None:
        """Delete a literal ``keepdims=False`` from a reduction call -- it is numpy's OWN default, so
        every reader here already reads its absence as False (:func:`read_axis_keepdims`) and the
        result rank is unchanged either way.

        Not cosmetic: dace's reductions declare no ``keepdims`` parameter at all
        (``_sum(pv, sdfg, state, a, axis=None)``), so forwarding the no-op refused the whole program
        with ``_sum() got an unexpected keyword argument 'keepdims'``.

        A TRUE one is deliberately left in place. It sets the result RANK, and restoring the reduced
        axis needs the operand's shape TOKENS, which this pass does not carry -- it knows ranks only.
        An unrestored axis broadcasts against the wrong one, which is a wrong answer rather than a
        refusal, and the native lowering consumes ``keepdims`` directly.
        """
        node.keywords = [
            k
            for k in node.keywords
            if not (
                k.arg == "keepdims"
                and isinstance(k.value, ast.Constant)
                and isinstance(k.value.value, (bool, int))
                and not k.value.value
            )
        ]

    def is_scalar(self, node: ast.expr) -> bool:
        """Rank 0 for certain: a numeric literal or a declared scalar parameter."""
        if isinstance(node, ast.Name):
            return node.id in self.scalars
        return expr_rank(node, self.ranks) == 0

    def is_linalg(self, func: ast.AST) -> bool:
        """``np.linalg.norm``'s callee shape, so a user helper called ``norm`` is not caught."""
        return isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute) and func.value.attr == "linalg"

    def axis_norm(self, node: ast.Call) -> ast.AST:
        """``np.linalg.norm(v, axis=k)`` -> ``np.sqrt(np.sum(abs(v) ** 2, axis=k))``.

        Only the default 2-norm; an explicit ``ord`` is a different reduction and is left alone.
        ``abs(v) ** 2`` rather than ``v * v`` because the two disagree for a COMPLEX operand -- and
        nothing downstream would have caught that, so the choice is made here where it is free.
        """
        kw = {k.arg: k.value for k in node.keywords if k.arg is not None}
        if "ord" in kw or len(node.args) > 1 or "axis" not in kw:
            return node
        operand = ast.unparse(node.args[0])
        # A TRUE keepdims rides along: dropping it silently changed the result RANK, and l2_norm's
        # ``x / np.linalg.norm(x, axis=1, keepdims=True)`` then broadcast against the wrong axis.
        # A false one is already gone -- :meth:`drop_noop_keepdims` takes it before this runs.
        keep = f", keepdims={ast.unparse(kw['keepdims'])}" if "keepdims" in kw else ""
        return self.rewrite_(f"np.sqrt(np.sum(np.abs({operand}) ** 2, axis={ast.unparse(kw['axis'])}{keep}))", node)

    def literal_axes(self, node: ast.Call) -> list[int] | None:
        kw = {k.arg: k.value for k in node.keywords if k.arg is not None}
        given = list(node.args[1:]) + ([kw["axis"]] if "axis" in kw else [])
        out: list[int] = []
        for a in given:
            if isinstance(a, ast.Constant) and isinstance(a.value, int) and not isinstance(a.value, bool):
                out.append(a.value)
            elif (
                isinstance(a, ast.UnaryOp)
                and isinstance(a.op, ast.USub)
                and isinstance(a.operand, ast.Constant)
                and isinstance(a.operand.value, int)
            ):
                out.append(-a.operand.value)
            else:
                return None
        return out or None

    def index_(self, operand: ast.expr, entries: list[ast.expr], node: ast.Call) -> ast.AST:
        """``operand[entries]``, merged into the operand's OWN index list when that is a basic one.

        Nested ``expand_dims`` / ``squeeze`` -- every instance-norm port reduces over
        ``np.expand_dims(np.expand_dims(z, 1), 1)`` -- otherwise builds the CHAIN
        ``z[:, None, :][:, None, :, :]``, and no shape resolver reads the extent of a subscript
        whose base is itself sliced. The reduction over it is then never sized, never hoisted to a
        temp, and reaches the emitter as an unlowered ``np.mean``.
        """
        merged = self.merge_index(operand, entries)
        subscript = ast.Subscript(
            value=operand if merged is None else operand.value,
            slice=index_slot(entries if merged is None else merged),
            ctx=ast.Load(),
        )
        return ast.fix_missing_locations(ast.copy_location(subscript, node))

    def merge_index(self, operand: ast.expr, entries: list[ast.expr]) -> list[ast.expr] | None:
        """``entries`` applied to ``operand``'s own index list, or ``None`` when they cannot merge.

        numpy basic indexing associates: an outer entry lands on the axis the inner subscript left
        (a scalar entry consumes its source axis and leaves none), and an outer newaxis inserts a
        fresh size-1 axis ahead of the axis it precedes. Only full slices, newaxes and int entries
        qualify -- a PARTIAL slice carries an offset an outer scalar index would drop
        (``a[2:5][0]`` is ``a[2]``, not ``a[0]``), and an Ellipsis or an index ARRAY does not map
        one entry to one axis. ``entries`` is this pass's own list, so it holds ``:`` / ``None`` /
        ``0`` and nothing else.
        """
        if not isinstance(operand, ast.Subscript):
            return None
        inner = slice_axes(operand)
        if not all(is_full_slice(e) or is_newaxis(e) or const_int(e) is not None for e in inner):
            return None
        if sum(1 for e in inner if const_int(e) is None) != sum(1 for e in entries if not is_newaxis(e)):
            return None  # the inner leaves source axes unspelled, so the positions do not line up
        merged: list[ast.expr] = []
        pos = 0
        for axis in inner:
            if const_int(axis) is not None:
                merged.append(axis)
                continue
            while is_newaxis(entries[pos]):
                merged.append(entries[pos])
                pos += 1
            outer = entries[pos]
            pos += 1
            if is_full_slice(outer):
                merged.append(axis)
            elif not is_newaxis(axis):
                merged.append(outer)  # ``x[None][0]`` drops the inserted axis instead
        merged.extend(entries[pos:])
        return merged

    def rewrite_(self, source: str, node: ast.Call) -> ast.AST:
        return ast.copy_location(ast.parse(source, mode="eval").body, node)


#: Calls whose ``axis`` selects WHICH loop the lowering writes -- a structural choice, so an axis
#: that is not a compile-time integer has no emittable form.
#:
#: Every op here had a way to swallow an unreadable axis rather than refuse it: a reduction read it
#: as "no axis" and reduced over ALL of them (``read_axis_keepdims`` returns ``None`` for both),
#: and an index op whose axis never resolved fell through to the emitter's scalar no-op path, which
#: dropped it outright -- ``np.flip(x, axis=dim)`` emitted a plain copy.
AXIS_STRUCTURAL_FNS = frozenset(REDUCE_FNS) | {
    "cumsum",
    "cumprod",
    "median",
    "count_nonzero",
    "squeeze",
    "expand_dims",
    "flip",
    "roll",
    "take",
    "repeat",
    "diff",
    "sort",
    "argsort",
    "swapaxes",
    "moveaxis",
    "concatenate",
    "stack",
    "split",
    "unique",
    "fft",
    "ifft",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
}


#: Which POSITIONAL slot each call puts ``axis`` in. Most reductions take it second; the ops that
#: take a count or an index first put it third. Reading the wrong slot let ``np.repeat(A, 2, ax)``
#: past the guard entirely (slot 1 held the literal repeat count) and made ``np.take(A, idx, ax)``
#: refuse with the INDEX ARRAY named as the offending axis.
#: ``np.fft.*`` takes a transform LENGTH first, so its axis is third too.
AXIS_POSITION: dict[str, int] = {
    "repeat": 2,
    "roll": 2,
    "take": 2,
    "diff": 2,
    "split": 2,
    "swapaxes": 1,
    "unique": 4,
    "fft": 2,
    "ifft": 2,
    "rfft": 2,
    "irfft": 2,
    "fftn": 2,
    "ifftn": 2,
}


def structural_constants(
    parameters: Mapping[str, object],
    scalars: Mapping[str, object],
    shapes_raw: Mapping[str, str],
    runtime_args: Sequence[str] = (),
) -> dict[str, int]:
    """Preset-constant integers that CANNOT be a size, so folding them into the body is safe.

    "Cannot be a size" is decided structurally: the name is absent from every ``init.shapes``
    expression. That matters because one emitted artifact serves every preset AND the harness may
    scale the declared sizes at run time -- a symbol that reaches an extent must stay a runtime
    argument. A symbol that reaches only an axis, a repeat count, or a slice bound has one value for
    the life of the artifact, and the loop nest cannot be built until it is a literal.

    A name in ``runtime_args`` is excluded whatever its manifest default says: it reaches the ABI, so
    the harness passes a value that need not be the default, and baking the default in is a
    miscompile. gmres declares ``max_iter`` in ``init.scalars`` AND takes it as an argument -- folding
    it turned the derived symbol ``m = min(max_iter, N)`` into ``min(100, N)``, pinning the iteration
    count to the manifest's value for every run. When such a name is an AXIS,
    :func:`specialize_runtime_axis` emits the nest for each axis and picks at run time; when it is a
    slice STEP it is carried symbolically (``lo + pos * step``), so neither slot needs the fold.
    """
    extent_names: set[str] = set()
    for shape in shapes_raw.values():
        try:
            parsed = ast.parse(str(shape).strip(), mode="eval")
        except SyntaxError:
            continue
        extent_names.update(n.id for n in ast.walk(parsed) if isinstance(n, ast.Name))
    runtime = frozenset(runtime_args)
    return {
        name: value
        for name, value in preset_constant_symbols(parameters, scalars).items()
        if name not in extent_names and name not in runtime
    }


def rebound_names(fn: ast.FunctionDef) -> frozenset[str]:
    """Every name ``fn`` BINDS anywhere in its body, targets unpacked.

    A manifest value is only the artifact's value while the name still HOLDS it, so both folds above
    consult this before substituting. EVERY binding form counts, not just ``=``: this is the sole
    barrier against folding a stale value into a slot that still compiles, so a form it misses is a
    wrong axis or a wrong stride with no error attached. ``:=``, ``with ... as``, ``except ... as``
    and a comprehension target bind exactly as an assignment does -- the comprehension's is its own
    scope, but treating it as a rebinding only costs a fold that was never necessary.
    """
    names: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            targets: list[ast.expr | None] = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor, ast.NamedExpr, ast.comprehension)):
            targets = [node.target]
        elif isinstance(node, ast.withitem):
            targets = [node.optional_vars]
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                names.add(node.name)
            continue
        else:
            continue
        names.update(
            leaf.id for tgt in targets if tgt is not None for leaf in ast.walk(tgt) if isinstance(leaf, ast.Name)
        )
    return frozenset(names)


class FoldConstantSymbols(ast.NodeTransformer):
    """Replace a load of a structural constant with its literal value.

    A name the body REBINDS is left alone. The conv ports normalise a scalar knob to a pair
    (``if isinstance(stride, int): stride = (stride, stride)``) and then read ``stride[0]``;
    folding the loads turned that read into ``1[0]``, since the rebinding is what makes it a tuple.
    """

    def __init__(self, const_syms: dict[str, int]) -> None:
        self.const_syms = const_syms

    def apply(self, fn: ast.FunctionDef) -> None:
        self.const_syms = {k: v for k, v in self.const_syms.items() if k not in rebound_names(fn)}
        self.visit(fn)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if not isinstance(node.ctx, ast.Load) or node.id not in self.const_syms:
            return node
        return ast.copy_location(ast.Constant(value=self.const_syms[node.id]), node)


def axis_argument(call: ast.Call) -> ast.expr | None:
    """The node sitting in ``call``'s axis slot, or ``None`` when it names no axis (or is not a
    call whose axis picks the loop nest)."""
    name = np_attr_name(call)
    if name not in AXIS_STRUCTURAL_FNS:
        return None
    kw = {k.arg: k.value for k in call.keywords if k.arg is not None}
    slot = AXIS_POSITION.get(name, 1)
    return kw.get("axis") or kw.get("axes") or (call.args[slot] if len(call.args) > slot else None)


def reject_symbolic_axis(fn: ast.FunctionDef) -> None:
    """Refuse a reduction / scan whose axis is present but not a literal.

    ``read_axis_keepdims`` reports an unreadable axis as ``None``, the SAME value it reports for
    ``np.sum(x)``, so ``np.sum(x, axis=dim)`` would otherwise lower as a FULL reduction.

    Reached only for an axis :func:`specialize_runtime_axis` could not dispatch on -- a runtime
    axis with a known operand rank is emitted as one specialised nest per axis, chosen at run time.
    """
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call) or np_attr_name(node) not in AXIS_STRUCTURAL_FNS:
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg is not None}
        axis = axis_argument(node)
        if axis is not None and not is_literal_axis(axis):
            raise NotImplementedError(
                f"{ast.unparse(node)}: axis must be a compile-time integer "
                f"(got {ast.unparse(axis)!r}); the emitted loop nest is chosen by it"
            )
        # keepdims decides the result RANK, and a non-literal one was read as False -- which then
        # broadcast the reduction against the wrong axis.
        keep = kw.get("keepdims")
        if keep is not None and not (isinstance(keep, ast.Constant) and isinstance(keep.value, (bool, int))):
            raise NotImplementedError(
                f"{ast.unparse(node)}: keepdims must be a compile-time constant "
                f"(got {ast.unparse(keep)!r}); it decides the result rank"
            )


def reject_unsupported_slices(fn: ast.FunctionDef) -> None:
    """Refuse the two slice forms the index lowering silently ignores.

    * an UNBOUNDED non-literal step: ``x[::s]`` emitted a contiguous copy, stride gone. A BOUNDED
      one (``x[lo:hi:s]``, the conv/pool tap) is lowered instead -- see below.
    * a NEGATIVE lower bound: it is added to the iterator verbatim rather than resolved against the
      axis length, so ``x[-3:]`` emitted ``x[i - 3]`` and read before the buffer. (A negative UPPER
      bound is fine: it only shortens the trip count, which comes from the target's extent.)

    Why the bound decides it. A symbolic step's SIGN is unknown at emit time, and the two signs
    index in opposite directions, so lowering has to pick one. With an upper bound present the
    choice is forced rather than assumed: under a negative step numpy flips the bound defaults, so
    ``lo:hi:k`` with ``lo < hi`` yields an EMPTY axis and the assignment consuming it already fails
    in numpy. Only the positive stride has a run to preserve, and that is what is emitted. Without
    an upper bound both signs produce a full-length axis and a forward index would silently be the
    wrong one, so that form keeps the refusal.
    """
    for node in ast.walk(fn):
        if not isinstance(node, ast.Slice):
            continue
        if node.step is not None and not is_literal_axis(node.step) and node.upper is None:
            raise NotImplementedError(
                f"slice step {ast.unparse(node.step)!r} needs an upper bound or a "
                f"compile-time integer; an unbounded symbolic step has no known "
                f"direction and would be emitted as a forward stride"
            )
        if isinstance(node.lower, ast.UnaryOp) and isinstance(node.lower.op, ast.USub):
            raise NotImplementedError(
                f"negative slice start {ast.unparse(node.lower)!r} is not resolved against "
                f"the axis length; write it as an explicit extent instead"
            )


def is_literal_axis(node: ast.expr) -> bool:
    """A literal axis: an int, a negated int, ``None``, or a tuple/list of those."""
    if isinstance(node, ast.Constant):
        return node.value is None or (isinstance(node.value, int) and not isinstance(node.value, bool))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return is_literal_axis(node.operand)
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(is_literal_axis(e) for e in node.elts)
    return False


def np_attr_name(node: ast.Call) -> str | None:
    """``np.sum(...)`` / ``x.sum(...)`` -> ``"sum"``, else ``None``."""
    return node.func.attr if isinstance(node.func, ast.Attribute) else None


#: Ceiling on the rank a runtime axis may dispatch over. The body is duplicated once per axis, and
#: every branch's temporaries are allocated whether or not that branch runs, so the cost is linear
#: in the rank. Past this the refusal -- which names the axis -- is the better answer.
MAX_DISPATCH_RANK = 4


def sequence_length(value: ast.expr, ranks: dict[str, int]) -> int | None:
    """Element count of a compile-time sequence, or ``None`` when it is not one.

    Covers the literal and the ``[<elt>] * <array>.ndim`` repeat the ports build a per-axis index
    list with; that count is what makes ``slices[dim]`` an AXIS index rather than a data index.
    """
    if isinstance(value, (ast.List, ast.Tuple)):
        return len(value.elts)
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult):
        for seq, count in ((value.left, value.right), (value.right, value.left)):
            if not isinstance(seq, (ast.List, ast.Tuple)):
                continue
            if isinstance(count, ast.Constant) and isinstance(count.value, int):
                return len(seq.elts) * count.value
            if (
                isinstance(count, ast.Attribute)
                and count.attr == "ndim"
                and isinstance(count.value, ast.Name)
                and count.value.id in ranks
            ):
                return len(seq.elts) * ranks[count.value.id]
    return None


#: Calls whose ``axis`` addresses the RESULT's axes -- one more than the operand's, since the call
#: inserts one. Reading their axis against the operand's rank would size the dispatch one short.
AXIS_INSERTS = frozenset({"expand_dims", "stack"})


def axis_index_spaces(fn: ast.FunctionDef, ranks: dict[str, int]) -> dict[int, int]:
    """``id(index node) -> how many AXES that index selects among``, for the two sequences an axis
    may legitimately index: ``x.shape`` and a rank-length per-axis list.

    A negative axis and its normalised form pick the same element only in a sequence with one entry
    per axis. Indexing anything else with ``dim`` is a DATA read, where ``-1`` means "last element"
    and substituting ``rank - 1`` would read a different one -- so this is what decides both the
    axis count and whether substituting into a use is legitimate at all.
    """
    out: dict[int, int] = {}
    bound: dict[str, list[ast.expr]] = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            bound.setdefault(node.targets[0].id, []).append(node.value)
    for node in ast.walk(fn):
        if not isinstance(node, ast.Subscript):
            continue
        base = node.value
        if (
            isinstance(base, ast.Attribute)
            and base.attr == "shape"
            and isinstance(base.value, ast.Name)
            and base.value.id in ranks
        ):
            out[id(node.slice)] = ranks[base.value.id]
        elif isinstance(base, ast.Name) and bound.get(base.id):
            lengths = {sequence_length(v, ranks) for v in bound[base.id]}
            length = lengths.pop() if len(lengths) == 1 else None
            if length is not None:
                out[id(node.slice)] = length
    return out


#: What a dispatch is: the axis ARGUMENT's name, and the RANK of the operand it indexes -- which is
#: also the branch count and what a negative axis resolves against.
AxisChoice = tuple[str, int]


def runtime_axis_dispatch(fn: ast.FunctionDef, scalars: frozenset[str], ranks: dict[str, int]) -> AxisChoice | None:
    """``(name, rank)`` of the one runtime axis to specialise over, or ``None``.

    What the manifest happens to set the axis to is deliberately NOT a condition. An argument that
    crosses the ABI is one the caller chooses, so a preset-constant default is a default and not a
    compile-time fact; a kernel for which it IS a fact says so by keeping the value out of
    ``input_args`` entirely (a keyword-only default the reference declares), and then there is no
    runtime axis here to dispatch on.

    Every remaining condition is a precondition for substituting a literal axis into a clone of the
    whole body, not a convenience:

    * ONE name only -- the branch count is ``rank`` per dispatched name, so two would multiply.
    * Every use of it is an AXIS: an axis slot, or an index into ``x.shape`` / a per-axis list.
      Only there do ``-1`` and ``rank - 1`` denote the same thing, which is what lets one branch
      serve both spellings.
    * Every use that reveals an axis COUNT reveals the same one. That count is the branch count and
      what a negative axis resolves against, so a body mixing two rank spaces has no single
      dispatch and keeps the refusal.
    * The kernel writes through its parameters. A RETURNED output is promoted from the body's
      trailing statement (:func:`synthesize_return_temps`), which a dispatch buries inside a
      branch -- the kernel would then emit with no output at all.
    """
    if any(isinstance(node, ast.Return) and node.value is not None for node in ast.walk(fn)):
        return None
    axis_names: OrderedSet[str] = OrderedSet()
    axis_spaces: dict[int, int | None] = {}
    for node in ast.walk(fn):
        axis = axis_argument(node) if isinstance(node, ast.Call) else None
        if axis is None:
            continue
        operand = expr_rank(node.args[0], ranks) if node.args else None
        insert = 1 if np_attr_name(node) in AXIS_INSERTS else 0
        axis_spaces[id(axis)] = None if operand is None else operand + insert
        if isinstance(axis, ast.Name) and axis.id in scalars:
            axis_names.add(axis.id)
    if len(axis_names) != 1:
        return None
    name = next(iter(axis_names))
    index_spaces = axis_index_spaces(fn, ranks)
    uses = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == name]
    if not all(id(u) in axis_spaces or id(u) in index_spaces for u in uses):
        return None
    # An operand whose rank the table does not know reveals nothing and is skipped; one that
    # disagrees is a second rank space and refuses the dispatch.
    counts = {index_spaces[id(u)] for u in uses if id(u) in index_spaces}
    for u in uses:
        space = axis_spaces.get(id(u))
        if space is not None:
            counts.add(space)
    if len(counts) != 1:
        return None
    rank = counts.pop()
    return (name, rank) if 1 <= rank <= MAX_DISPATCH_RANK else None


def specialize_runtime_axis(
    fn: ast.FunctionDef, name: str, rank: int, params: frozenset[str], resolve: Callable[[ast.FunctionDef], None]
) -> None:
    """Emit one specialised body per axis, selected at run time by ``name``.

    Scope is the WHOLE body, not the one call whose axis is symbolic: the axis reaches the narrow
    slice, the take, the expand_dims and the concatenate alike, and the temporaries between them
    have a different SHAPE per axis (``(N-1, M)`` against ``(N, M-1)``). A per-op dispatch would
    have to agree on one shape for each of those, so the branch has to contain every statement that
    produces or consumes an axis-dependent value -- which is all of them.

    Each branch is a full clone with the axis substituted, then run through ``resolve`` (the
    structural-axis stage) as if it were the whole kernel, so its nest is chosen exactly as a
    literal-axis kernel's is. Locals are prefixed per branch because one name cannot carry two
    shapes in the emitter's declaration table.

    An OUT-OF-RANGE axis matches no branch, so the kernel writes nothing and leaves every output
    buffer as the caller passed it. numpy raises ``AxisError`` here and a void kernel has no way to
    report that; declining to write is the one behaviour that is neither a wrong answer nor a
    silent one, since the harness compares against a reference that raised.
    """
    branches: list[list[ast.stmt]] = []
    for axis in range(rank):
        clone = copy.deepcopy(fn)
        SubstituteAxisLiteral(name, axis).visit(clone)
        rename = {n: f"__ax{axis}_{n}" for n in rebound_names(clone) - params}
        RenameLocals(rename).visit(clone)
        ast.fix_missing_locations(clone)
        resolve(clone)
        branches.append(clone.body)
    chain: list[ast.stmt] = []
    for axis in reversed(range(rank)):
        # Both spellings of the same axis share a branch; nothing else may enter one.
        test = ast.BoolOp(
            op=ast.Or(),
            values=[
                ast.Compare(
                    left=ast.Name(id=name, ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value=value)]
                )
                for value in (axis, axis - rank)
            ],
        )
        chain = [ast.If(test=test, body=branches[axis], orelse=chain)]
    fn.body = chain
    ast.fix_missing_locations(fn)


class SubstituteAxisLiteral(ast.NodeTransformer):
    """Replace every read of the dispatched axis with the literal that branch stands for."""

    def __init__(self, name: str, axis: int) -> None:
        self.name = name
        self.axis = axis

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id != self.name or not isinstance(node.ctx, ast.Load):
            return node
        return ast.copy_location(ast.Constant(value=self.axis), node)


class RenameLocals(ast.NodeTransformer):
    """Give one branch's locals their own names, so two branches can size the same source-level
    temp differently."""

    def __init__(self, rename: dict[str, str]) -> None:
        self.rename = rename

    def visit_Name(self, node: ast.Name) -> ast.AST:
        new = self.rename.get(node.id)
        return node if new is None else ast.copy_location(ast.Name(id=new, ctx=node.ctx), node)
