"""Slice fusion: slice-bearing assignments lowered to one fused loop nest."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.lib_nodes.dims import shape_exprs_equal
from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import iter_extent_of_, span_multiple_of
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import (
    slice_step_any,
    step_is_negative,
    step_node,
    const_,
    const_or_name,
)
from hpcagent_bench.translators.numpyto_common.lowering.complex import infer_complex_dtype
from hpcagent_bench.translators.numpyto_common.lowering.indexing import (
    binop,
    has_any_slice,
    iter_var_name,
    name_of_subscript,
    slice_dims,
)
from hpcagent_bench.translators.numpyto_common.lowering.shape_harvest import is_scalar_helper_call
from hpcagent_bench.translators.numpyto_common.lowering.shape_reads import is_newaxis, negative_literal_offset
from hpcagent_bench.translators.numpyto_common.lowering.slice_scalarize import SliceToScalarRewriter
from hpcagent_bench.translators.numpyto_common.lowering.views import refuse_scalarising_a_contraction
from hpcagent_bench.translators.numpyto_common.subscripts import has_slice_subscript


def strided_trip_count(start: ast.expr, stop: ast.expr, step) -> ast.expr:
    """Element count of ``start:stop:step`` for a POSITIVE step: ``ceil((stop - start) / step)``.

    Folded to a literal when both bounds AND the step are constants, so the common ``a[0:2 * n:2]``
    shape keeps a plain loop bound instead of pushing a division into every backend. A symbolic
    step keeps the division: it is an ABI argument, so no value of it may be baked in.

    A span that is a MULTIPLE of the step is folded even when both are symbolic:
    ``ceil(A * s / s) == A`` exactly, for every positive ``s``. The pooling kernels slice
    ``padded[kz:kz + out * stride:stride]``, and left unfolded that extent reads
    ``(out * stride + stride - 1) // stride`` -- the same number as ``out``, spelled so that no
    token comparison can see it. ``rhs_is_whole_array`` then declined ``out = np.maximum(out,
    window)`` as a shape mismatch, the whole-array expansion never ran, and the emitters rendered
    a scalar ``max`` of two pointers.
    """
    if isinstance(step, ast.expr):
        span = stop if (isinstance(start, ast.Constant) and start.value == 0) else binop(stop, ast.Sub(), start)
        multiple = span_multiple_of(span, step)
        if multiple is not None:
            return multiple
        return binop(binop(binop(span, ast.Add(), step), ast.Sub(), const_(1)), ast.FloorDiv(), step)
    if isinstance(start, ast.Constant) and isinstance(stop, ast.Constant):
        return const_(max(0, -(-(stop.value - start.value) // step)))
    span = stop if (isinstance(start, ast.Constant) and start.value == 0) else binop(stop, ast.Sub(), start)
    return binop(binop(span, ast.Add(), const_(step - 1)), ast.FloorDiv(), const_(step))


class SliceFusion(ast.NodeTransformer):
    """Rewrite slice-bearing assignments into a single fused loop.

    Handles the canonical jacobi-style pattern::

        A[a0:b0, a1:b1] = expr

    where ``expr`` may contain any number of nested ``B[c0:d0, c1:d1]``
    references that share the same logical shape as the LHS. The
    rewriter picks one iteration variable per axis and replaces every
    slice with a scalar subscript indexed by the iter var (plus the
    offset between the slice's start and the LHS slice's start).

    A POSITIVE ``step`` on the assignment target is supported: the axis
    iterates its LOGICAL position ``k`` in ``range(0, count)`` and the
    target is written at ``start + k * step``, so the RHS mapping (which
    reads ``k`` as the position) needs no division.

    Limitations -- raised as :class:`NotImplementedError`:

    * a NEGATIVE ``step`` on the assignment target -- numpy seeds the
      reverse start at ``axis_len - 1`` when the bound is omitted, which
      needs the axis length the local-array case does not always carry,
    * slices whose ``stop`` is omitted on an array whose shape we
      cannot resolve (the IR only carries shape symbols for declared
      parameters; for local arrays declared via ``np.zeros`` we have
      shape info too).
    """

    def __init__(self, array_shapes: dict[str, list[str]]) -> None:
        self.array_shapes = array_shapes
        #: Monotonic id for the invariant-read temps staged ahead of a fused nest.
        self.invariant_ctr: list[int] = [0]

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        return self.rewrite_(node.targets[0], node.value, aug_op=None) or node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        self.generic_visit(node)
        return self.rewrite_(node.target, node.value, aug_op=node.op) or node

    def rewrite_(self, target: ast.AST, value: ast.expr, aug_op: ast.AST | None) -> ast.AST | None:
        """Common slice-fusion path for both Assign and AugAssign.

        ``aug_op`` is ``None`` for plain Assign or the augmented operator
        (``ast.Add``, ``ast.Mult`` etc.) when invoked from AugAssign --
        the rewritten body becomes ``LHS = LHS op RHS`` per element so
        AugAssign semantics survive the per-element expansion.
        """
        if not has_any_slice(target):
            return None
        if not isinstance(target, ast.Subscript):
            return None
        refuse_scalarising_a_contraction(value)
        lhs_name = name_of_subscript(target)
        if lhs_name is None:
            return None
        lhs_dims = slice_dims(target)
        # Compute the per-axis iteration range = LHS slice bounds.
        # Negative-index slice bounds ``A[1:-1]`` (or any int < 0) are
        # numpy-style ``axis_length + K``; resolve here so downstream
        # passes see fully concrete bounds.
        # Each entry is ``(loop_lo, loop_hi, step, slice_start)``. For a unit step the iter var IS
        # the destination coordinate, so ``loop_lo == slice_start``; for a strided target the iter
        # var is the logical position and ``loop_lo`` is 0 -- consumers reading ``rng[0]`` as "what
        # to subtract from the iter var to get the position" stay correct in both cases.
        ranges: list[tuple[ast.AST, ast.AST, int, ast.AST]] = []
        for axis, d in enumerate(lhs_dims):
            if not isinstance(d, ast.Slice):
                ranges.append((d, d, 1, d))
                continue
            step = 1 if d.step is None else slice_step_any(d)
            if step is None:
                raise NotImplementedError(
                    f"slice step {ast.unparse(d.step)!r} on an assignment target must be a compile-time integer"
                )
            start = self.resolve_bound(d.lower, lhs_name, axis, default=const_(0))
            stop = self.resolve_bound(d.upper, lhs_name, axis, default=lambda: self.axis_length(lhs_name, axis))
            if step == 1:
                ranges.append((start, stop, 1, start))
                continue
            if step_is_negative(step):
                raise NotImplementedError(f"negative slice step {step} on an assignment target is not supported")
            ranges.append((const_(0), strided_trip_count(start, stop, step), step, start))
        # Build the per-axis scalarisation: iter var ``i_axis`` ranging
        # ``[start, stop)``; every RHS subscript gets the iter var
        # offset by the LHS slice's start.
        iter_vars: list[ast.Name] = []
        for axis in range(len(ranges)):
            if not isinstance(lhs_dims[axis], ast.Slice):
                iter_vars.append(None)  # type: ignore[arg-type]
                continue
            iter_vars.append(ast.Name(id=iter_var_name(axis), ctx=ast.Load()))

        # LHS subscript: iter var per slice axis, indexed absolute (not
        # relative to the slice's own start).
        new_lhs = ast.Subscript(
            value=target.value,
            slice=self.scalar_slice(lhs_dims, iter_vars, ranges, lhs_name),
            ctx=ast.Store(),
        )

        # A loop-invariant ELEMENT read of the array this statement writes is served from a
        # slot a previous iteration may already have stored to, so it is staged ahead of the
        # nest -- see :class:`HoistInvariantSelfReads`.
        hoister = HoistInvariantSelfReads(lhs_name, self.array_shapes, self.invariant_ctr)
        rhs_rewriter = SliceToScalarRewriter(self.array_shapes, iter_vars, ranges, lhs_name, lhs_dims)
        new_rhs = rhs_rewriter.visit(hoister.visit(copy.deepcopy(value)))
        # A top-level RHS Name (``corr[i+1:M, i] = __mm4``) isn't visited by
        # NodeTransformer unless asked -- subscriptify it explicitly.
        new_rhs = rhs_rewriter.maybe_subscriptify(new_rhs)

        if aug_op is None:
            inner: ast.stmt = ast.Assign(targets=[new_lhs], value=new_rhs)
        else:
            inner = ast.AugAssign(target=new_lhs, op=aug_op, value=new_rhs)

        # One ``for`` per slice axis (scalar dims are skipped).
        body: list[ast.stmt] = [inner]
        for axis in reversed(range(len(lhs_dims))):
            if not isinstance(lhs_dims[axis], ast.Slice):
                continue
            lo, hi = ranges[axis][0], ranges[axis][1]
            ivar = iter_vars[axis]
            body = [
                ast.For(
                    target=ast.Name(id=ivar.id, ctx=ast.Store()),
                    iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[lo, hi], keywords=[]),
                    body=body,
                    orelse=[],
                )
            ]
        if hoister.staged:
            return [*hoister.staged, *body]
        return body[0] if len(body) == 1 else body

    def axis_length(self, array_name: str, axis: int) -> ast.AST:
        shape = self.array_shapes.get(array_name)
        if shape is None or axis >= len(shape):
            raise NotImplementedError(
                f"slice with omitted stop on {array_name!r} axis {axis}: shape unknown to NumpyToC"
            )
        return const_or_name(shape[axis])

    def resolve_bound(self, bound: ast.AST | None, array_name: str, axis: int, default) -> ast.AST:
        """Resolve a slice bound, expanding numpy's negative-index form.

        A bound of ``None`` -> ``default`` (typically 0 for start,
        axis length for stop). A bound of ``-K`` (integer constant or
        ``UnaryOp(USub, Constant)``) -> ``axis_length - K``. All other
        bounds pass through unchanged so symbolic expressions like
        ``N-1`` survive.

        ``default`` may be a plain AST node or a zero-arg callable
        producing one; the stop bound's default calls ``axis_length``,
        which can raise ``NotImplementedError`` on an array with unknown
        shape, so it must only be evaluated when the bound is actually
        omitted -- not on every explicit-stop slice (e.g. ``a[:n]``).
        """
        if bound is None:
            return default() if callable(default) else default
        k = negative_literal_offset(bound)
        if k is not None:
            return binop(self.axis_length(array_name, axis), ast.Sub(), const_(k))
        return bound

    def scalar_slice(self, lhs_dims, iter_vars, ranges, name) -> ast.AST:
        """Build the LHS scalar subscript: iter vars per slice dim,
        original (negative-resolved) index per non-slice dim."""
        idx_nodes: list[ast.AST] = []
        for axis, d in enumerate(lhs_dims):
            if isinstance(d, ast.Slice):
                ivar = ast.Name(id=iter_vars[axis].id, ctx=ast.Load())
                step, slice_start = ranges[axis][2], ranges[axis][3]
                if step == 1:
                    idx_nodes.append(ivar)
                    continue
                scaled = binop(ivar, ast.Mult(), step_node(step))
                idx_nodes.append(
                    scaled
                    if (isinstance(slice_start, ast.Constant) and slice_start.value == 0)
                    else binop(slice_start, ast.Add(), scaled)
                )
            else:
                idx_nodes.append(self.resolve_scalar_index(d, name, axis))
        if len(idx_nodes) == 1:
            return idx_nodes[0]
        return ast.Tuple(elts=idx_nodes, ctx=ast.Load())

    def resolve_scalar_index(self, idx: ast.AST, name: str, axis: int) -> ast.AST:
        """A negative constant scalar index ``-K`` (e.g. ``y[:, -1]``)
        wraps to ``axis_length - K`` -- numpy semantics. C has no
        wrap-around, so leaving it literal indexes ``arr[... + (-1)]``
        out of bounds (the deriche heap corruption). Other indices pass
        through unchanged so ``N - 1`` etc. survive."""
        k = negative_literal_offset(idx)
        if k is not None:
            return binop(self.axis_length(name, axis), ast.Sub(), const_(k))
        return idx


#: Prefix of the temps :class:`HoistInvariantSelfReads` stages ahead of a fused loop nest.
INVARIANT_SELF_READ_PREFIX = "__sfinv"


class HoistInvariantSelfReads(ast.NodeTransformer):
    """Stage every loop-invariant ELEMENT read of the array a fused assignment writes.

    :class:`SliceFusion` turns ``A[k, k:] = A[k, k:] / A[k, k]`` into a loop that stores one
    element per iteration, and the ``si1 == k`` iteration overwrites the pivot ``A[k, k]``:
    every later iteration then divides by the value it just stored. numpy evaluates the whole
    RHS against the PRE-assignment array, so the pivot is read once, ahead of the nest. The
    Gauss-elimination family (``row -= factor * pivot_row``) is where this bites.

    Invariance is decided structurally -- full rank, no ``Slice``, no newaxis, no index array
    in any axis -- so the staged value is by construction the one every iteration would have
    loaded. That is why staging is also correct for a NON-aliasing kernel (cholesky / lu read
    ``A[k, k]`` while writing rows ``k + 1:``): same value, one load instead of a trip count's
    worth. A read under an ``IfExp`` / ``BoolOp`` keeps its guard -- hoisting past the test
    that exists to keep the element from being addressed would fault where numpy does not.
    """

    def __init__(self, lhs_name: str, array_shapes: dict[str, list[str]], counter: list[int]) -> None:
        self.lhs_name = lhs_name
        self.array_shapes = array_shapes
        self.counter = counter
        #: ``<name> = <read>`` assignments to place before the nest, in staging order.
        self.staged: list[ast.stmt] = []
        self._by_source: dict[str, str] = {}

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if not self.is_invariant_element(node):
            return node
        key = ast.unparse(node)
        name = self._by_source.get(key)
        if name is None:
            self.counter[0] += 1
            name = f"{INVARIANT_SELF_READ_PREFIX}{self.counter[0]}"
            self._by_source[key] = name
            self.staged.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=node))
        return ast.Name(id=name, ctx=ast.Load())

    def is_invariant_element(self, node: ast.Subscript) -> bool:
        """True when ``node`` reads ONE element of the written array at an index no iter var moves."""
        if not (isinstance(node.value, ast.Name) and node.value.id == self.lhs_name):
            return False
        if not isinstance(node.ctx, ast.Load):
            return False
        rank = len(self.array_shapes.get(self.lhs_name) or ())
        dims = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        if rank == 0 or len(dims) != rank:
            return False
        for dim in dims:
            if isinstance(dim, ast.Slice) or is_newaxis(dim):
                return False
            if isinstance(dim, ast.Attribute) and dim.attr == "newaxis":
                return False
            # An index ARRAY makes the read a gather, whose result is not one element.
            if any(isinstance(n, ast.Name) and n.id in self.array_shapes for n in ast.walk(dim)):
                return False
        return True


class LiftFreshArrayFromSlices(ast.NodeTransformer):
    """Convert ``lap_field = expr_with_slice_subscripts`` into

        lap_field = np.zeros((extent,));   # registered as a local
        lap_field[:] = expr

    so the existing :class:`SliceFusion` lowers the per-element form.

    Triggered when the LHS is a bare Name without a shape entry and
    the RHS contains at least one Subscript with a Slice axis whose
    iteration extent is derivable.

    The new ``Name = np.zeros(...)`` is a marker -- we don't emit any
    initializer; the LHS storage is declared by the emitter from
    ``zeros_locals``. The marker is dropped by stamping the RHS as
    ``__hpcagent_bench_zeros__()`` (which the emitter already swallows).
    """

    def __init__(
        self,
        shapes: dict[str, list[str]],
        local_dtypes: dict[str, str] | None = None,
        scalar_helpers: set[str] | None = None,
    ) -> None:
        self.shapes: dict[str, list[str]] = dict(shapes)
        #: By-value scalar helpers -- see :func:`is_scalar_helper_call`. A call to one is rank 0
        #: even though its ARGUMENTS carry slices (``bratu_dot(Q[p, :, :], w, N)``).
        self.scalar_helpers: set[str] = set(scalar_helpers or ())
        self.new_locals: dict[str, tuple[str, ...]] = {}
        # Side-effect: when the RHS contains a complex literal like
        # ``1j``, infer that the fresh local should be declared as
        # complex128 (mandelbrot ``C = X + Y[:, None] * 1j``).
        self.local_dtypes: dict[str, str] = local_dtypes if local_dtypes is not None else {}

    def run(self, tree: ast.AST) -> dict[str, tuple[str, ...]]:
        """Mutate ``tree`` in place and return the new-local shape map."""
        self.visit(tree)
        return self.new_locals

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            return node
        if is_scalar_helper_call(node.value, self.scalar_helpers):
            return node
        if not (has_slice_subscript(node.value) or self.is_array_binop(node.value)):
            return node
        ext = iter_extent_of_(node.value, self.shapes)
        if ext is None:
            return node
        shape_toks: tuple[str, ...] = tuple(ast.unparse(e) for e in ext)
        existing = self.shapes.get(target.id)
        # If the target already has a shape that matches the derived
        # extent, lift unconditionally (this is the
        # ``C = X + Y[:, None] * 1j`` case where an earlier rewriter
        # already deduced C's shape via broadcasting). Otherwise the
        # target must be a fresh local.
        rebind = existing is not None
        if rebind:
            if len(existing) != len(shape_toks) or not all(
                shape_exprs_equal(a, b) for a, b in zip(existing, shape_toks)
            ):
                return node
        else:
            self.new_locals[target.id] = shape_toks
        self.shapes[target.id] = list(shape_toks)
        # Infer complex dtype when the RHS contains any complex literal
        # ``1j`` (or operates on an already-complex array). C99
        # ``_Complex`` is assignment-compatible with real ``double`` in
        # C (with a warning) but the C++ emit path is a hard type error,
        # so we must tag the LHS as complex when the value is complex.
        if target.id not in self.local_dtypes:
            inferred = infer_complex_dtype(node.value, self.local_dtypes)
            if inferred is not None:
                self.local_dtypes[target.id] = inferred
        # A target that ALREADY had this shape is a live buffer being rebound, not a fresh local:
        # the bare marker reads as a genuine ``np.zeros`` reset, so the emitters memset the buffer
        # immediately before the loop that reads it. ``_conv3d``'s ``out = out + bias.reshape(..)``
        # had its whole convolution result wiped that way. Same sentinel pair
        # :meth:`WholeArrayAssignRewriter._expand` stamps, for the same reason -- the per-element
        # store below overwrites every element, so there is nothing to zero, and ``self_ref`` keeps
        # a deferred-malloc emitter from reallocating a buffer the RHS still reads.
        marker_args: list[ast.expr] = []
        if rebind:
            self_ref = any(isinstance(n, ast.Name) and n.id == target.id for n in ast.walk(node.value))
            marker_args = [ast.Constant(value="__reassign__"), ast.Constant(value=self_ref)]
        marker = ast.Assign(
            targets=[ast.Name(id=target.id, ctx=ast.Store())],
            value=ast.Call(func=ast.Name(id="__hpcagent_bench_zeros__", ctx=ast.Load()), args=marker_args, keywords=[]),
        )
        # ``C[:]`` only iterates the first axis; for multi-D targets
        # we need ``C[:, :]`` so slice fusion emits a per-element loop
        # nest covering every axis (mandelbrot's
        # ``C = X + Y[:, None] * 1j`` is (yn, xn) and requires 2 loops).
        rank = len(shape_toks)
        if rank == 1:
            slice_form: ast.expr = ast.Slice(lower=None, upper=None, step=None)
        else:
            slice_form = ast.Tuple(
                elts=[ast.Slice(lower=None, upper=None, step=None) for unused in range(rank)], ctx=ast.Load()
            )
        slice_lhs = ast.Subscript(value=ast.Name(id=target.id, ctx=ast.Load()), slice=slice_form, ctx=ast.Store())
        slice_assign = ast.Assign(targets=[slice_lhs], value=node.value)
        return [marker, slice_assign]

    def is_array_binop(self, expr):
        """``True`` for a BinOp / UnaryOp whose tree contains at least
        one bare-Name reference to an array of rank >= 1 in
        ``self.shapes``. Recognises forms like ``__cb1 = mass * vel``
        that the call-hoister synthesises -- the lifter then registers
        a fresh local with the broadcast extent so the per-element copy
        loop materialises the buffer.
        """
        if not isinstance(expr, (ast.BinOp, ast.UnaryOp)):
            return False
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Name):
                s = self.shapes.get(sub.id)
                if s:
                    return True
        return False
