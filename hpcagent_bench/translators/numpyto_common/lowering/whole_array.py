"""Whole-array assignments between named arrays lowered to loop nests."""

import ast
import copy
from typing import Any

from hpcagent_bench.translators.numpyto_common import dtypes
from hpcagent_bench.translators.numpyto_common.frontend import substitute_inlined_scalar_defs, fold_shape_expr
from hpcagent_bench.translators.numpyto_common.lib_nodes.constructors import MESHGRID_AXIS_KW, expand_meshgrid
from hpcagent_bench.translators.numpyto_common.lib_nodes.dims import shape_exprs_equal
from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import (
    is_integer_expr,
    iter_extent_of_,
    extent_is_scalar,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import slice_step_any
from hpcagent_bench.translators.numpyto_common.lib_nodes.scalarize import scalarize_at_iters
from hpcagent_bench.translators.numpyto_common.lowering.complex import (
    ctor_complex_tag,
    dtype_carrying_operands,
    scalar_expr_complex,
)
from hpcagent_bench.translators.numpyto_common.lowering.indexing import const_, slice_dims
from hpcagent_bench.translators.numpyto_common.lowering.shape_harvest import is_scalar_helper_call
from hpcagent_bench.translators.numpyto_common.lowering.shape_reads import is_newaxis, token_to_ast
from hpcagent_bench.translators.numpyto_common.lowering.slice_scalarize import SliceToScalarRewriter
from hpcagent_bench.translators.numpyto_common.lowering.subscriptify import SubscriptifyNames
from hpcagent_bench.translators.numpyto_common.lowering.views import is_rank_preserving_slice_view
from hpcagent_bench.translators.numpyto_common.ordered import OrderedSet


def is_constructor_call(node: ast.Call) -> bool:
    """``True`` for ``np.zeros / np.empty / np.ones / np.full /
    np.zeros_like / np.empty_like`` -- the constructors whose
    semantics is allocation, NOT a shape-preserving elementwise op.
    Used by the whole-array rewriter to refuse expanding these
    forms into per-element loops."""
    if not (
        isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "np"
    ):
        return False
    return node.func.attr in {
        "zeros",
        "empty",
        "ones",
        "full",
        "ndarray",
        "zeros_like",
        "empty_like",
        "ones_like",
        "full_like",
        "linspace",
        "arange",
        "eye",
        "identity",
        "mgrid",
        # Shape-changing ops: NOT elementwise. The dedicated
        # ``expand_reshape / expand_repeat / expand_transpose``
        # paths handle these via the LibNodeRewriter.
        "reshape",
        "repeat",
        "transpose",
    }


def normalise_shape(shape) -> tuple[str, ...]:
    """Normalise shape tokens for structural comparison.

    Tokens like ``"H + 2"`` and ``"(H + 2)"`` denote the same extent
    but mismatch on a plain ``==``. Re-parse compound tokens via
    :func:`ast.parse` (mode='eval') and unparse back -- the result
    is canonical Python syntax that compares correctly regardless of
    the original wrapper parens.

    Parens are not the only spelling difference that survives an unparse: neighbouring slices of one
    array give ``hi - lo``, ``hi + 1 - (lo + 1)`` and ``hi - 1 - (lo - 1)`` for the SAME extent, and
    fv3_xppm builds every PPM limiter out of exactly that. Textual inequality read as a broadcast
    mismatch, so the whole-array expansion declined and the assignment reached emit as arithmetic on
    two pointers. :func:`fold_shape_expr` gathers the literals of such a chain, which is enough to
    make the three agree; it is evaluation-preserving (test_shape_expr_folding re-evaluates every
    rewrite), so an extent it equates really is equal.
    """
    out = []
    for tok in shape:
        try:
            if tok and not str(tok).isdigit() and not str(tok).isidentifier():
                parsed = ast.parse(str(tok), mode="eval").body
                out.append(fold_shape_expr(ast.unparse(parsed)))
                continue
        except (SyntaxError, ValueError):
            pass
        out.append(str(tok))
    return tuple(out)


def is_bool_expr(node: ast.AST, local_dtypes: dict[str, str]) -> bool:
    """True when ``node`` evaluates to a BOOLEAN (array): a comparison, a
    boolean connective (``and``/``or``/``not``), a bitwise ``& | ^`` of boolean
    operands, or a reference to a boolean-typed array. Used to type a derived
    local array (``mask = cfl_clip & owner``) as bool on every backend."""
    if isinstance(node, (ast.Compare, ast.BoolOp)):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
        return is_bool_expr(node.left, local_dtypes) and is_bool_expr(node.right, local_dtypes)
    # ``~x`` inverts a boolean MASK (logical negation) when x is boolean; on an
    # integer operand it is bitwise NOT and stays integer.
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return is_bool_expr(node.operand, local_dtypes)
    if isinstance(node, ast.Name):
        return local_dtypes.get(node.id) in ("bool", "bool_")
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name):
            return local_dtypes.get(node.value.id) in ("bool", "bool_")
        # A broadcast-reshaped COMPARISON is still boolean: ``(cj_list == ci_sh)[:, None, None]``.
        # Reading only the Name form declared such a local real and then assigned a LOGICAL into
        # it. Deliberately narrow -- a chained subscript of a bool ARRAY is left alone, because
        # cloudsc's int-as-bool locals are read back through ``INT()`` and retyping them logical
        # breaks that intrinsic.
        if isinstance(node.value, (ast.Compare, ast.BoolOp)):
            return True
        return False
    return False


def has_index_array(node: ast.Subscript, shape_table) -> bool:
    """True when a Subscript uses at least one integer index-ARRAY index
    (advanced indexing) -- ``u2[q, r, s]`` with q/r/s in the shape table, or
    ``A[B[i]]``. Distinguishes a fancy gather (whole-array-expandable) from a
    plain slice / scalar read."""
    if not isinstance(node, ast.Subscript):
        return False
    sl = node.slice
    elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
    for e in elts:
        if isinstance(e, ast.Name) and e.id in shape_table:
            return True
        if isinstance(e, ast.Subscript) and isinstance(e.value, ast.Name) and e.value.id in shape_table:
            return True
    return False


def np_func_call(value: ast.AST, name: str) -> ast.Call | None:
    """Return ``value`` when it is ``np.<name>(...)`` / ``numpy.<name>(...)``,
    else ``None`` -- used to recognise ``np.ix_`` / ``np.meshgrid`` calls."""
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == name
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id in ("np", "numpy")
    ):
        return value
    return None


def ix_call_args(value: ast.AST) -> list[ast.expr] | None:
    """The open-mesh index arrays of an ``np.ix_(a, b, c)`` call, else ``None``."""
    call = np_func_call(value, "ix_")
    return list(call.args) if call is not None else None


class WholeArrayAssignRewriter(ast.NodeTransformer):
    """Turn whole-array Assign / AugAssign between named arrays into
    per-element loops.

    Examples (where ``shape_table[name]`` returns the array shape)::

        x1 += __mm1                ->  for i in range(N): x1[i] += __mm1[i]
        out = a                    ->  for i in range(N): out[i] = a[i]

    Without this, the emitter would render ``x1 += __mm1`` as pointer
    arithmetic in C and as undefined Fortran.
    """

    def __init__(
        self,
        shape_table: dict[str, Any],
        real_arrays: set[str] | None = None,
        local_dtypes: dict[str, str] | None = None,
        scalar_defs: dict[str, ast.expr] | None = None,
        scalar_helpers: set[str] | None = None,
    ) -> None:
        # We mutate ``shape_table`` to track Name aliases per Assign in
        # source order. Use a local copy so the caller's table is not
        # repeatedly clobbered when an alias gets reassigned.
        self.shape_table = dict(shape_table)
        #: Keys present in the caller's table at entry -- anything the pass adds
        #: beyond these (a meshgrid output, a ``gsq = gx**2 + ...`` broadcast local)
        #: is a genuinely NEW local whose shape the later slice-fusion pass needs;
        #: see :attr:`discovered_shapes`.
        self._input_keys = set(shape_table)
        #: Kernel helpers emitted as by-value scalar functions -- see :func:`is_scalar_helper_call`.
        self.scalar_helpers: set[str] = set(scalar_helpers or ())
        #: Shared dtype tag table. Alias / BinOp expansions propagate
        #: dtype here so the emitter sees the right C type for a
        #: complex-RHS local that was never directly declared.
        self.local_dtypes: dict[str, str] = local_dtypes if local_dtypes is not None else {}
        #: New local array names introduced by alias propagation
        #: (``x = __cb2`` where ``x`` wasn't previously an array) --
        #: emitter must declare them as stack arrays.
        self.alias_locals: dict[str, tuple[str, ...]] = {}
        #: Scalar-dim definitions (``ny = nhalo + nj + nhalo``), for shape comparison only.
        #: Inlining leaves one extent spelled through the local and its sibling spelled out,
        #: and the two never compare equal as text -- fv3_dycore's whole PPM stack declined on
        #: ``__inl1_ny`` vs ``nhalo + nj + nhalo`` and reached emit as a slice expression.
        self._scalar_defs: dict[str, str] = dict(scalar_defs or {})
        self._norm_memo: dict[tuple[str, ...], tuple[str, ...]] = {}
        #: Monotonic id for the buffered fancy ``A[idx] += rhs`` snapshot temps.
        self._scatter_ctr = 0
        #: Per-name list of shapes recorded in source order, one entry
        #: per ``Name = expr`` reassignment that the rewriter
        #: expanded to a per-element loop nest. Consumed by the
        #: source-order shape resolver so reassigned locals (lenet's
        #: ``x = relu(...); x = maxpool2d(x); ...``) carry the
        #: THEN-current shape at each ``arr.shape[i]`` reference.
        self._reassign_shapes: dict[str, list[tuple[str, ...]]] = {}
        # Names that existed as declared kernel arrays before any
        # alias / matmul-hoist temps were synthesised. When the LHS of
        # a whole-array alias is not in this set we register it as a
        # fresh local for the emitter to declare.
        self._known_arrays = set(real_arrays) if real_arrays else set(shape_table.keys())
        #: ``grid = np.ix_(a, b, c)`` bindings: grid name -> the open-mesh index
        #: arrays, resolved at each ``A[grid]`` gather / ``A[grid] (+)= rhs``
        #: scatter use site (numpy advanced open-mesh indexing).
        self._ix_grids: dict[str, list[ast.expr]] = {}
        #: Monotonic id for open-mesh (``np.ix_``) gather / scatter loop iters.
        self._ix_ctr = 0

    @property
    def discovered_shapes(self) -> dict[str, tuple[str, ...]]:
        """Shapes the pass inferred for locals the caller's table did not yet know
        (meshgrid outputs, broadcast-BinOp locals such as ``gsq``). Merged back into
        the shared table so the downstream slice-fusion scalarizer sizes them, rather
        than treating a shapeless denominator (``rho_g / gsq``) as a bare pointer."""
        return {k: tuple(v) for k, v in self.shape_table.items() if k not in self._input_keys and v}

    def expand_(
        self,
        target: ast.Name,
        value: ast.expr,
        op: ast.AST | None = None,
    ) -> list[ast.stmt]:
        shape = self.shape_table.get(target.id)
        if not shape:
            return []
        iters = [f"__w{i}" for i in range(len(shape))]
        idx = (
            ast.Name(id=iters[0], ctx=ast.Load())
            if len(iters) == 1
            else ast.Tuple(elts=[ast.Name(id=i, ctx=ast.Load()) for i in iters], ctx=ast.Load())
        )
        lhs_sub = ast.Subscript(value=ast.Name(id=target.id, ctx=ast.Load()), slice=idx, ctx=ast.Store())
        # A slice- or newaxis-bearing RHS (``delta = xi[None, :, None, :] - x[aj][:, None, :, :]``)
        # has no Name to subscriptify: the operands are already Subscripts, and left alone they
        # reach the emitter as whole-array slices. Scalarise them against the nest first, with the
        # same rewriter a sliced LHS uses -- the target is a fresh full-extent local, so every axis
        # is a full slice starting at 0.
        rhs = copy.deepcopy(value)
        if any(isinstance(n, ast.Slice) or is_newaxis(n) for n in ast.walk(rhs)):
            iter_nodes = [ast.Name(id=i, ctx=ast.Load()) for i in iters]
            full_dims = [ast.Slice(lower=None, upper=None, step=None) for unused in iters]
            zero_ranges = [(const_(0), const_(0)) for unused in iters]
            rewriter = SliceToScalarRewriter(self.shape_table, iter_nodes, zero_ranges, target.id, full_dims)
            rhs = rewriter.visit(rhs)
            # Its own ``maybe_subscriptify`` at the TOP level only, exactly as the sliced-LHS
            # driver does. ``SubscriptifyNames`` must NOT follow it: that walker rewrites every
            # bare array Name it meets, including the base of a subscript this pass has already
            # scalarised, which chained ``nbfp[i, j, k]`` onto ``nbfp[ti[..], tj[..], 0]``.
            rhs = rewriter.maybe_subscriptify(rhs)
        else:
            # Replace any Name(value) whose shape matches with a per-element
            # subscript; scalars pass through.
            rhs = SubscriptifyNames(self.shape_table, iters).visit(rhs)
        if op is None:
            body = [ast.Assign(targets=[lhs_sub], value=rhs)]
        else:
            body = [ast.AugAssign(target=lhs_sub, op=op, value=rhs)]
        out = body
        for var, bound in zip(reversed(iters), reversed(shape)):
            out = [
                ast.For(
                    target=ast.Name(id=var, ctx=ast.Store()),
                    iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[token_to_ast(bound)], keywords=[]),
                    body=out,
                    orelse=[],
                )
            ]
        # Prepend a ``Name = __hpcagent_bench_zeros__("__reassign__", self_ref)`` marker so
        # the source-order shape resolver (``ResolveArrShape``) can pick
        # up the THEN-current shape of the LHS. The marker is a no-op at
        # emit-time -- the emitter declares the LHS once at the function
        # start. The ``"__reassign__"`` sentinel distinguishes it from a
        # genuine ``np.zeros(...)`` reset: a reassignment is immediately
        # followed by a per-element loop that FULLY overwrites the LHS, so
        # the emitter must NOT re-zero (memset) the buffer first -- doing
        # so corrupts a self-referential reassignment like bicgstab's
        # ``p = r + beta * (p - omega * v)`` (the loop reads the old p).
        # ``self_ref`` says the same about a symbolic size: rayleigh_ritz's ``U = U * np.sign(...)``
        # reads the OLD U at every index, so a deferred-malloc emitter may not realloc it either.
        if op is None:
            self_ref = any(isinstance(n, ast.Name) and n.id == target.id for n in ast.walk(value))
            marker = ast.Assign(
                targets=[ast.Name(id=target.id, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="__hpcagent_bench_zeros__", ctx=ast.Load()),
                    args=[
                        ast.Constant(value="__reassign__"),
                        ast.Constant(value=self_ref),
                    ],
                    keywords=[],
                ),
            )
            self._reassign_shapes.setdefault(target.id, []).append(tuple(shape))
            out = [marker] + out
        return out

    def expand_partial(self, target: ast.Subscript, value: ast.Name) -> list[ast.stmt]:
        """Lower ``A[i] = B`` where ``A[i]`` is a PARTIAL subscript (a row /
        sub-array, fewer integer indices than ``A``'s rank) and ``B`` is a
        whole array of the remaining shape, into a per-element copy loop over
        the trailing dims. This is the ``back[t] = np.argmax(scores, axis=0)``
        pattern after the reduction is hoisted to a temp ``B`` -- without the
        expansion the emitter renders ``A[i] = B`` as a pointer store.
        """
        name = target.value.id
        shape = self.shape_table.get(name)
        if not shape:
            return []
        sl = target.slice
        given = list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]
        if any(isinstance(g, ast.Slice) for g in given):
            return []  # a Slice index is whole-array, handled elsewhere
        k = len(given)
        if k >= len(shape):
            return []  # fully indexed -> a scalar store, not a row copy
        remaining = shape[k:]
        rhs_shape = self.shape_table.get(value.id)
        if not rhs_shape or len(rhs_shape) != len(remaining):
            return []
        iters = [f"__w{i}" for i in range(len(remaining))]
        full_idx = list(given) + [ast.Name(id=i, ctx=ast.Load()) for i in iters]
        lhs_sub = ast.Subscript(
            value=ast.Name(id=name, ctx=ast.Load()), slice=ast.Tuple(elts=full_idx, ctx=ast.Load()), ctx=ast.Store()
        )
        rhs = SubscriptifyNames(self.shape_table, iters).visit(copy.deepcopy(value))
        out: list[ast.stmt] = [ast.Assign(targets=[lhs_sub], value=rhs)]
        for var, bound in zip(reversed(iters), reversed(remaining)):
            out = [
                ast.For(
                    target=ast.Name(id=var, ctx=ast.Store()),
                    iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[token_to_ast(bound)], keywords=[]),
                    body=out,
                    orelse=[],
                )
            ]
        return out

    def expand_fancy_scatter_store(self, target: ast.Subscript, value: ast.expr, op) -> list[ast.stmt]:
        """Lower a fancy-index scatter store ``A[idx, c] (op)= rhs`` where one or
        more lead components CONTAIN an INDEX ARRAY (``idx``, or an expression
        over it like ``(idx + m) % n``) and the rest are scalars, into a
        per-element loop ``for k: A[idx[k], ...] (op)= rhs[k]``.

        numpy iterates equal-length 1-D index arrays elementwise together, so a
        REPEATED index array (``lap[idx, idx] = c``, the diagonal) or the SAME
        array buried inside an arithmetic expression (``lap[idx, (idx + m) %
        n] += w``, chebyshev's circulant band) shares ONE loop iterator across
        every position that touches it -- not a separate one per occurrence, and
        not left as a raw array reference (which would emit invalid index-array
        pointer arithmetic, or silently leave stale un-iterated reads).

        The C/Fortran emitter has no notion of array-valued subscripts, so a
        raw ``facb[nl] = v`` / ``tg[nl, 0] = psi[:, i]`` would emit an invalid
        ``arr[ptr] = ...``. The index array(s) give the trip count (they must
        all agree on length); the RHS is scalarised at the loop iter.

        A sliced axis may sit beside the index array (fv3's ``al[ia, :, :nk]``) and gets a loop of
        its own over that slice's extent, offset by its ``lower``. The iters are emitted in numpy's
        RESULT order -- the single advanced position first when a ``:`` precedes it (numpy moves
        that group to the FRONT), otherwise in subscript order -- so the RHS, which the shared
        scalarizer right-aligns against those iters, pairs element for element with the LHS whatever
        axis each side happens to carry its own index array on. Two advanced positions broadcast
        together and are not claimed, nor is a strided or negative-bounded slice, whose iter would
        need arithmetic this loop does not do."""
        if not isinstance(target.value, ast.Name):
            return []
        name = target.value.id
        if name not in self.shape_table:
            return []
        lead = list(target.slice.elts) if isinstance(target.slice, ast.Tuple) else [target.slice]
        slice_axes = [k for k, e in enumerate(lead) if isinstance(e, ast.Slice)]

        def plain_bound(e: ast.expr) -> bool:
            """A unit-step slice with non-negative literal bounds -- the only kind this loop sizes."""
            if slice_step_any(e) not in (None, 1):
                return False
            return not any(
                isinstance(b, ast.Constant) and isinstance(b.value, int) and b.value < 0 for b in (e.lower, e.upper)
            )

        if any(not plain_bound(lead[k]) for k in slice_axes):
            return []
        if slice_axes and len(self.shape_table.get(name, ())) < len(lead):
            return []  # a lead longer than the target's rank is not a subscript this can size
        # Every Name referenced in ``lead`` AS A BARE ARRAY -- a whole position
        # (``idx``) or one buried inside a BinOp/Mod expression (``(idx + m) %
        # n``) -- that is itself a known 1-D index array. A repeat of the SAME
        # name counts once (OrderedSet keeps the pick below deterministic). A
        # Name that is already the BASE of a Subscript (``src[__sat1]``) is
        # NOT collected: that is an ALREADY-scalarised element read (this
        # rewriter's own prior output, or any other already-lowered gather),
        # not a raw array still needing its own iteration -- re-treating it as
        # one double-wraps it (``src[__sc0][__sat1]``) and corrupts the loop.
        idx_names: OrderedSet = OrderedSet()

        def collect_bare_arrays(node: ast.expr, is_subscript_base: bool) -> None:
            if isinstance(node, ast.Name):
                if not is_subscript_base and node.id != name and len(self.shape_table.get(node.id, ())) == 1:
                    idx_names.add(node.id)
                return
            if isinstance(node, ast.Subscript):
                collect_bare_arrays(node.value, True)
                collect_bare_arrays(node.slice, False)
                return
            for child in ast.iter_child_nodes(node):
                collect_bare_arrays(child, False)

        for e in lead:
            collect_bare_arrays(e, False)
        if not idx_names:
            return []
        carriers = [
            k for k, e in enumerate(lead) if any(isinstance(n, ast.Name) and n.id in idx_names for n in ast.walk(e))
        ]
        if slice_axes and len(carriers) != 1:
            # Two advanced positions broadcast into ONE result axis block; this loop gives each its
            # own iter, which is a different (wrong) pairing.
            return []
        extents = {self.shape_table[n][0] for n in idx_names}
        if len(extents) != 1:
            return []  # disagreeing lengths -- not one broadcastable iteration plane
        idx_name0 = next(iter(idx_names))
        # A GATHER needs integer indices. A boolean array in this position is a MASK, and the mask
        # rewriter declines whenever it cannot prove the dtype -- ``m = flags.astype(bool);
        # out[m] = 0`` then landed here and scattered through the 0/1 truth values, writing only
        # out[0] and out[1]. Unknown dtype is unsafe for the same reason, so require integer.
        for idx_name in idx_names:
            if not dtypes.is_integer(self.local_dtypes.get(idx_name, "")):
                raise NotImplementedError(
                    f"{ast.unparse(target)}: index array {idx_name!r} is not a known integer "
                    f"dtype; a boolean here is a MASK, not a gather"
                )
        extent = self.shape_table[idx_name0][0]
        it = "__sc0"

        class IndexArraysAtIter(ast.NodeTransformer):
            """Replace every occurrence of an index-array Name with ``name[it]``,
            wherever it sits -- a whole lead position, or buried in arithmetic."""

            def visit_Name(self, node: ast.Name) -> ast.AST:
                if node.id in idx_names:
                    return ast.Subscript(
                        value=ast.Name(id=node.id, ctx=ast.Load()),
                        slice=ast.Name(id=it, ctx=ast.Load()),
                        ctx=ast.Load(),
                    )
                return node

        new_lead: list[ast.expr] = []
        rhs_carries_index = any(isinstance(n, ast.Name) and n.id in idx_names for n in ast.walk(value))
        # ``(iter, bound)`` in numpy RESULT-axis order; ``new_lead`` places each at its own axis.
        plan: list[tuple[str, ast.expr]] = []
        n_sliced = 0
        for k, e in enumerate(lead):
            if k in slice_axes:
                ivar = f"__scs{n_sliced}"
                n_sliced += 1
                hi = e.upper if e.upper is not None else token_to_ast(self.shape_table[name][k])
                bound = hi if e.lower is None else ast.BinOp(left=hi, op=ast.Sub(), right=copy.deepcopy(e.lower))
                pos: ast.expr = ast.Name(id=ivar, ctx=ast.Load())
                if e.lower is not None:
                    pos = ast.BinOp(left=pos, op=ast.Add(), right=copy.deepcopy(e.lower))
                new_lead.append(pos)
                plan.append((ivar, bound))
            else:
                new_lead.append(IndexArraysAtIter().visit(copy.deepcopy(e)))
                # Only a position that actually CARRIES an index array opens the shared iter; a
                # plain scalar axis (``A[idx, 0, :]``) contributes no result axis at all, and
                # letting it open one puts the iters out of step with the RHS.
                if k in carriers and it not in [i for i, unused in plan]:
                    # A single advanced position keeps its place in numpy's result, so the iters
                    # follow subscript order -- an RHS that does not mention the index array (a
                    # broadcast column, lulesh's ``areaX[:, None]``) right-aligns against exactly
                    # those axes. An RHS that DOES carry it is the exception: the substitution
                    # above already consumed that result axis from it, so the shared iter has to
                    # lead for the axes it has left to right-align against the innermost iters.
                    plan.insert(0 if rhs_carries_index else len(plan), (it, token_to_ast(extent)))
        iters = [i for i, unused in plan]
        bounds = [b for unused, b in plan]
        lhs_slice = new_lead[0] if len(new_lead) == 1 else ast.Tuple(elts=new_lead, ctx=ast.Load())
        lhs = ast.Subscript(value=ast.Name(id=name, ctx=ast.Load()), slice=lhs_slice, ctx=ast.Store())
        # The RHS reads the same index arrays at the same iter. An index EXPRESSION
        # (``src[ia - 1, :]``) reaches emit with the array name still bare otherwise, which is
        # the invalid pointer arithmetic this rewriter exists to avoid; substituting first
        # leaves ``SubscriptifyNames`` an already-scalarised element read, which it keeps.
        rhs = SubscriptifyNames(self.shape_table, iters).visit(IndexArraysAtIter().visit(copy.deepcopy(value)))

        def loop_(body_stmt: ast.stmt) -> ast.stmt:
            for ivar, bound in zip(reversed(iters), reversed(bounds)):
                body_stmt = ast.For(
                    target=ast.Name(id=ivar, ctx=ast.Store()),
                    iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[copy.deepcopy(bound)], keywords=[]),
                    body=[body_stmt],
                    orelse=[],
                )
            return body_stmt

        if op is None:
            # Plain fancy store ``A[idx, c] = rhs`` -- a single per-element loop.
            # Sequential last-write-wins on a repeated index equals numpy's
            # buffered fancy assignment, so no snapshot is needed.
            out: list[ast.stmt] = [loop_(ast.Assign(targets=[lhs], value=rhs))]
        else:
            # numpy fancy ``A[idx] += rhs`` is BUFFERED: it reads the OLD A[idx],
            # applies the op against rhs, and scatters back with LAST-WRITE-WINS for
            # a repeated index -- it does NOT accumulate (that is ``np.add.at``,
            # routed elsewhere). Snapshot the gathered old values into a temp, then
            # store, so a duplicate index matches numpy (a single in-place ``+=``
            # loop would over-count). The snapshot is a rank-1 local of A's dtype.
            gname = f"__scg{self._scatter_ctr}"
            self._scatter_ctr += 1
            # What numpy buffers is the whole written PLANE, so the snapshot carries one axis per
            # result iter -- a ``:`` beside the index array (lulesh's ``normal[:, corners, 0] +=``)
            # makes that a rank-2 read, and a rank-1 vector would fold the slice axis away.
            self.alias_locals[gname] = tuple(ast.unparse(b) for b in bounds)
            if name in self.local_dtypes:
                self.local_dtypes[gname] = self.local_dtypes[name]

            def g_index() -> ast.expr:
                names = [ast.Name(id=i, ctx=ast.Load()) for i in iters]
                return names[0] if len(names) == 1 else ast.Tuple(elts=names, ctx=ast.Load())

            g_store = ast.Subscript(value=ast.Name(id=gname, ctx=ast.Load()), slice=g_index(), ctx=ast.Store())
            g_load = ast.Subscript(value=ast.Name(id=gname, ctx=ast.Load()), slice=g_index(), ctx=ast.Load())
            a_load = ast.Subscript(
                value=ast.Name(id=name, ctx=ast.Load()), slice=copy.deepcopy(lhs_slice), ctx=ast.Load()
            )
            gather = loop_(ast.Assign(targets=[g_store], value=a_load))
            store = loop_(ast.Assign(targets=[copy.deepcopy(lhs)], value=ast.BinOp(left=g_load, op=op, right=rhs)))
            out = [gather, store]
        for s in out:
            ast.fix_missing_locations(s)
        return out

    def resolve_ix_operands(self, sub: ast.Subscript) -> list[ast.expr] | None:
        """The open-mesh index arrays of a ``A[grid]`` / ``A[np.ix_(...)]``
        subscript -- resolving a bound ``grid = np.ix_(...)`` (recorded in
        :attr:`_ix_grids`) or an inline call -- else ``None``."""
        idx = sub.slice
        if isinstance(idx, ast.Name) and idx.id in self._ix_grids:
            return self._ix_grids[idx.id]
        return ix_call_args(idx)

    def ix_dims(self, ops: list[ast.expr]) -> list[ast.expr] | None:
        """Per-operand length for a list of 1-D index arrays (the open-mesh /
        meshgrid axis extents), or ``None`` when any operand's shape is unknown
        or not rank-1."""
        dims: list[ast.expr] = []
        for op in ops:
            ext = iter_extent_of_(op, self.shape_table)
            if ext is None or len(ext) != 1:
                return None
            dims.append(ext[0])
        return dims

    def empty_alloc(self, name: str, dims: list[ast.expr]) -> ast.Assign:
        """``name = np.empty((d0, d1, ...))`` marker so the zeros harvest declares
        the fresh gather / meshgrid output; its element type rides on
        :attr:`local_dtypes`."""
        shape_tuple = ast.Tuple(elts=[copy.deepcopy(d) for d in dims], ctx=ast.Load())
        return ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="empty", ctx=ast.Load()),
                args=[shape_tuple],
                keywords=[],
            ),
        )

    def ix_iters(self, prefix: str, k: int) -> list[str]:
        self._ix_ctr += 1
        return [f"{prefix}{self._ix_ctr}_{d}" for d in range(k)]

    def expand_ix_gather(self, target: ast.Name, arr: ast.Name, ops: list[ast.expr]) -> list[ast.stmt] | None:
        """``vloc = A[np.ix_(a, b, c)]`` -> ``vloc[i,j,k] = A[a[i], b[j], c[k]]``.

        The result has shape ``(len(a), len(b), len(c))`` (open-mesh gather);
        ``A`` is read at the Cartesian product of the index arrays."""
        dims = self.ix_dims(ops)
        if dims is None:
            return None
        k = len(ops)
        iters = self.ix_iters("__ixg", k)
        read = [
            scalarize_at_iters(copy.deepcopy(op), [ast.Name(id=it, ctx=ast.Load())], self.shape_table)
            for op, it in zip(ops, iters)
        ]
        read_slot = read[0] if k == 1 else ast.Tuple(elts=read, ctx=ast.Load())
        src = ast.Subscript(value=ast.Name(id=arr.id, ctx=ast.Load()), slice=read_slot, ctx=ast.Load())
        out_slot = (
            ast.Name(id=iters[0], ctx=ast.Load())
            if k == 1
            else ast.Tuple(elts=[ast.Name(id=it, ctx=ast.Load()) for it in iters], ctx=ast.Load())
        )
        store = ast.Subscript(value=ast.Name(id=target.id, ctx=ast.Load()), slice=out_slot, ctx=ast.Store())
        body: list[ast.stmt] = [ast.Assign(targets=[store], value=src)]
        for it, dim in zip(reversed(iters), reversed(dims)):
            body = [
                ast.For(
                    target=ast.Name(id=it, ctx=ast.Store()),
                    iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[copy.deepcopy(dim)], keywords=[]),
                    body=body,
                    orelse=[],
                )
            ]
        self.shape_table[target.id] = tuple(ast.unparse(d) for d in dims)
        dt = self.local_dtypes.get(arr.id)
        if dt is not None:
            self.local_dtypes[target.id] = dt
        out: list[ast.stmt] = [self.empty_alloc(target.id, dims)] + body
        for s in out:
            ast.fix_missing_locations(s)
        return out

    def expand_ix_scatter(
        self, arr: ast.Name, ops: list[ast.expr], value: ast.expr, op: ast.AST | None
    ) -> list[ast.stmt] | None:
        """``A[np.ix_(a, b, c)] (op)= rhs`` -> a nested loop
        ``A[a[i], b[j], c[k]] (op)= rhs[i, j, k]`` over the Cartesian product of
        the index arrays. The index arrays are distinct per axis (the LS3DF
        periodic fragment-box placement), so every scattered cell is unique and a
        plain accumulate matches numpy's buffered ``A[ix_] += rhs`` bit-for-bit."""
        dims = self.ix_dims(ops)
        if dims is None:
            return None
        k = len(ops)
        iters = self.ix_iters("__ixs", k)
        lhs_idx = [
            scalarize_at_iters(copy.deepcopy(o), [ast.Name(id=it, ctx=ast.Load())], self.shape_table)
            for o, it in zip(ops, iters)
        ]
        lhs_slot = lhs_idx[0] if k == 1 else ast.Tuple(elts=lhs_idx, ctx=ast.Load())
        lhs = ast.Subscript(value=ast.Name(id=arr.id, ctx=ast.Load()), slice=lhs_slot, ctx=ast.Store())
        rhs = SubscriptifyNames(self.shape_table, iters).visit(copy.deepcopy(value))
        stmt: ast.stmt = (
            ast.Assign(targets=[lhs], value=rhs) if op is None else ast.AugAssign(target=lhs, op=op, value=rhs)
        )
        body: list[ast.stmt] = [stmt]
        for it, dim in zip(reversed(iters), reversed(dims)):
            body = [
                ast.For(
                    target=ast.Name(id=it, ctx=ast.Store()),
                    iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[copy.deepcopy(dim)], keywords=[]),
                    body=body,
                    orelse=[],
                )
            ]
        for s in body:
            ast.fix_missing_locations(s)
        return body

    def expand_meshgrid_unpack(self, target: ast.Tuple, value: ast.expr) -> list[ast.stmt] | None:
        """``g0, ..., g_{k-1} = np.meshgrid(a0, ..., a_{k-1}, indexing=...)`` ->
        per-output allocator + broadcast-copy loop nest (via
        :func:`expand_meshgrid`). Each output is a fresh local of its input's
        dtype whose shape follows the ``ij`` / ``xy`` convention."""
        call = np_func_call(value, "meshgrid")
        if call is None:
            return None
        names = [e.id for e in target.elts if isinstance(e, ast.Name)]
        if len(names) != len(target.elts):
            return None
        args = list(call.args)
        if not args or len(args) != len(names):
            return None
        in_dims = self.ix_dims(args)
        if in_dims is None:
            return None
        indexing = "xy"
        for kw in call.keywords:
            if kw.arg == "indexing" and isinstance(kw.value, ast.Constant):
                indexing = kw.value.value
        if indexing not in ("ij", "xy"):
            return None
        k = len(args)
        # perm maps an output axis to the input axis whose length it takes: a
        # single 0<->1 swap for 'xy', identity for 'ij'.
        perm = list(range(k))
        if indexing == "xy" and k >= 2:
            perm[0], perm[1] = 1, 0
        out_dims = [in_dims[perm[p]] for p in range(k)]
        out_shape_tokens = tuple(ast.unparse(d) for d in out_dims)
        out_stmts: list[ast.stmt] = []
        for d, gname in enumerate(names):
            # Output d carries input d's element type (numpy meshgrid preserves
            # per-input dtype); shape follows the indexing convention.
            if isinstance(args[d], ast.Name):
                dt = self.local_dtypes.get(args[d].id)
                if dt is not None:
                    self.local_dtypes[gname] = dt
            self.shape_table[gname] = out_shape_tokens
            kwargs = [
                ast.keyword(arg="indexing", value=ast.Constant(value=indexing)),
                ast.keyword(arg=MESHGRID_AXIS_KW, value=ast.Constant(value=d)),
            ]
            loops = expand_meshgrid(
                ast.Name(id=gname, ctx=ast.Store()), [copy.deepcopy(a) for a in args], self.shape_table, kwargs=kwargs
            )
            out_stmts.append(self.empty_alloc(gname, out_dims))
            out_stmts.extend(loops)
        for s in out_stmts:
            ast.fix_missing_locations(s)
        return out_stmts

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        self.refuse_boolean_gather(node.value)
        target = node.targets[0]
        # ``g0, g1, ... = np.meshgrid(a0, a1, ..., indexing=...)`` multi-output
        # tuple unpack -> one broadcast-copy loop nest per output.
        if isinstance(target, ast.Tuple):
            meshed = self.expand_meshgrid_unpack(target, node.value)
            if meshed is not None:
                return meshed
            return node
        # ``grid = np.ix_(a, b, c)`` open-mesh index binding: record the operands
        # and drop the statement (resolved at each ``A[grid]`` use site below).
        if isinstance(target, ast.Name):
            ix_ops = ix_call_args(node.value)
            if ix_ops is not None:
                self._ix_grids[target.id] = ix_ops
                return None
        # ``vloc = A[grid]`` (or ``A[np.ix_(...)]``) open-mesh GATHER.
        if (
            isinstance(target, ast.Name)
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
        ):
            ops = self.resolve_ix_operands(node.value)
            if ops is not None:
                gathered = self.expand_ix_gather(target, node.value.value, ops)
                if gathered is not None:
                    return gathered
        # ``A[grid] = rhs`` open-mesh scatter store (plain, no accumulate).
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            ops = self.resolve_ix_operands(target)
            if ops is not None:
                scattered = self.expand_ix_scatter(target.value, ops, node.value, None)
                if scattered is not None:
                    return scattered
        # Fancy-index scatter store ``A[idx, c] = rhs`` (idx an index array):
        # a per-element loop. Runs first so the sliced-RHS form the emitter
        # rejects (vexx ``tg[nl, 0] = psi[:, i]``) is lowered here.
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            scattered = self.expand_fancy_scatter_store(target, node.value, None)
            if scattered:
                return scattered
        # Per-statement shape table update for reassigned locals.
        # Without this, resnet's ``x = (padded - mean) / sqrt(std + eps)``
        # (after batchnorm inlining) sees ``x`` with its harvest-time
        # final shape and skips the whole-array expansion.
        if (
            isinstance(target, ast.Name)
            and isinstance(node.value, (ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Call))
            and not (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "__hpcagent_bench_zeros__"
            )
        ):
            ext = (
                None
                if is_scalar_helper_call(node.value, self.scalar_helpers)
                else iter_extent_of_(node.value, self.shape_table)
            )
            # All-size-1 broadcast -> a scalar local, not a ``T x[1]`` array (see extent_is_scalar).
            if ext is not None and not extent_is_scalar(ext):
                self.shape_table[target.id] = tuple(ast.unparse(e) for e in ext)
        # ``C[:] = expr`` on a multi-D array means whole-array elementwise
        # assignment in numpy; lower to a per-element loop that walks the
        # full extent and subscripts every Name expression on the RHS.
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id in self.shape_table
            and isinstance(target.slice, ast.Slice)
            and target.slice.lower is None
            and target.slice.upper is None
        ):
            # Skip constructor-style calls (np.repeat / reshape /
            # transpose / mgrid / zeros etc.) -- they are NOT
            # shape-preserving elementwise ops and lowering them
            # per-element produces nonsense like
            # ``D[w0, w1, w2] = np.repeat(arr[w0, w1, 0], K, axis=2)``.
            if isinstance(node.value, ast.Call) and is_constructor_call(node.value):
                return node
            name = target.value.id
            expanded = self.expand_(ast.Name(id=name, ctx=ast.Store()), node.value, None)
            if expanded:
                return expanded
        # ``A[i] = B`` -- partial-subscript LHS (a row / sub-array) assigned a
        # whole array of the remaining shape (the reduction-into-a-row pattern
        # ``back[t] = np.argmax(scores, axis=0)`` once the RHS is hoisted to a
        # temp Name). Expand to a per-element copy over the trailing dims.
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id in self.shape_table
            and not isinstance(target.slice, ast.Slice)
            and isinstance(node.value, ast.Name)
            and node.value.id in self.shape_table
        ):
            expanded = self.expand_partial(target, node.value)
            if expanded:
                return expanded
        # ``A[b] = <Nd slice/BinOp expression>`` -- a partial-subscript LHS (a
        # sub-array, plain SCALAR leading index) assigned a whole arithmetic
        # expression of the remaining shape (stencil_4d's
        # ``out_grid[b] = w_dist[-1]*padded[...]``). The trailing residual axes
        # loop element-by-element, mirroring the AugAssign partial-subscript path.
        # Guarded narrowly so gather stores / scatter (index-array lead, bare
        # Subscript gather RHS) keep their own handling: the lead must be plain
        # scalars (no index arrays) and the RHS broadcast extent must match the
        # residual rank.
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id in self.shape_table
            and not isinstance(target.slice, ast.Slice)
            and isinstance(node.value, (ast.BinOp, ast.UnaryOp, ast.IfExp))
        ):
            shape = self.shape_table.get(target.value.id)
            lead = list(target.slice.elts) if isinstance(target.slice, ast.Tuple) else [target.slice]
            if (
                shape
                and not any(isinstance(e, ast.Slice) for e in lead)
                and not any(isinstance(e, ast.Name) and self.shape_table.get(e.id) for e in lead)
            ):
                n_trailing = len(shape) - len(lead)
                rhs_ext = iter_extent_of_(node.value, self.shape_table)
                if n_trailing > 0 and rhs_ext is not None and len(rhs_ext) == n_trailing:
                    expanded = self.expand_partial_subscript(target, node.value, None)
                    if expanded:
                        return expanded
        # Track Name = Name aliases in source order so a reassigned ``x``
        # gets the shape of whichever RHS preceded each use. If the LHS
        # is a fresh local (not already an array), record it so the
        # emitter declares it.
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Name) and node.value.id in self.shape_table:
            rhs_shape = self.shape_table[node.value.id]
            self.shape_table[target.id] = rhs_shape
            if target.id not in self._known_arrays:
                self.alias_locals[target.id] = tuple(rhs_shape)
            # Carry the RHS's dtype tag (notably ``complex128``) to
            # the LHS so a ``tmp = __cb3`` alias of a complex temp
            # keeps the complex tag for the C/C++ declaration.
            rhs_dt = self.local_dtypes.get(node.value.id)
            if rhs_dt and target.id not in self.local_dtypes:
                self.local_dtypes[target.id] = rhs_dt
        # ``z = arr[...]`` -- scalar local taking its dtype from a
        # known-dtype array. Lets ``abs(z)`` on a complex-array element
        # route through the ``cabs`` complex-intrinsic path.
        if (
            isinstance(target, ast.Name)
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
            and target.id not in self.local_dtypes
        ):
            src_dt = self.local_dtypes.get(node.value.value.id)
            if src_dt is not None:
                self.local_dtypes[target.id] = src_dt
        # ``X = np.zeros/ones/empty/eye(shape, Y.dtype | np.complexNN)`` -- a fresh
        # complex work array (the eigh reduction's L / Li / C / V). The shape is
        # tracked elsewhere; here we tag the complex element type.
        if isinstance(target, ast.Name) and target.id not in self.local_dtypes and isinstance(node.value, ast.Call):
            ctag = ctor_complex_tag(node.value, self.local_dtypes)
            if ctag is not None:
                self.local_dtypes[target.id] = ctag
        # ``X = Y.copy()`` / ``np.copy(Y)`` / ``np.ascontiguousarray(Y)`` -- inherit
        # the source's (complex) dtype (the Jacobi copies its working matrix).
        #
        # ``np.transpose`` / ``np.conj`` / ``np.where`` belong here for the same reason: they
        # rearrange or select values, they do not change what a value IS. Left out, a matrix
        # reached through one of them was UNTYPED, and ``RealConjDropper`` reads untyped as real
        # -- so it deleted the ``conj`` from the eigh Jacobi's own rotation. The rotation stopped
        # being unitary and the eigenvalues of ``np.linalg.eigh(np.transpose(m))`` came back as
        # zeros, with nothing failing to say so.
        if (
            isinstance(target, ast.Name)
            and target.id not in self.local_dtypes
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
        ):
            f = node.value.func
            args = node.value.args
            dt = None
            for src in dtype_carrying_operands(node.value):
                dt = self.local_dtypes.get(src)
                if dt:
                    break
            if dt is None and f.attr == "where" and len(args) == 3:
                # Either arm decides it: numpy promotes, so one complex arm makes the result complex.
                for arm in args[1:]:
                    arm_dt = self.local_dtypes.get(arm.id) if isinstance(arm, ast.Name) else None
                    if arm_dt and arm_dt.startswith("complex"):
                        dt = arm_dt
                        break
            if dt and dt.startswith("complex"):
                self.local_dtypes[target.id] = dt
        # ``X = <scalar complex arithmetic>`` (``ephi = apq / m``) -- a scalar
        # BinOp/UnaryOp over complex operands. The array-BinOp branch below only
        # fires when the value is a whole-array expr (``iter_extent_of_`` non-None);
        # a scalar complex temp needs its own tag or the emit declares it real.
        if (
            isinstance(target, ast.Name)
            and target.id not in self.local_dtypes
            and isinstance(node.value, (ast.BinOp, ast.UnaryOp))
            and scalar_expr_complex(node.value, self.local_dtypes)
        ):
            self.local_dtypes[target.id] = "complex128"
        # ``x = BinOp(array, array)`` where ``x`` is a Name: infer
        # x's shape from the broadcast extent of the RHS and treat as
        # whole-array assignment. ``iter_extent_of_`` returns ``None``
        # for purely-scalar RHS expressions like
        # ``__inl_H_out = x.shape[1] - K + 1`` so they are not
        # misclassified as arrays.
        if isinstance(target, ast.Name) and isinstance(
            node.value, (ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Compare, ast.BoolOp)
        ):
            ext = iter_extent_of_(node.value, self.shape_table)
            # An all-size-1 broadcast (``t = (a[i] > x)`` with ``x`` shape ``(1,)``) is a SCALAR, not a
            # ``T t[1]`` array: numpyto reads size-1 arrays element-wise as ``x[0]``, so registering ``t`` as
            # an array here desyncs its scalar declaration (from ``t = 0`` / ``if t`` / ``out[0] = t``) from
            # the array-style ``t[__w0] = ...`` writes the extent drives below (a mix that will not compile).
            if ext is not None and not extent_is_scalar(ext):
                rhs_widest = tuple(ast.unparse(e) for e in ext)
                # Propagate inferred shape to the LHS so downstream
                # uses pick it up; declare as a fresh local if new.
                self.shape_table[target.id] = rhs_widest
                if target.id not in self._known_arrays:
                    self.alias_locals[target.id] = tuple(rhs_widest)
                # A whole-array boolean expression -- a Compare / BoolOp
                # (``owner = mask != 0``), a ``not``, or a bitwise ``& | ^`` of
                # boolean operands (``mask = cfl_clip & owner``) -- yields a
                # BOOLEAN array. Type it so both backends declare it bool
                # (Fortran ``logical``, C ``bool``) rather than real.
                if is_bool_expr(node.value, self.local_dtypes) and target.id not in self.local_dtypes:
                    self.local_dtypes[target.id] = "bool_"
                # Complex-dtype propagation for BinOp / UnaryOp RHS:
                # a subtree carrying a complex literal or a complex-
                # tagged Name promotes the LHS to ``complex128`` so
                # the emit declares the right C dtype.
                # By the expression's OPERANDS, not by ast.walk: walking promoted the result of
                # ``np.abs(v)`` to complex merely because the complex ``v`` appears inside it, and
                # a magnitude is real. _scalar_expr_complex stops at the calls that return a real.
                if target.id not in self.local_dtypes:
                    if scalar_expr_complex(node.value, self.local_dtypes):
                        self.local_dtypes[target.id] = "complex128"
                    else:
                        # Integer-typed whole-array result (``q = j % nx`` where
                        # j is int64) stays integer -- so an index array derived
                        # from arange keeps its int dtype through the % / * chain
                        # (fft_3d's q/r/s gather indices).
                        if isinstance(node.value, ast.BinOp) and is_integer_expr(
                            node.value, self.local_dtypes, set(self.shape_table)
                        ):
                            self.local_dtypes[target.id] = "int64"
        if isinstance(target, ast.Name) and target.id in self.shape_table:
            if isinstance(node.value, ast.Name) and self.shape_table.get(node.value.id) == self.shape_table[target.id]:
                expanded = self.expand_(target, node.value, None)
                if expanded:
                    return expanded
            # Whole-array BinOp / UnaryOp / IfExp / Call on the RHS:
            # lower to per-element loop. The _SubscriptifyNames walker
            # rewrites every array reference inside the expression to
            # its subscripted form. ``Call`` covers cases like
            # ``x = fmax(x, 0)`` (relu post math-rename) or
            # ``x = sqrt(arr)`` where every array operand has the same
            # shape as the LHS.
            elif isinstance(
                node.value, (ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Call, ast.Subscript, ast.Compare, ast.BoolOp)
            ):
                # Skip constructor calls (np.zeros / np.empty etc.) --
                # those are NOT shape-preserving Calls, and lowering
                # them per-element would emit garbage like
                # ``N[w0, w1] = np.zeros(...)``.
                if isinstance(node.value, ast.Call) and is_constructor_call(node.value):
                    return node
                # A bare-Subscript RHS is handled here ONLY when it is a fancy
                # gather (>=1 integer index-ARRAY index) -- ``__cb = u2[q, r, s]``
                # whose result extent equals the LHS (fft_3d's checksum gather).
                # A plain slice/scalar read (``x = a[:, i]``) must NOT be
                # whole-array-expanded here; it has its own handling and doing so
                # corrupts dense kernels (resnet / softmax / mlp).
                if isinstance(node.value, ast.Subscript):
                    # A basic-slice view bound to a name and then used BARE keeps no subscript for
                    # ``fold_slice_view_aliases`` to compose into, so materialise the crop here
                    # (transposed conv's ``out = canvas[:, :, p:p + oh, p:p + ow]``). Left alone it
                    # reached the emitter as a whole-array slice, and the next reassignment of the
                    # same name allocated over it -- a zeroed crop, not a refusal.
                    if is_rank_preserving_slice_view(node.value, self.shape_table, len(self.shape_table[target.id])):
                        expanded = self.expand_(target, node.value, None)
                        if expanded:
                            return expanded
                    if not has_index_array(node.value, self.shape_table):
                        return node
                    # Fancy gather ``pos_nb = pos[nb]`` (cfd / lavamd): the
                    # result extent already equals the LHS shape (registered by
                    # the harvest), so expand DIRECTLY. ``rhs_is_whole_array``
                    # would mis-reject it -- it treats the index array ``nb``
                    # (shape ``(nboxes,)``) as a value operand that must
                    # broadcast against the (nboxes, npart, 3) result.
                    expanded = self.expand_(target, node.value, None)
                    if expanded:
                        return expanded
                    return node
                # Only attempt if every array-valued subexpression has
                # the same shape as the LHS.
                if self.rhs_is_whole_array(node.value, target.id):
                    expanded = self.expand_(target, node.value, None)
                    if expanded:
                        return expanded
        return node

    def refuse_boolean_gather(self, value: ast.expr) -> None:
        """Refuse ``arr[m]`` where ``m`` is a proven BOOLEAN array.

        That is numpy COMPACTION -- a shorter result holding the entries where ``m`` is true -- not
        a gather. Scalarised as a gather it indexes through the 0/1 truth values, which compiles
        cleanly and bins every element as if its index were 0 or 1: azimint_naive returned NaN in 7
        of 8 bins that way, and nothing on the path said a word. The scatter side already refuses
        this (see "index array ... is not a known integer dtype"); the read side did not.
        """
        for sub in ast.walk(value):
            if not (
                isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name) and isinstance(sub.slice, ast.Name)
            ):
                continue
            if self.local_dtypes.get(sub.slice.id) in ("bool", "bool_"):
                raise NotImplementedError(
                    f"{ast.unparse(sub)}: {sub.slice.id!r} is a boolean MASK, so this "
                    f"selects a SHORTER array, not one element per index; indexing "
                    f"through its 0/1 values would be a wrong answer"
                )

    def norm_(self, shape) -> tuple[str, ...]:
        """Shape tokens with scalar-dim locals resolved, then folded -- the comparison form."""
        key = tuple(str(t) for t in shape)
        got = self._norm_memo.get(key)
        if got is None:
            got = normalise_shape(substitute_inlined_scalar_defs(key, self._scalar_defs))
            self._norm_memo[key] = got
        return got

    def rhs_is_whole_array(self, expr: ast.AST, lhs_name: str) -> bool:
        """Check that every array Name referenced in ``expr`` has a
        shape compatible with ``lhs_name``: equal, or broadcastable
        (rank <= LHS rank with each axis either equal to LHS or
        equal to 1). Names that appear as the ``.value`` of a
        Subscript are SKIPPED -- the Subscript itself yields a
        possibly lower-rank value, so the bare Name's declared
        rank does not constrain whole-array compatibility.
        """
        target_shape = self.shape_table.get(lhs_name)
        if target_shape is None:
            return False
        target_norm = self.norm_(target_shape)
        # Collect Names that appear as a Subscript value -- those
        # are accessed in lower-rank form via the Subscript, so the
        # bare Name's full-rank shape is not the relevant constraint.
        subscript_targets: set[str] = set()
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name):
                subscript_targets.add(sub.value.id)
        has_array = False
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Name):
                if sub.id in subscript_targets:
                    continue
                shape = self.shape_table.get(sub.id)
                if shape is None:
                    continue
                shape_norm = self.norm_(shape)
                if shape_norm == target_norm:
                    has_array = True
                    continue
                if self.broadcastable_to(shape_norm, target_norm):
                    has_array = True
                    continue
                return False
        if has_array:
            return True
        # An expression built ONLY from SLICES (the vectorised stencils'
        # ``padded[R-r:R+N-r, ...] + padded[R+r:R+N+r, ...]``) leaves no bare array Name to
        # constrain -- every one is a Subscript value, skipped above -- so read the extent off the
        # slices themselves rather than decline the whole-array assignment this rewriter exists
        # to lower.
        for sub in ast.walk(expr):
            if not (isinstance(sub, ast.Subscript) and any(isinstance(e, ast.Slice) for e in slice_dims(sub))):
                continue
            extent = iter_extent_of_(sub, self.shape_table)
            if extent is None:
                return False
            extent_norm = self.norm_(tuple(ast.unparse(e) for e in extent))
            if not self.same_extent(extent_norm, target_norm):
                return False
            has_array = True
        return has_array

    def same_extent(self, extent: tuple[str, ...], target: tuple[str, ...]) -> bool:
        """Same extent, allowing two spellings of one bound: ``R + N - r - (R - r)`` and
        ``R + N + r - (R + r)`` are both ``N``, and only the symbolic compare says so."""
        if extent == target:
            return True
        if len(extent) == len(target) and all(a == b or shape_exprs_equal(a, b) for a, b in zip(extent, target)):
            return True
        return self.broadcastable_to(extent, target)

    @staticmethod
    def broadcastable_to(shape, target_shape):
        """Numpy broadcast rule: align shapes right-to-left; each
        dim must match the target or be 1; missing dims are treated
        as 1."""
        if len(shape) > len(target_shape):
            return False
        # Right-align by padding the shorter (``shape``) with implicit 1s.
        offset = len(target_shape) - len(shape)
        for i, s in enumerate(shape):
            if s == target_shape[offset + i]:
                continue
            if s == "1":
                continue
            return False
        return True

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.target, ast.Name) and node.target.id in self.shape_table:
            expanded = self.expand_(node.target, node.value, node.op)
            if expanded:
                return expanded
        # ``A[grid] += rhs`` open-mesh SCATTER-ADD (grid = np.ix_(a, b, c), or an
        # inline ``A[np.ix_(...)] += rhs``): a nested accumulate loop over the
        # Cartesian product of the (distinct-per-axis) index arrays.
        if isinstance(node.target, ast.Subscript) and isinstance(node.target.value, ast.Name):
            ops = self.resolve_ix_operands(node.target)
            if ops is not None:
                scattered = self.expand_ix_scatter(node.target.value, ops, node.value, node.op)
                if scattered is not None:
                    return scattered
        # Fancy-index scatter ``A[idx, c] += rhs`` (idx an index array): lowered by
        # ``expand_fancy_scatter_store`` to a snapshot-gather + store pair, matching
        # numpy's BUFFERED fancy ``+=`` (old value, last-write-wins on a duplicate
        # index -- NOT an accumulate) (the QE ultrasoft ``rhoc[nl] += aux2 * sf`` /
        # ``rhoc[box] += ...``).
        if isinstance(node.target, ast.Subscript) and isinstance(node.target.value, ast.Name):
            scattered = self.expand_fancy_scatter_store(node.target, node.value, node.op)
            if scattered:
                return scattered
        # Partial-index subscript target: ``Sigma[k, E, a] += __mm2`` on a
        # rank-5 Sigma indexes only 3 axes, leaving a (Norb, Norb) residual
        # slice. Loop the residual axes so the matrix accumulate lands
        # element-by-element (scattering_self_energies).
        if isinstance(node.target, ast.Subscript):
            expanded = self.expand_partial_subscript(node.target, node.value, node.op)
            if expanded:
                return expanded
        return node

    def expand_partial_subscript(self, target: ast.Subscript, value: ast.expr, op: ast.AST) -> list[ast.stmt]:
        """Expand ``arr[lead] (op)= rhs`` where ``arr[lead]`` indexes only
        the leading axes of a higher-rank ``arr`` -- the trailing axes form
        a residual slice looped element-by-element. Returns [] when the
        target is not a partial scalar index."""
        if not isinstance(target.value, ast.Name):
            return []
        name = target.value.id
        shape = self.shape_table.get(name)
        if not shape:
            return []
        lead = list(target.slice.elts) if isinstance(target.slice, ast.Tuple) else [target.slice]
        if any(isinstance(e, ast.Slice) for e in lead):
            return []
        if any(isinstance(e, ast.Constant) and e.value is None for e in lead):
            return []
        n_trailing = len(shape) - len(lead)
        if n_trailing <= 0:
            return []  # full index -> scalar; nothing to loop
        iters = [f"__w{i}" for i in range(n_trailing)]
        trailing = shape[-n_trailing:]
        lhs_idx = ast.Tuple(elts=list(lead) + [ast.Name(id=i, ctx=ast.Load()) for i in iters], ctx=ast.Load())
        lhs_sub = ast.Subscript(value=ast.Name(id=name, ctx=ast.Load()), slice=lhs_idx, ctx=ast.Store())
        rhs = SubscriptifyNames(self.shape_table, iters).visit(copy.deepcopy(value))
        # ``op is None`` -> a plain ``arr[lead] = rhs`` store (stencil_4d's
        # ``out_grid[b] = w_dist[-1] * padded[...]`` slice-expression RHS);
        # otherwise the augmented accumulate (``out_grid[b] += ...``).
        leaf: ast.stmt = (
            ast.Assign(targets=[lhs_sub], value=rhs) if op is None else ast.AugAssign(target=lhs_sub, op=op, value=rhs)
        )
        out: list[ast.stmt] = [leaf]
        for var, bound in zip(reversed(iters), reversed(trailing)):
            out = [
                ast.For(
                    target=ast.Name(id=var, ctx=ast.Store()),
                    iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[token_to_ast(bound)], keywords=[]),
                    body=out,
                    orelse=[],
                )
            ]
        return out
