"""The lowering pipeline: :class:`LoweringContext`, the ordered phases, and :func:`lower`."""

import ast
import copy
import operator
import os
import re
from collections.abc import Callable

from hpcagent_bench.translators.numpyto_common import dtypes
from hpcagent_bench.translators.numpyto_common.frontend import (
    collect_inlined_scalar_defs,
    resolve_shape_attr_tokens,
    substitute_inlined_scalar_defs,
)
from hpcagent_bench.translators.numpyto_common.ir import KernelIR, stamp_symbol_assumptions
from hpcagent_bench.translators.numpyto_common.lib_nodes.array_methods import ArrayMethodRewriter
from hpcagent_bench.translators.numpyto_common.lib_nodes.dims import DIM_IDENT_RE
from hpcagent_bench.translators.numpyto_common.lib_nodes.fft import FFT_LIBRARY_MARKER, FFTN_LIBRARY_MARKER
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import const_or_name
from hpcagent_bench.translators.numpyto_common.lib_nodes.linalg import reset_temp_counters
from hpcagent_bench.translators.numpyto_common.lib_nodes.rewriter import LibNodeRewriter
from hpcagent_bench.translators.numpyto_common.lowering.calls import (
    AstypeRewriter,
    ConditionalNoneAllocRewriter,
    EnumerateZipRewriter,
    MatmulCallRewriter,
    NpAliasRewriter,
    ReshapeMethodRewriter,
    ScalarTimesMatmulRewriter,
    TransposeRewriter,
)
from hpcagent_bench.translators.numpyto_common.lowering.casts import (
    BuiltinCastRewriter,
    ScalarFloatTagger,
    TrueDivisionPromoter,
)
from hpcagent_bench.translators.numpyto_common.lowering.chains import ChainedSubscriptFlattener
from hpcagent_bench.translators.numpyto_common.lowering.complex import (
    REAL_FOR_COMPLEX,
    PromoteMixedComplexIfExp,
    RealConjDropper,
    seed_complex_work_dtypes,
    walk_complex,
)
from hpcagent_bench.translators.numpyto_common.lowering.constructors import (
    CopyToAllocAndFill,
    EyeCallHoister,
    EyeToZerosDiagonal,
    FullCallHoister,
    FullLikeRewriter,
    MgridLowering,
    ZerosRewriter,
)
from hpcagent_bench.translators.numpyto_common.lowering.fft_grid import FftGridReshapeRewriter
from hpcagent_bench.translators.numpyto_common.lowering.forward_subst import (
    ForwardSubstituteInvariantScalars,
    SelfAssignDropper,
)
from hpcagent_bench.translators.numpyto_common.lowering.hoisting import ComputedIndexCallHoister, MethodCallRewriter
from hpcagent_bench.translators.numpyto_common.lowering.masks import (
    BooleanMaskReductionRewriter,
    BooleanMaskRewriter,
    collect_bool_names,
)
from hpcagent_bench.translators.numpyto_common.lowering.mathfuncs import MathRewriter
from hpcagent_bench.translators.numpyto_common.lowering.reshape_attr import ShapeAttrToReshape
from hpcagent_bench.translators.numpyto_common.lowering.scatter import ScatterAtRewriter
from hpcagent_bench.translators.numpyto_common.lowering.shape_harvest import collect_dim_aliases, harvest_local_shapes
from hpcagent_bench.translators.numpyto_common.lowering.shape_reads import (
    ResolveArrShape,
    ShapeMidExpressionRewriter,
    fold_shape_reads_in_table,
)
from hpcagent_bench.translators.numpyto_common.lowering.signature import (
    detect_output_and_index_arrays,
    fold_shape_aliases,
    promote_free_names_to_params,
    promote_shape_symbols_to_params,
    retype_int_helper_scalars,
)
from hpcagent_bench.translators.numpyto_common.lowering.slice_fusion import LiftFreshArrayFromSlices, SliceFusion
from hpcagent_bench.translators.numpyto_common.lowering.ssa import ssa_rename_reassigned
from hpcagent_bench.translators.numpyto_common.lowering.sugar import (
    DaceMapRewriter,
    MembershipToComparisons,
    UnrollConstRangeComprehension,
)
from hpcagent_bench.translators.numpyto_common.lowering.tuples import (
    ShapeTableTupleSplit,
    TupleLocalPropagator,
    TupleSubscriptFolder,
)
from hpcagent_bench.translators.numpyto_common.lowering.views import (
    EllipsisExpander,
    PadImplicitTrailingSlices,
    fold_slice_view_aliases,
    fold_subarray_aliases,
)
from hpcagent_bench.translators.numpyto_common.lowering.whole_array import WholeArrayAssignRewriter
from hpcagent_bench.translators.numpyto_common.ordered import OrderedSet
from hpcagent_bench.translators.numpyto_common.statement_desugar import DesugarArrayIteration, SplitChainedAssign

__all__ = [
    "INL_RE",
    "INVARIANT_ENV",
    "LOWER_PHASES",
    "LoweringContext",
    "assert_lowering_invariants",
    "dtype_verdict",
    "elementwise_store_base",
    "fix_real_scalar_dtypes",
    "fold_local_shape_attr_tokens",
    "lower",
    "lp_forward_substitute_invariant_scalars",
    "lp_libnode_expand",
    "lp_lower_helpers",
    "lp_normalize_calls",
    "lp_normalize_index_access",
    "lp_pre_libnode_normalize",
    "lp_promote_params",
    "lp_promote_true_division",
    "lp_resolve_inlined_shapes",
    "lp_scalarized_math_rename",
    "lp_scatter_at",
    "lp_seed_dtypes_and_harvest",
    "lp_seed_shape_table",
    "lp_slice_fusion_and_resolve",
    "lp_slice_normalize_and_lift",
    "lp_whole_array_and_zeros",
    "scalar_return_helpers",
    "tag_complex_locals",
]

#: Matches a residual inlined-scalar token (``__inl3_N``) or an unresolved
#: ``arr.shape[`` attribute access -- the never-worse guard in the inl resolver
#: keeps the original token whenever expansion would leave one of these behind.
INL_RE = re.compile(r"__inl\w*|\w+\.shape\[")


class LoweringContext:
    """Mutable state threaded across the ordered lowering phases in :func:`lower`.

    Each ``_lp_*`` phase reads and writes fields here. The finalised
    side-tables (``local_dtypes`` / ``zeros_locals`` / ``zeros_fills`` /
    ``reassign_shapes`` / ``int_locals`` / ``scalar_call_temps``) are written
    straight onto :attr:`kir` -- typed :class:`KernelIR` fields the emitter reads
    directly, not attributes monkey-patched onto ``tree.__dict__``.
    """

    __slots__ = (
        "arrays_shapes",
        "blas",
        "bool_names",
        "dim_aliases",
        "fft_library",
        "fft_library_nd",
        "inl_defs",
        "iter_rewriter",
        "kir",
        "lib_rewriter",
        "lib_shape_table",
        "local_dtypes",
        "native_call",
        "original_kir",
        "param_seed",
        "resolve_inl_table",
        "scalar_temps",
        "shapes",
        "sibling_scalar_helpers",
        "tree",
        "wa_rewriter",
        "zeros",
        "zeros_locals",
    )

    def __init__(self, original_kir: KernelIR, lowered: KernelIR) -> None:
        #: The un-lowered input IR -- source of ``.sparse`` and ``.helpers``.
        self.original_kir = original_kir
        #: Target's "I render this numpy call myself" predicate; see :func:`lower`.
        self.native_call: (
            Callable[[tuple[str, str], ast.Call, dict[str, tuple[str, ...]], dict[str, str]], bool] | None
        ) = None
        #: Target renders a dense 2-D float GEMM as a BLAS call; see :func:`lower`.
        self.blas: bool = False
        #: Target renders a whole-array 1-D np.fft.* as an FFT_LIBRARY_MARKER call; see :func:`lower`.
        self.fft_library: bool = False
        #: Target renders a batched / N-D np.fft.* as FFTN_LIBRARY_MARKER; see :func:`lower`.
        self.fft_library_nd: bool = False
        #: By-value scalar helpers this IR can call but does not itself list -- a HELPER body's
        #: own IR carries no helper list, so its siblings are handed down by :func:`lower`.
        self.sibling_scalar_helpers: set[str] = set()
        #: The working (lowered) IR -- what :func:`lower` returns.
        self.kir = lowered
        #: Shortcut to the function-body AST every pass rewrites in place.
        self.tree = lowered.tree
        #: Boolean-array names, harvested HERE because the producer of a mask
        #: (``I = np.less(...)``) is itself lowered to an explicit loop by an
        #: earlier phase than the mask consumers -- collecting later sees only
        #: ``I[i] = ...`` and cannot prove ``I`` boolean (mandelbrot1).
        self.bool_names: set[str] = collect_bool_names(lowered.tree, lowered.arrays)
        # Shape / dtype tables built up across phases and consumed downstream.
        self.arrays_shapes: dict[str, list[str]] = {}
        self.lib_shape_table: dict[str, object] = {}
        self.local_dtypes: dict[str, str] = {}
        self.zeros_locals: dict[str, tuple[str, ...]] = {}
        self.shapes: dict[str, list[str]] = {}
        self.scalar_temps: dict[str, tuple[str, ...]] = {}
        self.inl_defs: dict[str, object] = {}
        #: Dimension local -> its definition (``channels`` -> ``embed_dim``), for the matmul
        #: hoister's token comparison. See :func:`collect_dim_aliases`.
        self.dim_aliases: dict[str, str] = {}
        self.param_seed: dict[str, tuple[str, ...]] = {}
        #: Bound ``resolve_inl_table_`` closure, set in the resolve-inl phase and
        #: re-used by the slice-normalise phase (both resolve ``__inl`` tokens).
        self.resolve_inl_table: Callable[[dict], None] | None = None
        # Rewriter handles whose post-visit state a later phase consumes.
        self.iter_rewriter: DesugarArrayIteration | None = None
        self.wa_rewriter: WholeArrayAssignRewriter | None = None
        self.lib_rewriter: object = None
        self.zeros: ZerosRewriter | None = None


def lp_seed_shape_table(ctx: LoweringContext) -> None:
    """Seed the array-shape table, then resolve shape-mid-expressions / ellipses.

    Shape-mid-expression first -- legacy kernels use ``A.shape[0]`` inside loops /
    array constructors; everything downstream is easier if those are resolved to
    bare symbol names.
    """
    lowered = ctx.kir
    ctx.arrays_shapes = {a.name: list(a.shape) for a in lowered.arrays}
    # Sparse arrays carry CSR/etc. buffers, not a dense ArrayDesc, so they
    # are absent from ``lowered.arrays`` -- but the body still reads
    # ``A.shape[i]`` (cg/bicgstab/minres' ``n = A.shape[0]``). Seed the
    # shape table from each sparse desc's logical_shape so the resolver
    # maps ``A.shape[0]`` -> the logical dim symbol.
    for sname, sd in (lowered.sparse or {}).items():
        if sd.logical_shape:
            ctx.arrays_shapes.setdefault(sname, list(sd.logical_shape))
    ShapeMidExpressionRewriter(ctx.arrays_shapes).visit(ctx.tree)
    # Index-access normalisation (chained-flatten / ellipsis-expand / trailing-slice
    # pad) is deferred to the single ``normalize-index-access`` phase, which runs
    # AFTER the harvest/inlined-shape resolution so post-inline locals' ranks are
    # known (breaking the harvest<->ellipsis circular dependency for ls3df_scf).


def lp_normalize_calls(ctx: LoweringContext) -> None:
    """Canonicalise numpy call / method forms (alloc, matmul, transpose, math,
    casts, iteration, tuple-unpack) into the loop-lowerable subset."""
    tree = ctx.tree
    ash = ctx.arrays_shapes
    NpAliasRewriter().visit(tree)
    # Before anything reads an operand LIST: a constant-range comprehension is the list, spelled
    # as a loop (np.concatenate's im2col taps), and no later pass can see through one.
    UnrollConstRangeComprehension().visit(tree)
    MembershipToComparisons().visit(tree)
    # ``X = alloc(...) if cond else None`` -> ``X = alloc(...)`` before the zeros
    # harvester runs, so the conditionally-allocated buffer is seen as a plain local
    # (the backends have no ``None``; reads are guarded by the same ``cond``).
    ConditionalNoneAllocRewriter().visit(tree)
    # A nested ``np.full`` (the causal mask's ``np.triu(np.full(..., -inf), 1)``) is spilled
    # to a temp first, so the direct-assign rewriter below sees it.
    FullCallHoister().visit(tree)
    FullLikeRewriter().visit(tree)
    # ``np.eye`` / ``np.identity`` -> zeros + diagonal fill, BEFORE the zeros
    # harvest so the resulting ``np.zeros((n, n))`` is picked up normally. A nested
    # ``... + 1e-12 * np.eye(k)`` (LS3DF's RR jitter) is spilled to a temp first so
    # the direct-assign diagonal rewriter sees it.
    EyeCallHoister().visit(tree)
    EyeToZerosDiagonal().visit(tree)
    # Same reason, same place: a copy must DECLARE its buffer, not just share a shape token.
    CopyToAllocAndFill().visit(tree)
    MatmulCallRewriter().visit(tree)
    # ``np.<op>.at`` scatter lowering runs later (see ``lp_scatter_at``), once
    # ``np.arange``/reduction/einsum local temps it may need to size (vexx_k's
    # ``ikb``, icon_scatter's ``lev``) have been materialised by the LibNode
    # expander -- run too early, its index/value shapes are simply unknown.
    TransposeRewriter(set(ctx.original_kir.sparse or {})).visit(tree)
    # Local arrays too, not just declared parameters: fv3_dycore's y-stage reads
    # ``.astype(q_advected_x.dtype)`` off an intermediate, and an unresolved dtype drops the cast,
    # which leaves Fortran multiplying a REAL by the LOGICAL mask.
    AstypeRewriter(
        {
            **{k: v for k, v in ctx.local_dtypes.items() if v},
            **{a.name: a.dtype for a in ctx.kir.arrays if a.dtype},
        },
        default_float=next((a.dtype for a in ctx.kir.arrays if a.dtype and a.dtype.startswith("float")), ""),
    ).visit(tree)
    MethodCallRewriter().visit(tree)
    # A Call in subscript-index position (``v[np.argmax(np.abs(v))]``) is hoisted
    # to a fresh temp so the index is a bare Name the backends emit; the spilled
    # ``__ix = np.argmax(...)`` is expanded by the later LibNode reduction pass.
    ComputedIndexCallHoister().visit(tree)

    def leading_extent(array: str) -> ast.expr | None:
        shape = ash.get(array)
        return const_or_name(shape[0]) if shape else None

    ctx.iter_rewriter = DesugarArrayIteration(leading_extent, lambda target, ordinal: f"__ai{ordinal + 1}")
    ctx.iter_rewriter.visit(tree)
    EnumerateZipRewriter(leading_extent).visit(tree)
    BuiltinCastRewriter().visit(tree)
    # The LOCAL array shapes too, exactly as the two later MathRewriter sites do. With only
    # the declared arrays, an inlined helper's temps look like scalars, and np.maximum on two
    # of them took the scalar rename: __npb_fmax(double *, double *).
    MathRewriter(set(ash.keys()) | set(ctx.lib_shape_table.keys()), defer_array_capable=True).visit(tree)
    DaceMapRewriter().visit(tree)
    # ``a = b = v``: ``v`` once -- a temp for a scalar, one name for an array's one buffer.
    chain_ranks = {name: len(shape) for name, shape in ash.items()}
    chain_ranks.update((desc.name, 0) for desc in (*ctx.kir.scalars, *ctx.kir.symbols))
    SplitChainedAssign(lambda ordinal: f"__chain{ordinal}", seed_ranks=chain_ranks).visit(tree)
    tuple_rewriter = ShapeTableTupleSplit(ash)
    tuple_rewriter.visit(tree)
    # Stash the int-locals so the emitter can declare them.
    ctx.kir.int_locals = tuple_rewriter.int_locals
    # Last: delete the ``X = X`` statements the shape rewrites above leave behind, while
    # the arrays / shape tables that say which names must survive are still exact. Runs
    # before promote-params, which is indifferent to them either way.
    SelfAssignDropper(set(ctx.arrays_shapes) | set(ctx.kir.reassign_shapes)).visit(tree)


def lp_promote_params(ctx: LoweringContext) -> None:
    """Promote shape symbols / free names to params and flag output+index arrays.

    After tuple-unpack expansion, the kernel body may reference shape symbols that
    the JSON's input_args did not declare (a numpy kernel commonly reads
    ``n, k = a.shape`` then iterates over ``range(n)``). Promote those symbols to
    first-class kernel parameters so the emitted C signature carries them.
    """
    lowered = ctx.kir
    fold_shape_aliases(lowered)
    promote_shape_symbols_to_params(lowered)
    # Anything still referenced in the body that isn't a declared parameter,
    # builtin, or assigned local becomes an ``int`` parameter (symbolic strides /
    # chunk sizes in TSVC-2.5 kernels).
    promote_free_names_to_params(lowered)
    # Body-driven: detect writes (force is_output) and index-array usage (force
    # int64 dtype) so the emitter picks the right pointer qualifier / element type.
    # The helpers come along because a write inside one is still a write to the
    # CALLER's buffer -- see :func:`written_through_helpers`.
    detect_output_and_index_arrays(lowered, ctx.original_kir.helpers)


def lp_pre_libnode_normalize(ctx: LoweringContext) -> None:
    """Pre-LibNode normalisation: mgrid, ``x.shape =`` reshape, tuple-subscript
    fold, boolean-mask reduction fusion. Also seeds the LibNode shape table."""
    tree = ctx.tree
    ctx.lib_shape_table = dict(ctx.arrays_shapes)
    # Pre-pass: collapse chained subscripts ``A[i][j]`` -> ``A[i, j]`` (vexx_k's
    # ``tabxx_qr[ia][:, ijtoh[ih, jh]]`` / ``becxx[:, jbnd, ikq][ikb]``) so the
    # harvest, scalarizers and fancy-scatter store all see a single-level access.
    ChainedSubscriptFlattener(ctx.arrays_shapes, bool_names=ctx.bool_names, explicit_trailing_axes=True).visit(tree)
    ast.fix_missing_locations(tree)
    # Pre-pass: lower ``Xi, Yi = np.mgrid[a:b, c:d]`` tuple-unpack assignments to a
    # pair of per-element init loops -- before the main harvest so the resulting
    # fresh arrays get their shape registered like any other local.
    MgridLowering().visit(tree)
    ast.fix_missing_locations(tree)
    # Pre-pass: rewrite ``x.shape = expr`` -> ``x = np.reshape(x, expr)``. Handles
    # chained ``Xi.shape = Yi.shape = expr`` too. Mandelbrot2 canonical uses this.
    ShapeAttrToReshape().visit(tree)
    ast.fix_missing_locations(tree)
    # Forward-substitute a shape-tuple local (``shp = (Lb, Lb, Lb, nstate)`` --
    # the seed-time fold of a declared-array-based ``shp = Y.shape``) into its uses
    # BEFORE the harvest, so ``np.reshape(x, shp)`` is sized from the concrete tuple
    # (not a spurious 1-D ``(shp,)``) when the harvest records the reshape target.
    TupleLocalPropagator().run(tree)
    ast.fix_missing_locations(tree)
    # Fold ``(a, b, c)[K]`` Tuple subscripts -- comes from ``arr.shape[-2]`` when
    # the shape is a tuple literal.
    TupleSubscriptFolder().visit(tree)
    ast.fix_missing_locations(tree)
    # Peephole: fuse ``tmp = arr[mask]; X = np.<reduction>(tmp)`` into a single
    # masked iteration so we avoid materialising the dynamic-length compacted view
    # from boolean fancy indexing. Seeded with the kernel-array shapes so the loop
    # bound is the right symbol.
    BooleanMaskReductionRewriter(ctx.arrays_shapes, ctx.bool_names).visit(tree)
    ast.fix_missing_locations(tree)


def scalar_return_helpers(ctx: "LoweringContext") -> set[str]:
    """Names of the by-value SCALAR helpers reachable from the body being lowered."""
    own = {h.kernel_name for h in ctx.original_kir.helpers if h.return_kind == "scalar"}
    return own | ctx.sibling_scalar_helpers


def lp_seed_dtypes_and_harvest(ctx: LoweringContext) -> None:
    """Seed local dtypes (signature + boolean constructors), unify mixed-complex
    selects, SSA-rename reassigned locals, then harvest local-array shapes."""
    tree = ctx.tree
    # Seed with signature-array dtypes so downstream passes (call hoister
    # infer_complex, BinOp dtype propagation, emit-time decl) consistently treat
    # declared inputs/outputs the same way as locals. Every array goes in -- the
    # table is keyed by name so there is no cost to a uniform copy of all dtypes.
    ctx.local_dtypes = {}
    # Bind the finalised table onto the IR now; the phases below mutate it in
    # place, so the emitter reads the fully-populated dict after ``lower`` returns.
    ctx.kir.local_dtypes = ctx.local_dtypes
    for arr in ctx.kir.arrays:
        if arr.dtype:
            ctx.local_dtypes[arr.name] = arr.dtype
    # Seed boolean-typed locals from explicit ``np.zeros/empty/ones(..,
    # dtype=np.bool_)`` constructors (ICON cfl_clip / levmask) so a derived
    # ``mask = cfl_clip & owner`` is recognised as boolean (and declared bool /
    # logical) before the whole-array rewriter runs.
    for s_ in ast.walk(tree):
        if (
            isinstance(s_, ast.Assign)
            and len(s_.targets) == 1
            and isinstance(s_.targets[0], ast.Name)
            and isinstance(s_.value, ast.Call)
        ):
            for kw_ in s_.value.keywords:
                if kw_.arg != "dtype":
                    continue
                dv_ = kw_.value
                if (isinstance(dv_, ast.Attribute) and dv_.attr in ("bool_", "bool")) or (
                    isinstance(dv_, ast.Name) and dv_.id == "bool"
                ):
                    ctx.local_dtypes.setdefault(s_.targets[0].id, "bool_")
    # A local the mask harvest already PROVED boolean is declared boolean too. Left at the float
    # default, velocity_tendencies' ``lvl_active = levelmask[band] | levelmask[band_next]`` emitted
    # a bitwise-or on two doubles, which is not a C operation at all.
    for bool_name in ctx.bool_names:
        ctx.local_dtypes.setdefault(bool_name, "bool_")
    # Seed the complex work-array temps (and their directly-derived scalar reads)
    # that the eigh / eigvalsh cyclic-Jacobi lowering allocates from a complex
    # signature array's ``.dtype`` -- BEFORE the true-division and libnode-expand
    # phases, which otherwise consume those still-untyped temps and mis-lower the
    # complex divide (``apq / m``) and the ``np.conj`` on the reduction matrices.
    seed_complex_work_dtypes(tree, ctx.local_dtypes, {a.name: a.dtype for a in ctx.kir.arrays})
    # SSA-style rename for Names reassigned with different broadcast extents (hdiff
    # / vadv ``res = ...; res = ...`` with two distinct shapes). Runs BEFORE harvest
    # so each version registers under its own name and downstream passes (harvest /
    # LibNodeRewriter / lifter) see unambiguous shapes per local.
    ssa_rename_reassigned(tree, ctx.arrays_shapes)
    harvest_local_shapes(tree, ctx.lib_shape_table, ctx.local_dtypes, scalar_return_helpers(ctx))
    # Unify a mixed real/complex conditional's branches (``d = z.real if flag else
    # z``) so Fortran ``merge`` (strict same-type) and the JIT type unifiers see a
    # uniform complex select instead of a real-vs-complex pair (QE vexx gamma_only
    # path). Runs AFTER the harvest so a complex LOCAL branch (``deexx`` typed from
    # its ``np.zeros(.., complex128)`` constructor) is already known complex.
    PromoteMixedComplexIfExp(ctx.local_dtypes).visit(tree)


def lp_resolve_inlined_shapes(ctx: LoweringContext) -> None:
    """Resolve inlined-scalar dim tokens in the harvest table, inherit loop-var
    dtypes, and pre-lift ``alpha * A`` so the matmul hoister sees a bare Name."""
    tree = ctx.tree
    # Inlined-helper locals (conv2d's ``__inl1_output``) get their shape from
    # ``__inl<k>_`` scalar-dim locals (``__inl1_N`` ...) that are *assigned later
    # in the body* -- so an allocation sized from them at function top reads garbage,
    # and the tokens never bind. Build a resolver that substitutes each ``__inl<k>_``
    # dim-local away (fixpoint) and concretises the resulting ``param.shape[i]``
    # against the real param shapes; applied later to the declaration / malloc sink
    # (``zeros_locals`` / ``shapes``).
    ctx.inl_defs = collect_inlined_scalar_defs(tree)
    ctx.param_seed = {n: tuple(s) for n, s in ctx.arrays_shapes.items()}
    ctx.dim_aliases = collect_dim_aliases(tree, set(ctx.arrays_shapes) | set(ctx.lib_shape_table))

    def resolve_inl(shape):
        """Substitute ``__inl<k>_`` dim-locals away then resolve
        ``param.shape[i]`` -> a pure-param shape tuple.

        Best-effort and *never-worse*: a token is only rewritten when the
        result fully resolves to real parameters. If expansion would leave
        a residual ``__inl`` name or a ``.shape`` on a non-parameter local
        (the chained-inline case -- ``__inl3_N = x__v1.shape[0]`` where
        ``x__v1`` is itself a local), the ORIGINAL token is kept so the
        downstream source-order ``ResolveArrShape`` pass handles it."""
        if not ctx.inl_defs:
            return tuple(shape)
        subbed = substitute_inlined_scalar_defs(tuple(shape), ctx.inl_defs)
        resolved = resolve_shape_attr_tokens(subbed, ctx.param_seed)
        return tuple(new if not INL_RE.search(new) else str(orig) for orig, new in zip(shape, resolved))

    def resolve_inl_table_(table) -> None:
        for nm in list(table):
            table[nm] = list(resolve_inl(table[nm])) if isinstance(table[nm], list) else resolve_inl(table[nm])

    ctx.resolve_inl_table = resolve_inl_table_
    # Resolve the harvest table now so the passes that consume it (notably
    # ``WholeArrayAssignRewriter``, which decides whether ``__hcall1 + bias``
    # is a same-rank broadcast to expand into a loop vs. a raw pointer add)
    # see ``__inl1_output``'s real shape. The never-worse guard leaves the
    # chained-inline locals (whose dims reference another local's ``.shape``)
    # untouched for the source-order ``ResolveArrShape`` pass downstream.
    if ctx.inl_defs:
        resolve_inl_table_(ctx.lib_shape_table)
    # Loop-var dtype inheritance: ``for b in data:`` (where ``data`` is
    # uint8) declares ``b`` as the element dtype of ``data``.
    array_dtypes_by_name = {a.name: a.dtype for a in ctx.kir.arrays}
    for loop_var, source_arr in ctx.iter_rewriter.var_to_array.items():
        src_dt = array_dtypes_by_name.get(source_arr) or ctx.local_dtypes.get(source_arr)
        if src_dt is not None:
            ctx.local_dtypes[loop_var] = src_dt
    # Pre-matmul: lift ``alpha * A`` -> temp so the matmul hoister sees a bare Name
    # on the left of ``A @ B``. The pre-lift runs per Assign.
    ctx.scalar_temps = {}
    scalar_counter = [0]
    for stmt in list(tree.body):
        if isinstance(stmt, (ast.Assign, ast.AugAssign)):
            sm = ScalarTimesMatmulRewriter(ctx.lib_shape_table, ctx.scalar_temps, scalar_counter)
            stmt.value = sm.visit(stmt.value)
            # Prepend pre_stmts before the original statement.
            if sm.pre_stmts:
                idx = tree.body.index(stmt)
                tree.body[idx:idx] = sm.pre_stmts


def lp_normalize_index_access(ctx: LoweringContext) -> None:
    """Consolidated index-access normalisation, run once after every array shape is
    known (post ``resolve-inlined-shapes``). Three rewrites, in order:

    1. :class:`ChainedSubscriptFlattener` -- flatten a chained subscript
       ``A[f][..., 0]`` -> ``A[f, ..., 0]`` so the base is always a Name.
    2. :class:`EllipsisExpander` -- replace ``...`` with the explicit full slices
       its array's rank implies (``a[..., 0]`` on a 3-D array -> ``a[:, :, 0]``).
    3. :class:`PadImplicitTrailingSlices` -- make numpy's implicit trailing full
       slices explicit (``A[i, j]`` on a 3-D array -> ``A[i, j, :]``).

    Runs after the harvest / inlined-shape resolution to break the
    harvest<->ellipsis circular dependency: post-inline locals (``hx``, ``vloc``,
    ``psi_frag[f][..., 0]``) only get a shape at the harvest, which the ellipsis /
    trailing-slice rewrites need for each array's rank. Shape source is
    :attr:`ctx.lib_shape_table` (harvest + inlined-resolve table, covering both
    signature arrays and derived locals) -- not the raw :attr:`ctx.arrays_shapes`,
    which at this point holds only the declared arrays."""
    tree = ctx.tree
    shapes = ctx.lib_shape_table
    ChainedSubscriptFlattener(shapes, bool_names=ctx.bool_names).visit(tree)
    EllipsisExpander(shapes).visit(tree)
    PadImplicitTrailingSlices(shapes).visit(tree)
    # Re-fold ``<array-expr>.shape`` / ``.shape[k]`` now that every post-inline
    # local's shape is harvested: the seed-time pass only had the DECLARED-array
    # shapes, so a ``.shape`` read on an inlined local (``v[..., None].shape[-1]``
    # in ``_hpsi``'s ``X.reshape(-1, X.shape[-1])``, or the eigh helper's
    # ``a.shape[0]``) could not resolve then. With the full table the newaxis /
    # subscript base folds to concrete dims BEFORE the reshape / LibNode expander
    # bakes the (otherwise unresolved) token into a loop bound.
    ShapeMidExpressionRewriter(shapes).visit(tree)
    # ...and fold the same reads inside the table's own tokens, so the table agrees with the
    # body it describes. A shape an expander reads as a CONSTANT (squeeze's unit axis) is only
    # a constant once this runs.
    fold_shape_reads_in_table(shapes)
    # Re-run the tuple splitter for the same reason the fold above re-runs: its first pass
    # (normalize-calls) only had the DECLARED-array shapes, so ``n, c, oh, ow = x.shape`` on an
    # inlined local stayed a tuple and reached the emitter as a value -- "expression Tuple", the
    # single largest emit failure in the corpus. Extends int_locals rather than replacing it; the
    # first pass's names are still live.
    tuple_rewriter = ShapeTableTupleSplit(shapes)
    tuple_rewriter.visit(tree)
    ctx.kir.int_locals += [n for n in tuple_rewriter.int_locals if n not in ctx.kir.int_locals]
    TupleLocalPropagator().run(tree)
    TupleSubscriptFolder().visit(tree)
    ast.fix_missing_locations(tree)
    # Re-resolve inlined-scalar dims now that the folds above concretised the
    # reshape-shape locals (``__inl5_k = shp[-1]`` -> ``= nstate``). The earlier
    # resolve-inlined-shapes phase ran BEFORE that fold, so a size local defined only
    # in a LATER inlined solve (``w`` / ``X`` reassigned across the two Rayleigh-Ritz
    # inlines) kept its ``__inl<k>_`` token and drove a use-before-def allocation
    # (garbage size). Recollect and reapply against the now-fuller table.
    ctx.inl_defs = collect_inlined_scalar_defs(tree)
    if ctx.inl_defs and ctx.resolve_inl_table is not None:
        ctx.resolve_inl_table(ctx.lib_shape_table)


def lp_libnode_expand(ctx: LoweringContext) -> None:
    """FFT-grid + reshape normalisation, then the LibNode expander (reductions /
    matmul / linalg); a second free-name promotion for structural scalars."""
    tree = ctx.tree
    # Flat-grid FFT idiom (vexx invfft/fwfft: reshape-to-grid -> fftn over the
    # grid axes -> reshape-back) -> materialised reshape + fft temps, so the
    # reshape/DFT expanders below see bare Names with known shapes.
    FftGridReshapeRewriter(ctx.lib_shape_table, ctx.local_dtypes, [0]).visit(tree)
    ast.fix_missing_locations(tree)
    # Normalize ``X.reshape(a, b)`` method form to ``np.reshape(X, (a, b))``
    # AFTER the FFT idiom match (which consumes its own reshape chains) so the
    # single expand_reshape path serves lulesh's varargs spelling.
    ReshapeMethodRewriter().visit(tree)
    # ... and the reduction methods, for the same reason: one function-form path per op instead of
    # a receiver-shape special case in every expander.
    ArrayMethodRewriter(set(ctx.original_kir.sparse or {})).visit(tree)
    ast.fix_missing_locations(tree)
    # Seed the INTEGER/UINT element dtype of every kernel-parameter array into
    # ``local_dtypes`` so the library-node hoister can see, e.g., that ``idx`` in
    # ``int(np.max(idx))`` is int64 and tag the max-reduction temp int64 rather
    # than the float default. Without this a value-preserving reduction over an
    # int array declares a real accumulator, and the Fortran emit's ``merge(int,
    # real)`` update is a kind mismatch (gfortran rejects it). Only int/uint tags
    # are seeded: complex is handled by the dedicated complex-propagation pass, and
    # a float tag would flip untagged-float-default code paths. Param arrays are
    # declared from the ABI signature (not ``local_dtypes``), so tagging them here
    # never redeclares them.
    for arr in ctx.kir.arrays:
        if dtypes.is_integer(arr.dtype):
            ctx.local_dtypes.setdefault(arr.name, arr.dtype)
    # Library-node expansion -- reductions, matmul, etc. -- runs before
    # ``ZerosRewriter`` so any matmul temps the rewriter introduces are picked up
    # by the zeros pass as local arrays.
    ctx.lib_rewriter = LibNodeRewriter(
        ctx.lib_shape_table,
        scalar_helpers=scalar_return_helpers(ctx),
        known_arrays=set(ctx.arrays_shapes.keys()),
        local_dtypes=ctx.local_dtypes,
        sparse=ctx.original_kir.sparse,
        dim_aliases=ctx.dim_aliases,
        native_call=ctx.native_call,
        native_dtypes={**{arr.name: arr.dtype for arr in ctx.kir.arrays}, **ctx.local_dtypes},
        blas=ctx.blas,
        fft_library=ctx.fft_library,
        fft_library_nd=ctx.fft_library_nd,
    )
    ctx.lib_rewriter.visit(tree)
    # Second math rename: an intrinsic whose argument only becomes a SCALAR once the library
    # nodes expand. ``np.sqrt(w @ (cov @ w))`` (portfolio_optimization) defers the rename in
    # ``lp_normalize_calls`` -- the arg reads arrays, so the elementwise expander gets first
    # refusal -- but a 1-D x 1-D matmul lowers to a scalar dot temp, leaving ``np.sqrt(__mm2)``
    # that no later pass renames and the backends reject as an unsupported call. Re-running the
    # SAME rewriter (rather than special-casing the emitter) with the temps now known keeps the
    # rule intact: an argument that is still array-valued -- a vector matmul temp, which IS in
    # ``lib_shape_table`` -- keeps deferring; a scalar one renames to the math intrinsic.
    MathRewriter(set(ctx.arrays_shapes.keys()) | set(ctx.lib_shape_table.keys())).visit(tree)
    # Second free-name promotion: sparse matvec dispatchers introduce structural
    # scalar symbols that don't exist until the hoister runs (SELL-C-sigma's slice
    # height ``C``), so the first promotion at the top of lower() can't see them.
    # Re-run now -- matmul temps and the JDS scratch are subscript-store targets, so
    # they're treated as locals and excluded; only genuinely free Load names get
    # promoted.
    promote_free_names_to_params(ctx.kir)
    fix_real_scalar_dtypes(ctx)


def fix_real_scalar_dtypes(ctx: LoweringContext) -> None:
    """Re-derive the dtype of a local (scalar or array) that LibNodeRewriter
    tagged complex from a complex source but whose value is actually real.

    LibNodeRewriter propagates a complex source's element type onto every
    derived local, but ``.real``/``.imag``, ``abs``/``hypot``, or a
    ``np.float64(...)`` cast produce a real value (eigh/eigvalsh cyclic-Jacobi's
    ``app = A[p, p].real`` / ``m = hypot(...)`` chain; LS3DF GENPOT's ``v =
    ifftn(v_g).real + ...`` and its ``.mean()`` temp). Left complex, the emitter
    declares them complex and a comparison, ``conjg(<real>)``, or a real
    narrowing store fails to compile under C++ (C silently drops the imaginary
    part). Reuse :func:`walk_complex` (already classifies these forms real);
    retag to the matching real width when it disagrees with a complex tag.

    Iterates to a fixpoint: the cascade is transitive (``tau`` is real only once
    ``app``/``aqq``/``m`` are; GENPOT's ``v`` only once its ``.real`` result is).
    An accumulator's self-reference (``acc = acc + real[...]``) is neutral and
    resolves real; a genuinely complex injected value keeps it complex. Kernel
    parameter dtypes (outside ``local_dtypes``) still resolve names on a
    candidate's RHS, so a complex input is never misread as real. Finally drops
    stale conjugation on a now-real operand (:class:`RealConjDropper`). Same
    class of fix as ``abs(complex) -> double``: a real-returning op on a
    complex operand is real."""
    tree = ctx.tree
    ld = ctx.local_dtypes
    # Kernel-parameter array dtypes live outside ``local_dtypes``; resolve names
    # through both so a complex INPUT read on a candidate's RHS is seen as complex
    # (else the walk would call it real and unsoundly narrow the candidate).
    array_dtypes = {a.name: a.dtype for a in ctx.kir.arrays}

    def name_dtype(nm: str) -> str | None:
        dt = ld.get(nm)
        return dt if dt is not None else array_dtypes.get(nm)

    # Every complex-tagged LOCAL of known real width is a candidate -- scalars and
    # local temp arrays alike (an array's ``lib_shape_table`` shape is orthogonal
    # to its element dtype). Kernel input/output arrays are excluded: narrowing
    # their declaration would break the marshalled ABI.
    candidates = {n for n, dt in ld.items() if dt in REAL_FOR_COMPLEX and n not in array_dtypes}
    # A name FFT_LIBRARY_MARKER writes is excluded too: the marker call is a bare
    # ``Expr`` (``__fft_1d_library(out, src, n, inverse, norm)``), invisible to the
    # ``writes`` walk below (it only looks at Assign/AugAssign targets). Its output
    # temp's OWN init marker (``__cb1 = __hpcagent_bench_zeros__()``) then looks like
    # its only, real-valued write and gets narrowed back to real -- undoing the
    # unconditional complex128 tag the hoister gave it (lib_nodes.call_hoist.CallHoister,
    # every np.fft.* transform returns complex regardless of operand). Every 1-D FFT
    # library call's output is complex by construction, never a narrowing candidate.
    candidates -= {
        call.args[0].id
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id in (FFT_LIBRARY_MARKER, FFTN_LIBRARY_MARKER)
        and call.args
        and isinstance(call.args[0], ast.Name)
    }
    # Every value WRITTEN to a candidate -- a whole-name ``x = e`` / ``x += e`` or
    # a per-element ``x[i] = e`` / ``x[i] += e`` (an array is written elementwise).
    # A candidate is real only if EVERY write is real.
    writes: dict[str, list[ast.expr]] = {}
    for stmt in ast.walk(tree):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            tgt: ast.AST = stmt.targets[0]
        elif isinstance(stmt, ast.AugAssign):
            tgt = stmt.target
        else:
            continue
        base = (
            tgt.id
            if isinstance(tgt, ast.Name)
            else tgt.value.id
            if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name)
            else None
        )
        if base in candidates:
            writes.setdefault(base, []).append(stmt.value)

    def all_writes_real(name: str, vals: list[ast.expr]) -> bool:
        # The candidate's own self-reference is neutral (resolved real): it carries
        # the running accumulator value, so a mean / sum of a real array is real,
        # yet every OTHER operand must resolve real for the write to be real.
        def resolve(nm: str) -> str | None:
            return None if nm == name else name_dtype(nm)

        return all(walk_complex(v, resolve) is None for v in vals)

    changed = True
    while changed and candidates:
        changed = False
        for name in list(candidates):
            vals = writes.get(name)
            # Fixpoint: ``tau = (aqq - app) / (2 * m)`` -- or the GENPOT ``v`` array
            # and its ``v.mean()`` temp -- read real only once the locals they
            # derive from have themselves been retagged real (this pass).
            if vals and all_writes_real(name, vals):
                ld[name] = REAL_FOR_COMPLEX[ld[name]]
                candidates.discard(name)
                changed = True
    RealConjDropper(ld).visit(tree)
    # A library call in subscript-base position is materialised into a temp by THIS phase, so
    # ``np.transpose(d, perm)[..., None]`` still carried its Ellipsis through
    # ``normalize-index-access`` -- that pass only fires on a Name base. The temp now has a
    # harvested shape, so the rank the expansion needs is finally known.
    EllipsisExpander(ctx.lib_shape_table).visit(tree)
    ast.fix_missing_locations(tree)


def lp_scatter_at(ctx: LoweringContext) -> None:
    """Lower ``np.<op>.at(target, idx, vals)`` unbuffered scatters into explicit
    indexed loops (:class:`ScatterAtRewriter`).

    Deliberately runs AFTER ``lp_libnode_expand``: an idx/value expression is
    often a LOCAL temp built from a reduction/einsum/``np.arange`` (vexx_k's
    ``ikb = ofsbeta[:, None] + np.arange(nh)[None, :]``, icon_scatter's ``lev =
    np.arange(nlev)[None, :, None, None]``), whose shape only lands in
    ``ctx.lib_shape_table`` once the harvest and LibNode expander have run, and
    whose ``np.arange`` must already be a materialised array (not the raw call)
    for the SAME index-array Name path the gather side uses. Running any
    earlier (e.g. in the C/Fortran ABI-normalisation phase) leaves every
    non-parameter idx/value unresolvable."""
    # ``name = <base>.reshape(-1)`` / ``name = np.broadcast_to(base, shape)``
    # locals read bare inside ``.at()`` (icon_scatter's ``vals``, read twice) --
    # collect them so ``ScatterAtRewriter`` can look through the alias to the
    # wrapped operand exactly as it does for a wrapper call spelled inline.
    wrapper_defs: dict[str, ast.expr] = {}
    for stmt in ast.walk(ctx.tree):
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and ScatterAtRewriter.unwrap_wrapper_call(stmt.value) is not None
        ):
            wrapper_defs[stmt.targets[0].id] = stmt.value
    ScatterAtRewriter(ctx.lib_shape_table, ctx.bool_names, wrapper_defs).visit(ctx.tree)
    ast.fix_missing_locations(ctx.tree)
    # Every ``.at()`` use of a wrapper-defined name was just replaced by its
    # peeled (unwrapped) operand -- if that was the name's ONLY use, its
    # ``np.broadcast_to``/``.reshape(-1)`` definition is now dead code the
    # emitter has no lowering for (nothing left reads the wrapped result).
    # Drop it rather than leave an orphaned unsupported call.
    still_read = {n.id for n in ast.walk(ctx.tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    dead = set(wrapper_defs) - still_read
    if dead:

        class DropDeadWrapperAssign(ast.NodeTransformer):
            def visit_Assign(self, node: ast.Assign) -> ast.Assign | None:
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in dead:
                    return None
                return node

        DropDeadWrapperAssign().visit(ctx.tree)


def lp_whole_array_and_zeros(ctx: LoweringContext) -> None:
    """Whole-array assignment expansion, the zeros harvest, and the merged
    local-array declaration tables (``zeros_locals`` / ``zeros_fills`` /
    ``scalar_call_temps`` / ``shapes``)."""
    tree = ctx.tree
    # Whole-array Augmented / plain assignment between same-shape arrays (``x1 +=
    # temp`` or ``out = a``) is numpy's elementwise form; expand to a loop nest so
    # the C/Fortran emitter does not see pointer arithmetic. Pass the set of REAL
    # kernel arrays (not aliases that LibNodeRewriter added) so the whole-array
    # rewriter knows which aliases are fresh locals needing declaration.
    real_arrays = set(ctx.arrays_shapes.keys())
    # Boolean masking: ``arr[mask_expr] = value`` -> per-element loop with an ``if
    # mask_expr[i]:`` guard. Runs before the whole-array rewriter so the LHS is a
    # plain scalar subscript downstream.
    BooleanMaskRewriter(ctx.lib_shape_table, ctx.bool_names).visit(tree)
    ctx.wa_rewriter = WholeArrayAssignRewriter(
        ctx.lib_shape_table,
        real_arrays,
        local_dtypes=ctx.local_dtypes,
        scalar_defs=collect_inlined_scalar_defs(tree, None),
        scalar_helpers=scalar_return_helpers(ctx),
    )
    ctx.wa_rewriter.visit(tree)
    # Fold the shapes the whole-array pass inferred for genuinely-new locals
    # (meshgrid ``gx``/``gy``/``gz``, the broadcast ``gsq``) back into the shared
    # table, so the zeros harvest and slice-fusion scalarizer below size them --
    # otherwise a shapeless ``gsq`` denominator stays a bare pointer in ``v_g =
    # rho_g / gsq`` and the C division is ``complex / double *``.
    for nm_, shp_ in ctx.wa_rewriter.discovered_shapes.items():
        ctx.lib_shape_table.setdefault(nm_, shp_)
    ctx.zeros = ZerosRewriter(ctx.lib_shape_table)
    ctx.zeros.visit(tree)
    # Merge matmul-hoisted temps with the np.zeros locals -- both become C stack
    # arrays / Fortran locals in the prelude.
    zeros_locals = dict(ctx.zeros.zeros)
    zeros_locals.update(ctx.lib_rewriter.matmul_temps)
    zeros_locals.update(ctx.lib_rewriter.fresh_local_allocs)
    zeros_locals.update(ctx.scalar_temps)
    # setdefault, not update: an alias local is DERIVED (``padded = x``) while the entry already
    # here came from an allocation (``padded = np.zeros((n, c_in, length + 2 * pa))``). Letting the
    # alias win sized conv_standard_1d's zero-padded buffer like the unpadded input -- an
    # out-of-bounds write and wrong numbers at every output position that reads the pad.
    for nm_, shp_ in ctx.wa_rewriter.alias_locals.items():
        zeros_locals.setdefault(nm_, shp_)
    # Pre-pass harvested local arrays (corr = np.eye(M, ...), imgOut = np.copy(...),
    # etc.) that the LibNode expanders didn't rewrite. They must still be declared
    # so the emitter sees them.
    for name, shape in ctx.lib_shape_table.items():
        if name not in zeros_locals and name not in ctx.arrays_shapes:
            zeros_locals[name] = tuple(shape) if shape else ("1",)
    ctx.zeros_locals = zeros_locals
    ctx.kir.zeros_locals = zeros_locals
    # Fill kind per local (zeros / ones / empty / ...). Only the explicit
    # ``np.<ctor>`` path (``ZerosRewriter``) carries a meaningful kind; every other
    # source (matmul temps, slice-fusion lifts, alias locals) is a write-before-read
    # temp, so it defaults to ``empty``. The emitter consults this only when a local
    # name aliases an OUTPUT parameter -- to initialise the caller's buffer correctly
    # without a shadowing declaration.
    ctx.kir.zeros_fills = dict(ctx.zeros.fills)
    tag_complex_locals(ctx.kir, zeros_locals, ctx.zeros.dtype_src, ctx.zeros.dtype_literal)
    # Scalar call-hoist temps: declared as plain double locals by the emit walker
    # via its implicit-local logic (they appear as a bare Name on the LHS of an
    # Assign whose RHS is a Call).
    ctx.kir.scalar_call_temps = list(ctx.lib_rewriter.scalar_call_temps)
    # Re-collect the shape table -- np.zeros locals are included for slice fusion.
    shapes: dict[str, list[str]] = dict(ctx.arrays_shapes)
    for name, shape in ctx.zeros.zeros.items():
        shapes[name] = list(shape) if shape else ["1"]
    # Also include any shapes harvested by the pre-pass (np.eye / np.copy /
    # np.transpose / np.linalg.* etc) that aren't in arrays_shapes / zeros locals --
    # needed when slice fusion encounters omitted-stop slices on such temps.
    for name, shape in ctx.lib_shape_table.items():
        if name not in shapes:
            shapes[name] = list(shape)
    ctx.shapes = shapes


def lp_slice_normalize_and_lift(ctx: LoweringContext) -> None:
    """Normalise subscript forms, lift array-valued slice RHS to fresh locals, and
    re-resolve ``.size`` / ``.shape`` / ``__inl`` tokens over the new locals."""
    tree = ctx.tree
    shapes = ctx.shapes
    # Normalise subscript forms BEFORE the slice lifter so a row/column read
    # ``box = tabxx_box[ia]`` / ``qr = tabxx_qr[ia][:, k]`` (QE ultrasoft
    # augmentation) is materialised into a rank-1 local the fancy scatter / gather
    # can index. Flatten chained subscripts ``B[inner][outer]`` into one combined
    # index, then fold scalar-prefix sub-array aliases (``low = A[i, j]; low[k]`` ->
    # ``A[i, j, k]``, xsbench). Trailing-slice padding is done once, earlier, in the
    # ``normalize-index-access`` phase.
    ChainedSubscriptFlattener(shapes, bool_names=ctx.bool_names, explicit_trailing_axes=True).visit(tree)
    fold_subarray_aliases(tree, shapes)
    # Fold a name bound to a partial/strided VIEW (a real Slice with bounds/step,
    # not just a scalar prefix) into every further-subscripted use, composing the
    # offsets/strides -- grouped conv's ``x_g = padded[:, g*ipg:(g+1)*ipg]`` and
    # sibling machine_learning kernels, whose further-sliced ``x_g[...]`` uses
    # otherwise reach the emitter as a bare ``:`` value expression.
    # A folded-away alias may be a MATERIALISED staging local (the slice lifter's
    # ``__hcall`` copy): its uses now read the base array directly, so leaving it in
    # ``zeros_locals`` emits a malloc/free pair for a buffer nothing writes or reads,
    # hoisted to function top because it has no use to place it against.
    for dead_ in fold_slice_view_aliases(tree, shapes):
        ctx.kir.zeros_locals.pop(dead_, None)
        ctx.kir.zeros_fills.pop(dead_, None)
        shapes.pop(dead_, None)
    # Lift array-valued RHS (slice-bearing BinOp / Call / etc) on a bare-Name LHS to
    # a ``Name = np.zeros(extent); Name[:] = expr`` pair so slice fusion can lower
    # the per-element loop. Computes the shape from the iteration extent of the RHS,
    # registers the new local in both ``shapes`` and ``zeros_locals``.
    lifter = LiftFreshArrayFromSlices(shapes, local_dtypes=ctx.local_dtypes, scalar_helpers=scalar_return_helpers(ctx))
    new_locals = lifter.run(tree)
    if new_locals:
        for name, shape in new_locals.items():
            shapes[name] = list(shape)
            ctx.zeros_locals[name] = tuple(shape)
    # ``zeros_locals`` / ``local_dtypes`` are the same objects the IR already holds
    # (bound in earlier phases), so the lifter's in-place additions -- and the
    # complex dtypes it inferred -- are already visible to the emitter.
    # Re-resolve ``.size`` / ``.shape`` / ``len(..)`` over the NOW-materialised
    # locals (``box = tabxx_box[ia, :]`` -> a rank-1 local). The early pass at
    # parse-shape time saw only params, so ``box.size == 0`` (the QE ultrasoft empty-
    # box guard) survived unresolved; with ``box`` in ``shapes`` it folds to its
    # extent. Already-resolved references are bare Names now, so the other branches
    # are no-ops.
    ShapeMidExpressionRewriter(shapes).visit(tree)
    # ``ZerosRewriter`` re-derives the ``np.empty`` shape straight from the AST
    # tuple, so ``__inl<k>_`` tokens reappear in ``zeros_locals`` / ``shapes`` even
    # after the early ``lib_shape_table`` resolve. These are the declaration / malloc
    # / subscript-stride sink the emitter reads -- resolve them to pure params here
    # so the top-of-function malloc is sized from real parameters (not yet-unassigned
    # ``__inl1_N`` locals).
    if ctx.inl_defs:
        ctx.resolve_inl_table(ctx.zeros_locals)
        ctx.resolve_inl_table(shapes)


def fold_local_shape_attr_tokens(
    tuple_tables: list[dict[str, object]], reassign_shapes: dict[str, list] | None
) -> None:
    """Fold surviving ``arr.shape[i]`` STRING tokens in the finalised shape tables
    against the now-resolved shapes of the LOCAL arrays they name.

    The inl resolver (:meth:`LoweringContext.resolve_inl_table`) only resolves a
    ``.shape[i]`` token against PARAMETER shapes, so a token naming a LOCAL survives
    it. That happens when a temp is derived from a local operand whose shape is
    itself resolved LATE -- the eigh eigenvector temps aliased off
    ``M = Linv @ h_sub @ Linv.T``: the ``M.shape[0]`` token is recorded before the
    chained-matmul shape of ``M`` is known, and the AST ``ResolveArrShape`` pass
    rewrites body Attributes, not the malloc / row-major-stride TABLES the emitter
    reads. By this final phase every local's shape is known, so a token-level fold
    turns ``M.shape[0]`` into ``k`` in both the allocation extents and the subscript
    strides. Reuses the frontend's token substituter (it operates on shape tokens,
    never on source). Iterated to a bounded fixpoint so a temp whose dimension names
    ANOTHER temp still converges; never-worse (a token whose base is unknown is
    left untouched)."""
    tables = [t for t in tuple_tables if t is not None]
    for unused in range(4):
        seed: dict[str, tuple[str, ...]] = {}
        for t in tables:
            for nm, shp in t.items():
                seed[nm] = tuple(shp)
        changed = False
        for t in tables:
            for nm in list(t):
                new = resolve_shape_attr_tokens(tuple(t[nm]), seed)
                if new != tuple(t[nm]):
                    t[nm] = list(new) if isinstance(t[nm], list) else new
                    changed = True
        if reassign_shapes:
            for nm in list(reassign_shapes):
                new_list = [tuple(resolve_shape_attr_tokens(tuple(s), seed)) for s in reassign_shapes[nm]]
                if new_list != [tuple(s) for s in reassign_shapes[nm]]:
                    reassign_shapes[nm] = new_list
                    changed = True
        if not changed:
            break


def lp_slice_fusion_and_resolve(ctx: LoweringContext) -> None:
    """Slice fusion, source-order ``arr.shape[i]`` resolution, and forcing
    index-array locals to int64."""
    tree = ctx.tree
    shapes = ctx.shapes
    SliceFusion(shapes).visit(tree)
    # Final pass: resolve any surviving ``arr.shape[i]`` references to the concrete
    # shape token from the harvested table. These survive whenever a harvested helper
    # variable's shape was a string-form ``arr.shape[i]`` (e.g. inlined maxpool
    # ``np.empty([x.shape[0], x.shape[1] // 2, ...])`` -- the harvest stage resolves
    # what it can, but the body's range / loop bounds still reference the attribute
    # expression). The emit walker has no idea what to do with an Attribute, so we
    # substitute here.
    #
    # Seed the source-order resolver with bench-info parameter shapes only (not the
    # harvest's final-state shapes for reassigned locals). The resolver then builds
    # the current shape table per statement as it walks, so ``x.shape[i]`` at line K
    # resolves against the shape ``x`` had AT line K -- not after the kernel's final
    # reassignment.
    #
    # Stash a fresh copy of the reassign FIFO on the IR -- the emit walker consumes
    # it in source order to thread per-statement shape into multi-D subscript
    # flattening. The resolver below consumes its OWN copy (a fresh dict).
    ctx.kir.reassign_shapes = {k: list(v) for k, v in ctx.wa_rewriter._reassign_shapes.items()}
    ResolveArrShape(
        shapes,
        param_shapes={k: tuple(v) for k, v in ctx.arrays_shapes.items()},
        zeros_locals={k: tuple(v) for k, v in ctx.zeros_locals.items()},
        reassign_shapes={k: list(v) for k, v in ctx.wa_rewriter._reassign_shapes.items()},
    ).visit(tree)
    ast.fix_missing_locations(tree)
    # Force index-array LOCALS to int64. A local whose VALUES index another array
    # (``delv[neigh_safe[w0]]`` -- neigh_safe = np.clip(lxim, ..) is a local, so the
    # param-only detect_output_and_index_arrays misses it) must be integer;
    # C/Fortran reject a float subscript. A name used as a subscript index is always
    # integral, so this is sound.
    # Ordered: the loop below inserts into ``ctx.local_dtypes``, and that dict's order is what
    # the fp8 prelude and the Fortran declaration block iterate.
    idx_locals = OrderedSet()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            sl = node.slice
            elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            for e in elts:
                if isinstance(e, ast.Subscript) and isinstance(e.value, ast.Name):
                    idx_locals.add(e.value.id)  # A[B[i]] -> B is an index array
                elif isinstance(e, ast.Name):
                    idx_locals.add(e.id)  # A[B] (whole-array gather) -> B
    for nm_ in idx_locals:
        # A name the mask harvest PROVED boolean is a mask, never an index set: ``A[m]`` selects
        # the entries where ``m`` is true. Retyping it int64 is what let azimint_naive's
        # ``bin_id[valid]`` compile as a gather through 0/1 truth values -- every point binned as
        # if its index were 0 or 1, and a quiet wrong answer rather than a refusal.
        if nm_ in ctx.bool_names:
            continue
        if nm_ in shapes or nm_ in ctx.local_dtypes:
            dt_ = ctx.local_dtypes.get(nm_)
            if not (dt_ and dtypes.is_integer(dt_)):
                ctx.local_dtypes[nm_] = "int64"
    # Resolve any ``arr.shape[i]`` token left in the malloc / stride tables against
    # the now-fully-known LOCAL shapes (M => (k, k)), so eigh temps aliased off a
    # late-resolved chained-matmul operand allocate + index with concrete extents.
    fold_local_shape_attr_tokens([ctx.arrays_shapes, shapes, ctx.zeros_locals], ctx.kir.reassign_shapes)


def lp_promote_true_division(ctx: LoweringContext) -> None:
    """Promote all-integer ``/`` to a floating divide (numpy true division).

    Runs EARLY -- right after dtype seeding + harvest, BEFORE the LibNode /
    slice / reshape passes synthesize their own integer index arithmetic
    (row-major ``idx / stride`` decompositions that MUST stay integer). It
    therefore only rewrites divisions the numpy SOURCE wrote, never internally
    generated index math. Array names come from the harvested shape tables so a
    float array element is not mistaken for an integer; a later scalarization of
    a whole-array ``np.float64(a) / b`` recurses into the cast operand
    unchanged.

    The dtype table is ``local_dtypes`` (arrays) WIDENED with the declared scalar params
    and the body's provable float scalars: neither is in ``local_dtypes``, and
    ``is_integer_expr`` reads an untagged non-array Name as integer, so without them a
    float divide (chebyshev_filter_subspace's ``sigma1 / e``, over declared float64
    params) is promoted as if it were int/int -- baking an fp64 cast into an otherwise
    fp32 kernel."""
    array_names = set(ctx.lib_shape_table) | set(ctx.arrays_shapes)
    tags: dict[str, str] = {s.name: s.dtype for s in ctx.kir.scalars}
    tags.update(ctx.local_dtypes)  # the harvested tags win over the declaration
    ScalarFloatTagger(tags, array_names).visit(ctx.tree)
    TrueDivisionPromoter(tags, array_names).visit(ctx.tree)
    ast.fix_missing_locations(ctx.tree)


def lp_forward_substitute_invariant_scalars(ctx: LoweringContext) -> None:
    """Replay loop-invariant scalars deeper (POLYCC-001/006); slice fusion made them deep."""
    tables = (ctx.arrays_shapes, ctx.shapes, ctx.zeros_locals, ctx.lib_shape_table, ctx.kir.reassign_shapes)
    # Owning a shape means array; appearing in one sizes a declaration (dwt2d's ``s``).
    blocked = set(ctx.kir.sparse or {}) | {a.name for a in ctx.kir.arrays}
    for table in tables:
        blocked |= set(table)
        for value in table.values():
            for token in value if isinstance(value, (list, tuple)) else [value]:
                for tok in token if isinstance(token, (list, tuple)) else [token]:
                    blocked |= set(DIM_IDENT_RE.findall(str(tok)))
    subst = ForwardSubstituteInvariantScalars(blocked, set(ctx.kir.input_args)).run(ctx.tree)
    if not subst.substituted:
        return
    # Its declaration would now be an unused local, and unused is a warning is an error.
    gone = subst.substituted
    ctx.kir.int_locals = [n for n in ctx.kir.int_locals if n not in gone]
    ctx.kir.scalar_call_temps = [n for n in ctx.kir.scalar_call_temps if n not in gone]
    for name in gone:
        ctx.local_dtypes.pop(name, None)


def lp_scalarized_math_rename(ctx: LoweringContext) -> None:
    """Third and last math rename, once slice fusion has turned whole-array statements into
    per-element ones. An intrinsic whose operand only becomes a scalar HERE (hardsigmoid's
    ``np.clip((x + 3.0) / 6.0, 0, 1)`` -- array-valued at both earlier renames) has no later pass to
    catch it and reaches the emitter as an unsupported call. Same rewriter, later state."""
    MathRewriter(set(ctx.arrays_shapes.keys()) | set(ctx.lib_shape_table.keys())).visit(ctx.tree)


def lp_lower_helpers(ctx: LoweringContext) -> None:
    """Lower each non-inlinable helper the same way -- it is a self-contained
    sub-kernel (own params + body). Its early ``return`` survives lowering (the
    return-extraction is a parse_kernel step, not a lowering pass)."""
    for helper in ctx.original_kir.helpers:
        retype_int_helper_scalars(helper)
    # A helper body calls its SIBLINGS, and its own IR lists none of them; hand the by-value
    # scalar ones down so the shape passes read such a call as rank 0 rather than elementwise.
    siblings = scalar_return_helpers(ctx)
    ctx.kir.helpers = [lower(h, scalar_helpers=siblings) for h in ctx.original_kir.helpers]


#: The lowering pipeline as data: an ordered list of ``(name, phase)`` pairs run
#: over a shared :class:`LoweringContext`. The ORDER is load-bearing -- a slice
#: pass consults the array-shape table, so ``np.zeros`` locals must be registered
#: before it; see each phase's docstring / inline comments for the rationale.
LOWER_PHASES: list[tuple[str, Callable[["LoweringContext"], None]]] = [
    ("seed-shape-table", lp_seed_shape_table),
    ("normalize-calls", lp_normalize_calls),
    ("promote-params", lp_promote_params),
    ("pre-libnode-normalize", lp_pre_libnode_normalize),
    ("seed-dtypes-and-harvest", lp_seed_dtypes_and_harvest),
    ("promote-true-division", lp_promote_true_division),
    ("resolve-inlined-shapes", lp_resolve_inlined_shapes),
    ("normalize-index-access", lp_normalize_index_access),
    ("libnode-expand", lp_libnode_expand),
    ("scatter-at", lp_scatter_at),
    ("whole-array-and-zeros", lp_whole_array_and_zeros),
    ("slice-normalize-and-lift", lp_slice_normalize_and_lift),
    ("slice-fusion-and-resolve", lp_slice_fusion_and_resolve),
    ("forward-substitute-invariant-scalars", lp_forward_substitute_invariant_scalars),
    ("scalarized-math-rename", lp_scalarized_math_rename),
    ("lower-helpers", lp_lower_helpers),
]

#: Environment flag that turns on the between-phase invariant checks below.
#: Off by default (production emit pays nothing); flip it on to localise a
#: pipeline regression to the exact phase that corrupted the shared state.
INVARIANT_ENV = "HPCAGENT_BENCH_LOWER_INVARIANTS"


def assert_lowering_invariants(phase_name: str, ctx: LoweringContext) -> None:
    """Check the cross-phase invariants that must hold after ``phase_name``.

    Debug-only (gated by :data:`INVARIANT_ENV`). Every failure names the phase
    that broke it, so a side-table assigned the wrong container type or an AST
    the context stopped tracking is caught at the phase boundary that introduced
    it -- not later, as an inscrutable emit-time ``KeyError`` or ``AttributeError``.
    """
    kir = ctx.kir
    # The context's tree handle must remain the kir's tree: a phase that rebuilds
    # the AST has to write it back to both, or later phases rewrite an orphan.
    if ctx.tree is not kir.tree:
        raise AssertionError(
            f"lowering invariant after '{phase_name}': ctx.tree no "
            "longer aliases ctx.kir.tree (a phase rebuilt the AST "
            "without writing it back to both handles)"
        )
    if not isinstance(kir.tree, ast.FunctionDef):
        raise AssertionError(
            f"lowering invariant after '{phase_name}': kir.tree is {type(kir.tree).__name__}, expected ast.FunctionDef"
        )
    # The typed side-tables keep their declared container type -- a phase that
    # assigns the wrong shape surfaces here, not at the emitter reader.
    for fld, typ in (
        ("int_locals", list),
        ("local_dtypes", dict),
        ("zeros_locals", dict),
        ("zeros_fills", dict),
        ("scalar_call_temps", list),
        ("reassign_shapes", dict),
    ):
        val_ = operator.attrgetter(fld)(kir)
        if not isinstance(val_, typ):
            raise AssertionError(
                f"lowering invariant after '{phase_name}': kir.{fld} is {type(val_).__name__}, expected {typ.__name__}"
            )
    # The AST stays structurally well-formed: a rewriter that leaves a bad field
    # (a raw string where a node belongs, a Call missing args) fails to unparse.
    # Unparse a fixed-up copy -- synthetic nodes legitimately lack ``lineno``
    # mid-lowering, so filling locations on a throwaway keeps the check about
    # structure (and leaves the real tree untouched).
    try:
        ast.unparse(ast.fix_missing_locations(copy.deepcopy(kir.tree)))
    except Exception as exc:
        raise AssertionError(
            f"lowering invariant after '{phase_name}': kir.tree does not round-trip through ast.unparse ({exc})"
        ) from exc


def dtype_verdict(tag: str) -> str | None:
    """``"complex"`` / ``"real"`` for a dtype token, ``None`` for one that names no width here.

    ``np_float`` / ``np_complex`` are the framework's PRECISION GLOBALS: a reference binds them off
    the framework module so one source runs at either precision, and they arrive as bare names the
    dtype registry has never carried. Unknown stays unknown rather than defaulting -- such a token
    must neither pin a name real nor widen it, and the registry RAISES on one it does not know.
    """
    if tag in ("np_complex", "np_float"):
        return "complex" if tag == "np_complex" else "real"
    try:
        return "complex" if dtypes.canonical(tag).startswith("complex") else "real"
    except (KeyError, TypeError):
        return None


def elementwise_store_base(node: ast.AST) -> str | None:
    """``name`` written by an elementwise store, or None. AugAssign counts: a matmul temp is ZEROED by
    a plain assign and then ACCUMULATED into, so the accumulate is the only statement that carries
    its operands' dtype."""
    tgt = (
        node.targets[0]
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        else node.target
        if isinstance(node, ast.AugAssign)
        else None
    )
    if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name):
        return tgt.value.id
    return None


def tag_complex_locals(
    kir, zeros_locals: dict[str, tuple[str, ...]], dtype_src: dict[str, str], dtype_literal: dict[str, str]
) -> None:
    """Give every COMPLEX zeros-local its complex dtype, so the emitter does not default it to real.

    A local array the emitter has no tag for is declared at the kernel's default FLOAT width. For a
    buffer that holds complex values that is not an approximation, it is half the storage, and the
    imaginary part is dropped on the way in with nothing to say so -- eigh_test's Jacobi work matrix
    came out real and its eigenvalues were wrong by 0.24.

    Only the complex verdict is applied, and only where nothing has pinned the name already: real
    and integer locals already resolve elsewhere, so widening the change past the failure it fixes
    would re-type buffers across the whole corpus for no stated reason.
    """
    from hpcagent_bench.translators.numpyto_common.numpy_desugar import (
        dtype_kind,
        dtype_table_,
    )  # here: numpy_desugar imports this module

    try:
        complex_tag = dtypes.complex_dtype_for(kir.float_precision or "float64")
    except KeyError:
        return  # no nameable complex width at this precision -- nothing to tag

    # Two sources, run together to a fixpoint because each feeds the other: a ``zeros_like`` chain
    # (``scaled`` from ``bu`` from the eigh work matrix from the operand) resolves link by link, and
    # the assignment walk carries the answer across the matmul temps in between.
    seed = {a.name: ("complex" if dtype_verdict(a.dtype) == "complex" else "float") for a in kir.arrays}
    seed.update({n: "complex" for n, t in kir.local_dtypes.items() if dtype_verdict(t) == "complex"})
    # Names whose constructor stated a real dtype are settled; inference must not reach them.
    pinned_real = {n for n, lit in dtype_literal.items() if dtype_verdict(lit) == "real"}
    seed.update({n: "float" for n in pinned_real})

    stores = [(name, n.value) for n in ast.walk(kir.tree) for name in [elementwise_store_base(n)] if name is not None]
    for unused in range(8):
        # The WHOLE mapping, not its size: after the first pass propagation stops adding names and
        # only flips a name real -> complex, so a size comparison calls a fixpoint that has not been
        # reached. A chain whose stores do not appear in dependency order then stops one link short
        # and the last buffer is declared real -- the imaginary-part loss this function exists to
        # prevent, now silent.
        before = dict(seed)
        seed.update({n: k for n, k in dtype_table_(kir.tree, seed).items() if k})
        for name, src in dtype_src.items():
            if seed.get(src) == "complex":
                seed[name] = "complex"
        # A buffer written ELEMENTWISE from a complex value is a complex buffer. The whole-array
        # form (``x = <complex expr>``) is what the assignment walk reads, but by this point the
        # lowering has turned most of them into a store loop, so the name that gets declared is only
        # ever the base of a subscript.
        for base, value in stores:
            if base not in pinned_real and dtype_kind(value, seed) == "complex":
                seed[base] = "complex"
        if seed == before:
            break
    for name in zeros_locals:
        if name not in pinned_real and seed.get(name) == "complex":
            kir.local_dtypes.setdefault(name, complex_tag)


def lower(
    kir: KernelIR,
    native_call: Callable[[tuple[str, str], ast.Call, dict[str, tuple[str, ...]], dict[str, str]], bool] | None = None,
    blas: bool = False,
    fft_library: bool = False,
    scalar_helpers: set[str] | None = None,
    fft_library_nd: bool = False,
) -> KernelIR:
    """Return a lowered copy of ``kir`` ready for backend emission.

    ``native_call(key, call, shapes, dtypes)`` is the target's answer to "do you render this numpy
    call yourself?", asked with the array-shape and element-dtype tables: a per-axis form needs the
    operand's rank, and a semantics-sensitive one needs its element type (Fortran and numpy agree on
    a floating reduction and disagree on an integer one).
    A call it claims is left UNEXPANDED for the emitter -- Fortran uses it to keep ``SUM``/``MAXVAL``
    and friends as intrinsics instead of loop nests (see :mod:`numpyto_fortran.intrinsics`). The
    default claims nothing, which is C's answer and the behaviour every caller had before.

    The body is a fixed sequence of named phases (:data:`LOWER_PHASES`), each
    mutating a shared :class:`LoweringContext`. Pipeline shape: math rename ->
    ``np.zeros`` -> slice fusion. Order matters: the slice rewriter consults the
    array-shape table, and ``np.zeros`` locals must be registered first so their
    shapes are visible to it.

    Matmul (``A @ B`` / ``np.matmul`` -- normalised to ``@`` by
    :class:`MatmulCallRewriter`) is loop-lowered for every target EXCEPT a dense 2-D float
    contraction under ``blas``, which becomes a :data:`lib_nodes.BLAS_GEMM_MARKER` call the
    target's emitter renders as its own gemm. Every other shape -- batched, transposed, matvec,
    sparse, non-float -- keeps the loop nest on every target. The Fortran ``MATMUL`` intrinsic is
    reserved for the rare unresolved-shape case the loop hoister cannot lower (Fortran emitter).

    ``fft_library`` is the same "render a real library call instead of a loop nest" signal, for a
    whole-array 1-D ``np.fft.fft``/``ifft``/``fftn``/``ifftn`` (rank == 1 only -- a batched/N-D
    transform keeps the naive loop on every target, library or not): it becomes a
    :data:`lib_nodes.FFT_LIBRARY_MARKER` call, which C/C++/Fortran render as an FFTW3 call and
    numba renders as an ``objmode`` call into ``numpy.fft``. SEPARATE from ``blas`` -- a target
    with no BLAS_GEMM_MARKER renderer (Fortran, numba) can still opt into this one.
    ``fft_library_nd`` (numpyto_c only) also renders a batched / N-D transform as
    :data:`lib_nodes.FFTN_LIBRARY_MARKER` -- one ``fftw_plan_many_dft`` -- instead of the naive loop.

    Set :data:`INVARIANT_ENV` in the environment to run
    :func:`assert_lowering_invariants` after every phase.
    """
    check = assert_lowering_invariants if INVARIANT_ENV in os.environ else None
    # One lower() call == one translation unit, so this is where the lib_nodes scratch-name
    # counters start over; leaving them running makes the text depend on emission order.
    reset_temp_counters()
    ctx = LoweringContext(kir, copy.deepcopy(kir))
    ctx.native_call = native_call
    ctx.blas = blas
    ctx.fft_library = fft_library
    ctx.fft_library_nd = fft_library_nd
    ctx.sibling_scalar_helpers = set(scalar_helpers or ())
    for name_, phase in LOWER_PHASES:
        phase(ctx)
        if check is not None:
            check(name_, ctx)
    # The phases promote body shape names to symbols (:func:`promote_free_names_to_params`,
    # :func:`promote_shape_symbols_to_params`), which the manifest never saw. Re-stamp so a
    # promoted dimension carries the same positivity as a declared one.
    stamp_symbol_assumptions(ctx.kir)
    return ctx.kir
