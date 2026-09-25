"""The kernel IR builder behind :func:`parse_kernel`."""

import ast
import pathlib
import re

from hpcagent_bench.translators.numpyto_common import dtypes
from hpcagent_bench.translators.numpyto_common.ir import (
    ArrayDesc,
    KernelIR,
    ScalarDesc,
    SymbolDesc,
    stamp_symbol_assumptions,
)
from hpcagent_bench.translators.numpyto_common.ordered import OrderedSet
from hpcagent_bench.translators.numpyto_common.numpy_desugar import (
    EighCallHoister,
    EighLoopRewriter,
    module_kind_tables,
    eigh_alias_names,
    kind_of_dtype_str,
    fold_finfo_eps,
    fold_list_accumulators,
    rank_table,
    rewrite_curve_fit,
)
from hpcagent_bench.translators.numpyto_common.tuple_desugar import desugar_tuples
from hpcagent_bench.translators.numpyto_common.frontend.axes import (
    AxisReshapeToIndexing,
    FoldConstantSymbols,
    reject_symbolic_axis,
    reject_unsupported_slices,
    runtime_axis_dispatch,
    specialize_runtime_axis,
    structural_constants,
)
from hpcagent_bench.translators.numpyto_common.frontend.body_rewrites import (
    UnpackedOpenMeshToGrid,
    FoldParamNoneGuard,
    FoldTupleLocals,
    NonFiniteNormalizer,
    SubstituteParamAliases,
    rename_rebound_parameters,
    strip_framework_dtype_rebinding,
    native_desugar,
    version_rebound_locals,
)
from hpcagent_bench.translators.numpyto_common.frontend.helper_kirs import build_helper_kirs
from hpcagent_bench.translators.numpyto_common.frontend.initialize import dtypes_from_initialize, shapes_from_initialize
from hpcagent_bench.translators.numpyto_common.frontend.inlining import (
    HoistMultiStmtHelpers,
    InlineHelpers,
    collect_inlinable_helpers,
    flatten_nested_helpers,
    fuse_guarded_returns,
    unroll_const_list_loops,
)
from hpcagent_bench.translators.numpyto_common.frontend.int_usage import names_used_as_int
from hpcagent_bench.translators.numpyto_common.frontend.manifest import (
    PinnedValue,
    collect_bool_preset_names,
    collect_float_preset_names,
    collect_symbols,
    declared_ranks,
    default_array_dtype,
    fallback_shape_for_legacy,
    infer_scalar_dtype,
    load_bench_info,
    parse_shape_expression,
    as_block,
    as_list,
    declared_dtypes,
    declared_index_arrays,
    declared_shapes,
    pinned_config_in_use,
    pinned_values,
    shape_only_constants,
    symbol_sign_from_bindings,
)
from hpcagent_bench.translators.numpyto_common.frontend.module_constants import (
    ModuleConst,
    find_function,
    fold_consts_into_shapes,
    fold_default_args,
    inline_module_constants,
    materialize_const_arrays,
)
from hpcagent_bench.translators.numpyto_common.frontend.none_guarded import (
    SpliceNoneGuardedCalls,
    collect_none_guarded_helpers,
)
from hpcagent_bench.translators.numpyto_common.frontend.returns import (
    derive_returned_array_metadata,
    promote_scalar_returns,
    strip_trailing_return,
    synthesize_return_temps,
)
from hpcagent_bench.translators.numpyto_common.frontend.shapes import fold_dtype_aliases, resolve_shape_reads
from hpcagent_bench.translators.numpyto_common.frontend.sparse import PruneSparseDispatch, expand_sparse_arrays


def build_kernel_ir(
    numpy_py: pathlib.Path,
    bench_info: pathlib.Path,
    config: str | None = None,
    precision: str | None = None,
    keep_helpers: bool = False,
    open_mesh_grids: bool = True,
) -> KernelIR:
    """Build a :class:`KernelIR` from ``numpy_py`` + ``bench_info``.

    :param numpy_py: path to ``<short>_numpy.py``.
    :param bench_info: path to ``bench_info/<short>.json``.
    :param config: explicit sparse configuration key to emit (the
        deterministic path; the harness passes ``ResolvedBench.config_key``).
        Falls back to the canonical default when ``None``.
    :param precision: working float precision, for source-level desugars whose
        output embeds a precision-dependent constant (currently only
        curve_fit's finite-difference step). Dtypes aren't set here -- that's
        ``ir.apply_precision`` after lowering. ``None`` keeps constants at fp64.
    :param keep_helpers: leave ordinary helper calls in place instead of inlining them, so each
        helper is emitted as its own static function (see :func:`parse_kernel`). The forms with
        no standalone ABI are still spliced.
    :param open_mesh_grids: rewrite ``gx, gy = np.ix_(..)`` into one grid name for the native
        emitters. dace takes the unpacked names and refuses the packed one, so its path passes False.
    :raises ValueError: when the JSON is missing required fields, or no
        function in the Python file matches ``bench_info.func_name``.
    """
    info = load_bench_info(bench_info)
    init_block = as_block(info.get("init"))
    func_name = str(info["func_name"])
    array_args = [str(a) for a in as_list(info["array_args"])]
    input_args = [str(a) for a in as_list(info["input_args"])]
    output_args = [str(a) for a in as_list(info.get("output_args", []))]
    shapes_raw = declared_shapes(init_block)
    # The dtype half of the same declaration surface -- read here, beside the shapes, so both
    # halves see the same ``rename`` fixup below (see :func:`declared_dtypes`).
    dtypes_raw = declared_dtypes(init_block)
    parameters: dict[str, object] = as_block(info.get("parameters", {}))
    preset_symbols = collect_symbols(parameters)
    # Preset names with a non-integer value (e.g. solver ``tol``=1e-6) are float
    # SCALARS, not integer symbols -- else they'd declare ``int`` and truncate to 0.
    float_preset_names = collect_float_preset_names(parameters, as_block(init_block.get("scalars")))
    # Preset names with a boolean value are CONFIG FLAGS, not integer symbols --
    # so Fortran declares them ``logical`` and ``if (flag)``/``.not. flag`` type-check.
    bool_preset_names = collect_bool_preset_names(parameters)
    # The manifest's fixed scalar bindings -- the convolution knobs (``conv_padding``,
    # ``conv_stride``, ``*_groups``) live here, not in the presets. Evidence for
    # :func:`symbol_sign_from_bindings`, and for the promoted names the emitter declares.
    manifest_scalars = as_block(init_block.get("scalars"))
    # Every value each config knob takes; ``parameters`` holds only one representative.
    config_values_ = as_block(info.get("config_values"))

    src = numpy_py.read_text()
    tree = ast.parse(src, filename=str(numpy_py))
    if open_mesh_grids:
        UnpackedOpenMeshToGrid().visit(tree)
    ast.fix_missing_locations(tree)
    # Rewrite ``w, v = eigh(a[, b], ...)`` (np.linalg / scipy.linalg / the
    # ``_sci_eigh`` alias) to a self-contained complex-Hermitian eigh loop nest
    # BEFORE helper inlining, so the module-level alias import is still in scope
    # and the eigh in a helper (cegterg's ``_diaghg``) is lowered before it inlines.
    eigh_aliases = eigh_alias_names(tree)
    # A nested eigh/eigvalsh call (``float(np.linalg.eigvalsh(T).max()) + beta``)
    # must be materialised into its own ``__eigv = <call>`` assign first, so the
    # direct-assign loop rewriter below can lower it.
    EighCallHoister(eigh_aliases).visit(tree)
    ast.fix_missing_locations(tree)
    # dtype KIND (not the raw tag) for the loop rewriter's real/complex Jacobi choice: the declared
    # arrays plus the preset scalars, split the way the signature below types them. Carried across
    # helper calls by module_kind_tables, so an operand built in a helper is still resolvable.
    scalar_kinds = {
        **dict.fromkeys(preset_symbols, "int"),
        **dict.fromkeys(float_preset_names, "float"),
        **dict.fromkeys(bool_preset_names, "bool"),
    }
    array_kinds = {name: kind for name, dt in dtypes_raw.items() if (kind := kind_of_dtype_str(dt)) is not None}
    declared_kinds = {**scalar_kinds, **array_kinds}
    kind_tables = module_kind_tables(tree, func_name, declared_kinds)
    EighLoopRewriter(eigh_aliases, declared_kinds, kind_tables, dtypes_raw).visit(tree)
    # Canonicalise inf/nan spellings module-wide (see _NonFiniteNormalizer) so
    # both kernel and helpers are covered.
    NonFiniteNormalizer().visit(tree)
    ast.fix_missing_locations(tree)
    fn = find_function(tree, func_name)
    if fn is None:
        raise ValueError(f"{numpy_py}: no function named {func_name!r}")
    strip_framework_dtype_rebinding(fn)
    # Inline top-level helpers ABOVE the kernel whose body is a single
    # ``return expr`` by substituting the call with that expression (params
    # renamed to the call's args) -- lets NumpyToC handle e.g. nussinov's
    # ``match(b1, b2)`` without emitting a C/Fortran function for it.
    # bench_info.input_args is positional; when its names disagree with the
    # kernel signature (mandelbrot lists ``XN``/``YN`` for ``xn``/``yn``),
    # the harness still pairs by position, so align ``input_args`` to the
    # kernel's real parameter names and update ``array_args``/``output_args``.
    fn_param_names = [a.arg for a in fn.args.args]
    if len(input_args) == len(fn_param_names) and input_args != fn_param_names:
        rename = dict(zip(input_args, fn_param_names))
        input_args = list(fn_param_names)
        array_args = [rename.get(a, a) for a in array_args]
        output_args = [rename.get(a, a) for a in output_args]
        # ``parameters`` feeds ``preset_symbols`` -- rename here too so size
        # symbols still resolve as integer params.
        new_parameters: dict[str, object] = {}
        for preset, vals in parameters.items():
            new_parameters[preset] = {rename.get(k, k): v for k, v in as_block(vals).items()}
        parameters = new_parameters
        preset_symbols = collect_symbols(parameters)
        # The init declarations also key on the original names.
        shapes_raw = {rename.get(k, k): v for k, v in shapes_raw.items()}
        dtypes_raw = {rename.get(k, k): v for k, v in dtypes_raw.items()}

    # Inline module-level numeric constants (``BET_M = 0.5`` in vadv); left as
    # free Names they'd emit as bogus kernel parameters the harness can't
    # resolve. Only top-level ``NAME = <number>`` assigns the kernel neither
    # takes as a parameter nor reassigns locally are inlined. The folded names
    # are accumulated across every round below: shape tokens and the shape-symbol
    # promotion in lowering must both see that they are no longer free symbols.
    inlined_consts: dict[str, ModuleConst] = dict(inline_module_constants(tree, fn, input_args))
    # Fold kernel params that carry a DEFAULT and aren't in input_args into
    # body constants -- the harness only passes input_args, so e.g. the sp_*
    # solvers' ``max_iter=100``/``tol=1e-6`` stay fixed, not runtime params.
    # Otherwise a float ``tol`` mis-synthesized as int would never trip the
    # convergence break, so the solver iterates past convergence -> nan.
    fold_default_args(fn, input_args)
    # Drop the scipy-sparse dispatch branch: static backends are dense-only,
    # so ``sp.issparse(x)`` is statically False and the guarded path
    # (banded_mmt's sparse branch) is dead code; this leaves the dense path.
    PruneSparseDispatch().visit(fn)
    # Fold ``if <param> is None`` optional-default guards (params are always
    # supplied across the C ABI) -- drops the unlowerable ``None`` literal.
    FoldParamNoneGuard(input_args).visit(fn)
    # Substitute ``local = <param>`` whole-array aliases with the parameter so
    # write-through (``vt = p_diag_vt; vt[...] = ...``) reaches the output and
    # read-only aliases don't pay for a copy.
    alias_sub = SubstituteParamAliases(input_args)
    alias_sub.collect(fn)
    alias_sub.visit(fn)  # also drops no-op ``x = x`` self-assignments
    ast.fix_missing_locations(fn)

    # Rewrite ``popt, _ = curve_fit(model, x, y, p0=guess)`` to a naive
    # Levenberg-Marquardt loop nest (plus the list preludes that build p0 into
    # arrays). Runs BEFORE helper inlining, like the eigh rewriter above, so
    # the model ``def`` is still distinct and its varargs can rebind to the
    # array curve_fit conceptually passes; the fixpoint below then inlines
    # the LM's calls to the model.
    rewrite_curve_fit(tree, fn, precision)

    # A round-off bound written as ``np.finfo(y.dtype).eps`` becomes that precision's
    # epsilon literal. An accuracy requirement is a plain number and never comes here.
    fold_finfo_eps(tree, precision)

    # Strip every top-level helper's give-up paths: bail-only exception
    # handlers (``except np.linalg.LinAlgError: return None``) and
    # ``if <diverged>: return None`` sentinels. Runs BEFORE the inline
    # fixpoint, not with native_desugar (which runs after): these early
    # returns disqualify Form-3 (single-tail-return) inlining, and a
    # tuple-returning helper (distribution_search's ``solve_three_levels``)
    # has no emittable ABI unless inlined into its caller.

    # Flatten helpers NESTED inside other helpers first (lulesh's per-helper
    # ``def c(a, i): return a[:, i]`` shorthand): a helper containing a nested
    # def is rejected by _collect_inlinable_helpers (FunctionDef isn't an
    # allowed mid statement) and would never inline -- its nested def is only
    # exposed by inlining the parent, a deadlock otherwise.
    flatten_nested_helpers(tree)
    fuse_guarded_returns(tree)
    # Inline helper calls to a FIXPOINT: one pass only inlines the outermost
    # level (NodeTransformer doesn't re-visit spliced-in bodies), so a chain of
    # helpers calling helpers (lulesh's ``_lagrange_nodal`` -> ``_calc_force_
    # for_nodes`` -> ... -> ``_calc_shape_fn_derivatives``) needs repeated
    # passes. Each round re-collects (exposing a helper-local ``def`` freed by
    # inlining its parent) and re-inlines module constants now living in
    # spliced-in bodies.
    # Counters are shared across iterations so ``__inl<N>_``/``__hcall<N>``
    # prefixes stay globally unique -- a per-iteration reset could collide a
    # later-inlined nested helper with an earlier outer one.
    inl_counter: list[int] = [0]
    hcall_counter: list[int] = [0]

    def run_regular_inline_fixpoint() -> None:
        if keep_helpers:
            return
        for unused in range(64):
            helpers = collect_inlinable_helpers(tree, fn)
            if not helpers:
                break
            names = OrderedSet(helpers)
            # Hoist Form-3 (multi-statement-return) helper calls nested inside
            # expressions to standalone Assigns first (``relu(conv2d(input, w) + b)``
            # -> ``__hcall0 = conv2d(input, w); relu(__hcall0 + b)``), so
            # _InlineHelpers can inline via its visit_Assign path.
            # Unroll ``for x in [<const tuples>]: body`` (lulesh face-node loops)
            # BEFORE inlining, so per-iteration void-helper calls become concrete
            # statements the inliner can splice.
            unroll_const_list_loops(fn)
            HoistMultiStmtHelpers(helpers, hcall_counter).visit(fn)
            InlineHelpers(helpers, inl_counter).visit(fn)
            ast.fix_missing_locations(fn)
            inlined_consts.update(inline_module_constants(tree, fn, input_args))
            # Done when no call to a (still-inlinable) helper survives in the body.
            if not any(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names for n in ast.walk(fn)
            ):
                break

    run_regular_inline_fixpoint()
    # Splice any surviving "returns None-or-a-tuple" helper (see _collect_none_guarded_helpers)
    # into its call site, together with the caller's own "is None" guard and unpack -- Form 3 above
    # refuses these outright (an early return disqualifies it). Each round may expose a fresh call
    # to an ordinary (Form 1/2/3) helper nested in the spliced-in body (_transpose_taps's own call
    # to _ceil_div), so the regular fixpoint gets one more pass afterward.
    for unused in range(8):
        none_guarded = collect_none_guarded_helpers(tree, fn)
        if not none_guarded:
            break
        # Every owner, not just the kernel: ``_tap_range`` is called from ``_conv_transpose3d`` and
        # never from the kernel body, so splicing into ``fn`` alone left it a tuple-returning
        # function with no C ABI. Same rule the tuple splice in ``build_helper_kirs`` already
        # follows -- a sentinel return has no ABI ANYWHERE.
        owners = [fn] + [n for n in tree.body if isinstance(n, ast.FunctionDef) and n is not fn]
        splicer = SpliceNoneGuardedCalls(none_guarded, inl_counter)
        if not any([splicer.apply(owner) for owner in owners if owner.name not in none_guarded]):
            break
        ast.fix_missing_locations(fn)
        run_regular_inline_fixpoint()
    # Final unroll: the LAST inline round can splice in fresh ``for nk in
    # (n0,n1,n2,n3)`` tuple-literal loops (lulesh _sum_face_normal) after the
    # in-loop unroll already ran, so do one more pass once inlining settles.
    unroll_const_list_loops(fn)
    ast.fix_missing_locations(fn)
    # Re-fold ``local = param`` aliases EXPOSED BY INLINING. fv3_dycore's
    # copy_corners(field) (``f = field; f[corner] = f[...]``) becomes ``__inlN_f
    # = q; __inlN_f[corner] = ...`` after inlining; the earlier alias pass never
    # saw it, so a backend would copy q into a fresh buffer and lose the corner
    # writes (stale halo -> PPM reads garbage -> wrong fluxes). Re-running here
    # folds __inlN_f -> q so writes land on q.
    alias_sub_post = SubstituteParamAliases(input_args)
    alias_sub_post.collect(fn)
    alias_sub_post.visit(fn)
    ast.fix_missing_locations(fn)
    # Re-fold ``if <param> is None`` guards EXPOSED BY INLINING (same reason as
    # the alias re-fold above). A helper's own optional-default guard (lavamd's
    # ``lavamd_kernel(.., fv=None)`` -> ``if fv is None: fv = np.zeros(..)``) is
    # spliced in after the first fold already ran, leaving an unlowerable
    # ``None``/``is`` compare (params are always supplied across the ABI, so
    # it's dead). Must run AFTER the alias substitution: inlining renames the
    # param to ``__inlN_fv``, and only the alias fold maps that back onto the
    # real parameter for this pass to recognise it.
    FoldParamNoneGuard(input_args).visit(fn)
    ast.fix_missing_locations(fn)
    # Materialise module-level constant ARRAYS (lookup tables -- lulesh's
    # ``_VOLU_PERM = np.array([[...]], dtype=np.intp)``) into the kernel body as a
    # zeros local + element stores. Runs AFTER inlining so a table referenced only
    # inside a helper (lulesh's _calc_volume_derivative) is now in the kernel body.
    materialize_const_arrays(tree, fn, input_args)
    ast.fix_missing_locations(fn)
    # Native-backend desugars (newaxis, ufunc-out, roll-slice, complex accessors,
    # validation-guard drop, static-None fold). Applied here to the kernel body
    # AND, identically, to every non-inlined helper in ``build_helper_kirs`` so a
    # helper that survives inlining is not left with un-emittable constructs.
    native_desugar(fn)

    # Scalarize compile-time tuples AFTER inlining, so a tuple a helper built from its own
    # parameters (every KernelBench conv/pool port normalises a knob to ``(s, s)``) is folded
    # against the values the call site actually passed.
    scalar_names_ = frozenset(input_args) - frozenset(array_args)
    init_scalars = as_block(init_block.get("scalars"))

    def resolve_axes(target: ast.FunctionDef) -> None:
        """Put every structural position into the literal form the nest is built from, then refuse
        whatever is left symbolic. Applied to the body -- or, when the axis itself is a runtime
        argument, to each specialised clone of it."""
        # A structural constant becomes a literal BEFORE anything reads it: an axis, a repeat count
        # and a slice bound all pick the loop nest, none buildable from a runtime scalar.
        FoldConstantSymbols(structural_constants(parameters, init_scalars, shapes_raw, runtime_args=input_args)).apply(
            target
        )
        ast.fix_missing_locations(target)
        # expand_dims/swapaxes first: they become plain indexing, which the tuple pass can then rank.
        AxisReshapeToIndexing(rank_table(target, declared_ranks(shapes_raw)), scalar_names_).visit(target)
        ast.fix_missing_locations(target)
        desugar_tuples(
            target,
            int_scalars=scalar_names_ - frozenset(float_preset_names),
            float_scalars=frozenset(float_preset_names) & scalar_names_,
            arrays=frozenset(array_args),
            ranks=rank_table(target, declared_ranks(shapes_raw)),
        )
        # Whatever axis did not become a literal above has no emittable loop nest. Refuse it here
        # rather than let a downstream reader mistake it for "no axis at all". A slice step and a
        # negative slice start pick the nest the same way, so they are refused on the same pass.
        reject_symbolic_axis(target)
        reject_unsupported_slices(target)

    # An axis the ABI supplies has no single nest, but the operand's RANK is known, so the honest
    # emission is every nest it could pick plus the run-time choice between them -- never the
    # manifest default, which the harness need not pass.
    dispatch = runtime_axis_dispatch(fn, scalar_names_, rank_table(fn, declared_ranks(shapes_raw)))
    if dispatch is None:
        resolve_axes(fn)
    else:
        specialize_runtime_axis(fn, dispatch[0], dispatch[1], frozenset(input_args), resolve_axes)

    rename_rebound_parameters(fn, frozenset(array_args) - frozenset(output_args))
    version_rebound_locals(fn, frozenset(input_args) | frozenset(output_args) | frozenset(array_args))

    # Inline tuple-valued shape locals and fold tuple concatenation AFTER
    # inlining so references inside inlined helper bodies (vexx's invfft/fwfft
    # use the enclosing ``grid`` tuple in ``reshape(grid + (-1,))``) are caught.
    fold_tuples = FoldTupleLocals(input_args)
    fold_tuples.collect(fn)
    fold_tuples.visit(fn)
    ast.fix_missing_locations(fn)

    # Kernels may declare outputs via a final ``return X``/``return X, Y``
    # instead of in-place writes (mandelbrot / numpy-book style). Promote a
    # returned Name to an output array only when its shape is derivable --
    # otherwise the kernel would gain a bogus parameter (deriche's older
    # ``imgOut[:] = ...; return imgOut`` already declares its output via
    # bench_info and must not be promoted here).
    # Seed shapes from input arrays, so ``Q = np.zeros_like(A)`` (A a
    # parameter) mirrors A's shape; computed once, reused below.
    legacy_shapes = shapes_from_initialize(numpy_py, info)
    input_array_shapes: dict[str, str] = {}
    for a_ in array_args:
        s_ = shapes_raw.get(a_)
        if s_ is None:
            s_ = legacy_shapes.get(a_)
        if s_ is not None:
            input_array_shapes[a_] = s_

    # Every ``x.shape[k]`` becomes the extent the manifest declares. The emitted kernel has no
    # descriptor beside its buffers to read a shape out of, and one that survives here forks the
    # spelling of an extent the ABI already carries by name -- see :func:`resolve_shape_reads`.
    # Runs after the tuple fold, so a whole ``x.shape`` is already per-axis subscripts. The dtype
    # is a placeholder: only the shape half of the resolver's answer is read.
    shape_env = {
        n: ArrayDesc(name=n, dtype="float64", shape=parse_shape_expression(s), is_output=n in output_args)
        for n, s in input_array_shapes.items()
    }
    # The reads it could not resolve come back for the caller to report; nothing consumes
    # them yet, so an unresolved read still reaches the pass that owns its refusal.
    # A dtype read reached through a local name matches none of the ``x.dtype`` consumers; fold it
    # back to the attribute before any of them run (see :func:`fold_dtype_aliases`).
    fold_dtype_aliases(fn)
    # A list grown by ``append`` is an array written by a rule; fold it before lowering can read
    # ``len`` of it as an array extent (see :func:`fold_list_accumulators`).
    fold_list_accumulators(fn)
    resolve_shape_reads(fn, shape_env)
    # Synthesise temps for computed (non-Name) returns -- ``return A @ x``
    # -> ``__out0 = A @ x; return __out0`` -- so they promote like
    # ``return X``. ``revert_return`` undoes this if a shape can't be
    # derived (leaving the kernel untouched, i.e. an un-promoted skip).
    returned_outputs, revert_return = synthesize_return_temps(fn)
    if returned_outputs and not any(o in input_args for o in returned_outputs):
        returned_shapes, returned_dtypes = derive_returned_array_metadata(
            fn, returned_outputs, seed_shapes=input_array_shapes
        )
        if all(o in returned_shapes for o in returned_outputs):
            for out in returned_outputs:
                input_args.append(out)
                if out not in array_args:
                    array_args.append(out)
                if out not in output_args:
                    output_args.append(out)
            strip_trailing_return(fn)
            ast.fix_missing_locations(fn)
        elif not returned_shapes and not output_args:
            # SCALAR-only return with no other output would be silently dropped --
            # promote each to a 1-element float output buffer (grid_search's
            # binary-search index).
            for out in promote_scalar_returns(fn, returned_outputs):
                input_args.append(out)
                array_args.append(out)
                output_args.append(out)
                # Route through ``shapes_raw`` (runs ``parse_shape_expression``),
                # not ``returned_shapes``, so this parses to the ``('1',)`` dim
                # tuple the multidim subscript lowering expects (raw ``"(1,)"``
                # mis-tokenizes).
                shapes_raw[out] = "(1,)"
            ast.fix_missing_locations(fn)
        else:
            revert_return()
            returned_shapes, returned_dtypes = {}, {}
    else:
        revert_return()
        returned_shapes, returned_dtypes = {}, {}

    symbols: list[SymbolDesc] = []
    arrays: list[ArrayDesc] = []
    scalars: list[ScalarDesc] = []

    # Sparse layout expansion: any logical array carrying a non-dense
    # format for the chosen configuration becomes a set of physical
    # buffer arrays; the logical name is skipped from the dense/scalar
    # paths and recorded in ``sparse_descs`` for the matmul hoister.
    sparse_descs, sparse_buffer_arrays, logical_to_physical = expand_sparse_arrays(info, config)

    scalar_defaults = as_block(init_block.get("scalars"))
    fallback_shape = fallback_shape_for_legacy(preset_symbols)
    # Legacy HPCAgent-Bench JSONs (no array declarations at all) declare arrays
    # through an ``initialize`` function in a sibling Python module --
    # ``legacy_shapes`` was harvested above (reused here); recover dtypes
    # likewise before the 1-D fallback.
    legacy_dtypes = dtypes_from_initialize(numpy_py, info)
    index_names = declared_index_arrays(init_block)
    # The DECLARED dtypes (``init.arrays[<name>].dtype``, plus ``init.dtypes``
    # for the names that are not arrays) win over the initialize-harvest, so a
    # kernel like stockham_fft that allocates the output via
    # ``rng_complex(...)`` (not recognised by the constructor parser)
    # can still declare its complex outputs correctly.
    for k, v in dtypes_raw.items():
        legacy_dtypes[k] = v
    # Invariant over the per-arg loop: one full-tree walk hoisted out of it.
    int_names = names_used_as_int(fn)
    for arg in input_args:
        # Logical sparse arrays are expanded into physical buffers
        # separately (see ``sparse_buffer_arrays`` injection below) --
        # skip the dense / scalar treatment for the logical name.
        if arg in sparse_descs:
            continue
        if arg in array_args:
            # Return-style outputs: shape and dtype come from the
            # assignment-harvest, NOT bench_info (which does not list
            # them).
            if arg in returned_shapes:
                arrays.append(
                    ArrayDesc(
                        name=arg,
                        dtype=returned_dtypes.get(arg, default_array_dtype()),
                        shape=returned_shapes[arg],
                        is_output=True,
                        is_index=arg in index_names,
                    )
                )
                continue
            shape_expr = shapes_raw.get(arg)
            if shape_expr is None:
                shape_expr = legacy_shapes.get(arg)
            if shape_expr is None:
                if fallback_shape is None:
                    raise ValueError(
                        f"{bench_info}: array {arg!r} has no shape expression "
                        f"in init.shapes and no inferrable size symbol"
                    )
                shape_expr = fallback_shape
            arrays.append(
                ArrayDesc(
                    name=arg,
                    dtype=legacy_dtypes.get(arg, default_array_dtype()),
                    shape=parse_shape_expression(shape_expr),
                    is_output=arg in output_args,
                    is_index=arg in index_names,
                )
            )
        elif arg in preset_symbols and arg not in float_preset_names and arg not in bool_preset_names:
            declared_dt = legacy_dtypes.get(arg)
            symbols.append(
                SymbolDesc(
                    name=arg,
                    # A symbol is an integer by construction (the float and bool presets are
                    # routed to scalars above), so a declared non-integer dtype describes
                    # something else and must not narrow the binding's int64.
                    dtype=declared_dt if declared_dt and dtypes.is_integer(declared_dt) else "int64",
                    assumption=symbol_sign_from_bindings(arg, parameters, manifest_scalars, config_values_),
                )
            )
        elif arg in bool_preset_names:
            # A boolean config flag: a runtime ``bool`` scalar (C ``bool`` /
            # Fortran ``logical(c_bool)``), NOT an integer dimension.
            scalars.append(ScalarDesc(name=arg, dtype="bool", is_output=arg in output_args))
        else:
            # Plain scalar input (e.g. ``alpha`` in gemm): dtype comes from
            # ``init.scalars`` when present (int default -> int param, float
            # default -> double); otherwise falls back to double.
            # init.dtypes is authoritative for a scalar too, not only an array: srad's ROI
            # bounds have no init.scalars default to infer from.
            # legacy_dtypes, not dtypes_raw: the manifest is already merged on top of it, so a
            # declared dtype still wins, but a scalar the initializer BUILDS with its width --
            # compute's ``a = np.int64(4)`` -- is now typed like an array built the same way
            # instead of falling through to the run's float type and being called with an int64.
            inferred_dt = legacy_dtypes.get(arg) or infer_scalar_dtype(scalar_defaults.get(arg))
            # Promote to int when the kernel uses the scalar in an integer-only
            # context (``range(arg)`` / subscript / shape -- mirrors the C emit's
            # ``needs_int`` check), so e.g. nbody's ``Nt`` and lenet's
            # ``C_before_fc1`` declare ``int`` despite bench_info not pinning
            # their dtype. Plain ``int``, not ``int64``: must match the shape
            # symbols' kind, since Fortran's ``-std=f2018`` rejects mixed-kind
            # integer arithmetic (``int32_iter * int64_scalar``).
            # An array DIMENSION symbol is always integral even if the kernel
            # body never references it (vexx's ``npw`` only sizes ``psi``/``nl``):
            # otherwise it defaults to real and clashes with the array decl's
            # ``integer``.
            is_array_dim = any(re.search(rf"\b{re.escape(arg)}\b", str(tok)) for a in arrays for tok in a.shape)
            if inferred_dt in {"float64", "double", "float32"} and (arg in int_names or is_array_dim):
                inferred_dt = "int"
            scalars.append(
                ScalarDesc(
                    name=arg,
                    dtype=inferred_dt,
                    is_output=arg in output_args,
                    value=scalar_defaults.get(arg),
                )
            )

    # Inject the physical sparse buffer arrays + expand the logical
    # sparse names in input_args to their ordered physical buffers so
    # the emitted signature receives (A_indptr, A_indices, A_data, ...)
    # in place of the logical ``A``.
    if sparse_descs:
        arrays.extend(sparse_buffer_arrays)
        expanded_input: list[str] = []
        for arg in input_args:
            if arg in logical_to_physical:
                expanded_input.extend(logical_to_physical[arg])
            else:
                expanded_input.append(arg)
        input_args = expanded_input

    # The manifest shape tokens were written against the SOURCE names, so an
    # inlined module constant (cloudsc's ``nclv``) still spells the eliminated
    # name; fold it to its literal here, before anything derives symbols from
    # the shapes (helper params below, shape promotion in lowering).
    fold_consts_into_shapes(arrays, inlined_consts)

    short_name = info.get("short_name", func_name)
    pinned: dict[str, PinnedValue] = pinned_values(info.get("pinned_config"))
    kir = KernelIR(
        tree=fn,
        kernel_name=func_name,
        short_name=short_name,
        input_args=input_args,
        symbols=symbols,
        arrays=arrays,
        scalars=scalars,
        source_path=str(numpy_py),
        sparse=sparse_descs,
        inlined_consts=set(inlined_consts),
        # Pinned config knobs stay in ``symbols``/``scalars`` (the body reads them by name and
        # lowering has to resolve them) but leave the ABI: they are compile-time constants the
        # native emitters declare (see :attr:`KernelIR.pinned_consts`).
        pinned_consts=pinned_config_in_use(pinned, fn, arrays, input_args),
        # A manifest name only a declared shape spells is a constant no emitted artifact can
        # observe; the emitters that must prove an extent identity substitute it, the rest do not.
        shape_only_consts=shape_only_constants(parameters, init_scalars, arrays, fn, input_args, pinned),
    )
    # Helpers that survived the inlining fixpoint as CALLS (an early ``return`` /
    # recursion blocks inlining) become their own native functions -- the early
    # return is then just a native ``return``. Each helper param's type/shape is
    # inferred from the call site; :func:`lower` lowers every helper body too.
    kir.helpers = build_helper_kirs(tree, fn, kir, keep_helpers)
    # After the arrays exist: a symbol standing alone as a dimension is positive by allocation,
    # which is a stronger fact than the presets can give and applies to a helper's symbols too.
    stamp_symbol_assumptions(kir)
    for helper in kir.helpers:
        stamp_symbol_assumptions(helper)
    # A scalar that sizes an array is PROMOTED to a dc.symbol by the emitter, which owns no
    # descriptor for it and so had no sign to declare. Carry the evidence for every bound name
    # instead of re-deriving it there, where the manifest is out of scope.
    bound = set(manifest_scalars) | {n for preset in parameters.values() for n in as_block(preset)}
    kir.symbol_signs = {
        n: sign
        for n in sorted(bound)
        if (sign := symbol_sign_from_bindings(n, parameters, manifest_scalars, config_values_))
    }
    for helper in kir.helpers:
        helper.symbol_signs = kir.symbol_signs
    return kir
