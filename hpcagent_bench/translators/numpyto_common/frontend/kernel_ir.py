"""The kernel IR builder behind :func:`parse_kernel`."""

import ast
import dataclasses
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
    JsonBlock,
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


@dataclasses.dataclass(slots=True, kw_only=True)
class Signature:
    """The bench_info view of the kernel: argument lists, declarations, presets and bindings."""

    info: JsonBlock
    init_block: JsonBlock
    func_name: str
    input_args: list[str]
    array_args: list[str]
    output_args: list[str]
    shapes_raw: dict[str, str]
    dtypes_raw: dict[str, str]
    parameters: dict[str, object]
    preset_symbols: list[str]
    #: Preset names with a float value are float scalars, not integer symbols.
    float_presets: set[str]
    #: Preset names with a boolean value are config flags (``logical`` in Fortran).
    bool_presets: set[str]
    #: ``init.scalars``: fixed scalar bindings (conv knobs) and scalar defaults.
    scalars: JsonBlock
    #: Every value each config knob takes; ``parameters`` holds one representative.
    config_values: JsonBlock

    @classmethod
    def read(cls, bench_info: pathlib.Path) -> "Signature":
        info = load_bench_info(bench_info)
        init_block = as_block(info.get("init"))
        parameters: dict[str, object] = as_block(info.get("parameters", {}))
        return cls(
            info=info,
            init_block=init_block,
            func_name=str(info["func_name"]),
            array_args=[str(a) for a in as_list(info["array_args"])],
            input_args=[str(a) for a in as_list(info["input_args"])],
            output_args=[str(a) for a in as_list(info.get("output_args", []))],
            shapes_raw=declared_shapes(init_block),
            dtypes_raw=declared_dtypes(init_block),
            parameters=parameters,
            preset_symbols=collect_symbols(parameters),
            float_presets=collect_float_preset_names(parameters, as_block(init_block.get("scalars"))),
            bool_presets=collect_bool_preset_names(parameters),
            scalars=as_block(init_block.get("scalars")),
            config_values=as_block(info.get("config_values")),
        )

    def align_to(self, fn: ast.FunctionDef) -> None:
        """``input_args`` is positional: when its names disagree with the kernel signature, rename every
        manifest table to the signature's names (float/bool presets keep the manifest spelling)."""
        fn_param_names = [a.arg for a in fn.args.args]
        if len(self.input_args) != len(fn_param_names) or self.input_args == fn_param_names:
            return
        rename = dict(zip(self.input_args, fn_param_names))
        self.input_args = list(fn_param_names)
        self.array_args = [rename.get(a, a) for a in self.array_args]
        self.output_args = [rename.get(a, a) for a in self.output_args]
        self.parameters = {
            preset: {rename.get(k, k): v for k, v in as_block(vals).items()} for preset, vals in self.parameters.items()
        }
        self.preset_symbols = collect_symbols(self.parameters)
        self.shapes_raw = {rename.get(k, k): v for k, v in self.shapes_raw.items()}
        self.dtypes_raw = {rename.get(k, k): v for k, v in self.dtypes_raw.items()}

    def sign_of(self, name: str) -> str | None:
        return symbol_sign_from_bindings(name, self.parameters, self.scalars, self.config_values)


def build_kernel_ir(
    numpy_py: pathlib.Path,
    bench_info: pathlib.Path,
    config: str | None = None,
    precision: str | None = None,
    keep_helpers: bool = False,
    open_mesh_grids: bool = True,
) -> KernelIR:
    """Build a :class:`KernelIR` from ``numpy_py`` + ``bench_info``.

    :param config: sparse configuration key to emit; ``None`` picks the default one.
    :param precision: working float precision for desugars that embed a precision-dependent
        constant (curve_fit's finite-difference step, ``finfo`` eps). ``None`` keeps fp64.
    :param keep_helpers: leave helper calls in place so each helper is emitted as its own function
        (see :func:`parse_kernel`); forms with no standalone ABI are still spliced.
    :param open_mesh_grids: rewrite ``gx, gy = np.ix_(..)`` into one grid name (native emitters);
        dace refuses the packed form, so its path passes False.
    :raises ValueError: when the JSON lacks required fields or has no function ``func_name``.
    """
    sig = Signature.read(bench_info)
    tree, fn = parse_module(numpy_py, sig, open_mesh_grids)
    strip_framework_dtype_rebinding(fn)
    sig.align_to(fn)
    inlined_consts = prepare_body(tree, fn, sig.input_args, precision)
    inline_helpers(tree, fn, sig.input_args, keep_helpers, inlined_consts)
    native_desugar(fn)
    resolve_structural_axes(fn, sig)
    rename_rebound_parameters(fn, frozenset(sig.array_args) - frozenset(sig.output_args))
    version_rebound_locals(fn, frozenset(sig.input_args) | frozenset(sig.output_args) | frozenset(sig.array_args))
    # After inlining, so tuples referenced from inlined helper bodies fold too.
    fold_tuples = FoldTupleLocals(sig.input_args)
    fold_tuples.collect(fn)
    fold_tuples.visit(fn)
    ast.fix_missing_locations(fn)
    legacy_shapes = shapes_from_initialize(numpy_py, sig.info)
    input_array_shapes = input_array_shapes_(sig, legacy_shapes)
    resolve_shape_reads_(fn, sig, input_array_shapes)
    returned_shapes, returned_dtypes = promote_returns(fn, sig, input_array_shapes)
    sparse_descs, sparse_buffer_arrays, logical_to_physical = expand_sparse_arrays(sig.info, config)
    symbols, arrays, scalars = declare_arguments(
        ArgumentSources(
            fn=fn,
            sig=sig,
            bench_info=bench_info,
            sparse=frozenset(sparse_descs),
            legacy_shapes=legacy_shapes,
            legacy_dtypes=dtypes_from_initialize(numpy_py, sig.info) | sig.dtypes_raw,
            returned_shapes=returned_shapes,
            returned_dtypes=returned_dtypes,
        )
    )
    input_args = sig.input_args
    if sparse_descs:
        # The logical sparse name becomes its ordered physical buffers in the signature.
        arrays.extend(sparse_buffer_arrays)
        input_args = [phys for arg in input_args for phys in logical_to_physical.get(arg, [arg])]
    # Manifest shapes still spell inlined module constants; fold them before symbols are derived.
    fold_consts_into_shapes(arrays, inlined_consts)
    pinned: dict[str, PinnedValue] = pinned_values(sig.info.get("pinned_config"))
    kir = KernelIR(
        tree=fn,
        kernel_name=sig.func_name,
        short_name=sig.info.get("short_name", sig.func_name),
        input_args=input_args,
        symbols=symbols,
        arrays=arrays,
        scalars=scalars,
        source_path=str(numpy_py),
        sparse=sparse_descs,
        inlined_consts=set(inlined_consts),
        # Pinned knobs stay readable by name but leave the ABI as compile-time constants.
        pinned_consts=pinned_config_in_use(pinned, fn, arrays, input_args),
        shape_only_consts=shape_only_constants(sig.parameters, sig.scalars, arrays, fn, input_args, pinned),
    )
    # Helpers still called after inlining become their own functions.
    kir.helpers = build_helper_kirs(tree, fn, kir, keep_helpers)
    # A symbol standing alone as a dimension is positive by allocation.
    stamp_symbol_assumptions(kir)
    for helper in kir.helpers:
        stamp_symbol_assumptions(helper)
    # The sign evidence for every bound name, for scalars the emitter promotes to symbols.
    bound = set(sig.scalars) | {n for preset in sig.parameters.values() for n in as_block(preset)}
    kir.symbol_signs = {n: sign for n in sorted(bound) if (sign := sig.sign_of(n))}
    for helper in kir.helpers:
        helper.symbol_signs = kir.symbol_signs
    return kir


def parse_module(numpy_py: pathlib.Path, sig: Signature, open_mesh_grids: bool) -> tuple[ast.Module, ast.FunctionDef]:
    """Parse the kernel module and apply the module-wide rewrites that must precede inlining."""
    tree = ast.parse(numpy_py.read_text(), filename=str(numpy_py))
    if open_mesh_grids:
        UnpackedOpenMeshToGrid().visit(tree)
    ast.fix_missing_locations(tree)
    # eigh is lowered before inlining, while the module-level alias import is still in scope; a
    # nested eigh call is first hoisted into its own assignment.
    eigh_aliases = eigh_alias_names(tree)
    EighCallHoister(eigh_aliases).visit(tree)
    ast.fix_missing_locations(tree)
    # dtype KINDs for the real/complex Jacobi choice, carried across helper calls.
    scalar_kinds = {
        **dict.fromkeys(sig.preset_symbols, "int"),
        **dict.fromkeys(sig.float_presets, "float"),
        **dict.fromkeys(sig.bool_presets, "bool"),
    }
    array_kinds = {name: kind for name, dt in sig.dtypes_raw.items() if (kind := kind_of_dtype_str(dt)) is not None}
    declared_kinds = {**scalar_kinds, **array_kinds}
    kind_tables = module_kind_tables(tree, sig.func_name, declared_kinds)
    EighLoopRewriter(eigh_aliases, declared_kinds, kind_tables, sig.dtypes_raw).visit(tree)
    NonFiniteNormalizer().visit(tree)
    ast.fix_missing_locations(tree)
    fn = find_function(tree, sig.func_name)
    if fn is None:
        raise ValueError(f"{numpy_py}: no function named {sig.func_name!r}")
    return tree, fn


def prepare_body(
    tree: ast.Module, fn: ast.FunctionDef, input_args: list[str], precision: str | None
) -> dict[str, ModuleConst]:
    """The pre-inlining folds; returns the module constants folded so far."""
    inlined_consts: dict[str, ModuleConst] = dict(inline_module_constants(tree, fn, input_args))
    # The harness passes only ``input_args``, so a defaulted parameter outside them is a constant.
    fold_default_args(fn, input_args)
    PruneSparseDispatch().visit(fn)
    FoldParamNoneGuard(input_args).visit(fn)
    substitute_param_aliases(fn, input_args)
    # Before inlining, like eigh: the model ``def`` must still be distinct.
    rewrite_curve_fit(tree, fn, precision)
    fold_finfo_eps(tree, precision)
    # A nested def only becomes inlinable once its parent's is flattened.
    flatten_nested_helpers(tree)
    fuse_guarded_returns(tree)
    return inlined_consts


def substitute_param_aliases(fn: ast.FunctionDef, input_args: list[str]) -> None:
    """``local = <param>`` aliases -> the parameter, so writes through the alias reach it."""
    alias_sub = SubstituteParamAliases(input_args)
    alias_sub.collect(fn)
    alias_sub.visit(fn)
    ast.fix_missing_locations(fn)


def inline_helpers(
    tree: ast.Module,
    fn: ast.FunctionDef,
    input_args: list[str],
    keep_helpers: bool,
    inlined_consts: dict[str, ModuleConst],
) -> None:
    """Inline helpers to a fixpoint (unless kept), splice ``None``-guarded ones, then redo the folds
    inlining exposes. The name counters are shared so ``__inl<N>_`` prefixes stay unique."""
    counters = InlineCounters()
    inline_regular_helpers(tree, fn, input_args, keep_helpers, inlined_consts, counters)
    for unused in range(8):
        none_guarded = collect_none_guarded_helpers(tree, fn)
        if not none_guarded:
            break
        # Every owner: a sentinel-returning helper has no ABI anywhere.
        owners = [fn] + [n for n in tree.body if isinstance(n, ast.FunctionDef) and n is not fn]
        splicer = SpliceNoneGuardedCalls(none_guarded, counters.inl)
        if not any([splicer.apply(owner) for owner in owners if owner.name not in none_guarded]):
            break
        ast.fix_missing_locations(fn)
        inline_regular_helpers(tree, fn, input_args, keep_helpers, inlined_consts, counters)
    # The last inline round can splice in fresh constant-list loops.
    unroll_const_list_loops(fn)
    ast.fix_missing_locations(fn)
    # Aliases and optional-argument guards inlining exposed (the alias fold first: it maps the
    # renamed ``__inlN_`` parameter back onto the real one the guard fold recognises).
    substitute_param_aliases(fn, input_args)
    FoldParamNoneGuard(input_args).visit(fn)
    ast.fix_missing_locations(fn)
    # After inlining, so a table referenced only inside a helper is in the body now.
    materialize_const_arrays(tree, fn, input_args)
    ast.fix_missing_locations(fn)


@dataclasses.dataclass(slots=True)
class InlineCounters:
    inl: list[int] = dataclasses.field(default_factory=lambda: [0])
    hcall: list[int] = dataclasses.field(default_factory=lambda: [0])


def inline_regular_helpers(
    tree: ast.Module,
    fn: ast.FunctionDef,
    input_args: list[str],
    keep_helpers: bool,
    inlined_consts: dict[str, ModuleConst],
    counters: InlineCounters,
) -> None:
    """Repeat until no call to an inlinable helper survives: one pass inlines one nesting level."""
    if keep_helpers:
        return
    for unused in range(64):
        helpers = collect_inlinable_helpers(tree, fn)
        if not helpers:
            break
        names = OrderedSet(helpers)
        # Unroll constant-list loops first, so per-iteration helper calls become statements, and
        # hoist multi-statement helper calls out of expressions into assignments.
        unroll_const_list_loops(fn)
        HoistMultiStmtHelpers(helpers, counters.hcall).visit(fn)
        InlineHelpers(helpers, counters.inl).visit(fn)
        ast.fix_missing_locations(fn)
        inlined_consts.update(inline_module_constants(tree, fn, input_args))
        if not any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names for n in ast.walk(fn)
        ):
            break


def resolve_structural_axes(fn: ast.FunctionDef, sig: Signature) -> None:
    """Make every axis, repeat count and slice bound a literal, fold compile-time tuples, and refuse
    what stays symbolic. A runtime axis argument instead gets one specialised body per axis."""
    scalar_names = frozenset(sig.input_args) - frozenset(sig.array_args)

    def resolve(target: ast.FunctionDef) -> None:
        FoldConstantSymbols(
            structural_constants(sig.parameters, sig.scalars, sig.shapes_raw, runtime_args=sig.input_args)
        ).apply(target)
        ast.fix_missing_locations(target)
        # expand_dims/swapaxes become indexing first, which the tuple pass can then rank.
        AxisReshapeToIndexing(rank_table(target, declared_ranks(sig.shapes_raw)), scalar_names).visit(target)
        ast.fix_missing_locations(target)
        desugar_tuples(
            target,
            int_scalars=scalar_names - frozenset(sig.float_presets),
            float_scalars=frozenset(sig.float_presets) & scalar_names,
            arrays=frozenset(sig.array_args),
            ranks=rank_table(target, declared_ranks(sig.shapes_raw)),
        )
        reject_symbolic_axis(target)
        reject_unsupported_slices(target)

    dispatch = runtime_axis_dispatch(fn, scalar_names, rank_table(fn, declared_ranks(sig.shapes_raw)))
    if dispatch is None:
        resolve(fn)
    else:
        specialize_runtime_axis(fn, dispatch[0], dispatch[1], frozenset(sig.input_args), resolve)


def input_array_shapes_(sig: Signature, legacy_shapes: dict[str, str]) -> dict[str, str]:
    """Declared (else ``initialize()``-harvested) shape of every array argument that has one."""
    out: dict[str, str] = {}
    for a in sig.array_args:
        s = sig.shapes_raw.get(a)
        if s is None:
            s = legacy_shapes.get(a)
        if s is not None:
            out[a] = s
    return out


def resolve_shape_reads_(fn: ast.FunctionDef, sig: Signature, input_array_shapes: dict[str, str]) -> None:
    """Every ``x.shape[k]`` becomes the declared extent (the emitted kernel has no descriptor to read
    one from). Runs after the tuple fold, so a whole ``x.shape`` is per-axis subscripts already."""
    # Only the shape half of these descriptors is read.
    shape_env = {
        n: ArrayDesc(name=n, dtype="float64", shape=parse_shape_expression(s), is_output=n in sig.output_args)
        for n, s in input_array_shapes.items()
    }
    fold_dtype_aliases(fn)
    fold_list_accumulators(fn)
    resolve_shape_reads(fn, shape_env)


def promote_returns(
    fn: ast.FunctionDef, sig: Signature, input_array_shapes: dict[str, str]
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """A kernel returning its outputs gets them as output arrays when their shapes derive; a
    scalar-only return becomes a 1-element output. Otherwise the return is left as it was."""
    returned_outputs, revert_return = synthesize_return_temps(fn)
    if not returned_outputs or any(o in sig.input_args for o in returned_outputs):
        revert_return()
        return {}, {}
    returned_shapes, returned_dtypes = derive_returned_array_metadata(
        fn, returned_outputs, seed_shapes=input_array_shapes
    )
    if all(o in returned_shapes for o in returned_outputs):
        for out in returned_outputs:
            sig.input_args.append(out)
            if out not in sig.array_args:
                sig.array_args.append(out)
            if out not in sig.output_args:
                sig.output_args.append(out)
        strip_trailing_return(fn)
        ast.fix_missing_locations(fn)
        return returned_shapes, returned_dtypes
    if not returned_shapes and not sig.output_args:
        for out in promote_scalar_returns(fn, returned_outputs):
            sig.input_args.append(out)
            sig.array_args.append(out)
            sig.output_args.append(out)
            # Parsed later by _parse_shape_expression into the ``('1',)`` dim tuple.
            sig.shapes_raw[out] = "(1,)"
        ast.fix_missing_locations(fn)
        return returned_shapes, returned_dtypes
    revert_return()
    return {}, {}


@dataclasses.dataclass(slots=True, kw_only=True)
class ArgumentSources:
    """Everything an argument's descriptor is read from."""

    fn: ast.FunctionDef
    sig: Signature
    bench_info: pathlib.Path
    sparse: frozenset[str]
    legacy_shapes: dict[str, str]
    #: ``initialize()``-harvested dtypes with the declared ones on top.
    legacy_dtypes: dict[str, str]
    returned_shapes: dict[str, tuple[str, ...]]
    returned_dtypes: dict[str, str]


def declare_arguments(src: ArgumentSources) -> tuple[list[SymbolDesc], list[ArrayDesc], list[ScalarDesc]]:
    """One descriptor per (non-sparse) input argument: array, integer symbol, or scalar."""
    sig = src.sig
    symbols: list[SymbolDesc] = []
    arrays: list[ArrayDesc] = []
    scalars: list[ScalarDesc] = []
    fallback_shape = fallback_shape_for_legacy(sig.preset_symbols)
    index_names = declared_index_arrays(sig.init_block)
    int_names = names_used_as_int(src.fn)
    for arg in sig.input_args:
        if arg in src.sparse:
            continue  # expanded into its physical buffers by the caller
        if arg in sig.array_args:
            arrays.append(array_desc(arg, src, index_names, fallback_shape))
        elif arg in sig.preset_symbols and arg not in sig.float_presets and arg not in sig.bool_presets:
            declared_dt = src.legacy_dtypes.get(arg)
            # A symbol is an integer by construction; a declared non-integer dtype must not narrow it.
            dtype = declared_dt if declared_dt and dtypes.is_integer(declared_dt) else "int64"
            symbols.append(SymbolDesc(name=arg, dtype=dtype, assumption=sig.sign_of(arg)))
        elif arg in sig.bool_presets:
            scalars.append(ScalarDesc(name=arg, dtype="bool", is_output=arg in sig.output_args))
        else:
            scalars.append(scalar_desc(arg, src, arrays, int_names))
    return symbols, arrays, scalars


def array_desc(arg: str, src: ArgumentSources, index_names: set[str], fallback_shape: str | None) -> ArrayDesc:
    sig = src.sig
    if arg in src.returned_shapes:
        # A returned output: shape and dtype come from its assignment, not from the manifest.
        return ArrayDesc(
            name=arg,
            dtype=src.returned_dtypes.get(arg, default_array_dtype()),
            shape=src.returned_shapes[arg],
            is_output=True,
            is_index=arg in index_names,
        )
    shape_expr = sig.shapes_raw.get(arg)
    if shape_expr is None:
        shape_expr = src.legacy_shapes.get(arg)
    if shape_expr is None:
        if fallback_shape is None:
            raise ValueError(
                f"{src.bench_info}: array {arg!r} has no shape expression in init.shapes and no inferrable size symbol"
            )
        shape_expr = fallback_shape
    return ArrayDesc(
        name=arg,
        dtype=src.legacy_dtypes.get(arg, default_array_dtype()),
        shape=parse_shape_expression(shape_expr),
        is_output=arg in sig.output_args,
        is_index=arg in index_names,
    )


def scalar_desc(arg: str, src: ArgumentSources, arrays: list[ArrayDesc], int_names: set[str]) -> ScalarDesc:
    """A plain scalar: declared dtype, else inferred from its ``init.scalars`` default; a float one
    used in an integer position or naming an array dimension is an ``int`` (plain int, the shape
    symbols' kind, since Fortran rejects mixed-kind integer arithmetic)."""
    sig = src.sig
    inferred_dt = src.legacy_dtypes.get(arg) or infer_scalar_dtype(sig.scalars.get(arg))
    is_array_dim = any(re.search(rf"\b{re.escape(arg)}\b", str(tok)) for a in arrays for tok in a.shape)
    if inferred_dt in {"float64", "double", "float32"} and (arg in int_names or is_array_dim):
        inferred_dt = "int"
    return ScalarDesc(name=arg, dtype=inferred_dt, is_output=arg in sig.output_args, value=sig.scalars.get(arg))
