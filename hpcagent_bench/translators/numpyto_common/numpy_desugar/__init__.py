"""Desugar numpy forms the verbatim Python backends (numba / pythran / dace) cannot compile.

Entry point: :func:`desugar_for_python_backend`. Each submodule owns one family of rewrites.
"""

import ast

from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import (
    DesugarError,
    REDUCE_FNS,
    AUG_OP_SRC,
    const_int,
    eigh_alias_names,
    eigh_call_ab,
    np_submodule_attr,
)
from hpcagent_bench.translators.numpyto_common.numpy_desugar.constants import (
    DEFAULT_FOLDING_BACKENDS,
    ConstComprehensionFold,
    ListCompUnroll,
    const_name_values,
    fold_constant_helper_arguments,
    fold_finfo_eps,
    fold_kernel_defaults,
    has_defaulted_parameters,
)
from hpcagent_bench.translators.numpyto_common.numpy_desugar.counting import (
    HISTOGRAM_HOIST,
    REPEAT_AXIS_HOIST,
    AddAtInline,
    BincountInline,
    DiffToSliceDifference,
    RepeatCountsInline,
    SearchsortedMaterialize,
    StripAstypeCopyKwarg,
)
from hpcagent_bench.translators.numpyto_common.numpy_desugar.curve_fit import rewrite_curve_fit
from hpcagent_bench.translators.numpyto_common.numpy_desugar.eigh import (
    EighCallHoister,
    EighInline,
    EighLoopRewriter,
    eigh_stmts,
)
from hpcagent_bench.translators.numpyto_common.numpy_desugar.einsum import EINSUM_HOIST
from hpcagent_bench.translators.numpyto_common.numpy_desugar.fft import NATIVE_FFT_BACKENDS, FftInline
from hpcagent_bench.translators.numpyto_common.numpy_desugar.guards import (
    BoolOpIfToChain,
    DeadBranchElim,
    DropGuards,
    DropValidationGuards,
    IssubdtypeFold,
    SpliceErrstate,
)
from hpcagent_bench.translators.numpyto_common.numpy_desugar.hoist import HoistTables, ValueHoist
from hpcagent_bench.translators.numpyto_common.numpy_desugar.indexing import (
    FANCY_GATHER_HOIST,
    IxWriteToLoop,
    DecomposeRollSlice,
    FancySliceStoreToLoop,
    MaskedAssignToLoop,
    MgridInline,
)
from hpcagent_bench.translators.numpyto_common.numpy_desugar.kinds import (
    CallKinds,
    NO_CALLS,
    dtype_kind,
    dtype_table_,
    kind_of_dtype_str,
    promote_kind,
    infer_param_kinds,
    module_kind_tables,
)
from hpcagent_bench.translators.numpyto_common.numpy_desugar.linalg import (
    LINALG_HOIST,
    LINALG_LOWERABLE,
    LOWER_SOLVE_RHS_RANKS,
    NATIVE_LINALG,
)
from hpcagent_bench.translators.numpyto_common.numpy_desugar.lists import fold_list_accumulators
from hpcagent_bench.translators.numpyto_common.numpy_desugar.matmul import (
    INT_MATMUL_HOIST,
    BatchedMatmulToLoop,
    ReshapeContiguousInline,
    ReshapeMatmulInline,
    int_matmul_stmts,
    noncontig_names,
)
from hpcagent_bench.translators.numpyto_common.numpy_desugar.numba import (
    NdimFold,
    NumbaDtypeFixups,
    ReshapeFortranOrderInline,
    SliceObjectInline,
    OuterBroadcastPeel,
)
from hpcagent_bench.translators.numpyto_common.numpy_desugar.pad import PadInline
from hpcagent_bench.translators.numpyto_common.numpy_desugar.ranks import (
    infer_param_ranks,
    param_body_rank_evidence,
    agreed_param_ranks,
    expr_rank,
    extent_tokens,
    helper_return_ranks,
    name_binding_index,
    name_value_pairs,
    rank_table,
    shape_table,
)
from hpcagent_bench.translators.numpyto_common.numpy_desugar.reductions import (
    DACE_REDUCE_AXIS_HOIST,
    MASKED_REDUCE_HOIST,
    REDUCE_AXIS_HOIST,
    KeepdimsToNewaxis,
    NormalizeNegativeAxis,
    UfuncReduceToReducer,
    axis_list,
    masked_reduce_map,
    reduce_axis_stmts,
)
from hpcagent_bench.translators.numpyto_common.numpy_desugar.ssa import SsaRename
from hpcagent_bench.translators.numpyto_common.numpy_desugar.ufuncs import (
    UFUNC_OUTER_HOIST,
    CallFixups,
    ComplexAccessorToFunc,
    ElementalUfuncToPrimitive,
    FillDiagonalInline,
    UfuncOutInline,
)

__all__ = [
    "CallKinds",
    "DesugarError",
    "IxWriteToLoop",
    "NO_CALLS",
    "REDUCE_FNS",
    "AUG_OP_SRC",
    "AddAtInline",
    "BincountInline",
    "ComplexAccessorToFunc",
    "DecomposeRollSlice",
    "DiffToSliceDifference",
    "DropValidationGuards",
    "EighCallHoister",
    "EighLoopRewriter",
    "ElementalUfuncToPrimitive",
    "FillDiagonalInline",
    "NormalizeNegativeAxis",
    "RepeatCountsInline",
    "SpliceErrstate",
    "StripAstypeCopyKwarg",
    "UfuncOutInline",
    "UfuncReduceToReducer",
    "axis_list",
    "const_int",
    "dtype_kind",
    "dtype_table_",
    "eigh_alias_names",
    "eigh_call_ab",
    "eigh_stmts",
    "int_matmul_stmts",
    "kind_of_dtype_str",
    "param_body_rank_evidence",
    "promote_kind",
    "reduce_axis_stmts",
    "desugar_for_python_backend",
    "expr_rank",
    "extent_tokens",
    "fold_finfo_eps",
    "fold_list_accumulators",
    "module_kind_tables",
    "name_binding_index",
    "name_value_pairs",
    "np_submodule_attr",
    "rank_table",
    "rewrite_curve_fit",
    "shape_table",
]


def desugar_for_python_backend(source: str, kir, backend: str | None = None) -> str:
    """Rewrite ``source`` so numba/pythran/dace can compile it: expand numpy
    ops they don't support (batched ``@``/``np.matmul``, ``np.pad``,
    ``np.einsum``, ``np.fft.*``, ``np.mgrid``, axis reductions, ufunc.outer,
    multi-array fancy gather, ``np.ix_`` writes, 2-D boolean-mask assignment,
    ``np.ndarray``/``np.linspace(dtype=)``/``abs(array)``) into plain loops/
    broadcasts/``np.where``, fold constant comprehensions to literals, unroll a
    comprehension over a constant iterable, and SSA-rename a reassigned local
    (dace refuses both). EVERY function in the module is processed (helpers too --
    nbody's masked updates live in getAcc/getEnergy), each with its own rank
    table seeded from the kernel arrays (kir) or inferred call-site param
    ranks. Every pass is pattern-guarded; ``source`` returns byte-for-byte
    unchanged when none fire.

    ``backend`` selects the target's native-``np.linalg`` capability: an op it
    implements natively (numba/dace do cholesky/solve/inv) is left verbatim;
    one it lacks (pythran has no np.linalg) is lowered to explicit loops.
    ``None`` (default) lowers no linalg -- the safe backwards-compatible base.
    A capability can also be PARTIAL: :data:`LOWER_SOLVE_RHS_RANKS` lowers the
    ``solve`` right-hand-side ranks a nominally-native backend cannot expand."""
    lower_linalg = LINALG_LOWERABLE - NATIVE_LINALG.get(backend, LINALG_LOWERABLE)
    lower_solve_rhs_ranks = LOWER_SOLVE_RHS_RANKS.get(backend, frozenset())
    tree = ast.parse(source)
    changed = False
    kernel = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == kir.kernel_name), None)
    if backend in DEFAULT_FOLDING_BACKENDS and kernel is not None and has_defaulted_parameters(kernel):
        changed = fold_kernel_defaults(kernel, kir.input_args)
    all_funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if backend == "numba":
        changed = fold_constant_helper_arguments(tree, kir.kernel_name) or changed
        for fn in all_funcs:
            inline = SliceObjectInline(fn)
            inline.visit(fn)
            changed = changed or inline.changed
    kir_seed: dict[str, int] = {a.name: len(a.shape) for a in kir.arrays}
    kir_dtype_seed: dict[str, str] = {
        a.name: kind_of_dtype_str(vars(a).get("dtype")) for a in kir.arrays if kind_of_dtype_str(vars(a).get("dtype"))
    }
    # Concrete (not just KIND) dtypes, for passes that need an exact width -- e.g. _FftInline's
    # phase-divisor cast, which must match complex64 vs complex128, not just "is complex".
    # Read through ``vars()`` like the kind seed above: a KIR array carries no dtype at all in the
    # rank-only callers, and a direct attribute read makes the whole desugar raise for every backend.
    kir_array_dtypes: dict[str, str] = {a.name: vars(a)["dtype"] for a in kir.arrays if "dtype" in vars(a)}
    param_ranks = infer_param_ranks(all_funcs, kir.kernel_name, kir_seed)
    param_kinds = infer_param_kinds(all_funcs, kir.kernel_name, kir_dtype_seed) if backend == "numba" else {}
    return_ranks = (
        helper_return_ranks(all_funcs, param_ranks, kir.kernel_name, kir_seed) if backend == "numba" else None
    )
    agreed_ranks = (
        agreed_param_ranks(all_funcs, kir.kernel_name, param_ranks, kir_seed, return_ranks)
        if return_ranks is not None
        else {}
    )
    eigh_aliases = eigh_alias_names(tree)
    for fn in all_funcs or [tree]:
        is_kernel = vars(fn).get("name") == kir.kernel_name
        seed = dict(param_ranks.get(vars(fn).get("name"), {}))
        if is_kernel:
            seed.update(kir_seed)
        if return_ranks is not None and isinstance(fn, ast.FunctionDef):
            fold = NdimFold(fn, agreed_ranks.get(fn.name, {}))
            fold.visit(fn)
            changed = changed or fold.changed
        ranks = rank_table(fn, seed, call_returns=return_ranks)
        dtypes = dtype_table_(fn, kir_dtype_seed if is_kernel else {})
        noncontig = noncontig_names(fn)
        masked_gathers = masked_reduce_map(fn, ranks, dtypes)
        consts = const_name_values(fn)
        tables = HoistTables(ranks, dtypes, masked_gathers, lower_linalg, lower_solve_rhs_ranks)
        passes = [
            DropGuards(),
            # First: it splices statements OUT of a ``with`` body, and every pass below walks only
            # this scope's top-level statements.
            SpliceErrstate(),
            ConstComprehensionFold(consts),
            ListCompUnroll(consts),
            # Before every temp-minting pass below: an ``or`` clones its if-body, and two
            # clones sharing one hoisted temp name would redeclare it per branch. After the
            # const folds, whose "bound exactly once" table the clones would otherwise stale.
            BoolOpIfToChain(),
            NormalizeNegativeAxis(ranks),
            IxWriteToLoop(ranks, dtypes, fn),
            FancySliceStoreToLoop(ranks, dtypes),
            EighInline(ranks, eigh_aliases, dtypes, kir_array_dtypes),
            ValueHoist(LINALG_HOIST, tables),
            ReshapeMatmulInline(ranks),
            BatchedMatmulToLoop(ranks),
            PadInline(ranks, lower_symbolic_constant=backend == "dace"),
            ValueHoist(EINSUM_HOIST, tables),
            *([] if backend in NATIVE_FFT_BACKENDS else [FftInline(ranks, kir_array_dtypes)]),
            MgridInline(),
            ValueHoist(FANCY_GATHER_HOIST, tables),
            ValueHoist(DACE_REDUCE_AXIS_HOIST if backend == "dace" else REDUCE_AXIS_HOIST, tables),
            # Directly behind it: takes only the keepdims reductions the loop lowering
            # declined (an operand whose rank the table had to forget).
            KeepdimsToNewaxis(),
            ValueHoist(MASKED_REDUCE_HOIST, tables),
            CallFixups(ranks),
            IssubdtypeFold(dtypes),
            DeadBranchElim(),
            ValueHoist(UFUNC_OUTER_HOIST, tables),
            MaskedAssignToLoop(ranks, dtypes),
            AddAtInline(ranks),
            SearchsortedMaterialize(),
            # Bincount BEFORE the diff rewrite: the repeat lowering below reads ``np.diff(p)``
            # structurally to prove its output length telescopes.
            StripAstypeCopyKwarg(),
            RepeatCountsInline(ranks),
            BincountInline(ranks),
            # LAST of the three: the repeat lowering above reads ``np.diff(p)`` structurally, so the
            # slice rewrite has to come after it.
            DiffToSliceDifference(),
            ValueHoist(HISTOGRAM_HOIST, tables),
            ValueHoist(REPEAT_AXIS_HOIST, tables),
            ReshapeContiguousInline(noncontig),
            # numba only: its reshape takes no ``order=`` and its ``@`` / dtype typing is strict.
            *(
                [
                    ReshapeFortranOrderInline(),
                    NumbaDtypeFixups(dtype_table_(fn, param_kinds.get(vars(fn).get("name"), {}))),
                ]
                if backend == "numba"
                else []
            ),
            ValueHoist(INT_MATMUL_HOIST, tables),
            ComplexAccessorToFunc(conjugate_only=True),
            ElementalUfuncToPrimitive(),
            # numba only, and LAST of the rewrites: it peels an outer-product broadcast into a
            # loop, and a pass running after it would have to see through the loop to match.
            *([OuterBroadcastPeel(ranks)] if backend == "numba" else []),
            # LAST: every pass above matches BY NAME through a table built before the loop
            # (ranks / dtypes / consts / noncontig / masked_gathers), so a rename ahead of
            # them turns every lookup into a miss and silently switches those passes off.
            SsaRename(fn, set(kir_seed)),
        ]
        for p in passes:
            # Process THIS scope's own statements only; a nested def is its own
            # scope (its params carry different ranks) and is handled as its own
            # entry in ``all_funcs``, so skip it here to avoid a wrong-rank pass.
            new_body = []
            for stmt in fn.body:
                if isinstance(stmt, ast.FunctionDef):
                    new_body.append(stmt)
                    continue
                res = p.visit(stmt)
                if res is None:
                    continue
                new_body.extend(res if isinstance(res, list) else [res])
            fn.body = new_body
        changed = changed or any(p.changed for p in passes)
    if not changed:
        return source  # nothing matched -> leave the body verbatim
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
