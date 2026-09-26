"""Lowering: the numpy-numeric subset rewritten into plain loops, ready for backend emission.

:func:`lower` runs the ordered phases of :mod:`pipeline` over a copy of a
:class:`~numpyto_common.ir.KernelIR`: call and constructor normalisation, library-node
expansion (:mod:`numpyto_common.lib_nodes`), whole-array and slice fusion, and signature fixes.
Slice fusion turns ``A[1:N-1] = (B[:N-2] + B[2:] + B[1:N-1]) / 3.0`` into ONE loop over
``range(1, N-1)`` that reads ``B[i-1]``/``B[i+1]``/``B[i]``, not four temporary arrays.
"""

from hpcagent_bench.translators.numpyto_common.lowering.calls import (
    AstypeRewriter,
    ConditionalNoneAllocRewriter,
    MatmulCallRewriter,
    NpAliasRewriter,
    ReshapeMethodRewriter,
    TransposeRewriter,
)
from hpcagent_bench.translators.numpyto_common.lowering.casts import BuiltinCastRewriter
from hpcagent_bench.translators.numpyto_common.lowering.chains import ChainedSubscriptFlattener
from hpcagent_bench.translators.numpyto_common.lowering.complex import walk_complex
from hpcagent_bench.translators.numpyto_common.lowering.constructors import EyeToZerosDiagonal, FullCallHoister
from hpcagent_bench.translators.numpyto_common.lowering.hoisting import MethodCallRewriter
from hpcagent_bench.translators.numpyto_common.lowering.mathfuncs import MATH_INTRINSIC_NAMES, MathRewriter
from hpcagent_bench.translators.numpyto_common.lowering.pipeline import (
    INVARIANT_ENV,
    LoweringContext,
    assert_lowering_invariants,
    lower,
)
from hpcagent_bench.translators.numpyto_common.lowering.scatter import ScatterAtRewriter
from hpcagent_bench.translators.numpyto_common.lowering.shape_reads import (
    is_newaxis_result_axis,
    ShapeMidExpressionRewriter,
)
from hpcagent_bench.translators.numpyto_common.lowering.signature import (
    BUILTIN_NAMES,
    promote_free_names_to_params,
    helper_returns_int,
    integer_valued_locals,
)
from hpcagent_bench.translators.numpyto_common.lowering.slice_fusion import INVARIANT_SELF_READ_PREFIX, SliceFusion
from hpcagent_bench.translators.numpyto_common.lowering.slice_scalarize import SliceToScalarRewriter
from hpcagent_bench.translators.numpyto_common.lowering.ssa import ssa_rename_reassigned
from hpcagent_bench.translators.numpyto_common.lowering.subscriptify import SubscriptifyNames
from hpcagent_bench.translators.numpyto_common.lowering.tuples import ShapeTableTupleSplit, TupleLocalPropagator
from hpcagent_bench.translators.numpyto_common.lowering.views import (
    EllipsisExpander,
    fold_slice_view_aliases,
    PadImplicitTrailingSlices,
)
from hpcagent_bench.translators.numpyto_common.lowering.whole_array import WholeArrayAssignRewriter

__all__ = [
    "ChainedSubscriptFlattener",
    "INVARIANT_SELF_READ_PREFIX",
    "LoweringContext",
    "ShapeTableTupleSplit",
    "SliceFusion",
    "AstypeRewriter",
    "BUILTIN_NAMES",
    "BuiltinCastRewriter",
    "ConditionalNoneAllocRewriter",
    "EllipsisExpander",
    "EyeToZerosDiagonal",
    "FullCallHoister",
    "INVARIANT_ENV",
    "MATH_INTRINSIC_NAMES",
    "MathRewriter",
    "MatmulCallRewriter",
    "MethodCallRewriter",
    "NpAliasRewriter",
    "PadImplicitTrailingSlices",
    "ReshapeMethodRewriter",
    "ScatterAtRewriter",
    "ShapeMidExpressionRewriter",
    "SliceToScalarRewriter",
    "SubscriptifyNames",
    "TransposeRewriter",
    "TupleLocalPropagator",
    "WholeArrayAssignRewriter",
    "assert_lowering_invariants",
    "fold_slice_view_aliases",
    "is_newaxis_result_axis",
    "promote_free_names_to_params",
    "ssa_rename_reassigned",
    "walk_complex",
    "helper_returns_int",
    "integer_valued_locals",
    "lower",
]
