"""The expander registry: ``(module, name)`` of a numpy call -> its expander."""

import ast
import copy
from collections.abc import Callable

from hpcagent_bench.translators.numpyto_common.lib_nodes.axes import (
    expand_expand_dims,
    expand_flip,
    expand_moveaxis,
    expand_roll,
    expand_squeeze,
    expand_swapaxes,
    expand_take,
    expand_transpose,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.blas import expand_add_outer, expand_dot_2d, expand_outer
from hpcagent_bench.translators.numpyto_common.lib_nodes.constructors import (
    expand_arange,
    expand_copy,
    expand_eye,
    expand_fromfunction,
    expand_linspace,
    expand_meshgrid,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.contractions import (
    expand_einsum,
    expand_inner,
    expand_tensordot,
    expand_vdot,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.elementwise import (
    UNARY_C_MATH,
    binary_call_expander,
    unary_call_expander,
    unary_expr_expander,
    expand_add,
    expand_clip,
    expand_cos_arr,
    expand_divide,
    expand_equal,
    expand_exp_arr,
    expand_greater,
    expand_greater_equal,
    expand_less,
    expand_less_equal,
    expand_log_arr,
    expand_logical_and,
    expand_logical_not,
    expand_logical_or,
    expand_maximum,
    expand_minimum,
    expand_multiply,
    expand_negative,
    expand_not_equal,
    expand_power,
    expand_sin_arr,
    expand_sqrt_arr,
    expand_subtract,
    expand_tanh,
    expand_where,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.fft import (
    expand_fft,
    expand_fftfreq,
    expand_fftn,
    expand_ifft,
    expand_ifftn,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import REDUCTION_NAMES, const_
from hpcagent_bench.translators.numpyto_common.lib_nodes.histograms import expand_bincount, expand_histogram
from hpcagent_bench.translators.numpyto_common.lib_nodes.joins import expand_concatenate, expand_hstack, expand_stack
from hpcagent_bench.translators.numpyto_common.lib_nodes.linalg import (
    expand_cholesky,
    expand_linalg_det,
    expand_linalg_inv,
    expand_linalg_norm,
    expand_linalg_solve,
    expand_lstsq,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.pad import expand_pad
from hpcagent_bench.translators.numpyto_common.lib_nodes.reductions import (
    expand_all,
    expand_any,
    expand_argmax,
    expand_argmin,
    expand_count_nonzero,
    expand_max,
    expand_mean,
    expand_min,
    expand_prod,
    expand_std,
    expand_sum,
    expand_var,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.repeat import expand_repeat
from hpcagent_bench.translators.numpyto_common.lib_nodes.reshape import expand_reshape
from hpcagent_bench.translators.numpyto_common.lib_nodes.scans import (
    expand_cummax,
    expand_cummin,
    expand_cumprod,
    expand_cumsum,
    expand_diff,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.sorting import expand_median, expand_searchsorted, expand_sort
from hpcagent_bench.translators.numpyto_common.lib_nodes.triangular import (
    expand_diag,
    expand_diagonal,
    expand_trace,
    expand_tril,
    expand_triu,
)

#: Map of ``("np", attr) -> expander``. The expander signature is
#: ``(assign_target, call_args, shape_table) -> list[stmt]``.
NP_CALL_EXPANDERS: dict[tuple[str, str], Callable] = {
    # Reductions
    ("np", "sum"): expand_sum,
    # np.maximum/minimum.accumulate: a DaCe Scan re-emits cummax/cummin as these
    # (there's no np.cummax). ufunc.reduce forms are normalized to np.sum/...
    # upstream in native_desugar (_UfuncReduceToReducer), so they never reach here.
    ("np", "searchsorted"): expand_searchsorted,
    ("np", "maximum.accumulate"): expand_cummax,
    ("np", "minimum.accumulate"): expand_cummin,
    ("np", "sort"): expand_sort,
    ("np", "max"): expand_max,
    ("np", "min"): expand_min,
    ("np", "mean"): expand_mean,
    ("np", "prod"): expand_prod,
    ("np", "std"): expand_std,
    # Linear algebra
    ("np", "dot"): expand_dot_2d,
    ("np", "einsum"): expand_einsum,
    ("np", "tensordot"): expand_tensordot,
    ("np", "inner"): expand_inner,
    ("np", "vdot"): expand_vdot,
    ("np", "trace"): expand_trace,
    ("np", "diagonal"): expand_diagonal,
    ("np", "diag"): expand_diag,
    ("np", "cumsum"): expand_cumsum,
    ("np", "cumprod"): expand_cumprod,
    ("np", "median"): expand_median,
    ("np", "roll"): expand_roll,
    ("np", "tril"): expand_tril,
    ("np", "pad"): expand_pad,
    ("np", "outer"): expand_outer,
    ("np", "add.outer"): expand_add_outer,
    ("np", "transpose"): expand_transpose,
    ("np", "linalg.cholesky"): expand_cholesky,
    ("np", "linalg.norm"): expand_linalg_norm,
    ("np", "linalg.lstsq"): expand_lstsq,
    ("np", "linalg.inv"): expand_linalg_inv,
    ("np", "linalg.det"): expand_linalg_det,
    ("np", "linalg.solve"): expand_linalg_solve,
    ("np", "fft.fftn"): expand_fftn,
    ("np", "fft.ifftn"): expand_ifftn,
    ("np", "fft.fft"): expand_fft,
    ("np", "fft.ifft"): expand_ifft,
    ("np", "fft.fftfreq"): expand_fftfreq,
    ("np", "var"): expand_var,
    ("np", "argmax"): expand_argmax,
    ("np", "argmin"): expand_argmin,
    ("np", "any"): expand_any,
    ("np", "all"): expand_all,
    ("np", "count_nonzero"): expand_count_nonzero,
    # Memory / shape
    ("np", "copy"): expand_copy,
    # np.asarray/np.ascontiguousarray of an already-materialised array is a copy
    # (contiguity/dtype already hold for our buffers), so they lower exactly
    # like np.copy (dbcsr/minife pass inputs through np.asarray before indexing).
    ("np", "asarray"): expand_copy,
    ("np", "ascontiguousarray"): expand_copy,
    # ``np.array(<array expr>)`` is the same materialising copy. The literal-list and 0-d scalar
    # forms never reach here: the frontend rewrites both before lowering runs.
    ("np", "array"): expand_copy,
    ("np", "bincount"): expand_bincount,
    ("np", "reshape"): expand_reshape,
    ("np", "swapaxes"): expand_swapaxes,
    ("np", "moveaxis"): expand_moveaxis,
    ("np", "expand_dims"): expand_expand_dims,
    ("np", "squeeze"): expand_squeeze,
    ("np", "take"): expand_take,
    ("np", "repeat"): expand_repeat,
    ("np", "eye"): expand_eye,
    ("np", "meshgrid"): expand_meshgrid,
    ("np", "triu"): expand_triu,
    ("np", "hstack"): expand_hstack,
    ("np", "concatenate"): expand_concatenate,
    ("np", "stack"): expand_stack,
    ("np", "diff"): expand_diff,
    ("np", "flip"): expand_flip,
    ("np", "linspace"): expand_linspace,
    ("np", "arange"): expand_arange,
    ("np", "fromfunction"): expand_fromfunction,
    ("np", "histogram"): expand_histogram,
    # Elementwise
    ("np", "minimum"): expand_minimum,
    ("np", "maximum"): expand_maximum,
    ("np", "add"): expand_add,
    ("np", "multiply"): expand_multiply,
    # Comparison ops -> per-element Compare (boolean output array).
    ("np", "less"): expand_less,
    ("np", "less_equal"): expand_less_equal,
    ("np", "greater"): expand_greater,
    ("np", "greater_equal"): expand_greater_equal,
    ("np", "equal"): expand_equal,
    ("np", "not_equal"): expand_not_equal,
    # Logical ops -> per-element BoolOp / UnaryOp.
    ("np", "logical_and"): expand_logical_and,
    ("np", "logical_or"): expand_logical_or,
    ("np", "logical_not"): expand_logical_not,
    ("np", "subtract"): expand_subtract,
    ("np", "divide"): expand_divide,
    ("np", "true_divide"): expand_divide,
    ("np", "negative"): expand_negative,
    ("np", "power"): expand_power,
    ("np", "tanh"): expand_tanh,
    ("np", "clip"): expand_clip,
    ("np", "where"): expand_where,
    # Per-element math intrinsics. These also live in MATH_BUILTINS for
    # the scalar-arg form; the call-hoister catches the array form first.
    ("np", "exp"): expand_exp_arr,
    ("np", "log"): expand_log_arr,
    ("np", "sqrt"): expand_sqrt_arr,
    ("np", "sin"): expand_sin_arr,
    ("np", "cos"): expand_cos_arr,
}
for np_name, c_name_ in UNARY_C_MATH.items():
    NP_CALL_EXPANDERS[("np", np_name)] = unary_call_expander(c_name_)

# Inline-expression ufuncs valid in both C and Fortran: square -> x*x,
# reciprocal -> 1.0/x, degrees/radians via the exact double conversion factor
# (180/pi, pi/180) as a plain numeric literal -- no M_PI (C-only), no
# per-language divergence.
DEG_PER_RAD = 57.29577951308232  # 180 / pi
RAD_PER_DEG = 0.017453292519943295  # pi / 180
NP_CALL_EXPANDERS[("np", "square")] = unary_expr_expander(
    lambda x: ast.BinOp(left=copy.deepcopy(x), op=ast.Mult(), right=copy.deepcopy(x))
)
NP_CALL_EXPANDERS[("np", "reciprocal")] = unary_expr_expander(
    lambda x: ast.BinOp(left=const_(1.0), op=ast.Div(), right=x)
)
NP_CALL_EXPANDERS[("np", "degrees")] = unary_expr_expander(
    lambda x: ast.BinOp(left=x, op=ast.Mult(), right=const_(DEG_PER_RAD))
)
NP_CALL_EXPANDERS[("np", "rad2deg")] = NP_CALL_EXPANDERS[("np", "degrees")]
NP_CALL_EXPANDERS[("np", "radians")] = unary_expr_expander(
    lambda x: ast.BinOp(left=x, op=ast.Mult(), right=const_(RAD_PER_DEG))
)
NP_CALL_EXPANDERS[("np", "deg2rad")] = NP_CALL_EXPANDERS[("np", "radians")]
# ``sign`` has no both-language inline form (C bool arithmetic vs Fortran
# logicals), so emit a ``__npb_sign(x)`` marker each backend specialises in its
# own _emit_call. Kept out of the promotion pass via the math intrinsic name set.
NP_CALL_EXPANDERS[("np", "sign")] = unary_call_expander("__npb_sign")


for np_name, c_name_ in {
    "arctan2": "atan2",
    "hypot": "hypot",
    "copysign": "copysign",
    "fmod": "fmod",
    "fmax": "fmax",
    "fmin": "fmin",
}.items():
    NP_CALL_EXPANDERS[("np", np_name)] = binary_call_expander(c_name_)


#: Expander keys that accept a partial-slice assignment target
#: (``row_offsets[1:] = np.cumsum(m_sizes)``) in addition to a bare Name. The
#: full-slice form is canonicalised to a Name in ``visit_Assign``; only a
#: shifted slice reaches here, and only the cumulative scans honour the
#: lower-bound offset (via :func:`scan_target_offsets`).
#: Registered ops whose result shape is NOT the broadcast of its operands -- reductions,
#: constructors, shape-changers and contractions. Everything else that has an expander IS
#: elementwise, which is how :data:`ELEMENTWISE_SHAPE_OPS` below is derived.
#:
#: Derived, not restated, and computed AFTER every registration: the previous hand-written list was
#: missing `square`, `reciprocal`, `degrees`, `radians`, `asarray`, the comparison ufuncs and the
#: binary libm ufuncs, and each omission made ``derive_output_shape`` return None -- which does not
#: fail, it silently DECLINES to hoist and leaves a bare ``np.square(...)`` for the emitter.
NON_ELEMENTWISE_SHAPE_OPS: set[str] = set(REDUCTION_NAMES) | {
    "reshape",
    "repeat",
    "transpose",
    "flip",
    "roll",
    "triu",
    "tril",
    "concatenate",
    "stack",
    "pad",
    "diag",
    "diagonal",
    "diff",
    "trace",
    "einsum",
    "tensordot",
    "inner",
    "outer",
    "dot",
    "vdot",
    "matmul",
    "cumsum",
    "cumprod",
    "linspace",
    "arange",
    "zeros",
    "ones",
    "empty",
    "full",
    "eye",
    "identity",
    "fromfunction",
    "meshgrid",
    "mgrid",
    "sort",
    "argsort",
    "histogram",
    "unique",
    "take",
    "interp",
    "searchsorted",
    "nonzero",
    "kron",
    "cross",
    "split",
    "expand_dims",
    "squeeze",
    "swapaxes",
    "moveaxis",
    "ravel",
    "flatten",
    "tile",
    "broadcast_to",
    "atleast_1d",
    "atleast_2d",
    "append",
    "insert",
    "delete",
}

ELEMENTWISE_SHAPE_OPS: frozenset[str] = frozenset(
    name
    for module, name in NP_CALL_EXPANDERS
    if module == "np" and "." not in name and name not in NON_ELEMENTWISE_SHAPE_OPS
) | {"copy", "array", "where", "clip"}
