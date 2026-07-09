"""numpy-faithful semantics fixes in the SHARED translator lowering.

Two tiers:

* AST / source-level UNIT tests (fast, deterministic) that pin the exact loop /
  cast / dtype an expander or rank rule produces -- so a regression points
  straight at the cause.
* End-to-end ORACLE tests (``run_op`` / ``run_return_op``) that compile + run the
  kernel on every backend and compare bit-exact to numpy.

Covers: NaN-propagating max/min/argmax/argmin, int64 sum/prod accumulators,
zero-size reduction refusal, std/var ddof, sum/prod/max/min ``initial=``,
mean/std/var float-dtype preservation, concatenate negative axis, elementwise
broadcast of BOTH operands, symmetric size-1 broadcast, ellipsis subscript rank,
and int/int true division.
"""
import ast

import numpy as np
import pytest

from _op_oracle import run_op
from numpyto_common.lib_nodes import (_broadcast_extents, _const_or_name, expand_max, expand_min, expand_prod,
                                      expand_std, expand_sum)
from numpyto_common.numpy_desugar import _expr_rank, _reduce_axis_stmts

_ALL = ("c", "cpp", "fortran", "numba", "pythran", "jax")


def _ok(res):
    return all(v == "ok" or v.startswith("skip") for v in res.values()), res


def _call_args(src):
    call = ast.parse(src, mode="eval").body
    return call.args, call.keywords


def _target(name="out"):
    return ast.Name(id=name, ctx=ast.Store())


def _src(stmts):
    mod = ast.fix_missing_locations(ast.Module(body=list(stmts), type_ignores=[]))
    return ast.unparse(mod)


# --------------------------------------------------------------------------- #
# Fix 2: integer sum/prod accumulate in int64 (not a float 0.0 / 1.0)         #
# --------------------------------------------------------------------------- #


def test_sum_integer_uses_int_accumulator_and_int64_dtype():
    args, kws = _call_args("np.sum(a)")
    ld = {"a": "int64"}
    stmts = expand_sum(_target(), args, {"a": ("N", )}, kws, local_dtypes=ld)
    init = stmts[0]
    assert isinstance(init.value, ast.Constant)
    assert init.value.value == 0 and isinstance(init.value.value, int)  # not 0.0
    assert ld["out"] == "int64"  # result upcast to int64 (numpy rule)


def test_prod_integer_uses_int_accumulator():
    args, kws = _call_args("np.prod(a)")
    ld = {"a": "int32"}
    stmts = expand_prod(_target(), args, {"a": ("N", )}, kws, local_dtypes=ld)
    init = stmts[0]
    assert init.value.value == 1 and isinstance(init.value.value, int)
    assert ld["out"] == "int64"


def test_sum_float_keeps_float_accumulator():
    args, kws = _call_args("np.sum(a)")
    ld = {"a": "float64"}
    stmts = expand_sum(_target(), args, {"a": ("N", )}, kws, local_dtypes=ld)
    assert isinstance(stmts[0].value.value, float)  # 0.0
    assert "out" not in ld


def test_integer_sum_prod_numeric_all_backends():
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    src = "import numpy as np\ndef f(a, s, p):\n s[0] = np.sum(a)\n p[0] = np.prod(a)\n"
    res = run_op(src,
                 "f", {"a": a}, {
                     "s": (1, ),
                     "p": (1, )
                 }, {"N": 5},
                 shapes={
                     "a": "(N,)",
                     "s": "(1,)",
                     "p": "(1,)"
                 },
                 dtypes={
                     "a": "int64",
                     "s": "int64",
                     "p": "int64"
                 },
                 backends=_ALL)
    ok, _ = _ok(res)
    assert ok, res


# --------------------------------------------------------------------------- #
# Fix 1: NaN propagation in max / min / argmax / argmin                       #
# --------------------------------------------------------------------------- #


def test_max_emits_nan_test():
    args, kws = _call_args("np.max(a)")
    stmts = expand_max(_target(), args, {"a": ("N", )}, kws)
    # A NaN element must win: the update tests ``x != x`` (NaN self-inequality).
    assert "!=" in _src(stmts)


def test_min_emits_nan_test():
    args, kws = _call_args("np.min(a)")
    stmts = expand_min(_target(), args, {"a": ("N", )}, kws)
    assert "!=" in _src(stmts)


def test_max_min_nan_propagation_all_backends():
    a = np.array([1.0, np.nan, 2.0, -3.0])
    src = "import numpy as np\ndef f(a, mx, mn):\n mx[0] = np.max(a)\n mn[0] = np.min(a)\n"
    # pythran runs the full reduction natively, and its runtime ``np.max`` /
    # ``np.min`` SUPPRESS NaN (verified: returns 2.0 for this input) -- a pythran
    # limitation in code the shared translator does not own. The C / C++ /
    # Fortran path (which this fix targets) plus numba / jax propagate NaN.
    res = run_op(src,
                 "f", {"a": a}, {
                     "mx": (1, ),
                     "mn": (1, )
                 }, {"N": 4},
                 shapes={
                     "a": "(N,)",
                     "mx": "(1,)",
                     "mn": "(1,)"
                 },
                 backends=_ALL,
                 skip_backends={"pythran": "pythran np.max/np.min suppress NaN (runtime limitation)"})
    ok, _ = _ok(res)
    assert ok, res


def test_argmax_argmin_first_nan_index_all_backends():
    a = np.array([1.0, 5.0, np.nan, 2.0, np.nan])  # first NaN at index 2
    assert np.argmax(a) == 2 and np.argmin(a) == 2
    src = "import numpy as np\ndef f(a, i, j):\n i[0] = np.argmax(a)\n j[0] = np.argmin(a)\n"
    res = run_op(src,
                 "f", {"a": a}, {
                     "i": (1, ),
                     "j": (1, )
                 }, {"N": 5},
                 shapes={
                     "a": "(N,)",
                     "i": "(1,)",
                     "j": "(1,)"
                 },
                 dtypes={
                     "i": "int64",
                     "j": "int64"
                 },
                 backends=_ALL)
    ok, _ = _ok(res)
    assert ok, res


# --------------------------------------------------------------------------- #
# Fix 3: zero-size reduction refuses to lower (no OOB seed)                    #
# --------------------------------------------------------------------------- #


def test_max_zero_length_axis_refuses():
    args, kws = _call_args("np.max(a)")
    with pytest.raises(NotImplementedError):
        expand_max(_target(), args, {"a": ("0", )}, kws)


def test_min_zero_length_reduction_axis_refuses():
    args, kws = _call_args("np.min(a, axis=1)")
    with pytest.raises(NotImplementedError):
        expand_min(_target(), args, {"a": ("N", "0")}, kws)


def test_max_zero_length_kept_axis_ok():
    # A zero-length KEPT axis is fine (no reduction over it); only a zero-length
    # REDUCED axis has no identity.
    args, kws = _call_args("np.max(a, axis=1)")
    expand_max(_target(), args, {"a": ("0", "M")}, kws)  # must not raise


# --------------------------------------------------------------------------- #
# Fix 4: std / var honor ddof (divide by N - ddof)                            #
# --------------------------------------------------------------------------- #


def test_std_ddof_changes_divisor():
    args, kws = _call_args("np.std(a, ddof=1)")
    src = _src(expand_std(_target(), args, {"a": ("N", )}, kws))
    assert "- 1" in src  # divisor N - 1
    args0, kws0 = _call_args("np.std(a)")
    src0 = _src(expand_std(_target(), args0, {"a": ("N", )}, kws0))
    assert "- 1" not in src0  # default ddof=0 -> plain N


def test_var_ddof_numeric_all_backends():
    a = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    src = "import numpy as np\ndef f(a, v, s):\n v[0] = np.var(a, ddof=1)\n s[0] = np.std(a, ddof=2)\n"
    res = run_op(src,
                 "f", {"a": a}, {
                     "v": (1, ),
                     "s": (1, )
                 }, {"N": 8},
                 shapes={
                     "a": "(N,)",
                     "v": "(1,)",
                     "s": "(1,)"
                 },
                 backends=_ALL)
    ok, _ = _ok(res)
    assert ok, res


# --------------------------------------------------------------------------- #
# Fix 5: sum / prod / max / min honor initial=                                #
# --------------------------------------------------------------------------- #


def test_sum_initial_seeds_accumulator():
    args, kws = _call_args("np.sum(a, initial=5.0)")
    stmts = expand_sum(_target(), args, {"a": ("N", )}, kws)
    assert stmts[0].value.value == 5.0


def test_initial_numeric_all_backends():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    src = ("import numpy as np\ndef f(a, s, p, mx, mn):\n"
           " s[0] = np.sum(a, initial=10.0)\n"
           " p[0] = np.prod(a, initial=2.0)\n"
           " mx[0] = np.max(a, initial=100.0)\n"
           " mn[0] = np.min(a, initial=-5.0)\n")
    res = run_op(src,
                 "f", {"a": a}, {
                     "s": (1, ),
                     "p": (1, ),
                     "mx": (1, ),
                     "mn": (1, )
                 }, {"N": 4},
                 shapes={
                     "a": "(N,)",
                     "s": "(1,)",
                     "p": "(1,)",
                     "mx": "(1,)",
                     "mn": "(1,)"
                 },
                 backends=_ALL)
    ok, _ = _ok(res)
    assert ok, res


# --------------------------------------------------------------------------- #
# Fix 6: mean / std / var preserve a FLOAT input's dtype (float32 stays f32)  #
# --------------------------------------------------------------------------- #


def test_mean_float_input_preserves_dtype_in_desugar():
    float_src = _src(_reduce_axis_stmts("t", "s", "mean", [0], 2, 0, elem_is_float=True))
    assert "s.dtype" in float_src and "np.float64" not in float_src
    int_src = _src(_reduce_axis_stmts("t", "s", "mean", [0], 2, 0, elem_is_float=False))
    assert "np.float64" in int_src  # integer input upcasts to float64


def test_var_float_input_preserves_dtype_in_desugar():
    assert "s.dtype" in _src(_reduce_axis_stmts("t", "s", "var", [0], 2, 0, elem_is_float=True))
    assert "s.dtype" in _src(_reduce_axis_stmts("t", "s", "std", [0], 2, 0, elem_is_float=True))


# --------------------------------------------------------------------------- #
# Fix 7: concatenate accepts a negative-literal axis (axis=-1)                #
# --------------------------------------------------------------------------- #


def test_concatenate_negative_axis_all_backends():
    a = np.arange(6.0).reshape(2, 3)
    b = (np.arange(4.0) + 10).reshape(2, 2)
    src = "import numpy as np\ndef f(a, b, out):\n out[:] = np.concatenate((a, b), axis=-1)\n"
    res = run_op(src,
                 "f", {
                     "a": a,
                     "b": b
                 }, {"out": (2, 5)}, {
                     "M": 2,
                     "N": 3,
                     "K": 2
                 },
                 shapes={
                     "a": "(M, N)",
                     "b": "(M, K)",
                     "out": "(M, 5)"
                 },
                 backends=_ALL)
    ok, _ = _ok(res)
    assert ok, res


# --------------------------------------------------------------------------- #
# Fix 8/9: elementwise broadcasts BOTH operands (and symmetric size-1)         #
# --------------------------------------------------------------------------- #


def _ext(*toks):
    return tuple(_const_or_name(t) for t in toks)


def _unp(exts):
    return tuple(ast.unparse(e) for e in exts)


def test_broadcast_extents_size1_symmetric():
    assert _unp(_broadcast_extents(_ext("N", "1"), _ext("N", "M"))) == ("N", "M")
    assert _unp(_broadcast_extents(_ext("M"), _ext("N", "M"))) == ("N", "M")
    assert _unp(_broadcast_extents(_ext("1"), _ext("N"))) == ("N", )
    assert _unp(_broadcast_extents(_ext("N", "M"), _ext("N", "1"))) == ("N", "M")


def test_maximum_broadcast_lower_rank_first_operand_all_backends():
    a = np.array([1.0, 5.0, 3.0])  # (M,)
    B = np.array([
        [0.0, 6.0, 2.0],  # (N, M)
        [4.0, 1.0, 9.0]
    ])
    src = "import numpy as np\ndef f(a, B, out):\n out[:] = np.maximum(a, B)\n"
    res = run_op(src,
                 "f", {
                     "a": a,
                     "B": B
                 }, {"out": (2, 3)}, {
                     "M": 3,
                     "N": 2
                 },
                 shapes={
                     "a": "(M,)",
                     "B": "(N, M)",
                     "out": "(N, M)"
                 },
                 backends=_ALL)
    ok, _ = _ok(res)
    assert ok, res


def test_multiply_broadcast_row_vector_all_backends():
    a = np.array([[2.0, 3.0, 4.0]])  # (1, M)
    B = np.array([
        [1.0, 1.0, 1.0],  # (N, M)
        [5.0, 6.0, 7.0]
    ])
    src = "import numpy as np\ndef f(a, B, out):\n out[:] = np.multiply(a, B)\n"
    res = run_op(src,
                 "f", {
                     "a": a,
                     "B": B
                 }, {"out": (2, 3)}, {
                     "M": 3,
                     "N": 2
                 },
                 shapes={
                     "a": "(1, M)",
                     "B": "(N, M)",
                     "out": "(N, M)"
                 },
                 backends=_ALL)
    ok, _ = _ok(res)
    assert ok, res


# --------------------------------------------------------------------------- #
# Fix 10: an Ellipsis subscript entry fills all otherwise-unindexed axes       #
# --------------------------------------------------------------------------- #


def test_expr_rank_ellipsis():
    ranks = {"a": 3, "b": 4}
    assert _expr_rank(ast.parse("a[..., i]", mode="eval").body, ranks) == 2  # was 1 pre-fix
    assert _expr_rank(ast.parse("a[...]", mode="eval").body, ranks) == 3
    assert _expr_rank(ast.parse("b[..., 0, 1]", mode="eval").body, ranks) == 2
    assert _expr_rank(ast.parse("a[0, ...]", mode="eval").body, ranks) == 2


# --------------------------------------------------------------------------- #
# Fix 11: int / int is TRUE division (float64), not integer division          #
# --------------------------------------------------------------------------- #


def test_int_true_division_all_backends():
    a = np.array([7, 1, 9], dtype=np.int64)
    b = np.array([2, 4, 2], dtype=np.int64)
    src = "import numpy as np\ndef f(a, b, out):\n out[0] = a[0] / b[0]\n out[1] = a[1] / b[1]\n out[2] = a[2] / b[2]\n"
    res = run_op(src,
                 "f", {
                     "a": a,
                     "b": b
                 }, {"out": (3, )}, {"N": 3},
                 shapes={
                     "a": "(N,)",
                     "b": "(N,)",
                     "out": "(N,)"
                 },
                 dtypes={
                     "a": "int64",
                     "b": "int64"
                 },
                 backends=_ALL)
    ok, _ = _ok(res)
    assert ok, res


def test_explicit_float_cast_in_division_all_backends():
    a = np.array([7, 3], dtype=np.int64)
    b = np.array([2, 4], dtype=np.int64)
    src = "import numpy as np\ndef f(a, b, out):\n out[0] = float(a[0]) / b[0]\n out[1] = float(a[1]) / b[1]\n"
    res = run_op(src,
                 "f", {
                     "a": a,
                     "b": b
                 }, {"out": (2, )}, {"N": 2},
                 shapes={
                     "a": "(N,)",
                     "b": "(N,)",
                     "out": "(N,)"
                 },
                 dtypes={
                     "a": "int64",
                     "b": "int64"
                 },
                 backends=_ALL)
    ok, _ = _ok(res)
    assert ok, res
