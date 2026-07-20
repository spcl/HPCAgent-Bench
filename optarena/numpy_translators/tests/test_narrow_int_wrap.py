"""Narrow-int arithmetic and the element width: what the backends do and do NOT reproduce.

C, C++ and Fortran promote narrow reads (int8/16/32, uint8/16/32) to int64 and compute wide, so an
INTERMEDIATE that overflows the element width does not wrap. numpy evaluates the op at the operand
dtype and wraps there, so results diverge when an intermediate overflows before a non-linear step:
for int8 ``a = b = 100``, numpy's ``a + b`` wraps to -56 and ``// 2`` gives -28, while the wide form
computes 200 // 2 = 100. That gap is real and currently UNFIXED -- see the xfail below.

A per-op re-wrap was attempted and reverted: casting every narrow-int result back to its element
width broke more than it fixed (integer true division truncated in C/C++, int8-times-float truncated
in Fortran, ``**`` casting a libm double into a narrow int, and an undefined ``npb_wrap_*`` when a
non-inlined Fortran helper needed one), because "which numpy dtype does this subtree compute in" was
answered by two divergent hand-rolled oracles rather than one shared, differentially-tested one.

Everything below the xfail pins behaviour that must hold either way: no wrap where numpy PROMOTES,
and no truncation of results that are not integers at all. Those are the regression guards for a
future re-implementation.
"""
import numpy as np
import pytest
from _op_oracle import run_op

_NATIVE = ("c", "cpp", "fortran")


def _assert_ok(res):
    for backend, status in res.items():
        assert status == "ok" or status.startswith("skip"), f"{backend}: {status}"
    assert any(status == "ok" for status in res.values()), f"all skipped (vacuous): {res}"


def _run(src, ins, outs, dtypes, n):
    shapes = {name: "(N,)" for name in list(ins) + list(outs)}
    return run_op(src,
                  "f",
                  ins, {name: (n, )
                        for name in outs}, {"N": n},
                  shapes=shapes,
                  dtypes=dtypes,
                  backends=_NATIVE)


@pytest.mark.xfail(strict=True,
                   reason="per-op narrow-int wrap reverted; the backends compute wide, so an "
                   "intermediate that overflows the element width does not wrap. Re-implement in "
                   "numpyto_common with one shared oracle and a C/Fortran/numpy differential test.")
def test_int8_intermediate_overflow_wraps():
    # a + b overflows int8 (200 -> -56) BEFORE the floor-div, so wrapping changes the result. This
    # is the ONLY case in this file that distinguishes a per-op wrap from wrapping at the store --
    # the ring ops below compose identically either way, which is why they stayed green when the
    # feature was deleted and why they never protected it.
    src = ("import numpy as np\n"
           "def f(a, b, out):\n"
           "    for i in range(a.shape[0]):\n"
           "        out[i] = (a[i] + b[i]) // 2\n")
    a = np.array([100, 60, -100, 127], dtype=np.int8)
    b = np.array([100, 60, -100, 1], dtype=np.int8)
    assert np.array_equal((a + b) // 2, np.array([-28, 120 // 2, 28, -64], dtype=np.int8))  # numpy anchor
    _assert_ok(_run(src, {"a": a, "b": b}, ["out"], {"a": "int8", "b": "int8", "out": "int8"}, 4))


def test_int16_multiply_wraps():
    src = ("import numpy as np\n"
           "def f(x, out):\n"
           "    for i in range(x.shape[0]):\n"
           "        out[i] = x[i] * x[i]\n")
    x = np.array([30000, -30000, 181, 0], dtype=np.int16)
    _assert_ok(_run(src, {"x": x}, ["out"], {"x": "int16", "out": "int16"}, 4))


def test_unary_negation_of_int8_min_wraps():
    # -(-128) is -128 in int8, not 128 -- the unary op needs the same wrap as the binary ones.
    src = ("import numpy as np\n"
           "def f(m, out):\n"
           "    for i in range(m.shape[0]):\n"
           "        out[i] = -m[i]\n")
    m = np.array([-128, -1, 127], dtype=np.int8)
    _assert_ok(_run(src, {"m": m}, ["out"], {"m": "int8", "out": "int8"}, 3))


def test_uint8_subtraction_wraps_modulo():
    src = ("import numpy as np\n"
           "def f(a, b, out):\n"
           "    for i in range(a.shape[0]):\n"
           "        out[i] = a[i] - b[i]\n")
    a = np.array([0, 5, 255], dtype=np.uint8)
    b = np.array([1, 10, 255], dtype=np.uint8)
    assert np.array_equal(a - b, np.array([255, 251, 0], dtype=np.uint8))  # numpy anchor
    _assert_ok(_run(src, {"a": a, "b": b}, ["out"], {"a": "uint8", "b": "uint8", "out": "uint8"}, 3))


def test_int32_accumulator_wraps():
    src = ("import numpy as np\n"
           "def f(x, out):\n"
           "    for i in range(x.shape[0]):\n"
           "        out[i] = x[i] * x[i] + x[i]\n")
    x = np.array([2**15, 2**16, -(2**16), 3], dtype=np.int32)
    _assert_ok(_run(src, {"x": x}, ["out"], {"x": "int32", "out": "int32"}, 4))


# --- the wrap must NOT fire where numpy promotes -------------------------------------------------
def test_int64_operands_are_not_wrapped():
    # int64 IS the compute width; a wrap here would be a no-op at best and must not truncate.
    src = ("import numpy as np\n"
           "def f(x, out):\n"
           "    for i in range(x.shape[0]):\n"
           "        out[i] = x[i] * x[i]\n")
    x = np.array([2**20, 2**31, -(2**20)], dtype=np.int64)
    _assert_ok(_run(src, {"x": x}, ["out"], {"x": "int64", "out": "int64"}, 3))


def test_mixed_narrow_and_wide_promotes_and_is_not_wrapped():
    # numpy promotes int8 + int64 to int64, so the sum must NOT be truncated back to int8.
    src = ("import numpy as np\n"
           "def f(a, w, out):\n"
           "    for i in range(a.shape[0]):\n"
           "        out[i] = a[i] + w[i]\n")
    a = np.array([100, 100], dtype=np.int8)
    w = np.array([100, 10**6], dtype=np.int64)
    assert np.array_equal(a + w, np.array([200, 1000100], dtype=np.int64))  # promotes, no wrap
    _assert_ok(_run(src, {"a": a, "w": w}, ["out"], {"a": "int8", "w": "int64", "out": "int64"}, 2))


def test_logical_negation_is_not_wrapped():
    # `not x` yields a LOGICAL, not an integer. Wrapping it is a hard type error in Fortran
    # ("'a' argument of 'int' intrinsic must have a numeric type") and meaningless in C -- this is
    # what broke cloudsc, whose masks are narrow-int-backed booleans.
    src = ("import numpy as np\n"
           "def f(flag, x, out):\n"
           "    for i in range(x.shape[0]):\n"
           "        if not flag[i]:\n"
           "            out[i] = x[i]\n"
           "        else:\n"
           "            out[i] = 0\n")
    flag = np.array([0, 1, 0, 1], dtype=np.int32)
    x = np.array([5, 6, 7, 8], dtype=np.int32)
    res = run_op(src,
                 "f", {
                     "flag": flag,
                     "x": x
                 }, {"out": (4, )}, {"N": 4},
                 shapes={
                     "flag": "(N,)",
                     "x": "(N,)",
                     "out": "(N,)"
                 },
                 dtypes={
                     "flag": "int32",
                     "x": "int32",
                     "out": "int32"
                 },
                 backends=_NATIVE)
    _assert_ok(res)


def test_integer_true_division_is_not_truncated():
    """``/`` on ints is REAL division in numpy, and the wrap must not cast the quotient back.

    Integer ``a / b`` is desugared to ``np.float64(a) / b``, whose subtree reads only int arrays.
    The C wrap oracle saw int32 operands and no float, so it cast the double quotient to int32:
    7 / 2 emitted 3 where numpy says 3.5 -- a silent wrong ANSWER, not an overflow edge case, on
    every integer true division in every C and C++ kernel. Fortran was correct only because it
    already bailed on any call in the subtree.
    """
    src = ("import numpy as np\n"
           "def f(a, b, out):\n"
           "    for i in range(a.shape[0]):\n"
           "        out[i] = a[i] / b[i]\n")
    a = np.array([7, 9, 1, 5], dtype=np.int32)
    b = np.array([2, 2, 2, 2], dtype=np.int32)
    assert np.array_equal(a / b, np.array([3.5, 4.5, 0.5, 2.5]))  # numpy anchor: REAL division
    res = run_op(src,
                 "f", {
                     "a": a,
                     "b": b
                 }, {"out": (4, )}, {"N": 4},
                 shapes={
                     "a": "(N,)",
                     "b": "(N,)",
                     "out": "(N,)"
                 },
                 dtypes={
                     "a": "int32",
                     "b": "int32",
                     "out": "float64"
                 },
                 backends=_NATIVE)
    _assert_ok(res)


def test_narrow_true_division_is_not_truncated():
    # Same defect at int8, where the wrap is otherwise legitimately active.
    src = ("import numpy as np\n"
           "def f(a, b, out):\n"
           "    for i in range(a.shape[0]):\n"
           "        out[i] = a[i] / b[i]\n")
    a = np.array([7, 100, 3], dtype=np.int8)
    b = np.array([2, 8, 4], dtype=np.int8)
    _assert_ok(
        run_op(src,
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
                   "a": "int8",
                   "b": "int8",
                   "out": "float64"
               },
               backends=_NATIVE))


def test_call_result_is_not_wrapped():
    """A call's result dtype is not derivable from the operand dtypes below it, so the wrap must
    not fire through one. ``int(...)`` yields a Python int that numpy does NOT wrap at int8."""
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    for i in range(a.shape[0]):\n"
           "        out[i] = int(a[i]) * 3\n")
    a = np.array([100, 50, -100], dtype=np.int8)
    assert np.array_equal(np.array([int(x) * 3 for x in a]), np.array([300, 150, -300]))  # no wrap
    _assert_ok(
        run_op(src,
               "f", {"a": a}, {"out": (3, )}, {"N": 3},
               shapes={
                   "a": "(N,)",
                   "out": "(N,)"
               },
               dtypes={
                   "a": "int8",
                   "out": "int64"
               },
               backends=_NATIVE))


def test_float_operand_disables_the_int_wrap():
    # An int8 array combined with a float must compute (and stay) in floating point.
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    for i in range(a.shape[0]):\n"
           "        out[i] = a[i] * 3.5\n")
    a = np.array([100, 120], dtype=np.int8)
    _assert_ok(_run(src, {"a": a}, ["out"], {"a": "int8", "out": "float64"}, 2))
