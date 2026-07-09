# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Numerical lowering tests for the WEIRD / under-tested intrinsics, distilled from the more
complex npbench kernels (mandelbrot / cavity_flow / channel_flow / azimint / go_fast all lean on
np.maximum/minimum/clip/where/flip/std/tanh). Each case is a single-call kernel; run_op emits +
compiles + runs it on every backend and compares to numpy.

The edge cases here (NaN propagation, half-integer rounding, +/-inf, negative operands, 1-based
locations) are exactly where a naive intrinsic mapping diverges from numpy. Where a backend is
KNOWN to mis-lower an edge (see the ``skip_backends`` reason -> the review finding), it is skipped
rather than silently failing CI; drop the skip once the mapping is fixed and the case goes green.
The non-edge cases carry NO skips, so they are real all-backend coverage.
"""
import numpy as np
import pytest

# Reuse the standalone numerical oracle harness (build + run + numpy compare).
try:
    import _op_oracle as _oo
except ImportError:
    import importlib.util
    import pathlib
    _spec = importlib.util.spec_from_file_location("_op_oracle",
                                                   pathlib.Path(__file__).resolve().parent / "_op_oracle.py")
    _oo = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_oo)

_ALL = ("c", "cpp", "fortran", "numba", "pythran", "jax")

# Backends that mis-lower a given edge, keyed to the deep-review finding so the gap is documented,
# not hidden. C / C++ now route np.maximum/minimum through the NaN-propagating __npb_fmax/__npb_fmin
# helpers (numpyto_c/emit.py) and Fortran emits the NaN-propagating MERGE form (numpyto_fortran/
# emit.py), so both are NO LONGER skipped; only pythran (max/min drop NaN) remains.
_NAN_SUPPRESS = {"pythran": "pythran max/min do not propagate NaN"}


def _skip(status, label):
    fails = {b: s for b, s in status.items() if s.startswith("FAIL")}
    assert not fails, f"{label}: {fails}"


def _need_toolchain():
    import shutil
    if not (shutil.which("gcc") and shutil.which("g++") and shutil.which("gfortran")):
        pytest.skip("gcc/g++/gfortran needed for the native numerical check")


#: (label, source, inputs, out_shape, syms, shapes, skip_backends)
_CASES = [
    # -- np.maximum / np.minimum: value + broadcast (no NaN) -- all backends must agree ----------
    ("maximum_broadcast", "import numpy as np\ndef f(a, b, out):\n    out[:] = np.maximum(a, b)\n", {
        "a": np.array([-3.0, 5.0, 2.0, 9.0]),
        "b": np.array([1.0, 1.0, 1.0, 1.0])
    }, (4, ), {
        "N": 4
    }, {
        "a": "(N,)",
        "b": "(N,)",
        "out": "(N,)"
    }, {}),
    ("minimum_broadcast", "import numpy as np\ndef f(a, b, out):\n    out[:] = np.minimum(a, b)\n", {
        "a": np.array([-3.0, 5.0, 2.0, 9.0]),
        "b": np.array([1.0, 1.0, 1.0, 1.0])
    }, (4, ), {
        "N": 4
    }, {
        "a": "(N,)",
        "b": "(N,)",
        "out": "(N,)"
    }, {}),
    # -- np.maximum with NaN in the data -- numpy PROPAGATES; fmax/fmin/Fortran MAX do not --------
    ("maximum_nan_propagation", "import numpy as np\ndef f(a, b, out):\n    out[:] = np.maximum(a, b)\n", {
        "a": np.array([1.0, np.nan, 2.0, np.nan]),
        "b": np.array([0.0, 1.0, np.nan, np.nan])
    }, (4, ), {
        "N": 4
    }, {
        "a": "(N,)",
        "b": "(N,)",
        "out": "(N,)"
    }, _NAN_SUPPRESS),
    # -- np.clip: in-range + clamp both ends (clip = maximum(minimum)) ----------------------------
    ("clip_bounds", "import numpy as np\ndef f(a, out):\n    out[:] = np.clip(a, 0.0, 5.0)\n", {
        "a": np.array([-2.0, 0.0, 3.0, 5.0, 9.0])
    }, (5, ), {
        "N": 5
    }, {
        "a": "(N,)",
        "out": "(N,)"
    }, {}),
    # -- np.where: elementwise select (mandelbrot / go_fast idiom) ---------------------------------
    ("where_select", "import numpy as np\ndef f(a, b, out):\n    out[:] = np.where(a > b, a, b)\n", {
        "a": np.array([1.0, 4.0, -1.0, 7.0]),
        "b": np.array([2.0, 2.0, 2.0, 2.0])
    }, (4, ), {
        "N": 4
    }, {
        "a": "(N,)",
        "b": "(N,)",
        "out": "(N,)"
    }, {}),
    # -- np.flip: full reverse (channel_flow / cavity idiom). The dedicated np.flip path emits the
    #    correct Fortran arr(SIZE:1:-1); all backends agree (raw x[::-1] slicing is the buggy path,
    #    review emit.py:989 -- NOT exercised here).
    ("flip_reverse", "import numpy as np\ndef f(a, out):\n    out[:] = np.flip(a)\n", {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    }, (5, ), {
        "N": 5
    }, {
        "a": "(N,)",
        "out": "(N,)"
    }, {}),
    # -- np.std (ddof=0 default): srad/azimint statistic ------------------------------------------
    ("std_default", "import numpy as np\ndef f(a, out):\n    out[0] = np.std(a)\n", {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    }, (1, ), {
        "N": 6
    }, {
        "a": "(N,)",
        "out": "(1,)"
    }, {}),
    # -- np.tanh: transcendental (deep_learning / activation) ------------------------------------
    ("tanh_elementwise", "import numpy as np\ndef f(a, out):\n    out[:] = np.tanh(a)\n", {
        "a": np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    }, (5, ), {
        "N": 5
    }, {
        "a": "(N,)",
        "out": "(N,)"
    }, {}),
    # -- np.round on exact halves -- numpy is half-to-EVEN. Fortran now emits a half-even form
    #    (ANINT + tie-to-even correction), C/C++ round are half-even, so no backend is skipped.
    ("round_half_even", "import numpy as np\ndef f(a, out):\n    out[:] = np.round(a)\n", {
        "a": np.array([0.5, 1.5, 2.5, 3.5, -0.5, -2.5])
    }, (6, ), {
        "N": 6
    }, {
        "a": "(N,)",
        "out": "(N,)"
    }, {}),
    # -- np.floor: numpy returns FLOAT; finite values round-trip on every backend (the integer-
    #    intrinsic overflow, review emit.py:53, only bites +/-inf / |x|>=2^63, not exercised here).
    ("floor_finite", "import numpy as np\ndef f(a, out):\n    out[:] = np.floor(a)\n", {
        "a": np.array([-1.5, -0.1, 0.9, 2.5, 3.0])
    }, (5, ), {
        "N": 5
    }, {
        "a": "(N,)",
        "out": "(N,)"
    }, {}),
    # -- np.sign: value + NaN -- numpy sign(nan)=nan, C (x>0)-(x<0) gives 0 -----------------------
    ("sign_with_nan", "import numpy as np\ndef f(a, out):\n    out[:] = np.sign(a)\n", {
        "a": np.array([-3.0, 0.0, 4.0, np.nan])
    }, (4, ), {
        "N": 4
    }, {
        "a": "(N,)",
        "out": "(N,)"
    }, {
        "pythran": "sign gives 0 for NaN"
    }),
    # -- np.argmax: 0-based index. The Fortran emitter DOES adjust MAXLOC to 0-based (the review's
    #    off-by-one, emit.py:1148, does not reproduce here); first-max tie goes to the lowest index.
    ("argmax_index", "import numpy as np\ndef f(a, out):\n    out[0] = np.argmax(a)\n", {
        "a": np.array([3.0, 9.0, 1.0, 9.0, 2.0])
    }, (1, ), {
        "N": 5
    }, {
        "a": "(N,)",
        "out": "(1,)"
    }, {}),
]


@pytest.mark.parametrize("label,src,inputs,out_shape,syms,shapes,skip", _CASES, ids=[c[0] for c in _CASES])
def test_weird_intrinsic_matches_numpy(label, src, inputs, out_shape, syms, shapes, skip):
    """Each weird-intrinsic kernel must match numpy on every backend that does not carry a
    documented lowering gap (skip_backends)."""
    _need_toolchain()
    out_name = "out"
    status = _oo.run_op(src, "f", inputs, {out_name: out_shape}, syms, shapes=shapes, backends=_ALL, skip_backends=skip)
    _skip(status, label)


# ------------------------------------------------------------------------------------------------
# Dtype-sensitive edges (int64 width, float32 precision). These need a non-float64 ``dtypes``
# override, which the float64-only parametrized table above cannot thread, so each is a dedicated
# test. They target the numpyto_c (C / C++) width + precision lowering; a non-owned backend that
# genuinely cannot express the edge carries a documented skip.
# ------------------------------------------------------------------------------------------------


def test_int_cast_past_2_31():
    """``int(x)`` for a value beyond 2^31 must cast to int64, not a 32-bit ``int`` that truncates.
    3_000_000_000 > 2^31 (2_147_483_648): a 32-bit cast wraps to a negative int."""
    _need_toolchain()
    src = "import numpy as np\ndef f(a, out):\n    out[0] = int(a[0])\n"
    status = _oo.run_op(src,
                        "f", {"a": np.array([3_000_000_000.0])}, {"out": (1, )}, {"N": 1},
                        shapes={
                            "a": "(N,)",
                            "out": "(1,)"
                        },
                        backends=_ALL,
                        dtypes={"out": "int64"})
    _skip(status, "int_cast_past_2_31")


def test_int64_abs_large_magnitude():
    """``abs`` on an int64 with |x| > 2^31 must use ``llabs`` (64-bit); C's 32-bit ``abs`` truncates."""
    _need_toolchain()
    src = "import numpy as np\ndef f(a, out):\n    out[0] = abs(a[0])\n"
    status = _oo.run_op(src,
                        "f", {"a": np.array([-3_000_000_000], dtype=np.int64)}, {"out": (1, )}, {"N": 1},
                        shapes={
                            "a": "(N,)",
                            "out": "(1,)"
                        },
                        backends=_ALL,
                        dtypes={
                            "a": "int64",
                            "out": "int64"
                        })
    _skip(status, "int64_abs_large_magnitude")


def test_float_floor_division():
    """``a // b`` on FLOAT operands is ``floor(a / b)`` (numpy floor_divide). The unconditional
    integer ``int_floor`` uses ``%`` / ``/`` which C rejects on doubles -- an outright compile fail."""
    _need_toolchain()
    src = "import numpy as np\ndef f(a, b, out):\n    out[:] = a // b\n"
    status = _oo.run_op(src,
                        "f", {
                            "a": np.array([7.5, -7.5, 7.0, 10.0]),
                            "b": np.array([2.0, 2.0, 3.0, 4.0])
                        }, {"out": (4, )}, {"N": 4},
                        shapes={
                            "a": "(N,)",
                            "b": "(N,)",
                            "out": "(N,)"
                        },
                        backends=_ALL)
    _skip(status, "float_floor_division")


def test_float32_transcendental_precision():
    """A float32 kernel must round each op in float32: the ``f``-suffixed literal (``0.1f``) and the
    single-precision ``sqrtf`` reproduce numpy's per-op float32 rounding. A double literal / double
    ``sqrt`` would compute in double and round once at the store -- a ~1e-7 float32 discrepancy."""
    _need_toolchain()
    src = "import numpy as np\ndef f(a, out):\n    out[:] = np.sqrt(a) * 0.1\n"
    status = _oo.run_op(src,
                        "f", {"a": np.array([1.1, 2.2, 3.3, 0.7, 5.5], dtype=np.float32)}, {"out": (5, )}, {"N": 5},
                        shapes={
                            "a": "(N,)",
                            "out": "(N,)"
                        },
                        backends=_ALL,
                        dtypes={
                            "a": "float32",
                            "out": "float32"
                        },
                        skip_backends={
                            "fortran": "the `* 0.1` literal is float64, promoting the float32 multiply to double",
                            "numba": "numba types the `0.1` literal float64, promoting the float32 multiply to double",
                        })
    _skip(status, "float32_transcendental_precision")


def test_complex_dot_no_conjugation():
    """A rank-1 ``a @ b`` on COMPLEX operands is ``sum(a*b)`` -- numpy dot / matmul do NOT
    conjugate. Fortran's DOT_PRODUCT conjugates its first arg, so the emitter must emit the
    non-conjugating form; distinct real+imag parts make a conjugated result differ."""
    _need_toolchain()
    src = "import numpy as np\ndef f(a, b, out):\n    out[0] = a @ b\n"
    a = np.array([1 + 2j, 3 - 1j, -2 + 4j, 0.5 + 0.5j], dtype=np.complex128)
    b = np.array([2 - 1j, 1 + 1j, 3 + 2j, -1 + 0.5j], dtype=np.complex128)
    status = _oo.run_op(src,
                        "f", {
                            "a": a,
                            "b": b
                        }, {"out": (1, )}, {"N": 4},
                        shapes={
                            "a": "(N,)",
                            "b": "(N,)",
                            "out": "(1,)"
                        },
                        backends=_ALL,
                        dtypes={"out": "complex128"})
    _skip(status, "complex_dot_no_conjugation")


def test_signed_right_shift_arithmetic():
    """``x >> n`` on a SIGNED int64 is ARITHMETIC (sign-preserving) in numpy. Fortran must emit
    SHIFTA, not ISHFT(x, -n) (a logical / zero-fill shift that mangles negative values)."""
    _need_toolchain()
    src = "import numpy as np\ndef f(a, out):\n    out[:] = a >> 2\n"
    status = _oo.run_op(src,
                        "f", {"a": np.array([-8, -1, -100, 17, 1024], dtype=np.int64)}, {"out": (5, )}, {"N": 5},
                        shapes={
                            "a": "(N,)",
                            "out": "(N,)"
                        },
                        backends=_ALL,
                        dtypes={
                            "a": "int64",
                            "out": "int64"
                        })
    _skip(status, "signed_right_shift_arithmetic")


def test_int64_floor_division_precision():
    """``a // b`` on large int64 operands must divide in DOUBLE, not the default single-precision
    ``REAL()`` that drops mantissa bits above 2^24. Values near 2^40 expose the single-precision loss."""
    _need_toolchain()
    src = "import numpy as np\ndef f(a, b, out):\n    out[:] = a // b\n"
    a = np.array([1099511627776, 1099511627777, -1099511627776, 999999999999], dtype=np.int64)
    b = np.array([7, 7, 7, 3], dtype=np.int64)
    status = _oo.run_op(src,
                        "f", {
                            "a": a,
                            "b": b
                        }, {"out": (4, )}, {"N": 4},
                        shapes={
                            "a": "(N,)",
                            "b": "(N,)",
                            "out": "(N,)"
                        },
                        backends=_ALL,
                        dtypes={
                            "a": "int64",
                            "b": "int64",
                            "out": "int64"
                        })
    _skip(status, "int64_floor_division_precision")
