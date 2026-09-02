"""Compound-assign and float floor-division emit gaps (C / C++ / Fortran).

* ``t //= v`` / ``t %= v`` had no numpy-faithful compound form: C ``//=`` raised, C ``%=``
  emitted raw dividend-sign modulo, Fortran ``//=`` did truncating ``/`` and ``%=`` emitted
  the invalid infix ``x MOD y``. Both emitters now expand to ``t = t <op> v`` through the
  BinOp path (int_floor / python_mod).
* Fortran float ``a // b`` lowered to ``FLOOR(.., int64)`` -- an integer, undefined for
  ``|a/b| > 2^63`` (1e20 // 2 wrapped instead of 5e19). It now yields a real floor.
"""

import numpy as np
from _op_oracle import run_op

_NATIVE = ("c", "cpp", "fortran")


def _assert_native_ok(res):
    for backend, status in res.items():
        assert status == "ok" or status.startswith("skip"), f"{backend}: {status}"
    assert any(status == "ok" for status in res.values()), f"all backends skipped (vacuous): {res}"


def test_compound_int_floordiv_and_mod_match_numpy():
    # //= and %= over the full sign matrix: numpy // floors toward -inf and % takes the
    # divisor's sign, unlike C/Fortran's native truncate/dividend-sign forms.
    # The compound ops run DIRECTLY on the output buffers -- the shape a real kernel writes.
    # (This used to be routed through scalar locals to dodge run_op's complex-scratch probe,
    # which raised TypeError on `//=` over complex; the probe now skips itself instead.)
    src = (
        "import numpy as np\n"
        "def f(a, b, q, r):\n"
        "    for i in range(a.shape[0]):\n"
        "        q[i] = a[i]\n"
        "        q[i] //= b[i]\n"
        "        r[i] = a[i]\n"
        "        r[i] %= b[i]\n"
    )
    a = np.array([-7, 7, -8, 8], dtype=np.int64)
    b = np.array([2, -2, 3, -3], dtype=np.int64)
    res = run_op(
        src,
        "f",
        {"a": a, "b": b},
        {"q": (4,), "r": (4,)},
        {"N": 4},
        shapes={"a": "(N,)", "b": "(N,)", "q": "(N,)", "r": "(N,)"},
        dtypes={"a": "int64", "b": "int64", "q": "int64", "r": "int64"},
        backends=_NATIVE,
    )
    _assert_native_ok(res)


def test_literal_grid_unpack_does_not_overflow_int32():
    # tuple-unpack int_locals were declared as bare 32-bit C int; a literal grid whose
    # pairwise product exceeds 2^31 (46341*46341 = 2147488281) wrapped negative.
    src = "import numpy as np\ndef f(out):\n    nx, ny = 46341, 46341\n    out[0] = nx * ny\n"
    res = run_op(src, "f", {}, {"out": (1,)}, {}, shapes={"out": "(1,)"}, backends=("c", "cpp"))
    _assert_native_ok(res)


def test_floordiv_float_matches_numpy_over_overflow_range():
    # 1e20 // 2 == 5e19 in numpy; the old Fortran FLOOR(.., int64) overflowed int64 here.
    src = "import numpy as np\ndef f(a, b, out):\n    for i in range(a.shape[0]):\n        out[i] = a[i] // b[i]\n"
    a = np.array([1e20, -7.5, 7.5, 3.0], dtype=np.float64)
    b = np.array([2.0, 2.0, -2.0, 2.0], dtype=np.float64)
    res = run_op(
        src,
        "f",
        {"a": a, "b": b},
        {"out": (4,)},
        {"N": 4},
        shapes={"a": "(N,)", "b": "(N,)", "out": "(N,)"},
        backends=_NATIVE,
    )
    _assert_native_ok(res)


def test_int_cast_floordiv_and_mod_floor_like_numpy():
    # int(a[i]) is an INTEGER however float a is, so ``int(a[i]) // 2`` must take the
    # integer floor-div branch. The float-operand test used to walk into the cast, see
    # float ``a`` and pick the float path ``floor((x) / (y))`` -- but both emitted operands
    # are already int64 there, so C truncated toward zero and the floor was a no-op:
    # int(-7.5) // 2 gave -3 instead of numpy's -4 (same for the divisor-sign %).
    src = (
        "import numpy as np\n"
        "def f(a, q, r):\n"
        "    for i in range(a.shape[0]):\n"
        "        q[i] = int(a[i]) // 2\n"
        "        r[i] = int(a[i]) % 3\n"
    )
    a = np.array([-7.5, -1.5, 3.5, 7.5], dtype=np.float64)
    res = run_op(
        src,
        "f",
        {"a": a},
        {"q": (4,), "r": (4,)},
        {"N": 4},
        shapes={"a": "(N,)", "q": "(N,)", "r": "(N,)"},
        dtypes={"q": "int64", "r": "int64"},
        backends=_NATIVE,
    )
    _assert_native_ok(res)


def test_integer_floordiv_index_emits_integer_index():
    # b[i // 2] is an integer floor-div used as an array index. It must take the integer
    # FloorDiv branch and emit an INTEGER index -- a real-valued index is rejected by
    # Fortran -std=f2018 (regression: loop induction vars must type as int64).
    src = "import numpy as np\ndef f(a, b):\n    for i in range(a.shape[0]):\n        b[i // 2] = b[i // 2] + a[i]\n"
    a = np.arange(1.0, 9.0)
    res = run_op(src, "f", {"a": a}, {"b": (4,)}, {"N": 8}, shapes={"a": "(N,)", "b": "(4,)"}, backends=_NATIVE)
    _assert_native_ok(res)
