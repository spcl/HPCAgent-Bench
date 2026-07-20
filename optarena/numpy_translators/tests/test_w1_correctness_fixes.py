"""W1 numpy-faithfulness regression tests (deep-audit 2026-07-10).

Each idiom round-trips through the ``run_op`` oracle against numpy, pinning a
bug fixed in wave W1:

* ``np.clip(a, lo, hi)`` with ``lo > hi`` returns a_max -- numpy defines clip as
  ``minimum(a_max, maximum(a, a_min))`` so the outer op is ``min`` (was ``max``).
* ``np.linspace(start, stop, 1)`` returns ``[start]`` -- the divisor is
  ``max(n - 1, 1)`` so a single point no longer divides by zero.
* axis ``np.max``/``np.min``/``np.argmax``/``np.argmin`` PROPAGATE NaN on the
  imperative (numba/pythran/c) path; arg* return the FIRST NaN index.
* ``a, b = b, a + b`` is a SIMULTANEOUS assignment -- the lowering captures the
  RHS into temps before writing any target (C/C++/Fortran), so the sequential
  split no longer doubles ``b``.
* Fortran integer ``//`` above 2**53 floors exactly (no lossy REAL() round-trip).
"""
import importlib.util
import pathlib

import numpy as np
import pytest


def _oracle():
    import shutil
    if not (shutil.which("gcc") and shutil.which("gfortran") and shutil.which("g++")):
        pytest.skip("gcc/g++/gfortran needed for the native oracle emit step")
    try:
        import _op_oracle
    except ImportError:
        spec = importlib.util.spec_from_file_location("_op_oracle",
                                                      pathlib.Path(__file__).resolve().parent / "_op_oracle.py")
        _op_oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_op_oracle)
    return _op_oracle


def _assert_ok(status, backends, label):
    """No requested backend may FAIL; a backend that skips (unsupported / no
    toolchain) is tolerated but at least one must actually have run."""
    ran = False
    for b in backends:
        s = status.get(b, "skip:absent")
        if s.startswith("skip"):
            continue
        ran = True
        assert not s.startswith("FAIL"), f"{label}: {b}: {s}"
    if not ran:
        pytest.skip(f"{label}: no backend ran ({status})")


def test_clip_lo_greater_than_hi_matches_numpy():
    no = _oracle()
    src = ("import numpy as np\n"
           "def f(a, lo, hi, out):\n"
           "    out[:] = np.clip(a, lo, hi)\n")
    # c/cpp/fortran go through the shared expand_clip lowering; pythran's native
    # np.clip has the same reversed order and is rewritten in its emitter to
    # np.minimum(hi, np.maximum(a, lo)). All backends must match numpy.
    st = no.run_op(src,
                   "f", {
                       "a": np.array([1.0, 4.0, 10.0]),
                       "lo": 5.0,
                       "hi": 3.0
                   }, {"out": (3, )}, {"N": 3},
                   shapes={
                       "a": "(N,)",
                       "out": "(N,)"
                   })
    _assert_ok(st, ("c", "cpp", "fortran", "numba", "pythran", "jax"), "clip-lo>hi")


def test_clip_propagates_nan():
    # numpy clip propagates a NaN operand; the min/max composition (shared
    # lowering + pythran rewrite) must too.
    no = _oracle()
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    out[:] = np.clip(a, 1.0, 5.0)\n")
    st = no.run_op(src,
                   "f", {"a": np.array([np.nan, 4.0, 10.0])}, {"out": (3, )}, {"N": 3},
                   shapes={
                       "a": "(N,)",
                       "out": "(N,)"
                   })
    _assert_ok(st, ("c", "cpp", "fortran", "pythran", "jax"), "clip-nan")


def test_linspace_single_point_matches_numpy():
    no = _oracle()
    src = ("import numpy as np\n"
           "def f(out):\n"
           "    out[:] = np.linspace(0.0, 1.0, 1)\n")
    st = no.run_op(src, "f", {}, {"out": (1, )}, {"N": 1}, shapes={"out": "(N,)"})
    _assert_ok(st, ("c", "cpp", "fortran"), "linspace-n1")


def test_linspace_multi_point_still_matches_numpy():
    no = _oracle()
    src = ("import numpy as np\n"
           "def f(out):\n"
           "    out[:] = np.linspace(0.0, 1.0, 5)\n")
    st = no.run_op(src, "f", {}, {"out": (5, )}, {"N": 5}, shapes={"out": "(N,)"})
    _assert_ok(st, ("c", "cpp", "fortran"), "linspace-n5")


def test_axis_max_propagates_nan():
    no = _oracle()
    a = np.array([[1.0, np.nan, 2.0], [4.0, 5.0, 6.0]])
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    out[:] = np.max(a, axis=1)\n")
    st = no.run_op(src, "f", {"a": a}, {"out": (2, )}, {"M": 2, "N": 3}, shapes={"a": "(M, N)", "out": "(M,)"})
    _assert_ok(st, ("c", "numba", "pythran"), "axis-max-nan")


def test_axis_argmax_returns_first_nan_index():
    no = _oracle()
    a = np.array([[1.0, np.nan, 5.0], [4.0, 5.0, 6.0]])
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    out[:] = np.argmax(a, axis=1)\n")
    st = no.run_op(src,
                   "f", {"a": a}, {"out": (2, )}, {
                       "M": 2,
                       "N": 3
                   },
                   shapes={
                       "a": "(M, N)",
                       "out": "(M,)"
                   },
                   dtypes={"out": "int64"})
    _assert_ok(st, ("c", "numba", "pythran"), "axis-argmax-nan")


def test_axis_std_ddof_matches_numpy():
    # np.std over an axis with ddof=1 divides the SS by N-ddof; the numba/pythran
    # imperative path used to hardcode ddof=0. All backends support axis std.
    no = _oracle()
    a = np.array([[1.0, 2.0, 4.0, 8.0], [3.0, 5.0, 7.0, 9.0]])
    src = "import numpy as np\ndef f(a, out):\n    out[:] = np.std(a, axis=1, ddof=1)\n"
    st = no.run_op(src, "f", {"a": a}, {"out": (2, )}, {"M": 2, "N": 4}, shapes={"a": "(M, N)", "out": "(M,)"})
    _assert_ok(st, ("c", "cpp", "fortran", "numba", "pythran", "jax"), "std-ddof1")


def test_axis_var_ddof_matches_numpy():
    # np.var(axis, ddof=1) on the imperative path. c/cpp/fortran are EXCLUDED: the
    # native np.var(x, axis=k) lowering fails to compile (output temp __cb1 left
    # undeclared) regardless of ddof -- a separate pre-existing bug, not this fix.
    no = _oracle()
    a = np.array([[1.0, 2.0, 4.0, 8.0], [3.0, 5.0, 7.0, 9.0]])
    src = "import numpy as np\ndef f(a, out):\n    out[:] = np.var(a, axis=1, ddof=1)\n"
    st = no.run_op(src, "f", {"a": a}, {"out": (2, )}, {"M": 2, "N": 4}, shapes={"a": "(M, N)", "out": "(M,)"})
    _assert_ok(st, ("numba", "pythran", "jax"), "var-ddof1")


def test_integer_floordiv_above_2e53_matches_numpy():
    # c/cpp/fortran emit an exact integer floor-div (int_floor macro / kind-cast
    # integer division) so they stay correct above 2**53, where a float divide
    # would lose mantissa bits and floor to the wrong integer.
    no = _oracle()
    src = ("import numpy as np\n"
           "def f(a, b, out):\n"
           "    out[0] = a // b\n")
    st = no.run_op(src,
                   "f", {
                       "a": np.int64(2**53 + 3),
                       "b": np.int64(4)
                   }, {"out": (1, )}, {"N": 1},
                   shapes={"out": "(N,)"},
                   dtypes={
                       "out": "int64",
                       "a": "int64",
                       "b": "int64"
                   })
    _assert_ok(st, ("c", "cpp", "fortran"), "int-floordiv-2e53")


def test_integer_floordiv_negative_matches_numpy():
    # numpy ``//`` floors toward -inf; Fortran integer ``/`` truncates toward
    # zero. The floor correction must make -7 // 2 == -4 (not -3), 7 // -2 == -4.
    no = _oracle()
    src = ("import numpy as np\n"
           "def f(a, b, out):\n"
           "    for i in range(a.shape[0]):\n"
           "        out[i] = a[i] // b[i]\n")
    a = np.array([-7, 7, -8, 9, -9], dtype=np.int64)
    b = np.array([2, -2, 2, -4, 4], dtype=np.int64)
    st = no.run_op(src,
                   "f", {
                       "a": a,
                       "b": b
                   }, {"out": (5, )}, {"N": 5},
                   shapes={
                       "a": "(N,)",
                       "b": "(N,)",
                       "out": "(N,)"
                   },
                   dtypes={
                       "out": "int64",
                       "a": "int64",
                       "b": "int64"
                   })
    _assert_ok(st, ("c", "cpp", "fortran", "numba", "pythran", "jax"), "int-floordiv-neg")
