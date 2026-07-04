"""Shape-manipulation aliases -> the transpose / reshape loop-lowering.

``np.swapaxes`` / ``np.expand_dims`` / ``np.squeeze`` are NumPy sugar over transpose and
reshape; the C / Fortran backends (which emit explicit loops, unlike numba / pythran / jax
that run the numpy verbatim) get them for free by delegating to the existing transpose /
reshape expanders, once ``_iter_extent_of`` learns their output shape. These validate the
emitted code numerically against numpy for param AND intermediate-local operands (the
ML-reshape case), across the full backend matrix (C / C++ / Fortran + numba / pythran / jax,
skip-tolerant). The kept ``swapaxes`` negative-axes and ``expand_dims`` middle-axis (keyword
form) cases subsume the positive-axis and trailing-axis variants.
"""
import numpy as np
from _op_oracle import run_op

_ALL = ("c", "cpp", "fortran", "numba", "pythran", "jax")


def _ok(res):
    return all(v == "ok" or v.startswith("skip") for v in res.values()), res


def test_swapaxes_negative_axes():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    src = ("import numpy as np\n"
           "def k(a, out):\n"
           "    b = np.swapaxes(a, -1, -2)\n"
           "    for i in range(out.shape[0]):\n"
           "        for j in range(out.shape[1]):\n"
           "            out[i, j] = b[i, j]\n")
    ok, res = _ok(
        run_op(src,
               "k", {"a": a}, {"out": (4, 3)}, {
                   "M": 3,
                   "N": 4
               },
               shapes={
                   "a": "(M, N)",
                   "out": "(N, M)"
               },
               backends=_ALL))
    assert ok, res


def test_swapaxes_intermediate_local_operand():
    """The operand is an intermediate local (``tmp = a + 1``), whose shape the machinery
    infers -- the ML case where a reshape follows a computed tensor."""
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    src = ("import numpy as np\n"
           "def k(a, out):\n"
           "    tmp = a + 1.0\n"
           "    b = np.swapaxes(tmp, 0, 1)\n"
           "    for i in range(out.shape[0]):\n"
           "        for j in range(out.shape[1]):\n"
           "            out[i, j] = b[i, j]\n")
    ok, res = _ok(
        run_op(src,
               "k", {"a": a}, {"out": (4, 3)}, {
                   "M": 3,
                   "N": 4
               },
               shapes={
                   "a": "(M, N)",
                   "out": "(N, M)"
               },
               backends=_ALL))
    assert ok, res


def test_expand_dims_middle_axis():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    src = ("import numpy as np\n"
           "def k(a, out):\n"
           "    b = np.expand_dims(a, axis=1)\n"
           "    for i in range(a.shape[0]):\n"
           "        for j in range(a.shape[1]):\n"
           "            out[i, 0, j] = b[i, 0, j]\n")
    ok, res = _ok(
        run_op(src,
               "k", {"a": a}, {"out": (3, 1, 4)}, {
                   "M": 3,
                   "N": 4
               },
               shapes={
                   "a": "(M, N)",
                   "out": "(M, 1, N)"
               },
               backends=_ALL))
    assert ok, res


def test_squeeze_named_axis():
    a = np.arange(12, dtype=np.float64).reshape(3, 1, 4)
    src = ("import numpy as np\n"
           "def k(a, out):\n"
           "    b = np.squeeze(a, 1)\n"
           "    for i in range(out.shape[0]):\n"
           "        for j in range(out.shape[1]):\n"
           "            out[i, j] = b[i, j]\n")
    ok, res = _ok(
        run_op(src,
               "k", {"a": a}, {"out": (3, 4)}, {
                   "M": 3,
                   "N": 4
               },
               shapes={
                   "a": "(M, 1, N)",
                   "out": "(M, N)"
               },
               backends=_ALL))
    assert ok, res


def test_squeeze_all_unit_axes():
    a = np.arange(12, dtype=np.float64).reshape(1, 3, 1, 4)
    src = ("import numpy as np\n"
           "def k(a, out):\n"
           "    b = np.squeeze(a)\n"
           "    for i in range(out.shape[0]):\n"
           "        for j in range(out.shape[1]):\n"
           "            out[i, j] = b[i, j]\n")
    ok, res = _ok(
        run_op(src,
               "k", {"a": a}, {"out": (3, 4)}, {
                   "M": 3,
                   "N": 4
               },
               shapes={
                   "a": "(1, M, 1, N)",
                   "out": "(M, N)"
               },
               backends=_ALL))
    assert ok, res
