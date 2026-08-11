"""Shape-manipulation aliases -> the transpose / reshape loop-lowering.

``np.swapaxes`` / ``np.expand_dims`` / ``np.squeeze`` are NumPy sugar over transpose and
reshape; the C / Fortran backends (which emit explicit loops, unlike numba / pythran / jax
that run the numpy verbatim) get them for free by delegating to the existing transpose /
reshape expanders, once ``_iter_extent_of`` learns their output shape. These validate the
emitted code numerically against numpy for param AND intermediate-local operands (the
ML-reshape case), across the full backend matrix (C / C++ / Fortran + numba / pythran / jax,
skip-tolerant). The kept ``swapaxes`` negative-axes and ``expand_dims`` middle-axis (keyword
form) cases subsume the positive-axis and trailing-axis variants.

The last two cover the way the ML corpus writes a squeeze that never reaches the expander at
all: BACK TO BACK on the trailing axes, which the front end rewrites to a chained subscript
first, plus the rank-independence guard the extent fold behind it rests on.
"""
import ast

import numpy as np
from _op_oracle import run_op

from numpyto_common.lowering import _is_newaxis_result_axis

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


def test_squeeze_back_to_back_on_the_trailing_axes():
    """``np.squeeze(np.squeeze(b, axis=-1), axis=-1)`` (the global-pool tail) is rewritten to the
    CHAINED subscript ``b[:, :, :, 0][:, :, 0]`` before any expander runs. ``b`` is a local, so
    its rank is not declared -- but the outer indices land inside the inner ``:`` positions, where
    the collapse to ``b[:, :, 0, 0]`` holds at every rank."""
    a = np.arange(12, dtype=np.float64).reshape(3, 4, 1, 1)
    src = ("import numpy as np\n"
           "def k(a, out):\n"
           "    b = a * 2.0\n"
           "    b = np.squeeze(np.squeeze(b, axis=-1), axis=-1)\n"
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
                   "a": "(M, N, 1, 1)",
                   "out": "(M, N)"
               },
               backends=_ALL))
    assert ok, res


def test_a_newaxis_extent_folds_without_the_operand_rank_but_nothing_else_does():
    """``X[:, None, :].shape[1]`` is 1 for every ``X``, which is what lets an ``expand_dims``
    operand's extent resolve before that operand's own shape is harvested -- the squeeze in
    ``np.squeeze(_pool(np.expand_dims(x, 1)), axis=1)`` is only provably dropping a unit axis
    because of it. Every other position is rank-DEPENDENT: a scalar index consumes a source
    axis, an Ellipsis stands for an unknown number of them, and a trailing axis is not named
    at all, so folding any of those without the rank would state the wrong extent."""

    def axis(text: str, k: int) -> bool:
        return _is_newaxis_result_axis(ast.parse(text, mode="eval").body, k)

    assert axis("X[:, None, :]", 1)
    assert axis("X[None, :]", 0)
    assert not axis("X[:, None, :]", 0), "a full slice takes its extent from the operand"
    assert not axis("X[0, None, :]", 1), "a scalar index consumes a source axis, shifting the map"
    assert not axis("X[..., None]", 1), "an Ellipsis stands for an unknown number of axes"
    assert not axis("X[idx, None]", 1), "a gather contributes its index array's own rank"
    assert not axis("X[:, None]", -1), "a negative axis counts from a rank that is not known"
    assert not axis("X[:, None]", 4), "past the named axes the result axis is a trailing one"
