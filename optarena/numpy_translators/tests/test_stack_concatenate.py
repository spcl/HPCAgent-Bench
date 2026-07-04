"""``np.stack`` (new axis) and ``np.concatenate`` (existing axis) -> copy loop nests.

``np.concatenate`` was already lowered (join operands along an existing axis); ``np.stack``
adds a NEW axis (out rank = operand rank + 1, the k-th extent = number of operands) and is
lowered here. Both emit per-operand copy loops for the C / Fortran backends (numba / pythran
/ jax run the numpy verbatim). Validated numerically vs numpy across the full backend matrix
(C / C++ / Fortran + numba / pythran / jax, skip-tolerant). ``test_stack_negative_axis_appends``
subsumes the positive last-axis case (``-1`` normalizes to it).
"""
import numpy as np
from _op_oracle import run_op

_ALL = ("c", "cpp", "fortran", "numba", "pythran", "jax")


def _ok(res):
    return all(v == "ok" or v.startswith("skip") for v in res.values()), res


_A = np.arange(6, dtype=np.float64).reshape(2, 3)
_B = np.arange(6, 12, dtype=np.float64).reshape(2, 3)


def _stack3d(axis, out_shape, out_sym, n_operands=2):
    seq = "(a, b)" if n_operands == 2 else "(a, b, a)"
    src = ("import numpy as np\n"
           "def k(a, b, out):\n"
           f"    c = np.stack({seq}, axis={axis})\n"
           "    for i in range(out.shape[0]):\n"
           "        for j in range(out.shape[1]):\n"
           "            for l in range(out.shape[2]):\n"
           "                out[i, j, l] = c[i, j, l]\n")
    return run_op(src, "k", {"a": _A, "b": _B}, {"out": out_shape}, {"M": 2, "N": 3},
                  shapes={"a": "(M, N)", "b": "(M, N)", "out": out_sym}, backends=_ALL)


def test_stack_axis0():
    ok, res = _ok(_stack3d(0, (2, 2, 3), "(2, M, N)"))
    assert ok, res


def test_stack_axis1():
    ok, res = _ok(_stack3d(1, (2, 2, 3), "(M, 2, N)"))
    assert ok, res


def test_stack_negative_axis_appends():
    """``axis=-1`` (a UnaryOp literal, not Constant(-1)) appends the new axis."""
    ok, res = _ok(_stack3d(-1, (2, 3, 2), "(M, N, 2)"))
    assert ok, res


def test_stack_three_operands():
    ok, res = _ok(_stack3d(0, (3, 2, 3), "(3, M, N)", n_operands=3))
    assert ok, res


# --------------------------------------------------------------------------- #
# concatenate regression (join along an existing axis) -- already supported.   #
# --------------------------------------------------------------------------- #


def _concat(axis, out_shape, out_sym, n_operands=2):
    seq = "(a, b)" if n_operands == 2 else "(a, b, a)"
    src = ("import numpy as np\n"
           "def k(a, b, out):\n"
           f"    c = np.concatenate({seq}, axis={axis})\n"
           "    for i in range(out.shape[0]):\n"
           "        for j in range(out.shape[1]):\n"
           "            out[i, j] = c[i, j]\n")
    return run_op(src, "k", {"a": _A, "b": _B}, {"out": out_shape}, {"M": 2, "N": 3},
                  shapes={"a": "(M, N)", "b": "(M, N)", "out": out_sym}, backends=_ALL)


def test_concatenate_axis0():
    ok, res = _ok(_concat(0, (4, 3), "(2 * M, N)"))
    assert ok, res


def test_concatenate_axis1():
    ok, res = _ok(_concat(1, (2, 6), "(M, 2 * N)"))
    assert ok, res


def test_concatenate_three_operands():
    ok, res = _ok(_concat(0, (6, 3), "(3 * M, N)", n_operands=3))
    assert ok, res
