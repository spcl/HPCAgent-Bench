"""Numpy-op gaps surfaced auditing the KernelBench ML corpus (PR#5).

Three fixes, each validated numerically vs numpy across the full backend matrix (C / C++ /
Fortran + numba / pythran / jax, skip-tolerant):

* ``np.inf`` / ``np.nan`` on FORTRAN -- lowered to the C ``INFINITY`` / ``NAN`` names, which
  Fortran expresses via ``ieee_value`` (masked-softmax / pooling init use ``-inf``);
* ``np.flip`` with an ``axis`` and for N-D (was 1-D only, and dropped the axis keyword);
* ``np.reshape(..., -1)`` -- the inferred dimension (``x.reshape(batch, -1)``, ubiquitous
  in ML flattens), resolved to the source element count over the other target dims.
"""
import numpy as np
from _op_oracle import run_op

_ALL = ("c", "cpp", "fortran", "numba", "pythran", "jax")


def _ok(res):
    return all(v == "ok" or v.startswith("skip") for v in res.values()), res


def _run(src, ins, outs, syms, shapes):
    return _ok(run_op(src, "f", ins, outs, syms, shapes=shapes, backends=_ALL))


_X = np.linspace(-2.0, 2.0, 6)
_A = np.arange(24, dtype=np.float64).reshape(4, 6)


# --- np.inf / np.nan (Fortran ieee_value) --------------------------------- #


def test_neg_inf_masking():
    ok, res = _run("import numpy as np\ndef f(x, out):\n out[:] = np.where(x > 0.0, x, -np.inf)\n",
                   {"x": _X}, {"out": (6, )}, {"N": 6}, {"x": "(N,)", "out": "(N,)"})
    assert ok, res


def test_nan_fill():
    ok, res = _run("import numpy as np\ndef f(x, out):\n out[:] = np.where(x > 5.0, np.nan, x)\n",
                   {"x": _X}, {"out": (6, )}, {"N": 6}, {"x": "(N,)", "out": "(N,)"})
    assert ok, res


# --- np.flip (N-D, axis-aware) --------------------------------------------- #


def test_flip_axis0_and_axis1():
    for axis in (0, 1, -1):
        ok, res = _run(f"import numpy as np\ndef f(a, out):\n out[:] = np.flip(a, axis={axis})\n",
                       {"a": _A}, {"out": (4, 6)}, {"M": 4, "N": 6}, {"a": "(M, N)", "out": "(M, N)"})
        assert ok, (axis, res)


def test_flip_all_axes():
    ok, res = _run("import numpy as np\ndef f(a, out):\n out[:] = np.flip(a)\n",
                   {"a": _A}, {"out": (4, 6)}, {"M": 4, "N": 6}, {"a": "(M, N)", "out": "(M, N)"})
    assert ok, res


# --- np.reshape(-1) -------------------------------------------------------- #


def test_reshape_row_neg1():
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    b = a.reshape(2, -1)\n"
           "    for i in range(2):\n"
           "        for j in range(12):\n"
           "            out[i, j] = b[i, j]\n")
    ok, res = _run(src, {"a": _A}, {"out": (2, 12)}, {"M": 4, "N": 6}, {"a": "(M, N)", "out": "(2, 12)"})
    assert ok, res


def test_reshape_neg1_on_intermediate_local():
    """``t.reshape(-1)`` where t is a computed local -- the shape is resolved from t's
    inferred extent, not just a parameter's."""
    ok, res = _run("import numpy as np\ndef f(a, out):\n t = a * 2.0\n b = t.reshape(-1)\n out[:] = b\n",
                   {"a": _A}, {"out": (24, )}, {"M": 4, "N": 6}, {"a": "(M, N)", "out": "(M*N,)"})
    assert ok, res


# --- np.ones_like (was missing from NP_ZEROS_ALIASES) ---------------------- #


def test_ones_like():
    ok, res = _run("import numpy as np\ndef f(a, out):\n b = np.ones_like(a)\n out[:] = a + b\n",
                   {"a": np.arange(6, dtype=np.float64)}, {"out": (6, )}, {"N": 6}, {"a": "(N,)", "out": "(N,)"})
    assert ok, res
