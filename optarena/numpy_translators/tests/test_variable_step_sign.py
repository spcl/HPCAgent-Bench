"""A range step whose sign is only known at RUNTIME must still iterate the Python direction.

Both native emitters used to decide the loop direction from the emitted TEXT of the step
(``step.startswith("-")``), which is only right for a literal. With ``s = -1`` held in a variable
the text is ``s``, so both picked the positive-step form and diverged silently:

* C emitted ``for (i = lo; i < hi; i += s)`` -- the guard is false at entry, so the loop ran ZERO
  times and the output kept whatever it was initialised with.
* Fortran emitted ``do i = lo, hi - 1, s``. Fortran's DO honours the runtime sign, but the
  inclusive-bound adjustment went the wrong way, so ``range(n, 0, -1)`` ran two iterations too far
  (down to ``-1``) -- out-of-range indices, not merely a wrong count.

Neither failed loudly, which is why this is pinned per backend rather than left to a kernel test.
"""
import numpy as np
from _op_oracle import run_op

_NATIVE = ("c", "cpp", "fortran")


def _assert_ok(res):
    for backend, status in res.items():
        assert status == "ok" or status.startswith("skip"), f"{backend}: {status}"
    assert any(status == "ok" for status in res.values()), f"all skipped (vacuous): {res}"


def _run(src, ins, n):
    names = list(ins) + ["out"]
    return run_op(src,
                  "f",
                  ins, {"out": (n, )}, {"N": n},
                  shapes={name: "(N,)"
                          for name in names},
                  dtypes={name: "float64"
                          for name in names},
                  backends=_NATIVE)


def test_negative_step_from_a_variable_runs_backwards():
    # s is -1 only at runtime; a text-sign check sees "s" and picks the forward form.
    src = ("import numpy as np\n"
           "def f(x, out):\n"
           "    n = x.shape[0]\n"
           "    s = -1\n"
           "    for i in range(n - 1, -1, s):\n"
           "        out[i] = x[i] * 2.0\n")
    _assert_ok(_run(src, {"x": np.arange(6, dtype=np.float64)}, 6))


def test_negative_step_variable_carries_a_running_value():
    # The reverse scan is order-dependent, so a wrong direction or trip count cannot cancel out.
    src = ("import numpy as np\n"
           "def f(x, out):\n"
           "    n = x.shape[0]\n"
           "    s = -1\n"
           "    acc = 0.0\n"
           "    for i in range(n - 1, -1, s):\n"
           "        acc = acc + x[i]\n"
           "        out[i] = acc\n")
    _assert_ok(_run(src, {"x": np.arange(1, 7, dtype=np.float64)}, 6))


def test_positive_step_from_a_variable_still_runs_forwards():
    # The fix must not flip the common case: an unknown-sign step that is POSITIVE at runtime.
    src = ("import numpy as np\n"
           "def f(x, out):\n"
           "    n = x.shape[0]\n"
           "    s = 2\n"
           "    for i in range(0, n, s):\n"
           "        out[i] = x[i] + 1.0\n")
    _assert_ok(_run(src, {"x": np.arange(7, dtype=np.float64)}, 7))


def test_literal_negative_step_unaffected():
    # The statically-known form keeps the plain reverse loop -- guards against a regression there.
    src = ("import numpy as np\n"
           "def f(x, out):\n"
           "    for i in range(x.shape[0] - 1, -1, -1):\n"
           "        out[i] = x[i] * 3.0\n")
    _assert_ok(_run(src, {"x": np.arange(5, dtype=np.float64)}, 5))
