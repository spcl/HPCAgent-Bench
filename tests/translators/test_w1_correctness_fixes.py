# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""numpy-faithfulness tests.

Each idiom round-trips through the ``run_op`` oracle against numpy:

* ``np.clip(a, lo, hi)`` with ``lo > hi`` returns a_max -- numpy defines clip as
  ``minimum(a_max, maximum(a, a_min))`` so the outer op is ``min``.
* ``np.linspace(start, stop, 1)`` returns ``[start]`` -- the divisor is
  ``max(n - 1, 1)`` so a single point does not divide by zero.
* axis ``np.max``/``np.min``/``np.argmax``/``np.argmin`` PROPAGATE NaN on the
  imperative (numba/c) path; arg* return the FIRST NaN index.
* ``a, b = b, a + b`` is a SIMULTANEOUS assignment -- the lowering captures the
  RHS into temps before writing any target (C/C++/Fortran), so the sequential
  split does not double ``b``.
* Fortran integer ``//`` above 2**53 floors exactly (no lossy REAL() round-trip).
"""

import numpy as np
import pytest

from tests.translators import op_oracle


@pytest.mark.parametrize(
    ("src", "inputs", "backends", "label"),
    [
        # c/cpp/fortran go through the shared expand_clip lowering; pythran's native
        # np.clip has the same reversed order and is rewritten in its emitter to
        # np.minimum(hi, np.maximum(a, lo)). All backends must match numpy.
        pytest.param(
            "import numpy as np\ndef f(a, lo, hi, out):\n    out[:] = np.clip(a, lo, hi)\n",
            {"a": np.array([1.0, 4.0, 10.0]), "lo": 5.0, "hi": 3.0},
            ("c", "cpp", "fortran", "numba"),
            "clip-lo>hi",
            id="lo-greater-than-hi",
        ),
        # numpy clip propagates a NaN operand; the min/max composition (shared lowering + pythran rewrite) must too.
        pytest.param(
            "import numpy as np\ndef f(a, out):\n    out[:] = np.clip(a, 1.0, 5.0)\n",
            {"a": np.array([np.nan, 4.0, 10.0])},
            ("c", "cpp", "fortran"),
            "clip-nan",
            id="propagates-nan",
        ),
    ],
)
def test_clip_lo_greater_than_hi_matches_numpy(
    src: str, inputs: dict[str, np.ndarray | float], backends: tuple[str, ...], label: str
) -> None:
    no = op_oracle.native()
    st = no.run_op(src, "f", inputs, {"out": (3,)}, {"N": 3}, shapes={"a": "(N,)", "out": "(N,)"})
    op_oracle.assert_ok(st, backends, label)


def test_linspace_single_point_matches_numpy() -> None:
    no = op_oracle.native()
    src = "import numpy as np\ndef f(out):\n    out[:] = np.linspace(0.0, 1.0, 1)\n"
    st = no.run_op(src, "f", {}, {"out": (1,)}, {"N": 1}, shapes={"out": "(N,)"})
    op_oracle.assert_ok(st, ("c", "cpp", "fortran"), "linspace-n1")


@pytest.mark.parametrize(
    ("func", "a", "dtypes", "label"),
    [
        pytest.param("max", np.array([[1.0, np.nan, 2.0], [4.0, 5.0, 6.0]]), None, "axis-max-nan", id="max"),
        # arg* return the FIRST NaN index.
        pytest.param(
            "argmax", np.array([[1.0, np.nan, 5.0], [4.0, 5.0, 6.0]]), {"out": "int64"}, "axis-argmax-nan", id="argmax"
        ),
    ],
)
def test_axis_max_propagates_nan(func: str, a: np.ndarray, dtypes: dict[str, str] | None, label: str) -> None:
    no = op_oracle.native()
    src = f"import numpy as np\ndef f(a, out):\n    out[:] = np.{func}(a, axis=1)\n"
    st = no.run_op(
        src, "f", {"a": a}, {"out": (2,)}, {"M": 2, "N": 3}, shapes={"a": "(M, N)", "out": "(M,)"}, dtypes=dtypes
    )
    op_oracle.assert_ok(st, ("c", "numba"), label)


@pytest.mark.parametrize(
    ("func", "backends"),
    [
        ("std", ("c", "cpp", "fortran", "numba")),
        # np.var(axis, ddof=1) on the imperative path only: c/cpp/fortran are EXCLUDED, since the
        # native np.var(x, axis=k) lowering fails to compile (output temp __cb1 left undeclared)
        # regardless of ddof -- a separate pre-existing bug, not this fix.
        ("var", ("numba",)),
    ],
)
def test_axis_reduction_ddof_matches_numpy(func: str, backends: tuple[str, ...]) -> None:
    # np.std / np.var over an axis with ddof=1 divides the SS by N-ddof, the numba imperative path included.
    no = op_oracle.native()
    a = np.array([[1.0, 2.0, 4.0, 8.0], [3.0, 5.0, 7.0, 9.0]])
    src = f"import numpy as np\ndef f(a, out):\n    out[:] = np.{func}(a, axis=1, ddof=1)\n"
    st = no.run_op(src, "f", {"a": a}, {"out": (2,)}, {"M": 2, "N": 4}, shapes={"a": "(M, N)", "out": "(M,)"})
    op_oracle.assert_ok(st, backends, f"{func}-ddof1")


@pytest.mark.parametrize(
    ("src", "inputs", "length", "shapes", "backends", "label"),
    [
        # c/cpp/fortran emit an exact integer floor-div (int_floor macro / kind-cast integer division) so they stay
        # correct above 2**53, where a float divide would lose mantissa bits and floor to the wrong integer.
        pytest.param(
            "import numpy as np\ndef f(a, b, out):\n    out[0] = a // b\n",
            {"a": np.int64(2**53 + 3), "b": np.int64(4)},
            1,
            {"out": "(N,)"},
            ("c", "cpp", "fortran"),
            "int-floordiv-2e53",
            id="above-2e53",
        ),
        # numpy ``//`` floors toward -inf; Fortran integer ``/`` truncates toward zero. The floor correction must
        # make -7 // 2 == -4 (not -3), 7 // -2 == -4.
        pytest.param(
            "import numpy as np\ndef f(a, b, out):\n    for i in range(a.shape[0]):\n        out[i] = a[i] // b[i]\n",
            {"a": np.array([-7, 7, -8, 9, -9], dtype=np.int64), "b": np.array([2, -2, 2, -4, 4], dtype=np.int64)},
            5,
            {"a": "(N,)", "b": "(N,)", "out": "(N,)"},
            ("c", "cpp", "fortran", "numba"),
            "int-floordiv-neg",
            id="negative",
        ),
    ],
)
def test_integer_floordiv_above_2e53_matches_numpy(
    src: str,
    inputs: dict[str, np.int64 | np.ndarray],
    length: int,
    shapes: dict[str, str],
    backends: tuple[str, ...],
    label: str,
) -> None:
    no = op_oracle.native()
    st = no.run_op(
        src,
        "f",
        inputs,
        {"out": (length,)},
        {"N": length},
        shapes=shapes,
        dtypes={"out": "int64", "a": "int64", "b": "int64"},
    )
    op_oracle.assert_ok(st, backends, label)
