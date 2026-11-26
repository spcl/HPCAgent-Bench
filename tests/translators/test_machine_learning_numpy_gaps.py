# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Numpy-op gaps surfaced auditing the KernelBench ML corpus (PR#5).

Three fixes, each validated numerically vs numpy across the full backend matrix (C / C++ /
Fortran + numba, skip-tolerant):

* ``np.inf`` / ``np.nan`` on FORTRAN -- lowered to the C ``INFINITY`` / ``NAN`` names, which
  Fortran expresses via ``ieee_value`` (masked-softmax / pooling init use ``-inf``);
* ``np.flip`` with an ``axis`` and for N-D (was 1-D only, and dropped the axis keyword);
* ``np.reshape(..., -1)`` -- the inferred dimension (``x.reshape(batch, -1)``, ubiquitous
  in ML flattens), resolved to the source element count over the other target dims.
"""

import numpy as np
import pytest

from tests.translators.op_oracle import all_ok, run_op

ALL = ("c", "cpp", "fortran", "numba")


def run_(
    src: str, ins: dict[str, np.ndarray], outs: dict[str, tuple[int, ...]], syms: dict[str, int], shapes: dict[str, str]
) -> tuple[bool, dict[str, str]]:
    return all_ok(run_op(src, "f", ins, outs, syms, shapes=shapes, backends=ALL))


X = np.linspace(-2.0, 2.0, 6)
A = np.arange(24, dtype=np.float64).reshape(4, 6)

# np.inf / np.nan (Fortran ieee_value)


def test_neg_inf_masking() -> None:
    ok, res = run_(
        "import numpy as np\ndef f(x, out):\n out[:] = np.where(x > 0.0, x, -np.inf)\n",
        {"x": X},
        {"out": (6,)},
        {"N": 6},
        {"x": "(N,)", "out": "(N,)"},
    )
    assert ok, res


# np.flip (N-D, axis-aware)


@pytest.mark.parametrize("axis", ["axis=0", "axis=1", "axis=-1", ""])
def test_flip_matches_numpy_for_every_axis_spelling(axis: str) -> None:
    ok, res = run_(
        f"import numpy as np\ndef f(a, out):\n out[:] = np.flip(a{', ' if axis else ''}{axis})\n",
        {"a": A},
        {"out": (4, 6)},
        {"M": 4, "N": 6},
        {"a": "(M, N)", "out": "(M, N)"},
    )
    assert ok, (axis, res)


# np.reshape(-1)


@pytest.mark.parametrize(
    ("src", "out_shape", "shapes"),
    [
        (
            "import numpy as np\n"
            "def f(a, out):\n"
            "    b = a.reshape(2, -1)\n"
            "    for i in range(2):\n"
            "        for j in range(12):\n"
            "            out[i, j] = b[i, j]\n",
            (2, 12),
            {"a": "(M, N)", "out": "(2, 12)"},
        ),
        # ``t.reshape(-1)`` where t is a computed local: the shape is resolved from t's inferred extent, not just a
        # parameter's.
        (
            "import numpy as np\ndef f(a, out):\n t = a * 2.0\n b = t.reshape(-1)\n out[:] = b\n",
            (24,),
            {"a": "(M, N)", "out": "(M*N,)"},
        ),
    ],
    ids=["row", "intermediate-local"],
)
def test_reshape_row_neg1(src: str, out_shape: tuple[int, ...], shapes: dict[str, str]) -> None:
    ok, res = run_(src, {"a": A}, {"out": out_shape}, {"M": 4, "N": 6}, shapes)
    assert ok, res


# np.ones_like (was missing from NP_ZEROS_ALIASES)


def test_ones_like() -> None:
    ok, res = run_(
        "import numpy as np\ndef f(a, out):\n b = np.ones_like(a)\n out[:] = a + b\n",
        {"a": np.arange(6, dtype=np.float64)},
        {"out": (6,)},
        {"N": 6},
        {"a": "(N,)", "out": "(N,)"},
    )
    assert ok, res
