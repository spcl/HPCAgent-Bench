# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``len(array)`` -> the array's symbolic first-dimension size (native backends).

numpy's ``len(a)`` is ``a.shape[0]``. C / C++ have no array ``len`` (the emitted
literal ``len(a)`` fails to compile) and Fortran's ``len`` is the CHARACTER-length
intrinsic, so a native kernel that reads ``len(a)`` -- e.g. the GROMACS NBNxM
kernel's ``len(coulomb_table_f)`` bound -- would not compile. ``_ShapeMidExpression
Rewriter`` maps it to the first-dim shape symbol, alongside ``a.shape[k]`` /
``a.size`` / ``a.ndim``. The python backends (numba) run the body
verbatim and keep the builtin, so they are unaffected.
"""

import numpy as np
import pytest

from tests.translators.op_oracle import all_ok, run_op

ALL = ("c", "cpp", "fortran", "numba")


@pytest.mark.parametrize(
    ("src", "a", "params", "shape"),
    [
        # ``len`` of a 2-D array is the leading extent, not the total size.
        pytest.param(
            "import numpy as np\ndef f(a, out):\n out[0] = float(len(a))\n",
            np.arange(12, dtype=np.float64).reshape(3, 4),
            {"M": 3, "N": 4},
            "(M, N)",
            id="2d-first-dim",
        ),
        # the GROMACS pattern: ``len(table)`` used as an extent inside the kernel.
        pytest.param(
            "import numpy as np\ndef f(a, out):\n s = 0.0\n for i in range(len(a)):\n  s += a[i]\n out[0] = s\n",
            np.arange(5, dtype=np.float64),
            {"N": 5},
            "(N,)",
            id="loop-bound",
        ),
    ],
)
def test_len_of_2d_array_is_first_dim(src: str, a: np.ndarray, params: dict[str, int], shape: str) -> None:
    ok, res = all_ok(
        run_op(src, "f", {"a": a}, {"out": (1,)}, params, shapes={"a": shape, "out": "(1,)"}, backends=ALL)
    )
    assert ok, res
