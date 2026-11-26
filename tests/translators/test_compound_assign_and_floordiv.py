# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compound-assign and float floor-division emit gaps (C / C++ / Fortran).

* ``t //= v`` / ``t %= v`` have no direct numpy-faithful form (C ``%=`` is dividend-sign modulo,
  Fortran ``/`` truncates and ``x MOD y`` is invalid infix), so both emitters expand to
  ``t = t <op> v`` through the BinOp path (int_floor / python_mod).
* Fortran float ``a // b`` yields a real floor, not ``FLOOR(.., int64)`` -- an integer, undefined
  for ``|a/b| > 2^63`` (1e20 // 2 must be 5e19).
"""

import numpy as np
import pytest

from tests.translators.op_oracle import assert_all_ok, run_op

NATIVE = ("c", "cpp", "fortran")


INT64_ABQR = {"a": "int64", "b": "int64", "q": "int64", "r": "int64"}


@pytest.mark.parametrize(
    ("src", "inputs", "outputs", "symbols", "shapes", "dtypes", "backends"),
    [
        # //= and %= over the full sign matrix: numpy // floors toward -inf and % takes the divisor's sign, unlike
        # C/Fortran's native truncate/dividend-sign forms. The compound ops run DIRECTLY on the output buffers --
        # the shape a real kernel writes (run_op's complex-scratch probe skips itself for `//=` / `%=`, which have
        # no complex loop).
        (
            "import numpy as np\n"
            "def f(a, b, q, r):\n"
            "    for i in range(a.shape[0]):\n"
            "        q[i] = a[i]\n"
            "        q[i] //= b[i]\n"
            "        r[i] = a[i]\n"
            "        r[i] %= b[i]\n",
            {"a": np.array([-7, 7, -8, 8], dtype=np.int64), "b": np.array([2, -2, 3, -3], dtype=np.int64)},
            {"q": (4,), "r": (4,)},
            {"N": 4},
            {"a": "(N,)", "b": "(N,)", "q": "(N,)", "r": "(N,)"},
            INT64_ABQR,
            NATIVE,
        ),
        # tuple-unpack int_locals were declared as bare 32-bit C int; a literal grid whose pairwise product exceeds
        # 2^31 (46341*46341 = 2147488281) wrapped negative.
        (
            "import numpy as np\ndef f(out):\n    nx, ny = 46341, 46341\n    out[0] = nx * ny\n",
            {},
            {"out": (1,)},
            {},
            {"out": "(1,)"},
            None,
            ("c", "cpp"),
        ),
        # 1e20 // 2 == 5e19 in numpy; a Fortran FLOOR(.., int64) overflows int64 here.
        (
            "import numpy as np\ndef f(a, b, out):\n    for i in range(a.shape[0]):\n        out[i] = a[i] // b[i]\n",
            {"a": np.array([1e20, -7.5, 7.5, 3.0], dtype=np.float64), "b": np.array([2.0, 2.0, -2.0, 2.0])},
            {"out": (4,)},
            {"N": 4},
            {"a": "(N,)", "b": "(N,)", "out": "(N,)"},
            None,
            NATIVE,
        ),
        # int(a[i]) is an INTEGER however float a is, so ``int(a[i]) // 2`` must take the integer floor-div branch.
        # A float-operand test that walks into the cast sees float ``a`` and picks ``floor((x) / (y))``, but both
        # emitted operands are int64 there, so C truncates toward zero and the floor is a no-op: int(-7.5) // 2
        # gives -3 instead of numpy's -4 (same for %).
        (
            "import numpy as np\n"
            "def f(a, q, r):\n"
            "    for i in range(a.shape[0]):\n"
            "        q[i] = int(a[i]) // 2\n"
            "        r[i] = int(a[i]) % 3\n",
            {"a": np.array([-7.5, -1.5, 3.5, 7.5], dtype=np.float64)},
            {"q": (4,), "r": (4,)},
            {"N": 4},
            {"a": "(N,)", "q": "(N,)", "r": "(N,)"},
            {"q": "int64", "r": "int64"},
            NATIVE,
        ),
        # b[i // 2] is an integer floor-div used as an array index. It must take the integer FloorDiv branch and
        # emit an INTEGER index -- a real-valued index is rejected by Fortran -std=f2018 (regression: loop
        # induction vars must type as int64).
        (
            "import numpy as np\ndef f(a, b):\n    for i in range(a.shape[0]):\n        b[i // 2] = b[i // 2] + a[i]\n",
            {"a": np.arange(1.0, 9.0)},
            {"b": (4,)},
            {"N": 8},
            {"a": "(N,)", "b": "(4,)"},
            None,
            NATIVE,
        ),
    ],
    ids=["int-compound-floordiv-mod", "literal-grid-unpack", "float-floordiv-overflow", "int-cast", "floordiv-index"],
)
def test_compound_int_floordiv_and_mod_match_numpy(
    src: str,
    inputs: dict[str, np.ndarray],
    outputs: dict[str, tuple[int, ...]],
    symbols: dict[str, int],
    shapes: dict[str, str],
    dtypes: dict[str, str] | None,
    backends: tuple[str, ...],
) -> None:
    res = run_op(src, "f", inputs, outputs, symbols, shapes=shapes, dtypes=dtypes, backends=backends)
    assert_all_ok(res)
