# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``np.stack`` (new axis) and ``np.concatenate`` (existing axis) -> copy loop nests.

``np.concatenate`` was already lowered (join operands along an existing axis); ``np.stack``
adds a NEW axis (out rank = operand rank + 1, the k-th extent = number of operands) and is
lowered here. Both emit per-operand copy loops for the C / Fortran backends (numba
runs the numpy verbatim). Validated numerically vs numpy across the full backend matrix
(C / C++ / Fortran + numba, skip-tolerant). The ``negative_axis_appends`` case
subsumes the positive last-axis case (``-1`` normalizes to it).
"""

import numpy as np
import pytest

from tests.translators.op_oracle import all_ok, run_op

ALL = ("c", "cpp", "fortran", "numba")


A = np.arange(6, dtype=np.float64).reshape(2, 3)
B = np.arange(6, 12, dtype=np.float64).reshape(2, 3)


def stack3d(axis: int, out_shape: tuple[int, int, int], out_sym: str, n_operands: int = 2) -> dict[str, str]:
    seq = "(a, b)" if n_operands == 2 else "(a, b, a)"
    src = (
        "import numpy as np\n"
        "def k(a, b, out):\n"
        f"    c = np.stack({seq}, axis={axis})\n"
        "    for i in range(out.shape[0]):\n"
        "        for j in range(out.shape[1]):\n"
        "            for l in range(out.shape[2]):\n"
        "                out[i, j, l] = c[i, j, l]\n"
    )
    return run_op(
        src,
        "k",
        {"a": A, "b": B},
        {"out": out_shape},
        {"M": 2, "N": 3},
        shapes={"a": "(M, N)", "b": "(M, N)", "out": out_sym},
        backends=ALL,
    )


@pytest.mark.parametrize(
    ("axis", "out_shape", "out_sym"),
    [
        (0, (2, 2, 3), "(2, M, N)"),
        (1, (2, 2, 3), "(M, 2, N)"),
        # ``axis=-1`` (a UnaryOp literal, not Constant(-1)) appends the new axis.
        (-1, (2, 3, 2), "(M, N, 2)"),
    ],
    ids=["axis0", "axis1", "negative_axis_appends"],
)
def test_stack_inserts_the_new_axis_where_asked(axis: int, out_shape: tuple[int, int, int], out_sym: str) -> None:
    ok, res = all_ok(stack3d(axis, out_shape, out_sym))
    assert ok, res


# concatenate regression (join along an existing axis) -- already supported.   #


def concat(axis: int, out_shape: tuple[int, int], out_sym: str, n_operands: int = 2) -> dict[str, str]:
    seq = "(a, b)" if n_operands == 2 else "(a, b, a)"
    src = (
        "import numpy as np\n"
        "def k(a, b, out):\n"
        f"    c = np.concatenate({seq}, axis={axis})\n"
        "    for i in range(out.shape[0]):\n"
        "        for j in range(out.shape[1]):\n"
        "            out[i, j] = c[i, j]\n"
    )
    return run_op(
        src,
        "k",
        {"a": A, "b": B},
        {"out": out_shape},
        {"M": 2, "N": 3},
        shapes={"a": "(M, N)", "b": "(M, N)", "out": out_sym},
        backends=ALL,
    )


@pytest.mark.parametrize(
    ("axis", "out_shape", "out_sym", "n_operands"),
    [(0, (4, 3), "(2 * M, N)", 2), (1, (2, 6), "(M, 2 * N)", 2), (0, (6, 3), "(3 * M, N)", 3)],
    ids=["axis0", "axis1", "three_operands"],
)
def test_concatenate_joins_operands_along_an_existing_axis(
    axis: int, out_shape: tuple[int, int], out_sym: str, n_operands: int
) -> None:
    ok, res = all_ok(concat(axis, out_shape, out_sym, n_operands))
    assert ok, res
