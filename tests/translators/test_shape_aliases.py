# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shape-manipulation aliases -> the transpose / reshape loop-lowering.

``np.swapaxes`` / ``np.expand_dims`` / ``np.squeeze`` are NumPy sugar over transpose and
reshape; the C / Fortran backends (which emit explicit loops, unlike numba
that run the numpy verbatim) get them for free by delegating to the existing transpose /
reshape expanders, once ``iter_extent_of`` learns their output shape. These validate the
emitted code numerically against numpy for param AND intermediate-local operands (the
ML-reshape case), across the full backend matrix (C / C++ / Fortran + numba,
skip-tolerant). The kept ``swapaxes`` negative-axes and ``expand_dims`` middle-axis (keyword
form) cases subsume the positive-axis and trailing-axis variants.

The last two cover the way the ML corpus writes a squeeze that never reaches the expander at
all: BACK TO BACK on the trailing axes, which the front end rewrites to a chained subscript
first, plus the rank-independence guard the extent fold behind it rests on.
"""

import ast

import numpy as np
import pytest

from hpcagent_bench.translators.numpyto_common.lowering import is_newaxis_result_axis
from tests.translators.op_oracle import all_ok, run_op

ALL = ("c", "cpp", "fortran", "numba")


def copy_src(setup: str) -> str:
    """A kernel running ``setup`` (binding ``b``) and copying the 2-D ``b`` into ``out``."""
    return (
        "import numpy as np\n"
        "def k(a, out):\n"
        f"{setup}"
        "    for i in range(out.shape[0]):\n"
        "        for j in range(out.shape[1]):\n"
        "            out[i, j] = b[i, j]\n"
    )


EXPAND_DIMS_SRC = (
    "import numpy as np\n"
    "def k(a, out):\n"
    "    b = np.expand_dims(a, axis=1)\n"
    "    for i in range(a.shape[0]):\n"
    "        for j in range(a.shape[1]):\n"
    "            out[i, 0, j] = b[i, 0, j]\n"
)


@pytest.mark.parametrize(
    ("src", "a_shape", "out_shape", "a_decl", "out_decl"),
    [
        (copy_src("    b = np.swapaxes(a, -1, -2)\n"), (3, 4), (4, 3), "(M, N)", "(N, M)"),
        # The operand is an intermediate local (``tmp = a + 1``), whose shape the machinery
        # infers -- the ML case where a reshape follows a computed tensor.
        (copy_src("    tmp = a + 1.0\n    b = np.swapaxes(tmp, 0, 1)\n"), (3, 4), (4, 3), "(M, N)", "(N, M)"),
        (EXPAND_DIMS_SRC, (3, 4), (3, 1, 4), "(M, N)", "(M, 1, N)"),
        (copy_src("    b = np.squeeze(a, 1)\n"), (3, 1, 4), (3, 4), "(M, 1, N)", "(M, N)"),
        (copy_src("    b = np.squeeze(a)\n"), (1, 3, 1, 4), (3, 4), "(1, M, 1, N)", "(M, N)"),
        # ``np.squeeze(np.squeeze(b, axis=-1), axis=-1)`` (the global-pool tail) is rewritten to the
        # CHAINED subscript ``b[:, :, :, 0][:, :, 0]`` before any expander runs. ``b`` is a local, so
        # its rank is not declared -- but the outer indices land inside the inner ``:`` positions, where
        # the collapse to ``b[:, :, 0, 0]`` holds at every rank.
        (
            copy_src("    b = a * 2.0\n    b = np.squeeze(np.squeeze(b, axis=-1), axis=-1)\n"),
            (3, 4, 1, 1),
            (3, 4),
            "(M, N, 1, 1)",
            "(M, N)",
        ),
    ],
    ids=[
        "swapaxes_negative_axes",
        "swapaxes_intermediate_local_operand",
        "expand_dims_middle_axis",
        "squeeze_named_axis",
        "squeeze_all_unit_axes",
        "squeeze_back_to_back_on_the_trailing_axes",
    ],
)
def test_shape_alias_matches_numpy_on_every_backend(
    src: str, a_shape: tuple[int, ...], out_shape: tuple[int, ...], a_decl: str, out_decl: str
) -> None:
    a = np.arange(12, dtype=np.float64).reshape(a_shape)
    ok, res = all_ok(
        run_op(
            src,
            "k",
            {"a": a},
            {"out": out_shape},
            {"M": 3, "N": 4},
            shapes={"a": a_decl, "out": out_decl},
            backends=ALL,
        )
    )
    assert ok, res


def test_a_newaxis_extent_folds_without_the_operand_rank_but_nothing_else_does() -> None:
    """``X[:, None, :].shape[1]`` is 1 for every ``X``, which is what lets an ``expand_dims``
    operand's extent resolve before that operand's own shape is harvested -- the squeeze in
    ``np.squeeze(_pool(np.expand_dims(x, 1)), axis=1)`` is only provably dropping a unit axis
    because of it. Every other position is rank-DEPENDENT: a scalar index consumes a source
    axis, an Ellipsis stands for an unknown number of them, and a trailing axis is not named
    at all, so folding any of those without the rank would state the wrong extent."""

    def axis(text: str, k: int) -> bool:
        return is_newaxis_result_axis(ast.parse(text, mode="eval").body, k)

    assert axis("X[:, None, :]", 1)
    assert axis("X[None, :]", 0)
    assert not axis("X[:, None, :]", 0), "a full slice takes its extent from the operand"
    assert not axis("X[0, None, :]", 1), "a scalar index consumes a source axis, shifting the map"
    assert not axis("X[..., None]", 1), "an Ellipsis stands for an unknown number of axes"
    assert not axis("X[idx, None]", 1), "a gather contributes its index array's own rank"
    assert not axis("X[:, None]", -1), "a negative axis counts from a rank that is not known"
    assert not axis("X[:, None]", 4), "past the named axes the result axis is a trailing one"
