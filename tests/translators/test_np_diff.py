# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``np.diff`` is the first difference, so it lowers to ``A[..., 1:, ...] - A[..., :-1, ...]``.

It reached the native emitters unexpanded and died there with ``call to np.diff not supported`` --
a refusal, not a miscompile, but one that pushed kernels into writing the subtraction by hand.
cp2k_density_matrix_trs4 hit it on ``np.repeat(np.arange(n), np.diff(row_ptr))``, the standard CSR
row-length idiom.

The axis shortens by exactly one, which is the part a shape-preserving assumption gets wrong: the
output is NOT the operand's shape, so ``diff`` also has to leave ``ELEMENTWISE_SHAPE_OPS``. Only the
first difference expands -- ``n > 1`` needs a temporary per stage, and ``prepend=``/``append=`` are a
concatenate the caller can spell.
"""

import ast

import numpy as np
import pytest

from hpcagent_bench.translators.numpyto_common.lib_nodes import expand_diff
from tests.translators.op_oracle import assert_all_ok, run_op

NATIVE = ("c", "cpp", "fortran")


def expand_(src: str, shapes: dict) -> str:
    """Run the registered expander over the single ``out = np.diff(...)`` statement in ``src``."""
    assign = ast.parse(src).body[0]
    stmts = expand_diff(assign.targets[0], assign.value.args, shapes, assign.value.keywords)
    return "\n".join(ast.unparse(ast.fix_missing_locations(s)) for s in stmts)


@pytest.mark.parametrize(
    ("src", "shapes", "expected"),
    [
        (
            "out = np.diff(a)",
            {"a": ("n",)},
            "for __df0 in range(n - 1):\n    out[__df0] = a[__df0 + 1] - a[__df0]",
        ),
        (
            "out = np.diff(a)",
            {"a": ("r", "c")},
            "for __df0 in range(r):\n"
            "    for __df1 in range(c - 1):\n"
            "        out[__df0, __df1] = a[__df0, __df1 + 1] - a[__df0, __df1]",
        ),
        (
            "out = np.diff(a, axis=0)",
            {"a": ("r", "c")},
            "for __df0 in range(r - 1):\n"
            "    for __df1 in range(c):\n"
            "        out[__df0, __df1] = a[__df0 + 1, __df1] - a[__df0, __df1]",
        ),
    ],
    ids=["1d_neighbour_subtraction", "2d_defaults_to_last_axis", "2d_axis_zero_walks_rows"],
)
def test_diff_expands_to_neighbour_subtraction_along_its_axis(src: str, shapes: dict, expected: str) -> None:
    assert expand_(src, shapes) == expected


@pytest.mark.parametrize(
    ("src", "shapes", "match"),
    [
        ("out = np.diff(a, 2)", {"a": ("n",)}, "first difference"),
        ("out = np.diff(a, prepend=0)", {"a": ("n",)}, "concatenate"),
        ("out = np.diff(a, axis=k)", {"a": ("r", "c")}, "constant int"),
    ],
    ids=["n_greater_than_one", "prepend_as_concatenate", "nonconstant_axis"],
)
def test_diff_forms_beyond_the_first_difference_are_refused(src: str, shapes: dict, match: str) -> None:
    with pytest.raises(NotImplementedError, match=match):
        expand_(src, shapes)


CSR_ROW_PTR = np.array([0, 2, 2, 5, 9], dtype=np.int64)
SQUARES = np.arange(12, dtype=np.float64).reshape(3, 4) ** 2


@pytest.mark.parametrize(
    ("src", "inputs", "out_shape", "symbols", "shapes", "dtypes"),
    [
        # The idiom that found the gap: row lengths out of a CSR ``row_ptr``.
        pytest.param(
            "import numpy as np\n\ndef k(row_ptr, out):\n    out[:] = np.diff(row_ptr)\n",
            {"row_ptr": CSR_ROW_PTR},
            (4,),
            {"n": 5},
            {"row_ptr": "(5,)", "out": "(4,)"},
            {"row_ptr": "int64", "out": "int64"},
            id="csr-row-lengths",
        ),
        pytest.param(
            "import numpy as np\n\ndef k(a, out):\n    out[:, :] = np.diff(a, axis=0)\n",
            {"a": SQUARES},
            (2, 4),
            {},
            {"a": "(3,4)", "out": "(2,4)"},
            None,
            id="2d-axis-zero",
        ),
    ],
)
def test_numeric_row_lengths_from_csr_row_ptr(
    src: str,
    inputs: dict[str, np.ndarray],
    out_shape: tuple[int, ...],
    symbols: dict[str, int],
    shapes: dict[str, str],
    dtypes: dict[str, str] | None,
) -> None:
    options = {} if dtypes is None else {"dtypes": dtypes}
    assert_all_ok(run_op(src, "k", inputs, {"out": out_shape}, symbols, shapes=shapes, backends=NATIVE, **options))
