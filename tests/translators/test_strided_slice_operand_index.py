# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A STRIDED slice used as an operand must be READ with its stride.

``iter_extent_of`` has always divided a strided slice's trip count by ``|step|``
(``a[0::2]`` over 12 elements -> 6 iterations), but its partner
``scalarize_at_iters`` -- which renders the operand at each of those iterations --
read only ``slice.lower`` and ignored ``step`` entirely. Extent and index therefore
disagreed: the loop ran the right NUMBER of times over the WRONG elements, walking
the first ``n // k`` entries contiguously instead of every ``k``-th one, and a
``a[::-1]`` operand was read FORWARD.

Silent in every case -- the shapes match, so nothing OOB, nothing to warn about:

* ``np.maximum(a[0::2], b)`` over ``a = 1..12`` gave ``[1 2 3 4 5 6]``, not ``[1 3 5 7 9 11]``;
* ``np.dot(a[0::2], ones(6))`` gave 21 instead of 36;
* ``np.maximum(a[::-1], 0)`` gave the un-reversed array.

The bare strided slice-ASSIGN (``lo[:] = b[0::2]``, dwt2d's Haar) goes through
lowering's own slice rewriters and was always correct; it is pinned here too so the
shared-index fix cannot regress it.
"""

import ast

import numpy as np
import pytest

from hpcagent_bench.translators.numpyto_common.lib_nodes import scalarize_at_iters
from tests.translators.op_oracle import assert_all_ok, run_op

NATIVE = ("c", "cpp", "fortran")
A12 = np.arange(1.0, 13.0)
ZERO6 = np.zeros(6)


def index_of(expr: str, shape: tuple[str]) -> str:
    """Render ``expr`` at a single iter ``i`` and return the unparsed index text."""
    node = ast.parse(expr, mode="eval").body
    out = scalarize_at_iters(node, [ast.Name(id="i", ctx=ast.Load())], {"a": shape})
    return ast.unparse(ast.fix_missing_locations(out))


# structural: the rendered index carries the stride


def test_full_axis_reverse_counts_down_from_the_last_element() -> None:
    assert index_of("a[::-1]", ("N",)) == "a[i * -1 + (N - 1)]"
    assert index_of("a[::-2]", ("12",)) == "a[i * -2 + (12 - 1)]"


# numerical: every backend matches numpy


@pytest.mark.parametrize(
    ("src", "inputs", "outputs", "n", "shapes"),
    [
        (
            "import numpy as np\ndef f(a, b, out):\n    out[:] = np.maximum(a[0::2], b)\n",
            {"a": A12, "b": ZERO6},
            {"out": (6,)},
            12,
            {"a": "(12,)", "b": "(6,)", "out": "(6,)"},
        ),
        (
            "import numpy as np\ndef f(a, b, out):\n    out[:] = np.maximum(a[::-1], b)\n",
            {"a": np.arange(1.0, 7.0), "b": ZERO6},
            {"out": (6,)},
            6,
            {"a": "(6,)", "b": "(6,)", "out": "(6,)"},
        ),
        (
            "import numpy as np\ndef f(a, b, out):\n    out[0] = np.dot(a[0::2], b)\n",
            {"a": A12, "b": np.ones(6)},
            {"out": (1,)},
            12,
            {"a": "(12,)", "b": "(6,)", "out": "(1,)"},
        ),
        # dwt2d's Haar shape: a bare strided slice ASSIGN, which lowering (not the
        # operand scalarizer) rewrites. Was always correct -- pinned against regression.
        (
            "import numpy as np\ndef f(a, lo, hi):\n    lo[:] = a[0::2]\n    hi[:] = a[1::2]\n",
            {"a": A12},
            {"lo": (6,), "hi": (6,)},
            12,
            {"a": "(12,)", "lo": "(6,)", "hi": "(6,)"},
        ),
    ],
    ids=["elementwise_strided_operand", "elementwise_reverse_operand", "dot_strided_operand", "strided_slice_assign"],
)
def test_strided_operand_matches_numpy(
    src: str, inputs: dict[str, np.ndarray], outputs: dict[str, tuple[int, ...]], n: int, shapes: dict[str, str]
) -> None:
    assert_all_ok(run_op(src, "f", inputs, outputs, {"N": n}, shapes=shapes, backends=NATIVE))
