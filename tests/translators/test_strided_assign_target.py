# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A STRIDED slice is a legal ASSIGNMENT TARGET, not only a legal read.

``SliceFusion`` scalarises a slice-assign into one loop per slice axis, and must honour a
``step`` on the target as the read side does (``lo[:] = b[0::2]``), or the interleave half of
every wavelet/FFT/pack kernel refuses (``NotImplementedError: slice step != 1 not supported``)
while the deinterleave half lowers.

The store iterates the LOGICAL position ``k`` and writes ``start + k * step``, which is
also why the RHS mapping needs no division: it already reads the iter var as the
position. A unit step keeps the plain shape exactly -- iter var IS the coordinate, no
stray ``* 1``, no extra bound arithmetic.

A NEGATIVE step on the target stays refused: numpy seeds the reverse start at
``axis_len - 1`` when the bound is omitted, and a local array's axis length is not
always in ``array_shapes`` -- silently emitting ``-k`` would write before the buffer.
"""

import ast

import numpy as np
import pytest

from hpcagent_bench.translators.numpyto_common.lowering import SliceFusion
from tests.translators.op_oracle import assert_all_ok, run_op

NATIVE = ("c", "cpp", "fortran")


def fuse(src: str, shapes: dict) -> str:
    tree = SliceFusion(shapes).visit(ast.parse(src))
    return ast.unparse(ast.fix_missing_locations(tree))


# structural: the emitted store carries the stride


def test_strided_target_writes_every_kth_element() -> None:
    assert fuse("out[0::2] = a[:]", {"out": ["12"], "a": ["6"]}) == (
        "for si0 in range(0, 6):\n    out[si0 * 2] = a[si0]"
    )


def test_stride_applies_per_axis_only_where_written() -> None:
    assert fuse("out[0::2, :] = c[:, :]", {"out": ["4", "3"], "c": ["2", "3"]}) == (
        "for si0 in range(0, 2):\n    for si1 in range(0, 3):\n        out[si0 * 2, si1] = c[si0, si1]"
    )


def test_negative_step_target_is_refused_not_miscompiled() -> None:
    with pytest.raises(NotImplementedError, match="negative slice step"):
        fuse("out[::-1] = a[:]", {"out": ["6"], "a": ["6"]})


# numerical: every native backend matches numpy


def test_interleave_matches_numpy() -> None:
    # The inverse of dwt2d's Haar deinterleave, written as a strided slice assignment
    # rather than an index loop.
    src = "import numpy as np\ndef f(lo, hi, out):\n    out[0::2] = lo\n    out[1::2] = hi\n"
    assert_all_ok(
        run_op(
            src,
            "f",
            {"lo": np.arange(1.0, 7.0), "hi": np.arange(101.0, 107.0)},
            {"out": (12,)},
            {"N": 12},
            shapes={"lo": "(6,)", "hi": "(6,)", "out": "(12,)"},
            backends=NATIVE,
        )
    )


def test_strided_target_with_offset_start_matches_numpy() -> None:
    src = "import numpy as np\ndef f(a, out):\n    out[1:10:3] = a\n"
    assert_all_ok(
        run_op(
            src,
            "f",
            {"a": np.arange(1.0, 4.0)},
            {"out": (12,)},
            {"N": 12},
            shapes={"a": "(3,)", "out": "(12,)"},
            backends=NATIVE,
        )
    )


def test_strided_target_of_a_strided_read_matches_numpy() -> None:
    src = "import numpy as np\ndef f(a, out):\n    out[0::2] = a[1::2]\n"
    assert_all_ok(
        run_op(
            src,
            "f",
            {"a": np.arange(1.0, 13.0)},
            {"out": (12,)},
            {"N": 12},
            shapes={"a": "(12,)", "out": "(12,)"},
            backends=NATIVE,
        )
    )


def test_strided_row_target_of_a_2d_array_matches_numpy() -> None:
    src = "import numpy as np\ndef f(a, out):\n    out[0::2, :] = a\n"
    assert_all_ok(
        run_op(
            src,
            "f",
            {"a": np.arange(1.0, 7.0).reshape(2, 3)},
            {"out": (4, 3)},
            {"N": 4},
            shapes={"a": "(2,3)", "out": "(4,3)"},
            backends=NATIVE,
        )
    )
