# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A float32 kernel compares against a float constant expression in double, as its references do (``1.0 - 1e-8``
is exactly 1.0 in float)."""

from collections.abc import Callable

import pytest

from hpcagent_bench.translators.numpyto_c.emit import emit_c
from hpcagent_bench.translators.numpyto_common.ir import KernelIR, apply_precision
from hpcagent_bench.translators.numpyto_common.lowering import lower
from hpcagent_bench.translators.numpyto_fortran.emit import emit_fortran
from tests.translators.op_oracle import parse_source

SOURCE = (
    "import numpy as np\n"
    "def f(za, out):\n"
    "    for i in range(N):\n"
    "        if za[i] > 1.0 - 1e-8:\n"
    "            out[i] = 1.0\n"
)
SHAPES = {"za": "(N,)", "out": "(N,)"}


def lowered(precision: str) -> KernelIR:
    """The kernel at ``precision`` the way the oracle emits it (``--precision``: :func:`apply_precision`)."""
    return apply_precision(lower(parse_source(SOURCE, "f", ["za"], ["out"], SHAPES, {"N": 8})), precision)


@pytest.mark.parametrize(
    ("emit", "present", "absent"),
    [
        (emit_c, "0.99999999", "1e-08f"),
        (emit_fortran, "0.99999999_c_double", None),
    ],
    ids=["c", "fortran"],
)
def test_a_float32_kernel_compares_against_the_folded_double_constant(
    emit: Callable[..., str], present: str, absent: str | None
) -> None:
    src = emit(lowered("float32"), fn_name="f")
    assert present in src
    if absent is not None:
        assert absent not in src
