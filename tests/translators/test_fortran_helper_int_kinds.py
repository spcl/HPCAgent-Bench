# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A kept helper types its integer locals' bitwise literals exactly as the kernel body does.

``IAND`` / ``IOR`` / ``IEOR`` need both arguments of one kind, so a bare literal paired with a
typed local takes that local's kind suffix. The kernel body read the kind from the local's
declaration; the contained helper mapped every non-int64 integer local to ``int32``, so an
``integer(c_int16_t)`` local met a ``3_c_int32_t`` literal and gfortran rejected the ``IAND``.
"""

import json
import pathlib
import tempfile

from hpcagent_bench.translators.numpyto_common.frontend import parse_kernel
from hpcagent_bench.translators.numpyto_common.lowering import lower
from hpcagent_bench.translators.numpyto_fortran.emit import emit_fortran
from hpcagent_bench.translators.numpyto_fortran.intrinsics import renders_natively

# The early return keeps ``h`` a contained subroutine instead of inlining it.
SRC = (
    "import numpy as np\n"
    "def h(x):\n"
    " m = np.int16(x[0])\n"
    " r = m & 3\n"
    " if m < 0:\n"
    "  return 0.0\n"
    " return r * 1.0\n"
    "def f(x, out):\n"
    " out[0] = h(x)\n"
    " m = np.int16(x[1])\n"
    " out[0] += m & 3\n"
)


def emitted() -> str:
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "k_numpy.py").write_text(SRC)
    bench = {
        "name": "k",
        "short_name": "k",
        "relative_path": "",
        "module_name": "k",
        "func_name": "f",
        "parameters": {"S": {"N": 8}},
        "input_args": ["x", "out"],
        "array_args": ["x", "out"],
        "output_args": ["out"],
        "init": {"shapes": {"x": "(N,)", "out": "(1,)"}},
    }
    (d / "bi.json").write_text(json.dumps({"benchmark": bench}))
    kir = lower(parse_kernel(d / "k_numpy.py", d / "bi.json"), native_call=renders_natively)
    return emit_fortran(kir, fn_name="f")


def test_helper_and_kernel_suffix_an_int16_bitwise_literal_alike() -> None:
    src = emitted()
    kernel, helper = src.split("contains", 1)
    assert "IAND(m, 3_c_int16_t)" in kernel, kernel
    assert "IAND(m, 3_c_int16_t)" in helper, helper
    assert "3_c_int32_t" not in src, src
