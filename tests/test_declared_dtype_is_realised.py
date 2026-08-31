# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A manifest's declared array dtype must be the dtype the run actually materialises.

The declaration is not documentation: it is what the emitters type the C/Fortran parameter with.
When the two disagree the buffer still binds -- ctypes hands over an address and nothing checks
what is behind it -- so the kernel reads one width through a pointer of another and returns
numbers rather than an error. floyd_warshall handed a float64 ``path`` through ``int32_t *``;
needleman_wunsch and smith_waterman did the same with their base-code sequences.

The disagreement only appears at a precision the initializer was not written for: each of those
three honours its ``datatype`` argument for an array the manifest pins, so they agree at the
initializer's own default and diverge the moment a run asks for fp64. That is why this test pins
the REALISED dtype at an explicit precision rather than trusting the default.

``int4`` is declared and stored as ``int8`` (numpy has no int4), so the comparison is against
:func:`hpcagent_bench.dtypes.storage_dtype`, which is the same rule ``sizing.working_bytes`` uses
to weigh the buffer.
"""
from typing import Dict, List, Tuple

import numpy as np
import pytest

from hpcagent_bench.dtypes import storage_dtype
from hpcagent_bench.frameworks.benchmark import Benchmark
from hpcagent_bench.spec import KERNELS

#: The precision the check runs at. fp64 is what every job submission asks for
#: (scripts/submit_xl.sbatch pins DATATYPE=float64), so it is the precision a disagreement is
#: reached at in practice.
PRECISION = "float64"

KERNEL_NAMES = sorted(KERNELS.select_keys("all"))


def declared_dtypes(spec) -> Dict[str, str]:
    """``{array: dtype}`` the manifest declares, from either spelling of the init block."""
    out: Dict[str, str] = {}
    if spec.init is None:
        return out
    for name, entry in (spec.init.shapes or {}).items():
        if isinstance(entry, dict) and "dtype" in entry:
            out[name] = entry["dtype"]
    for name, dtype in (spec.init.dtypes or {}).items():
        out.setdefault(name, dtype)
    return out


def disagreements(key: str) -> List[Tuple[str, str, str]]:
    """``(array, declared, realised)`` for every array of ``key`` whose dtype does not match."""
    spec = KERNELS.specs()[key]
    declared = declared_dtypes(spec)
    if not declared:
        return []
    data = Benchmark(key).get_data("S", datatype=PRECISION)
    bad = []
    for name, want in declared.items():
        value = data.get(name)
        if not isinstance(value, np.ndarray):
            continue
        expected = np.dtype(storage_dtype(want)).name
        if value.dtype.name != expected:
            bad.append((name, expected, value.dtype.name))
    return bad


@pytest.mark.parametrize("key", KERNEL_NAMES)
def test_every_declared_array_dtype_is_the_one_materialised(key: str) -> None:
    bad = disagreements(key)
    assert not bad, ("declared dtype is not the one the run materialises at "
                     f"{PRECISION}: " + ", ".join(f"{n}: declared {w}, got {g}" for n, w, g in bad))
