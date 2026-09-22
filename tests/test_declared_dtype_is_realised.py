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

#: The precision the check runs at where the kernel supports it. fp64 is what every job submission
#: asks for (scripts/submit_xl.sbatch pins DATATYPE=float64), so it is the precision a
#: disagreement is reached at in practice.
PRECISION = "float64"
#: The Precision-enum spelling of :data:`PRECISION`, to test a manifest's ``precisions`` list against.
PRECISION_NAME = "fp64"

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


def check_precision(spec) -> str:
    """The precision to materialise ``spec`` at: :data:`PRECISION` where the manifest declares
    fp64, else the first precision it DOES declare.

    A kernel that pins one narrow precision -- the bf16 distributed ML operators -- is never run
    at fp64, so materialising it at fp64 compares a declaration against a run that does not
    exist. Every fp64-declaring kernel (678 of the 689) keeps the fp64 check unchanged.
    """
    return PRECISION if PRECISION_NAME in spec.precisions else spec.precisions[0]


def disagreements(key: str) -> List[Tuple[str, str, str]]:
    """``(array, declared, realised)`` for every array of ``key`` whose dtype does not match."""
    spec = KERNELS.specs()[key]
    declared = declared_dtypes(spec)
    if not declared:
        return []
    data = Benchmark(key).get_data("S", datatype=check_precision(spec))
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
    at = check_precision(KERNELS.specs()[key])
    assert not bad, f"declared dtype is not the one the run materialises at {at}: " + ", ".join(
        f"{n}: declared {w}, got {g}" for n, w, g in bad
    )


def undeclared_integer_arrays(key: str) -> List[Tuple[str, str]]:
    """``(array, realised)`` for every integer array of ``key`` the manifest leaves undeclared."""
    spec = KERNELS.specs()[key]
    if spec.init is None:
        return []
    declared = declared_dtypes(spec)
    data = Benchmark(key).get_data("S", datatype=PRECISION)
    bad = []
    for name in spec.array_args:
        value = data.get(name)
        if isinstance(value, np.ndarray) and value.dtype.kind in "iub" and name not in declared:
            bad.append((name, value.dtype.name))
    return bad


@pytest.mark.parametrize("key", KERNEL_NAMES)
def test_every_integer_array_declares_its_dtype(key: str) -> None:
    """The other direction: an array left undeclared is typed at the run precision.

    That is right for a float array and a wrong ABI for an integer one: the emitted signature takes
    ``double *`` while the harness binds an int32 buffer, so the kernel reads the bytes as doubles.
    lavamd left its three box tables undeclared: the run read a garbage
    neighbour count out of ``neighbor_counts`` and sized a transient from it, and the CPU canon
    column died on ``std::bad_array_new_length``. pathfinder carried the same declaration gap.
    """
    bad = undeclared_integer_arrays(key)
    assert not bad, "an integer array the initializer materialises is not declared: " + ", ".join(
        f"{name}: {dtype}" for name, dtype in bad
    )
