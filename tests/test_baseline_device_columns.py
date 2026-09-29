# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A baseline sweep must know which columns build for a device, or it runs one with no GPU behind it.

The sweep decides by NAME (``baseline.DEVICE_COLUMNS``), so it needs no Python environment to ask. The framework
decides by its own table (``cpp_runtime.FRAMEWORK_LANG``). Nothing tied the two together, and they had drifted: the
test was ``*gpu*``, which matched ``dace_gpu*`` and missed every PPCG column, so ``ppcg_hip``, the AMD CUDA->HIP
column, was submitted with no GPU at all.
"""

import pytest

from hpcagent_bench.benchmarks.cpp_runtime import FRAMEWORK_LANG
from hpcagent_bench.cluster import baseline

DEVICE_LANGUAGES = ("hip", "cuda")


@pytest.mark.parametrize("column", sorted(c for c, lang in FRAMEWORK_LANG.items() if lang in DEVICE_LANGUAGES))
def test_every_device_column_the_framework_builds_is_known_to_the_sweep(column: str) -> None:
    assert baseline.is_device_column(column), (
        f"{column} builds {FRAMEWORK_LANG[column]} but the sweep takes it for a CPU one"
    )


@pytest.mark.parametrize("column", ["dace_gpu", "dace_gpu_canonicalize", "dace_gpu_autoopt"])
def test_the_dace_device_columns_are_known_to_the_sweep(column: str) -> None:
    """dace is not a cpp column, so it is absent from FRAMEWORK_LANG and needs its own check."""
    assert baseline.is_device_column(column)


@pytest.mark.parametrize(
    "column", ["numba", "cc", "cpp", "fortran", "pluto", "dace_cpu", "dace_cpu_parallel", "dace_cpu_canonicalize"]
)
def test_a_cpu_column_is_not_a_device_column(column: str) -> None:
    assert not baseline.is_device_column(column), f"{column} would be reported as missing GPUs it never uses"
