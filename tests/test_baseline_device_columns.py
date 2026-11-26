# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A baseline sweep must know which columns build for a device, or it runs one with no GPU behind it.

The sweep asks the column's registered device (``arch``). The framework decides by its own table
(``native_runtime.FRAMEWORK_LANG``) for the native columns, and the two had drifted before: a name rule (``*gpu*``)
matched ``dace_gpu*`` and missed every PPCG column, so ``ppcg_hip``, the AMD CUDA->HIP column, was submitted with no
GPU at all.
"""

import pytest

from hpcagent_bench.categories import Language
from hpcagent_bench.cluster import baseline
from hpcagent_bench.frameworks.native_runtime import FRAMEWORK_LANG

DEVICE_LANGUAGES = (Language.HIP, Language.CUDA)


DEVICE_COLUMNS = sorted(c for c, lang in FRAMEWORK_LANG.items() if lang in DEVICE_LANGUAGES)
# dace and tvm are not cpp columns, so they are absent from FRAMEWORK_LANG and need their own entries.
GPU_FLAVORS = ["dace_gpu_parallel", "dace_gpu_canon", "dace_gpu_autoopt", "tvm_gpu"]
CPU_COLUMNS = ["numba", "cc", "cpp", "fortran", "pluto", "dace_cpu_parallel", "dace_cpu_canon", "tvm_cpu"]


@pytest.mark.parametrize(
    ("column", "is_device"),
    [(c, True) for c in DEVICE_COLUMNS] + [(c, True) for c in GPU_FLAVORS] + [(c, False) for c in CPU_COLUMNS],
)
def test_every_device_column_the_framework_builds_is_known_to_the_sweep(column: str, is_device: bool) -> None:
    """A device column is one the framework builds for a GPU language or a dace/tvm GPU flavor; a CPU column
    would be reported as missing GPUs it never uses."""
    assert baseline.is_device_column(column) is is_device, (
        f"{column} builds {FRAMEWORK_LANG.get(column)} and is {'a device' if is_device else 'a CPU'} column"
    )
