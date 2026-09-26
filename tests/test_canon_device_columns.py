# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The canon submitter must request GPUs for every column the framework builds for a device.

submit-canon.sh decides by NAME, in shell, so it needs no Python environment at submit time.
The framework decides by its own table (cpp_runtime.FRAMEWORK_LANG). Nothing tied the two
together, and they had drifted: the shell test was `*gpu*`, which matched dace_gpu* and missed
every PPCG column -- so ppcg_hip, the AMD CUDA->HIP column, was submitted with no GPU at all.
"""

import fnmatch
import re

import pytest

from hpcagent_bench import paths
from hpcagent_bench.benchmarks.cpp_runtime import FRAMEWORK_LANG

SUBMITTER = paths.ROOT / "experiments" / "submit-canon.sh"
DEVICE_LANGUAGES = ("hip", "cuda")


def _shell_device_patterns() -> list[str]:
    """The glob patterns on the line that sets --gres."""
    for line in SUBMITTER.read_text().splitlines():
        if "gres=(--gres=gpu" in line and "[[" in line:
            return re.findall(r'"\$\{(?:col|one)\}" == (\S+?)\s*(?:\|\||\]\])', line)
    raise AssertionError("no --gres decision line found in submit-canon.sh")


def _shell_says_device(column: str) -> bool:
    return any(fnmatch.fnmatchcase(column, pattern) for pattern in _shell_device_patterns())


@pytest.mark.parametrize("column", sorted(c for c, lang in FRAMEWORK_LANG.items() if lang in DEVICE_LANGUAGES))
def test_every_device_column_the_framework_builds_is_given_gpus(column: str) -> None:
    assert _shell_says_device(column), (
        f"{column} builds {FRAMEWORK_LANG[column]} but submit-canon.sh would submit it with "
        f"no GPU (patterns: {_shell_device_patterns()})"
    )


@pytest.mark.parametrize("column", ["dace_gpu", "dace_gpu_canonicalize", "dace_gpu_autoopt"])
def test_the_dace_device_columns_are_given_gpus(column: str) -> None:
    """dace is not a cpp column, so it is absent from FRAMEWORK_LANG and needs its own check."""
    assert _shell_says_device(column)


@pytest.mark.parametrize(
    "column", ["numba", "cc", "cpp", "fortran", "pluto", "dace_cpu", "dace_cpu_parallel", "dace_cpu_canonicalize"]
)
def test_a_cpu_column_is_not_given_gpus(column: str) -> None:
    """--exclusive already takes the node; a spurious --gres only lengthens the queue wait."""
    assert not _shell_says_device(column), f"{column} would be submitted asking for GPUs it never uses"


def _packed_needs_gpu(packed: str) -> bool:
    """Mirror the submitter: a packed job needs GPUs if ANY member column does."""
    return any(_shell_says_device(one) for one in packed.split(","))


@pytest.mark.parametrize(
    "packed, wants_gpu",
    [
        ("numba,ppcg_hip", True),  # the smoke job that exposed this: a device column packed second
        ("ppcg_hip,numba", True),
        ("numba,cc,fortran", False),
        ("dace_cpu,dace_gpu", True),
    ],
)
def test_a_packed_job_gets_gpus_if_any_member_is_a_device_column(packed: str, wants_gpu: bool) -> None:
    """With TIME_LIMIT_ONE_JOB the columns ride one comma-joined value. Testing the JOINED string
    missed "numba,ppcg_hip" -- it neither contains "gpu" nor starts with "ppcg" -- so the submitter
    has to test each member, and this checks the per-member rule rather than the joined one."""
    text = SUBMITTER.read_text()
    assert "for one in ${col//,/ }" in text, "submitter no longer tests each packed column separately"
    assert _packed_needs_gpu(packed) is wants_gpu
