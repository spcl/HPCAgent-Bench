# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The GPU profiler's trace-and-read paths (:mod:`hpcagent_bench.harness.gpu_profiling`) end to end.

Runs on a host with no GPU, no ``nsys`` and no ROCm: the subprocess boundaries answer with the real
CSV fixtures ``tests/test_gpu_profiling.py`` carries, so everything between the profiler's exit and
the ``/profile`` payload is the production code.
"""

import pytest

from hpcagent_bench.harness import gpu_profiling
from tests.test_gpu_profiling import NSYS_STATS, ROCPROF_CSVS


def without_share_column(text: str) -> str:
    """``text`` with its share column renamed to something no alias matches."""
    return text.replace("Time (%)", "Share of time").replace('"Percentage"', '"Share"')


@pytest.mark.parametrize(
    "rows",
    [
        gpu_profiling.parse_csv(
            without_share_column(gpu_profiling.split_reports(NSYS_STATS)[gpu_profiling.KERNEL_REPORT])
        ),
        gpu_profiling.parse_csv(without_share_column(ROCPROF_CSVS[gpu_profiling.KERNEL_STATS_CSV])),
    ],
    ids=["nsys", "rocprofv3"],
)
def test_a_kernel_report_without_a_share_column_is_refused_rather_than_read_as_zero(rows) -> None:
    """Read as 0.0, every kernel fell below min_percent: the profile came back with no kernels,
    every one counted as omitted, and the empty-trace check let it through as a measurement."""
    with pytest.raises(gpu_profiling.GpuProfilerUnavailable) as caught:
        gpu_profiling.kernel_stats(rows, 1.0)
    assert caught.value.cause == "kernel_share_missing", caught.value.cause
    assert "Share" in str(caught.value), str(caught.value)
