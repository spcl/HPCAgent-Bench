# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The GPU profiler's trace-and-read paths (:mod:`hpcagent_bench.harness.gpu_profiling`) end to end.

Runs on a host with no GPU, no ``nsys`` and no ROCm: the subprocess boundaries answer with the real
CSV fixtures ``tests/test_gpu_profiling.py`` carries, so everything between the profiler's exit and
the ``/profile`` payload is the production code.
"""

import json
import pathlib
import subprocess

import pytest

from hpcagent_bench.harness import gpu_profiling, profiling
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


def wedge(argv: list[str], **kwargs: object) -> None:
    """A profiler still running at its deadline, as subprocess reports one: by raising."""
    raise subprocess.TimeoutExpired(argv, 3.0)


def result_line(elapsed_ns: int = 600_000, reps: int = 3) -> str:
    return profiling.RESULT_PREFIX + json.dumps({"elapsed_ns": elapsed_ns, "reps": reps}) + "\n"


def nsys_records_then_stats_wedge(monkeypatch: pytest.MonkeyPatch) -> None:
    def record(argv: list[str], *, cwd: str, timeout: float) -> subprocess.CompletedProcess[str]:
        (pathlib.Path(cwd) / (gpu_profiling.REPORT_STEM + ".nsys-rep")).write_text("")
        return subprocess.CompletedProcess(argv, 0, stdout=result_line(), stderr="")

    monkeypatch.setattr(gpu_profiling, "run_command", record)
    monkeypatch.setattr(gpu_profiling.subprocess, "run", wedge)


@pytest.mark.parametrize(
    "language,stage",
    [
        ("cuda", lambda mp: mp.setattr(gpu_profiling, "run_command", wedge)),
        ("cuda", nsys_records_then_stats_wedge),
        ("hip", lambda mp: mp.setattr(gpu_profiling, "run_command", wedge)),
    ],
    ids=["nsys-record", "nsys-stats", "rocprof-record"],
)
def test_a_wedged_gpu_profiler_is_a_timed_out_refusal_not_a_raw_timeout(tmp_path, monkeypatch, language, stage) -> None:
    """subprocess signals a deadline by raising, and the route turned that raw exception into a 500
    with no cause, which an agent cannot tell apart from a broken judge."""
    monkeypatch.setattr(gpu_profiling, "nsys_check", lambda language: "/fake/bin/nsys")
    monkeypatch.setattr(gpu_profiling, "rocprof_check", lambda: ("rocprofv3", "/fake/bin/rocprofv3"))
    stage(monkeypatch)
    with pytest.raises(gpu_profiling.GpuProfilerUnavailable) as caught:
        gpu_profiling.profile_gpu_once(
            tmp_path, tmp_path / "request.json", language=language, timeout=3.0, min_percent=1.0
        )
    assert caught.value.cause == "timed_out", caught.value.cause
    assert "3s" in str(caught.value), str(caught.value)
