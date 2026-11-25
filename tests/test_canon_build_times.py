# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The build-time distribution a paper quotes (p50/p90/max and the slowest kernel) per reported phase -- optimize,
tune, compile (codegen plus compile) -- read from canon rows (:func:`hpcagent_bench.stats.canon.build_times`)."""

from hpcagent_bench.stats import canon


def row(
    kernel: str,
    transform: float | None,
    codegen: float | None,
    compile_: float | None,
    tune: float | None = None,
    status: str = "ok",
) -> dict[str, object]:
    return {
        "column": "dace_cpu_canon",
        "kernel": kernel,
        "status": status,
        "transform_time": transform,
        "tune_time": tune,
        "codegen_time": codegen,
        "compile_time": compile_,
    }


def test_each_reported_phase_and_the_total_are_summarized_with_the_slowest_kernel() -> None:
    """Compile is codegen plus compile; a column with no tune phase reports none."""
    rows = [row("gemm", 1.0, 0.5, 2.0), row("cloudsc", 157.0, 10.0, 60.0), row("jacobi", 0.5, 0.2, 1.0)]
    by_phase = {entry.phase: entry for entry in canon.build_times(rows)}
    assert set(by_phase) == {"optimize", "compile", "total"}
    assert (by_phase["optimize"].max, by_phase["optimize"].slowest) == (157.0, "cloudsc")
    assert by_phase["compile"].p50 == 2.5
    assert (by_phase["total"].p50, by_phase["total"].kernels) == (3.5, 3)


def test_a_tuner_reports_its_tune_time_in_the_total() -> None:
    by_phase = {entry.phase: entry for entry in canon.build_times([row("gemm", 1.0, 0.5, 2.0, tune=40.0)])}
    assert (by_phase["tune"].p50, by_phase["total"].p50) == (40.0, 43.5)


def test_an_unmeasured_or_failed_row_counts_in_no_distribution() -> None:
    rows = [row("gemm", None, None, None), row("crashed", 99.0, 1.0, 1.0, status="crash")]
    assert canon.build_times(rows) == []


def test_the_table_names_every_summarized_row() -> None:
    text = canon.format_build_times(canon.build_times([row("gemm", 1.0, 0.5, 2.0, tune=3.0)]))
    assert text.count("dace_cpu_canon") == 4
    assert [line.split()[1] for line in text.splitlines()[1:]] == ["optimize", "tune", "compile", "total"]

