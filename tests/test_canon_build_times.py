# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The build-time distribution a paper quotes (p50/p90/max and the slowest kernel), read from canon rows'
``transform_time`` / ``codegen_time`` / ``compile_time`` (:func:`hpcagent_bench.stats.canon.build_times`)."""

from hpcagent_bench.phase_clock import Phase
from hpcagent_bench.stats import canon


def row(
    kernel: str, transform: float | None, codegen: float | None, compile_: float | None, status: str = "ok"
) -> dict[str, object]:
    return {
        "column": "dace_cpu_canon",
        "kernel": kernel,
        "status": status,
        "transform_time": transform,
        "codegen_time": codegen,
        "compile_time": compile_,
    }


def test_each_phase_and_the_total_are_summarized_with_the_slowest_kernel() -> None:
    rows = [row("gemm", 1.0, 0.5, 2.0), row("cloudsc", 157.0, 10.0, 60.0), row("jacobi", 0.5, 0.2, 1.0)]
    by_phase = {entry.phase: entry for entry in canon.build_times(rows)}
    assert set(by_phase) == {Phase.TRANSFORM, Phase.CODEGEN, Phase.COMPILE, None}
    assert by_phase[Phase.TRANSFORM].max == 157.0
    assert by_phase[Phase.TRANSFORM].slowest == "cloudsc"
    assert by_phase[None].p50 == 3.5
    assert by_phase[None].kernels == 3


def test_an_unmeasured_or_failed_row_counts_in_no_distribution() -> None:
    rows = [row("gemm", 1.0, None, 2.0), row("crashed", 99.0, 1.0, 1.0, status="crash")]
    by_phase = {entry.phase: entry for entry in canon.build_times(rows)}
    assert Phase.CODEGEN not in by_phase, "a NULL phase is not measured"
    assert None not in by_phase, "no total without every phase"
    assert by_phase[Phase.TRANSFORM].kernels == 1


def test_the_table_names_every_summarized_row() -> None:
    text = canon.format_build_times(canon.build_times([row("gemm", 1.0, 0.5, 2.0)]))
    assert text.count("dace_cpu_canon") == 4
    assert [line.split()[1] for line in text.splitlines()[1:]] == ["transform", "codegen", "compile", "total"]
