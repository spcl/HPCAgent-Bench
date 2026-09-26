# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The MI300A -> GH200 transfer figure and the platform column that keeps the two grades apart."""

import math
import pathlib

import pandas as pd
import pytest

from hpcagent_bench import experiments
from hpcagent_bench.harness.timing import FINAL_GRADE_REDUCTION
from hpcagent_bench.stats import population, style
from hpcagent_bench.stats.figures import transfer

CPU_ARM = "cpf-llr-focus40-qwen38-c"
GPU_ARM = "gpu-llr-focus40-oss120b-hip"
DB = "hpcagent-bench-runs/cpf-llr-focus40-20260917/640098/judge/rank-0/hpcagent_bench0.db"


def observation(ts: int, speedup: float, platform: str, record: str = "submission", status: str = "graded") -> dict:
    """One graded row with the extractor's column names."""
    return {
        "run_root": "cpf-llr-focus40-20260917",
        "job": "640098",
        "db": DB,
        "row_kind": record,
        "run_id": f"{CPU_ARM}.n0.p{ts}.w{ts}",
        "arm": CPU_ARM,
        "language": "c",
        "benchmark": f"k{ts}",
        "ts_ms": ts,
        "attempt_index": 0,
        "speedup": speedup,
        "timing_suspect": 0,
        "timing_reduction": FINAL_GRADE_REDUCTION,
        "grade_final_status": status,
        "platform": platform,
    }


def paired_rows(rows: list[tuple[str, float, float, str]]) -> pd.DataFrame:
    """A paired frame of ``(language, mi300a, gh200, status)`` rows, one kernel each."""
    return pd.DataFrame(
        [
            {
                "benchmark": f"k{index}",
                "arm": GPU_ARM if language in ("hip", "triton") else CPU_ARM,
                "model": "oss120b" if language in ("hip", "triton") else "qwen38",
                "language": language,
                "speedup_mi300a": x,
                "mi300a_grade": transfer.FINAL_GRADE,
                "speedup_gh200": y,
                "status_gh200": status,
            }
            for index, (language, x, y, status) in enumerate(rows)
        ],
        columns=list(transfer.PAIRED_COLUMNS),
    )


#: CPU: four answers correct on both machines (x ranks 1 2 3 4, y ranks 1 3 2 4: Spearman 0.8), one
#: that failed on GH200, one the judge failed to grade there, and two C answers GH200 could not build.
CPU_ROWS = [
    ("c", 1.0, 1.5, transfer.CORRECT),
    ("c", 2.0, 6.0, transfer.CORRECT),
    ("fortran", 3.0, 4.0, transfer.CORRECT),
    ("c", 4.0, 9.0, transfer.CORRECT),
    ("c", 5.0, math.nan, transfer.FAILED),
    ("c", 6.0, math.nan, transfer.ERRORED),
    ("c", math.nan, math.nan, transfer.NOT_PORTABLE),
    ("c", math.nan, math.nan, transfer.NOT_PORTABLE),
]


def test_existing_readers_select_mi300a_and_never_see_a_gh200_row(tmp_path: pathlib.Path) -> None:
    """The GH200 re-timing shares its answer's key; read_observations keeps it out unless asked."""
    path = tmp_path / "obs.csv"
    pd.DataFrame([observation(1, 2.0, "mi300a"), observation(1, 8.0, "gh200"), observation(2, 3.0, "")]).to_csv(
        path, index=False
    )
    default = experiments.read_observations(path)
    assert sorted(default["speedup"]) == [2.0, 3.0]
    assert list(experiments.read_observations(path, population.GH200_PLATFORM)["speedup"]) == [8.0]


def test_a_statistic_over_both_platforms_is_refused() -> None:
    frame = pd.DataFrame([observation(1, 2.0, "mi300a"), observation(2, 8.0, "gh200")])
    with pytest.raises(population.MixedPopulationError, match="mixes platforms"):
        population.graded_episode_rows(frame, tainted=())
    assert len(population.graded_episode_rows(population.on_platform(frame), tainted=())) == 1


def test_a_frame_without_the_column_is_mi300a() -> None:
    frame = pd.DataFrame([observation(1, 2.0, "mi300a")]).drop(columns="platform")
    assert len(population.on_platform(frame)) == 1 and population.on_platform(frame, "gh200").empty


def test_panel_reports_the_rank_correlation_and_correct_share() -> None:
    stats = transfer.panel_stats(paired_rows(CPU_ROWS), "CPU")
    assert stats.spearman == pytest.approx(0.8)
    assert (stats.correct, stats.failed, stats.errored) == (4, 2, 1), "a judge error is a failure on GH200"
    assert stats.correct_share == pytest.approx(4 / 6)
    assert stats.not_portable == {"c": 2, "fortran": 0}


def test_the_geomean_is_over_answers_solved_on_both_machines() -> None:
    """C holds three answers solved on both machines; the failed, errored and unportable ones enter
    no mean. Three answers are under six, so the interval is withheld."""
    table = transfer.geomean_table(paired_rows(CPU_ROWS)).set_index(["language", "model"])
    c = table.loc[("c", "qwen38")]
    assert c["n"] == 3
    assert c["gm_mi300a"] == pytest.approx((1.0 * 2.0 * 4.0) ** (1 / 3))
    assert c["gm_gh200"] == pytest.approx((1.5 * 6.0 * 9.0) ** (1 / 3))
    assert math.isnan(c["ci_low_mi300a"]) and math.isnan(c["ci_high_gh200"])
    assert table.loc[("fortran", "qwen38"), "n"] == 1


def test_each_slot_draws_mi300a_filled_beside_gh200_hollow() -> None:
    fig = transfer.geomean_figure(paired_rows(CPU_ROWS))
    marks = [c for c in fig.axes[0].collections if len(c.get_offsets())]
    filled = [c for c in marks if c.get_facecolors()[0][:3].tolist() != [1.0, 1.0, 1.0]]
    hollow = [c for c in marks if c.get_facecolors()[0][:3].tolist() == [1.0, 1.0, 1.0]]
    assert len(filled) == len(hollow) == 2, "one MI300A and one GH200 mark per language with answers"
    assert all(f.get_offsets()[0][0] < h.get_offsets()[0][0] for f, h in zip(filled, hollow))


def test_a_panel_with_no_answer_solved_on_both_is_a_pending_stub(tmp_path: pathlib.Path) -> None:
    fig = transfer.geomean_figure(paired_rows(CPU_ROWS))
    gpu = fig.axes[1]
    assert [c.get_gid() for c in gpu.collections] == [style.PENDING_GID]
    style.save(fig, tmp_path / "transfer", formats=("pdf",), width_in=style.ICLR_WRAP_WIDTH_IN)
    assert fig.get_size_inches()[1] <= 3.2


def test_the_scatter_draws_only_answers_solved_on_both_machines(tmp_path: pathlib.Path) -> None:
    """The unsolved answer and the judge error are counted as failures (summary table), never drawn;
    the unportable answers neither."""
    fig = transfer.scatter_figure(paired_rows(CPU_ROWS))
    cpu, gpu = fig.axes
    assert cpu.get_title() == "CPU  $\\rho$ = 0.80"
    xs = sorted(x for c in cpu.collections for x, _ in c.get_offsets())
    assert xs == [1.0, 2.0, 3.0, 4.0]
    assert [c.get_gid() for c in gpu.collections] == [style.PENDING_GID]
    style.save(fig, tmp_path / "scatter", formats=("pdf",), width_in=style.ICLR_WRAP_WIDTH_IN)
    assert fig.get_size_inches()[1] <= 3.6


def test_a_model_with_no_answer_in_a_language_takes_no_slot() -> None:
    """Fortran holds one qwen38 answer and no oss120b one: one Fortran slot, beside C's one."""
    rows = [*CPU_ROWS, ("c", 2.0, 3.0, transfer.CORRECT)]
    paired = paired_rows(rows)
    paired.loc[paired.index[-1], ["arm", "model"]] = ["cpf-llr-focus40-oss120b-c", "oss120b"]
    cpu = transfer.geomean_figure(paired).axes[0]
    counts = sorted(text.get_text() for text in cpu.texts)
    assert counts == ["1", "1", "3"], "C qwen38 3, C oss120b 1, Fortran qwen38 1; no Fortran oss120b 0"


def test_a_wide_scatter_axis_labels_every_other_decade_from_1x() -> None:
    assert transfer.decade_ticks(0.05, 50.0) == [0.1, 1.0, 10.0]
    assert transfer.decade_ticks(0.2, 3000.0) == [1.0, 100.0]


def test_models_outside_the_paper_are_left_out() -> None:
    rows = paired_rows([("c", 2.0, 3.0, transfer.CORRECT)])
    glm = rows.assign(arm="cpf-llr-focus40-glm53-c-skills", model="glm53")
    assert transfer.kept_arms(pd.concat([rows, glm])).model.tolist() == ["qwen38"]


def test_observations_pair_each_gh200_row_with_the_answer_it_re_timed() -> None:
    mi300a = pd.DataFrame([observation(1, 2.0, "mi300a"), observation(2, 3.0, "mi300a"), observation(3, 4.0, "mi300a")])
    gh200 = pd.DataFrame(
        [
            observation(1, 6.0, "gh200"),
            observation(2, math.nan, "gh200", record="attempt", status="unsolved"),
            observation(3, math.nan, "gh200", status="error"),
        ]
    )
    paired = transfer.paired_from_observations(mi300a, gh200).set_index("benchmark")
    assert list(paired["status_gh200"]) == [transfer.CORRECT, transfer.FAILED, transfer.ERRORED]
    assert list(paired["speedup_mi300a"]) == [2.0, 3.0, 4.0]
    assert paired.loc["k1", "speedup_gh200"] == 6.0 and paired["speedup_gh200"].isna().sum() == 2
    assert set(paired["language"]) == {"c"} and set(paired["mi300a_grade"]) == {transfer.FINAL_GRADE}


def test_the_join_table_falls_back_to_the_live_grade_and_drops_retired_arms() -> None:
    table = pd.DataFrame(
        {
            "arm": [CPU_ARM, CPU_ARM, GPU_ARM, "gpu-llr-focus40-qwen38-triton"],
            "benchmark": ["k0", "k1", "k2", "k3"],
            "backend": ["c", "c", "hip", "triton"],
            "s_bar_mi300a": [2.0, math.nan, math.nan, 3.0],
            "grade_live_speedup": [1.5, 4.0, math.nan, 3.0],
            "s_bar_gh200": [5.0, math.nan, math.nan, 2.0],
            "status_mi300a": ["graded", "", "", "graded"],
            "status_gh200": ["graded", "graded", "not-portable", "graded"],
        }
    )
    paired = transfer.paired_from_csv(table).set_index("benchmark")
    assert list(paired.index) == ["k0", "k1", "k2"], "the host-resident triton arm is a dropped arm"
    assert list(paired["status_gh200"]) == [transfer.CORRECT, transfer.FAILED, transfer.NOT_PORTABLE]
    assert paired.loc["k1", "speedup_mi300a"] == 4.0 and paired.loc["k1", "mi300a_grade"] == transfer.LIVE_GRADE
    assert list(paired["model"]) == ["qwen38", "qwen38", "oss120b"]
