# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``hpcagent_bench.stats.figures.kernel_comparison`` -- the llr-focus40 canon-vs-agents figure.

Condition comes from the ARM NAME (:data:`kernel_comparison.ARM_PATTERN`), never the
``language``/``packet`` columns, because the pre-regrade extraction records those inconsistently
for the same arm. Completeness is roster coverage (:func:`population.complete_arms`), applied
before any per-kernel value is read, and a model whose every arm is incomplete gets no panel.

ORIENTATION: the MEASURED quantity is on Y (speed-up on the top panel, tokens on the bottom) and the
kernel NAMES are on X, rotated, on one axis both panels share; the geomean/median summary is a group
past the last kernel at the RIGHT END of that axis, behind a dashed vertical separator. The drawing
itself is :mod:`hpcagent_bench.stats.figures.per_kernel`'s (tests/test_plot_per_kernel.py); what is
asserted here is that this figure's data reaches it with the right values and statuses.
"""

import math
import pathlib

import matplotlib.axes
import pandas as pd
import pytest

from hpcagent_bench.stats import cost, palette, population
from hpcagent_bench.stats import style as plotstyle
from hpcagent_bench.stats.figures import kernel_comparison, per_kernel, results

ROSTER: tuple[str, ...] = ("k1", "k2", "k3")


def submission_rows(arm: str, benchmark_speedups: dict[str, float], baseline: str = "numba") -> list[dict[str, object]]:
    """One graded episode per (arm, kernel): the columns ``population.kernel_answers`` needs.

    ``baseline`` is the denominator the judge stamped on the row -- ``numba`` for loop_level_reasoning
    and ``c-autopar`` for scientific_computing (``harness.grading.TRACK_DEFAULT_BASELINE``)."""
    rows = []
    for benchmark, speedup in benchmark_speedups.items():
        run = f"{arm}-{benchmark}"
        rows.append(
            {
                "run_root": "j1",
                "job": "j1",
                "run_id": run,
                "arm": arm,
                "row_kind": "submission",
                "benchmark": benchmark,
                "speedup": speedup,
                "baseline_ns": 1000.0,
                "native_ns": 1000.0 / speedup,
                "baseline": baseline,
                "timing_suspect": 0,
                "ts_ms": 1,
                "attempt_index": 1,
                "timing_reduction": "mwd-v2",
            }
        )
    return rows


def call_rows(arm: str, benchmark_tokens: dict[str, float]) -> list[dict[str, object]]:
    """One ``call`` row per (arm, kernel): a running count mid-task, NEVER a token cost (T4) --
    used only to test that a frame with call rows and no task rows is refused for tokens."""
    rows = []
    for benchmark, tokens in benchmark_tokens.items():
        run = f"{arm}-{benchmark}"
        rows.append(
            {
                "run_root": "j1",
                "job": "j1",
                "run_id": run,
                "arm": arm,
                "row_kind": "call",
                "benchmark": benchmark,
                "tokens": tokens,
                "ts_ms": 1,
                "attempt_index": 1,
            }
        )
    return rows


def task_rows(
    arm: str, benchmark_tokens: dict[str, float], ts_ms: int = 1, run_suffix: str = ""
) -> list[dict[str, object]]:
    """One ``row_kind=task`` row per (arm, kernel): the columns ``population.kernel_tokens`` needs
    (T2-T4) -- one row per worker directory, ``tokens`` the task's own effective total.
    ``run_suffix`` distinguishes several tasks of the same kernel (a rerun or a designed repeat)."""
    rows = []
    for benchmark, tokens in benchmark_tokens.items():
        run = f"{arm}-{benchmark}{run_suffix}"
        rows.append(
            {
                "run_root": f"j1{run_suffix}",
                "job": f"j1{run_suffix}",
                "run_id": run,
                "arm": arm,
                "row_kind": "task",
                "benchmark": benchmark,
                "tokens": tokens,
                # the task total as fresh input alone, so every cost card prices it at ``tokens``
                "tokens_fresh_input": tokens,
                "tokens_cached_input": 0.0,
                "tokens_output": 0.0,
                "ts_ms": ts_ms,
            }
        )
    return rows


def observations(rows: list[dict[str, object]]) -> pd.DataFrame:
    """``rows`` as an extracted frame, which always carries ``tokens`` and its components (blank off a task row)."""
    frame = pd.DataFrame(rows)
    token_columns = ("tokens", *cost.COMPONENT_COLUMNS)
    return frame.reindex(columns=[*frame.columns, *(c for c in token_columns if c not in frame.columns)])


def canon_frame(rows: list[tuple[str, str, float, str]]) -> pd.DataFrame:
    """A ``canon`` table frame: (column, kernel, median_ms, validated)."""
    return pd.DataFrame(
        [{"run": "r1", "column": c, "kernel": k, "median_ms": ms, "validated": v} for c, k, ms, v in rows]
    )


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        ("cpf-llr-focus40-qwen38-c", ("qwen38", "")),
        ("cpf-llr-focus40-qwen38-c-cpf", ("qwen38", "cpf")),
        ("cpf-llr-focus40-oss120b-c-cpfsrc", ("oss120b", "cpfsrc")),
        ("cpf-llr-focus40-qwen38-fortran", None),
        ("cpf-llr-focus40-qwen38-c-skills", None),
    ],
)
def test_parse_arm_reads_model_and_condition_from_the_arm_name_only(arm: str, expected: tuple[str, str] | None) -> None:
    """The arm name is the one column every row of an arm agrees on in the pre-regrade db; language
    and packet are not read here at all."""
    assert kernel_comparison.parse_arm(arm) == expected


def test_candidate_arms_keeps_only_arms_the_pattern_names() -> None:
    frame = observations(
        [
            *submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0}),
            *submission_rows("cpf-llr-focus40-qwen38-fortran", {"k1": 2.0}),
            *submission_rows("cpf-llr-focus40-qwen38-c-skills", {"k1": 2.0}),
        ]
    )
    assert kernel_comparison.candidate_arms(frame) == {"cpf-llr-focus40-qwen38-c": ("qwen38", "")}


def test_an_incomplete_arm_is_dropped_and_its_coverage_reported() -> None:
    """An arm served on 2 of 3 roster kernels cannot be scored over the roster without inventing a
    value for the third, so it is dropped rather than entered at any policy's stand-in."""
    frame = observations(
        [
            *submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0, "k3": 2.0}),
            *submission_rows("cpf-llr-focus40-qwen38-c-cpf", {"k1": 2.0, "k2": 2.0}),
        ]
    )
    panels, _canon, dropped = kernel_comparison.build_panels(frame, ROSTER)
    kept_arms = {series.key for arms in panels.values() for series in arms}
    assert kept_arms == {"cpf-llr-focus40-qwen38-c"}
    assert dropped == {"cpf-llr-focus40-qwen38-c-cpf": 2}


def test_a_model_whose_every_arm_is_incomplete_gets_no_panel_at_all() -> None:
    """A model with one candidate arm, and that arm short of the roster, draws nothing -- not an
    empty panel."""
    frame = observations([*submission_rows("cpf-llr-focus40-oss120b-c", {"k1": 2.0})])
    panels, _canon, dropped = kernel_comparison.build_panels(frame, ROSTER)
    assert panels == {}
    assert dropped == {"cpf-llr-focus40-oss120b-c": 1}


def test_include_incomplete_draws_a_partial_arm_anyway() -> None:
    frame = observations([*submission_rows("cpf-llr-focus40-oss120b-c", {"k1": 2.0})])
    panels, _canon, dropped = kernel_comparison.build_panels(frame, ROSTER, include_incomplete=True)
    assert "oss120b" in panels
    assert dropped == {}


def test_the_canon_series_is_none_when_no_canon_frame_is_given() -> None:
    """git-scicomp reuses this module with no deterministic reference column at all."""
    frame = observations([*submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0, "k3": 2.0})])
    panels, canon_mark, _dropped = kernel_comparison.build_panels(frame, ROSTER)
    assert canon_mark is None
    assert "qwen38" in panels


def test_the_canon_series_reads_the_column_over_the_chosen_baseline() -> None:
    frame = observations([*submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0, "k3": 2.0})])
    canon = canon_frame(
        [
            ("numba", "k1", 100.0, "True"),
            ("numba", "k2", 100.0, "True"),
            ("numba", "k3", 100.0, "True"),
            ("dace_cpu_canonicalize", "k1", 25.0, "True"),
            ("dace_cpu_canonicalize", "k2", 50.0, "True"),
            ("dace_cpu_canonicalize", "k3", 100.0, "True"),
        ]
    )
    _panels, canon_mark, _dropped = kernel_comparison.build_panels(frame, ROSTER, canon_frame=canon)
    assert canon_mark is not None
    assert canon_mark.values == {"k1": 4.0, "k2": 2.0, "k3": 1.0}


def test_an_arms_condition_series_carries_the_packet_colour_the_arm_summary_figure_uses() -> None:
    """cpf and cpfsrc must draw as two distinct colours -- the treatments the paper compares -- and
    neither may collide with the control's neutral grey."""
    frame = observations(
        [
            *submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0, "k3": 2.0}),
            *submission_rows("cpf-llr-focus40-qwen38-c-cpf", {"k1": 2.0, "k2": 2.0, "k3": 2.0}),
            *submission_rows("cpf-llr-focus40-qwen38-c-cpfsrc", {"k1": 2.0, "k2": 2.0, "k3": 2.0}),
        ]
    )
    panels, _canon, _dropped = kernel_comparison.build_panels(frame, ROSTER)
    colours = {series.condition: series.color for series in panels["qwen38"]}
    assert len(set(colours.values())) == 3
    assert colours[""] == "#4d4d4d"


def test_a_kernel_with_no_verified_answer_is_absent_from_a_series_own_values() -> None:
    """``Series.values`` (what a mark is drawn from) follows the framework's own scoring policy,
    never inventing a served-but-unsolved value: an arm complete on the roster (a row exists) but
    with a negative/unusable speed-up on one kernel simply has no value there. The kernel still
    reaches the table and the figure as a missing mark -- see the ``table_rows`` tests below."""
    rows = submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0})
    rows.append(
        {
            "run_root": "j1",
            "job": "j1",
            "run_id": "cpf-llr-focus40-qwen38-c-k3",
            "arm": "cpf-llr-focus40-qwen38-c",
            "row_kind": "submission",
            "benchmark": "k3",
            "speedup": -1.0,
            "baseline_ns": 1000.0,
            "native_ns": 1000.0,
            "baseline": "numba",
            "timing_suspect": 0,
            "ts_ms": 1,
            "attempt_index": 1,
        }
    )
    frame = observations(rows)
    panels, _canon, dropped = kernel_comparison.build_panels(frame, ROSTER, include_incomplete=True)
    values = panels["qwen38"][0].values
    assert set(values) == {"k1", "k2"}
    assert dropped == {}


def test_table_rows_carries_one_row_per_series_per_roster_kernel_not_only_the_verified_ones() -> None:
    """Every roster kernel gets a row for every series, verified or not: a missing answer is a
    readable fact in the table (``status=no_verified_answer``, blank ``speedup``), never a silently
    absent row. Schema note: the table now also carries a ``tokens`` column and, past the per-kernel
    rows, one ``row=summary`` row per series (deliberately widened from the pre-tokens schema)."""
    frame = observations([*submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0})])
    canon = canon_frame([("numba", "k1", 100.0, "True"), ("dace_cpu_canonicalize", "k1", 50.0, "True")])
    panels, canon_mark, _dropped = kernel_comparison.build_panels(
        frame, ROSTER, canon_frame=canon, include_incomplete=True
    )
    table = kernel_comparison.table_rows(panels, canon_mark, ROSTER)

    assert set(table.columns) == {
        "kernel",
        "series",
        "kind",
        "model",
        "condition",
        "speedup",
        "tokens",
        "tokens_min",
        "tokens_max",
        "status",
        "row",
        "statistic",
        "value",
        "low",
        "high",
        "n_kernels",
    }
    kernel_rows = table[table.row == kernel_comparison.ROW_KERNEL]
    assert (kernel_rows.kind == "canon").sum() == len(ROSTER)  # every roster kernel, canon solved 1 of 3
    assert (kernel_rows.kind == "arm").sum() == len(ROSTER)  # every roster kernel, the arm solved 2 of 3

    canon_k3 = kernel_rows[(kernel_rows.kind == "canon") & (kernel_rows.kernel == "k3")].iloc[0]
    assert canon_k3.status == kernel_comparison.STATUS_MISSING
    assert canon_k3.speedup == ""

    arm_k3 = kernel_rows[(kernel_rows.kind == "arm") & (kernel_rows.kernel == "k3")].iloc[0]
    assert arm_k3.status == kernel_comparison.STATUS_MISSING
    assert arm_k3.speedup == ""

    arm_k1 = kernel_rows[(kernel_rows.kind == "arm") & (kernel_rows.kernel == "k1")].iloc[0]
    assert arm_k1.status == kernel_comparison.STATUS_VERIFIED
    assert arm_k1.speedup == 2.0


def test_summary_row_carries_the_geomean_of_the_plotted_per_kernel_speedups() -> None:
    """The ``row=summary`` ``statistic=geomean`` row's ``value`` equals the geometric mean of that
    series' own per-kernel speed-ups over the roster -- the same values the speed-up panel draws."""
    import math

    frame = observations([*submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 8.0, "k3": 2.0})])
    panels, _canon, _dropped = kernel_comparison.build_panels(frame, ROSTER)
    table = kernel_comparison.table_rows(panels, None, ROSTER)
    summary_row = table[(table.row == kernel_comparison.ROW_SUMMARY) & (table.statistic == "geomean")].iloc[0]
    assert math.isclose(float(summary_row.value), (2.0 * 8.0 * 2.0) ** (1.0 / 3.0))
    assert summary_row.n_kernels == 3


def test_summary_row_carries_the_geomean_interval_the_summary_slot_draws() -> None:
    """The table is read apart from the figure, so its summary row has to be the slot's statistic
    exactly: the geomean over the solved kernels (k3 unanswered is left out); two kernels are below
    the 6-kernel floor, so the interval is blank in both."""
    frame = observations([*submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 8.0})])
    panels, _canon_mark, _dropped = kernel_comparison.build_panels(frame, ROSTER, include_incomplete=True)
    series = panels["qwen38"][0]
    table = kernel_comparison.table_rows(panels, None, ROSTER)
    row = table[(table.row == kernel_comparison.ROW_SUMMARY) & (table.statistic == "geomean")].iloc[0]
    point, low, high = per_kernel.summary_geomean(kernel_comparison.speedup_series(series, ROSTER).cells)
    assert (row.value, row.n_kernels) == pytest.approx((point, 2))
    assert point == pytest.approx(4.0) and math.isnan(low) and math.isnan(high)
    assert (row.low, row.high) == ("", "")


def test_summary_row_carries_the_geomean_of_the_plotted_per_kernel_tokens() -> None:
    """The ``row=summary`` ``statistic=geomean_tokens`` row's ``value`` is the geometric mean of that
    series' own per-kernel token totals (paper rule): (100 * 300 * 200)^(1/3) = 181.71, not the median 200."""
    frame = observations(
        [
            *submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0, "k3": 2.0}),
            *task_rows("cpf-llr-focus40-qwen38-c", {"k1": 100.0, "k2": 300.0, "k3": 200.0}),
        ]
    )
    panels, _canon, _dropped = kernel_comparison.build_panels(frame, ROSTER)
    table = kernel_comparison.table_rows(panels, None, ROSTER)
    summary_row = table[(table.row == kernel_comparison.ROW_SUMMARY) & (table.statistic == "geomean_tokens")].iloc[0]
    assert summary_row.value == pytest.approx(181.71205928)
    assert summary_row.n_kernels == 3


def test_the_canon_series_carries_no_summary_token_row() -> None:
    """Canon has no tokens, so its series gets only the geomean speed-up summary row, never a median
    tokens one."""
    frame = observations([*submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0, "k3": 2.0})])
    canon = canon_frame(
        [
            ("numba", "k1", 100.0, "True"),
            ("numba", "k2", 100.0, "True"),
            ("numba", "k3", 100.0, "True"),
            ("dace_cpu_canonicalize", "k1", 50.0, "True"),
            ("dace_cpu_canonicalize", "k2", 50.0, "True"),
            ("dace_cpu_canonicalize", "k3", 50.0, "True"),
        ]
    )
    panels, canon_mark, _dropped = kernel_comparison.build_panels(frame, ROSTER, canon_frame=canon)
    table = kernel_comparison.table_rows(panels, canon_mark, ROSTER)
    canon_summary = table[(table.row == kernel_comparison.ROW_SUMMARY) & (table.series == "canon")]
    assert list(canon_summary.statistic) == ["geomean"]


def test_arm_tokens_reads_one_tasks_total_never_a_sum() -> None:
    """``arm_tokens`` reads a kernel's token total off its task record (T1-T4), never sums call
    rows -- a kernel with one task simply reports that task's own total."""
    frame = observations(
        [
            *submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0}),
            *task_rows("cpf-llr-focus40-qwen38-c", {"k1": 100.0}),
        ]
    )
    values, low, high = kernel_comparison.arm_tokens(frame, "cpf-llr-focus40-qwen38-c")
    assert values == {"k1": 100.0}
    assert low == {} and high == {}


def test_arm_tokens_reads_a_rerun_kernels_latest_task_total_not_the_sum_of_both() -> None:
    """A rerun kernel's token cell is the LATEST task's own total (R4): summing both tasks would
    bill an arm twice for being resubmitted, which the pre-2026-09-15 reduction did (spec F1)."""
    arm = "cpf-llr-focus40-qwen38-c"
    frame = observations(
        [
            *submission_rows(arm, {"k1": 2.0}),
            *task_rows(arm, {"k1": 400.0}, ts_ms=10, run_suffix="-w0"),
            *task_rows(arm, {"k1": 250.0}, ts_ms=30, run_suffix="-w1"),
        ]
    )
    values, low, high = kernel_comparison.arm_tokens(frame, arm)
    assert values == {"k1": 250.0}
    assert low == {} and high == {}


def test_arm_tokens_refuses_a_frame_with_call_rows_and_no_task_records() -> None:
    """``calls.tokens`` is a running count of the CURRENT attempt at a judge call (T4): a frame
    extracted before task records existed cannot cost an arm off it, and must say so rather than
    silently reading a partial total."""
    frame = observations(
        [
            *submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0}),
            *call_rows("cpf-llr-focus40-qwen38-c", {"k1": 900.0}),
        ]
    )
    with pytest.raises(population.MixedPopulationError, match="no task records"):
        kernel_comparison.arm_tokens(frame, "cpf-llr-focus40-qwen38-c")


def test_repeats_median_series_and_table_carry_tokens_min_and_max() -> None:
    """Designed repeats (R5): the kernel's token cell is the median over its tasks, bracketed by
    the minimum and maximum over those SAME tasks -- both on the ``Series`` and in the written
    table, where they draw as a whisker on the token panel."""
    arm = "cpf-llr-focus40-qwen38-c"
    rows = submission_rows(arm, {"k1": 2.0, "k2": 2.0, "k3": 2.0})
    for suffix, tokens, ts_ms in (("-w0", 100.0, 10), ("-w1", 400.0, 11), ("-w2", 250.0, 12)):
        rows += task_rows(arm, {"k1": tokens}, ts_ms=ts_ms, run_suffix=suffix)
    frame = observations(rows)

    panels, _canon, _dropped = kernel_comparison.build_panels(frame, ROSTER, repeats="median")
    series = panels["qwen38"][0]
    assert series.tokens == {"k1": 250.0}
    assert series.tokens_min == {"k1": 100.0}
    assert series.tokens_max == {"k1": 400.0}

    table = kernel_comparison.table_rows(panels, None, ROSTER)
    row = table[(table.row == kernel_comparison.ROW_KERNEL) & (table.kernel == "k1")].iloc[0]
    assert row.tokens == 250.0
    assert row.tokens_min == 100
    assert row.tokens_max == 400


def test_repeats_latest_series_and_table_carry_no_tokens_min_or_max() -> None:
    """Under the default ``--repeats latest`` one task IS the kernel's value, so there is nothing
    to bracket: the range columns stay blank."""
    frame = observations(
        [
            *submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0, "k3": 2.0}),
            *task_rows("cpf-llr-focus40-qwen38-c", {"k1": 100.0}),
        ]
    )
    panels, _canon, _dropped = kernel_comparison.build_panels(frame, ROSTER)
    series = panels["qwen38"][0]
    assert series.tokens_min == {} and series.tokens_max == {}

    table = kernel_comparison.table_rows(panels, None, ROSTER)
    row = table[(table.row == kernel_comparison.ROW_KERNEL) & (table.kernel == "k1")].iloc[0]
    assert row.tokens_min == "" and row.tokens_max == ""


def test_table_rows_writes_whole_token_totals_and_kernel_counts_as_integers() -> None:
    """N3: a raw task token total and a kernel count must be written as python ints, never floats
    with a trailing ``.0`` -- a table reader must not have to guess whether ``5.0`` means 5 exactly
    or a rounded statistic."""
    arm = "cpf-llr-focus40-qwen38-c"
    frame = observations(
        [
            *submission_rows(arm, {"k1": 2.0, "k2": 8.0, "k3": 2.0}),
            *task_rows(arm, {"k1": 100.0, "k2": 300.0, "k3": 200.0}),
        ]
    )
    panels, _canon, _dropped = kernel_comparison.build_panels(frame, ROSTER)
    table = kernel_comparison.table_rows(panels, None, ROSTER)

    kernel_row = table[(table.row == kernel_comparison.ROW_KERNEL) & (table.kernel == "k1")].iloc[0]
    assert isinstance(kernel_row.tokens, int)

    summary_row = table[(table.row == kernel_comparison.ROW_SUMMARY) & (table.statistic == "geomean")].iloc[0]
    assert isinstance(summary_row.n_kernels, int)


def test_the_control_condition_reads_no_packet_not_the_registry_skill_wording() -> None:
    """This figure's treatments (CPF page, CPF as source) are not skills, so its control must not
    borrow the skills experiments' "No Skill Packet" wording -- see
    ``hpcagent_bench.packets.control_label``."""
    assert kernel_comparison.condition_label("") == "No Packet"


def test_cpfsrc_reads_as_source_and_cpf_reads_as_the_page() -> None:
    """The two treatments the paper contrasts must read as two different THINGS, not two
    abbreviations of the same phrase."""
    assert kernel_comparison.condition_label("cpfsrc") == "Canonical Parallel Form as Source"
    assert kernel_comparison.condition_label("cpf") == "Canonical Parallel Form Page"


def test_git_scicomps_two_conditions_both_read_as_proper_names() -> None:
    """git-scicomp's own condition axis (no packet, no CPF): ``repo`` already read "Whole
    Repository" off the registry, but ``kernel`` fell through to the bare arm-name token because
    nothing named it there -- the legend read "kernel" beside "Repository Formulation", one condition
    properly named and the other not."""
    assert kernel_comparison.condition_label("repo") == "Git Reformulation"  # registry display name since 2026-09-25
    assert kernel_comparison.condition_label("kernel") == "Bare Kernel"


def figure_key(frame: pd.DataFrame) -> list[str]:
    """The labels of the one key :func:`kernel_comparison.figure` draws for ``frame``."""
    import matplotlib.pyplot as plt

    panels, canon_mark, _dropped = kernel_comparison.build_panels(frame, ROSTER, include_incomplete=True)
    fig = kernel_comparison.figure(panels, canon_mark, list(ROSTER), False, "title")
    try:
        (legend,) = fig.legends
        return [text.get_text() for text in legend.get_texts()]
    finally:
        plt.close(fig)


def test_the_key_names_the_placeholder_cross_only_when_a_kernel_draws_one() -> None:
    """An entry for a mark that is not on the figure is one more thing to read and find nowhere; a
    kernel nobody answered draws the cross, and then the key has to say what it means."""
    solved = observations([*submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0, "k3": 2.0})])
    missing = observations([*submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0})])
    assert plotstyle.NOT_DELIVERED_LABEL not in figure_key(solved)
    assert plotstyle.NOT_DELIVERED_LABEL in figure_key(missing)


def test_the_two_panels_share_one_kernel_x_axis_and_keep_their_own_value_y_scales() -> None:
    """Every model draws into the SAME two panels, so one model reaching 120x (or 900K tokens) must
    not stretch a scale the others are read against: the speed-up axis spans every series' values,
    not the first one's. The two panels share the kernel axis -- a kernel is at the same x in both --
    while their value axes stay separate, log2 ratio vs log10 count being different quantities."""
    import matplotlib

    matplotlib.use("Agg")
    frame = observations(
        [
            *submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0, "k3": 2.0}),
            *submission_rows("cpf-llr-focus40-oss120b-c", {"k1": 120.0, "k2": 120.0, "k3": 120.0}),
            *task_rows("cpf-llr-focus40-qwen38-c", {"k1": 100.0, "k2": 100.0, "k3": 100.0}),
            *task_rows("cpf-llr-focus40-oss120b-c", {"k1": 900000.0, "k2": 900000.0, "k3": 900000.0}),
        ]
    )
    panels, canon_mark, _dropped = kernel_comparison.build_panels(frame, ROSTER)
    fig = kernel_comparison.figure(panels, canon_mark, list(ROSTER), False, "title")
    try:
        assert len(fig.axes) == 2
        speedup_ax, token_ax = fig.axes
        assert list(speedup_ax.get_xticks()) == list(token_ax.get_xticks())
        assert speedup_ax.get_xlim() == token_ax.get_xlim()
        assert speedup_ax.get_ylim() != token_ax.get_ylim()
        # both models' extremes fit the ONE speed-up axis, and both spends the ONE token axis
        assert speedup_ax.get_ylim()[1] >= 120.0 and speedup_ax.get_ylim()[0] <= 2.0
        assert token_ax.get_ylim()[1] >= 900000.0 and token_ax.get_ylim()[0] <= 100.0
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_the_kernel_names_are_the_rotated_x_tick_labels_of_the_bottom_panel_only() -> None:
    """The kernel axis carries NAMES and runs along X, rotated so 40 of them fit; the two panels
    share it, so it is labelled once, at the foot of the figure. The top panel's own labels would
    repeat them into the gap between the panels."""
    import matplotlib

    matplotlib.use("Agg")
    frame = observations([*submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0, "k3": 2.0})])
    panels, canon_mark, _dropped = kernel_comparison.build_panels(frame, ROSTER)
    fig = kernel_comparison.figure(panels, canon_mark, list(ROSTER), False, "title")
    try:
        speedup_ax, token_ax = fig.axes
        labels = token_ax.get_xticklabels()
        # both panels summarize by the geomean, so the statistic is the shared axis' last tick
        assert [label.get_text() for label in labels] == [*ROSTER, "Geomean"]
        assert all(label.get_rotation() == 90.0 for label in labels[: len(ROSTER)])
        assert not [label for label in speedup_ax.get_xticklabels() if label.get_text() in ROSTER]
        # and never on a VALUE axis: that one carries the measured quantity, in both panels
        for ax in fig.axes:
            assert not {label.get_text() for label in ax.get_yticklabels()} & set(ROSTER)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_the_summary_group_sits_past_the_last_kernel_behind_a_dashed_vertical_separator() -> None:
    """The geomean/median summary is a GROUP at the right end of the kernel axis, marked off by a
    dashed vertical rule: it is one statistic per series, not a 41st kernel, so it must not sit
    inside the kernel slots -- and the axis has to reach past it, or the group is clipped."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.collections import PathCollection

    frame = observations(
        [
            *submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0, "k3": 2.0}),
            *task_rows("cpf-llr-focus40-qwen38-c", {"k1": 100.0, "k2": 200.0, "k3": 300.0}),
        ]
    )
    panels, canon_mark, _dropped = kernel_comparison.build_panels(frame, ROSTER)
    fig = kernel_comparison.figure(panels, canon_mark, list(ROSTER), False, "title")
    try:
        separator_x = per_kernel.summary_separator_x(len(ROSTER))
        summary_x = per_kernel.summary_slot_x(len(ROSTER), 0)
        assert separator_x > len(ROSTER) - 1 and summary_x > separator_x
        for ax in fig.axes:
            assert ax.get_xlim()[1] > summary_x
            dashed = [
                line
                for line in ax.get_lines()
                if line.get_xdata()[0] == line.get_xdata()[1] and line.get_linestyle() != "-"
            ]
            assert [line.get_xdata()[0] for line in dashed] == [separator_x]
            marks = {
                round(float(x), 6)
                for collection in ax.collections
                if isinstance(collection, PathCollection)
                for x, _y in collection.get_offsets()
            }
            assert [x for x in marks if x > separator_x], marks
            assert max(x for x in marks if x < separator_x) < len(ROSTER)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_token_panel_draws_no_canon_series() -> None:
    """The canon column runs no agent and spends no tokens, so the token panel must draw only the
    model's own arms -- never a canon mark, even a hollow "missing" one."""
    import matplotlib

    matplotlib.use("Agg")
    frame = observations([*submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0, "k3": 2.0})])
    canon = canon_frame(
        [
            ("numba", "k1", 100.0, "True"),
            ("numba", "k2", 100.0, "True"),
            ("numba", "k3", 100.0, "True"),
            ("dace_cpu_canonicalize", "k1", 50.0, "True"),
            ("dace_cpu_canonicalize", "k2", 50.0, "True"),
            ("dace_cpu_canonicalize", "k3", 50.0, "True"),
        ]
    )
    panels, canon_mark, _dropped = kernel_comparison.build_panels(frame, ROSTER, canon_frame=canon)
    fig = kernel_comparison.figure(panels, canon_mark, list(ROSTER), False, "title")
    try:
        speedup_ax, token_ax = fig.axes
        # canon's marker (a diamond) is drawn once per kernel slot on the speed-up panel and never on
        # the token panel: one PathCollection per drawn mark (plotstyle.point_mark), two per point
        # (white disc + coloured mark), so the token panel has strictly fewer than the speed-up one.
        assert len(token_ax.collections) < len(speedup_ax.collections)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_a_kernel_with_no_token_total_draws_no_mark_on_the_token_panel() -> None:
    """A missing token total is no measurement (R7); a hollow mark at the axis edge would read as the
    smallest spend on the panel. The speed-up panel still marks a kernel with no answer."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.collections import PathCollection

    frame = observations(
        [
            *submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0, "k3": 2.0}),
            *task_rows("cpf-llr-focus40-qwen38-c", {"k1": 100.0}),
        ]
    )
    panels, canon_mark, _dropped = kernel_comparison.build_panels(frame, ROSTER)
    fig = kernel_comparison.figure(panels, canon_mark, list(ROSTER), False, "title")
    try:
        token_ax = fig.axes[1]
        separator_x = per_kernel.summary_separator_x(len(ROSTER))
        slots = {
            round(float(x), 6)
            for collection in token_ax.collections
            if isinstance(collection, PathCollection)
            for x, _y in collection.get_offsets()
            if x < separator_x
        }
        assert slots.isdisjoint({1.0, 2.0}), slots
        assert 0.0 in slots, slots
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_an_incomplete_arm_appears_in_neither_the_speedup_nor_the_token_panel() -> None:
    """An arm dropped for incomplete roster coverage must not leak into the token panel either --
    completeness is decided once, before either metric is read."""
    frame = observations(
        [
            *submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0, "k3": 2.0}),
            *submission_rows("cpf-llr-focus40-qwen38-c-cpf", {"k1": 2.0}),
            *task_rows("cpf-llr-focus40-qwen38-c-cpf", {"k1": 100.0}),
        ]
    )
    panels, _canon, dropped = kernel_comparison.build_panels(frame, ROSTER)
    kept_keys = {series.key for series_list in panels.values() for series in series_list}
    assert "cpf-llr-focus40-qwen38-c-cpf" not in kept_keys
    assert dropped == {"cpf-llr-focus40-qwen38-c-cpf": 1}


def test_roster_of_reads_every_kernel_the_canon_frame_names() -> None:
    canon = canon_frame([("numba", "k1", 1.0, "True"), ("numba", "k2", 1.0, "True")])
    assert kernel_comparison.roster_of(canon) == ["k1", "k2"]


def test_rank_condition_keeps_the_declared_order_for_known_conditions() -> None:
    order = ("", "cpf", "cpfsrc")
    ranked = sorted(("cpfsrc", "", "cpf"), key=lambda condition: kernel_comparison.rank_condition(condition, order))
    assert ranked == ["", "cpf", "cpfsrc"]


def test_rank_condition_sorts_an_axis_outside_the_declared_order_alphabetically() -> None:
    """git-scicomp's arm names carry ``kernel``/``repo``, neither a skill packet; the default
    CONDITION_ORDER must not raise on them, and unknowns sort after every known condition."""
    order = ("", "cpf", "cpfsrc")
    ranked = sorted(("repo", "kernel"), key=lambda condition: kernel_comparison.rank_condition(condition, order))
    assert ranked == ["kernel", "repo"]


def test_build_panels_draws_a_non_packet_condition_axis_from_a_custom_arm_pattern() -> None:
    """git-scicomp's own use: condition is the arm's kernel/repo suffix, no canon column, and the
    condition order is passed explicitly rather than left to the packet default."""
    import re

    pattern = re.compile(r"^git-scicomp-(?P<model>[a-z0-9]+)-(?P<condition>kernel|repo)$")
    frame = observations(
        [
            *submission_rows("git-scicomp-qwen38-kernel", {"k1": 2.0, "k2": 2.0, "k3": 2.0}),
            *submission_rows("git-scicomp-qwen38-repo", {"k1": 3.0, "k2": 3.0, "k3": 3.0}),
        ]
    )
    panels, canon_mark, dropped = kernel_comparison.build_panels(
        frame, ROSTER, pattern=pattern, condition_order=("kernel", "repo")
    )
    assert canon_mark is None
    assert dropped == {}
    assert [series.condition for series in panels["qwen38"]] == ["kernel", "repo"]


def test_a_rerun_of_the_script_writes_byte_identical_files(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A published figure is regenerated and diffed against the committed one."""
    import importlib.util
    import sys

    repo = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "plot_kernel_comparison", repo / "statistics" / "plot_kernel_comparison.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = script
    spec.loader.exec_module(script)

    obs_csv = tmp_path / "obs.csv"
    observations(
        [
            *submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0, "k3": 2.0}),
            *submission_rows("cpf-llr-focus40-qwen38-c-cpfsrc", {"k1": 3.0, "k2": 3.0, "k3": 3.0}),
        ]
    ).to_csv(obs_csv, index=False)
    roster_file = tmp_path / "roster.txt"
    roster_file.write_text("\n".join(ROSTER))

    for epoch, folder in (("0", "first"), ("86400", "second")):
        monkeypatch.setenv("SOURCE_DATE_EPOCH", epoch)
        out_dir = tmp_path / folder
        rc = script.run(
            obs_csv,
            None,
            kernel_comparison.CANON_COLUMN,
            kernel_comparison.CANON_BASELINE,
            roster_file,
            kernel_comparison.ARM_PATTERN.pattern,
            False,
            False,
            "",
            out_dir / "kernel_comparison.pdf",
            out_dir / "kernel_comparison.csv",
        )
        assert rc == 0

    for name in ("kernel_comparison.pdf", "kernel_comparison.png", "kernel_comparison.csv"):
        first = (tmp_path / "first" / name).read_bytes()
        second = (tmp_path / "second" / name).read_bytes()
        assert first == second, f"{name} depends on when it was rendered"


def test_a_partial_arm_is_printed_as_dropped_with_its_coverage(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib.util
    import sys

    repo = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "plot_kernel_comparison", repo / "statistics" / "plot_kernel_comparison.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = script
    spec.loader.exec_module(script)

    obs_csv = tmp_path / "obs.csv"
    observations(
        [
            *submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0, "k3": 2.0}),
            *submission_rows("cpf-llr-focus40-qwen38-c-cpf", {"k1": 2.0}),
        ]
    ).to_csv(obs_csv, index=False)
    roster_file = tmp_path / "roster.txt"
    roster_file.write_text("\n".join(ROSTER))

    rc = script.run(
        obs_csv,
        None,
        kernel_comparison.CANON_COLUMN,
        kernel_comparison.CANON_BASELINE,
        roster_file,
        kernel_comparison.ARM_PATTERN.pattern,
        False,
        False,
        "",
        tmp_path / "out" / "kernel_comparison.pdf",
        tmp_path / "out" / "kernel_comparison.csv",
    )

    assert rc == 0
    err = capsys.readouterr().err
    assert "dropped cpf-llr-focus40-qwen38-c-cpf: 1/3" in err


def test_a_series_shape_is_the_model_and_its_colour_is_the_condition() -> None:
    """One channel per entity, from the shared registry: the MODEL takes the shape (which survives
    greyscale) and the CONDITION takes the colour, so a packet means the same thing here as in
    every other figure in the repo."""
    frame = observations(
        [
            *submission_rows("cpf-llr-focus40-qwen38-c-cpf", {"k1": 2.0, "k2": 2.0, "k3": 2.0}),
            *submission_rows("cpf-llr-focus40-oss120b-c-cpf", {"k1": 2.0, "k2": 2.0, "k3": 2.0}),
        ]
    )
    panels, _canon, _dropped = kernel_comparison.build_panels(frame, ROSTER)

    for model, series_list in panels.items():
        for series in series_list:
            assert series.marker == palette.marker(model)
            assert series.color == palette.color(series.condition)
    assert panels["qwen38"][0].marker != panels["oss120b"][0].marker
    assert panels["qwen38"][0].color == panels["oss120b"][0].color


def test_both_panels_rule_the_value_axis_and_leave_the_kernel_axis_bare() -> None:
    """A major and a lighter minor grid on the measured axis -- which is Y here (user, 2026-09-22).
    The kernel axis carries NAMES, where a guide line per category measures nothing."""
    import matplotlib.pyplot as plt

    frame = observations(
        [
            *submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0, "k3": 2.0}),
            *task_rows("cpf-llr-focus40-qwen38-c", {"k1": 100.0, "k2": 200.0, "k3": 300.0}),
        ]
    )
    panels, canon_mark, _dropped = kernel_comparison.build_panels(frame, ROSTER)
    fig = kernel_comparison.figure(panels, canon_mark, list(ROSTER), False, "title")
    try:
        for ax in fig.axes:
            assert any(line.get_visible() for line in ax.yaxis.get_gridlines())
            assert [tick for tick in ax.yaxis.get_minor_ticks() if tick.gridline.get_visible()]
            assert not [tick for tick in ax.xaxis.get_major_ticks() if tick.gridline.get_visible()]
    finally:
        plt.close(fig)


def test_a_kernel_with_no_verified_answer_enters_the_speedup_series_crossed_at_one() -> None:
    """A real 1.0x speed-up and a kernel nobody answered land on the same coordinate -- 1x, what a
    served but unsolved kernel leaves standing -- so the two may not draw as one mark: the
    placeholder is an UNDELIVERED cell, which per_kernel draws hollow and crossed and keeps out of
    the summary."""
    values = kernel_comparison.SeriesValues("arm", "Arm", "#4d4d4d", "o", "qwen38", "", {"k1": 2.0}, {}, {}, {})
    cells = {cell.kernel: cell for cell in kernel_comparison.speedup_series(values, ROSTER).cells}
    assert set(cells) == set(ROSTER)
    assert cells["k1"].delivered and cells["k1"].episodes == (2.0,)
    assert not cells["k3"].delivered and cells["k3"].episodes == (population.NOT_DELIVERED,)
    assert "1x" in plotstyle.NOT_DELIVERED_LABEL


def test_a_kernel_with_no_token_total_is_absent_from_the_token_series() -> None:
    """No token count is a neutral cost (R7): the kernel gets no cell rather than a stand-in."""
    values = kernel_comparison.SeriesValues("arm", "Arm", "#4d4d4d", "o", "qwen38", "", {}, {"k1": 50.0}, {}, {})
    assert [cell.kernel for cell in kernel_comparison.token_series(values, ROSTER).cells] == ["k1"]


def test_a_repeats_median_kernel_carries_its_task_range_as_its_interval() -> None:
    """R5: the median over designed repeats is drawn with the minimum and maximum of those same
    tasks, the whisker the token panel shows."""
    values = kernel_comparison.SeriesValues(
        "arm", "Arm", "#4d4d4d", "o", "qwen38", "", {}, {"k1": 250.0}, {"k1": 100.0}, {"k1": 400.0}
    )
    (cell,) = kernel_comparison.token_series(values, ROSTER).cells
    assert cell.episodes == (250.0,) and cell.interval == (100.0, 400.0)


def test_an_arm_sits_at_the_same_x_in_both_panels_although_canon_has_no_tokens() -> None:
    """The canon column keeps an empty series on the token panel, so every arm keeps its dodge
    offset: a reader matches a mark across the two panels by its position."""
    import matplotlib.pyplot as plt

    frame = observations(
        [
            *submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0, "k3": 2.0}),
            *task_rows("cpf-llr-focus40-qwen38-c", {"k1": 100.0, "k2": 100.0, "k3": 100.0}),
        ]
    )
    canon = canon_frame(
        [("numba", k, 100.0, "True") for k in ROSTER]
        + [(kernel_comparison.CANON_COLUMN, k, 50.0, "True") for k in ROSTER]
    )
    panels, canon_mark, _dropped = kernel_comparison.build_panels(frame, ROSTER, canon_frame=canon)
    fig = kernel_comparison.figure(panels, canon_mark, list(ROSTER), False, "title")
    try:
        colour = panels["qwen38"][0].color
        speed, tokens = (marks_in(ax, colour, per_kernel.summary_separator_x(len(ROSTER))) for ax in fig.axes)
    finally:
        plt.close(fig)
    assert speed == tokens != set(), (speed, tokens)


def marks_in(ax: matplotlib.axes.Axes, colour: str, before: float) -> set[float]:
    """The x of every scatter mark edged in ``colour`` left of ``before`` (the kernel columns)."""
    import matplotlib.colors
    from matplotlib.collections import PathCollection

    return {
        round(float(x), 6)
        for collection in ax.collections
        if isinstance(collection, PathCollection)
        and len(collection.get_edgecolors())
        and matplotlib.colors.to_hex(collection.get_edgecolors()[0]) == colour
        for x in collection.get_offsets()[:, 0]
        if x < before
    }


def test_a_double_column_render_is_the_page_width_and_a_standalone_one_follows_its_kernels() -> None:
    """``--double-column`` is a page insert, so its width is the page's; a standalone render asks
    for room per kernel instead, so it is wider than the page for forty kernels."""
    import matplotlib.pyplot as plt

    roster = [f"k{i}" for i in range(40)]
    frame = observations([*submission_rows("cpf-llr-focus40-qwen38-c", {k: 2.0 for k in roster})])
    panels, canon_mark, _dropped = kernel_comparison.build_panels(frame, roster)
    insert = kernel_comparison.figure(panels, canon_mark, roster, True, "title")
    standalone = kernel_comparison.figure(panels, canon_mark, roster, False, "title")
    try:
        assert insert.get_size_inches()[0] == pytest.approx(plotstyle.DOUBLE_COLUMN_WIDTH)
        assert standalone.get_size_inches()[0] > plotstyle.DOUBLE_COLUMN_WIDTH
    finally:
        plt.close(insert)
        plt.close(standalone)


def test_a_never_answered_kernel_draws_a_hollow_crossed_mark_at_one_on_the_speedup_panel() -> None:
    """The placeholder is DRAWN, not just configured: an unfilled mark plus a cross, at 1x on the
    value axis and at the kernel's own slot on the kernel axis."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.collections import PathCollection

    rows = submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0})
    frame = observations(rows)
    panels, canon_mark, _dropped = kernel_comparison.build_panels(frame, ROSTER, include_incomplete=True)
    fig = kernel_comparison.figure(panels, canon_mark, list(ROSTER), False, "title")
    try:
        speedup_ax = fig.axes[0]
        crosses = [
            collection
            for collection in speedup_ax.collections
            if isinstance(collection, PathCollection)
            for x, y in collection.get_offsets()
            if x == 2.0 and y == population.NOT_DELIVERED
        ]
        # the white halo, the hollow mark and the cross plotstyle.point_mark lays over it
        assert len(crosses) == 3
        assert any(len(collection.get_facecolors()) == 0 for collection in crosses)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_the_speedup_axis_names_the_denominator_the_judge_recorded() -> None:
    """The baseline is a property of the DATA, not of the figure. scientific_computing grades
    against ``c-autopar`` (``harness.grading.TRACK_DEFAULT_BASELINE``), so a panel drawn over its
    rows must say so; a fixed "vs Numba" labels those scores with a denominator none of them saw.
    The canon series already names its own denominator this way."""
    frame = observations(
        [
            *submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0, "k3": 2.0}, baseline="c-autopar"),
            *task_rows("cpf-llr-focus40-qwen38-c", {"k1": 10.0, "k2": 10.0, "k3": 10.0}),
        ]
    )
    panels, canon_mark, _dropped = kernel_comparison.build_panels(frame, ROSTER)
    fig = kernel_comparison.figure(
        panels,
        canon_mark,
        list(ROSTER),
        False,
        "title",
        kernel_comparison.CONDITION_ORDER,
        results.baseline_of(frame),
    )
    try:
        # the words, not the line breaks: a label taller than its panel is broken onto two lines
        assert " ".join(fig.axes[0].get_ylabel().split()) == "Speed-Up vs c-autopar"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_the_script_hands_the_figure_the_baseline_its_observations_carry(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wiring, not just the label: the script reads the denominator off the rows it loaded."""
    import importlib.util
    import sys

    repo = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "plot_kernel_comparison", repo / "statistics" / "plot_kernel_comparison.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = script
    spec.loader.exec_module(script)

    obs_csv = tmp_path / "obs.csv"
    observations(
        [
            *submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0, "k3": 2.0}, baseline="c-autopar"),
            *task_rows("cpf-llr-focus40-qwen38-c", {"k1": 10.0, "k2": 10.0, "k3": 10.0}),
        ]
    ).to_csv(obs_csv, index=False)
    roster_file = tmp_path / "roster.txt"
    roster_file.write_text("\n".join(ROSTER))

    seen: list[str] = []
    real = kernel_comparison.figure

    def record(*args: object, **kwargs: object) -> object:
        seen.append(str(args[6]) if len(args) > 6 else str(kwargs["baseline"]))
        return real(*args, **kwargs)  # pyright: ignore[reportArgumentType, reportCallIssue]

    monkeypatch.setattr(script.kernel_comparison, "figure", record)
    rc = script.run(
        obs_csv,
        None,
        kernel_comparison.CANON_COLUMN,
        kernel_comparison.CANON_BASELINE,
        roster_file,
        kernel_comparison.ARM_PATTERN.pattern,
        False,
        False,
        "",
        tmp_path / "out" / "kernel_comparison.pdf",
        tmp_path / "out" / "kernel_comparison.csv",
    )

    assert rc == 0
    assert seen == ["c-autopar"]


# ---------------------------------------------------------------------------------------------
# llr40_model_figure -- colour=model, one point per (kernel, model) under a packet SELECTION.
# ---------------------------------------------------------------------------------------------


def test_llr40_arm_names_the_control_and_each_packet() -> None:
    assert kernel_comparison.llr40_arm("qwen38", "") == "cpf-llr-focus40-qwen38-c"
    assert kernel_comparison.llr40_arm("qwen38", "lang-skills") == "cpf-llr-focus40-qwen38-c-skills"
    assert kernel_comparison.llr40_arm("oss120b", "perf-playbook-cpu") == "cpf-llr-focus40-oss120b-c-perf-playbook-cpu"


def test_median_over_packets_takes_the_median_of_whichever_packets_answered_a_kernel() -> None:
    per_packet = [{"k1": 2.0, "k2": 4.0}, {"k1": 4.0}, {"k1": 6.0, "k2": 8.0}]
    # k1: median(2,4,6)=4; k2: median(4,8)=6 -- one packet's absence never dilutes k2's median.
    assert kernel_comparison.median_over_packets(per_packet) == {"k1": 4.0, "k2": 6.0}


def test_llr40_model_panels_reads_the_control_arm_by_default() -> None:
    frame = observations(
        [
            *submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 4.0, "k2": 2.0}),
            *task_rows("cpf-llr-focus40-qwen38-c", {"k1": 100.0, "k2": 200.0}),
            *submission_rows("cpf-llr-focus40-oss120b-c", {"k1": 8.0}),
            *task_rows("cpf-llr-focus40-oss120b-c", {"k1": 300.0}),
        ]
    )
    panels = kernel_comparison.llr40_model_panels(frame, ["qwen38", "oss120b"], "", "latest")
    assert set(panels) == {"qwen38", "oss120b"}
    qwen = panels["qwen38"][0]
    assert qwen.values == {"k1": 4.0, "k2": 2.0}
    assert qwen.tokens == {"k1": 100.0, "k2": 200.0}
    assert qwen.color == palette.model_color("qwen38")
    # Every model's mark shares ONE shape under the control/median modes -- colour alone tells
    # the series apart, the settled rule this figure shares with the efficacy panels.
    assert qwen.marker == panels["oss120b"][0].marker == palette.CONTROL_MARKER


def test_llr40_model_panels_median_mode_aggregates_the_packet_arms_one_model_has() -> None:
    frame = observations(
        [
            *submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0}),
            *task_rows("cpf-llr-focus40-qwen38-c", {"k1": 100.0}),
            *submission_rows("cpf-llr-focus40-qwen38-c-cpf", {"k1": 8.0}),
            *task_rows("cpf-llr-focus40-qwen38-c-cpf", {"k1": 300.0}),
        ]
    )
    panels = kernel_comparison.llr40_model_panels(frame, ["qwen38"], "median", "latest")
    series = panels["qwen38"][0]
    # Only 2 of the 5 registered packets have an arm here; the other 3 contribute nothing and do
    # not drag the median toward a value neither arm reported.
    assert series.values == {"k1": 5.0}
    assert series.tokens == {"k1": 200.0}


def test_llr40_model_panels_drops_a_model_with_no_value_under_the_chosen_packet() -> None:
    frame = observations(submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0}))
    panels = kernel_comparison.llr40_model_panels(frame, ["qwen38"], "cpf", "latest")
    assert panels == {}


def test_llr40_model_figure_draws_one_mark_per_kernel_per_model_plus_a_summary_column() -> None:
    frame = observations(
        [
            *submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 4.0, "k2": 2.0}),
            *task_rows("cpf-llr-focus40-qwen38-c", {"k1": 100.0, "k2": 200.0}),
            *submission_rows("cpf-llr-focus40-oss120b-c", {"k1": 8.0}),
            *task_rows("cpf-llr-focus40-oss120b-c", {"k1": 300.0}),
        ]
    )
    fig = kernel_comparison.llr40_model_figure(
        frame, list(ROSTER), ["qwen38", "oss120b"], packet_mode="", repeats="latest"
    )
    try:
        speedup_ax, token_ax = fig.axes[0], fig.axes[1]
        # 2 models * 3 roster kernels, plus each model's own summary mark past the separator.
        assert len(speedup_ax.collections) >= 1
        assert len(token_ax.collections) >= 1
        # NO title anywhere -- a paper caption carries it (module docstring).
        assert fig._suptitle is None
        assert speedup_ax.get_title() == ""
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_llr40_model_figure_refuses_when_no_model_has_a_value() -> None:
    frame = observations(submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0}))
    with pytest.raises(ValueError, match="no model"):
        kernel_comparison.llr40_model_figure(frame, list(ROSTER), ["oss120b"], packet_mode="cpf", repeats="latest")
