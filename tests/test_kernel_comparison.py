# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``hpcagent_bench.stats.figures.kernel_comparison`` -- the llr-focus40 canon-vs-agents figure.

Condition comes from the ARM NAME (:data:`kernel_comparison.ARM_PATTERN`), never the
``language``/``packet`` columns, because the pre-regrade extraction records those inconsistently
for the same arm. Completeness is roster coverage (:func:`population.complete_arms`), applied
before any per-kernel value is read, and a model whose every arm is incomplete gets no panel.
"""

import pathlib

import pandas as pd
import pytest

from hpcagent_bench.stats.figures import kernel_comparison

ROSTER: tuple[str, ...] = ("k1", "k2", "k3")


def submission_rows(arm: str, benchmark_speedups: dict[str, float]) -> list[dict[str, object]]:
    """One graded episode per (arm, kernel): the columns ``population.kernel_answers`` needs."""
    rows = []
    for benchmark, speedup in benchmark_speedups.items():
        run = f"{arm}-{benchmark}"
        rows.append(
            {
                "run_root": "j1",
                "job": "j1",
                "run_id": run,
                "arm": arm,
                "record": "submission",
                "benchmark": benchmark,
                "speedup": speedup,
                "baseline_ns": 1000.0,
                "native_ns": 1000.0 / speedup,
                "baseline": "numba",
                "suspect": 0,
                "ts_ms": 1,
                "attempt_index": 1,
            }
        )
    return rows


def observations(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


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
    panels, canon_mark, _dropped = kernel_comparison.build_panels(frame, ROSTER, canon_frame=canon)
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


def test_a_kernel_with_no_verified_answer_has_no_row_rather_than_a_stand_in_value() -> None:
    """The figure follows the framework's own scoring policy, never inventing a served-but-unsolved
    value: an arm complete on the roster (a row exists) but with a negative/unusable speed-up on one
    kernel simply has no mark there."""
    rows = submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0})
    rows.append(
        {
            "run_root": "j1",
            "job": "j1",
            "run_id": "cpf-llr-focus40-qwen38-c-k3",
            "arm": "cpf-llr-focus40-qwen38-c",
            "record": "submission",
            "benchmark": "k3",
            "speedup": -1.0,
            "baseline_ns": 1000.0,
            "native_ns": 1000.0,
            "baseline": "numba",
            "suspect": 0,
            "ts_ms": 1,
            "attempt_index": 1,
        }
    )
    frame = observations(rows)
    panels, _canon, dropped = kernel_comparison.build_panels(frame, ROSTER, include_incomplete=True)
    values = panels["qwen38"][0].values
    assert set(values) == {"k1", "k2"}
    assert dropped == {}


def test_table_rows_carries_one_row_per_series_and_kernel_with_a_value() -> None:
    frame = observations([*submission_rows("cpf-llr-focus40-qwen38-c", {"k1": 2.0, "k2": 2.0, "k3": 2.0})])
    canon = canon_frame([("numba", "k1", 100.0, "True"), ("dace_cpu_canonicalize", "k1", 50.0, "True")])
    panels, canon_mark, _dropped = kernel_comparison.build_panels(frame, ROSTER, canon_frame=canon)
    table = kernel_comparison.table_rows(panels, canon_mark)
    assert set(table.columns) == {"kernel", "series", "kind", "model", "condition", "speedup"}
    assert (table.kind == "canon").sum() == 1
    assert (table.kind == "arm").sum() == 3


def test_roster_of_reads_every_kernel_the_canon_frame_names() -> None:
    canon = canon_frame([("numba", "k1", 1.0, "True"), ("numba", "k2", 1.0, "True")])
    assert kernel_comparison.roster_of(canon) == ["k1", "k2"]


def test_a_rerun_of_the_script_writes_byte_identical_files(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A published figure is regenerated and diffed against the committed one."""
    import importlib.util
    import sys

    repo = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "plot_kernel_comparison", repo / "scripts" / "plot_kernel_comparison.py"
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
        "plot_kernel_comparison", repo / "scripts" / "plot_kernel_comparison.py"
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
