# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""statistics/gate_sensitivity.py: task-level dispersion-gate rules A (current), B and C.

A CELL IS A (config, shape) TIMED MEASUREMENT OF ONE SUBMISSION -- never an episode, a rerun or a
repetition across waves (module docstring). The per-cell fixtures below are therefore synthetic,
CSV rows the module's own contract defines (``arm, benchmark, cell, ratio``), not observations rows:
no stored artifact carries real per-cell ratios today, so there is no "production shape" to mirror
for that part of the module, only the contract :func:`gate_sensitivity.load_percell_ratios` states.

The A-vs-B/C ground truth below (g, gsd, credited) was checked independently with plain
``math``/``statistics`` before it went into this file, not read back off :mod:`gate_sensitivity`
itself -- a test that recomputes the implementation checks nothing.

THE ``relevance`` FIXTURES ARE THE PRODUCTION SHAPE (tests/test_paired_arms.py's own rule): a graded
``submission`` row exactly as ``reproducibility/llr40/extract_llr40.py`` writes it.
"""

import importlib.util
import math
import pathlib
import sys
from types import ModuleType

import pandas as pd
import pytest

STATISTICS = pathlib.Path(__file__).resolve().parents[1] / "statistics"


def load_gate_sensitivity() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gate_sensitivity", STATISTICS / "gate_sensitivity.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="gs")
def gate_sensitivity_fixture() -> ModuleType:
    return load_gate_sensitivity()


# --------------------------------------------------------------------------------------------
# Rules A/B/C: pure gate arithmetic, built directly so the boundary is exact and hand-checkable.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("g", "gsd", "min_cell", "want_a", "want_b", "want_c"),
    [
        # single-cell task: gsd=1, min==g, so B/C collapse to A's g>1 test -- the three can only
        # diverge once a task has >=2 cells.
        (5.0, 1.0, 5.0, True, True, True),
        (0.5, 1.0, 0.5, False, False, False),
        # tie between g and gsd: A requires strict g>gsd, so an exact tie is NOT credited (B/C do
        # not read gsd at all, and both clear their own min_cell>=1 test here).
        (2.0, 2.0, 2.0, False, True, True),
        # min_cell exactly 1x: B's >=1 admits it, C's strict >1 does not -- the B-vs-C boundary.
        (3.0, 1.0, 1.0, True, True, False),
        # the real llr-focus40-cpu-shaped case: high dispersion, no cell below 1x.
        (2.655842946402823, 3.4486175336784184, 1.003, False, True, True),
        # one regressed cell among two strong wins: A clears its own (smaller) gsd, B/C do not.
        (4.626065009182743, 3.800758634187982, 0.99, True, False, False),
    ],
)
def test_credited_matches_the_literal_rule_text(
    gs: ModuleType, g: float, gsd: float, min_cell: float, want_a: bool, want_b: bool, want_c: bool
) -> None:
    t = gs.TaskScore("exp", "armA", "k1", (), g, gsd, min_cell, True)
    assert t.credited("A") == want_a
    assert t.credited("B") == want_b
    assert t.credited("C") == want_c


def test_an_unsolved_task_is_never_credited_under_any_rule(gs: ModuleType) -> None:
    t = gs.TaskScore("exp", "armA", "k1", (5.0,), 5.0, 1.0, 5.0, False)
    assert not t.credited("A") and not t.credited("B") and not t.credited("C")
    assert t.score("A") == t.score("B") == t.score("C") == 1.0


def test_score_clamps_a_credited_task_to_c_max(gs: ModuleType) -> None:
    t = gs.TaskScore("exp", "armA", "k1", (9999.0,) * 3, 9999.0, 1.0, 9999.0, True)
    assert t.score("A") == pytest.approx(2000.0)  # the shipped default c_max, never part of the gate


def test_geomean_gsd_matches_independently_checked_numbers(gs: ModuleType) -> None:
    g, gsd = gs.geomean_gsd([1.746, 10.697, 1.003])
    assert g == pytest.approx(2.655842946402823)
    assert gsd == pytest.approx(3.4486175336784184)


def test_geomean_gsd_of_one_cell_has_no_dispersion(gs: ModuleType) -> None:
    g, gsd = gs.geomean_gsd([7.0])
    assert g == pytest.approx(7.0) and gsd == pytest.approx(1.0)


# --------------------------------------------------------------------------------------------
# load_percell_ratios: refuses rather than substitutes.
# --------------------------------------------------------------------------------------------


def write_percell(rows: list[dict[str, object]], tmp_path: pathlib.Path, name: str = "percell.csv") -> pathlib.Path:
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_a_missing_percell_file_raises_not_falls_back(gs: ModuleType, tmp_path: pathlib.Path) -> None:
    with pytest.raises(gs.MissingPerCellDataError):
        gs.load_percell_ratios(tmp_path / "does-not-exist.csv")


def test_a_percell_file_missing_a_required_column_raises(gs: ModuleType, tmp_path: pathlib.Path) -> None:
    path = write_percell([{"arm": "armA", "benchmark": "k1", "ratio": 2.0}], tmp_path)  # no 'cell'
    with pytest.raises(gs.MissingPerCellDataError, match="cell"):
        gs.load_percell_ratios(path)


def test_a_non_positive_ratio_raises(gs: ModuleType, tmp_path: pathlib.Path) -> None:
    path = write_percell([{"arm": "armA", "benchmark": "k1", "cell": 0, "ratio": 0.0}], tmp_path)
    with pytest.raises(gs.MissingPerCellDataError):
        gs.load_percell_ratios(path)


def test_a_well_formed_percell_file_loads(gs: ModuleType, tmp_path: pathlib.Path) -> None:
    rows = [
        {"arm": "armA", "benchmark": "k1", "cell": 0, "ratio": 1.5},
        {"arm": "armA", "benchmark": "k1", "cell": 1, "ratio": 2.0},
    ]
    frame = gs.load_percell_ratios(write_percell(rows, tmp_path))
    assert len(frame) == 2 and list(frame["ratio"]) == [1.5, 2.0]


# --------------------------------------------------------------------------------------------
# build_task_scores over real (synthetic) per-cell rows.
# --------------------------------------------------------------------------------------------


def test_build_task_scores_groups_cells_by_arm_and_benchmark(gs: ModuleType, tmp_path: pathlib.Path) -> None:
    rows = [
        {"arm": "armA", "benchmark": "k1", "cell": "cfg0:large0", "ratio": 1.746},
        {"arm": "armA", "benchmark": "k1", "cell": "cfg0:large1", "ratio": 10.697},
        {"arm": "armA", "benchmark": "k1", "cell": "cfg0:large2", "ratio": 1.003},
    ]
    (t,) = gs.build_task_scores(gs.load_percell_ratios(write_percell(rows, tmp_path)), "exp")
    assert t.cells == (1.746, 10.697, 1.003)
    assert t.g == pytest.approx(2.655842946402823) and t.gsd == pytest.approx(3.4486175336784184)
    assert t.solved
    assert not t.credited("A") and t.credited("B") and t.credited("C")


def test_a_clean_suffix_arm_is_folded_into_its_base_arm(gs: ModuleType, tmp_path: pathlib.Path) -> None:
    """USER 2026-09-18: a '-clean' arm is the SAME condition as its bare counterpart."""
    rows = [
        {"arm": "armA", "benchmark": "k1", "cell": 0, "ratio": 2.0},
        {"arm": "armA-clean", "benchmark": "k1", "cell": 1, "ratio": 3.0},
    ]
    tasks = gs.build_task_scores(gs.load_percell_ratios(write_percell(rows, tmp_path)), "exp")
    assert {t.arm for t in tasks} == {"armA"}
    assert sorted(tasks[0].cells) == [2.0, 3.0]


def test_flips_table_lists_every_disagreement_and_nothing_else(gs: ModuleType, tmp_path: pathlib.Path) -> None:
    """Deliverable must be complete, not sampled: an all-rules-agree task must not appear, and every
    disagreeing task must -- both directions checked in one table."""
    rows = [
        {"arm": "armA", "benchmark": "agree", "cell": 0, "ratio": 5.0},
        {"arm": "armA", "benchmark": "flip", "cell": 0, "ratio": 1.746},
        {"arm": "armA", "benchmark": "flip", "cell": 1, "ratio": 10.697},
        {"arm": "armA", "benchmark": "flip", "cell": 2, "ratio": 1.003},
    ]
    tasks = gs.build_task_scores(gs.load_percell_ratios(write_percell(rows, tmp_path)), "exp")
    table = gs.flips_table(tasks)
    assert list(table["benchmark"]) == ["flip"]
    assert bool(table.iloc[0]["flips_a_vs_b"]) and bool(table.iloc[0]["flips_a_vs_c"])


def test_counts_table_all_rollup_sums_the_named_experiments(gs: ModuleType, tmp_path: pathlib.Path) -> None:
    exp1 = gs.build_task_scores(
        gs.load_percell_ratios(
            write_percell([{"arm": "armA", "benchmark": "k1", "cell": 0, "ratio": 5.0}], tmp_path, "e1.csv")
        ),
        "exp1",
    )
    exp2 = gs.build_task_scores(
        gs.load_percell_ratios(
            write_percell([{"arm": "armB", "benchmark": "k1", "cell": 0, "ratio": 0.5}], tmp_path, "e2.csv")
        ),
        "exp2",
    )
    counts = gs.counts_table(exp1 + exp2)
    rollup = counts[(counts.experiment == "all") & (counts.model == "all") & (counts.rule == "A")].iloc[0]
    assert rollup["n_tasks"] == 2 and rollup["credited"] == 1 and rollup["scored_one"] == 1


def test_arm_geomeans_table_is_over_real_cells_only(gs: ModuleType, tmp_path: pathlib.Path) -> None:
    rows = [
        {"arm": "armA", "benchmark": "k1", "cell": 0, "ratio": 4.0},
        {"arm": "armA", "benchmark": "k2", "cell": 0, "ratio": 1.0},
    ]
    tasks = gs.build_task_scores(gs.load_percell_ratios(write_percell(rows, tmp_path)), "exp")
    table = gs.arm_geomeans_table(tasks)
    row = table[table.arm == "armA"].iloc[0]
    assert row["n_tasks"] == 2
    assert row["geomean_A"] == pytest.approx(math.sqrt(4.0 * 1.0))


# --------------------------------------------------------------------------------------------
# `relevance`: today's one-ratio-per-task reality, production-shaped observations fixtures.
# --------------------------------------------------------------------------------------------


def submitted_row(
    arm: str, kernel: str, speedup: float, *, job: str = "j1", ts: int = 1000, index: int = 1, suspect: int = 0
) -> dict[str, object]:
    return {
        "run_root": "stamp",
        "job": job,
        "record": "submission",
        "run_id": f"{arm}.n0.p{kernel}.w0",
        "arm": arm,
        "benchmark": kernel,
        "speedup": speedup,
        "suspect": suspect,
        "ts_ms": ts,
        "attempt_index": index,
        "timing_reduction": "mwd-v2",
    }


def obs_frame(rows: list[dict[str, object]], tmp_path: pathlib.Path) -> pd.DataFrame:
    path = tmp_path / "observations.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return pd.read_csv(path, low_memory=False)


@pytest.mark.parametrize(
    ("name", "want"), [("llr-focus40-cpu", "latest"), ("llr-focus40-blind", "latest"), ("git-scicomp", "median")]
)
def test_default_repeat_policy_matches_the_campaigns_own_design(gs: ModuleType, name: str, want: str) -> None:
    assert gs.default_repeat_policy(name) == want


def test_relevance_table_reports_the_one_stored_ratio_not_a_gate_decision(
    gs: ModuleType, tmp_path: pathlib.Path
) -> None:
    rows = [submitted_row("armA", "k1", 8.0)]
    table = gs.relevance_table({"exp": obs_frame(rows, tmp_path)})
    assert len(table) == 1
    got = table.iloc[0]
    assert got["raw_ratio"] == pytest.approx(8.0)
    assert got["abs_ln_ratio"] == pytest.approx(abs(math.log(8.0)))


def test_relevance_table_drops_a_suspect_answer(gs: ModuleType, tmp_path: pathlib.Path) -> None:
    """A suspect ratio never reaches the gate (score_rule.credit excludes it) regardless of gsd, so
    it must not enter the |ln ratio| distribution either."""
    rows = [submitted_row("armA", "k1", 5000.0, suspect=1)]
    table = gs.relevance_table({"exp": obs_frame(rows, tmp_path)})
    assert table.empty


def test_relevance_table_keeps_only_the_latest_wave_for_a_reran_kernel(gs: ModuleType, tmp_path: pathlib.Path) -> None:
    """The campaign's own rule (module docstring): a rerun kernel counts as its LATEST run, not a
    pool of every wave -- this is exactly the distinction the coordinator's correction turned on."""
    rows = [
        submitted_row("armA", "k1", 2.0, job="wave1", ts=1000),
        submitted_row("armA", "k1", 9.0, job="wave2", ts=2000),
    ]
    table = gs.relevance_table({"exp": obs_frame(rows, tmp_path)})
    assert len(table) == 1
    assert table.iloc[0]["raw_ratio"] == pytest.approx(9.0)


def test_relevance_summary_counts_tasks_at_risk_under_each_illustrative_gsd(gs: ModuleType) -> None:
    table = pd.DataFrame({"abs_ln_ratio": [0.0, math.log(1.2), math.log(4.0)]})
    summary = gs.relevance_summary(table)
    row = summary[summary.illustrative_gsd == 1.5].iloc[0]
    # 0.0 and ln(1.2) both clear ln(1.5) or fall under it; ln(4.0) does not.
    assert row["n_tasks"] == 3
    assert row["at_risk_if_gsd_this_big"] == 2  # 0.0 and ln(1.2) <= ln(1.5)
    assert row["safe_regardless"] == 1


def test_relevance_summary_of_an_empty_table_is_well_formed(gs: ModuleType) -> None:
    summary = gs.relevance_summary(pd.DataFrame({"abs_ln_ratio": []}))
    assert (summary["n_tasks"] == 0).all() and (summary["at_risk_if_gsd_this_big"] == 0).all()
