# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""paired_arms.py: arm-against-arm comparisons over a declared family.

THE FIXTURE IS THE PRODUCTION SHAPE. Every episode here emits a graded ``submission`` row carrying a
speed-up and NO token count, plus a ``call`` row carrying a token count and NO speed-up, because that
is what the judge and the trajectory writer actually record. Three earlier tests put both columns on
one row, which is why a filter that AND-ed them -- and so kept only call rows and dropped every
graded submission -- passed its tests and reached a published table.

The replicate case is checked directly: two jobs of one arm reuse the same ``run_id``, because a
launcher derives it from the rank layout, so a reduction keyed on ``run_id`` alone silently discards
one replicate. They must come out as two episodes whose MAXIMUM stands.
"""

import importlib.util
import math
import pathlib
import sys
from types import ModuleType

import pandas as pd
import pytest

from hpcagent_bench.stats import population, summary

EXPERIMENTS = pathlib.Path(__file__).resolve().parents[1] / "experiments"

#: One kernel roster the fixtures draw names from, so a coverage count has something to be over.
KERNELS = ("k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8")


def load_experiment_module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="paired_arms")
def paired_arms_fixture() -> ModuleType:
    return load_experiment_module("paired_arms")


def graded(
    arm: str,
    kernel: str,
    speedup: float,
    job: str = "j1",
    ts: int = 1000,
    index: int = 1,
    optimizer: str = "a-model",
) -> dict[str, object]:
    """One graded submission: timings, no tokens. ``run_id`` is the rank spelling, which repeats
    across jobs exactly as a launcher writes it. ``optimizer`` carries the recovery tag when the row
    is one nobody submitted."""
    return {
        "optimizer": optimizer,
        "run_root": "stamp",
        "job": job,
        "record": "submission",
        "run_id": f"{arm}.n0.p0.w0",
        "arm": arm,
        "benchmark": kernel,
        "speedup": speedup,
        "tokens": "",
        "baseline": "numba",
        "ts_ms": ts,
        "attempt_index": index,
    }


def call(arm: str, kernel: str, tokens: float, job: str = "j1", ts: int = 1000, index: int = 1) -> dict[str, object]:
    """One trajectory call: a CUMULATIVE token count, no timings."""
    return {
        "optimizer": "a-model",
        "run_root": "stamp",
        "job": job,
        "record": "call",
        "run_id": f"{arm}.n0.p0.w0",
        "arm": arm,
        "benchmark": kernel,
        "speedup": "",
        "tokens": tokens,
        "baseline": "numba",
        "ts_ms": ts,
        "attempt_index": index,
    }


def frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def observations(rows: list[dict[str, object]], tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "llr40_observations.csv"
    frame(rows).to_csv(path, index=False)
    return path


def episode(arm: str, kernel: str, speedup: float, tokens: float, job: str = "j1") -> list[dict[str, object]]:
    """One agent on one kernel, in the two rows the harness writes for it."""
    return [graded(arm, kernel, speedup, job=job), call(arm, kernel, tokens, job=job)]


def test_within_an_episode_the_last_submission_wins_not_the_best(paired_arms: ModuleType) -> None:
    """Evaluation is single-shot, so the answer the agent stopped at is the answer; a max over an
    episode's rows would score best-of-N."""
    rows = [graded("a", "k1", 4.0, ts=1000, index=1), graded("a", "k1", 2.0, ts=2000, index=2)]
    best = paired_arms.best_by_arm_kernel(frame(rows))
    assert best.speedup.tolist() == [2.0]


def test_replicate_jobs_are_separate_episodes_and_the_maximum_stands(paired_arms: ModuleType) -> None:
    """Two jobs of one arm reuse the run_id, so keying on it alone would drop a whole replicate.

    Replicate 1 ends at 5.0 and replicate 2 at 3.0. They are two episodes: the arm's value is the
    maximum over them (5.0), not the later replicate's 3.0 and not a pooled max over rows.
    """
    rows = [
        graded("a", "k1", 5.0, job="j1", ts=1000),
        graded("a", "k1", 9.0, job="j2", ts=2000, index=1),
        graded("a", "k1", 3.0, job="j2", ts=3000, index=2),
    ]
    episodes = population.last_per_episode(frame(rows), paired_arms.SUBMISSION_ORDER)
    assert len(episodes) == 2
    assert paired_arms.best_by_arm_kernel(frame(rows)).speedup.tolist() == [5.0]


def test_the_score_leg_keeps_a_kernel_that_has_no_call_row(paired_arms: ModuleType, tmp_path: pathlib.Path) -> None:
    """A graded row carries no tokens, so intersecting the two legs would drop every kernel whose
    call rows are missing. The score leg is over the kernels both arms SOLVED and nothing else."""
    rows: list[dict[str, object]] = []
    for kernel in KERNELS:
        rows += episode("a", kernel, 2.0, 100.0)
        rows += episode("b", kernel, 1.5, 100.0)
    rows.append(graded("a", "k9", 3.0))
    rows.append(graded("b", "k9", 2.0))

    path = observations(rows, tmp_path)
    obs = paired_arms.load_observations(path)
    graded_rows = paired_arms.graded_rows(obs, ["a", "b"])
    best = paired_arms.best_by_arm_kernel(graded_rows)
    table = paired_arms.arm_aggregates(best, paired_arms.served_by_arm(obs), "numba")
    tokens = paired_arms.tokens_by_arm_kernel(obs)

    assert ("a", "k9") not in tokens
    change, n_pairs = paired_arms.score_leg(table["a"], table["b"])
    assert n_pairs == len(KERNELS) + 1
    assert change.n == len(KERNELS) + 1
    assert paired_arms.cost_leg("a", "b", tokens)[1] == len(KERNELS)


def test_the_cost_leg_reads_a_cumulative_counter_as_its_episode_maximum(paired_arms: ModuleType) -> None:
    """``calls.tokens`` is cumulative through a call, so an episode's spend is its maximum and a
    kernel's is the sum over its episodes -- summing the rows counts every earlier call again."""
    rows = [
        call("a", "k1", 100.0, job="j1", ts=1000, index=1),
        call("a", "k1", 250.0, job="j1", ts=2000, index=2),
        call("a", "k1", 400.0, job="j2", ts=3000, index=1),
    ]
    tokens = paired_arms.tokens_by_arm_kernel(frame(rows))
    assert tokens[("a", "k1")] == 650.0


def test_two_denominators_are_refused_rather_than_pooled(paired_arms: ModuleType) -> None:
    """The judge stamps the reference it divided by; two of them are not one quantity."""
    rows = [graded("a", "k1", 2.0), graded("b", "k1", 2.0)]
    rows[1]["baseline"] = "c"
    with pytest.raises(population.MixedPopulationError):
        paired_arms.graded_rows(frame(rows), ["a", "b"])


def test_a_pair_reports_what_the_intersection_dropped(paired_arms: ModuleType, tmp_path: pathlib.Path) -> None:
    """Arm b solves two kernels arm a never did. The row must carry them as ``n_only_b`` and test
    the discordance, not quietly compare the six that survived."""
    rows: list[dict[str, object]] = []
    for kernel in KERNELS[:6]:
        rows += episode("a", kernel, 2.0, 100.0)
        rows += episode("b", kernel, 2.0, 100.0)
    for kernel in ("k7", "k8"):
        rows += episode("b", kernel, 2.0, 100.0)
        rows.append(call("a", kernel, 100.0))

    path = observations(rows, tmp_path)
    obs = paired_arms.load_observations(path)
    graded_rows = paired_arms.graded_rows(obs, ["a", "b"])
    best = paired_arms.best_by_arm_kernel(graded_rows)
    table = paired_arms.arm_aggregates(best, paired_arms.served_by_arm(obs), "numba")
    reported = paired_arms.pair_rows([("a", "b")], table, paired_arms.tokens_by_arm_kernel(obs), list(KERNELS), "f")

    speed = next(row for row in reported if row["leg"] == "speedup")
    assert (speed["n_a"], speed["n_b"], speed["n_both"]) == (6, 8, 6)
    assert (speed["n_only_a"], speed["n_only_b"]) == (0, 2)
    assert speed["coverage_p"] == pytest.approx(population.mcnemar_exact(0, 2))


def test_a_leg_below_the_interval_floor_reports_underpowered(paired_arms: ModuleType, tmp_path: pathlib.Path) -> None:
    """Under ``MIN_PAIRS_FOR_INTERVAL`` no interval and no p can be had, and the verdict says so
    rather than reading a bootstrap flag that is a coin toss at n = 2. The cost leg here is
    degenerate as well -- both arms spent the same on every kernel -- which is equally not a test."""
    rows: list[dict[str, object]] = []
    for kernel in KERNELS[:3]:
        rows += episode("a", kernel, 3.0, 100.0)
        rows += episode("b", kernel, 2.0, 100.0)

    path = observations(rows, tmp_path)
    obs = paired_arms.load_observations(path)
    best = paired_arms.best_by_arm_kernel(paired_arms.graded_rows(obs, ["a", "b"]))
    table = paired_arms.arm_aggregates(best, paired_arms.served_by_arm(obs), "numba")
    reported = paired_arms.pair_rows([("a", "b")], table, paired_arms.tokens_by_arm_kernel(obs), list(KERNELS), "f")

    assert len(KERNELS[:3]) < summary.MIN_PAIRS_FOR_INTERVAL
    for row in reported:
        assert row["verdict"] == "underpowered"
        assert math.isnan(float(row["ci_low"])) and math.isnan(float(row["ci_high"]))


def test_the_correction_runs_over_every_leg_of_every_pair(paired_arms: ModuleType, tmp_path: pathlib.Path) -> None:
    """The family is the whole table: two pairs on two legs is four tests, and a per-row threshold
    applied four times is the multiplicity error the correction exists to prevent."""
    rows: list[dict[str, object]] = []
    for index, kernel in enumerate(KERNELS):
        rows += episode("a", kernel, 2.0 + index * 0.1, 100.0 + index)
        rows += episode("b", kernel, 1.0 + index * 0.1, 200.0 + index)
        rows += episode("c", kernel, 1.5 + index * 0.1, 300.0 + index)

    path = observations(rows, tmp_path)
    obs = paired_arms.load_observations(path)
    best = paired_arms.best_by_arm_kernel(paired_arms.graded_rows(obs, ["a", "b", "c"]))
    table = paired_arms.arm_aggregates(best, paired_arms.served_by_arm(obs), "numba")
    tokens = paired_arms.tokens_by_arm_kernel(obs)
    reported = paired_arms.pair_rows([("a", "b"), ("a", "c")], table, tokens, list(KERNELS), "f")

    assert len(reported) == 4
    assert all(row["p_adjusted"] >= row["p_value"] for row in reported)
    assert {row["leg"] for row in reported} == {"speedup", "tokens"}


def test_the_estimate_is_the_hodges_lehmann_of_the_paired_logs(paired_arms: ModuleType, tmp_path: pathlib.Path) -> None:
    """The point, the interval and the p must describe ONE parameter. A ratio of geometric means --
    exp of the MEAN of the same logs -- is a different one, and on a skewed set the two disagree."""
    values = (1.05, 1.1, 0.95, 1.2, 0.9, 1.15, 1.02, 40.0)
    rows: list[dict[str, object]] = []
    for kernel, ratio in zip(KERNELS, values, strict=True):
        rows += episode("a", kernel, 2.0 * ratio, 100.0)
        rows += episode("b", kernel, 2.0, 100.0)

    path = observations(rows, tmp_path)
    obs = paired_arms.load_observations(path)
    best = paired_arms.best_by_arm_kernel(paired_arms.graded_rows(obs, ["a", "b"]))
    table = paired_arms.arm_aggregates(best, paired_arms.served_by_arm(obs), "numba")
    change, _ = paired_arms.score_leg(table["a"], table["b"])

    expected = summary.paired_change([math.log(value) for value in values])
    assert math.exp(change.estimate) == pytest.approx(math.exp(expected.estimate))
    assert math.exp(change.estimate) < summary.geomean(values)


def test_the_recovery_tags_match_the_writer(paired_arms: ModuleType) -> None:
    """``promote_unsubmitted.py`` writes these two spellings into ``submissions.optimizer`` and this
    module reads them. Two literals, one contract: a rename there must break here, not silently turn
    every recovered row into an ordinary submission."""
    writer = load_experiment_module("promote_unsubmitted")
    assert paired_arms.HARVESTED_TAG == writer.HARVESTED_TAG
    assert paired_arms.PROMOTED_TAG == writer.PROMOTED_TAG


def test_an_arm_row_counts_the_answers_nobody_submitted(paired_arms: ModuleType, tmp_path: pathlib.Path) -> None:
    """An arm whose rows are mostly recovered measured its agents' code and not their decision to
    ship it, so the counts sit in the table rather than in a footnote."""
    rows = [
        *episode("a", "k1", 2.0, 100.0),
        graded("a", "k2", 3.0, optimizer=paired_arms.HARVESTED_TAG),
        call("a", "k2", 100.0),
        graded("a", "k3", 4.0, optimizer=paired_arms.PROMOTED_TAG),
        call("a", "k3", 100.0),
    ]

    path = observations(rows, tmp_path)
    obs = paired_arms.load_observations(path)
    graded_frame = paired_arms.graded_rows(obs, ["a"])
    best = paired_arms.best_by_arm_kernel(graded_frame)
    served = paired_arms.served_by_arm(obs)
    table = paired_arms.arm_aggregates(best, served, "numba")
    row = paired_arms.arm_rows(best, graded_frame, table, served, paired_arms.tokens_by_arm_kernel(obs))[0]

    assert (row["n_solved"], row["n_harvested"], row["n_promoted"]) == (3, 1, 1)
