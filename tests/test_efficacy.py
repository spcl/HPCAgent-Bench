# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Intervention efficacy: the properties the definition claims, asserted rather than assumed.

Every test here pins a property the metric is CHOSEN for -- scale invariance, antisymmetry, zero at
no effect, cost read the right way round -- because those are what make two arms comparable at all.
A metric that silently loses one of them still prints a plausible percentage.
"""

import math

import numpy as np
import pytest

from hpcagent_bench.harness import efficacy as eff
from hpcagent_bench.stats import summary

#: ``log(c_best_su / fortran_best_su)`` for every kernel in the shipped
#: ``reproducibility/llr40/analysis/per_language_kernel.csv`` that both languages reached: the
#: real shape a paired delta has here, right-tailed with exact ties from the 1% speedup ladder.
#: A Gaussian fixture would measure a distribution this analysis never sees.
LLR40_LOG_DELTAS: tuple[float, ...] = (
    0.358216,
    1.303498,
    0.039798,
    1.194033,
    1.034826,
    0.348268,
    -0.019896,
    -1.860710,
    0.169168,
    0.049760,
    -0.636832,
    0.069630,
    0.676656,
    0.009941,
    0.000000,
    0.417899,
    -0.417953,
    -0.039774,
    0.169167,
    0.089531,
    -0.636824,
    0.009952,
    2.378006,
    0.039796,
    0.000000,
    1.273632,
    0.000000,
    0.159217,
    -0.089515,
    0.179090,
    0.039808,
    1.701534,
    -0.149319,
    0.159276,
    -0.059627,
    0.298405,
    0.348472,
    -0.676398,
    0.000000,
)


def arms(before_s, after_s, before_c, after_c):
    """Four mappings keyed by task name, from four equal-length sequences."""
    names = [f"k{i}" for i in range(len(before_s))]
    return (
        dict(zip(names, before_s)),
        dict(zip(names, after_s)),
        dict(zip(names, before_c)),
        dict(zip(names, after_c)),
    )


def test_no_effect_is_exactly_one_and_zero() -> None:
    """The anchor the whole scale hangs on: identical arms must read no effect, not 'almost none'."""
    b, a, bc, ac = arms([1.0, 2.0, 0.5], [1.0, 2.0, 0.5], [10.0, 20.0, 5.0], [10.0, 20.0, 5.0])
    r = eff.efficacy(b, a, bc, ac)
    assert r.score.rho == pytest.approx(1.0)
    assert r.cost.rho == pytest.approx(1.0)
    assert r.q == pytest.approx(0.0)
    assert r.overall_effect == pytest.approx(1.0)
    assert r.score.wins == 0 and r.score.losses == 0 and r.score.ties == 3
    assert r.score.pvalue == 1.0 and r.cost.pvalue == 1.0


def test_swapping_the_arms_negates_q() -> None:
    """Antisymmetry. Without it the metric would answer differently depending on which arm the
    caller happened to call 'before', and no ranking built on it would mean anything."""
    b, a, bc, ac = arms([1.0, 2.0, 4.0], [2.0, 2.0, 1.0], [10.0, 30.0, 5.0], [20.0, 10.0, 5.0])
    forward = eff.efficacy(b, a, bc, ac)
    backward = eff.efficacy(a, b, ac, bc)
    assert backward.q == pytest.approx(-forward.q)
    assert backward.score.rho == pytest.approx(1.0 / forward.score.rho)
    assert backward.cost.rho == pytest.approx(1.0 / forward.cost.rho)
    assert backward.overall_effect == pytest.approx(1.0 / forward.overall_effect)


def test_a_cheaper_arm_is_an_improvement_not_a_regression() -> None:
    """rho_C is INVERTED on purpose. Read the other way round, every intervention that saved tokens
    would be reported as having made things worse -- the sign error the inversion exists to stop."""
    b, a, bc, ac = arms([1.0, 1.0], [1.0, 1.0], [100.0, 200.0], [50.0, 100.0])
    r = eff.efficacy(b, a, bc, ac)
    assert r.cost.rho == pytest.approx(2.0), "halving the tokens is a 2x improvement"
    assert r.cost.pct_change == pytest.approx(100.0)
    assert r.cost.wins == 2 and r.cost.losses == 0
    assert r.q > 0.0


def test_the_aggregate_is_scale_invariant_across_tasks() -> None:
    """The reason it is a geometric mean. A kernel timed in nanoseconds and one timed in seconds must
    move the aggregate by the same factor for the same RELATIVE change, or the unit picks the winner."""
    b, a, bc, ac = arms([1.0, 1.0, 1.0], [2.0, 3.0, 0.5], [10.0, 10.0, 10.0], [5.0, 20.0, 10.0])
    plain = eff.efficacy(b, a, bc, ac)
    scaled_b = {k: v * 1e6 for k, v in b.items()}
    scaled_a = {k: v * 1e6 for k, v in a.items()}
    scaled = eff.efficacy(scaled_b, scaled_a, bc, ac)
    assert scaled.score.rho == pytest.approx(plain.score.rho)
    assert scaled.q == pytest.approx(plain.q)


def test_log_rho_is_the_mean_of_the_deltas() -> None:
    """ln rho and mean(d_i) are the same quantity, and the median, the counts and the interval are
    all computed over d. If these ever disagree the pairing is wrong, not the rounding."""
    b, a, bc, ac = arms([1.0, 2.0, 4.0, 8.0], [3.0, 1.0, 9.0, 2.0], [7.0, 5.0, 11.0, 2.0], [1.0, 9.0, 3.0, 4.0])
    r = eff.efficacy(b, a, bc, ac)
    assert r.score.log_rho == pytest.approx(math.log(r.score.rho))
    assert r.cost.log_rho == pytest.approx(math.log(r.cost.rho))
    assert r.q == pytest.approx(0.5 * r.score.log_rho + 0.5 * r.cost.log_rho)


def test_the_median_and_the_counts_expose_a_tail_the_mean_hides() -> None:
    """The robustness check earning its place: one kernel that moved 100x carries a mean that nine
    regressions should have sunk. The geomean says improvement, the median and the count say not."""
    before = [1.0] * 10
    after = [100.0] + [0.9] * 9
    b, a, bc, ac = arms(before, after, [1.0] * 10, [1.0] * 10)
    r = eff.efficacy(b, a, bc, ac)
    assert r.score.rho > 1.0, "the mean of the logs is carried by the one big win"
    assert r.score.median_delta < 0.0, "the median must show the typical task got worse"
    assert r.score.losses == 9 and r.score.wins == 1


def test_tasks_are_paired_by_name_not_by_position() -> None:
    """An arm that crashed on a kernel has no row for it. Zipping would pair kernel k against k+1
    and report a difference between two different kernels as an effect."""
    before_s = {"a": 1.0, "b": 2.0, "c": 4.0}
    after_s = {"a": 2.0, "c": 8.0}
    costs_b = {"a": 10.0, "b": 10.0, "c": 10.0}
    costs_a = {"a": 10.0, "c": 10.0}
    r = eff.efficacy(before_s, after_s, costs_b, costs_a)
    assert r.tasks == ("a", "c"), "only the shared tasks are comparable"
    assert r.score.tasks == 2
    assert r.score.rho == pytest.approx(2.0)


def test_arms_that_share_no_task_are_refused() -> None:
    """Nothing paired means nothing to say, and an empty geomean of 1.0 would say 'no effect'."""
    with pytest.raises(ValueError, match="share no task"):
        eff.efficacy({"a": 1.0}, {"b": 1.0}, {"a": 1.0}, {"b": 1.0})


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_a_value_that_is_not_a_ratio_is_refused(bad) -> None:
    """A zero or negative speedup is a MISSING measurement, not a small one. Skipping it silently
    would change which tasks the pairing covers without the report ever saying so."""
    b, a, bc, ac = arms([1.0, 1.0], [1.0, bad], [1.0, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        eff.efficacy(b, a, bc, ac)


def test_the_interval_is_deterministic_for_the_same_input() -> None:
    """The bounds go in a paper. The same inputs have to give the same interval on a rerun."""
    deltas = [0.1, -0.2, 0.4, 0.05, -0.01, 0.9, -0.3]
    first = eff.bootstrap_interval(deltas, resamples=500)
    second = eff.bootstrap_interval(deltas, resamples=500)
    assert first == second
    assert first != eff.bootstrap_interval(deltas, resamples=500, seed=eff.BOOTSTRAP_SEED + 1)


def test_the_bootstrap_interval_brackets_the_mean_it_bounds() -> None:
    """The bootstrap interval is FOR ``rho``, so it has to contain it. It decides nothing: its
    measured coverage against a null is 0.70 at n = 4, which is not a 5% statement about anything."""
    b, a, bc, ac = arms([1.0, 1.0, 1.0, 1.0], [2.0, 0.5, 2.0, 0.5], [1.0] * 4, [1.0] * 4)
    r = eff.efficacy(b, a, bc, ac, resamples=2000)
    assert r.score.ci_low <= r.score.log_rho <= r.score.ci_high
    assert r.score.ci_low <= 0.0 <= r.score.ci_high


def test_gains_and_losses_that_cancel_are_not_significant() -> None:
    """The null case the whole decision exists to get right: six tasks, three up and three down by
    the same factor, is no effect -- not a small one, and not an effect whose sign the mean picked."""
    b, a, bc, ac = arms([1.0] * 6, [2.0, 0.5, 2.0, 0.5, 2.0, 0.5], [1.0] * 6, [1.0] * 6)
    r = eff.efficacy(b, a, bc, ac)
    rows = eff.family_rows({"cancel": r})
    assert rows[0]["score_verdict"] == eff.NOT_SIGNIFICANT
    assert r.score.pvalue > 0.05


def test_a_pair_with_too_few_tasks_reports_underpowered_rather_than_a_boolean() -> None:
    """A two-task pair cannot reach any alpha whatever it measured, and the llr40 skill pairs run
    at n = 2, 3 and 4. A boolean column has only 'yes' and 'no' to say, and both are wrong there."""
    b, a, bc, ac = arms([1.0, 1.0], [4.0, 4.0], [100.0, 100.0], [25.0, 25.0])
    r = eff.efficacy(b, a, bc, ac)
    assert r.score.underpowered and math.isnan(r.score.pvalue)
    rows = eff.family_rows({"tiny": r})
    assert rows[0]["score_verdict"] == eff.UNDERPOWERED and rows[0]["cost_verdict"] == eff.UNDERPOWERED
    assert rows[0]["score_hl_pct"] > 0.0, "the ESTIMATE still stands; only the interval is withheld"


def test_a_single_task_cannot_bound_anything() -> None:
    """One paired observation has no spread to resample, and must not print a narrow interval as if
    it did -- a degenerate one at its own value is the honest answer."""
    r = eff.efficacy({"a": 1.0}, {"a": 4.0}, {"a": 1.0}, {"a": 1.0})
    assert r.score.tasks == 1
    assert r.score.ci_low == pytest.approx(r.score.ci_high) == pytest.approx(math.log(4.0))


def test_weights_must_sum_to_one_and_stay_non_negative() -> None:
    b, a, bc, ac = arms([1.0], [2.0], [1.0], [1.0])
    with pytest.raises(ValueError, match="sum to 1"):
        eff.efficacy(b, a, bc, ac, score_weight=0.7, cost_weight=0.7)
    with pytest.raises(ValueError, match="negative weight"):
        eff.efficacy(b, a, bc, ac, score_weight=1.5, cost_weight=-0.5)


def test_a_weighting_can_favour_either_axis_without_moving_the_point() -> None:
    """Q is a PROXY. Changing the weights must move the ranking number and leave the two ratios --
    the thing Pareto dominance is decided on -- exactly where they were."""
    b, a, bc, ac = arms([1.0, 1.0], [4.0, 4.0], [1.0, 1.0], [2.0, 2.0])
    even = eff.efficacy(b, a, bc, ac)
    score_led = eff.efficacy(b, a, bc, ac, score_weight=1.0)
    assert score_led.point == even.point
    assert score_led.q == pytest.approx(math.log(4.0))
    assert score_led.q > even.q, "the cost got worse, so weighting it out must raise Q"


def test_dominance_needs_both_axes_and_a_strict_gain_on_one() -> None:
    def at(rho_s, rho_c):
        b, a, bc, ac = arms([1.0], [rho_s], [1.0], [1.0 / rho_c])
        return eff.efficacy(b, a, bc, ac)

    strong, weak, traded = at(2.0, 2.0), at(1.5, 1.5), at(4.0, 0.5)
    assert eff.dominates(strong, weak)
    assert not eff.dominates(weak, strong)
    assert not eff.dominates(strong, traded), "faster but costlier is a trade no weighting settles"
    assert not eff.dominates(traded, strong)
    assert not eff.dominates(strong, strong), "dominance is strict; nothing dominates itself"


def test_the_front_keeps_every_intervention_nothing_dominates() -> None:
    def at(rho_s, rho_c):
        b, a, bc, ac = arms([1.0], [rho_s], [1.0], [1.0 / rho_c])
        return eff.efficacy(b, a, bc, ac)

    front = eff.pareto_front({"cheap": at(1.2, 4.0), "fast": at(4.0, 1.2), "dominated": at(1.1, 1.1)})
    assert set(front) == {"cheap", "fast"}


def test_the_row_reports_percentages_and_carries_the_robustness_checks() -> None:
    """What lands in the CSV is what a reader sees. A row that dropped the counts would let a
    tail-carried result print as a clean percentage."""
    b, a, bc, ac = arms([1.0] * 8, [2.0, 3.0] * 4, [10.0] * 8, [5.0] * 8)
    row = eff.as_row("skills", eff.efficacy(b, a, bc, ac))
    assert row["intervention"] == "skills" and row["tasks"] == 8
    assert row["score_pct"] > 0.0 and row["cost_pct"] == pytest.approx(100.0)
    for key in ("score_wins", "score_losses", "score_median_delta", "score_ci_low_pct", "score_p_value"):
        assert key in row, f"the row dropped {key}, which is the check the percentage cannot make"
    assert row["score_verdict"] == eff.UNCORRECTED, "a row outside a family is not a finding"
    assert math.isnan(float(row["score_p_adjusted"]))


def flat_pair(score_ratio: float, cost_ratio: float, tasks: int = 12) -> eff.Efficacy:
    """One intervention whose every task moved by the same two factors, for counting flags."""
    b, a, bc, ac = arms([1.0] * tasks, [score_ratio] * tasks, [100.0] * tasks, [100.0 / cost_ratio] * tasks)
    return eff.efficacy(b, a, bc, ac)


def test_every_flag_in_a_table_is_corrected_across_the_family_it_belongs_to() -> None:
    """Three models x two languages x two axes is twelve tests, and twelve uncorrected 5%
    thresholds fire at least once on 46% of tables where nothing happened. The correction has to
    be over the family, not over whichever row the reader is looking at."""
    members = {f"m{i}": flat_pair(1.02, 1.02, tasks=6 + i) for i in range(6)}
    rows = eff.family_rows(members, family="skills")
    assert len(rows) == 6
    for row in rows:
        assert row["score_family"] == "skills" and row["cost_family"] == "skills"
        assert float(row["score_p_adjusted"]) >= float(row["score_p_value"]), (
            "an adjusted p below the raw one would be a correction in the wrong direction"
        )
    raw = [float(row["score_p_value"]) for row in rows]
    adjusted = [float(row["score_p_adjusted"]) for row in rows]
    assert min(adjusted) > min(raw), "twelve tests corrected as one must move at least one threshold"


def test_a_pooled_row_built_from_the_family_is_not_counted_as_a_thirteenth_test() -> None:
    """The pooled row re-reads the same tasks the pair rows already carry. Entered into the
    correction it would weaken every member with its own evidence and then present that evidence a
    second time as a finding of its own."""
    members = {f"m{i}": flat_pair(2.0, 1.5) for i in range(3)}
    pooled = flat_pair(2.0, 1.5, tasks=36)
    with_pool = eff.family_rows(members, dependent={"skills:all": pooled})
    without = eff.family_rows(members)
    assert len(with_pool) == 4
    assert [row["score_p_adjusted"] for row in with_pool[:3]] == [row["score_p_adjusted"] for row in without]
    assert with_pool[-1]["score_verdict"] == eff.NOT_INDEPENDENT
    assert math.isnan(float(with_pool[-1]["score_p_adjusted"]))


def test_a_test_that_was_never_run_does_not_enter_the_correction() -> None:
    """An underpowered pair performed no test, so counting it in m would raise every real member's
    threshold to pay for a claim nobody made."""
    tested = {"big": flat_pair(2.0, 1.5)}
    mixed = {"big": flat_pair(2.0, 1.5), "tiny": flat_pair(2.0, 1.5, tasks=2)}
    alone = eff.family_rows(tested)
    beside = eff.family_rows(mixed)
    assert beside[1]["score_verdict"] == eff.UNDERPOWERED
    assert float(beside[0]["score_p_adjusted"]) == pytest.approx(float(alone[0]["score_p_adjusted"]))


#: Trials per size in the coverage simulation, and the seed that pins it. A coverage claim has to
#: be measured; at 500 trials the Monte-Carlo error on a 5% rate is about 1 point.
COVERAGE_TRIALS: int = 500
COVERAGE_SEED: int = 20260911


def false_positive_rate(population: np.ndarray, n: int) -> tuple[float, float, float]:
    """``(false positives, coverage of the emitted intervals, share withheld)`` over one size."""
    rng = np.random.default_rng(COVERAGE_SEED)
    false_positives, emitted = 0, 0
    for _ in range(COVERAGE_TRIALS):
        change = summary.paired_change(rng.choice(population, size=n, replace=True))
        if not math.isfinite(change.low):
            continue
        emitted += 1
        false_positives += int(not change.low <= 0.0 <= change.high)
    coverage = math.nan if emitted == 0 else 1.0 - false_positives / emitted
    return false_positives / COVERAGE_TRIALS, coverage, 1.0 - emitted / COVERAGE_TRIALS


@pytest.mark.parametrize(
    "n_pairs, max_false_positive, min_coverage, withheld_share",
    [
        pytest.param(2, 0.0, math.nan, 1.0, id="n=2 -- withheld, an llr40 skill pair"),
        pytest.param(4, 0.0, math.nan, 1.0, id="n=4 -- withheld, an llr40 skill pair"),
        pytest.param(10, 0.08, 0.90, 0.0, id="n=10"),
        pytest.param(20, 0.08, 0.90, 0.0, id="n=20"),
        pytest.param(39, 0.08, 0.90, 0.0, id="n=39 -- the focus40 roster"),
    ],
)
def test_the_significance_decision_holds_its_nominal_level_on_the_real_delta_shape(
    n_pairs: int, max_false_positive: float, min_coverage: float, withheld_share: float
) -> None:
    """The decision behind ``score_verdict`` fires on a population with no effect at most 5% of the
    time, or the flag is the least reliable number in the table printed as the most confident one.
    The percentile bootstrap of the mean it replaced missed the same null on 27% of samples at
    n = 4 and 7% at n = 39."""
    population = np.asarray(LLR40_LOG_DELTAS, dtype=float)
    population = population - summary.hodges_lehmann(population)
    rate, coverage, withheld = false_positive_rate(population, n_pairs)
    assert withheld == pytest.approx(withheld_share)
    assert rate <= max_false_positive, f"fired on {rate:.1%} of null samples at n={n_pairs}"
    if not math.isnan(min_coverage):
        assert coverage >= min_coverage, f"covered the null on {coverage:.3f} of samples at n={n_pairs}"
