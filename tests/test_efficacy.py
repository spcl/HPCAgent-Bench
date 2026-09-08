# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Intervention efficacy: the properties the definition claims, asserted rather than assumed.

Every test here pins a property the metric is CHOSEN for -- scale invariance, antisymmetry, zero at
no effect, cost read the right way round -- because those are what make two arms comparable at all.
A metric that silently loses one of them still prints a plausible percentage.
"""

import math

import pytest

from hpcagent_bench.harness import efficacy as eff


def arms(before_s, after_s, before_c, after_c):
    """Four mappings keyed by task name, from four equal-length sequences."""
    names = [f"k{i}" for i in range(len(before_s))]
    return (
        dict(zip(names, before_s)),
        dict(zip(names, after_s)),
        dict(zip(names, before_c)),
        dict(zip(names, after_c)),
    )


def test_no_effect_is_exactly_one_and_zero():
    """The anchor the whole scale hangs on: identical arms must read no effect, not 'almost none'."""
    b, a, bc, ac = arms([1.0, 2.0, 0.5], [1.0, 2.0, 0.5], [10.0, 20.0, 5.0], [10.0, 20.0, 5.0])
    r = eff.efficacy(b, a, bc, ac)
    assert r.score.rho == pytest.approx(1.0)
    assert r.cost.rho == pytest.approx(1.0)
    assert r.q == pytest.approx(0.0)
    assert r.overall_effect == pytest.approx(1.0)
    assert r.score.wins == 0 and r.score.losses == 0 and r.score.ties == 3
    assert not r.score.significant, "an interval that cannot exclude zero must not read as an effect"
    assert not r.cost.significant


def test_swapping_the_arms_negates_q():
    """Antisymmetry. Without it the metric would answer differently depending on which arm the
    caller happened to call 'before', and no ranking built on it would mean anything."""
    b, a, bc, ac = arms([1.0, 2.0, 4.0], [2.0, 2.0, 1.0], [10.0, 30.0, 5.0], [20.0, 10.0, 5.0])
    forward = eff.efficacy(b, a, bc, ac)
    backward = eff.efficacy(a, b, ac, bc)
    assert backward.q == pytest.approx(-forward.q)
    assert backward.score.rho == pytest.approx(1.0 / forward.score.rho)
    assert backward.cost.rho == pytest.approx(1.0 / forward.cost.rho)
    assert backward.overall_effect == pytest.approx(1.0 / forward.overall_effect)


def test_a_cheaper_arm_is_an_improvement_not_a_regression():
    """rho_C is INVERTED on purpose. Read the other way round, every intervention that saved tokens
    would be reported as having made things worse -- the sign error the inversion exists to stop."""
    b, a, bc, ac = arms([1.0, 1.0], [1.0, 1.0], [100.0, 200.0], [50.0, 100.0])
    r = eff.efficacy(b, a, bc, ac)
    assert r.cost.rho == pytest.approx(2.0), "halving the tokens is a 2x improvement"
    assert r.cost.pct_change == pytest.approx(100.0)
    assert r.cost.wins == 2 and r.cost.losses == 0
    assert r.q > 0.0


def test_the_aggregate_is_scale_invariant_across_tasks():
    """The reason it is a geometric mean. A kernel timed in nanoseconds and one timed in seconds must
    move the aggregate by the same factor for the same RELATIVE change, or the unit picks the winner."""
    b, a, bc, ac = arms([1.0, 1.0, 1.0], [2.0, 3.0, 0.5], [10.0, 10.0, 10.0], [5.0, 20.0, 10.0])
    plain = eff.efficacy(b, a, bc, ac)
    scaled_b = {k: v * 1e6 for k, v in b.items()}
    scaled_a = {k: v * 1e6 for k, v in a.items()}
    scaled = eff.efficacy(scaled_b, scaled_a, bc, ac)
    assert scaled.score.rho == pytest.approx(plain.score.rho)
    assert scaled.q == pytest.approx(plain.q)


def test_log_rho_is_the_mean_of_the_deltas():
    """ln rho and mean(d_i) are the same quantity, and the median, the counts and the interval are
    all computed over d. If these ever disagree the pairing is wrong, not the rounding."""
    b, a, bc, ac = arms([1.0, 2.0, 4.0, 8.0], [3.0, 1.0, 9.0, 2.0], [7.0, 5.0, 11.0, 2.0], [1.0, 9.0, 3.0, 4.0])
    r = eff.efficacy(b, a, bc, ac)
    assert r.score.log_rho == pytest.approx(math.log(r.score.rho))
    assert r.cost.log_rho == pytest.approx(math.log(r.cost.rho))
    assert r.q == pytest.approx(0.5 * r.score.log_rho + 0.5 * r.cost.log_rho)


def test_the_median_and_the_counts_expose_a_tail_the_mean_hides():
    """The robustness check earning its place: one kernel that moved 100x carries a mean that nine
    regressions should have sunk. The geomean says improvement, the median and the count say not."""
    before = [1.0] * 10
    after = [100.0] + [0.9] * 9
    b, a, bc, ac = arms(before, after, [1.0] * 10, [1.0] * 10)
    r = eff.efficacy(b, a, bc, ac)
    assert r.score.rho > 1.0, "the mean of the logs is carried by the one big win"
    assert r.score.median_delta < 0.0, "the median must show the typical task got worse"
    assert r.score.losses == 9 and r.score.wins == 1


def test_tasks_are_paired_by_name_not_by_position():
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


def test_arms_that_share_no_task_are_refused():
    """Nothing paired means nothing to say, and an empty geomean of 1.0 would say 'no effect'."""
    with pytest.raises(ValueError, match="share no task"):
        eff.efficacy({"a": 1.0}, {"b": 1.0}, {"a": 1.0}, {"b": 1.0})


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_a_value_that_is_not_a_ratio_is_refused(bad):
    """A zero or negative speedup is a MISSING measurement, not a small one. Skipping it silently
    would change which tasks the pairing covers without the report ever saying so."""
    b, a, bc, ac = arms([1.0, 1.0], [1.0, bad], [1.0, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        eff.efficacy(b, a, bc, ac)


def test_the_interval_is_deterministic_for_the_same_input():
    """The bounds go in a paper. The same inputs have to give the same interval on a rerun."""
    deltas = [0.1, -0.2, 0.4, 0.05, -0.01, 0.9, -0.3]
    first = eff.bootstrap_interval(deltas, resamples=500)
    second = eff.bootstrap_interval(deltas, resamples=500)
    assert first == second
    assert first != eff.bootstrap_interval(deltas, resamples=500, seed=eff.BOOTSTRAP_SEED + 1)


def test_the_interval_brackets_the_mean_and_reads_no_effect_when_it_covers_zero():
    b, a, bc, ac = arms([1.0, 1.0, 1.0, 1.0], [2.0, 0.5, 2.0, 0.5], [1.0] * 4, [1.0] * 4)
    r = eff.efficacy(b, a, bc, ac, resamples=2000)
    assert r.score.ci_low <= r.score.log_rho <= r.score.ci_high
    assert r.score.ci_low <= 0.0 <= r.score.ci_high
    assert not r.score.significant, "gains and losses that cancel are no effect, not a small one"


def test_a_single_task_cannot_bound_anything():
    """One paired observation has no spread to resample, and must not print a narrow interval as if
    it did -- a degenerate one at its own value is the honest answer."""
    r = eff.efficacy({"a": 1.0}, {"a": 4.0}, {"a": 1.0}, {"a": 1.0})
    assert r.score.tasks == 1
    assert r.score.ci_low == pytest.approx(r.score.ci_high) == pytest.approx(math.log(4.0))


def test_weights_must_sum_to_one_and_stay_non_negative():
    b, a, bc, ac = arms([1.0], [2.0], [1.0], [1.0])
    with pytest.raises(ValueError, match="sum to 1"):
        eff.efficacy(b, a, bc, ac, score_weight=0.7, cost_weight=0.7)
    with pytest.raises(ValueError, match="negative weight"):
        eff.efficacy(b, a, bc, ac, score_weight=1.5, cost_weight=-0.5)


def test_a_weighting_can_favour_either_axis_without_moving_the_point():
    """Q is a PROXY. Changing the weights must move the ranking number and leave the two ratios --
    the thing Pareto dominance is decided on -- exactly where they were."""
    b, a, bc, ac = arms([1.0, 1.0], [4.0, 4.0], [1.0, 1.0], [2.0, 2.0])
    even = eff.efficacy(b, a, bc, ac)
    score_led = eff.efficacy(b, a, bc, ac, score_weight=1.0)
    assert score_led.point == even.point
    assert score_led.q == pytest.approx(math.log(4.0))
    assert score_led.q > even.q, "the cost got worse, so weighting it out must raise Q"


def test_dominance_needs_both_axes_and_a_strict_gain_on_one():
    def at(rho_s, rho_c):
        b, a, bc, ac = arms([1.0], [rho_s], [1.0], [1.0 / rho_c])
        return eff.efficacy(b, a, bc, ac)

    strong, weak, traded = at(2.0, 2.0), at(1.5, 1.5), at(4.0, 0.5)
    assert eff.dominates(strong, weak)
    assert not eff.dominates(weak, strong)
    assert not eff.dominates(strong, traded), "faster but costlier is a trade no weighting settles"
    assert not eff.dominates(traded, strong)
    assert not eff.dominates(strong, strong), "dominance is strict; nothing dominates itself"


def test_the_front_keeps_every_intervention_nothing_dominates():
    def at(rho_s, rho_c):
        b, a, bc, ac = arms([1.0], [rho_s], [1.0], [1.0 / rho_c])
        return eff.efficacy(b, a, bc, ac)

    front = eff.pareto_front({"cheap": at(1.2, 4.0), "fast": at(4.0, 1.2), "dominated": at(1.1, 1.1)})
    assert set(front) == {"cheap", "fast"}


def test_the_row_reports_percentages_and_carries_the_robustness_checks():
    """What lands in the CSV is what a reader sees. A row that dropped the counts would let a
    tail-carried result print as a clean percentage."""
    b, a, bc, ac = arms([1.0, 1.0], [2.0, 3.0], [10.0, 10.0], [5.0, 5.0])
    row = eff.as_row("skills", eff.efficacy(b, a, bc, ac))
    assert row["intervention"] == "skills" and row["tasks"] == 2
    assert row["score_pct"] > 0.0 and row["cost_pct"] == pytest.approx(100.0)
    for key in ("score_wins", "score_losses", "score_median_delta", "score_ci_low_pct", "score_significant"):
        assert key in row, f"the row dropped {key}, which is the check the percentage cannot make"
