# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The one definition of each statistic every figure in this repo reports.

These are the properties a figure relies on without restating them: the signed axis is odd about
no change, the geometric mean is taken in log space and refuses what is not a ratio, a set with no
spread gets an interval that says so rather than one that pretends to bound something, and the
paired estimate, its interval and its p value all describe the same quantity.
"""

import math

import numpy as np
import pytest
from scipy.stats import wilcoxon

from hpcagent_bench.stats import inference, summary


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [(2.0, 1.0), (3.0, 2.0), (1.0, 0.0), (0.5, -1.0), (0.25, -3.0), (4.0, 3.0), (1.0 / 3.0, -2.0)],
)
def test_signed_axis_mapping(ratio: float, expected: float) -> None:
    assert summary.signed_change(ratio) == pytest.approx(expected)


@pytest.mark.parametrize("ratio", [1.25, 2.0, 3.0, 7.5, 100.0])
def test_signed_axis_is_odd_about_no_change(ratio: float) -> None:
    """A win and the identical loss must be the same distance from zero, or the axis flatters one."""
    assert summary.signed_change(1.0 / ratio) == pytest.approx(-summary.signed_change(ratio))


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_an_unmeasured_cell_is_not_no_change(bad: float) -> None:
    """0 is the exact value of "measured, and nothing changed". An absent measurement may not claim it."""
    assert math.isnan(summary.signed_change(bad))


def test_geomean_is_the_ratio_whose_product_matches() -> None:
    assert summary.geomean([2.0, 8.0]) == pytest.approx(4.0)
    assert summary.geomean([2.0, 0.5]) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [[], [2.0, 0.0], [2.0, -1.0], [2.0, math.nan], [2.0, math.inf]])
def test_geomean_refuses_what_is_not_a_set_of_ratios(bad: list[float]) -> None:
    """A zero or a negative is a MISSING measurement, not a slow one. Six copies of this function
    disagreed about that, and the one that clamped reported a catastrophic regression that never
    happened -- so the arithmetic raises and the caller has to say that dropping is what it means."""
    with pytest.raises(ValueError):
        summary.geomean(bad)


def test_geomean_asked_to_drop_skips_missing_measurements_and_has_none_for_an_empty_set() -> None:
    """``unusable="drop"`` is the caller saying a zero, a negative or a non-finite entry is a missing
    measurement. What is left is averaged; nothing left has no geometric mean, which is NaN rather
    than the 0.0 of a collapse or the 1.0 of no change."""
    assert summary.geomean([2.0, 0.0, -1.0, math.nan, 8.0], unusable=summary.Unusable.DROP) == pytest.approx(4.0)
    assert math.isnan(summary.geomean([0.0, -1.0], unusable=summary.Unusable.DROP))
    assert math.isnan(summary.geomean([], unusable=summary.Unusable.DROP))


def test_usable_ratios_drops_and_warns() -> None:
    with pytest.warns(UserWarning, match="not finite positive ratios"):
        kept = summary.usable_ratios([2.0, 0.0, -1.0, math.nan, 8.0], label="cell")
    assert kept.tolist() == [2.0, 8.0]


def test_single_point_interval_collapses() -> None:
    """n=1 has no spread to estimate, so the interval is the point -- and must not raise."""
    interval = summary.geomean_ci([5.0])
    assert interval.point == pytest.approx(5.0)
    assert interval.low == pytest.approx(5.0) and interval.high == pytest.approx(5.0)
    assert interval.n == 1


def test_geomean_interval_brackets_the_centre() -> None:
    interval = summary.geomean_ci([2.0, 4.0, 8.0])
    assert interval.point == pytest.approx(4.0)
    assert interval.low < interval.point < interval.high


def test_the_geomean_interval_is_asymmetric_on_the_ratio_scale() -> None:
    """exp is not linear, so a symmetric +/- half-width would be wrong on a ratio axis. The ends
    are equidistant in LOG space and therefore not in ratio space."""
    interval = summary.geomean_ci([1.5, 3.0, 0.25, 8.0])
    assert interval.point - interval.low != pytest.approx(interval.high - interval.point)
    assert math.log(interval.point) - math.log(interval.low) == pytest.approx(
        math.log(interval.high) - math.log(interval.point)
    )


def test_the_geomean_interval_is_the_95_percent_log_t_interval() -> None:
    """Ratios 1,2,4,1,2,4: log2 values 0,1,2,0,1,2, mean 1, sd sqrt(0.8), t(0.975, 5) = 2.5706.
    The ends are 2^(1 -/+ 2.5706 * sqrt(0.8/6)) = 1.04345 and 3.83345, by hand."""
    interval = summary.geomean_ci([1.0, 2.0, 4.0, 1.0, 2.0, 4.0])
    assert (interval.point, interval.low, interval.high) == pytest.approx((2.0, 1.0434462081689584, 3.833451086107456))


@pytest.mark.parametrize(("n", "has_interval"), [(5, False), (6, True)])
def test_a_figure_geomean_draws_its_log_t_interval_only_from_six_values(n: int, has_interval: bool) -> None:
    """Below 6 values the point stays and the interval is withheld; from 6 it is the log-t interval."""
    interval = summary.geomean_interval([1.0, 2.0, 4.0, 1.0, 2.0, 4.0][:n])
    assert interval.method == ("log-t" if has_interval else "underpowered")
    assert math.isfinite(interval.low) == has_interval == math.isfinite(interval.high)


def test_the_paired_geomean_is_a_paired_t_test_with_a_log_t_interval() -> None:
    """The same six log ratios as a paired leg: the same interval, and the paired t p at
    t = 1 / sqrt(0.8/6) = 2.7386 on 5 degrees of freedom, 0.04086."""
    change = summary.paired_geomean([math.log(2.0) * value for value in (0, 1, 2, 0, 1, 2)])
    assert change.method == "paired-t"
    assert math.exp(change.low) == pytest.approx(1.0434462081689584)
    assert math.exp(change.high) == pytest.approx(3.833451086107456)
    assert change.pvalue == pytest.approx(0.040859403859295894)


def test_the_interval_names_what_it_is_for() -> None:
    assert summary.geomean_ci([2.0, 4.0]).label() == "95% log-t CI for geomean"


def test_hodges_lehmann_is_the_median_of_the_walsh_averages() -> None:
    """The six Walsh averages of [1, 2, 4] are 1, 1.5, 2, 2.5, 3, 4 -- their median is 2.25, which
    is not the sample median (2) and is the point the signed-rank test inverts to."""
    assert summary.walsh_averages([1.0, 2.0, 4.0]).tolist() == [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    assert summary.hodges_lehmann([1.0, 2.0, 4.0]) == pytest.approx(2.25)


def test_paired_change_agrees_with_its_own_test() -> None:
    """The point, the interval and the p value describe ONE quantity, so a significant result
    cannot have an interval covering zero."""
    rng = np.random.default_rng(3)
    result = summary.paired_change(rng.normal(0.6, 0.4, 30))
    assert result.pvalue < 0.05
    assert result.low > 0.0 and result.low < result.estimate < result.high
    assert result.method == "signed-rank-exact"


def test_paired_change_uses_the_exact_null_where_one_exists() -> None:
    """A normal approximation above n=25 was one copy's cutoff and scipy's exact test disagreed
    with it by 4e-3 in p at n=40. The exact distribution is computed wherever it is available."""
    rng = np.random.default_rng(7)
    rng.normal(0.2, 1.0, 12)
    rng.normal(0.15, 0.8, 25)
    result = summary.paired_change(rng.normal(0.15, 0.8, 40))
    assert result.method == "signed-rank-exact"
    assert result.pvalue == pytest.approx(0.18761708125202858)


def test_paired_change_drops_zero_differences_from_n() -> None:
    """A zero supports neither direction; keeping it would inflate n and shrink p for free."""
    result = summary.paired_change([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert result.n == 6 and result.ties == 2 and result.wins == 6 and result.losses == 0


def test_paired_change_says_underpowered_instead_of_drawing_an_interval() -> None:
    """Below six pairs the two-sided test cannot reach 0.05 whatever the data says, so an interval
    would be decoration."""
    result = summary.paired_change([1.0, 2.0, 3.0, 4.0])
    assert result.method == "underpowered"
    assert math.isnan(result.low) and math.isnan(result.high) and math.isnan(result.pvalue)


def test_paired_change_on_nothing_is_not_an_effect() -> None:
    result = summary.paired_change([0.0, 0.0, 0.0])
    assert result.method == "degenerate" and result.estimate == 0.0 and result.pvalue == 1.0


def test_a_tied_sample_falls_back_and_says_so() -> None:
    """scipy has no exact null with ties, so the method string has to record the approximation
    rather than let a reader assume the exact test ran."""
    result = summary.paired_change([0.5] * 8 + [-0.5] * 3)
    assert result.method == "signed-rank-approx"


def test_the_estimator_does_not_change_with_n() -> None:
    """A row with too few kernels for an interval still reports the SAME statistic as the rows
    beside it. Reporting a plain median there and a Hodges-Lehmann estimate elsewhere puts two
    statistics on one axis, distinguishable only by counting each row's kernels."""
    few = [0.1, 0.9, -0.4]
    assert summary.paired_change(few).estimate == pytest.approx(summary.hodges_lehmann(few))
    assert summary.paired_change(few).estimate != pytest.approx(float(np.median(few)))


def test_paired_geomean_is_the_geometric_mean_of_the_paired_ratios() -> None:
    """An arm comparison is reported as the geomean of its per-kernel ratios, so one kernel at 40x
    moves the estimate exactly as it moves that geomean."""
    ratios = [1.05, 1.1, 0.95, 1.2, 0.9, 1.15, 1.02, 40.0]
    change = summary.paired_geomean([math.log(ratio) for ratio in ratios])
    assert math.exp(change.estimate) == pytest.approx(summary.geomean(ratios))


@pytest.mark.parametrize("shift", [0.0, 0.1, 0.2, 0.35, 0.6])
def test_paired_geomean_interval_excludes_no_change_exactly_when_its_test_rejects(shift: float) -> None:
    """The interval and the p value are on the same mean log, so a reader cannot get a starred point
    whose interval crosses 1x, or an interval clear of 1x without a star."""
    change = summary.paired_geomean(np.random.default_rng(11).normal(shift, 0.5, 25))
    clear_of_zero = change.low > 0.0 or change.high < 0.0
    assert clear_of_zero == (change.pvalue < summary.DEFAULT_ALPHA), (change.low, change.high, change.pvalue)


def test_paired_geomean_keeps_the_kernels_that_did_not_change() -> None:
    """Dropping the zero logs would report the change of the kernels that moved as the arm's change."""
    change = summary.paired_geomean([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    assert (change.n, change.ties) == (6, 4)
    assert change.estimate == pytest.approx(1.0 / 3.0)


def test_paired_geomean_withholds_interval_and_p_below_the_floor() -> None:
    change = summary.paired_geomean([0.1] * (summary.MIN_PAIRS_FOR_INTERVAL - 2) + [0.3])
    assert change.method == "underpowered"
    assert math.isnan(change.low) and math.isnan(change.pvalue)


def test_paired_geomean_without_spread_reports_no_test() -> None:
    """Every kernel changing by exactly the same ratio has no t statistic; a p of 0 or 1 there would
    enter a correction as a test that never ran."""
    change = summary.paired_geomean([0.2] * 8)
    assert change.method == "degenerate"
    assert change.estimate == pytest.approx(0.2) and math.isnan(change.pvalue)
    assert math.isnan(change.low) and math.isnan(change.high)


def test_paired_geomean_over_no_pairs_has_no_estimate() -> None:
    """An empty leg is not a 1x ratio: exp(0) would plot as no change."""
    change = summary.paired_geomean([])
    assert (change.n, change.method) == (0, "degenerate")
    assert math.isnan(change.estimate)


def test_a_timing_comparison_and_a_paired_change_report_one_signed_rank_p() -> None:
    """Two call sites into scipy chose exact-or-approximate independently and disagreed on tied data;
    both now read the one test, so the same differences give the same p wherever they are tested."""
    before = np.array([10.0, 12.0, 9.0, 14.0, 11.0, 13.0, 10.5, 12.5, 9.5, 15.0])
    after = np.array([9.0, 11.0, 9.5, 12.0, 10.0, 12.0, 9.5, 12.0, 9.0, 13.0])
    expected = summary.signed_rank_test(after - before)[1]
    assert inference.compare(after, before, paired=True).pvalue == expected
    assert summary.paired_change(after - before).pvalue == expected


def test_a_tied_sample_takes_the_corrected_approximation_not_an_exact_count() -> None:
    """The exact null counts subsets of DISTINCT ranks, so on tied differences it is wrong rather than
    precise. scipy's automatic choice counted exactly there: llr40v11-oss120b-c read p = 0.38052
    instead of 0.38708."""
    differences = [0.1, 0.1, 0.2, -0.3, 0.4, 0.4, 0.5, -0.1, 0.6, 0.7]
    _, pvalue, method, n = summary.signed_rank_test(differences)
    assert (method, n) == ("signed-rank-approx", 10)
    assert pvalue == pytest.approx(float(wilcoxon(differences, method="approx", correction=True).pvalue))


def test_a_cell_below_the_interval_floor_reports_its_median_without_an_interval() -> None:
    median, low, high, dropped = summary.median_ci([1.0, 2.0, 4.0], drop=False, min_n=summary.MIN_INTERVAL_SAMPLES)
    assert (median, dropped) == (2.0, 0)
    assert math.isnan(low) and math.isnan(high)


def test_a_bootstrap_over_log_ratios_keeps_the_negative_values() -> None:
    """A log speedup below zero is a slow-down, not a broken timer reading; cleaning it away would move
    every interval of a regressing arm toward zero."""
    logs = [-1.0, -0.8, -0.6, -0.5, -0.4, -0.2, 0.1, 0.3]
    interval = summary.bootstrap_ci(logs, np.median, "median", n_resamples=999, method="percentile")
    assert interval.n == len(logs)
    assert interval.point == pytest.approx(-0.45)
    assert interval.low < interval.point < interval.high


def test_a_rank_sum_over_two_identical_samples_finds_no_difference() -> None:
    assert summary.rank_sum_test([3.0, 3.0, 3.0], [3.0, 3.0, 3.0])[1] == pytest.approx(1.0)
