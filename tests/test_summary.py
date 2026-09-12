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
import pandas as pd
import pytest

from hpcagent_bench.stats import summary


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


def test_signed_changes_matches_the_scalar_everywhere() -> None:
    ratios = [0.25, 0.5, 1.0, 2.0, 4.0]
    assert summary.signed_changes(ratios).tolist() == [summary.signed_change(r) for r in ratios]


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
    assert summary.geomean([2.0, 0.0, -1.0, math.nan, 8.0], unusable="drop") == pytest.approx(4.0)
    assert math.isnan(summary.geomean([0.0, -1.0], unusable="drop"))
    assert math.isnan(summary.geomean([], unusable="drop"))


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


def test_the_interval_names_what_it_is_for() -> None:
    assert summary.geomean_ci([2.0, 4.0]).label() == "95% log-t CI for geomean"


def test_median_per_kernel_weights_every_kernel_once() -> None:
    """An agent that resubmits a kernel ten times must not weight it ten times."""
    frame = pd.DataFrame(
        {"benchmark": ["a"] * 5 + ["b"], "speedup": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]},
    )
    per_kernel = summary.median_per_kernel(frame, "speedup")
    assert per_kernel.to_dict() == {"a": 3.0, "b": 10.0}


def test_median_per_kernel_reduces_within_an_episode_first() -> None:
    """A token count is a per-EPISODE total, so the episode is a max over its rows before the
    kernel is a median over its episodes -- otherwise a chatty agent votes once per judge call."""
    frame = pd.DataFrame(
        {
            "benchmark": ["a", "a", "a", "a"],
            "run_id": ["r1", "r1", "r2", "r2"],
            "tokens": [10.0, 40.0, 60.0, 20.0],
        }
    )
    assert summary.median_per_kernel(frame, "tokens", within=("run_id",)).to_dict() == {"a": 50.0}


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
