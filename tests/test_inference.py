# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for hpcagent_bench.inference: normality verdicts, verdict-selected confidence intervals,
two-sample significance / equivalence, and multiple-comparison correction.

The tests are written as PROPERTIES over generated samples of KNOWN distribution: a check that
accepts a normal sample is worthless unless it also rejects a lognormal one, and a test that
detects a known shift is worthless unless it stays quiet on two identical distributions. Both
directions are asserted for every decision this module makes, because the false-POSITIVE side is
the one such code usually leaves untested.
"""
import math
import warnings
from typing import Dict, Tuple

import numpy as np
import pytest
from scipy.stats import anderson as scipy_anderson

from hpcagent_bench import inference

#: Distinct seeds per property so a lucky draw cannot carry a passing assertion between tests.
SEEDS = (0, 1, 2, 3, 4)

#: A realistic repeat count: measurement.repeat defaults to 50 (harness/timing.py:105).
REPEAT = 50


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def normal_sample(seed: int, n: int = 200, loc: float = 100.0, scale: float = 5.0) -> np.ndarray:
    """Positive-mean normal: inference.clean drops non-positive values (wall-clock is positive),
    so a mean-zero normal would be silently truncated into a half-normal."""
    return rng(seed).normal(loc, scale, n)


def lognormal_sample(seed: int, n: int = 200, sigma: float = 0.5) -> np.ndarray:
    return 100.0 * rng(seed).lognormal(0.0, sigma, n)


def exponential_sample(seed: int, n: int = 200) -> np.ndarray:
    """Shifted exponential -- the classic "bounded below, heavy right tail" timing shape."""
    return 100.0 + rng(seed).exponential(20.0, n)


# --------------------------------------------------------------------------------------------
# Normality
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_normality_accepts_known_normal(seed: int) -> None:
    verdict = inference.check_normality(normal_sample(seed))
    assert verdict.normal, verdict.reason
    assert verdict.test == "shapiro"
    assert abs(verdict.skew) < 0.5
    assert verdict.qq_departure < 0.05  # a near-straight QQ line


@pytest.mark.parametrize("seed", SEEDS)
def test_normality_rejects_known_lognormal(seed: int) -> None:
    verdict = inference.check_normality(lognormal_sample(seed))
    assert not verdict.normal, verdict.reason
    assert verdict.rejected and not verdict.negligible
    assert verdict.skew > 0.5  # right-skewed, as the veto bound requires to reject


@pytest.mark.parametrize("seed", SEEDS)
def test_normality_rejects_known_exponential(seed: int) -> None:
    verdict = inference.check_normality(exponential_sample(seed))
    assert not verdict.normal, verdict.reason
    assert verdict.excess_kurtosis > inference.MAX_ABS_EXCESS_KURTOSIS


def test_normality_switches_to_anderson_darling_above_shapiro_limit() -> None:
    """n > 5000 must leave the Shapiro branch, where scipy documents the p-value as unreliable."""
    small = inference.check_normality(normal_sample(0, n=inference.SHAPIRO_MAX_N))
    large = inference.check_normality(normal_sample(0, n=inference.SHAPIRO_MAX_N + 1))
    assert small.test == "shapiro"
    assert large.test == "anderson-darling"
    assert large.normal, large.reason  # still a normal sample; only the test changed


def test_anderson_darling_statistic_matches_scipy() -> None:
    """Cross-check the hand-rolled A^2 against scipy's, which is the reference implementation.

    scipy's ``anderson`` is only used HERE as an oracle: the module computes A^2 itself because
    scipy's return schema is mid-migration (1.17 warns it will change in 1.19), which is exactly
    the kind of churn a benchmark's published numbers must not ride on."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # the migration warning; the value is unaffected
        for n in (50, 500, 5000):
            x = normal_sample(11, n=n)
            reference = float(scipy_anderson(x, dist="norm").statistic)
            assert inference.anderson_darling_statistic(x) == pytest.approx(reference, rel=1e-9, abs=1e-12)


def test_anderson_darling_pvalue_is_monotone_and_finite() -> None:
    """The published piecewise fit turns back up outside its range and overflows exp(); the
    implementation caps it. Assert both the monotonicity and the absence of an overflow."""
    values = [inference.anderson_darling_pvalue(a2, 1000) for a2 in (0.1, 0.25, 0.4, 0.8, 3.0, 50.0, 1e6)]
    assert all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in values)
    assert values == sorted(values, reverse=True)


def test_normality_rejection_is_vetoed_when_the_deviation_is_trivial() -> None:
    """The whole point of carrying an effect measure: at huge n a test rejects on a deviation
    far too small to change any downstream number, and a p-value alone must not decide."""
    x = rng(5).normal(100.0, 5.0, 60000)
    x[:400] += 1.0  # a real but microscopic contamination
    verdict = inference.check_normality(x)
    assert verdict.rejected or verdict.negligible  # either it did not reject, or the veto applies
    if verdict.rejected:
        assert verdict.negligible and verdict.normal, verdict.reason


def test_normality_refuses_to_guess_on_a_tiny_sample() -> None:
    """Too few samples must route to the non-parametric branch, not to a lucky "normal"."""
    verdict = inference.check_normality([1.0, 2.0, 3.0])
    assert not verdict.normal
    assert verdict.test == "insufficient"


def test_normality_on_a_zero_spread_sample_is_not_normal() -> None:
    """A timer-resolution floor is a degenerate sample, never a Gaussian."""
    verdict = inference.check_normality([7.0] * 40)
    assert not verdict.normal
    assert verdict.test == "insufficient"


def test_clean_drops_nonpositive_and_nonfinite() -> None:
    kept = inference.clean([1.0, -2.0, 0.0, float("nan"), float("inf"), 3.0])
    assert kept.tolist() == [1.0, 3.0]


# --------------------------------------------------------------------------------------------
# Confidence intervals
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_interval_for_picks_parametric_on_normal_and_bootstrap_otherwise(seed: int) -> None:
    parametric, normal_verdict = inference.interval_for(normal_sample(seed))
    nonparam, skewed_verdict = inference.interval_for(lognormal_sample(seed))
    assert normal_verdict.normal and parametric.method == "t" and parametric.statistic == "mean"
    assert not skewed_verdict.normal
    assert nonparam.method.startswith("bootstrap") and nonparam.statistic == "median"


def test_interval_label_names_both_the_statistic_and_the_method() -> None:
    """A figure caption must not leave "parametric or bootstrap?" to be inferred."""
    interval, _ = inference.interval_for(normal_sample(0))
    assert "95%" in interval.label() and "t" in interval.label() and "mean" in interval.label()


@pytest.mark.parametrize("seed", SEEDS)
def test_t_interval_covers_the_true_mean(seed: int) -> None:
    interval = inference.mean_ci_t(normal_sample(seed, n=400, loc=100.0, scale=5.0))
    assert interval.low < 100.0 < interval.high
    assert interval.low < interval.point < interval.high


def test_t_interval_coverage_rate_is_near_nominal() -> None:
    """The property that makes it a 95% interval: over many independent samples it must contain
    the truth about 95% of the time. A too-narrow interval is the failure this catches."""
    generator = rng(99)
    trials = 400
    hits = 0
    for _ in range(trials):
        interval = inference.mean_ci_t(generator.normal(100.0, 5.0, REPEAT))
        hits += int(interval.low < 100.0 < interval.high)
    assert 0.90 <= hits / trials <= 0.99, f"coverage {hits / trials:.3f} is not ~0.95"


@pytest.mark.parametrize("seed", SEEDS)
def test_bootstrap_median_interval_covers_the_true_median(seed: int) -> None:
    """Lognormal(0, 0.5) scaled by 100 has median exactly 100."""
    interval = inference.bootstrap_ci(lognormal_sample(seed, n=400), np.median, "median", n_resamples=2000)
    assert interval.low < 100.0 < interval.high
    assert interval.method in ("bootstrap-BCa", "bootstrap-percentile")


@pytest.mark.parametrize("seed", SEEDS)
def test_rank_median_interval_covers_the_true_median(seed: int) -> None:
    """The distribution-free cross-check: no bootstrap, no shape assumption."""
    interval = inference.median_rank_ci(lognormal_sample(seed, n=400))
    assert interval.low < 100.0 < interval.high
    assert interval.method == "rank-median"


def test_bootstrap_interval_is_reproducible_from_its_seed() -> None:
    """A published figure must redraw identically from the same DB."""
    x = lognormal_sample(0)
    first = inference.bootstrap_ci(x, np.median, "median", n_resamples=2000, seed=7)
    second = inference.bootstrap_ci(x, np.median, "median", n_resamples=2000, seed=7)
    assert (first.low, first.high) == (second.low, second.high)


def test_min_of_k_interval_labels_the_inconsistent_case() -> None:
    """k == n is the n-out-of-n bootstrap of an extreme order statistic, which is inconsistent.
    It may be reported, but it must never be labelled as if it were a consistent CI."""
    x = lognormal_sample(0, n=200)
    consistent = inference.min_of_k_ci(x, k=10, n_resamples=2000)
    degenerate = inference.min_of_k_ci(x, k=200, n_resamples=2000)
    assert consistent.method == "bootstrap-min-of-k"
    assert "inconsistent" in degenerate.method
    assert consistent.statistic == "min_of_10" and degenerate.statistic == "min_of_200"


def test_min_of_k_band_widens_as_k_shrinks() -> None:
    """Fewer repeats -> a worse expected minimum -> a wider, higher band. If this inverted, the
    min-of-k interval would be describing something other than the minimum."""
    x = lognormal_sample(1, n=400)
    narrow = inference.min_of_k_ci(x, k=50, n_resamples=4000)
    wide = inference.min_of_k_ci(x, k=3, n_resamples=4000)
    assert (wide.high - wide.low) > (narrow.high - narrow.low)


# --------------------------------------------------------------------------------------------
# Ratio (speed-up) intervals
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_speedup_interval_covers_a_known_ratio(seed: int) -> None:
    """Baseline median 200, candidate median 100 -> a true speed-up of exactly 2."""
    generator = rng(seed)
    baseline = 200.0 * generator.lognormal(0.0, 0.15, 120)
    candidate = 100.0 * generator.lognormal(0.0, 0.15, 120)
    interval = inference.speedup_ci(baseline, candidate, n_resamples=3000)
    assert interval.low < 2.0 < interval.high
    assert interval.statistic == "speedup(median)"


def test_speedup_interval_is_not_the_ratio_of_two_separate_intervals() -> None:
    """The bug this function exists to prevent: dividing endpoint by endpoint. That naive
    construction is strictly wider than the bootstrapped ratio, so the two must differ."""
    generator = rng(21)
    baseline = 200.0 * generator.lognormal(0.0, 0.25, 120)
    candidate = 100.0 * generator.lognormal(0.0, 0.25, 120)
    ratio = inference.speedup_ci(baseline, candidate, n_resamples=3000)
    b_ci = inference.bootstrap_ci(baseline, np.median, "median", n_resamples=3000)
    c_ci = inference.bootstrap_ci(candidate, np.median, "median", n_resamples=3000)
    naive_low, naive_high = b_ci.low / c_ci.high, b_ci.high / c_ci.low
    assert naive_low < ratio.low and ratio.high < naive_high


def test_speedup_interval_is_inversion_consistent() -> None:
    """Intervaling base/cand and flipping it must reproduce intervaling cand/base.

    Agreement is up to MONTE-CARLO error, not exact: each call draws its own replicates. The
    tolerance is still ~100x tighter than the asymmetry a ``point +/- z*se`` interval would show
    on a skewed ratio, which is the construction this asserts we are NOT using."""
    generator = rng(31)
    baseline = 200.0 * generator.lognormal(0.0, 0.2, 100)
    candidate = 100.0 * generator.lognormal(0.0, 0.2, 100)
    forward = inference.speedup_ci(baseline, candidate, n_resamples=20000, seed=5)
    reverse = inference.speedup_ci(candidate, baseline, n_resamples=20000, seed=5)
    assert 1.0 / reverse.high == pytest.approx(forward.low, rel=1e-3)
    assert 1.0 / reverse.low == pytest.approx(forward.high, rel=1e-3)


def test_speedup_interval_is_not_symmetric_about_the_point_estimate() -> None:
    """A skewed ratio must produce an asymmetric band. Equal arms would mean someone had
    reverted this to a normal approximation."""
    generator = rng(32)
    baseline = 200.0 * generator.lognormal(0.0, 0.45, 60)
    candidate = 100.0 * generator.lognormal(0.0, 0.45, 60)
    interval = inference.speedup_ci(baseline, candidate, n_resamples=20000)
    lower_arm, upper_arm = interval.point - interval.low, interval.high - interval.point
    assert abs(upper_arm - lower_arm) / max(lower_arm, upper_arm) > 0.02


@pytest.mark.parametrize("seed", SEEDS)
def test_fieller_interval_covers_a_known_ratio_of_means(seed: int) -> None:
    generator = rng(seed)
    interval = inference.fieller_ratio_ci(generator.normal(200.0, 10.0, 120), generator.normal(100.0, 5.0, 120))
    assert interval.low < 2.0 < interval.high
    assert interval.method == "fieller"


def test_fieller_reports_an_unbounded_interval_when_the_denominator_straddles_zero() -> None:
    """Fieller's honest failure mode: the delta method would return a finite, wrong interval.

    The denominator must stay POSITIVE (inference.clean drops non-positive timings), so the
    unbounded case is reached the way it is actually reachable for wall-clock: a tiny, wildly
    dispersed sample whose mean is not separated from zero by its own standard error."""
    generator = rng(41)
    volatile = np.exp(generator.normal(0.0, 2.5, 5))  # CV > 1 on 5 samples
    interval = inference.fieller_ratio_ci(generator.normal(200.0, 10.0, 30), volatile)
    assert interval.low == -math.inf and interval.high == math.inf


def test_fieller_is_bounded_on_a_well_conditioned_denominator() -> None:
    """The complement of the test above: the unbounded branch must not fire on ordinary data."""
    generator = rng(42)
    interval = inference.fieller_ratio_ci(generator.normal(200.0, 10.0, 60), generator.normal(100.0, 5.0, 60))
    assert math.isfinite(interval.low) and math.isfinite(interval.high)


# --------------------------------------------------------------------------------------------
# Significance between two systems
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_mann_whitney_detects_a_known_shift(seed: int) -> None:
    generator = rng(seed)
    candidate = 100.0 * generator.lognormal(0.0, 0.15, REPEAT)
    baseline = 130.0 * generator.lognormal(0.0, 0.15, REPEAT)
    result = inference.compare(candidate, baseline)
    assert result.significant and result.pvalue < 0.05
    assert result.effect < 0.0  # candidate stochastically smaller == faster
    assert result.ratio > 1.0  # median(baseline) / median(candidate)


@pytest.mark.parametrize("seed", SEEDS)
def test_mann_whitney_does_not_fire_on_two_identical_distributions(seed: int) -> None:
    """The false-positive check most such code lacks."""
    generator = rng(seed)
    a = 100.0 * generator.lognormal(0.0, 0.15, REPEAT)
    b = 100.0 * generator.lognormal(0.0, 0.15, REPEAT)
    result = inference.compare(a, b)
    assert not result.significant, f"false positive at p={result.pvalue}"
    assert abs(result.effect) < 0.4


def test_mann_whitney_false_positive_rate_is_near_alpha() -> None:
    """Over many all-null pairs the test must fire about alpha of the time -- not more."""
    generator = rng(1234)
    fired = 0
    trials = 300
    for _ in range(trials):
        a = 100.0 * generator.lognormal(0.0, 0.15, REPEAT)
        b = 100.0 * generator.lognormal(0.0, 0.15, REPEAT)
        fired += int(inference.compare(a, b).significant)
    assert fired / trials <= 0.10, f"false-positive rate {fired / trials:.3f} exceeds alpha by too much"


def test_effect_size_is_reported_even_when_a_huge_n_certifies_a_trivial_difference() -> None:
    """A p-value on 10000 repetitions can certify a 0.1% difference. The effect size must show
    it is trivial; that is the whole reason it travels beside the p-value."""
    generator = rng(77)
    a = 100.0 * generator.lognormal(0.0, 0.15, 10000)
    b = 100.1 * generator.lognormal(0.0, 0.15, 10000)
    result = inference.compare(a, b)
    assert abs(result.effect) < 0.1  # negligible by Cliff's delta convention
    assert result.ratio == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("seed", SEEDS)
def test_wilcoxon_detects_a_known_paired_shift(seed: int) -> None:
    """Paired collection is not what the harness does today (see the module docstring), but the
    test must be correct for the interleaved collector it is reserved for."""
    generator = rng(seed)
    baseline = 100.0 * generator.lognormal(0.0, 0.15, REPEAT)
    candidate = baseline * 0.8  # same rep, same machine, a real per-pair improvement
    result = inference.compare(candidate, baseline, paired=True)
    assert result.test == "wilcoxon-signed-rank"
    assert result.significant and result.ratio > 1.0


def test_wilcoxon_does_not_fire_on_independent_draws_from_one_distribution() -> None:
    generator = rng(303)
    a = 100.0 * generator.lognormal(0.0, 0.15, REPEAT)
    b = 100.0 * generator.lognormal(0.0, 0.15, REPEAT)
    assert not inference.compare(a, b, paired=True).significant


def test_wilcoxon_refuses_unequal_lengths() -> None:
    """Pairing needs a pairing; silently truncating would invent correspondence."""
    with pytest.raises(ValueError, match="equal-length"):
        inference.compare([1.0, 2.0, 3.0], [1.0, 2.0], paired=True)


# --------------------------------------------------------------------------------------------
# Equivalence (TOST)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_tost_declares_equivalence_for_two_identical_distributions(seed: int) -> None:
    """"This optimisation changed nothing measurable" as a POSITIVE claim."""
    generator = rng(seed)
    a = 100.0 * generator.lognormal(0.0, 0.05, 200)
    b = 100.0 * generator.lognormal(0.0, 0.05, 200)
    result = inference.tost_equivalence(a, b, margin=0.10)
    assert result.equivalent and result.pvalue < 0.05


@pytest.mark.parametrize("seed", SEEDS)
def test_tost_refuses_equivalence_for_a_real_difference(seed: int) -> None:
    generator = rng(seed)
    a = 100.0 * generator.lognormal(0.0, 0.05, 200)
    b = 140.0 * generator.lognormal(0.0, 0.05, 200)
    assert not inference.tost_equivalence(a, b, margin=0.10).equivalent


def test_tost_is_not_the_same_as_failing_to_reject() -> None:
    """The distinction the reviewer asked for: an underpowered sample fails to reject a
    difference AND fails to establish equivalence. Only TOST separates "no effect" from
    "no power"."""
    generator = rng(88)
    a = 100.0 * generator.lognormal(0.0, 0.30, 8)
    b = 100.0 * generator.lognormal(0.0, 0.30, 8)
    assert not inference.compare(a, b).significant  # no evidence of a difference ...
    assert not inference.tost_equivalence(a, b, margin=0.02).equivalent  # ... and none of sameness


def test_tost_rejects_a_nonsensical_margin() -> None:
    with pytest.raises(ValueError, match="margin"):
        inference.tost_equivalence([1.0, 2.0], [1.0, 2.0], margin=0.0)


# --------------------------------------------------------------------------------------------
# Multiple comparisons
# --------------------------------------------------------------------------------------------


def all_null_corpus(seed: int, kernels: int = 300, n: int = REPEAT) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """A corpus where EVERY kernel's two samples come from the same distribution: every
    rejection is by construction a false positive."""
    generator = rng(seed)
    cells: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for i in range(kernels):
        cells[f"kernel{i:03d}"] = (100.0 * generator.lognormal(0.0, 0.15, n), 100.0 * generator.lognormal(0.0, 0.15, n))
    return cells


def test_fdr_correction_reduces_false_positives_on_an_all_null_corpus() -> None:
    """~578 kernels at alpha=0.05 manufacture ~29 false positives by construction. The whole
    point of the correction is that this number collapses."""
    cells = all_null_corpus(2026, kernels=300)
    rows = inference.compare_corpus(cells)
    raw_hits = sum(int(r.comparison.significant) for r in rows)
    adjusted_hits = sum(int(r.significant_adjusted) for r in rows)
    assert raw_hits >= 5, "the simulated corpus should produce uncorrected false positives to correct"
    assert adjusted_hits < raw_hits
    assert adjusted_hits <= 1


def test_holm_is_at_least_as_conservative_as_fdr() -> None:
    """FWER control cannot claim more discoveries than FDR control on the same p-values."""
    cells = all_null_corpus(4242, kernels=200)
    fdr = sum(int(r.significant_adjusted) for r in inference.compare_corpus(cells, method="fdr_bh"))
    holm = sum(int(r.significant_adjusted) for r in inference.compare_corpus(cells, method="holm"))
    assert holm <= fdr


def test_correction_still_finds_a_real_effect_planted_in_a_null_corpus() -> None:
    """A correction that killed every discovery would be useless. A genuine 30% win must survive."""
    cells = all_null_corpus(7, kernels=200)
    generator = rng(555)
    cells["planted"] = (100.0 * generator.lognormal(0.0, 0.15, REPEAT), 130.0 * generator.lognormal(0.0, 0.15, REPEAT))
    rows = {r.key: r for r in inference.compare_corpus(cells)}
    assert rows["planted"].significant_adjusted
    assert rows["planted"].pvalue_adjusted < 0.05


def test_adjusted_pvalues_never_shrink_below_the_raw_ones() -> None:
    raw = [0.001, 0.01, 0.03, 0.2, 0.5, 0.9]
    for method in ("fdr_bh", "holm"):
        adjusted = inference.adjust_pvalues(raw, method=method)
        assert all(a >= p - 1e-12 for a, p in zip(adjusted, raw)), method
        assert all(0.0 <= a <= 1.0 for a in adjusted), method


def test_holm_matches_the_textbook_step_down_values() -> None:
    """Worked by hand: n=4, sorted p = .01 .02 .03 .04 -> .04 .06 .06 .06 after monotonicity."""
    assert inference.adjust_pvalues([0.01, 0.02, 0.03, 0.04], method="holm") == pytest.approx([0.04, 0.06, 0.06, 0.06])


def test_corpus_comparison_preserves_input_order() -> None:
    """Order reaches the report table, so it must come from the mapping, not from hashing."""
    cells = all_null_corpus(9, kernels=25)
    keys = list(cells.keys())
    assert [r.key for r in inference.compare_corpus(cells)] == keys


def test_adjust_pvalues_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown multiple-comparison method"):
        inference.adjust_pvalues([0.1, 0.2], method="bonferroni-ish")


def test_adjust_pvalues_handles_an_empty_corpus() -> None:
    assert inference.adjust_pvalues([], method="fdr_bh") == []
    assert inference.adjust_pvalues([], method="holm") == []


# --------------------------------------------------------------------------------------------
# The reduced statistic the harness credits
# --------------------------------------------------------------------------------------------


def test_min_of_k_of_a_normal_sample_is_not_normal() -> None:
    """⛔ The trap this module is built around: min-of-k is an EXTREME-VALUE statistic, so
    normality of the RAW repeats does not transfer to the reduced number the harness credits
    (reduce_min_of_k, harness/timing.py:144). Testing normality on the reduction answers a
    question nobody asked.

    Asserted at the harness's real repeat count (measurement.repeat defaults to 50,
    harness/timing.py:105), where the departure is unambiguous."""
    generator = rng(2027)
    raw = generator.normal(100.0, 5.0, 4000)
    assert inference.check_normality(raw).normal, "the raw draws ARE normal -- that is the premise"

    minima = np.array([float(np.min(generator.normal(100.0, 5.0, REPEAT))) for _ in range(4000)])
    verdict = inference.check_normality(minima)
    assert verdict.rejected  # the test sees it
    assert verdict.skew < 0.0  # left-skewed, as an extreme-value minimum must be
    assert not verdict.normal, verdict.reason  # and the departure is past the practical bounds


def test_min_of_k_departure_from_normality_grows_with_k() -> None:
    """The severity is GRADED, and the practical-normality veto tracks it: at k=5 the minimum of
    a normal sample is only mildly skewed and the veto (correctly) calls it close enough for a
    parametric interval, while at the harness's k=50 it is not. This is why the verdict carries
    an effect measure and not just a p-value -- both k reject at p < 1e-15."""
    generator = rng(2029)
    verdicts = {}
    for k in (5, REPEAT):
        minima = np.array([float(np.min(generator.normal(100.0, 5.0, k))) for _ in range(6000)])
        verdicts[k] = inference.check_normality(minima)
    assert verdicts[5].rejected and verdicts[REPEAT].rejected  # a p-value alone cannot tell them apart
    assert abs(verdicts[REPEAT].skew) > abs(verdicts[5].skew)  # the effect measure can
    assert verdicts[5].negligible and not verdicts[REPEAT].negligible


def test_min_of_k_distribution_shifts_with_k() -> None:
    """And it shifts with k, so a min-of-10 number and a min-of-50 number are not comparable."""
    generator = rng(2028)
    of_5 = np.mean([float(np.min(generator.normal(100.0, 5.0, 5))) for _ in range(2000)])
    of_50 = np.mean([float(np.min(generator.normal(100.0, 5.0, 50))) for _ in range(2000)])
    assert of_50 < of_5 - 1.0


def test_summarize_returns_the_verdict_the_interval_and_the_credited_statistic() -> None:
    summary = inference.summarize(lognormal_sample(0, n=120))
    assert summary["n"] == 120
    assert summary["normal"] is False
    assert summary["interval"].statistic == "median"
    assert "bootstrap" in summary["interval_label"]
    assert summary["min_of_k"].statistic == "min_of_120"
