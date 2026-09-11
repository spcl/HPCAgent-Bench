# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Inferential-statistics audit: the properties a number has to have before it is a claim.

Each test here states ONE property that the campaign tables and figures rely on and that an
audit found broken or unverified on real campaign data. Where a test is red, the property is
the correct one and the code is what has to move -- the numbers in the tables were checked
against a second, independently written route before the property was written down.

The paired sets used below have the shape the real ones do: per-kernel log speed-up ratios,
right-tailed, a handful of kernels per arm pair (llr40's skill pairs are n = 2, 3 and 4), and a
1% geometric quantisation ladder that guarantees ties in |d|.
"""

import importlib.util
import math
import pathlib
from types import ModuleType

import numpy as np
import pytest

from hpcagent_bench.harness import efficacy, metric
from hpcagent_bench.stats import signed_rank, summary

#: The real paired set the published C-vs-Fortran claim rests on: ``log(c_best_su / fortran_best_su)``
#: for every kernel in ``reproducibility/llr40/analysis/per_language_kernel.csv`` that both languages
#: reached. n = 39, four exact ties (the 1% geometric ladder collides), skew +0.54, excess kurtosis
#: +3.1. A synthetic Gaussian fixture would test a distribution this analysis never sees.
LLR40_C_OVER_FORTRAN_LOG_DELTAS: tuple[float, ...] = (
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

#: The same population shifted so its MEAN is exactly zero -- the null the percentile bootstrap of a
#: mean-of-logs claims to cover 95% of the time.
ZERO_MEAN_DELTAS: np.ndarray = np.asarray(LLR40_C_OVER_FORTRAN_LOG_DELTAS, dtype=float)
ZERO_MEAN_DELTAS = ZERO_MEAN_DELTAS - ZERO_MEAN_DELTAS.mean()


def bootstrap_false_positive_rate(population: np.ndarray, n: int, trials: int, seed: int) -> float:
    """Share of samples of size ``n`` on which ``efficacy.Ratio.significant`` would fire although
    the population mean is exactly zero."""
    rng = np.random.default_rng(seed)
    misses = 0
    for trial in range(trials):
        deltas = list(rng.choice(population, size=n, replace=True))
        low, high = efficacy.bootstrap_interval(deltas, resamples=499, seed=trial)
        if not low <= 0.0 <= high:
            misses += 1
    return misses / trials


def paired_change_false_positive_rate(population: np.ndarray, n: int, trials: int, seed: int) -> float:
    """The same for :func:`summary.paired_change`, whose interval inverts the signed-rank test."""
    rng = np.random.default_rng(seed)
    misses = 0
    for _ in range(trials):
        change = summary.paired_change(rng.choice(population, size=n, replace=True))
        if math.isfinite(change.low) and not change.low <= 0.0 <= change.high:
            misses += 1
    return misses / trials


@pytest.mark.parametrize(
    "n_pairs, max_false_positive_rate",
    [
        pytest.param(4, 0.08, id="n=4 -- the llr40 oss120b/qwen38 skill pairs"),
        pytest.param(10, 0.08, id="n=10"),
        pytest.param(39, 0.08, id="n=39 -- the focus40 roster"),
    ],
)
def test_the_efficacy_significance_flag_holds_its_nominal_level_on_skewed_paired_deltas(
    n_pairs: int, max_false_positive_rate: float
) -> None:
    """``score_significant`` / ``cost_significant`` are written into the shipped efficacy CSV as
    hard booleans, so a flag that fires far more often than 5% under a true null turns an absent
    effect into a published finding."""
    rate = bootstrap_false_positive_rate(ZERO_MEAN_DELTAS, n_pairs, trials=1500, seed=20260911)
    assert rate <= max_false_positive_rate, (
        f"efficacy.bootstrap_interval missed a zero-mean population on {rate:.1%} of samples at "
        f"n={n_pairs}; the 95% interval promises at most 5%"
    )


@pytest.mark.parametrize(
    "n_pairs, max_false_positive_rate",
    [
        pytest.param(6, 0.08, id="n=6 -- MIN_PAIRS_FOR_INTERVAL"),
        pytest.param(20, 0.08, id="n=20"),
        pytest.param(39, 0.08, id="n=39 -- the focus40 roster"),
    ],
)
def test_the_hodges_lehmann_interval_holds_its_nominal_level_on_skewed_paired_deltas(
    n_pairs: int, max_false_positive_rate: float
) -> None:
    """The rank interval is the one the figures draw and the one the signed-rank p inverts; if it
    drifted off its level the whole paired half of the analysis would move with it."""
    population = ZERO_MEAN_DELTAS - summary.paired_change(ZERO_MEAN_DELTAS).estimate
    rate = paired_change_false_positive_rate(population, n_pairs, trials=1500, seed=20260911)
    assert rate <= max_false_positive_rate, (
        f"summary.paired_change missed its own pseudo-median on {rate:.1%} of samples at n={n_pairs}"
    )


def test_the_reported_effect_and_the_p_value_describe_the_same_parameter() -> None:
    """One pairs-table row carries ``rho_score`` (a ratio of geometric means) beside a ``p_value``
    that inverts the Hodges-Lehmann pseudo-median, and on a skewed set the two point opposite ways
    -- a reader then gets an effect and a test that disagree about which arm is ahead."""
    deltas = ZERO_MEAN_DELTAS + 0.02
    ratio_of_geomeans = math.exp(float(np.mean(deltas)))
    hodges_lehmann = math.exp(summary.paired_change(deltas).estimate)
    assert (ratio_of_geomeans - 1.0) * (hodges_lehmann - 1.0) > 0.0, (
        f"exp(mean log) = {ratio_of_geomeans:.4f} and Hodges-Lehmann = {hodges_lehmann:.4f} "
        "straddle 1.0, so the row's effect column and its p-value disagree in direction"
    )


@pytest.mark.parametrize(
    "n, w_plus, exact, approximate",
    [
        pytest.param(35, 219.0, 0.118674, 0.117769, id="n=35 -- the llr40 C-vs-Fortran pairing"),
        pytest.param(40, 293.0, 0.118149, 0.117369, id="n=40 -- the focus40 roster"),
        pytest.param(97, 1943.0, 0.119557, 0.119225, id="n=97 -- the pooled model/kernel pairing"),
        pytest.param(210, 9705.0, 0.119812, 0.119658, id="n=210 -- above EXACT_MAX_N"),
    ],
)
def test_the_normal_signed_rank_approximation_never_reports_a_smaller_p_than_the_exact_null(
    n: int, w_plus: float, exact: float, approximate: float
) -> None:
    """llr40 speed-ups sit on a 1% geometric ladder, so |d| ties are the rule and ``use_exact``
    sends these tables down the approximate branch; an approximation that is uniformly below the
    exact p manufactures significance on exactly the data the campaign reports."""
    got_exact = signed_rank.exact_p(w_plus, n)
    got_approx = signed_rank.normal_p(w_plus, n, [float(i) for i in range(n)])
    assert got_exact == pytest.approx(exact, abs=5e-6), got_exact
    assert got_approx == pytest.approx(approximate, abs=5e-6), got_approx
    assert got_approx >= got_exact, (
        f"the normal approximation reports p={got_approx:.6f} against an exact {got_exact:.6f} at "
        f"n={n}: {got_exact - got_approx:.2e} on the anti-conservative side"
    )


@pytest.mark.parametrize(
    "values, description",
    [
        pytest.param([], "no scored kernel at all", id="empty"),
        pytest.param([0.0], "one unscored cell", id="single-zero"),
        pytest.param([0.0, 0.0], "every cell unscored", id="all-zero"),
    ],
)
def test_an_absent_measurement_reads_the_same_way_at_every_geomean_call_site(
    values: list[float], description: str
) -> None:
    """A missing speed-up is neutral in ``harness.metric`` and a total collapse in the CLI summary
    the same run prints, so one absence is reported as two different results depending on which
    line of the harness the reader is looking at."""
    grading = metric.geomean(values)
    cli_style = metric.geomean(values) if values else 0.0  # hpcagent_bench/cli.py:308
    assert grading == cli_style, (
        f"{description}: the grading path scores {grading} and the CLI summary prints {cli_style} for the same absence"
    )


def test_a_paired_comparison_reports_how_many_units_it_dropped() -> None:
    """The complement waves re-run only the kernels with no judge row, so the two arms of a pairing
    cover different kernel sets; an intersection that names only what it KEPT lets a claim about
    forty kernels be made on two, with nothing in the record saying so."""
    before = {"a": 2.0, "b": 3.0, "c": 4.0, "d": 5.0}
    after = {"a": 2.2, "b": 3.3}
    costs_before = {k: 1000.0 for k in before}
    costs_after = {k: 1000.0 for k in after}
    item = efficacy.efficacy(before, after, costs_before, costs_after)
    dropped = (set(before) | set(after)) - set(item.tasks)
    assert hasattr(item, "unmatched"), (
        f"efficacy() paired {len(item.tasks)} of {len(set(before) | set(after))} tasks and dropped "
        f"{sorted(dropped)} without recording them anywhere in the result"
    )


ARTIFACT = pathlib.Path(__file__).resolve().parents[1] / "reproducibility" / "llr40"


def load_analyze_llr40() -> ModuleType:
    """``analyze_llr40.py`` is a script beside its artifact, not an installed module."""
    spec = importlib.util.spec_from_file_location("analyze_llr40", ARTIFACT / "analyze_llr40.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "arm, published_geomean",
    [
        pytest.param("llr40v10-qwen38-c", 15.269, id="llr40v10-qwen38-c -- rank 2 of 21"),
        pytest.param("llr40v10-qwen38-fortran", 11.629, id="llr40v10-qwen38-fortran"),
        pytest.param("llr40v10-kimi27sglang-c", 10.659, id="llr40v10-kimi27sglang-c"),
        pytest.param("llr40v10-oss120b-c", 9.313, id="llr40v10-oss120b-c"),
        pytest.param("llr40v9-oss120b-fortran", 4.853, id="llr40v9-oss120b-fortran -- single episode"),
    ],
)
def test_the_shipped_llr40_arm_table_reproduces_from_the_shipped_observations(
    arm: str, published_geomean: float
) -> None:
    """The tables and the figure in ``reproducibility/llr40/analysis`` are the artifact a reader
    checks the campaign against; a table built by a reduction the script no longer performs ranks
    the arms by how often each agent resubmitted rather than by what it produced."""
    module = load_analyze_llr40()
    observations = module.load_observations(ARTIFACT)
    best = module.best_per_arm_kernel(module.submissions_with_sources(ARTIFACT, observations))
    recomputed = module.geomean(best[best.arm == arm].best_speedup)
    assert recomputed == pytest.approx(published_geomean, rel=1e-3), (
        f"{arm}: the shipped per_arm_summary.csv says {published_geomean}x, the current reduction "
        f"gives {recomputed:.3f}x"
    )
