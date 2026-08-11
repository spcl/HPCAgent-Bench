# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Statistical inference for timing claims: normality verdicts, confidence intervals chosen by
that verdict, and significance / equivalence tests between two systems.

Sibling of :mod:`hpcagent_bench.stats` (which owns robust outlier rejection and the median
bootstrap CI the heatmap already prints). This module answers the question a reviewer asks of a
speed-up table: *is this above measurement noise?*

WHAT THE HARNESS ACTUALLY MEASURES (the facts these choices rest on)
-------------------------------------------------------------------
* Raw per-repeat samples exist. ``timing.sampled_reps`` (harness/timing.py:124) runs
  ``warmup + repeat`` reps and returns the kept ns list; ``measurement.repeat`` defaults to 50
  (harness/timing.py:105).
* The credited number is a MIN-OF-K. ``reduce_min_of_k`` (harness/timing.py:144) is the default
  backend: ``speedup = min(baseline) / min(candidate)``. A minimum of k draws is an
  EXTREME-VALUE statistic -- it is not normal even when the underlying samples are, and its
  distribution shifts with k. So every normality question here is asked of the RAW repeats, and
  every interval names the statistic it is FOR.
* Candidate and baseline are NOT interleaved. In ``scoring.py`` the baselines are timed first
  (scoring.py:518-594), in their own child processes, and the candidate afterwards in a separate
  ``_call_isolated`` (scoring.py:625). On the framework track each framework is measured in its
  own process (``Framework.measure``, frameworks/framework.py:528); NumPy is run with
  ``repeat=1`` for validation only on a non-NumPy run (frameworks/test.py:170), and its timed
  rows come from a separate ``--framework numpy`` invocation. Rep *i* of one side therefore has
  no correspondence to rep *i* of the other: the samples are INDEPENDENT, and Mann-Whitney (not
  Wilcoxon signed-rank) is the correct test. :func:`wilcoxon_signed_rank` is provided for a
  future interleaved collector and is never reached from the current harness.
* Persistence is split. The framework ``results`` table keeps ONE ROW PER SAMPLE
  (``Result.time``, frameworks/schema.py:23; written per-sample at frameworks/test.py:275), so
  full inference is possible there. The agent-track ``submissions`` table keeps only the reduced
  ``native_ns`` / ``baseline_ns`` / ``speedup`` (recording.py:115-117) -- the raw repeats die
  with the scoring call, so agent-track cells can only be intervaled once they are persisted.
"""
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binom, bootstrap, false_discovery_control, kurtosis, mannwhitneyu, norm, shapiro
from scipy.stats import skew, t, wilcoxon

from hpcagent_bench.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES

#: Default two-sided error rate for every test and interval here.
DEFAULT_ALPHA: float = 0.05
#: The confidence level and bootstrap replicate count come from :mod:`hpcagent_bench.stats`, not
#: restated here. Both modules quote intervals of the same resolution BY CONSTRUCTION -- a second
#: copy of the number is a second thing to keep in step, and the comment that used to sit here said
#: exactly that while still holding a copy.

#: Every bootstrap seeds from here, so a published figure is reproducible from the same DB.
DEFAULT_SEED: int = 0

#: Below this the normality question has no power: any test accepts almost anything, so we
#: report "not normal" and route the caller to the non-parametric branch. Being wrong toward
#: non-parametric costs a slightly wider interval; being wrong toward normal produces the
#: misleading fitted-Gaussian plot this module exists to prevent.
MIN_NORMALITY_N: int = 8

#: scipy's Shapiro-Wilk p-value is documented as unreliable above ~5000 samples; past that we
#: switch to Anderson-Darling, which stays calibrated and is more sensitive in the tails.
SHAPIRO_MAX_N: int = 5000

#: Practical-normality bounds used to VETO a rejection. With large n, Shapiro-Wilk and
#: Anderson-Darling reject on deviations far too small to change any downstream number, so a
#: p-value alone must not decide. These two shape statistics are (unlike a QQ correlation)
#: n-independent under the alternative, which is what a veto needs.
MAX_ABS_SKEW: float = 0.5
MAX_ABS_EXCESS_KURTOSIS: float = 1.0


@dataclass(frozen=True, slots=True)
class NormalityVerdict:
    """Machine-readable normality verdict -- what downstream code branches on, never a printed
    sentence. ``normal`` is the single bit :func:`interval_for` and the plotting layer consume.

    ``qq_departure`` is ``1 - r**2`` of the normal probability plot (Filliben correlation): 0 is
    a perfect straight QQ line. It is REPORTED, not thresholded -- under true normality it
    shrinks with n (~0.02 at n=50, ~0.0002 at n=10000), so no fixed cut is meaningful across
    sample sizes. The veto uses skew / excess kurtosis instead."""
    normal: bool
    test: str  # "shapiro" | "anderson-darling" | "insufficient"
    statistic: float
    pvalue: float
    n: int
    skew: float
    excess_kurtosis: float
    qq_departure: float
    rejected: bool  # did the test reject at alpha, before the practical-significance veto
    negligible: bool  # is the departure too small to matter (the veto)
    reason: str


@dataclass(frozen=True, slots=True)
class Interval:
    """A confidence interval and, critically, WHAT IT IS FOR. An interval around the mean is not
    an interval around the min-of-k; ``statistic`` keeps the two from being confused in a table
    or a figure caption."""
    statistic: str  # "mean" | "median" | "min_of_k" | "speedup(min_of_k)" | ...
    point: float
    low: float
    high: float
    confidence: float
    method: str  # "t" | "bootstrap-BCa" | "bootstrap-percentile" | "rank-median" | "fieller"
    n: int

    def label(self) -> str:
        """One-line figure/table label naming both the statistic and the interval kind."""
        return f"{int(round(self.confidence * 100))}% {self.method} CI for {self.statistic}"


@dataclass(frozen=True, slots=True)
class Comparison:
    """A two-sample significance result: p-value AND effect size, never one without the other.

    A p-value on 10000 repetitions can certify a 0.1% difference that means nothing, so
    ``effect`` (Cliff's delta, identical to the rank-biserial correlation for Mann-Whitney) and
    ``ratio`` (the median-scale ratio, a physical effect size) travel with it."""
    test: str  # "mann-whitney" | "wilcoxon-signed-rank"
    statistic: float
    pvalue: float
    effect: float  # Cliff's delta / rank-biserial in [-1, 1]; <0 means `a` is smaller (faster)
    effect_name: str
    ratio: float  # median(b) / median(a) -- the speed-up on the median scale
    n_a: int
    n_b: int
    significant: bool  # pvalue < alpha (UNADJUSTED; see adjust_pvalues for the corpus level)
    alpha: float


@dataclass(frozen=True, slots=True)
class Equivalence:
    """A TOST result: "this optimisation changed nothing measurable" as a POSITIVE claim.

    Not the same as failing to reject a difference -- a null result can mean "no effect" or "no
    power", and only TOST separates them. ``margin`` is fractional and multiplicative (0.05 =
    within +/-5%), which is the right scale for ratio-scale wall-clock."""
    equivalent: bool
    pvalue: float  # max of the two one-sided p-values -- the TOST p
    pvalue_lower: float
    pvalue_upper: float
    margin: float
    ratio: float
    test: str
    alpha: float


@dataclass(frozen=True, slots=True)
class CorpusComparison:
    """One kernel's row in a corpus-wide sweep, carrying both the raw and the multiplicity-
    adjusted p-value. Only ``significant_adjusted`` may be quoted as a finding."""
    key: str
    comparison: Comparison
    pvalue_adjusted: float
    significant_adjusted: bool


def clean(samples: Sequence[float]) -> np.ndarray:
    """Finite, strictly positive samples as a float array. Wall-clock is positive by
    construction, so a zero or negative entry is a broken timer reading, not a fast run."""
    x = np.asarray(samples, dtype=float)
    return x[np.isfinite(x) & (x > 0.0)]


def qq_departure(x: np.ndarray) -> float:
    """``1 - r**2`` between the sorted samples and their normal (Blom / Filliben) plotting
    quantiles -- the fraction of the QQ plot's variance the normal fit does not explain."""
    n = x.size
    if n < 3:
        return float("nan")
    xs = np.sort(x)
    if float(np.ptp(xs)) == 0.0:
        return 0.0  # constant sample: degenerate, but the QQ line is exactly flat
    offset = 0.375 if n <= 10 else 0.5  # Blom for small n, Hazen otherwise
    probs = (np.arange(1, n + 1) - offset) / (n + 1 - 2 * offset)
    r = float(np.corrcoef(xs, norm.ppf(probs))[0, 1])
    return 1.0 - r * r


def anderson_darling_statistic(x: np.ndarray) -> float:
    """Anderson-Darling ``A**2`` against a normal fitted to the sample (Stephens' case 3: both
    mean and variance estimated).

    Computed here rather than through :func:`scipy.stats.anderson` because that function's
    return schema is mid-migration across scipy versions (1.17 warns that a ``method`` argument
    will become mandatory and that ``critical_value`` will disappear in 1.19). The statistic is
    four lines, so owning it costs less than pinning a scipy floor."""
    n = x.size
    xs = np.sort(x)
    sd = float(np.std(xs, ddof=1))
    if sd == 0.0:
        return float("nan")
    z = (xs - float(np.mean(xs))) / sd
    cdf = np.clip(norm.cdf(z), 1e-15, 1.0 - 1e-15)  # clip: ln(0) at an extreme tail sample
    weights = 2.0 * np.arange(1, n + 1) - 1.0
    return float(-n - np.sum(weights * (np.log(cdf) + np.log(1.0 - cdf[::-1]))) / n)


def anderson_darling_pvalue(statistic: float, n: int) -> float:
    """D'Agostino-Stephens p-value approximation for :func:`anderson_darling_statistic`. A
    p-value keeps the verdict schema uniform with the Shapiro-Wilk branch, so the same
    multiplicity correction consumes both."""
    # The published fit is a piecewise quadratic in A*^2 valid near the critical region; past
    # A*^2 ~ 10 its last branch turns back UP and eventually overflows exp(). Everything beyond
    # that is p < 1e-23 anyway, so cap the argument rather than extrapolate a curve into nonsense.
    a2 = min(statistic * (1.0 + 0.75 / n + 2.25 / (n * n)), 10.0)
    if a2 < 0.2:
        return 1.0 - math.exp(-13.436 + 101.14 * a2 - 223.73 * a2 * a2)
    if a2 < 0.34:
        return 1.0 - math.exp(-8.318 + 42.796 * a2 - 59.938 * a2 * a2)
    if a2 < 0.6:
        return math.exp(0.9177 - 4.279 * a2 - 1.38 * a2 * a2)
    return math.exp(1.2937 - 5.709 * a2 + 0.0186 * a2 * a2)


def check_normality(samples: Sequence[float], alpha: float = DEFAULT_ALPHA) -> NormalityVerdict:
    """Is this sample close enough to normal to justify a parametric interval?

    ⛔ Feed the RAW per-repeat timings. Feeding the reduced ``min_of_k`` number tests an
    extreme-value statistic's distribution, which answers a question nobody asked.

    Test selection: Shapiro-Wilk for ``n <= 5000`` -- the most powerful omnibus normality test at
    the sizes the harness produces (``measurement.repeat`` default 50, so n is tens to low
    thousands) -- and Anderson-Darling above that, where scipy documents the Shapiro p-value as
    unreliable and the extra tail sensitivity is what matters anyway.

    The verdict is NOT the p-value alone. With large n both tests reject on deviations too small
    to change any downstream number, so a rejection is vetoed when the shape statistics are
    within practical-normality bounds (``|skew| <= 0.5``, ``|excess kurtosis| <= 1``). Expect
    ``normal=False`` to be the common outcome on real timing data: wall-clock is bounded below by
    the hardware, right-skewed and heavy-tailed (interrupts, frequency scaling, page faults, a
    co-runner), so the non-parametric branch is the DEFAULT path, not the fallback.
    """
    x = clean(samples)
    n = int(x.size)
    if n < MIN_NORMALITY_N:
        return NormalityVerdict(False, "insufficient", float("nan"), float("nan"), n, float("nan"), float("nan"),
                                float("nan"), False, False,
                                f"n={n} < {MIN_NORMALITY_N}: no power to establish normality; use non-parametric")
    if float(np.ptp(x)) == 0.0:
        return NormalityVerdict(False, "insufficient", float("nan"), float("nan"), n, 0.0, 0.0, 0.0, False, False,
                                "zero spread: timer resolution floor, not a normal sample")

    sk = float(skew(x, bias=False))
    ek = float(kurtosis(x, fisher=True, bias=False))
    dep = qq_departure(x)

    if n <= SHAPIRO_MAX_N:
        result = shapiro(x)
        test, statistic, pvalue = "shapiro", float(result.statistic), float(result.pvalue)
    else:
        test = "anderson-darling"
        statistic = anderson_darling_statistic(x)
        pvalue = anderson_darling_pvalue(statistic, n)

    rejected = pvalue < alpha
    negligible = abs(sk) <= MAX_ABS_SKEW and abs(ek) <= MAX_ABS_EXCESS_KURTOSIS
    normal = (not rejected) or negligible
    if not rejected:
        reason = f"{test} p={pvalue:.3g} >= alpha={alpha}"
    elif negligible:
        reason = (f"{test} p={pvalue:.3g} < alpha={alpha} but skew={sk:.3f}, excess kurtosis={ek:.3f} are within "
                  f"practical bounds -- rejection is too small to matter at n={n}")
    else:
        reason = f"{test} p={pvalue:.3g} < alpha={alpha}, skew={sk:.3f}, excess kurtosis={ek:.3f}"
    return NormalityVerdict(normal, test, statistic, pvalue, n, sk, ek, dep, rejected, negligible, reason)


def mean_ci_t(samples: Sequence[float], confidence: float = DEFAULT_CONFIDENCE) -> Interval:
    """Student-t interval for the MEAN. t, not z: sigma is estimated from the same sample."""
    x = clean(samples)
    n = int(x.size)
    mean = float(np.mean(x)) if n else float("nan")
    if n < 2:
        return Interval("mean", mean, mean, mean, confidence, "t", n)
    half = float(t.ppf(0.5 + confidence / 2.0, n - 1)) * float(np.std(x, ddof=1)) / math.sqrt(n)
    return Interval("mean", mean, mean - half, mean + half, confidence, "t", n)


def median_rank_ci(samples: Sequence[float], confidence: float = DEFAULT_CONFIDENCE) -> Interval:
    """Distribution-free order-statistic interval for the MEDIAN.

    The number of samples below the population median is Binomial(n, 1/2) under nothing but
    independence, so the ranks ``[k, n+1-k]`` bracket it with exact coverage -- no bootstrap, no
    resampling seed, no shape assumption. Coverage is conservative (it steps in integer ranks),
    which is why :func:`bootstrap_ci` remains the default for the median; this one is the
    assumption-free cross-check when a reviewer distrusts the bootstrap."""
    x = np.sort(clean(samples))
    n = int(x.size)
    med = float(np.median(x)) if n else float("nan")
    if n < 6:  # below this no rank pair reaches 95% coverage; report a point interval honestly
        return Interval("median", med, med, med, confidence, "rank-median", n)
    tail = (1.0 - confidence) / 2.0
    k = int(binom.ppf(tail, n, 0.5))  # largest k with P(X < k) <= tail
    k = max(1, min(k, n // 2))
    return Interval("median", med, float(x[k - 1]), float(x[n - k]), confidence, "rank-median", n)


def bootstrap_ci(samples: Sequence[float],
                 statistic: Callable[[np.ndarray], float] = np.median,
                 name: str = "median",
                 confidence: float = DEFAULT_CONFIDENCE,
                 n_resamples: int = DEFAULT_RESAMPLES,
                 method: str = "BCa",
                 seed: int = DEFAULT_SEED) -> Interval:
    """Non-parametric bootstrap interval for whatever ``statistic`` actually gets reported.

    BCa by default -- it corrects both the median bias and the skew of the bootstrap
    distribution, which matters precisely because timing samples are skewed. BCa's acceleration
    term needs a jackknife and degenerates on tiny or near-constant samples, so a failure falls
    back to the percentile interval and SAYS SO in ``method``; a silent method swap would make
    two differently-derived intervals look alike in a table."""
    x = clean(samples)
    n = int(x.size)
    point = float(statistic(x)) if n else float("nan")
    if n < 3 or float(np.ptp(x)) == 0.0:
        return Interval(name, point, point, point, confidence, f"bootstrap-{method}", n)

    def vectorized(a: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.apply_along_axis(statistic, axis, a)

    for attempt in (method, "percentile"):
        try:
            res = bootstrap((x, ),
                            vectorized,
                            confidence_level=confidence,
                            n_resamples=n_resamples,
                            method=attempt,
                            vectorized=True,
                            random_state=np.random.default_rng(seed))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        low, high = float(res.confidence_interval.low), float(res.confidence_interval.high)
        if math.isfinite(low) and math.isfinite(high):
            return Interval(name, point, low, high, confidence, f"bootstrap-{attempt}", n)
    return Interval(name, point, point, point, confidence, "bootstrap-degenerate", n)


def min_of_k_ci(samples: Sequence[float],
                k: int,
                confidence: float = DEFAULT_CONFIDENCE,
                n_resamples: int = DEFAULT_RESAMPLES,
                seed: int = DEFAULT_SEED) -> Interval:
    """Interval for the statistic the harness ACTUALLY credits: ``min`` of ``k`` repeats.

    Resamples ``k`` values with replacement from the observed ``n`` and takes their minimum,
    ``n_resamples`` times. With ``k`` held FIXED this is a bootstrap of a fixed-dimension
    functional and is consistent -- it answers "what min would a fresh k-repeat run produce?".

    ⛔ The classic ``k == n`` case is NOT that. The n-out-of-n bootstrap of an extreme order
    statistic is inconsistent (Bickel & Freedman 1981): ~63% of the resamples reproduce the
    observed minimum exactly, so the bootstrap law does not converge to the law of the minimum.
    ``method`` records ``bootstrap-min-of-k`` when ``k < n`` and ``bootstrap-min-of-n
    (inconsistent, reproducibility band)`` when ``k >= n``, so a caller cannot quote the second
    as if it were the first. When an honest interval for a location parameter is wanted, interval
    the MEDIAN instead -- that is why :func:`speedup_ci` defaults to the median statistic.
    """
    x = clean(samples)
    n = int(x.size)
    k = max(1, int(k))
    point = float(np.min(x)) if n else float("nan")
    consistent = k < n
    method = "bootstrap-min-of-k" if consistent else "bootstrap-min-of-n (inconsistent, reproducibility band)"
    if n < 2 or float(np.ptp(x)) == 0.0:
        return Interval(f"min_of_{k}", point, point, point, confidence, method, n)
    rng = np.random.default_rng(seed)
    draws = np.min(rng.choice(x, size=(n_resamples, k), replace=True), axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return Interval(f"min_of_{k}", point, float(low), float(high), confidence, method, n)


def interval_for(samples: Sequence[float],
                 verdict: Optional[NormalityVerdict] = None,
                 confidence: float = DEFAULT_CONFIDENCE,
                 alpha: float = DEFAULT_ALPHA,
                 n_resamples: int = DEFAULT_RESAMPLES,
                 seed: int = DEFAULT_SEED) -> Tuple[Interval, NormalityVerdict]:
    """The interval the normality verdict SELECTS: t-interval for the mean when normal, BCa
    bootstrap interval for the median when not. Returns the verdict alongside so the caller can
    label the figure with which branch it took."""
    verdict = verdict if verdict is not None else check_normality(samples, alpha=alpha)
    if verdict.normal:
        return mean_ci_t(samples, confidence=confidence), verdict
    return bootstrap_ci(samples, np.median, "median", confidence, n_resamples, "BCa", seed), verdict


def fieller_ratio_ci(numerator: Sequence[float],
                     denominator: Sequence[float],
                     confidence: float = DEFAULT_CONFIDENCE) -> Interval:
    """Fieller's theorem interval for the RATIO OF MEANS of two INDEPENDENT normal samples.

    A ratio's interval is not the ratio of two intervals: the denominator's uncertainty enters
    non-linearly, and when the denominator's own interval approaches zero the ratio's interval is
    genuinely UNBOUNDED. Fieller shows that honestly (``g >= 1`` -> infinite bounds) where the
    delta method quietly returns a finite, wrong interval. For timings ``g >= 1`` should never
    happen -- it means the denominator is not distinguishable from a zero runtime."""
    a = clean(numerator)
    b = clean(denominator)
    na, nb = int(a.size), int(b.size)
    ma, mb = (float(np.mean(a)) if na else float("nan")), (float(np.mean(b)) if nb else float("nan"))
    ratio = ma / mb if mb else float("nan")
    if na < 2 or nb < 2 or not mb:
        return Interval("ratio_of_means", ratio, float("nan"), float("nan"), confidence, "fieller", min(na, nb))
    v_a = float(np.var(a, ddof=1)) / na
    v_b = float(np.var(b, ddof=1)) / nb
    crit = float(t.ppf(0.5 + confidence / 2.0, na + nb - 2))
    g = crit * crit * v_b / (mb * mb)
    if g >= 1.0:
        return Interval("ratio_of_means", ratio, float("-inf"), float("inf"), confidence, "fieller", min(na, nb))
    radicand = v_a * (1.0 - g) + ratio * ratio * v_b
    half = (crit / abs(mb)) * math.sqrt(max(radicand, 0.0))
    low, high = (ratio - half) / (1.0 - g), (ratio + half) / (1.0 - g)
    return Interval("ratio_of_means", ratio, low, high, confidence, "fieller", min(na, nb))


def speedup_ci(baseline: Sequence[float],
               candidate: Sequence[float],
               statistic: Callable[[np.ndarray], float] = np.median,
               name: str = "median",
               confidence: float = DEFAULT_CONFIDENCE,
               n_resamples: int = DEFAULT_RESAMPLES,
               seed: int = DEFAULT_SEED) -> Interval:
    """Interval for the SPEED-UP ``stat(baseline) / stat(candidate)`` -- the ratio itself.

    ⛔ An interval on the numerator and one on the denominator do not compose into an interval on
    the ratio -- dividing endpoint by endpoint gives a strictly and needlessly wider band. This
    bootstraps the RATIO: each replicate resamples both sides independently (they ARE independent
    -- see the module docstring) and divides, so the denominator's variability is carried through
    the non-linearity instead of being bolted on afterwards.

    The percentile interval is equivariant under monotone transformation of the replicates, so
    the band is not forced symmetric about the point estimate the way a ``point +/- z*se``
    interval is -- which matters because a ratio's sampling distribution is skewed. Inverting the
    arguments reproduces the reciprocal interval up to MONTE-CARLO error only: each call draws
    its own replicates, so the two agree to O(1/sqrt(n_resamples)), not exactly.

    Each side draws from its OWN spawned stream, so adding samples to one side does not perturb
    the other side's replicates -- a published interval then moves only for the reason it should.

    Pass ``statistic=np.min`` to interval the min-of-k speed-up the default backend credits;
    ``np.median`` is the default because the minimum's bootstrap is only a reproducibility band
    (see :func:`min_of_k_ci`).
    """
    b = clean(baseline)
    c = clean(candidate)
    point = (float(statistic(b)) / float(statistic(c))) if (b.size and c.size and float(statistic(c))) else float("nan")
    n = min(int(b.size), int(c.size))
    if b.size < 2 or c.size < 2:
        return Interval(f"speedup({name})", point, point, point, confidence, "bootstrap-percentile", n)
    stream_b, stream_c = np.random.default_rng(seed).spawn(2)
    bi = np.apply_along_axis(statistic, 1, stream_b.choice(b, size=(n_resamples, b.size), replace=True))
    ci = np.apply_along_axis(statistic, 1, stream_c.choice(c, size=(n_resamples, c.size), replace=True))
    ratios = bi[ci > 0.0] / ci[ci > 0.0]
    if ratios.size == 0:
        return Interval(f"speedup({name})", point, point, point, confidence, "bootstrap-degenerate", n)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(ratios, [tail, 1.0 - tail])
    return Interval(f"speedup({name})", point, float(low), float(high), confidence, "bootstrap-percentile", n)


def cliffs_delta(u_statistic: float, n_a: int, n_b: int) -> float:
    """Cliff's delta from a Mann-Whitney U: ``2U/(n_a n_b) - 1``, in ``[-1, 1]``.

    Identical to the rank-biserial correlation for this test. It is the probability that a random
    draw from ``a`` exceeds one from ``b`` minus the reverse -- so on timings a NEGATIVE delta
    means ``a`` is faster. Scale-free and unaffected by n, which is exactly what a p-value is
    not."""
    if n_a <= 0 or n_b <= 0:
        return float("nan")
    return 2.0 * u_statistic / (n_a * n_b) - 1.0


def mann_whitney(a: Sequence[float], b: Sequence[float], alpha: float = DEFAULT_ALPHA) -> Comparison:
    """Two-sided Mann-Whitney U for INDEPENDENT samples -- the correct test for this harness.

    Justification (see the module docstring for the file:line trail): the baselines are timed in
    their own child processes before the candidate's ``_call_isolated`` on the agent track, and in
    a wholly separate process invocation on the framework track. There is no rep-to-rep pairing to
    exploit, so Wilcoxon signed-rank would be pairing unrelated observations and inventing
    precision. Rank-based, so a heavy right tail does not drag the test the way a t-test's mean
    would."""
    x, y = clean(a), clean(b)
    n_a, n_b = int(x.size), int(y.size)
    if n_a < 2 or n_b < 2:
        return Comparison("mann-whitney", float("nan"), float("nan"), float("nan"), "cliffs-delta", float("nan"), n_a,
                          n_b, False, alpha)
    try:
        result = mannwhitneyu(x, y, alternative="two-sided")
        statistic, pvalue = float(result.statistic), float(result.pvalue)
    except ValueError:  # every value identical on both sides -> no rank information
        statistic, pvalue = float("nan"), 1.0
    med_a, med_b = float(np.median(x)), float(np.median(y))
    ratio = (med_b / med_a) if med_a else float("nan")
    effect = cliffs_delta(statistic, n_a, n_b) if math.isfinite(statistic) else 0.0
    return Comparison("mann-whitney", statistic, pvalue, effect, "cliffs-delta", ratio, n_a, n_b, bool(pvalue < alpha),
                      alpha)


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float], alpha: float = DEFAULT_ALPHA) -> Comparison:
    """Two-sided Wilcoxon signed-rank for PAIRED samples -- rep ``i`` of both sides measured
    back-to-back on the same input and machine.

    ⛔ Nothing in the harness today collects that way (see the module docstring): using this on
    the harness's independently-collected passes would pair unrelated observations. It exists so
    an INTERLEAVED collector -- which would be the stronger design, since it cancels slow drift
    such as thermal throttling -- has the right test waiting for it, and so the choice between the
    two tests is a documented decision rather than an accident."""
    x, y = clean(a), clean(b)
    if x.size != y.size:
        raise ValueError(f"paired test needs equal-length samples; got {x.size} and {y.size}")
    n = int(x.size)
    if n < 2 or np.all(x == y):
        return Comparison("wilcoxon-signed-rank", float("nan"), 1.0, 0.0, "rank-biserial", 1.0, n, n, False, alpha)
    result = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
    statistic, pvalue = float(result.statistic), float(result.pvalue)
    # Matched-pairs rank-biserial: signed-rank sums normalised by the total rank mass.
    nonzero = int(np.count_nonzero(x - y))
    total = nonzero * (nonzero + 1) / 2.0
    effect = (2.0 * statistic / total - 1.0) if total else 0.0
    med_a, med_b = float(np.median(x)), float(np.median(y))
    ratio = (med_b / med_a) if med_a else float("nan")
    return Comparison("wilcoxon-signed-rank", statistic, pvalue, effect, "rank-biserial", ratio, n, n,
                      bool(pvalue < alpha), alpha)


def compare(a: Sequence[float], b: Sequence[float], paired: bool = False, alpha: float = DEFAULT_ALPHA) -> Comparison:
    """Significance between two systems. ``paired=False`` (the default, and what the harness
    produces) -> Mann-Whitney U; ``paired=True`` -> Wilcoxon signed-rank."""
    return wilcoxon_signed_rank(a, b, alpha) if paired else mann_whitney(a, b, alpha)


def tost_equivalence(a: Sequence[float],
                     b: Sequence[float],
                     margin: float = 0.05,
                     alpha: float = DEFAULT_ALPHA,
                     paired: bool = False) -> Equivalence:
    """Two one-sided tests for EQUIVALENCE: can we assert "this changed nothing measurable"?

    ⛔ Failing to reject a difference is NOT evidence of no difference -- it is equally consistent
    with having no power. TOST inverts the burden: the null is "the difference is at least
    ``margin``", and REJECTING it on both sides is a positive claim of equivalence.

    ``margin`` is multiplicative because wall-clock is ratio-scale: ``0.05`` means "within +/-5%".
    The two one-sided tests are run non-parametrically (Mann-Whitney, or Wilcoxon when
    ``paired``) against the baseline scaled to each bound, so no normality is assumed -- the same
    reason every other test here is rank-based. Equivalent iff BOTH one-sided tests reject, i.e.
    ``max(p_lower, p_upper) < alpha``."""
    if not 0.0 < margin < 1.0:
        raise ValueError(f"margin must be a fraction in (0, 1); got {margin!r}")
    x, y = clean(a), clean(b)
    med_a, med_b = (float(np.median(x)) if x.size else float("nan")), (float(np.median(y)) if y.size else float("nan"))
    ratio = (med_b / med_a) if med_a else float("nan")
    test = "wilcoxon-tost" if paired else "mann-whitney-tost"
    if x.size < 2 or y.size < 2:
        return Equivalence(False, float("nan"), float("nan"), float("nan"), margin, ratio, test, alpha)

    def one_sided(scaled: np.ndarray, alternative: str) -> float:
        if paired:
            if np.all(x == scaled):
                return 1.0
            return float(wilcoxon(x, scaled, alternative=alternative, zero_method="wilcox").pvalue)
        try:
            return float(mannwhitneyu(x, scaled, alternative=alternative).pvalue)
        except ValueError:
            return 1.0

    # Upper: a is stochastically BELOW the b*(1+margin) bound. Lower: a is ABOVE b*(1-margin).
    p_upper = one_sided(y * (1.0 + margin), "less")
    p_lower = one_sided(y * (1.0 - margin), "greater")
    pvalue = max(p_upper, p_lower)
    return Equivalence(bool(pvalue < alpha), pvalue, p_lower, p_upper, margin, ratio, test, alpha)


def holm_bonferroni(pvalues: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down ADJUSTED p-values (input order preserved).

    Controls the family-wise error rate: the probability of even ONE false claim across the
    family. Uniformly more powerful than plain Bonferroni and assumption-free."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return []
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1.0, (n - np.arange(n)) * p[order])
    adjusted = np.maximum.accumulate(adjusted)  # step-down monotonicity
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return [float(v) for v in out]


def benjamini_hochberg(pvalues: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg FDR-adjusted p-values (input order preserved), via
    :func:`scipy.stats.false_discovery_control`."""
    p = np.asarray(pvalues, dtype=float)
    if p.size == 0:
        return []
    return [float(v) for v in false_discovery_control(p, method="bh")]


def adjust_pvalues(pvalues: Sequence[float], method: str = "fdr_bh") -> List[float]:
    """Multiplicity-adjusted p-values. ``fdr_bh`` (default) or ``holm``.

    ⛔ The corpus is ~578 kernels. Testing each at alpha=0.05 manufactures ~29 false positives by
    construction, so an unadjusted per-kernel p-value is not a finding.

    BH FDR is the DEFAULT because the corpus question is a screening one -- "which kernels sped
    up?" -- where controlling the expected PROPORTION of false discoveries among the claims keeps
    almost all the power. Holm controls the probability of ANY false claim and is the right
    choice for a single family-wide assertion ("no kernel regressed"), at a large cost in power
    over ~578 tests."""
    if method == "holm":
        return holm_bonferroni(pvalues)
    if method == "fdr_bh":
        return benjamini_hochberg(pvalues)
    raise ValueError(f"unknown multiple-comparison method {method!r}; use 'fdr_bh' or 'holm'")


def compare_corpus(cells: Mapping[str, Tuple[Sequence[float], Sequence[float]]],
                   paired: bool = False,
                   alpha: float = DEFAULT_ALPHA,
                   method: str = "fdr_bh") -> List[CorpusComparison]:
    """Corpus-wide comparison with multiplicity correction ALREADY APPLIED.

    ``cells`` maps a key (kernel, or ``kernel@framework``) to ``(candidate_samples,
    baseline_samples)``; iteration order of the mapping is preserved into the result list, so a
    report built from it is deterministic. This is the entry point corpus-level callers use --
    per-kernel :func:`compare` is for a single claim in isolation, and quoting a table of its raw
    p-values is exactly the mistake ``method`` exists to stop."""
    keys = list(cells.keys())
    comparisons = [compare(cells[k][0], cells[k][1], paired=paired, alpha=alpha) for k in keys]
    raw = [c.pvalue if math.isfinite(c.pvalue) else 1.0 for c in comparisons]
    adjusted = adjust_pvalues(raw, method=method)
    return [CorpusComparison(key, comp, adj, bool(adj < alpha)) for key, comp, adj in zip(keys, comparisons, adjusted)]


def summarize(samples: Sequence[float],
              confidence: float = DEFAULT_CONFIDENCE,
              alpha: float = DEFAULT_ALPHA,
              seed: int = DEFAULT_SEED) -> Dict[str, object]:
    """One sample's full honest summary: the verdict, the interval it selected, and the
    min-of-k reproducibility band the default backend credits. A plain dict so a report writer
    or a JSON dump consumes it without importing this module's dataclasses."""
    x = clean(samples)
    interval, verdict = interval_for(x, confidence=confidence, alpha=alpha, seed=seed)
    k = int(x.size)
    return {
        "n": int(x.size),
        "normal": verdict.normal,
        "normality": verdict,
        "interval": interval,
        "interval_label": interval.label(),
        "min_of_k": min_of_k_ci(x, k, confidence=confidence, seed=seed),
    }
