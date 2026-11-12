# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every statistic a figure or a table reports, defined exactly once: outlier rejection, medians, geometric and
arithmetic means with bootstrap intervals, the signed-change transform and sign-flip building blocks. No normality
is assumed (SC15 Rule 6); test choice lives in :mod:`hpcagent_bench.stats.significance`."""

import enum
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

__all__ = [
    "default_alpha",
    "DEFAULT_CI_METHOD",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_MAD_Z",
    "DEFAULT_RESAMPLES",
    "MAD_TO_SIGMA",
    "MEANAD_TO_SIGMA",
    "MIN_INTERVAL_SAMPLES",
    "MIN_PAIRS_FOR_INTERVAL",
    "SIGN_FLIP_DRAWS",
    "SIGN_FLIP_EXACT_MAX_N",
    "FloatArray",
    "Interval",
    "Samples",
    "Signs",
    "Statistic",
    "Unusable",
    "bootstrap_ci",
    "drop_outliers",
    "geomean",
    "geomean_ci",
    "geomean_interval",
    "log2_change",
    "mean_ci",
    "mean_interval",
    "mean_ratio_ci",
    "median_ci",
    "sign_flip_interval",
    "sign_flip_pvalue",
    "sign_flips",
    "signed_change",
    "signs",
    "usable_ratios",
]

# scipy is imported INSIDE the functions that need it: the grading path imports this module.

#: One timing sample per element. float64 is what ``np.asarray(..., dtype=float)`` produces.
FloatArray = npt.NDArray[np.float64]

#: What both entry points accept: a plain sequence of numbers, or an already-built float array.
Samples = Sequence[float] | FloatArray

#: A numpy reduction taking ``axis``: ``np.median``, ``np.mean`` or ``np.min``.
Statistic = Callable[..., "float | FloatArray"]

#: MAD -> normal-sigma consistency constant (1 / 0.6745). A modified z-score of ``k`` is
#: ``k`` robust standard deviations above the median.
MAD_TO_SIGMA: float = 1.4826

#: Mean-absolute-deviation -> normal-sigma constant (sqrt(pi/2)). Used only as the fallback
#: robust scale when MAD == 0 (>= half the samples identical) -- Iglewicz-Hoaglin.
MEANAD_TO_SIGMA: float = 1.253314

#: Default upper modified-z threshold. 5 keeps the median plus a ~5 robust-sigma slow tail --
#: "very bad" (OS-hiccup) samples, not ordinary run-to-run jitter.
DEFAULT_MAD_Z: float = 5.0

#: Fewest samples a percentile bootstrap of a MEDIAN can say anything about.
MIN_INTERVAL_SAMPLES: int = 5

#: Bootstrap median-CI defaults, exposed so callers and docs can report them.
DEFAULT_CONFIDENCE: float = 0.95
DEFAULT_RESAMPLES: int = 9999
DEFAULT_CI_METHOD: str = "percentile"


def drop_outliers(
    samples: Samples, threshold: float = DEFAULT_MAD_Z, warn: bool = True, label: str = ""
) -> tuple[FloatArray, FloatArray]:
    """Drop upper-tail outliers by robust modified z-score ``(x - median) / (1.4826 * MAD)`` above ``threshold``
    (slow side only); returns ``(kept, dropped)``.

    When ``warn``, every drop is warned about (prefixed by ``label``). Fewer than 3 samples or an all-identical
    sample yield no drops; MAD == 0 falls back to the mean absolute deviation."""
    x: FloatArray = np.asarray(samples, dtype=np.float64)
    empty: FloatArray = np.empty(0, dtype=np.float64)
    if x.size < 3:
        return x, empty
    med = float(np.median(x))
    abs_dev: FloatArray = np.abs(x - med)
    scale = float(np.median(abs_dev)) * MAD_TO_SIGMA
    if scale == 0.0:
        # MAD is 0 (>= half the samples equal the median): fall back to the mean absolute deviation (Iglewicz-Hoaglin).
        scale = float(np.mean(abs_dev)) * MEANAD_TO_SIGMA
    if scale == 0.0:
        return x, empty  # truly all identical: no robust scale, nothing to flag
    modified_z: FloatArray = (x - med) / scale
    drop_mask: npt.NDArray[np.bool_] = modified_z > threshold  # upper (slow) tail only
    kept: FloatArray = x[~drop_mask]
    dropped: FloatArray = x[drop_mask]
    if warn and dropped.size != 0:
        prefix = f"{label}: " if label else ""
        warnings.warn(
            f"{prefix}dropped {dropped.size} slow outlier sample(s) "
            f"(modified z > {threshold}, median={med:.4g}): {np.round(dropped, 4).tolist()}",
            stacklevel=2,
        )
    return kept, dropped


def median_ci(
    samples: Samples,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_RESAMPLES,
    method: str = DEFAULT_CI_METHOD,
    drop: bool = True,
    warn: bool = True,
    label: str = "",
    seed: int = 0,
    min_n: int = 0,
) -> tuple[float, float, float, int]:
    """``(median, ci_low, ci_high, n_dropped)``: a percentile bootstrap CI after optional outlier rejection
    (:func:`drop_outliers`); fewer than ``min_n`` samples get a NaN interval, no spread a point interval."""
    x: FloatArray = np.asarray(samples, dtype=np.float64)
    n_dropped = 0
    if drop:
        x, dropped = drop_outliers(x, warn=warn, label=label)
        n_dropped = int(dropped.size)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan"), n_dropped
    if x.size < min_n:
        return float(np.median(x)), math.nan, math.nan, n_dropped
    interval = bootstrap_ci(x, np.median, "median", confidence, n_resamples, method, seed)
    return interval.point, interval.low, interval.high, n_dropped


def default_alpha() -> float:
    """``statistics.alpha``: the two-sided error rate every interval and test here defaults to."""
    from hpcagent_bench import config

    return config.current().statistics.alpha


#: Below this a two-sided sign-flip test cannot reach alpha = 0.05 whatever the data says
#: (its smallest attainable p at n = 5 is 0.0625), so an interval would be decoration.
MIN_PAIRS_FOR_INTERVAL: int = 6


@dataclass(frozen=True, slots=True)
class Interval:
    """A confidence interval and WHAT IT IS FOR (``statistic``), so a table never confuses two."""

    statistic: str  # "mean" | "median" | "geomean" | "min_of_k" | "speedup(min_of_k)" | ...
    point: float
    low: float
    high: float
    confidence: float
    method: str  # "t" | "log-t" | "bootstrap-BCa" | "bootstrap-percentile" | "rank-median" | ...
    n: int

    def label(self) -> str:
        """One-line figure/table label naming both the statistic and the interval kind."""
        return f"{round(self.confidence * 100)}% {self.method} CI for {self.statistic}"


def bootstrap_ci(
    samples: Samples,
    statistic: Statistic = np.median,
    name: str = "median",
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_RESAMPLES,
    method: str = "BCa",
    seed: int = 0,
) -> Interval:
    """Non-parametric bootstrap interval for ``statistic`` (a numpy reduction taking ``axis``) of ``samples`` as
    given; a BCa failure falls back to percentile and SAYS SO in ``method``. Fewer than 3 samples or no spread is a
    degenerate interval."""
    x: FloatArray = np.asarray(samples, dtype=np.float64)
    n = int(x.size)
    point = float(statistic(x)) if n else math.nan
    if n < 3 or float(np.ptp(x)) == 0.0:
        return Interval(name, point, point, point, confidence, f"bootstrap-{method}", n)
    from scipy.stats import bootstrap  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

    for attempt in dict.fromkeys((method, "percentile")):
        try:
            res = bootstrap(  # pyright: ignore[reportUnknownVariableType] -- unstubbed scipy
                (x,),
                statistic,
                confidence_level=confidence,
                n_resamples=n_resamples,
                method=attempt,
                vectorized=True,
                random_state=np.random.default_rng(seed),  # pyright: ignore[reportCallIssue] -- scipy compat spelling
            )
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        low = float(res.confidence_interval.low)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        high = float(res.confidence_interval.high)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        if math.isfinite(low) and math.isfinite(high):
            return Interval(name, point, low, high, confidence, f"bootstrap-{attempt}", n)
    return Interval(name, point, point, point, confidence, "bootstrap-degenerate", n)


def mean_ratio_ci(
    numerator: Samples,
    denominator: Samples,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Interval:
    """``mean(numerator) / mean(denominator)`` over paired samples with its paired BCa bootstrap interval (a cost
    change, SC15); percentile fallback named in ``method``; fewer than 3 pairs or no spread is the point alone."""
    top: FloatArray = np.asarray(numerator, dtype=np.float64)
    bottom: FloatArray = np.asarray(denominator, dtype=np.float64)
    n = int(top.size)
    point = float(np.mean(top) / np.mean(bottom)) if n else math.nan
    if n < 3 or (float(np.ptp(top)) == 0.0 and float(np.ptp(bottom)) == 0.0):
        return Interval("mean ratio", point, point, point, confidence, "bootstrap-degenerate", n)
    from scipy.stats import bootstrap  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

    def ratio(a: FloatArray, b: FloatArray, axis: int = -1) -> FloatArray:
        return np.mean(a, axis=axis) / np.mean(b, axis=axis)

    for attempt in ("BCa", "percentile"):
        try:
            res = bootstrap(  # pyright: ignore[reportUnknownVariableType] -- unstubbed scipy
                (top, bottom),
                ratio,
                paired=True,
                confidence_level=confidence,
                n_resamples=n_resamples,
                method=attempt,
                vectorized=True,
                random_state=np.random.default_rng(seed),  # pyright: ignore[reportCallIssue] -- scipy compat spelling
            )
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        low = float(res.confidence_interval.low)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        high = float(res.confidence_interval.high)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        if math.isfinite(low) and math.isfinite(high):
            return Interval("mean ratio", point, low, high, confidence, f"bootstrap-{attempt}", n)
    return Interval("mean ratio", point, point, point, confidence, "bootstrap-degenerate", n)


def usable_ratios(values: Samples, label: str = "", warn: bool = True) -> FloatArray:
    """The entries of ``values`` a geometric mean may be taken over: finite and strictly positive. Anything else is
    a MISSING measurement, dropped (never clamped) with a warning prefixed by ``label``."""
    x: FloatArray = np.asarray(values, dtype=np.float64)
    keep: npt.NDArray[np.bool_] = np.isfinite(x) & (x > 0.0)
    dropped: FloatArray = x[~keep]
    if warn and dropped.size != 0:
        prefix = f"{label}: " if label else ""
        warnings.warn(
            f"{prefix}dropped {dropped.size} value(s) that are not finite positive ratios: "
            f"{np.round(dropped, 6).tolist()}",
            stacklevel=2,
        )
    return x[keep]


class Unusable(enum.Enum):
    """What :func:`geomean` does with a zero, negative or non-finite entry."""

    RAISE = enum.auto()
    DROP = enum.auto()


def geomean(values: Samples, unusable: Unusable = Unusable.RAISE) -> float:
    """Geometric mean of strictly positive ``values``, in log space. :attr:`Unusable.RAISE` raises on an empty set or
    a non-positive entry; :attr:`Unusable.DROP` drops zero, negative and non-finite entries, and NaN when none is left."""
    x: FloatArray = np.asarray(values, dtype=np.float64)
    if unusable == Unusable.DROP:
        x = x[np.isfinite(x) & (x > 0.0)]
        if x.size == 0:
            return math.nan
    if x.size == 0:
        raise ValueError("the geometric mean of no values is undefined")
    bad: FloatArray = x[~(np.isfinite(x) & (x > 0.0))]
    if bad.size != 0:
        raise ValueError(f"every value must be finite and strictly positive; got {bad[:4].tolist()}")
    return float(np.exp(math.fsum(np.log(x).tolist()) / x.size))


def geomean_ci(values: Samples, confidence: float | None = None) -> Interval:
    """Geometric mean and its BCa bootstrap interval, resampled in LOG space (ends asymmetric about the point);
    fewer than 3 values or no spread is the point alone."""
    confidence = 1.0 - default_alpha() if confidence is None else confidence
    x: FloatArray = np.asarray(values, dtype=np.float64)
    point = geomean(x)
    logs = bootstrap_ci(np.log(x), np.mean, name="geomean", confidence=confidence)
    return Interval("geomean", point, math.exp(logs.low), math.exp(logs.high), confidence, logs.method, int(x.size))


def geomean_interval(values: Samples, confidence: float | None = None) -> Interval:
    """The geometric mean and the interval every figure and table draws for it: :func:`geomean_ci`'s
    bootstrap interval, withheld (NaN ends, ``method`` ``"underpowered"``) below
    :data:`MIN_PAIRS_FOR_INTERVAL` values. An empty set is all NaN."""
    confidence = 1.0 - default_alpha() if confidence is None else confidence
    x: FloatArray = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return Interval("geomean", math.nan, math.nan, math.nan, confidence, "underpowered", 0)
    if x.size < MIN_PAIRS_FOR_INTERVAL:
        return Interval("geomean", geomean(x), math.nan, math.nan, confidence, "underpowered", int(x.size))
    return geomean_ci(x, confidence=confidence)


def mean_ci(values: Samples, confidence: float | None = None) -> Interval:
    """Arithmetic mean and its BCa bootstrap interval (a COST summary, SC15 Rule 3); fewer than 3 values or no spread
    is the point alone."""
    confidence = 1.0 - default_alpha() if confidence is None else confidence
    return bootstrap_ci(np.asarray(values, dtype=np.float64), np.mean, name="mean", confidence=confidence)


def mean_interval(values: Samples, confidence: float | None = None) -> Interval:
    """:func:`mean_ci`, withheld below :data:`MIN_PAIRS_FOR_INTERVAL` values exactly as
    :func:`geomean_interval` is. An empty set is all NaN."""
    confidence = 1.0 - default_alpha() if confidence is None else confidence
    x: FloatArray = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return Interval("mean", math.nan, math.nan, math.nan, confidence, "underpowered", 0)
    if x.size < MIN_PAIRS_FOR_INTERVAL:
        return Interval("mean", float(np.mean(x)), math.nan, math.nan, confidence, "underpowered", int(x.size))
    return mean_ci(x, confidence=confidence)


def signed_change(ratio: float) -> float:
    """Speedup ratio -> signed relative change: ``r - 1`` for ``r >= 1``, ``-(1/r - 1)`` below, so
    ``signed_change(1/r) == -signed_change(r)``. Not a finite positive ratio -> NaN, never 0.0."""
    if not math.isfinite(ratio) or ratio <= 0.0:
        return math.nan
    return ratio - 1.0 if ratio >= 1.0 else -(1.0 / ratio - 1.0)


def log2_change(ratio: float) -> float:
    """Speedup ratio -> its base-2 logarithm (same zero and sign as :func:`signed_change`, one unit per doubling);
    not a finite positive ratio -> NaN."""
    if not math.isfinite(ratio) or ratio <= 0.0:
        return math.nan
    return math.log2(ratio)


class Signs(NamedTuple):
    """How many paired changes favour the first setup, the second, and neither."""

    wins: int
    losses: int
    ties: int


def signs(differences: Samples) -> Signs:
    """The :class:`Signs` of the finite ``differences``: a win is above 0, a loss below, a tie exactly 0."""
    x: FloatArray = np.asarray(differences, dtype=np.float64)
    x = x[np.isfinite(x)]
    return Signs(int(np.count_nonzero(x > 0.0)), int(np.count_nonzero(x < 0.0)), int(np.count_nonzero(x == 0.0)))


#: Up to this many pairs the sign-flip test enumerates every sign vector (2**16 = 65536 of them).
SIGN_FLIP_EXACT_MAX_N: int = 16

#: Above :data:`SIGN_FLIP_EXACT_MAX_N`, the seeded sign vectors drawn (plus the identity).
SIGN_FLIP_DRAWS: int = 19_999


def sign_flips(n: int, seed: int = 0) -> FloatArray:
    """The sign vectors of a sign-flip test on ``n`` pairs, one per row, the identity first: every one of
    the ``2**n`` up to :data:`SIGN_FLIP_EXACT_MAX_N`, else the identity and :data:`SIGN_FLIP_DRAWS` drawn
    with ``default_rng(seed)``."""
    if n <= SIGN_FLIP_EXACT_MAX_N:
        bits: npt.NDArray[np.int64] = (np.arange(2**n, dtype=np.int64)[:, None] >> np.arange(n, dtype=np.int64)) & 1
        return np.where(bits == 1, -1.0, 1.0)
    drawn = np.random.default_rng(seed).choice((-1.0, 1.0), size=(SIGN_FLIP_DRAWS, n))
    return np.vstack([np.ones((1, n)), drawn])


def sign_flip_pvalue(differences: Samples, flips: FloatArray) -> float:
    """Two-sided sign-flip p of the mean of paired ``differences``: the share of sign vectors whose flipped mean is
    at least as far from 0 as the observed one."""
    x: FloatArray = np.asarray(differences, dtype=np.float64)
    observed = abs(float(np.mean(x)))
    flipped = np.abs(flips @ x) / x.size
    return float(np.mean(flipped >= observed - 1e-12 * max(observed, 1.0)))


def sign_flip_interval(differences: Samples, flips: FloatArray, alpha: float | None = None) -> tuple[float, float]:
    """The shifts ``delta`` the sign-flip test does NOT reject at ``alpha`` on ``differences - delta``, those with a p
    of at least ``alpha`` (the interval that inverts :func:`sign_flip_pvalue`, so it excludes 0 exactly when that p is
    below ``alpha``), found by bisection; ends hold within the p comparison's 1e-12 tolerance. Every sign vector's
    indicator is monotone in ``delta`` on each side of the mean, so the shifts not rejected form one interval."""
    alpha = default_alpha() if alpha is None else alpha
    x: FloatArray = np.asarray(differences, dtype=np.float64)
    centre = float(np.mean(x))
    span = float(np.ptp(x))

    def edge(outer: float) -> float:
        inner = centre
        for _ in range(60):
            middle = (inner + outer) / 2.0
            if sign_flip_pvalue(x - middle, flips) >= alpha:
                inner = middle
            else:
                outer = middle
        return inner

    return edge(float(np.min(x)) - span), edge(float(np.max(x)) + span)
