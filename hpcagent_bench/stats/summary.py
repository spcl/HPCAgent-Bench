# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Every statistic a figure or a table in this repo reports, defined exactly once.

A number with two definitions is a number nobody can check. Seven copies of the geometric mean
agreed on the arithmetic and disagreed on what an empty set or a zero ratio means, which is the
failure mode that does not show up as a wrong plot -- it shows up as two plots of the same data
that do not match.

WHAT LIVES HERE. Robust outlier rejection, the median and its bootstrap interval, the geometric
mean and its log-space interval, the signed-change axis transform, the one-value-per-kernel
reduction, and the paired Hodges-Lehmann estimate with its signed-rank test. :mod:`..inference`
sits on top of this for hypothesis testing across a corpus; it imports from here and never the
other way round.

Timing samples are right-skewed -- a run is never faster than the hardware minimum, but an OS
hiccup can make a single run arbitrarily slow. So a sample is summarized with the MEDIAN and a
non-parametric bootstrap CI (:func:`scipy.stats.bootstrap`) rather than a mean +/- std, and the
VERY bad upper outliers are dropped first by a robust (median / MAD) rule that a lone 10x sample
cannot mask. Every drop is warned about -- a silently discarded sample reads as clean data.

A set of RATIOS is summarized with the geometric mean, and Hoefler et al. (SC15) Rule 4 applies:
the ratio is never the only thing reported. See :mod:`hpcagent_bench.stats.rules`.

Reported defaults (so a run's rigor is documented, not implicit):
* outlier rule -- modified z-score ``(x - median) / (1.4826 * MAD)``, upper tail only,
  threshold :data:`DEFAULT_MAD_Z` (5.0);
* CI -- :func:`scipy.stats.bootstrap` of the median, ``confidence_level``
  :data:`DEFAULT_CONFIDENCE` (0.95), ``n_resamples`` :data:`DEFAULT_RESAMPLES` (9999),
  ``method`` :data:`DEFAULT_CI_METHOD` (``"percentile"`` -- the robust choice for a median,
  whose BCa acceleration estimate is unstable);
* paired test -- Wilcoxon signed-rank, exact or approximate by the ONE rule in
  :mod:`hpcagent_bench.stats.signed_rank`, whose threshold ``experiments/ablation_stats.py``
  obeys too; the method is passed to scipy explicitly rather than left to its ``auto`` heuristic.
"""

from __future__ import annotations
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from hpcagent_bench.stats import signed_rank

# scipy and pandas are imported INSIDE the three functions that need them, not here. The grading
# path takes its geometric mean from this module and already pays for numpy; making it pay for
# scipy as well would put a second of import into every judge process to reach ten lines of
# arithmetic. pandas is only ever an annotation here, so it never loads at runtime at all.
if TYPE_CHECKING:
    import pandas as pd

#: One timing sample per element. float64 is what ``np.asarray(..., dtype=float)`` produces.
FloatArray = npt.NDArray[np.float64]

#: What both entry points accept: a plain sequence of numbers, or an already-built float array.
Samples = Sequence[float] | FloatArray

#: MAD -> normal-sigma consistency constant (1 / 0.6745). A modified z-score of ``k`` is
#: ``k`` robust standard deviations above the median.
_MAD_TO_SIGMA: float = 1.4826

#: Mean-absolute-deviation -> normal-sigma constant (sqrt(pi/2)). Used only as the fallback
#: robust scale when MAD == 0 (>= half the samples identical) -- Iglewicz-Hoaglin.
_MEANAD_TO_SIGMA: float = 1.253314

#: Default upper modified-z threshold. 5 keeps the median plus a ~5 robust-sigma slow tail --
#: "very bad" (OS-hiccup) samples, not ordinary run-to-run jitter.
DEFAULT_MAD_Z: float = 5.0

#: Bootstrap median-CI defaults, exposed so callers and docs can report them.
DEFAULT_CONFIDENCE: float = 0.95
DEFAULT_RESAMPLES: int = 9999
DEFAULT_CI_METHOD: str = "percentile"


def drop_outliers(
    samples: Samples, threshold: float = DEFAULT_MAD_Z, warn: bool = True, label: str = ""
) -> tuple[FloatArray, FloatArray]:
    """Drop upper-tail outliers by robust modified z-score, one-sided (slow side only).

    ``modified_z = (x - median) / (1.4826 * MAD)``; a sample with ``modified_z > threshold``
    is dropped. Using the median and MAD (median absolute deviation) makes the rule immune
    to the very outliers it removes -- a lone 10x sample cannot inflate the scale the way a
    mean / std would. Only the SLOW side is trimmed: a timing sample is never faster than the
    hardware minimum, so a low value is real signal, not a hiccup.

    Returns ``(kept, dropped)`` as float arrays. When ``warn`` and anything is dropped, emits
    a :class:`UserWarning` naming the count and the dropped values -- a dropped sample must
    never vanish silently. ``label`` (e.g. ``"<kernel>@<framework>"``) prefixes the warning.
    Fewer than 3 samples, or a degenerate spread (MAD == 0, i.e. at least half identical),
    yield no drops.
    """
    x: FloatArray = np.asarray(samples, dtype=np.float64)
    empty: FloatArray = np.empty(0, dtype=np.float64)
    if x.size < 3:
        return x, empty
    med = float(np.median(x))
    abs_dev: FloatArray = np.abs(x - med)
    scale = float(np.median(abs_dev)) * _MAD_TO_SIGMA
    if scale == 0.0:
        # >= half the samples equal the median, so MAD is 0 and the modified z is undefined.
        # Fall back to the mean absolute deviation about the median (Iglewicz-Hoaglin) so a
        # clear outlier above an otherwise-constant cluster is still caught.
        scale = float(np.mean(abs_dev)) * _MEANAD_TO_SIGMA
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
) -> tuple[float, float, float, int]:
    """Median and a non-parametric bootstrap CI, after robust outlier rejection.

    Runs :func:`scipy.stats.bootstrap` on the median with the module defaults
    (``method='percentile'``, ``confidence_level=0.95``, ``n_resamples=9999``). Returns
    ``(median, ci_low, ci_high, n_dropped)``. With ``drop`` the upper outliers are removed
    first (:func:`drop_outliers`, which warns). A point CI ``(m, m, m)`` is returned when
    there is no spread or too few samples to bootstrap.
    """
    x: FloatArray = np.asarray(samples, dtype=np.float64)
    n_dropped = 0
    if drop:
        x, dropped = drop_outliers(x, warn=warn, label=label)
        n_dropped = int(dropped.size)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan"), n_dropped
    med = float(np.median(x))
    if x.size < 3 or float(np.ptp(x)) == 0.0:
        return med, med, med, n_dropped
    from scipy.stats import bootstrap  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

    res = bootstrap(  # pyright: ignore[reportUnknownVariableType] -- unstubbed scipy
        (x,),
        np.median,
        confidence_level=confidence,
        n_resamples=n_resamples,
        method=method,
        vectorized=True,
        random_state=np.random.default_rng(seed),  # pyright: ignore[reportCallIssue] -- scipy compat spelling
    )
    low = float(res.confidence_interval.low)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    high = float(res.confidence_interval.high)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    return med, low, high, n_dropped


#: Two-sided error rate every interval and test here defaults to.
DEFAULT_ALPHA: float = 0.05

#: Below this a two-sided signed-rank test cannot reach ``DEFAULT_ALPHA`` whatever the data says
#: (its smallest attainable p at n = 5 is 0.0625), so an interval would be decoration.
MIN_PAIRS_FOR_INTERVAL: int = 6


@dataclass(frozen=True, slots=True)
class Interval:
    """A confidence interval and, critically, WHAT IT IS FOR. An interval around the mean is not
    an interval around the min-of-k; ``statistic`` keeps the two from being confused in a table
    or a figure caption."""

    statistic: str  # "mean" | "median" | "geomean" | "min_of_k" | "speedup(min_of_k)" | ...
    point: float
    low: float
    high: float
    confidence: float
    method: str  # "t" | "log-t" | "bootstrap-BCa" | "bootstrap-percentile" | "rank-median" | ...
    n: int

    def label(self) -> str:
        """One-line figure/table label naming both the statistic and the interval kind."""
        return f"{int(round(self.confidence * 100))}% {self.method} CI for {self.statistic}"


@dataclass(frozen=True, slots=True)
class PairedChange:
    """The paired per-kernel change: one estimate, one interval and one p value that agree.

    The Hodges-Lehmann estimator is the location the signed-rank test inverts, so the point, the
    interval and the p value all describe the same quantity. A bootstrap mean beside a rank test
    does not: the two can disagree about which arm is ahead, and a reader cannot tell which to
    believe.
    """

    estimate: float
    low: float
    high: float
    pvalue: float
    n: int
    wins: int
    losses: int
    ties: int
    method: str  # "signed-rank-exact" | "signed-rank-approx" | "underpowered" | "degenerate"

    def interval(self, name: str, confidence: float = 1.0 - DEFAULT_ALPHA) -> Interval:
        """The same estimate as an :class:`Interval`, for a figure that draws one."""
        return Interval(name, self.estimate, self.low, self.high, confidence, self.method, self.n)


def usable_ratios(values: Samples, label: str = "", warn: bool = True) -> FloatArray:
    """The entries of ``values`` a geometric mean may be taken over: finite and strictly positive.

    A zero, a negative or a non-finite ratio is a MISSING measurement, not a slow one. Dropping it
    is the only defensible reading, and clamping it to a small epsilon -- which one copy of this
    did by way of :func:`scipy.stats.gmean`, whose ``log(0)`` sends the whole geomean to 0.0 --
    enters an absent datum as a catastrophic regression that never happened.

    Every drop is warned about, naming the count and the values. ``label`` prefixes the warning.
    """
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


def geomean(values: Samples) -> float:
    """Geometric mean of strictly positive ``values``, in log space so a long product cannot overflow.

    Raises on an empty sequence or a non-positive entry rather than skipping it: both are the
    caller handing over something that is not a set of ratios, and dropping one silently changes
    which kernels the summary is over without saying so. Filter with :func:`usable_ratios` first
    when dropping IS the intent; it warns.
    """
    x: FloatArray = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        raise ValueError("the geometric mean of no values is undefined")
    bad: FloatArray = x[~(np.isfinite(x) & (x > 0.0))]
    if bad.size != 0:
        raise ValueError(f"every value must be finite and strictly positive; got {bad[:4].tolist()}")
    return float(np.exp(math.fsum(np.log(x).tolist()) / x.size))


def geomean_ci(values: Samples, confidence: float = 1.0 - DEFAULT_ALPHA) -> Interval:
    """Geometric mean and its Student-t interval, computed in LOG space and mapped back.

    The ends come back as ratios, not as a half-width: ``exp`` is not linear, so a symmetric
    ``+/-`` would be wrong on a ratio axis and on the signed-change axis alike. A single
    observation has no spread to estimate, so its interval is the point itself.
    """
    x: FloatArray = np.asarray(values, dtype=np.float64)
    point = geomean(x)
    if x.size < 2:
        return Interval("geomean", point, point, point, confidence, "log-t", int(x.size))
    from scipy.stats import t  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

    logs: FloatArray = np.log(x)
    centre = math.fsum(logs.tolist()) / x.size
    half = float(t.ppf(0.5 + confidence / 2.0, x.size - 1)) * float(np.std(logs, ddof=1)) / math.sqrt(x.size)
    return Interval(
        "geomean", point, math.exp(centre - half), math.exp(centre + half), confidence, "log-t", int(x.size)
    )


def signed_change(ratio: float) -> float:
    """Speed-up ratio -> signed relative change. ``2x -> +1``, ``1x -> 0``, ``0.5x -> -1``.

    ``r >= 1`` maps to ``r - 1`` and ``r < 1`` to ``-(1/r - 1)``, so a 2x win (+1) and a 2x
    slow-down (-1) sit the same distance from 0 and the transform is odd about no change:
    ``signed_change(1/r) == -signed_change(r)`` exactly. A raw ratio axis crushes every slow-down
    into the 0..1 sliver and gives every speed-up an unbounded tail, so the eye reads a 0.5x
    regression as a SMALLER event than a 1.5x win when they are the same magnitude.

    Anything that is not a finite POSITIVE ratio -- 0, negative, +/-inf, NaN, a cell that was
    never measured -- returns NaN, never 0.0: 0 is the exact value of "measured, and nothing
    changed", and an absent measurement must not be able to claim it.
    """
    if not math.isfinite(ratio) or ratio <= 0.0:
        return math.nan
    return ratio - 1.0 if ratio >= 1.0 else -(1.0 / ratio - 1.0)


def signed_changes(ratios: Samples) -> FloatArray:
    """:func:`signed_change` over an array, NaN where a ratio is not plottable."""
    x: FloatArray = np.asarray(ratios, dtype=np.float64)
    out: FloatArray = np.full(x.shape, np.nan, dtype=np.float64)
    good: npt.NDArray[np.bool_] = np.isfinite(x) & (x > 0.0)
    out[good] = np.where(x[good] >= 1.0, x[good] - 1.0, -(1.0 / x[good] - 1.0))
    return out


def median_per_kernel(
    frame: pd.DataFrame, value: str, kernel: str = "benchmark", within: Sequence[str] = ()
) -> pd.Series:
    """One value per kernel -- the unit every corpus-level statistic here is taken over.

    A summary must never be pooled over RAW rows. An agent that resubmits a kernel ten times
    contributes it ten times to a pooled mean, which weights a kernel by the agent's patience
    rather than by the corpus; and a kernel timed at more repetitions than its neighbours would
    outvote them for the same non-reason. ``within`` names the columns that must be reduced BEFORE
    the kernel is (``run_id`` for a per-episode quantity such as a token count, where the episode
    total is a max over its rows rather than a median of them).
    """
    if within:
        frame = frame.groupby([kernel, *within], as_index=False)[value].max()
    return frame.groupby(kernel)[value].median()


def walsh_averages(values: Samples) -> FloatArray:
    """Sorted ``(v_i + v_j) / 2`` for ``i <= j`` -- what the Hodges-Lehmann estimate is a median of."""
    x: FloatArray = np.asarray(values, dtype=np.float64)
    i, j = np.triu_indices(x.size, k=0)
    return np.sort((x[i] + x[j]) / 2.0)


def hodges_lehmann(values: Samples) -> float:
    """The Hodges-Lehmann point estimate: the median of every Walsh average ``(v_i + v_j) / 2``.

    The location estimate the signed-rank test is consistent with, and robust where a mean is not:
    a per-kernel change is heavy-tailed (one kernel at 40x against a median near 2x), and a mean --
    even a mean in log space -- still lets that kernel carry the estimate.
    """
    return float(np.median(walsh_averages(values)))


def paired_change(differences: Samples, alpha: float = DEFAULT_ALPHA) -> PairedChange:
    """Hodges-Lehmann estimate, distribution-free interval and signed-rank p for paired ``differences``.

    ``differences`` is one number per KERNEL, already paired -- typically ``log(after / before)``,
    which makes the estimate a ratio once mapped back through ``exp`` and makes a win and its exact
    inverse cancel. Pairing is most of the precision: per-kernel spread is far larger than any
    treatment effect this repo measures, and the unpaired sibling of this test sees almost nothing
    at n = 40.

    The interval is the k-th smallest and k-th largest Walsh average, k taken from the signed-rank
    null -- no normality assumption and no resampling, so a published end point cannot move because
    a seed changed. Zero differences are dropped (Wilcoxon's original treatment): they support
    neither direction, and keeping them would inflate n and shrink the p value for free.

    The ESTIMATOR DOES NOT CHANGE WITH n. Below :data:`MIN_PAIRS_FOR_INTERVAL` the interval and the
    p value are withheld and ``method`` says ``underpowered``, but the point is still the
    Hodges-Lehmann estimate. Switching to a plain median down there -- which one caller did -- makes
    the marks on one figure two different statistics, and the reader is told which only by counting
    the kernels behind each row.
    """
    x: FloatArray = np.asarray(differences, dtype=np.float64)
    x = x[np.isfinite(x)]
    wins, losses = int(np.count_nonzero(x > 0.0)), int(np.count_nonzero(x < 0.0))
    ties = int(np.count_nonzero(x == 0.0))
    nonzero: FloatArray = x[x != 0.0]
    n = int(nonzero.size)
    if n == 0:
        return PairedChange(0.0, math.nan, math.nan, 1.0, 0, wins, losses, ties, "degenerate")
    point = hodges_lehmann(nonzero)
    if n < MIN_PAIRS_FOR_INTERVAL:
        return PairedChange(point, math.nan, math.nan, math.nan, n, wins, losses, ties, "underpowered")
    from scipy.stats import (  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
        norm,
        wilcoxon,
    )

    exact = signed_rank.use_exact(np.abs(nonzero).tolist())
    method = "signed-rank-exact" if exact else "signed-rank-approx"
    # The method is passed EXPLICITLY. scipy's "auto" is a library default that has moved before and
    # can move again on a bump, and the moment it does this path stops agreeing with the stdlib one.
    # correction=True for the same reason: scipy defaults it OFF, the stdlib normal_p applies the
    # half-step, and the two would report different p on every tied sample without it.
    result = wilcoxon(nonzero, method="exact" if exact else "approx", zero_method="wilcox", correction=True)
    walsh = walsh_averages(nonzero)
    mean = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = float(norm.ppf(1.0 - alpha / 2.0))
    cutoff = min(max(math.floor(mean - z * sd), 0), walsh.size // 2 - 1)
    low, high = float(walsh[cutoff]), float(walsh[walsh.size - 1 - cutoff])
    return PairedChange(point, low, high, float(result.pvalue), n, wins, losses, ties, method)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
