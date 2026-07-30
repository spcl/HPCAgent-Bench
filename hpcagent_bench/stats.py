# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Non-parametric statistics for measurement samples: robust outlier rejection and a
median confidence interval.

Timing samples are right-skewed -- a run is never faster than the hardware minimum, but an
OS hiccup can make a single run arbitrarily slow. So we summarize with the MEDIAN and a
non-parametric bootstrap CI (``scipy.stats.bootstrap``) rather than a mean +/- std, and we
first drop only the VERY bad upper outliers with a robust (median / MAD) rule that a lone
10x sample cannot mask. Every drop is warned about -- a silently discarded sample would
read as clean data.

Reported defaults (so a run's rigor is documented, not implicit):
* outlier rule -- modified z-score ``(x - median) / (1.4826 * MAD)``, upper tail only,
  threshold :data:`DEFAULT_MAD_Z` (5.0);
* CI -- :func:`scipy.stats.bootstrap` of the median, ``confidence_level``
  :data:`DEFAULT_CONFIDENCE` (0.95), ``n_resamples`` :data:`DEFAULT_RESAMPLES` (9999),
  ``method`` :data:`DEFAULT_CI_METHOD` (``"percentile"`` -- the robust choice for a median,
  whose BCa acceleration estimate is unstable).
"""
import warnings
from typing import Sequence, Tuple

import numpy as np
from scipy.stats import bootstrap

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


def drop_outliers(samples: Sequence[float],
                  threshold: float = DEFAULT_MAD_Z,
                  warn: bool = True,
                  label: str = "") -> Tuple[np.ndarray, np.ndarray]:
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
    x = np.asarray(samples, dtype=float)
    empty = np.empty(0, dtype=float)
    if x.size < 3:
        return x, empty
    med = float(np.median(x))
    abs_dev = np.abs(x - med)
    scale = float(np.median(abs_dev)) * _MAD_TO_SIGMA
    if scale == 0.0:
        # >= half the samples equal the median, so MAD is 0 and the modified z is undefined.
        # Fall back to the mean absolute deviation about the median (Iglewicz-Hoaglin) so a
        # clear outlier above an otherwise-constant cluster is still caught.
        scale = float(np.mean(abs_dev)) * _MEANAD_TO_SIGMA
    if scale == 0.0:
        return x, empty  # truly all identical: no robust scale, nothing to flag
    modified_z = (x - med) / scale
    drop_mask = modified_z > threshold  # upper (slow) tail only
    kept, dropped = x[~drop_mask], x[drop_mask]
    if warn and dropped.size:
        prefix = f"{label}: " if label else ""
        warnings.warn(
            f"{prefix}dropped {dropped.size} slow outlier sample(s) "
            f"(modified z > {threshold}, median={med:.4g}): {np.round(dropped, 4).tolist()}",
            stacklevel=2)
    return kept, dropped


def median_ci(samples: Sequence[float],
              confidence: float = DEFAULT_CONFIDENCE,
              n_resamples: int = DEFAULT_RESAMPLES,
              method: str = DEFAULT_CI_METHOD,
              drop: bool = True,
              warn: bool = True,
              label: str = "",
              seed: int = 0) -> Tuple[float, float, float, int]:
    """Median and a non-parametric bootstrap CI, after robust outlier rejection.

    Runs :func:`scipy.stats.bootstrap` on the median with the module defaults
    (``method='percentile'``, ``confidence_level=0.95``, ``n_resamples=9999``). Returns
    ``(median, ci_low, ci_high, n_dropped)``. With ``drop`` the upper outliers are removed
    first (:func:`drop_outliers`, which warns). A point CI ``(m, m, m)`` is returned when
    there is no spread or too few samples to bootstrap.
    """
    x = np.asarray(samples, dtype=float)
    n_dropped = 0
    if drop:
        x, dropped = drop_outliers(x, warn=warn, label=label)
        n_dropped = int(dropped.size)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan"), n_dropped
    med = float(np.median(x))
    if x.size < 3 or float(np.ptp(x)) == 0.0:
        return med, med, med, n_dropped
    res = bootstrap((x, ),
                    np.median,
                    confidence_level=confidence,
                    n_resamples=n_resamples,
                    method=method,
                    vectorized=True,
                    random_state=np.random.default_rng(seed))
    return med, float(res.confidence_interval.low), float(res.confidence_interval.high), n_dropped
