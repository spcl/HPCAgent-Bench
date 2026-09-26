# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Multiplicity corrections for a family of p-values."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.stats import false_discovery_control  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

FloatArray = npt.NDArray[np.float64]


def holm_bonferroni(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (input order preserved).

    Controls the family-wise error rate (probability of any false claim across the family).
    """
    p: FloatArray = np.asarray(pvalues, dtype=np.float64)
    n = int(p.size)
    if n == 0:
        return []
    order = np.argsort(p, kind="stable")
    adjusted: FloatArray = np.minimum(1.0, (n - np.arange(n)) * p[order])
    adjusted = np.maximum.accumulate(adjusted)  # step-down monotonicity
    out: FloatArray = np.empty(n, dtype=np.float64)
    out[order] = adjusted
    return [float(v) for v in out]


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg FDR-adjusted p-values (input order preserved)."""
    p: FloatArray = np.asarray(pvalues, dtype=np.float64)
    if p.size == 0:
        return []
    return [float(v) for v in false_discovery_control(p, method="bh")]


def adjust_pvalues(pvalues: Sequence[float], method: str = "fdr_bh") -> list[float]:
    """Multiplicity-adjusted p-values: ``fdr_bh`` (default, screening power) or ``holm`` (strict
    family-wise control). An unadjusted per-kernel p-value over a large corpus is not a finding.
    """
    if method == "holm":
        return holm_bonferroni(pvalues)
    if method == "fdr_bh":
        return benjamini_hochberg(pvalues)
    raise ValueError(f"unknown multiple-comparison method {method!r}; use 'fdr_bh' or 'holm'")
