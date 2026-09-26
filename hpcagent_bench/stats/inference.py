# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Multiplicity corrections for a family of p-values."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.stats import false_discovery_control  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

FloatArray = npt.NDArray[np.float64]


def holm_bonferroni(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down ADJUSTED p-values (input order preserved).

    Controls the family-wise error rate: the probability of even ONE false claim across the
    family. Uniformly more powerful than plain Bonferroni and assumption-free."""
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
    """Benjamini-Hochberg FDR-adjusted p-values (input order preserved), via
    :func:`scipy.stats.false_discovery_control`."""
    p: FloatArray = np.asarray(pvalues, dtype=np.float64)
    if p.size == 0:
        return []
    return [float(v) for v in false_discovery_control(p, method="bh")]


def adjust_pvalues(pvalues: Sequence[float], method: str = "fdr_bh") -> list[float]:
    """Multiplicity-adjusted p-values. ``fdr_bh`` (default) or ``holm``.

    WARNING: The corpus is ~578 kernels. Testing each at alpha=0.05 manufactures ~29 false positives by
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
