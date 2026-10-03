# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Wilcoxon signed-rank test reports exact only where the exact null applies.

The same test on the same 40 kernels was once published as p = 0.18329 and p = 0.18762 because two paths used
different exact/approximate cutoffs, and the approximation is the anti-conservative side. The cutoff is one
constant (:data:`summary.EXACT_MAX_N`) and a tie always takes the corrected approximation.
"""

import numpy as np
import pytest
from scipy.stats import wilcoxon

from hpcagent_bench.stats import summary


@pytest.mark.parametrize("n", (8, 15, 26, 40, 80))
def test_a_tied_sample_takes_the_corrected_approximation(n: int) -> None:
    """The exact null counts subsets of the DISTINCT ranks 1..n; with a tie the ranks are midranks and an "exact"
    p is wrong rather than imprecise. ``correction=True`` is explicit because scipy defaults it off."""
    values = np.round(np.random.default_rng(n).normal(0.3, 1.0, n), 1).tolist()
    nonzero = [v for v in values if v != 0.0]
    absolute = [abs(v) for v in nonzero]
    assert len(set(absolute)) < len(absolute), "fixture is not tied; the property is untested"
    assert not summary.use_exact(absolute)
    _statistic, p, method, used = summary.signed_rank_test(values)
    expected = float(wilcoxon(nonzero, method="approx", zero_method="wilcox", correction=True).pvalue)
    assert (method, used) == ("signed-rank-approx", len(nonzero))
    assert p == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_the_threshold_sits_where_the_exact_null_is_still_affordable() -> None:
    """The cutoff is measured, not inherited from scipy: never under the sizes these tables reach, never past what
    the exact DP pays for."""
    assert 40 <= summary.EXACT_MAX_N, "the llr focus tag is 40 kernels and must stay exact"
    assert summary.EXACT_MAX_N <= 250
    assert summary.use_exact([float(i) for i in range(1, summary.EXACT_MAX_N + 1)])
    assert not summary.use_exact([float(i) for i in range(1, summary.EXACT_MAX_N + 2)])
