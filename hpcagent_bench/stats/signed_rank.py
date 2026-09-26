# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Wilcoxon signed-rank null: one rule deciding exact vs approximate, and a stdlib
implementation of the exact distribution.

This repo computes the signed-rank p twice: figures take it from scipy, and
``statistics/ablation_stats.py`` is stdlib-only because it runs on a login node whose shell never
activated the benchmark environment. :data:`EXACT_MAX_N` and :func:`use_exact` are the single rule
both paths obey, so the cutoff cannot drift apart between them; :mod:`hpcagent_bench.stats.summary`
passes scipy the method this rule chose explicitly, so a scipy release cannot move it under us.
``tests/test_signed_rank.py`` checks the two implementations agree wherever both are exact.

200 is the largest ``n`` where the exact subset-sum DP stays fast (cost grows as n^3 past it) and
covers every paired-kernel count these tables reach; above it the tie-corrected normal
approximation is close enough not to matter. The count for one ``n`` is cached, so a table
comparing many arm pairs at the same ``n`` pays the DP once.
"""

import functools
import math
from collections.abc import Sequence

__all__ = ["EXACT_MAX_N", "average_ranks", "exact_p", "normal_p", "null_counts", "signed_rank_p", "use_exact"]

#: Sample sizes up to this get the exact null; above it the tie-corrected normal approximation.
EXACT_MAX_N: int = 200


def average_ranks(values: Sequence[float]) -> list[float]:
    """Ranks 1..n of ``values``, ties sharing their block's mean rank (the midrank convention the
    variance correction in :func:`normal_p` assumes)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        shared = (start + stop) / 2.0 + 1.0
        for position in range(start, stop + 1):
            ranks[order[position]] = shared
        start = stop + 1
    return ranks


def use_exact(absolute: Sequence[float]) -> bool:
    """Whether the exact null is affordable (``n <= EXACT_MAX_N``) and valid (no ties in
    ``absolute``): a tie breaks the rank lattice the exact count assumes.
    """
    n = len(absolute)
    return 0 < n <= EXACT_MAX_N and len(set(absolute)) == n


@functools.lru_cache(maxsize=64, typed=True)
def null_counts(n: int) -> tuple[int, ...]:
    """How many of the ``2**n`` sign assignments give each possible ``W+``, by subset-sum DP."""
    counts = [0] * (n * (n + 1) // 2 + 1)
    counts[0] = 1
    for rank in range(1, n + 1):
        for total in range(len(counts) - 1, rank - 1, -1):
            counts[total] += counts[total - rank]
    return tuple(counts)


def exact_p(statistic: float, n: int) -> float:
    """Two-sided exact p for the signed-rank statistic ``min(W+, W-)`` at sample size ``n``.

    ``statistic`` is rounded up, the conservative choice when it lands between two lattice points.
    """
    counts = null_counts(n)
    cutoff = min(len(counts) - 1, math.ceil(statistic - 1e-12))
    return min(1.0, 2.0 * sum(counts[: cutoff + 1]) / (2**n))


def normal_p(w_plus: float, n: int, absolute: Sequence[float]) -> float:
    """Two-sided normal-approximation p, tie-corrected on the variance and continuity-corrected on
    the deviation, matching scipy's ``correction=True`` / R's ``wilcox.test correct`` so both paths
    stay one test.
    """
    mean = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    groups: dict[float, int] = {}
    for value in absolute:
        groups[value] = groups.get(value, 0) + 1
    variance -= sum((size * size * size) - size for size in groups.values()) / 48.0
    if variance <= 0.0:
        return 1.0
    deviation = abs(abs(w_plus - mean) - 0.5)
    return min(1.0, math.erfc(deviation / math.sqrt(2.0 * variance)))


def signed_rank_p(diffs: Sequence[float]) -> tuple[int, float, str]:
    """Paired Wilcoxon signed-rank over ``diffs``; returns ``(n used, two-sided p, method)``.

    Zero differences are dropped (Wilcoxon's original treatment): they support neither direction,
    and keeping them would inflate n and shrink the p for free. Everything zero, or nothing to
    test, leaves ``n = 0``, ``p = 1`` and ``method = "degenerate"``.
    """
    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return 0, 1.0, "degenerate"
    absolute = [abs(d) for d in nonzero]
    ranks = average_ranks(absolute)
    w_plus = math.fsum(rank for rank, diff in zip(ranks, nonzero) if diff > 0.0)
    w_minus = math.fsum(ranks) - w_plus
    if use_exact(absolute):
        return n, exact_p(min(w_plus, w_minus), n), "signed-rank-exact"
    return n, normal_p(w_plus, n, absolute), "signed-rank-approx"
