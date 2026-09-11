# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Wilcoxon signed-rank null: the one rule deciding exact vs approximate, and a stdlib
implementation of the exact distribution.

TWO IMPLEMENTATIONS, ONE RULE. This repo computes the signed-rank p twice, and it has to: the
figures take it from scipy, and ``experiments/ablation_stats.py`` is deliberately stdlib-only
because it runs on a login node from a shell that never activated the benchmark environment. Two
implementations are fine. Two independently chosen CUTOFFS are not -- one module switched to the
normal approximation above n = 25 while the other inherited scipy's ``auto`` heuristic and stayed
exact through n = 50, so the same test on the same 40 kernels returned p = 0.18329 in one published
table and p = 0.18762 in another, with the approximation on the anti-conservative side.

So :data:`EXACT_MAX_N` and :func:`use_exact` live here, and both paths obey them. The stdlib module
loads this file BY PATH (stdlib ``importlib``), which needs no environment and no third-party
import; :mod:`hpcagent_bench.stats.summary` imports it normally and passes scipy the method this
rule chose, EXPLICITLY, so a scipy release cannot move the cutoff under us.
``tests/test_signed_rank.py`` proves the two implementations agree wherever both are exact.

THE CUTOFF IS MEASURED, not inherited. The exact null is a subset-sum count over the ranks
``1..n``: ``n * n(n+1)/2`` states of unbounded-integer addition. Timed in this interpreter:

    n         25     40     50    100    150     200     300     578
    DP     0.5ms  3.0ms  6.5ms   58ms  200ms   474ms   1.60s  11.79s

and the continuity-corrected normal approximation's worst absolute error against the exact p, over
effects spanning p = 0.001 to 0.9:

    n          25       40      100      200      578
    max|dp|  6.6e-3   4.1e-3   1.7e-3   8.3e-4   2.9e-4

200 is where the DP stops being free -- it is the last size under half a second, and the cost grows
as n^3 with big-integer coefficients past it. It also covers every paired-kernel count these tables
reach: the llr focus roster is 40 kernels and the largest campaign roster is 242 problems, of which
a PAIR covers fewer. Above 200 the approximation is within 8.3e-4 absolute of the exact p, which is
a fifth of the 4.3e-3 discrepancy that made this rule necessary, and it keeps shrinking.

The count for one ``n`` is cached, so a table comparing many arm pairs at the same ``n`` pays the
DP once.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence

#: Sample sizes up to this get the EXACT null; above it the tie-corrected normal approximation.
#: Measured, not inherited -- see the module docstring for the timings and the error curve. Both
#: implementations read this one name, so the cutoff cannot drift apart again.
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
    """THE RULE. Is the exact null both affordable and VALID for these ``|d|`` values?

    Two conditions, and the second is the one a size check alone misses. The exact distribution
    counts subsets of the DISTINCT ranks ``1..n``; with a tie the ranks are midranks, the lattice
    the count is over no longer holds, and the resulting p is wrong rather than merely imprecise.
    Neither implementation may claim exactness there, so a tied sample takes the tie-corrected
    normal approximation in both.
    """
    n = len(absolute)
    return 0 < n <= EXACT_MAX_N and len(set(absolute)) == n


@functools.lru_cache(maxsize=64, typed=True)
def null_counts(n: int) -> tuple[int, ...]:
    """How many of the ``2**n`` sign assignments give each possible ``W+``, by subset-sum DP.

    Under the null every rank 1..n is added to ``W+`` or not with probability 1/2 independently, so
    the exact distribution is the number of subsets of ``{1..n}`` summing to each total. Cached by
    ``n``: a pairs table comparing many arms at one kernel count pays this once.
    """
    counts = [0] * (n * (n + 1) // 2 + 1)
    counts[0] = 1
    for rank in range(1, n + 1):
        for total in range(len(counts) - 1, rank - 1, -1):
            counts[total] += counts[total - rank]
    return tuple(counts)


def exact_p(statistic: float, n: int) -> float:
    """Two-sided exact p for the signed-rank statistic ``min(W+, W-)`` at sample size ``n``.

    ``statistic`` is rounded UP: it can land half way between two lattice points, and rounding up
    is the conservative choice (a larger p) rather than one that could manufacture significance.
    """
    counts = null_counts(n)
    cutoff = min(len(counts) - 1, math.ceil(statistic - 1e-12))
    return min(1.0, 2.0 * sum(counts[: cutoff + 1]) / (2**n))


def normal_p(w_plus: float, n: int, absolute: Sequence[float]) -> float:
    """Two-sided normal-approximation p, tie-corrected on the variance and continuity-corrected on
    the deviation.

    Tied ``|d|`` values share a midrank, which makes ``W+`` less variable than the tie-free formula
    assumes; without the correction the test would be anti-conservative exactly on the data where
    ties are common (many kernels landing on the same speedup).

    ``W+`` is a lattice variable of spacing 1 and the normal density is continuous, so the tail it
    stands in for runs to the lattice point's outer EDGE: the deviation loses the half step. Both
    reference implementations subtract it (scipy ``correction=True``, R ``wilcox.test`` correct) and
    :mod:`hpcagent_bench.stats.summary` asks scipy for it, so the two paths stay one test. Measured
    over the whole lattice at exact p <= 0.10, it cuts the worst anti-conservative gap against the
    exact null by 5x to 11x: 1.7e-3 to 3.2e-4 at n = 40, 2.0e-4 to 7.5e-5 at n = 200.
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
