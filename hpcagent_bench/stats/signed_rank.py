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

and the normal approximation's worst absolute error against the exact p, over effects spanning
p = 0.001 to 0.9:

    n          25       40      100      200      578
    max|dp|  1.5e-2   8.4e-3   2.7e-3   1.2e-3   3.4e-4

200 is where the DP stops being free -- it is the last size under half a second, and the cost grows
as n^3 with big-integer coefficients past it. It also covers every paired-kernel count these tables
reach: the llr focus roster is 40 kernels and the largest campaign roster is 242 problems, of which
a PAIR covers fewer. Above 200 the approximation is within 1.2e-3 absolute of the exact p, which is
a quarter of the 4.3e-3 discrepancy that made this rule necessary, and it keeps shrinking.

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


def standard_normal_cdf(z: float) -> float:
    """``P(Z <= z)`` for the standard normal, from ``math.erfc`` (stdlib only, by module policy)."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def standard_normal_pdf(z: float) -> float:
    """The standard normal density at ``z``."""
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def normal_p(w_plus: float, n: int, absolute: Sequence[float]) -> float:
    """Two-sided normal-approximation p for ``W+``: CONTINUITY-CORRECTED, kurtosis-corrected, and
    conservative by construction.

    Three things a plain ``erfc(|W+ - mean| / sqrt(2 var))`` gets wrong, in the order they bite.

    TIES. Tied ``|d|`` values share a midrank, which makes ``W+`` less variable than the tie-free
    formula assumes. Every cumulant here is summed over the ACTUAL midranks, so the tie correction
    is not a bolt-on: ``sum(r^2)/4`` is identically ``n(n+1)(2n+1)/24 - sum(t^3 - t)/48``, and the
    same summation extends the correction to the fourth cumulant, which no closed form covers.

    CONTINUITY. ``W+`` lives on a lattice. The exact two-sided p is ``2 P(W+ <= w)``, whose normal
    image is the half-line up to the cell EDGE ``w + 1/2``, so the deviation carried into the tail
    is ``|W+ - mean| - 1/2``. Omitting that half step reports a smaller p than the exact null at
    every n -- 2.8e-3 too small at n = 35, still 2.9e-4 too small at n = 210.

    KURTOSIS. The signed-rank null is LIGHT-tailed: its fourth cumulant ``-sum(r^4)/8`` is
    negative, so the normal understates ``P(W+ <= w)`` through the whole shoulder ``|z| < sqrt(3)``
    -- which is where p = 0.05 to 0.15 lands, exactly the range a significance claim turns on. The
    Edgeworth term ``phi(z) (gamma2/24) He3(z)`` removes that O(1/n) error.

    The Edgeworth series is truncated, and its remainder is not signed, so the magnitude of the
    last retained term is ADDED as the truncation guard. That is what makes the result conservative
    rather than merely accurate: the reported p is never below the exact null's at any n these
    tables reach, and it costs at most 8.1e-4 of excess p at n = 35, shrinking as 1/n.
    """
    ranks = average_ranks(absolute)
    mean = math.fsum(ranks) / 2.0
    variance = math.fsum(rank * rank for rank in ranks) / 4.0
    if variance <= 0.0:
        return 1.0
    kurtosis = -math.fsum(rank**4 for rank in ranks) / (8.0 * variance * variance)
    z = -max(0.0, abs(w_plus - mean) - 0.5) / math.sqrt(variance)
    edgeworth = standard_normal_pdf(z) * (kurtosis / 24.0) * (z * z * z - 3.0 * z)
    return min(1.0, max(0.0, 2.0 * (standard_normal_cdf(z) - edgeworth + abs(edgeworth))))


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
