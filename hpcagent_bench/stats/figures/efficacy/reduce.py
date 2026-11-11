# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The efficacy figure's DATA: pairing a setup with its control over shared kernels, the speedup and token
reductions with their intervals, the significance verdicts and the emitted table."""

import dataclasses
import math
from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd

from hpcagent_bench import study_tags
from hpcagent_bench.stats import population, summary
from hpcagent_bench.stats.significance import Conclusion, Finding

__all__ = [
    "CONCLUSION_COLUMNS",
    "COST_SIG_LABEL",
    "COST_SIG_MARK",
    "EMPTY_POINT",
    "EQUIVALENT_LABEL",
    "EQUIVALENT_MARK",
    "JOINT_SIG_LABEL",
    "NO_SIGNIFICANCE",
    "SCORE_SIG_LABEL",
    "SCORE_SIG_MARK",
    "SIG_MEASURE_NAMES",
    "SPEEDUP_OVER",
    "Series",
    "SetupPoint",
    "Significance",
    "UNCHANGED_POINT",
    "absolute_columns",
    "any_significance",
    "axis_significance",
    "drawn_symbols",
    "family_sizes",
    "interval_note",
    "leg_labels",
    "per_kernel_ci",
    "setup_groups",
    "setup_point",
    "skill_halves",
    "speedup_mask",
]

#: What a speedup aggregate is taken over. ``solved``: the kernels BOTH setups answered
#: correctly -- a wrong answer is no speedup at all, so it is counted by the success rate and not
#: scored as the baseline, and both setups are timed on the same kernels, so solving only the easy
#: ones buys no speedup. ``served``: every kernel, a failure at 1x (the fallback reading: what a
#: user who keeps the baseline on a wrong answer gets). Token cost is over every served kernel
#: either way -- a failed episode still spent them.
SPEEDUP_OVER: population.KernelPolicy = population.KernelPolicy.SOLVED


def speedup_mask(paired: pd.DataFrame, over: population.KernelPolicy = SPEEDUP_OVER) -> "np.ndarray":
    """The rows of :func:`paired_kernels`' frame a speedup aggregate is taken over."""
    if over == population.KernelPolicy.SERVED:
        return np.ones(len(paired), dtype=bool)
    return (paired.control_solved.astype(bool) & paired.treated_solved.astype(bool)).to_numpy(dtype=bool)


@dataclasses.dataclass(frozen=True, slots=True)
class Series:
    """One setup's paired-per-kernel comparison against its control, as :func:`pairs_table` records
    it: ``x`` the speedup change as ``log2(ratio)``, a geomean of per-kernel ratios; ``y`` the token-cost
    ratio (treated over control), a ratio of arithmetic means; each with its 95% bootstrap interval
    ``*_low``/``*_high``.
    """

    x: float
    x_low: float
    x_high: float
    y: float
    y_low: float
    y_high: float
    kernels: int
    delivered: int
    baseline_ns: float
    native_ns: float
    treated_baseline_ns: float
    treated_native_ns: float
    control_tokens: float
    treated_tokens: float
    #: Kernels the token leg (``y``) is over: those of ``kernels`` with a task total on both sides.
    token_kernels: int


@dataclasses.dataclass(frozen=True, slots=True)
class SetupPoint:
    """ONE SETUP's own position on a dot row: its geomean speedup over the experiment's
    BASELINE as ``log2(ratio)``, its per-kernel MEAN token cost as a COUNT (SC15: costs summarize
    with the arithmetic mean; :func:`per_kernel_ci`'s bootstrap interval, like the speedup's)."""

    x: float
    x_low: float
    x_high: float
    y: float
    y_low: float
    y_high: float
    kernels: int
    token_kernels: int
    #: Kernels this setup answered correctly, of the ``served`` ones of its pair: the success rate.
    solved: int = 0
    served: int = 0


def per_kernel_ci(values: "np.ndarray") -> tuple[float, float, float]:
    """What ONE KERNEL cost, and its 95% interval: the mean over the kernels of the pair, with the
    bootstrap mean interval around it.

    Per kernel rather than per tag so the row is comparable across experiments whose tags are
    different sizes -- gitscicomp10's ten against llr40's forty -- which a total is not.
    """
    if values.size == 0:
        return math.nan, math.nan, math.nan
    spend = summary.bootstrap_ci(values, np.mean, "mean")
    return spend.point, spend.low, spend.high


def setup_point(
    speedup: "pd.Series", tokens: "pd.Series", priced: "np.ndarray", timed: "np.ndarray", solved: "pd.Series"
) -> SetupPoint:
    """One setup's geomean speedup over the ``timed`` kernels, its PER-KERNEL token spend over the
    ``priced`` ones (a token total on BOTH sides, so the two setups of a pair are costed over one
    population), and how many of the pair's kernels it solved."""
    values = speedup.to_numpy(dtype=float)[timed]
    speed = summary.geomean_ci(values) if values.size else None
    spend, spend_low, spend_high = per_kernel_ci(tokens.to_numpy(dtype=float)[priced])
    return SetupPoint(
        x=summary.log2_change(speed.point) if speed else math.nan,
        x_low=summary.log2_change(speed.low) if speed else math.nan,
        x_high=summary.log2_change(speed.high) if speed else math.nan,
        y=spend,
        y_low=spend_low,
        y_high=spend_high,
        kernels=int(timed.sum()),
        token_kernels=int(priced.sum()),
        solved=int(np.sum(solved)),
        served=len(tokens),
    )


def setup_groups(frame: pd.DataFrame) -> Iterator[tuple[str, str, pd.DataFrame]]:
    """``(model, leg, rows)`` for each setup of ``frame``."""
    for key, pair in frame.assign(leg=leg_labels(frame)).groupby(["model", "leg"]):
        model, leg = population.key_parts(key, 2)
        yield model, leg, pair


def skill_halves(pair: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``pair`` split into its control rows and its skills rows."""
    skills = population.series_of(pair, "skills").astype(bool)
    return pair.loc[~skills], pair.loc[skills]


def leg_labels(frame: pd.DataFrame) -> pd.Series:
    """``frame``'s per-setup LEG label: what the setup DELIVERED, unless ``frame`` already carries a
    resolved ``leg`` (an explicit pair list can hold several legs in one language).

    Read off the SETUP name wherever there is one. An extracted observations table records a GPU C
    offload setup's language as plain ``c``, so "C" next to "HIP" and "Triton" names the host
    language and hides the OpenMP target kernels the agent actually wrote
    (:func:`~hpcagent_bench.study_tags.setup_delivery_name`).
    """
    if "leg" in frame:
        return population.series_of(frame, "leg").astype(str)
    if "setup" in frame:
        return population.series_of(frame, "setup").astype(str).map(study_tags.setup_delivery_name)
    return population.series_of(frame, "language").astype(str).map(study_tags.language_name)


#: One mark's significance superscript, per axis -- ``*`` for the SPEEDUP axis, ``+`` for the
#: TOKEN-COST one. Concatenated onto the mark's own label, never onto the mark itself: a symbol
#: drawn on top of a small shape is easy to miss, one beside a label a reader is already reading is
#: not. ``+`` rather than a dagger: the dagger is a footnote mark in running text and half the
#: readers of a printed panel read it as one.
SCORE_SIG_MARK: str = "*"
COST_SIG_MARK: str = "+"

#: The superscript of a measure that is not significantly different and passed the equivalence test (the
#: pair table's ``conclusion``); one mark for both measures, since it sits on the measure's own row.
EQUIVALENT_MARK: str = "="
EQUIVALENT_LABEL: str = "Equivalent"

#: Per (model, leg): whether the speedup and the token cost each cleared BH and, after them and only when the
#: stats table carries conclusions, whether each was found equivalent.
Significance = tuple[bool, ...]
NO_SIGNIFICANCE: Significance = (False, False)

#: What each superscript MEANS, as the legend spells it. Both name a CHANGE, because both
#: Benjamini-Hochberg tests behind them are on the packet's effect -- the setup against its own
#: no-packet twin -- and never on the setup's distance from the experiment's baseline, which nothing
#: here tests. One row each, symbol first: a reader looking a symbol up wants it at the start of
#: the row, not inside a sentence.
SCORE_SIG_LABEL: str = "Speedup Significant"
COST_SIG_LABEL: str = "Cost Significant"
#: Every drawn superscript in one key row, for a key that must fit four columns: this, then the
#: measures in :data:`SIG_MEASURE_NAMES` order.
JOINT_SIG_LABEL: str = "Significant"
#: Each superscript's measure, as the joint key row names it.
SIG_MEASURE_NAMES: tuple[str, ...] = ("Speedup", "Cost")


#: The stats table's conclusion per measure (``statistics/paired_setups.py`` under ``statistics.equivalence``), absent otherwise.
CONCLUSION_COLUMNS: tuple[str, str] = ("score_conclusion", "cost_conclusion")


def axis_significance(stats: pd.DataFrame) -> dict[tuple[str, str], Significance]:
    """Per (model, leg), ``(score cleared BH, cost cleared BH)`` -- the independent verdicts a
    mark's superscript reads off (:data:`SCORE_SIG_MARK`, :data:`COST_SIG_MARK`)."""
    flags: dict[tuple[str, str], Significance] = {}
    if stats.empty:
        # A stub panel's table has no columns at all, so there is no leg to read.
        return flags
    legs = leg_labels(stats)
    for (_, row), leg in zip(stats.iterrows(), legs, strict=True):
        score_sig = str(row.get("score_verdict", "")) == Finding.SIGNIFICANT.text
        cost_sig = str(row.get("cost_verdict", "")) == Finding.SIGNIFICANT.text
        same = [str(row[name]) == Conclusion.EQUIVALENT.text for name in CONCLUSION_COLUMNS if name in row]
        flags[(str(row["model"]), str(leg))] = (score_sig, cost_sig, *same)
    return flags


def family_sizes(stats: pd.DataFrame) -> tuple[int, int]:
    """How many tests the figure's speed and cost marks were each corrected over (``statistics.correction_family``:
    the same pooled count, or each leg's own); 0 when the table carries none."""
    if stats.empty or "score_family_size" not in stats:
        return 0, 0
    return int(stats.score_family_size.iloc[0]), int(stats.cost_family_size.iloc[0])


def interval_note(statistic: str) -> str:
    """ONE axis's own interval -- fixed text, so the key carries it once however many columns draw
    it.

    The estimator is NOT named here. "95% log-t CI" on two of five legend rows was the densest text
    in the figure, and which interval it is belongs in the caption beside the test it came from."""
    return f"{statistic}, 95% CI"


def drawn_symbols(stats: pd.DataFrame) -> Significance:
    """Whether ANY mark of ``stats`` wears each superscript. The legend explains a symbol only when
    the panel draws one: a row for a mark nothing carries is a lookup a reader makes for nothing."""
    if stats.empty:
        return NO_SIGNIFICANCE
    return any_significance(axis_significance(stats).values())


def any_significance(flags: Iterable[Significance]) -> Significance:
    """Per measure, whether any of ``flags`` is set."""
    listed = list(flags)
    width = max(map(len, listed), default=len(NO_SIGNIFICANCE))
    return tuple(any(flag[index] for flag in listed if index < len(flag)) for index in range(width))


#: The 1x reference every relative row's control is: no change in either measure.
UNCHANGED_POINT = SetupPoint(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)


#: An SetupPoint with nothing in it: the slot a delivery that has not been measured yet keeps, so a
#: column's spacing is its FINAL spacing and the figure does not re-lay out when the data lands.
EMPTY_POINT = SetupPoint(math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, 0, 0)


def absolute_columns(side: str, point: SetupPoint) -> dict[str, float]:
    """``side``'s absolute level as table columns: ``<side>_speedup`` (a ratio, not log2) and
    ``<side>_mean_tokens`` per kernel, each with its ``_low``/``_high`` 95% interval."""
    return {
        f"{side}_speedup": 2.0**point.x,
        f"{side}_speedup_low": 2.0**point.x_low,
        f"{side}_speedup_high": 2.0**point.x_high,
        f"{side}_mean_tokens": point.y,
        f"{side}_mean_tokens_low": point.y_low,
        f"{side}_mean_tokens_high": point.y_high,
    }
