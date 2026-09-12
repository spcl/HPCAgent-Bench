# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The benchmarking rules a figure in this repo has to obey, as checks rather than as prose.

From T. Hoefler and R. Belli, "Scientific Benchmarking of Parallel Computing Systems", SC15. The
rules this module enforces, quoted from the paper:

* **Rule 4** -- "Avoid summarizing ratios; summarize the costs or rates that the ratios base on
  instead. Only if these are not available use the geometric mean for summarizing ratios."
* **Rule 5** -- "Report if the measurement values are deterministic. For nondeterministic data,
  report confidence intervals of the measurement."
* **Rule 7** -- "Compare nondeterministic data in a statistically sound way, e.g., using
  non-overlapping confidence intervals or ANOVA."
* **Rule 12** -- "Plot as much information as needed to interpret the experimental results. Only
  connect measurements by lines if they indicate trends and the interpolation is valid."

WHY THEY ARE FUNCTIONS. A rule written in a style guide is a rule a figure breaks silently: the
plot still renders, the numbers still look plausible, and nothing tells the author that the ratio
axis they drew has no costs behind it or that the line they drew between a control and a treatment
claims a trend across a two-point axis that has no order. These raise instead, at the point the
figure is built, and name the rule so the fix is a decision rather than a search.

Every check takes what the figure ALREADY has -- its data table, its interval columns, its x
values -- so obeying a rule and emitting the table a reader needs are the same act.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs] -- pandas ships none

#: Paper and rule text this module enforces, for a caption or an error message.
CITATION: str = 'Hoefler and Belli, "Scientific Benchmarking of Parallel Computing Systems", SC15'

RULE_TEXT: dict[int, str] = {
    4: (
        "Avoid summarizing ratios; summarize the costs or rates that the ratios base on instead. "
        "Only if these are not available use the geometric mean for summarizing ratios."
    ),
    5: (
        "Report if the measurement values are deterministic. For nondeterministic data, report "
        "confidence intervals of the measurement."
    ),
    7: (
        "Compare nondeterministic data in a statistically sound way, e.g., using non-overlapping "
        "confidence intervals or ANOVA."
    ),
    12: (
        "Plot as much information as needed to interpret the experimental results. Only connect "
        "measurements by lines if they indicate trends and the interpolation is valid."
    ),
}


class RuleViolation(ValueError):
    """A figure broke one of the rules. Carries the rule number so the message names the source."""

    def __init__(self, rule: int, detail: str) -> None:
        super().__init__(f"SC15 Rule {rule}: {RULE_TEXT[rule]}\n  {detail}\n  ({CITATION})")
        self.rule: int = rule


def require_costs(table: pd.DataFrame, ratio: str, costs: Sequence[str]) -> pd.DataFrame:
    """Rule 4. A table carrying a ratio column must carry the costs the ratio was taken over.

    A speed-up alone is uninterpretable: 1.4x on a kernel that runs for 3 ms and 1.4x on one that
    runs for 3 s are different results, and the reader cannot tell them apart from the ratio. So
    the numerator and denominator travel with it, in the SAME table the figure emits, and a figure
    that cannot supply them has to say so rather than ship the ratio on its own.
    """
    if ratio not in table.columns:
        raise RuleViolation(4, f"the table has no {ratio!r} column to check")
    missing = [c for c in costs if c not in table.columns]
    if missing:
        raise RuleViolation(4, f"{ratio!r} is summarized with no costs behind it; add {missing}")
    if table.empty:
        return table  # nothing was summarized, so no ratio is standing without its costs
    empty = [c for c in costs if not table[c].notna().any()]
    if empty:
        raise RuleViolation(4, f"{ratio!r} has cost columns {empty} that are entirely missing")
    return table


def require_interval(table: pd.DataFrame, point: str, low: str, high: str, deterministic: bool = False) -> pd.DataFrame:
    """Rules 5 and 7. A nondeterministic measurement is reported with an interval, or declared.

    ``deterministic=True`` is the escape hatch the paper allows, and it is an ASSERTION about the
    data rather than a way past the check: the caller is saying the values do not vary between
    runs, which the figure's caption then has to say too. Anything else must carry ``low`` and
    ``high`` beside ``point``, because two point estimates without intervals cannot be compared --
    which is Rule 7, and is the comparison every figure here is actually making.

    A row whose interval is absent is named, not silently dropped: the missing interval is
    usually a cell with too few repetitions, and that is a fact about the run.
    """
    if deterministic:
        return table
    missing = [c for c in (point, low, high) if c not in table.columns]
    if missing:
        raise RuleViolation(5, f"nondeterministic data plotted without an interval; add columns {missing}")
    usable = table[point].notna()
    if not usable.any():
        return table
    bare = table.index[usable & (table[low].isna() | table[high].isna())]
    if len(bare) == len(table.index[usable]):
        raise RuleViolation(5, f"every plotted point in {point!r} has an empty interval in {low!r}/{high!r}")
    inverted = table.index[usable & table[low].notna() & table[high].notna() & (table[low] > table[high])]
    if len(inverted):
        raise RuleViolation(5, f"interval ends are the wrong way round on rows {list(inverted)[:5]}")
    return table


def separated(low_a: float, high_a: float, low_b: float, high_b: float) -> bool:
    """Rule 7. Do two intervals fail to overlap -- the weakest sound statement a figure can make.

    Non-overlap implies a difference at the stated level; OVERLAP IMPLIES NOTHING, which is the
    half readers get wrong. A figure that needs to claim a difference from overlapping intervals
    needs a paired test instead (:func:`hpcagent_bench.stats.summary.paired_change`), not a
    narrower-looking pair of bars.
    """
    if not all(np.isfinite([low_a, high_a, low_b, high_b])):
        return False
    return high_a < low_b or high_b < low_a


def require_ordered_x(values: Sequence[float], connected: bool) -> None:
    """Rule 12. Points may be joined by a line only where the x axis has a meaningful order.

    ``connected`` is what the figure intends to draw. A line asserts two things -- that the order
    is real and that the space between the points is interpolable -- so an axis of CONDITIONS
    (control, treatment) or of NAMES (kernels) may never carry one: the line would read as a trend
    across a gap that does not exist. Draw the difference instead, which is what
    :func:`difference_segment` builds.
    """
    if not connected:
        return
    x = np.asarray(values, dtype=np.float64)
    if x.size < 2:
        raise RuleViolation(12, "a line through fewer than two points indicates no trend")
    if not np.all(np.diff(x) > 0):
        raise RuleViolation(12, f"x values {x.tolist()[:6]} are not strictly increasing, so a line is not a trend")


def difference_segment(before: float, after: float, at: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Rule 12. The control-to-treatment link as a DIFFERENCE: one vertical segment at one x.

    Two conditions are not a trend and their axis has no interpolable interior, so the pair is
    drawn at a single x position with the segment spanning the two values. The length of the
    segment IS the effect, read against the value axis, and there is no horizontal run for the eye
    to extrapolate along.

    Returns ``((x0, x1), (y0, y1))``, ready for ``ax.plot``.
    """
    return (at, at), (before, after)
