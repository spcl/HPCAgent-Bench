# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A forest plot for any family CSV ``experiments/paired_arms.py`` writes: one row per pair.

NEVER RECOMPUTES A STATISTIC. Every number drawn here -- the ratio, its interval, its
Benjamini-Hochberg verdict -- is read straight off the columns ``paired_arms.py`` already wrote;
this script only lays them out. A pair's row label is its own ``arm_a``/``arm_b`` columns, never a
naming convention re-parsed here, so any family that script produces draws with the same code --
``blind-vs-scored``, a skill-packet family, a git-scicomp episode-count family, whatever comes next.

TWO PANELS, NEVER ONE AXIS: a pair's speed-up ratio and its token ratio are different quantities
(SC15 Rule 4, ``hpcagent_bench/stats/rules.py``), and a shared scale would let a reader compare
their positions as if they were the same measurement. Both are ratios, so both get the SAME log2
axis treatment ``hpcagent_bench/stats/figures/per_kernel.py`` gives its own speed-up panel: powers
of two, ticks read back as ratios ("2x", "1/2x"), a 1x reference line. A pair with no tokens leg
(``paired_arms.py`` found no call rows shared by the two arms) leaves that row's right panel blank
rather than failing.

Rows are one per PAIR, not one per (pair, leg): the family CSV interleaves a pair's speed-up and
tokens rows one after the other, and ``rows_for`` re-groups them by ``(arm_a, arm_b)`` so both
panels share one categorical y axis (``hpcagent_bench/stats/style.py``'s ``row_axis``). Colour and
marker are the model either arm names, read off the shared model registry
(``hpcagent_bench/experiment_tags.py``) rather than off a hyphen convention -- palette's own rule
is that shape is always the model -- and fall back to a neutral grey circle for a pair naming none.
"""

import argparse
import dataclasses
import math
import pathlib
from collections.abc import Sequence

import matplotlib.axes
import matplotlib.figure
import pandas as pd

from hpcagent_bench import experiment_tags
from hpcagent_bench.harness import efficacy
from hpcagent_bench.stats import palette
from hpcagent_bench.stats import style as plotstyle
from hpcagent_bench.stats.figures import per_kernel

plotstyle.apply()
import matplotlib.pyplot as plt  # pyplot must follow plotstyle.apply()

#: One row's vertical budget, inches. A row label is two lines (both arm names), so this is taller
#: than per_kernel.py's single-line kernel columns.
ROW_HEIGHT_IN: float = 0.55

#: A floor under the plotted rows' own height, so a one- or two-pair family still gets a panel
#: rather than a sliver between the title and the x axis.
MIN_AXES_IN: float = 1.4

#: ``plotstyle.title``'s own fixed cost, in inches: its text sits ``0.34/height`` from the top and
#: returns an axes top ``0.30/height`` further down, so the title block is 0.64in regardless of the
#: figure's own height. Named here so this script's own blocks below can be added to it exactly,
#: rather than guessing a total that fights ``title``'s formula on a short figure.
TITLE_BLOCK_IN: float = 0.64

#: Fixed inches for the x-axis ticks and label, and for the legend beneath them. All THREE blocks
#: here are converted to a per-figure height FRACTION in ``build_figure``, the same trick
#: ``plotstyle.title`` uses for its own gap -- a constant fraction wastes growing whitespace on a
#: tall many-pair figure, where the ticks and the legend still cost the same inches.
XLABEL_BLOCK_IN: float = 0.75
LEGEND_BLOCK_IN: float = 0.55

#: A figure's width without ``--double-column``: wide enough that a long arm-name label does not
#: force the panels themselves down to a sliver.
STANDALONE_WIDTH_IN: float = 9.0

#: The panel gap (``subplots_adjust(wspace=...)``) starts here and grows until the boundary tick
#: labels clear; a family whose ratios never get wide edge labels ("4x") keeps this narrow gap.
INITIAL_WSPACE: float = 0.08
WSPACE_STEP: float = 0.02
MAX_WSPACE: float = 0.6

#: Minimum clear space between the two panels' boundary tick labels, in points -- enough that the
#: glyphs read as two separate numbers rather than touching.
MIN_LABEL_GAP_PT: float = 4.0

Pair = tuple[str, str]


@dataclasses.dataclass(frozen=True, slots=True)
class Row:
    """One pair's one leg: the ratio ``paired_arms.py`` computed, its interval, and whether the
    family's Benjamini-Hochberg correction still calls it significant."""

    pair: Pair
    estimate: float
    low: float
    high: float
    significant: bool
    n: int


def pair_order(table: pd.DataFrame) -> list[Pair]:
    """Every pair, in the order its first row appears in the table -- the family's own declared
    order (a caller's model/language/skills loop), not a re-sort."""
    seen: dict[Pair, None] = {}
    for record in table.itertuples(index=False):
        seen.setdefault((str(record.arm_a), str(record.arm_b)), None)
    return list(seen)


def rows_for(table: pd.DataFrame, leg: str, order: Sequence[Pair]) -> list[Row]:
    """One ``Row`` per pair of ``order``, in that order; a pair missing this leg (no shared call
    rows for ``tokens``) gets a NaN row so both panels keep one shared y axis."""
    by_pair = {(str(r.arm_a), str(r.arm_b)): r for r in table[table.leg == leg].itertuples(index=False)}
    rows: list[Row] = []
    for pair in order:
        record = by_pair.get(pair)
        if record is None:
            rows.append(Row(pair, math.nan, math.nan, math.nan, False, 0))
            continue
        rows.append(
            Row(
                pair,
                float(record.estimate_a_over_b),
                float(record.ci_low),
                float(record.ci_high),
                str(record.verdict) == efficacy.SIGNIFICANT,
                int(record.n_tested),
            )
        )
    return rows


def pair_label(pair: Pair) -> str:
    """A pair's row label: its own two arm names, exactly as the table names them."""
    return f"{pair[0]}\nvs {pair[1]}"


def named_model(pair: Pair, models: Sequence[str]) -> str:
    """The registered model tag named by either arm of ``pair``, or "" when none is -- a colour and
    marker key, read off the shared registry rather than off a naming convention this script would
    have to keep in step with a new experiment's."""
    tokens = set(pair[0].split("-")) | set(pair[1].split("-"))
    return next((model for model in models if model in tokens), "")


def pair_style(pairs: Sequence[Pair], models: Sequence[str]) -> tuple[dict[Pair, str], dict[Pair, str]]:
    """Colour and marker per pair: the model either arm names (palette's rule -- shape is always the
    model), or a neutral grey circle for a pair naming none of the registered models."""
    detected = {pair: named_model(pair, models) for pair in pairs}
    known = sorted({model for model in detected.values() if model})
    hues = palette.model_colors(known) if known else {}
    shapes = palette.model_markers(known) if known else {}
    colors = {pair: hues.get(model, plotstyle.MUTED) for pair, model in detected.items()}
    markers = {pair: shapes.get(model, "o") for pair, model in detected.items()}
    return colors, markers


def ratio_ticks(rows: Sequence[Row]) -> list[float]:
    """Powers of two spanning every finite value drawn, always at least ``1/4x .. 4x`` --
    ``per_kernel.speedup_yticks``'s own rule, over a row's ratio and interval instead of a kernel's
    raw episodes."""
    values = [v for row in rows for v in (row.estimate, row.low, row.high) if math.isfinite(v) and v > 0.0]
    low, high = (min(values), max(values)) if values else (1.0, 1.0)
    low_exp = min(-2, math.floor(math.log2(low)))
    high_exp = max(2, math.ceil(math.log2(high)))
    return [2.0**exp for exp in range(low_exp, high_exp + 1)]


def style_ratio_axis(ax: matplotlib.axes.Axes, rows: Sequence[Row]) -> None:
    """The log2 ratio x axis: ``per_kernel.style_speedup_axis``'s own tick and 1x-line rule, on x
    instead of y -- a forest plot's rows run down the page, so its ratio is the horizontal axis."""
    ax.set_xscale("log", base=2)
    ticks = ratio_ticks(rows)
    ax.set_xticks(ticks)
    ax.set_xticklabels([per_kernel.speedup_tick_label(tick) for tick in ticks], fontsize=plotstyle.TICK_PT * 0.75)
    ax.axvline(1.0, color=plotstyle.REFERENCE, linewidth=0.9, zorder=1)


def draw_forest(
    ax: matplotlib.axes.Axes, rows: Sequence[Row], colors: dict[Pair, str], shapes: dict[Pair, str]
) -> None:
    """One row per pair: a whisker for the interval, a mark at the point estimate, a star where the
    family's adjusted verdict still calls the ratio significant."""
    for i, row in enumerate(rows):
        if not math.isfinite(row.estimate):
            continue
        color, shape = colors[row.pair], shapes[row.pair]
        if math.isfinite(row.low) and math.isfinite(row.high) and row.low != row.high:
            ax.hlines(i, row.low, row.high, color=color, linewidth=1.3, alpha=0.6, zorder=2)
        plotstyle.point_mark(ax, row.estimate, i, color, shape, True, size=70.0)
        if row.significant:
            at = row.high if math.isfinite(row.high) else row.estimate
            ax.annotate(
                "*",
                xy=(at, i),
                xytext=(5, -1),
                textcoords="offset points",
                va="center",
                fontsize=plotstyle.ANNOTATION_PT,
                color=plotstyle.INK,
            )


def model_legend(pairs: Sequence[Pair], colors: dict[Pair, str], shapes: dict[Pair, str]) -> list[plt.Line2D]:
    """One legend handle per registered model the family names, in registry order."""
    models = experiment_tags.order("models")
    by_model = {named_model(pair, models): (colors[pair], shapes[pair]) for pair in pairs}
    return [
        plt.Line2D(
            [], [], marker=shape, linestyle="none", color=color, markersize=8, label=experiment_tags.model_name(model)
        )
        for model, (color, shape) in by_model.items()
        if model
    ]


def widen_gap_until_labels_clear(
    fig: matplotlib.figure.Figure, ax_left: matplotlib.axes.Axes, ax_right: matplotlib.axes.Axes
) -> None:
    """Grow the panel gap (``wspace``) until the left panel's rightmost tick label and the right
    panel's leftmost one stop touching, measured from the RENDERED glyphs rather than guessed from
    a font size.

    The labels that collide are whichever ratio happens to need the widest text -- "1/16x" is
    wider than "4x" -- so a fixed gap sized for one family's numbers collides on another's. Each
    step redraws and re-measures with ``get_window_extent``, which is what the renderer actually
    placed, in device pixels, so the check is correct at any DPI or font substitution rather than
    an estimate of one.
    """
    wspace = INITIAL_WSPACE
    while True:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        left_box = ax_left.get_xticklabels()[-1].get_window_extent(renderer)
        right_box = ax_right.get_xticklabels()[0].get_window_extent(renderer)
        min_gap_px = MIN_LABEL_GAP_PT * fig.dpi / 72.0
        if right_box.x0 - left_box.x1 >= min_gap_px or wspace >= MAX_WSPACE:
            return
        wspace = min(wspace + WSPACE_STEP, MAX_WSPACE)
        fig.subplots_adjust(wspace=wspace)


def build_figure(table: pd.DataFrame, label: str, double_column: bool) -> matplotlib.figure.Figure:
    """One figure: a shared row per pair, speed-up on the left, tokens on the right, colour and
    marker by the model either arm names.

    The height is BUDGETED in inches, not guessed as a fraction: the title's fixed cost
    (:data:`TITLE_BLOCK_IN`, matching ``plotstyle.title``'s own formula), the x-axis block and the
    legend block are added to the rows' own height, so ``subplots_adjust`` can convert every block
    back to an exact fraction of THIS figure instead of fighting ``title``'s fraction on a short one.
    """
    pairs = pair_order(table)
    if not pairs:
        raise SystemExit("paired_arms.py table has no rows to plot")
    speed_rows = rows_for(table, "speedup", pairs)
    token_rows = rows_for(table, "tokens", pairs)
    colors, shapes = pair_style(pairs, experiment_tags.order("models"))
    handles = model_legend(pairs, colors, shapes)

    axes_in = max(len(pairs) * ROW_HEIGHT_IN, MIN_AXES_IN)
    bottom_in = XLABEL_BLOCK_IN + (LEGEND_BLOCK_IN if handles else 0.0)
    height = axes_in + TITLE_BLOCK_IN + bottom_in
    width = plotstyle.DOUBLE_COLUMN_WIDTH if double_column else STANDALONE_WIDTH_IN
    fig, (ax_speed, ax_tokens) = plt.subplots(1, 2, sharey=True, figsize=(width, height))

    draw_forest(ax_speed, speed_rows, colors, shapes)
    style_ratio_axis(ax_speed, speed_rows)
    ax_speed.set_xlabel("Speedup Ratio (a / b)", fontsize=plotstyle.LABEL_PT * 0.8)
    plotstyle.row_axis(ax_speed, [pair_label(pair) for pair in pairs])

    draw_forest(ax_tokens, token_rows, colors, shapes)
    style_ratio_axis(ax_tokens, token_rows)
    ax_tokens.set_xlabel("Token Ratio (a / b)", fontsize=plotstyle.LABEL_PT * 0.8)
    plotstyle.despine(ax_tokens)
    ax_tokens.tick_params(axis="y", length=0, labelleft=False)

    # left/right/wspace first, and TOP/BOTTOM ONLY on the call below: subplots_adjust leaves every
    # parameter it is not given at its current value, so widening the gap here survives the later
    # call that sets top and bottom for the title and the legend.
    fig.subplots_adjust(left=0.36, right=0.98, wspace=INITIAL_WSPACE)
    widen_gap_until_labels_clear(fig, ax_speed, ax_tokens)

    if handles:
        plotstyle.legend_below(fig, handles, y=0.02)
    top = plotstyle.title(fig, label)
    fig.subplots_adjust(top=top, bottom=bottom_in / height)
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table", type=pathlib.Path, help="the pairs CSV experiments/paired_arms.py wrote (--out)")
    parser.add_argument("--label", default="", help="figure title")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("figures/paired_arms.pdf"))
    parser.add_argument(
        "--double-column", action="store_true", default=False, help="cap the figure width at style.DOUBLE_COLUMN_WIDTH"
    )
    args = parser.parse_args(argv)

    table = pd.read_csv(args.table)
    fig = build_figure(table, args.label or args.table.stem, args.double_column)
    written = plotstyle.save(fig, args.out.with_suffix(""))
    print(f"{len(pair_order(table))} pairs -> {written} (+ .png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
