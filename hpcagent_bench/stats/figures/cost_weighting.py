# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""How the choice of cost weighting changes an intervention's token-cost ratio.

Every cost card (:mod:`hpcagent_bench.stats.cost`) is a linear weight on the same three per-task token
counts, but the efficacy ratio rho_C is a ratio of GEOMEANS of those weighted sums, so it does not
convert from one card to another: it is recomputed per card from the per-task counts. This module does
that for a list of (treated, control) pairs and draws one mark per pair and card.

rho_C = GM(C_control) / GM(C_treated) over the kernels both arms have a token count for, the paper's
orientation: above 1 the intervention SAVES tokens. It is the paired geomean
(:func:`hpcagent_bench.stats.summary.paired_geomean`) of log(C_control / C_treated) with its 95%
t-interval, the same statistic every cost leg in this repo reports.

The figure is a wrap figure beside the text, on the shared print scale
(:data:`~hpcagent_bench.stats.style.PRINT_SCALE`): colour is the model (a second pair of one model
takes a lighter shade), shape the treatment (:func:`~hpcagent_bench.stats.palette.packet_marker` or
the harness's), and the key names both.

The cost leg needs token counts only, never a grade, so it can be drawn before the grades land.
"""

import math
import pathlib
from collections.abc import Sequence

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from hpcagent_bench import experiment_tags
from hpcagent_bench.stats import cost, palette, population, summary
from hpcagent_bench.stats import style as plotstyle

#: The pseudo-card that prices each pair with its own model's list price, ``usd-<model>`` in the
#: shipped cost cards: dollars depend on the model, the token weightings do not.
USD_CARD: str = "usd"

#: The cards the figure compares, in X order: the three token weightings, then list-price dollars.
DEFAULT_CARDS: tuple[str, ...] = (*cost.PROXY_CARDS, USD_CARD)

#: Columns of :func:`pair_cost_ratios`, in order.
COLUMNS: tuple[str, ...] = ("arm_a", "arm_b", "card", "n", "rho_c", "ci_low", "ci_high")

#: Horizontal spread of the marks of one card, as a fraction of the gap between two cards.
DODGE_SPAN: float = 0.6

TYPE: plotstyle.TypeScale = plotstyle.PRINT_SCALE
WIDTH_IN: float = plotstyle.ICLR_WRAP_WIDTH_IN
BODY_HEIGHT_IN: float = plotstyle.PRINT_BODY_HEIGHT_IN

#: The key's entry for the USD slot, which is a price vector per model rather than one weighting.
USD_NOTE: str = "USD $w_m$: list price per model"
YLABEL: str = "$\\rho_C = C_{\\mathrm{control}}/C_{\\mathrm{treated}}$\n(above 1: treatment cheaper)"
XLABEL: str = "Weights $w$ (in, cached, out)"


def arm_tokens(observations: pd.DataFrame, card: cost.CostModel, repeats: population.RepeatPolicy) -> dict:
    """``(arm, kernel) -> tokens`` with every task priced by ``card``."""
    priced = population.condition_rows(cost.priced(observations, card))
    totals = population.kernel_tokens(priced, ("arm", "benchmark"), repeats=repeats)
    return {(str(arm), str(kernel)): float(spend) for (arm, kernel), spend in totals.items()}


def pair_cost_ratios(
    observations: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    cards: Sequence[str] = DEFAULT_CARDS,
    repeats: population.RepeatPolicy = "latest",
) -> pd.DataFrame:
    """One row per (pair, card): rho_C of TREATED ``arm_a`` against CONTROL ``arm_b`` and its interval.

    Under :data:`USD_CARD` each pair is priced with ``usd-<model>``, its model's list price."""
    rows: list[dict[str, object]] = []
    for key in cards:
        by_card: dict[str, dict] = {}
        for treated, control in pairs:
            name = f"{USD_CARD}-{experiment_tags.model_of(treated)}" if key == USD_CARD else key
            if name not in by_card:
                by_card[name] = arm_tokens(observations, cost.resolve(name), repeats)
            tokens = by_card[name]
            shared = sorted(
                {k for a, k in tokens if a == treated and tokens[(a, k)] > 0}
                & {k for a, k in tokens if a == control and tokens[(a, k)] > 0}
            )
            change = summary.paired_geomean([math.log(tokens[(control, k)] / tokens[(treated, k)]) for k in shared])
            finite = math.isfinite(change.low) and math.isfinite(change.high)
            rows.append(
                {
                    "arm_a": treated,
                    "arm_b": control,
                    "card": key,
                    "n": len(shared),
                    "rho_c": math.exp(change.estimate) if shared else math.nan,
                    "ci_low": math.exp(change.low) if finite else math.nan,
                    "ci_high": math.exp(change.high) if finite else math.nan,
                }
            )
    return pd.DataFrame(rows, columns=list(COLUMNS))


def treatment_marker(arm: str) -> str:
    """The treatment's registered shape: its packet's, else its harness's."""
    packet = experiment_tags.packet_of(arm)
    if packet:
        return palette.packet_marker(packet)
    tokens = arm.split("-")
    for harness in palette.hue_order("harnesses"):
        if harness in tokens:
            return palette.harness_marker(harness)
    return palette.packet_marker(packet)


def pair_styles(arms: Sequence[str]) -> dict[str, tuple[str, str]]:
    """``arm -> (colour, marker)``: the model's colour, one lighter shade per further pair of that
    model, and the treatment's shape."""
    seen: dict[str, int] = {}
    styles: dict[str, tuple[str, str]] = {}
    for arm in arms:
        model = experiment_tags.model_of(arm)
        step = seen.get(model, 0)
        seen[model] = step + 1
        styles[arm] = (palette.model_shade(model, step), treatment_marker(arm))
    return styles


def model_key(arms: Sequence[str]) -> list[Patch]:
    """One colour swatch per model, naming how many shades it wears when it has several pairs."""
    counts: dict[str, int] = {}
    for arm in arms:
        model = experiment_tags.model_of(arm)
        counts[model] = counts.get(model, 0) + 1
    return [
        Patch(
            color=palette.model_color(model),
            label=experiment_tags.model_name(model) + (f" ({shades} shades)" if shades > 1 else ""),
        )
        for model, shades in counts.items()
    ]


def short_tick(card: str) -> str:
    """A compact weight-vector tick, ``(1,.1,1)``; the axis label names the three components."""
    if card == USD_CARD:
        return "USD $w_m$"
    model = cost.resolve(card)
    return (
        "(" + ",".join(f"{w:g}".replace("0.", ".") for w in (model.fresh_input, model.cached_input, model.output)) + ")"
    )


def mark(ax: matplotlib.axes.Axes, x: float, row: object, hue: str, shape: str) -> None:
    """One estimate with its 95% interval, the interval omitted when too few kernels support one."""
    ok = math.isfinite(row.ci_low) and math.isfinite(row.ci_high)
    ax.errorbar(
        x,
        row.rho_c,
        yerr=[[row.rho_c - row.ci_low], [row.ci_high - row.rho_c]] if ok else None,
        color=hue,
        marker=shape,
        markersize=TYPE.marker_size,
        linestyle="",
        elinewidth=0.9,
        capsize=1.8,
        zorder=5,
    )


def ratio_axis(ax: matplotlib.axes.Axes) -> None:
    """The log2 ratio Y axis at print type sizes, with the no-change line."""
    ax.axhline(1.0, color=plotstyle.REFERENCE, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.set_yscale("log", base=2)
    plotstyle.value_axis(ax, "y", log_base=2.0)
    ax.yaxis.set_major_formatter(FuncFormatter(plotstyle.ratio_tick))
    ax.tick_params(axis="both", labelsize=TYPE.tick_pt)
    ax.set_ylabel(YLABEL, fontsize=TYPE.label_pt)
    plotstyle.despine(ax)


def figure_cost_points(
    points: pd.DataFrame, labels: dict[str, str], width: float = WIDTH_IN
) -> matplotlib.figure.Figure | None:
    """One slot per card (its weight vector, then USD), one dodged mark per pair. Marks of one card
    are never joined: a line between two cards would suggest a quantity between them."""
    drawn = points.dropna(subset=["rho_c"])
    if drawn.empty:
        return None
    arms = list(dict.fromkeys(drawn.arm_a))
    cards = list(dict.fromkeys(drawn.card))
    styles = pair_styles(arms)
    fig, ax = plt.subplots(figsize=(width, BODY_HEIGHT_IN))
    step = DODGE_SPAN / max(1, len(arms))
    for i, arm in enumerate(arms):
        hue, shape = styles[arm]
        for row in drawn[drawn.arm_a == arm].itertuples():
            mark(ax, cards.index(row.card) + (i - (len(arms) - 1) / 2) * step, row, hue, shape)
    ax.set_xticks(range(len(cards)))
    ax.set_xticklabels([short_tick(card) for card in cards])
    ax.set_xlim(-0.6, len(cards) - 0.4)
    ax.set_xlabel(XLABEL, fontsize=TYPE.label_pt)
    ratio_axis(ax)
    handles: list = [
        Line2D([], [], color=styles[a][0], marker=styles[a][1], linestyle="", label=labels.get(a, a)) for a in arms
    ]
    handles += model_key(arms)
    fig.tight_layout()
    below = plotstyle.below_protrusion_in(fig, ax) + 0.04
    # The USD note is a line of its own under the key: as a sixth entry it widens the key's second
    # column past the wrap width and the key falls back to one tall column.
    note = fig.text(0.5, 0.0, USD_NOTE if USD_CARD in cards else "", ha="center", va="bottom", fontsize=TYPE.legend_pt)
    renderer = fig.canvas.get_renderer()
    note_in = note.get_window_extent(renderer).height / fig.dpi + 0.03 if note.get_text() else 0.0
    key_in = plotstyle.legend_below(
        fig, handles, ncol=2, y=0.01, fontsize=TYPE.legend_pt, markerscale=1.0, **plotstyle.COMPACT_KEY
    )
    top = fig.subplotpars.top * BODY_HEIGHT_IN
    total = BODY_HEIGHT_IN + key_in + note_in
    fig.set_size_inches(width, total)
    fig.legends[-1].set_bbox_to_anchor((0.5, (note_in + 0.02) / total), transform=fig.transFigure)
    note.set_position((0.5, 0.02 / total))
    fig.subplots_adjust(bottom=(key_in + note_in + below) / total, top=(top + key_in + note_in) / total)
    return fig


def save(fig: matplotlib.figure.Figure, stem: pathlib.Path) -> pathlib.Path:
    """Write the PDF and the PNG under ``stem`` at the figure's placed width, and close the figure."""
    return plotstyle.save(fig, stem, width_in=float(fig.get_size_inches()[0]))
