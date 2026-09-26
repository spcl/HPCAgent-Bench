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

import dataclasses
import enum
import math
import pathlib
from collections.abc import Sequence

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from hpcagent_bench import experiment_tags
from hpcagent_bench.stats import cost, palette, population, summary
from hpcagent_bench.stats import style as plotstyle


class Card(enum.Enum):
    """One X slot of the figure, in slot order: the three token weightings, then list-price dollars.
    ``USD`` is a pseudo-card: each pair is priced with ``usd-<model>``, its own model's card, since
    dollars depend on the model and the token weightings do not."""

    EFFECTIVE = "effective"
    BILLED = "billed"
    TOTAL = "total"
    USD = "usd"


#: The token weightings every figure draws; ``Card.USD`` joins them when every model has a price.
TOKEN_CARDS: tuple[Card, ...] = (Card.EFFECTIVE, Card.BILLED, Card.TOTAL)


@dataclasses.dataclass(frozen=True, slots=True)
class Pair:
    """One intervention of one model: its TREATED arm, its CONTROL arm and the key text."""

    treated: str
    control: str
    label: str = ""

    @property
    def model(self) -> str:
        return experiment_tags.model_of(self.treated)


def usd_card_name(model: str) -> str:
    """The price card of ``model``: ``usd-<model>``."""
    return f"{Card.USD.value}-{model}"


def figure_cards(pairs: Sequence[Pair], extra: pathlib.Path | None = None) -> tuple[Card, ...]:
    """The slots a figure over ``pairs`` draws: the token weightings, and USD only when EVERY
    model has a price card (shipped, or in ``extra``) -- a dollar slot missing some models would
    compare a subset of the marks with the rest."""
    known = {**cost.shipped_cards(), **(cost.load_cards(extra) if extra is not None else {})}
    priced = all(usd_card_name(pair.model) in known for pair in pairs)
    return (*TOKEN_CARDS, Card.USD) if priced and pairs else TOKEN_CARDS


#: Columns of :func:`pair_cost_ratios`, in order.
COLUMNS: tuple[str, ...] = ("arm_a", "arm_b", "card", "n", "rho_c", "ci_low", "ci_high")

#: Horizontal spread of the marks of one card, as a fraction of the gap between two cards.
DODGE_SPAN: float = 0.6

TYPE: plotstyle.TypeScale = plotstyle.PRINT_SCALE
WIDTH_IN: float = plotstyle.ICLR_WRAP_WIDTH_IN
#: A quarter under 0.735 (user, 2026-09-26: the figure sits beside a paragraph, not in its own row);
#: earlier 0.735 was 50% taller than a too-short 0.49.
BODY_HEIGHT_IN: float = 0.55 * plotstyle.PRINT_BODY_HEIGHT_IN

#: Slot ticks by the weighting's name in the paper; the text gives each weight vector.
TICKS: dict[Card, str] = {Card.EFFECTIVE: "eff", Card.BILLED: "bill", Card.TOTAL: "tot", Card.USD: r"\$"}
#: The paper's name for rho_C = C_control / C_treated; above 1 the treated setup is cheaper.
YLABEL: str = "Cost ratio"
XLABEL: str = "Weighting"


def arm_tokens(observations: pd.DataFrame, card: cost.CostModel, repeats: population.RepeatPolicy) -> dict:
    """``(arm, kernel) -> tokens`` with every task priced by ``card``."""
    priced = population.condition_rows(cost.priced(observations, card))
    totals = population.kernel_tokens(priced, ("arm", "benchmark"), repeats=repeats)
    return {(str(arm), str(kernel)): float(spend) for (arm, kernel), spend in totals.items()}


def shared_kernels(tokens: dict, treated: str, control: str) -> list[str]:
    """The kernels both arms spent tokens on, sorted: the pairing a paired geomean needs."""
    spent = {(arm, kernel) for (arm, kernel), count in tokens.items() if count > 0}
    return sorted({k for a, k in spent if a == treated} & {k for a, k in spent if a == control})


def ratio_row(tokens: dict, pair: Pair, card: str) -> dict[str, object]:
    """One :data:`COLUMNS` row: rho_C of ``pair`` over its shared kernels under one priced ``tokens``
    table, NaN where no kernel is shared or too few support an interval."""
    shared = shared_kernels(tokens, pair.treated, pair.control)
    change = summary.paired_geomean([math.log(tokens[(pair.control, k)] / tokens[(pair.treated, k)]) for k in shared])
    finite = math.isfinite(change.low) and math.isfinite(change.high)
    return {
        "arm_a": pair.treated,
        "arm_b": pair.control,
        "card": card,
        "n": len(shared),
        "rho_c": math.exp(change.estimate) if shared else math.nan,
        "ci_low": math.exp(change.low) if finite else math.nan,
        "ci_high": math.exp(change.high) if finite else math.nan,
    }


def pair_cost_ratios(
    observations: pd.DataFrame,
    pairs: Sequence[Pair | tuple[str, str]],
    cards: Sequence[Card | str] = (*TOKEN_CARDS, Card.USD),
    repeats: population.RepeatPolicy = population.RepeatPolicy.LATEST,
    extra: pathlib.Path | None = None,
) -> pd.DataFrame:
    """One row per (pair, card): rho_C of TREATED ``arm_a`` against CONTROL ``arm_b`` and its interval.

    Under :attr:`Card.USD` each pair is priced with ``usd-<model>``, its model's list price."""
    pairs = [pair if isinstance(pair, Pair) else Pair(*pair) for pair in pairs]
    priced: dict[str, dict] = {}
    rows: list[dict[str, object]] = []
    for key in cards:
        card = key.value if isinstance(key, Card) else key
        for pair in pairs:
            name = usd_card_name(pair.model) if card == Card.USD.value else card
            if name not in priced:
                priced[name] = arm_tokens(observations, cost.resolve(name, extra), repeats)
            rows.append(ratio_row(priced[name], pair, card))
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


def fit_xlabel(fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, width: float) -> None:
    """Centred under the axes the label can run past a narrow canvas; it then ends at the axes' right edge."""
    if ax.xaxis.label.get_window_extent(fig.canvas.get_renderer()).x1 / fig.dpi > width:
        ax.xaxis.label.set_x(1.0)
        ax.xaxis.label.set_horizontalalignment("right")


def fitted_key(fig: matplotlib.figure.Figure, handles: list, width: float) -> float:
    """The key under the axes in two columns, or in one when two overrun the placed width; its height in inches."""
    for ncol in (2, 1):
        key_in = plotstyle.legend_below(
            fig, handles, ncol=ncol, y=0.01, fontsize=TYPE.legend_pt, markerscale=1.0, **plotstyle.COMPACT_KEY
        )
        if ncol == 1 or fig.legends[-1].get_window_extent(fig.canvas.get_renderer()).width / fig.dpi <= width:
            return key_in
        fig.legends[-1].remove()
    raise AssertionError("unreachable")


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
    body_in = BODY_HEIGHT_IN
    fig, ax = plt.subplots(figsize=(width, body_in))
    step = DODGE_SPAN / max(1, len(arms))
    for i, arm in enumerate(arms):
        hue, shape = styles[arm]
        for row in drawn[drawn.arm_a == arm].itertuples():
            mark(ax, cards.index(row.card) + (i - (len(arms) - 1) / 2) * step, row, hue, shape)
    ax.set_xticks(range(len(cards)))
    ax.set_xticklabels([TICKS[Card(card)] for card in cards])
    ax.set_xlim(-0.6, len(cards) - 0.4)
    ax.set_xlabel(XLABEL, fontsize=TYPE.label_pt)
    ratio_axis(ax)
    handles: list = [
        Line2D([], [], color=styles[a][0], marker=styles[a][1], linestyle="", label=labels.get(a, a)) for a in arms
    ]
    # The model is the hue (a lighter shade per further pair), named in the caption: a swatch row
    # per model would double the key.
    fig.tight_layout()
    fit_xlabel(fig, ax, width)
    below = plotstyle.below_protrusion_in(fig, ax) + 0.04
    key_in = fitted_key(fig, handles, width)
    top = fig.subplotpars.top * body_in
    total = body_in + key_in
    fig.set_size_inches(width, total)
    fig.legends[-1].set_bbox_to_anchor((0.5, 0.02 / total), transform=fig.transFigure)
    fig.subplots_adjust(bottom=(key_in + below) / total, top=(top + key_in) / total)
    plotstyle.fill_width(fig)
    return fig


def save(fig: matplotlib.figure.Figure, stem: pathlib.Path) -> pathlib.Path:
    """Write the PDF and the PNG under ``stem`` at the figure's placed width, and close the figure."""
    return plotstyle.save(fig, stem, width_in=float(fig.get_size_inches()[0]))
