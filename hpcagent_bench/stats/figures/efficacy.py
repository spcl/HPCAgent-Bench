# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The efficacy figure: did an intervention buy speed-up, and what did it cost in tokens?

ONE PANEL, TWO AXES, ONE MARK PER ARM, PAIRED PER KERNEL against the arm's own control -- the same
kernels, same model, same judge (SC15 Rule 4: a ratio ships with the costs it was taken over).
X is the speed-up geomean as ``log2(ratio)``
(:func:`hpcagent_bench.stats.summary.log2_change` of
:func:`~hpcagent_bench.stats.summary.geomean_ci`'s point and interval): 0 is no change, +1 is 2x
faster, -1 is 2x slower, +2 is 4x -- a LINEAR scale in the exponent, so a 74x kernel does not drag a
modest win halfway across the panel the way a raw ``ratio - 1`` axis would, with the ticks read back
in ratios (:func:`~hpcagent_bench.stats.figures.per_kernel.speedup_tick_label`) exactly as every
other speed-up axis in this repo, never a bare ``1x``/``2x``-ticked ratio axis on its own. Y is the
token-cost geomean, ALSO paired per kernel and treated over control, ALSO a
:func:`~hpcagent_bench.stats.summary.geomean_ci` interval -- a ratio, not a median, so a comparison
with no effect on either axis draws its mark at ``(0, 1)``, which is where the hollow control
reference sits by construction.

EACH ARM IS ITS SUMMARY MARK: the geomean crossed with its 95% interval on both axes, plus the
control reference at ``(0, 1)``. The per-kernel paired ratios (SC15 Rules 5, 7 and 12) draw only
behind ``show_cloud=True`` (default off, :func:`draw_panel`) -- one comparison's cloud already
crowds a square panel past legibility once every kernel is a dot, and the reader who wants it can
still ask. A kernel served and never delivered still scores 1x and still counts its tokens in the
geomean either way (:data:`~hpcagent_bench.stats.population.NOT_DELIVERED`); only its OWN dot or
cross stops drawing. NOTHING IS JOINED BY A LINE -- an arm's mark is one measurement, not a trend,
the same discipline :mod:`hpcagent_bench.stats.figures.signed` draws its own rows under.

COLOUR IS THE MODEL, SHAPE IS THE PACKET -- the inverse of most figures in this repo
(:mod:`hpcagent_bench.stats.palette`'s module docstring has the reasoning). One panel already
belongs to one packet, so its shape never has to separate anything; the two or three models sharing
that panel do, and a hue tells two overlapping summary marks apart at a glance where a
circle-vs-square edge does not. :func:`hpcagent_bench.stats.palette.model_color` for the mark,
:func:`~hpcagent_bench.stats.palette.control_color` for the control reference, and
:func:`~hpcagent_bench.stats.palette.packet_marker` for the one shape the whole panel wears. EVERY
DRAWN MARK IS FILLED; the control reference is the ONLY hollow one, so hollow always means "no
packet" and never doubles as a second signal.

SIGNIFICANCE IS A SUPERSCRIPT, not fill: ``*`` beside a mark's label means the SPEED-UP axis cleared
the Benjamini-Hochberg-adjusted 5% threshold for that (model, leg), ``\N{DAGGER}`` means the
TOKEN-COST axis did, over the figure's own family of tests; a mark can carry either, both or
neither. Fill was tried first and dropped: the control reference is ALREADY the hollow mark, so a
hollow treated mark read as "maybe another control" instead of "not significant." The legend spells
the rule once, in ONE row, rather than marking every point with a symbol a reader has to look up
twice.

Several comparisons join as ONE ROW of square panels (:func:`panel_side`, :func:`figure_row`), sized
either to a panel's natural width or to a paper's own text width (:data:`~hpcagent_bench.stats.
style.ICLR_TEXT_WIDTH_IN`), never stacked: they are alternatives against one control, not a
sequence. NEITHER FORM DRAWS A WHOLE-FIGURE TITLE -- a paper's caption is the title, and a joined
row's own per-panel subtitle (drawn small, :data:`FigureConfig.subtitle_pt`) is the most either
draws; :func:`figure_one`'s own optional ``title`` is the single-panel equivalent, blank by default.

EVERY SIZE A CALLER MIGHT WANT TO HAND-TUNE LIVES IN ONE PLACE, :class:`FigureConfig`: tick, label,
subtitle and legend point sizes, legend columns, mark and cloud size, the axis margin and its floor,
and the minor grid's step and shade. Change a field there (or pass a new instance to any drawing
function's ``config`` argument) rather than a magic number inside a function.
"""

import dataclasses
import math
import pathlib
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Annotation
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter

from hpcagent_bench import experiment_tags, packets
from hpcagent_bench.harness import efficacy
from hpcagent_bench.stats import palette, population, rules, style, summary
from hpcagent_bench.stats.figures.per_kernel import speedup_tick_label


@dataclasses.dataclass(frozen=True, slots=True)
class FigureConfig:
    """Every knob a reader of this figure might want to change by hand, in one place, sized for an
    ICLR-column figure by default (:data:`~hpcagent_bench.stats.style.ICLR_TEXT_WIDTH_IN`). Pass a
    replacement instance to a drawing function's ``config`` argument rather than editing a constant
    inside it; a figure's own inch geometry (panel side, row width) stays a separate concern, set by
    :func:`panel_side`/``--row-width`` -- this dataclass is TYPE size and marker/grid density, not
    layout.
    """

    #: Axis tick label size, points.
    tick_pt: float = 7.5
    #: Axis (X/Y) label size, points.
    label_pt: float = 8.0
    #: A joined row's small per-panel subtitle, and :func:`figure_one`'s own optional ``title``.
    subtitle_pt: float = 7.0
    #: Legend entry text size, points.
    legend_pt: float = 6.75
    #: The legend's column count ceiling (:func:`~hpcagent_bench.stats.style.legend_below` wraps a
    #: row that does not fit the canvas onto fewer columns, never more than this).
    legend_ncol: int = 4
    #: A summary mark's own size (points^2, matplotlib's ``s=``).
    mark_size: float = 90.0
    #: The per-kernel cloud's dot size (``show_cloud=True`` only) and alpha.
    cloud_size: float = 10.0
    cloud_alpha: float = 0.45
    #: Fractional padding ``ax.margins`` adds around the plotted extent on each axis -- SMALL: the
    #: panel fits the data and its 95% intervals plus a little air, not a fixed wide window.
    margin: float = 0.15
    #: The narrowest total span EITHER axis is ever drawn at, in octaves (log2 units): a panel whose
    #: every arm moved a kernel by a few percent still gets more than the one tick a degenerate
    #: sub-octave window would leave (:func:`~hpcagent_bench.stats.style.value_axis`'s own log2
    #: branch). Small on purpose -- large enough for >=2 ticks, not so large it reopens the "huge
    #: empty area" a wider floor left around a tightly clustered result.
    min_span: float = 1.0
    #: A minor gridline every this many octaves (0.5 = a half power of two, between each major);
    #: ``0`` draws no minor grid at all.
    minor_grid_step: float = 0.5
    #: The minor grid's own line weight and colour, lighter than the major grid
    #: (:data:`~hpcagent_bench.stats.style.RULE`) so it reads as texture under the marks, not a
    #: second reference.
    minor_grid_width: float = 0.35
    minor_grid_color: str = "#e8e8ea"


#: The default a caller draws with unless it hands a replacement in.
DEFAULT_CONFIG = FigureConfig()

#: Back-compat aliases some tests and callers pin by name; both read off :data:`DEFAULT_CONFIG`.
CLOUD_SIZE: float = DEFAULT_CONFIG.cloud_size
CLOUD_ALPHA: float = DEFAULT_CONFIG.cloud_alpha
MARK_SIZE: float = DEFAULT_CONFIG.mark_size

#: Both axes here are always a log-space Student-t interval (SC15 Rules 5/7); named once so the
#: emitted table's own column names agree with it.
GEOMEAN_METHOD: str = "log-t"

#: Columns of :func:`paired_kernels`' frame: everything a drawn interval or a significance test on
#: one comparison needs, and the raw costs Rule 4 requires travel with it.
PAIRED_COLUMNS: tuple[str, ...] = (
    "control_speedup",
    "treated_speedup",
    "control_tokens",
    "treated_tokens",
    "baseline_ns",
    "native_ns",
    "delivered",
)


def paired_kernels(
    control: pd.DataFrame, treated: pd.DataFrame, repeats: population.RepeatPolicy = "latest"
) -> pd.DataFrame:
    """One row per kernel BOTH sides cover on speed-up AND on tokens.

    The population every statistic on this comparison -- the drawn interval and the corrected
    significance test alike -- is taken over, so the two can never disagree about which kernels are
    in it. ``delivered`` is True only when BOTH sides verified an answer there; a kernel either side
    only served (:data:`~hpcagent_bench.stats.population.NOT_DELIVERED`) is a placeholder ratio, not
    a measurement, and the cloud draws it as a cross.
    """
    control_answers = population.kernel_answers(control, repeats=repeats)
    treated_answers = population.kernel_answers(treated, repeats=repeats)
    control_tokens = population.kernel_tokens(control, repeats=repeats)
    treated_tokens = population.kernel_tokens(treated, repeats=repeats)
    kernels = control_answers.index.intersection(treated_answers.index)
    kernels = kernels.intersection(control_tokens.index).intersection(treated_tokens.index)
    if len(kernels) == 0:
        return pd.DataFrame(columns=PAIRED_COLUMNS)
    has_delivered = population.DELIVERED_COLUMN in control_answers and population.DELIVERED_COLUMN in treated_answers
    frame = pd.DataFrame(
        {
            "control_speedup": control_answers.loc[kernels, "speedup"].astype(float),
            "treated_speedup": treated_answers.loc[kernels, "speedup"].astype(float),
            "control_tokens": control_tokens.loc[kernels].astype(float),
            "treated_tokens": treated_tokens.loc[kernels].astype(float),
            "baseline_ns": control_answers.loc[kernels, "baseline_ns"].astype(float)
            if "baseline_ns" in control_answers
            else math.nan,
            "native_ns": control_answers.loc[kernels, "native_ns"].astype(float)
            if "native_ns" in control_answers
            else math.nan,
            "delivered": (
                control_answers.loc[kernels, population.DELIVERED_COLUMN].astype(bool)
                & treated_answers.loc[kernels, population.DELIVERED_COLUMN].astype(bool)
            )
            if has_delivered
            else True,
        },
        index=kernels,
    )
    usable = (
        np.isfinite(frame.control_speedup)
        & (frame.control_speedup > 0.0)
        & np.isfinite(frame.treated_speedup)
        & (frame.treated_speedup > 0.0)
    )
    return frame[usable]


@dataclasses.dataclass(frozen=True, slots=True)
class Series:
    """One arm's paired-per-kernel comparison against its control: the cloud of ratios and the
    geomean summary both axes are drawn from.

    ``cloud`` is indexed by kernel and carries ``x`` (signed speed-up change), ``y`` (the raw
    token-cost ratio, treated over control) and ``delivered``. The summary fields are the same two
    quantities at their geomean, ``*_low``/``*_high`` the 95% log-t interval.
    """

    cloud: pd.DataFrame
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
    control_tokens: float
    treated_tokens: float


def reduce_pair(
    control: pd.DataFrame, treated: pd.DataFrame, repeats: population.RepeatPolicy = "latest"
) -> Series | None:
    """``(control, treated)`` as a :class:`Series`; ``None`` when they share no usable kernel."""
    paired = paired_kernels(control, treated, repeats)
    if paired.empty:
        return None
    score_ratio = (paired.treated_speedup / paired.control_speedup).to_numpy(dtype=float)
    cost_ratio = (paired.treated_tokens / paired.control_tokens).to_numpy(dtype=float)
    score, cost = summary.geomean_ci(score_ratio), summary.geomean_ci(cost_ratio)
    cloud = pd.DataFrame(
        {"x": summary.log2_changes(score_ratio), "y": cost_ratio, "delivered": paired.delivered.to_numpy(dtype=bool)},
        index=paired.index,
    )
    return Series(
        cloud=cloud,
        x=summary.log2_change(score.point),
        x_low=summary.log2_change(score.low),
        x_high=summary.log2_change(score.high),
        y=cost.point,
        y_low=cost.low,
        y_high=cost.high,
        kernels=len(paired),
        delivered=int(paired.delivered.sum()),
        baseline_ns=float(paired.baseline_ns.median()),
        native_ns=float(paired.native_ns.median()),
        control_tokens=float(paired.control_tokens.median()),
        treated_tokens=float(paired.treated_tokens.median()),
    )


def ratio_tick(value: float, position: int = 0) -> str:
    """A base-2 major on the token-cost axis read back as the ratio it is: ``1x``, ``2x``, ``1/2x``
    -- the same spelling :func:`~hpcagent_bench.stats.figures.per_kernel.speedup_tick_label` gives
    every other speed-up/ratio axis in this repo, so a ratio below 1 never prints as a decimal."""
    del position
    return speedup_tick_label(value)


def log2_tick(value: float, position: int = 0) -> str:
    """A major on the LOG2 speed-up axis read back as the ratio it is: the axis holds
    ``log2(ratio)`` (``+1`` is 2x, ``-1`` is 0.5x), and :func:`~hpcagent_bench.stats.figures.
    per_kernel.speedup_tick_label` already spells the ratio the same way this repo's other
    speed-up axes do."""
    del position
    return speedup_tick_label(2.0**value)


def draw_series(
    ax: Axes, series: Series, colour: str, shape: str, show_cloud: bool = False, config: FigureConfig = DEFAULT_CONFIG
) -> None:
    """One arm's crossed 95% interval and summary mark, plus its per-kernel cloud when
    ``show_cloud`` asks for it (default off: :func:`draw_panel`'s own docstring). Always FILLED --
    the control reference, drawn separately (:func:`style_panel`), is the only hollow mark this
    figure ever wears; significance is a superscript on the mark's own label instead
    (:func:`draw_treatment_marks`).

    The cloud splits on ``delivered``: a verified kernel is a small dot, a served-and-never-answered
    placeholder a small cross, both in the arm's own colour.
    """
    if show_cloud:
        delivered, missing = series.cloud[series.cloud.delivered], series.cloud[~series.cloud.delivered]
        if not delivered.empty:
            ax.scatter(
                delivered.x, delivered.y, s=config.cloud_size, color=colour, alpha=config.cloud_alpha, linewidth=0,
                zorder=style.FILL_Z,
            )  # fmt: skip
        if not missing.empty:
            ax.scatter(
                missing.x, missing.y, s=config.cloud_size, marker="x", color=colour, alpha=config.cloud_alpha,
                linewidth=1.1, zorder=style.FILL_Z,
            )  # fmt: skip
    if np.isfinite(series.x_low) and np.isfinite(series.x_high):
        ax.hlines(
            series.y, series.x_low, series.x_high, color=colour, linewidth=1.2, alpha=0.75, zorder=style.CONNECTOR_Z
        )
    if np.isfinite(series.y_low) and np.isfinite(series.y_high):
        ax.vlines(
            series.x, series.y_low, series.y_high, color=colour, linewidth=1.2, alpha=0.75, zorder=style.CONNECTOR_Z
        )
    style.point_mark(ax, series.x, series.y, colour, shape, True, size=config.mark_size)


def leg_labels(frame: pd.DataFrame) -> pd.Series:
    """``frame``'s per-arm LEG label: the language, unless ``frame`` already carries a resolved
    ``leg`` (an explicit pair list can hold several legs in one language)."""
    if "leg" in frame:
        return frame["leg"].astype(str)
    return frame["language"].astype(str).map(experiment_tags.language_name)


#: One mark's significance superscript, per axis -- ``*`` for the SPEED-UP axis, DAGGER for the
#: TOKEN-COST one (:data:`significance_note`'s legend text names both). Concatenated onto the
#: mark's own label, never onto the mark itself: a symbol drawn on top of a small shape is easy to
#: miss, one beside a label a reader is already reading is not.
SCORE_SIG_MARK: str = "*"
COST_SIG_MARK: str = "\N{DAGGER}"


def axis_significance(stats: pd.DataFrame) -> dict[tuple[str, str], tuple[bool, bool]]:
    """Per (model, leg), ``(score axis cleared BH, cost axis cleared BH)`` -- the two independent
    verdicts a mark's superscript reads off (:data:`SCORE_SIG_MARK`/:data:`COST_SIG_MARK`)."""
    flags: dict[tuple[str, str], tuple[bool, bool]] = {}
    legs = leg_labels(stats)
    for (_, row), leg in zip(stats.iterrows(), legs, strict=True):
        score_sig = str(row.get("score_verdict", "")) == efficacy.SIGNIFICANT
        cost_sig = str(row.get("cost_verdict", "")) == efficacy.SIGNIFICANT
        flags[(str(row["model"]), str(leg))] = (score_sig, cost_sig)
    return flags


def significance_suffix(score_sig: bool, cost_sig: bool) -> str:
    """The superscript text one mark's label carries: neither, one or both of
    :data:`SCORE_SIG_MARK`/:data:`COST_SIG_MARK`, space-joined onto the label by the caller."""
    return (SCORE_SIG_MARK if score_sig else "") + (COST_SIG_MARK if cost_sig else "")


def family_size(stats: pd.DataFrame) -> int:
    """How many tests the figure's marks were corrected over; 0 when the table carries none."""
    if stats.empty or "family_size" not in stats:
        return 0
    return int(stats.family_size.iloc[0])


def interval_note(statistic: str) -> str:
    """ONE axis's own interval, named with its estimator (:data:`GEOMEAN_METHOD`) -- fixed text, so
    every panel's copy is byte-identical and a joined row's legend (:func:`figure_row`) collapses
    them into ONE shared entry instead of one per panel."""
    return f"{statistic}, 95% {GEOMEAN_METHOD} CI"


#: The legend's ONE line spelling the significance rule (:data:`SCORE_SIG_MARK`/
#: :data:`COST_SIG_MARK`) -- fixed text, so it dedupes across a joined row's panels
#: (:func:`figure_row`) the same way :func:`interval_note` does. No per-panel test count: the
#: family size differs panel to panel and a caller after the exact number already has it from
#: ``report()``'s own printed line or the emitted stats CSV.
SIGNIFICANCE_NOTE: str = f"{SCORE_SIG_MARK} Speed-Up, {COST_SIG_MARK} Token-Cost Significant (BH-Adjusted p < 0.05)"


def model_legend_marks(models: Sequence[str]) -> list[Line2D]:
    """One legend row per model: a neutral circle in the model's own colour -- the shape the mark
    itself wears is the panel's packet, never the model's, so the swatch does not pretend to draw
    it."""
    return [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=palette.model_color(name),
            markersize=9,
            label=experiment_tags.model_name(name),
        )  # fmt: skip
        for name in palette.in_order(models)
    ]


def control_legend_mark(control_over: Sequence[str], control_name: str = "") -> Line2D:
    """The hollow control reference's one legend row. ``control_over`` is every treatment the
    FIGURE reads against this one control -- a joined row takes the whole set so the text
    (:func:`hpcagent_bench.packets.control_label`) is not read off one panel's own treatment while
    the row draws several. ``control_name`` overrides that text outright, for a control that is not
    the absence of a packet."""
    return Line2D(
        [], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=palette.control_color(),
        markeredgewidth=1.8, markersize=9, label=control_name or packets.control_label(list(control_over)),
    )  # fmt: skip


def packet_legend_mark(treatment: str) -> Line2D:
    """One packet's shape, in neutral ink -- colour is the model's on this mark, so the swatch
    carries only the shape."""
    return Line2D(
        [], [], marker=palette.packet_marker(treatment), linestyle="none", color=style.MUTED, markersize=9,
        label=experiment_tags.packet_name(treatment),
    )  # fmt: skip


def legend_tail(show_cloud: bool) -> list[Line2D]:
    """The rows every panel's legend ends on: the cloud's cross (only with ``show_cloud``), the two
    interval notes and the significance rule -- every one FIXED TEXT
    (:func:`interval_note`/:data:`SIGNIFICANCE_NOTE`), so a joined row's per-panel legends
    (:func:`figure_row`) collapse the repeats into one shared entry each instead of one per panel."""
    handles: list[Line2D] = []
    if show_cloud:
        handles.append(
            Line2D(
                [], [], marker="x", linestyle="none", color=style.MUTED, markersize=7, label=style.NOT_DELIVERED_LABEL
            )  # fmt: skip
        )
    handles += [
        Line2D([], [], linestyle="-", linewidth=1.3, color=style.MUTED, label=interval_note("Speed-Up Geomean")),
        Line2D([], [], linestyle="-", linewidth=1.3, color=style.MUTED, label=interval_note("Token-Cost Geomean")),
        Line2D([], [], linestyle="none", marker="", label=SIGNIFICANCE_NOTE),
    ]
    return handles


def legend_handles(
    treatment: str,
    models: Sequence[str],
    control_over: Sequence[str],
    control_name: str = "",
    show_cloud: bool = False,
) -> list[Line2D]:  # fmt: skip
    """The figure's one key: a MODEL is a colour, the PACKET is the one shape the whole panel wears
    (:data:`SIGNIFICANCE_NOTE` explains the superscript)."""
    handles = model_legend_marks(models) + [
        control_legend_mark(control_over, control_name),
        packet_legend_mark(treatment),
    ]
    return handles + legend_tail(show_cloud)


def multi_legend_handles(
    treatments: Sequence[str],
    models: Sequence[str],
    control_name: str = "",
    show_cloud: bool = False,
) -> list[Line2D]:
    """:func:`legend_handles` for a panel drawing SEVERAL packets against one control
    (:func:`draw_multi_panel`): a model is still one colour, but now every drawn packet gets its own
    shape row instead of the panel's single one."""
    handles = model_legend_marks(models) + [control_legend_mark(treatments, control_name)]
    handles += [packet_legend_mark(treatment) for treatment in treatments]
    return handles + legend_tail(show_cloud)


def widen_x_axis(ax: Axes, config: FigureConfig) -> None:
    """Pad ``ax``'s X limits to at least :data:`FigureConfig.min_span`, centred where they already
    are -- a panel whose every arm moved a kernel by a few percent otherwise autoscales to a window
    under one octave wide, which gets exactly ONE labelled tick under a fixed whole-ratio spacing
    (the same trap :func:`~hpcagent_bench.stats.style.value_axis` names for a log axis under two
    decades)."""
    low, high = ax.get_xlim()
    if high - low < config.min_span:
        centre = (low + high) / 2.0
        ax.set_xlim(centre - config.min_span / 2.0, centre + config.min_span / 2.0)


def widen_y_axis(ax: Axes, config: FigureConfig) -> None:
    """Pad ``ax``'s Y limits (a base-2 log scale) to at least :data:`FigureConfig.min_span` octaves,
    centred in log space where they already are -- :func:`widen_x_axis`'s own floor, applied in log
    space since a sub-floor Y window draws the single tick
    :func:`~hpcagent_bench.stats.style.value_axis` warns a sub-two-octave window leaves once its 1.5x
    sub-tick is gone."""
    low, high = ax.get_ylim()
    log_low, log_high = math.log2(low), math.log2(high)
    if log_high - log_low < config.min_span:
        centre = (log_low + log_high) / 2.0
        ax.set_ylim(2.0 ** (centre - config.min_span / 2.0), 2.0 ** (centre + config.min_span / 2.0))


#: The most labelled ticks the X axis draws before its whole-ratio spacing widens. A few outlier
#: kernels (one crashed to 1/512x, another ran away to 256x) autoscale the window past twenty
#: octaves, and :data:`MultipleLocator(1.0)` -- a tick at EVERY power of 2 -- smears that many labels
#: into one panel's width until they overlap into a solid bar.
MAX_X_TICKS: int = 9


def x_tick_step(span: float) -> int:
    """The whole-ratio spacing (in log2 units: 1 is every power of 2, 2 every power of 4, ...) that
    keeps the X axis under :data:`MAX_X_TICKS` labelled ticks for a window ``span`` wide. Doubled
    rather than picked from an arbitrary "nice number" table, so a tick always lands on an INTEGER
    log2 value -- the only kind :func:`log2_tick` spells as a clean ratio."""
    step = 1
    while span / step > MAX_X_TICKS - 1:
        step *= 2
    return step


def minor_log2_grid(ax: Axes, axis: Literal["x", "y"], config: FigureConfig) -> None:
    """A light minor gridline every :data:`FigureConfig.minor_grid_step` octaves on ``axis`` -- a
    half power of two by default, between each major (:func:`x_tick_step`/:func:`ratio_tick`'s own
    majors) -- with NO minor tick labels: a number at every half-octave would double the axis' own
    text. ``minor_grid_step`` of 0 (or a caller who wants only the major grid) draws nothing."""
    if config.minor_grid_step <= 0.0:
        return
    target = ax.yaxis if axis == "y" else ax.xaxis
    if axis == "y":
        target.set_minor_locator(LogLocator(base=2.0, subs=(2.0**config.minor_grid_step,), numticks=40))
    else:
        target.set_minor_locator(MultipleLocator(config.minor_grid_step))
    target.set_minor_formatter(NullFormatter())
    ax.grid(
        axis=axis, which="minor", color=config.minor_grid_color, linewidth=config.minor_grid_width, zorder=0
    )  # fmt: skip


#: :func:`style_panel`'s default axis labels -- a caller overrides either to fold in a cost card's
#: own weights (:data:`~hpcagent_bench.stats.cost.CostModel.key` is not this module's to name) or to
#: blank the X label where :func:`figure_row`'s ``shared_x_label`` draws it once for the whole row.
DEFAULT_XLABEL: str = "Geomean Speed-Up"
DEFAULT_YLABEL: str = "Token-Cost (x)"


def style_panel(
    ax: Axes, config: FigureConfig = DEFAULT_CONFIG, xlabel: str = DEFAULT_XLABEL, ylabel: str = DEFAULT_YLABEL
) -> None:
    """One SQUARE panel: the speed-up geomean on X as ``log2(ratio)`` (0 = no change, +1 = 2x, -1 =
    0.5x), ticks read back in ratios like every other speed-up axis in this repo
    (:func:`~hpcagent_bench.stats.figures.per_kernel.speedup_tick_label`); the token-cost ratio on Y
    (1x = no change), log-scaled. Both are log-space quantities, on their own scales, with the
    hollow control reference drawn at their shared origin ``(0, 1)`` and an equal box aspect so
    joined panels are one shape. NO TITLE: a paper's caption carries that, and the caller's own small
    subtitle (:func:`figure_one`/:func:`figure_row`) is the most a panel draws. ``xlabel``/``ylabel``
    blank to ``""`` draw no label at all -- :func:`figure_row`'s own Y-dedup and ``shared_x_label``
    both blank every panel but the one that keeps it.
    """
    ax.axvline(0.0, color=style.REFERENCE, linewidth=1.0, zorder=1)
    ax.axhline(1.0, color=style.REFERENCE, linewidth=1.0, zorder=1)
    style.point_mark(ax, 0.0, 1.0, palette.control_color(), "o", False, size=config.mark_size)
    ax.set_yscale("log", base=2.0)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=config.label_pt)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=config.label_pt)
    ax.tick_params(axis="both", labelsize=config.tick_pt)
    style.value_axis(ax, "x")
    style.value_axis(ax, "y", log_base=2.0)
    ax.margins(x=config.margin, y=config.margin)
    widen_x_axis(ax, config)
    widen_y_axis(ax, config)
    low, high = ax.get_xlim()
    ax.xaxis.set_major_locator(MultipleLocator(x_tick_step(high - low)))
    ax.xaxis.set_major_formatter(FuncFormatter(log2_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(ratio_tick))
    minor_log2_grid(ax, "x", config)
    minor_log2_grid(ax, "y", config)
    ax.set_box_aspect(1.0)
    style.despine(ax)


def draw_treatment_marks(
    ax: Axes,
    frame: pd.DataFrame,
    stats: pd.DataFrame,
    shape: str,
    label_prefix: str,
    repeats: population.RepeatPolicy,
    show_cloud: bool,
    config: FigureConfig,
) -> set[str]:
    """Every (model, leg) mark ONE packet's already-tagged ``frame`` draws (``skills`` True/False
    for the two conditions) -- the loop :func:`draw_panel` and :func:`draw_multi_panel` share, so a
    panel drawing one packet or several marks every point the same way. Returns the models actually
    drawn, for the caller's own legend.

    ``label_prefix`` goes before the point's own leg label (the language, unless ``frame`` carries
    an explicit ``leg``) -- empty for one packet's own panel, the packet's name when several packets
    share one panel and a bare language would no longer say which mark is which. The label's own
    SUFFIX is its significance superscript (:func:`significance_suffix`).
    """
    significance = axis_significance(stats)
    drawn_models: set[str] = set()
    for (model, leg), pair in frame.assign(leg=leg_labels(frame)).groupby(["model", "leg"]):
        series = reduce_pair(pair[~pair.skills], pair[pair.skills], repeats)
        if series is None:
            continue
        drawn_models.add(str(model))
        draw_series(ax, series, palette.model_color(str(model)), shape, show_cloud, config)
        score_sig, cost_sig = significance.get((str(model), str(leg)), (False, False))
        suffix = significance_suffix(score_sig, cost_sig)
        text = f"{label_prefix}{leg}{f' {suffix}' if suffix else ''}"
        ax.annotate(
            text,
            (series.x, series.y),
            textcoords="offset points",
            xytext=(13, 0),
            fontsize=config.subtitle_pt,
            color=style.MUTED,
            va="center",
            zorder=style.MARK_Z + 2.0,
        )
    return drawn_models


def draw_panel(
    ax: Axes,
    frame: pd.DataFrame,
    stats: pd.DataFrame,
    treatment: str,
    control_over: Sequence[str] = (),
    control_name: str = "",
    repeats: population.RepeatPolicy = "latest",
    show_cloud: bool = False,
    config: FigureConfig = DEFAULT_CONFIG,
    xlabel: str = DEFAULT_XLABEL,
    ylabel: str = DEFAULT_YLABEL,
) -> list[Line2D]:
    """One comparison: every arm's summary mark (and, with ``show_cloud``, its paired cloud) on one
    panel.

    ``frame`` is the RAW tagged observations (one row per record, ``skills`` True/False for the two
    conditions) -- the per-kernel cloud needs the individual kernels, which an already-reduced table
    cannot give back. Grouped by (model, leg): a leg is the language, unless ``frame`` carries an
    explicit one (:func:`leg_labels`).
    """
    drawn_models = draw_treatment_marks(
        ax, frame, stats, palette.packet_marker(treatment), "", repeats, show_cloud, config
    )
    style_panel(ax, config, xlabel, ylabel)
    return legend_handles(
        treatment, sorted(drawn_models), list(control_over) or [treatment], control_name, show_cloud
    )  # fmt: skip


def draw_multi_panel(
    ax: Axes,
    frames: dict[str, pd.DataFrame],
    stats: dict[str, pd.DataFrame],
    control_name: str = "",
    repeats: population.RepeatPolicy = "latest",
    show_cloud: bool = False,
    config: FigureConfig = DEFAULT_CONFIG,
    xlabel: str = DEFAULT_XLABEL,
    ylabel: str = DEFAULT_YLABEL,
) -> list[Line2D]:
    """SEVERAL packets sharing one panel and one control, each its own SHAPE
    (:data:`palette.packet_marker`) with the mark's colour still the model's -- the whole-panel
    version of :func:`draw_panel`, for a reader comparing every intervention against the same
    control at a glance instead of hunting across a row of panels.

    ``frames``/``stats`` are keyed by packet, each value in :func:`draw_panel`'s own shape (a
    ``skills``-tagged frame, that packet's verdict table); a packet whose frame draws no arm at all
    (empty, or every group paired to nothing) is silently absent from the legend rather than drawn
    as a shape nothing wears. Each packet's significance is corrected within its OWN family
    (:func:`~hpcagent_bench.harness.efficacy.correct_family` ran once per packet, upstream, exactly
    as a row of one-packet panels would) -- a panel drawing several packets does not re-run BH
    jointly across them.
    """
    drawn_models: set[str] = set()
    drawn_treatments: list[str] = []
    for treatment, frame in frames.items():
        if frame.empty:
            continue
        prefix = f"{experiment_tags.packet_name(treatment)} "
        models = draw_treatment_marks(
            ax, frame, stats.get(treatment, pd.DataFrame()), palette.packet_marker(treatment), prefix, repeats,
            show_cloud, config,
        )  # fmt: skip
        if models:
            drawn_treatments.append(treatment)
        drawn_models |= models
    style_panel(ax, config, xlabel, ylabel)
    return multi_legend_handles(drawn_treatments, sorted(drawn_models), control_name, show_cloud)


#: A point label's candidate places around its mark, tried in order: (dx, dy) in points, then the
#: horizontal and vertical alignment. Right of the mark first, where the label has always sat.
LABEL_PLACES: tuple[tuple[float, float, str, str], ...] = (
    (13.0, 0.0, "left", "center"),
    (-13.0, 0.0, "right", "center"),
    (0.0, 11.0, "center", "bottom"),
    (0.0, -11.0, "center", "top"),
    (13.0, 11.0, "left", "bottom"),
    (-13.0, 11.0, "right", "bottom"),
    (13.0, -11.0, "left", "top"),
    (-13.0, -11.0, "right", "top"),
)


def boxes_touch(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """Two ``(x0, y0, x1, y1)`` display boxes share area."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def untangle_labels(ax: Axes) -> None:
    """Move each point label (an :class:`~matplotlib.text.Annotation`) to the first of
    :data:`LABEL_PLACES` where its RENDERED text touches no mark and no label settled before it; a
    label with every place taken keeps the first. Call once the layout is final: a label's offset is
    in points, so a place clear before ``subplots_adjust`` need not be clear after it."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    taken: list[tuple[float, ...]] = []
    for collection in ax.collections:
        if not isinstance(collection, PathCollection):
            continue
        offsets = collection.get_offsets()
        if len(offsets):
            sizes = collection.get_sizes()
            half = (math.sqrt(float(np.max(sizes))) / 2.0 * fig.dpi / 72.0) if sizes.size else 0.0
            for px, py in collection.get_offset_transform().transform(offsets):
                taken.append((px - half, py - half, px + half, py + half))
    for note in [text for text in ax.texts if isinstance(text, Annotation)]:
        box: tuple[float, ...] = ()
        for dx, dy, ha, va in (*LABEL_PLACES, LABEL_PLACES[0]):
            note.xyann = (dx, dy)
            note.set_horizontalalignment(ha)
            note.set_verticalalignment(va)
            box = tuple(note.get_window_extent(renderer).extents)
            if not any(boxes_touch(box, other) for other in taken):
                break
        taken.append(box)


#: A single comparison's SQUARE panel side, inches, when only one is drawn.
PANEL_SIDE: float = 5.0
PANEL_SIZE: tuple[float, float] = (PANEL_SIDE + 2.0, PANEL_SIDE + 3.1)
#: ``top`` reserves only a hair: neither :func:`figure_one` nor :func:`figure_row` draws a
#: whole-figure title any more, so nothing sits above the panel box itself.
PANEL_MARGINS: dict[str, float] = {"left": 0.135, "right": 0.97, "top": 0.98, "bottom": 0.30}

#: A single panel's side, inches, when several comparisons join in one row at their NATURAL size
#: (no target row width given).
ROW_PANEL_SIDE: float = 3.6
ROW_PANEL_GAP: float = 0.25

#: The fixed chrome around a joined row, in INCHES: a band costs the same inches whether the row's
#: panels are 1.7in or 3.6in on a side, and a constant FRACTION of the figure gives a wide row
#: whitespace it does not need and a narrow one less than it does. ``ROW_TITLE_IN`` is a hair of top
#: padding only -- no whole-row title is drawn; each panel's own subtitle sits INSIDE its box
#: (:func:`figure_row`'s own ``build``).
ROW_TITLE_IN: float = 0.05
ROW_XLABEL_IN: float = 0.55

#: :func:`figure_row`'s worst-case GUESS at the legend's height, for the PROBE pass only -- big
#: enough that the probe legend never wraps onto more rows than the real one will. The real bottom
#: margin is the legend's MEASURED height (:func:`~hpcagent_bench.stats.style.legend_below` already
#: returns it), not this constant: a two-model, one-treatment legend rendered here at under half of
#: it, and the unused rest sat as dead space between the panels and the key.
ROW_LEGEND_IN: float = 1.55

#: Clearance added past a measurement, inches -- the same margin :func:`~hpcagent_bench.stats.style.
#: title` and :func:`~hpcagent_bench.stats.style.legend_below` leave past their own measured boxes.
MEASURE_PAD_IN: float = 0.08


def required_left_margin(fig: Figure, ax: Axes) -> float:
    """How far left of ``ax``'s own box its Y ticks and axis label protrude, in inches, plus
    :data:`MEASURE_PAD_IN` -- what :func:`figure_row` must reserve so a long Y label, or a
    wide-ranging axis's longest tick (``0.0078125x``, wider than the fixed fraction this used to
    reserve), never renders past the canvas's own left edge."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_left = ax.get_window_extent(renderer).x0
    label_left = ax.yaxis.get_tightbbox(renderer).x0
    protrusion_in = max(0.0, axes_left - label_left) / fig.dpi
    return protrusion_in + MEASURE_PAD_IN


def panel_side(n: int, row_width_in: float | None = None) -> float:
    """One square panel's side for ``n`` panels joined in a row.

    ``row_width_in``, when given, is the row's OWN budget in inches -- a paper's text width
    (:data:`~hpcagent_bench.stats.style.ICLR_TEXT_WIDTH_IN`, an ACM column or text width) so the
    figure drops into the page at scale 1.0 with the type still legible, rather than being shrunk by
    ``\\includegraphics``. ``None`` keeps every panel at its natural :data:`ROW_PANEL_SIDE` and lets
    the row grow with ``n``.
    """
    if row_width_in is None:
        return ROW_PANEL_SIDE
    side = (row_width_in - ROW_PANEL_GAP * (n - 1)) / n
    return max(1.2, side)


def figure_one(
    frame: pd.DataFrame,
    stats: pd.DataFrame,
    treatment: str,
    out: pathlib.Path,
    control_name: str = "",
    repeats: population.RepeatPolicy = "latest",
    show_cloud: bool = False,
    title: str = "",
    config: FigureConfig = DEFAULT_CONFIG,
) -> pathlib.Path:
    """ONE comparison: its square panel and its own legend. NO whole-figure title -- a paper's
    caption is that; ``title``, blank by default, draws a small subtitle INSIDE the panel, the same
    place :func:`figure_row` draws one for each of its own panels."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    fig.set_dpi(style.SAVE_DPI)  # measure the legend's fit at the dpi save() actually writes
    handles = draw_panel(
        ax, frame, stats, treatment, control_name=control_name, repeats=repeats, show_cloud=show_cloud, config=config
    )
    fig.subplots_adjust(**PANEL_MARGINS)
    style.legend_below(fig, handles, ncol=config.legend_ncol, y=0.01, fontsize=config.legend_pt)
    if title:
        ax.text(
            0.5, 0.98, title, transform=ax.transAxes, ha="center", va="top", fontsize=config.subtitle_pt,
            color=style.INK, zorder=7,
        )  # fmt: skip
    untangle_labels(ax)
    return style.save(fig, out.with_suffix(""), fixed=True)


#: One panel of a joined row: ``title`` (what the panel is CALLED) plus either the SINGLE-treatment
#: shape (:func:`draw_panel`'s own ``treatment: str``, ``stats``/``frame`` each one table) or the
#: MULTI-treatment one (:func:`draw_multi_panel`'s ``treatments: Sequence[str]``, ``stats``/``frame``
#: each a ``{treatment: table}`` dict, several packets sharing this one panel and control).
Panel = (
    tuple[str, str, pd.DataFrame, pd.DataFrame]
    | tuple[str, Sequence[str], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]
)


def flat_treatments(spec: str | Sequence[str]) -> list[str]:
    """``spec`` as a flat list, whether it names one treatment or several."""
    return [spec] if isinstance(spec, str) else list(spec)


def resolve_row_repeats(
    repeats: population.RepeatPolicy | Sequence[population.RepeatPolicy], n: int
) -> list[population.RepeatPolicy]:
    """``repeats`` as one policy per panel: a bare policy repeats for all ``n``; a sequence must
    already have length ``n`` -- a git-scicomp panel (designed 3x repeats, median) and an
    llr-focus40 panel (reruns, latest) share no policy, so ONE row's panels are never forced onto
    ONE value."""
    if isinstance(repeats, str):
        return [repeats] * n
    resolved = list(repeats)
    if len(resolved) != n:
        raise ValueError(f"repeats names {len(resolved)} polic{'y' if len(resolved) == 1 else 'ies'}, panels {n}")
    return resolved


def figure_row(
    panels: Sequence[Panel],
    out: pathlib.Path,
    row_width_in: float | None = None,
    repeats: population.RepeatPolicy | Sequence[population.RepeatPolicy] = "latest",
    show_cloud: bool = False,
    config: FigureConfig = DEFAULT_CONFIG,
    shared_x_label: bool = False,
    xlabel: str = DEFAULT_XLABEL,
    ylabel: str = DEFAULT_YLABEL,
) -> pathlib.Path:
    """N comparisons as ONE ROW of N square panels, every one against its own control.

    Square and joined side by side rather than stacked: the comparisons are alternatives, not a
    sequence, so a reader compares them by panel shape as well as by content. Each panel is
    ``(title, treatment, stats, frame)`` -- ``title`` is what the panel is CALLED (a caller's own
    "Kernel Formulation" or the packet's own :func:`~hpcagent_bench.packets.label`), drawn small
    INSIDE the panel's own box (no whole-row title: a paper's caption is that); ``treatment`` is the
    registry key (or keys, :data:`Panel`) the panel is SHAPED by, differing from ``title`` whenever a
    joined figure names its panels for something other than the packet itself. ``repeats`` is either
    ONE policy for every panel or one PER panel (:func:`resolve_row_repeats`) -- a joined row's
    comparisons need not share a repeat-reduction policy.

    Every panel already shares ONE Y label (the leftmost panel's; the rest blank theirs) since every
    panel reads the same quantity. ``shared_x_label`` gives the X axis -- itself already the SAME
    quantity in every panel -- the identical treatment: every panel's own X label is blanked and
    ``xlabel`` is drawn ONCE, centred under the whole row, instead of repeating verbatim under each
    square. Off by default so an existing caller's per-panel X label is unchanged.
    """
    import matplotlib.pyplot as plt

    n = len(panels)
    side = panel_side(n, row_width_in)
    data_width = side * n + ROW_PANEL_GAP * (n - 1)
    treatments_here = [
        t for panel_title, treatment, treated_arm, control_arm in panels for t in flat_treatments(treatment)
    ]
    panel_xlabel = "" if shared_x_label else xlabel
    panel_repeats = resolve_row_repeats(repeats, n)

    def build(width: float, height: float) -> tuple[Figure, list[Axes], list[Line2D]]:
        fig, axes = plt.subplots(1, n, figsize=(width, height), squeeze=False)
        fig.set_dpi(style.SAVE_DPI)  # measure legend/margins at the dpi save() writes
        handles_by_label: dict[str, Line2D] = {}
        rows = zip(axes[0], panels, panel_repeats, strict=True)
        for ax, (title, treatment, stats, frame), one_repeats in rows:
            if isinstance(treatment, str):
                handles = draw_panel(
                    ax, frame, stats, treatment, treatments_here, repeats=one_repeats, show_cloud=show_cloud,
                    config=config, xlabel=panel_xlabel, ylabel=ylabel,
                )  # fmt: skip
            else:
                handles = draw_multi_panel(
                    ax, frame, stats, repeats=one_repeats, show_cloud=show_cloud, config=config,
                    xlabel=panel_xlabel, ylabel=ylabel,
                )  # fmt: skip
            for handle in handles:
                handles_by_label.setdefault(handle.get_label(), handle)
            if title:
                ax.text(
                    0.5, 0.98, title, transform=ax.transAxes, ha="center", va="top", fontsize=config.subtitle_pt,
                    color=style.INK, zorder=7,
                )  # fmt: skip
        for ax in axes[0][1:]:
            ax.set_ylabel("")
        return fig, list(axes[0]), list(handles_by_label.values())

    def dress(fig: Figure, handles: list[Line2D]) -> float:
        """The shared legend, drawn once per pass; returns its own measured height (in)."""
        return style.legend_below(
            fig, handles, ncol=min(len(handles), config.legend_ncol), y=0.005, fontsize=config.legend_pt
        )

    # Pass 1 (a throwaway figure): :data:`ROW_LEGEND_IN` is a worst-case guess at how tall the
    # legend's row wrap will come out and :data:`PANEL_MARGINS`-style left fraction is a guess at
    # how far a Y label and its ticks protrude -- both measured for real here, so pass 2 reserves
    # exactly what this row's own content needs instead of a constant sized for a wider one.
    probe_height = side + ROW_TITLE_IN + ROW_XLABEL_IN + ROW_LEGEND_IN
    probe_fig, probe_axes, probe_handles = build(data_width, probe_height)
    legend_h = dress(probe_fig, probe_handles)
    left_in = required_left_margin(probe_fig, probe_axes[0])
    plt.close(probe_fig)

    bottom_in = ROW_XLABEL_IN + legend_h + MEASURE_PAD_IN
    height = side + ROW_TITLE_IN + bottom_in
    # A page-budgeted row (``row_width_in`` given) keeps its CONTRACTED width and shrinks the data
    # area to fit the Y label inside it -- the promise that width exists to keep. A natural row
    # makes none, so the label gets its OWN canvas instead of eating into the square panel's side.
    width = data_width if row_width_in is not None else data_width + left_in
    fig, axes, handles = build(width, height)
    dress(fig, handles)
    fig.subplots_adjust(
        left=min(0.4, left_in / width),
        right=0.99,
        top=1.0 - ROW_TITLE_IN / height,
        bottom=bottom_in / height,
        wspace=0.5,
    )  # fmt: skip
    if shared_x_label:
        # Between the legend (below, up to legend_h/height) and each panel's own tick numbers
        # (above, right under bottom_in/height) -- the same ROW_XLABEL_IN band a per-panel X label
        # used to sit in, now drawn once for the row instead of once per panel.
        fig.text(
            0.5, (legend_h + MEASURE_PAD_IN) / height, xlabel, ha="center", va="bottom",
            fontsize=config.label_pt, color=style.INK,
        )  # fmt: skip
    for ax in axes:
        untangle_labels(ax)
    return style.save(fig, out.with_suffix(""), fixed=True)


def pairs_table(frame: pd.DataFrame, repeats: population.RepeatPolicy = "latest") -> pd.DataFrame:
    """One row per (model, leg): the drawn point behind :func:`draw_panel`'s mark, as the CSV record
    beside the figure (SC15 Rule 4: the costs a ratio was taken over travel with it)."""
    rows = []
    for (model, leg), pair in frame.assign(leg=leg_labels(frame)).groupby(["model", "leg"]):
        series = reduce_pair(pair[~pair.skills], pair[pair.skills], repeats)
        if series is None:
            continue
        rows.append(
            {
                "model": model,
                "leg": leg,
                "score_change": series.x,
                "score_change_low": series.x_low,
                "score_change_high": series.x_high,
                "cost_ratio": series.y,
                "cost_ratio_low": series.y_low,
                "cost_ratio_high": series.y_high,
                "kernels": series.kernels,
                "delivered": series.delivered,
                "baseline_ns": series.baseline_ns,
                "native_ns": series.native_ns,
                "control_tokens": series.control_tokens,
                "treated_tokens": series.treated_tokens,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    rules.require_costs(table, "score_change", ["baseline_ns", "native_ns"])
    rules.require_costs(table, "cost_ratio", ["control_tokens", "treated_tokens"])
    rules.require_interval(table, "score_change", "score_change_low", "score_change_high")
    return rules.require_interval(table, "cost_ratio", "cost_ratio_low", "cost_ratio_high")
