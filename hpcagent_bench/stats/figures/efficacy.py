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

EVERY ARM IS A CLOUD PLUS ITS SUMMARY (SC15 Rules 5, 7 and 12): the per-kernel paired ratios,
scattered at low alpha, and the geomean crossed with its 95% interval on both axes. NOTHING IS
JOINED BY A LINE -- an arm's mark is one measurement, not a trend, the same discipline
:mod:`hpcagent_bench.stats.figures.signed` draws its own rows under.

COLOUR IS THE INTERVENTION, SHAPE IS THE MODEL. :func:`hpcagent_bench.stats.palette.color` for the
treated mark, :func:`~hpcagent_bench.stats.palette.control_color` for the control reference, and
:func:`~hpcagent_bench.stats.palette.model_markers` for the shape -- colouring by model instead
would spend the intervention's channel on the entity the shape already carries.

Several comparisons join as ONE ROW of square panels (:func:`panel_side`, :func:`figure_row`), sized
either to a panel's natural width or to a paper's own text width (:data:`~hpcagent_bench.stats.
style.ICLR_TEXT_WIDTH_IN`), never stacked: they are alternatives against one control, not a
sequence.
"""

import dataclasses
import math
import pathlib
from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Annotation
from matplotlib.ticker import FuncFormatter, MultipleLocator

from hpcagent_bench import experiment_tags, packets
from hpcagent_bench.harness import efficacy
from hpcagent_bench.stats import palette, population, rules, style, summary
from hpcagent_bench.stats.figures.per_kernel import speedup_tick_label

#: The per-kernel cloud's dot size and alpha -- matches figures.signed's own cloud so the two read
#: as one visual language.
CLOUD_SIZE: float = 12.0
CLOUD_ALPHA: float = 0.45

#: The summary mark's own size, well above the cloud's so it reads as the arm's headline point.
MARK_SIZE: float = 130.0

#: Both axes here are always a log-space Student-t interval (SC15 Rules 5/7); named once so the
#: legend text and the emitted table agree.
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
    """A base-2 major on the token-cost axis read back as the ratio it is: ``1x``, ``2x``, ``0.5x``."""
    del position
    return "1x" if value == 1.0 else f"{value:g}x"


def log2_tick(value: float, position: int = 0) -> str:
    """A major on the LOG2 speed-up axis read back as the ratio it is: the axis holds
    ``log2(ratio)`` (``+1`` is 2x, ``-1`` is 0.5x), and :func:`~hpcagent_bench.stats.figures.
    per_kernel.speedup_tick_label` already spells the ratio the same way this repo's other
    speed-up axes do."""
    del position
    return speedup_tick_label(2.0**value)


def draw_series(ax: Axes, series: Series, colour: str, shape: str, filled: bool) -> None:
    """One arm's per-kernel cloud, its crossed 95% interval and its summary mark.

    The cloud splits on ``delivered``: a verified kernel is a small dot, a served-and-never-answered
    placeholder a small cross, both in the arm's own colour -- the shape and the fill still say
    which model and which condition the mark itself belongs to.
    """
    delivered, missing = series.cloud[series.cloud.delivered], series.cloud[~series.cloud.delivered]
    if not delivered.empty:
        ax.scatter(
            delivered.x, delivered.y, s=CLOUD_SIZE, color=colour, alpha=CLOUD_ALPHA, linewidth=0, zorder=style.FILL_Z
        )
    if not missing.empty:
        ax.scatter(
            missing.x, missing.y, s=CLOUD_SIZE, marker="x", color=colour, alpha=CLOUD_ALPHA, linewidth=1.1,
            zorder=style.FILL_Z,
        )  # fmt: skip
    if np.isfinite(series.x_low) and np.isfinite(series.x_high):
        ax.hlines(
            series.y, series.x_low, series.x_high, color=colour, linewidth=1.2, alpha=0.75, zorder=style.CONNECTOR_Z
        )
    if np.isfinite(series.y_low) and np.isfinite(series.y_high):
        ax.vlines(
            series.x, series.y_low, series.y_high, color=colour, linewidth=1.2, alpha=0.75, zorder=style.CONNECTOR_Z
        )
    style.point_mark(ax, series.x, series.y, colour, shape, filled, size=MARK_SIZE)


def leg_labels(frame: pd.DataFrame) -> pd.Series:
    """``frame``'s per-arm LEG label: the language, unless ``frame`` already carries a resolved
    ``leg`` (an explicit pair list can hold several legs in one language)."""
    if "leg" in frame:
        return frame["leg"].astype(str)
    return frame["language"].astype(str).map(experiment_tags.language_name)


def significant_flags(stats: pd.DataFrame) -> dict[tuple[str, str], bool]:
    """Per (model, leg), whether EITHER axis' corrected verdict is significant -- one star per arm,
    since the mark now carries both intervals at once."""
    flags: dict[tuple[str, str], bool] = {}
    legs = leg_labels(stats)
    for (row_index, row), leg in zip(stats.iterrows(), legs, strict=True):
        verdicts = (str(row.get("score_verdict", "")), str(row.get("cost_verdict", "")))
        flags[(str(row["model"]), str(leg))] = efficacy.SIGNIFICANT in verdicts
    return flags


def family_size(stats: pd.DataFrame) -> int:
    """How many tests the figure's marks were corrected over; 0 when the table carries none."""
    if stats.empty or "family_size" not in stats:
        return 0
    return int(stats.family_size.iloc[0])


def span(counts: Sequence[int]) -> str:
    """``counts`` as an n for a legend: one number, a range, or ``?`` when there are none."""
    if not counts:
        return "?"
    ordered = sorted(counts)
    return str(ordered[0]) if ordered[0] == ordered[-1] else f"{ordered[0]}-{ordered[-1]}"


def interval_note(counts: Sequence[int], statistic: str) -> str:
    """ONE axis's own interval, named with its estimator and the kernel count it is over. Both axes
    here always draw the same estimator (:data:`GEOMEAN_METHOD`)."""
    return f"{statistic}, 95% {GEOMEAN_METHOD} Interval, n={span(counts)}"


def legend_handles(
    treatment: str, models: Sequence[str], notes: Sequence[str], control_over: Sequence[str], family: int,
    control_name: str = "",
) -> list[Line2D]:  # fmt: skip
    """The figure's one key: a MODEL is a shape in neutral ink, a CONDITION is a colour.

    ``control_over`` is every treatment the FIGURE reads against this one control -- a joined row
    takes the whole set so the hollow mark's text (:func:`hpcagent_bench.packets.control_label`) is
    not read off one panel's own treatment while the row draws several. ``control_name`` overrides
    that text outright, for a control that is not the absence of a packet.
    """
    shapes = palette.model_markers(models)
    marks = [
        Line2D(
            [],
            [],
            marker=shapes[name],
            linestyle="none",
            color=style.MUTED,
            markersize=9,
            label=experiment_tags.model_name(name),
        )  # fmt: skip
        for name in palette.in_order(models)
    ]
    return marks + [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=palette.control_color(),
            markeredgewidth=1.8,
            markersize=9,
            label=control_name or packets.control_label(list(control_over)),
        ),  # fmt: skip
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=palette.color(treatment),
            markersize=9,
            label=experiment_tags.packet_name(treatment),
        ),  # fmt: skip
        Line2D(
            [], [], marker="x", linestyle="none", color=style.MUTED, markersize=7, label=style.NOT_DELIVERED_LABEL
        ),  # fmt: skip
        *[Line2D([], [], linestyle="-", linewidth=1.3, color=style.MUTED, label=text) for text in notes],
        Line2D(
            [], [], marker="*", linestyle="none", color=style.MUTED, markersize=11, label=f"BH q < 0.05 of {family}"
        ),  # fmt: skip
    ]


#: The narrowest span, in exponent units, the X axis is ever drawn at: two whole-ratio steps either
#: side of centre, so :data:`MultipleLocator(1.0)` always lands at least THREE labelled ticks (a
#: panel whose every arm moved a kernel by a few percent otherwise autoscales to a window under one
#: octave wide, which gets exactly ONE labelled tick under a fixed whole-ratio spacing -- the same
#: trap :func:`~hpcagent_bench.stats.style.value_axis` names for a log axis under two decades).
MIN_X_SPAN: float = 2.2


def widen_x_axis(ax: Axes) -> None:
    """Pad ``ax``'s X limits to at least :data:`MIN_X_SPAN`, centred where they already are."""
    low, high = ax.get_xlim()
    if high - low < MIN_X_SPAN:
        centre = (low + high) / 2.0
        ax.set_xlim(centre - MIN_X_SPAN / 2.0, centre + MIN_X_SPAN / 2.0)


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


def style_panel(ax: Axes, compact: bool) -> None:
    """One SQUARE panel: the speed-up geomean on X as ``log2(ratio)`` (0 = no change, +1 = 2x, -1 =
    0.5x), ticks read back in ratios like every other speed-up axis in this repo
    (:func:`~hpcagent_bench.stats.figures.per_kernel.speedup_tick_label`); the token-cost ratio on Y
    (1x = no change), log-scaled. Both are log-space quantities, on their own scales, with the
    hollow control reference drawn at their shared origin ``(0, 1)`` and an equal box aspect so
    joined panels are one shape.
    """
    ax.axvline(0.0, color=style.REFERENCE, linewidth=1.0, zorder=1)
    ax.axhline(1.0, color=style.REFERENCE, linewidth=1.0, zorder=1)
    style.point_mark(ax, 0.0, 1.0, palette.control_color(), "o", False, size=MARK_SIZE)
    ax.set_yscale("log", base=2.0)
    label_pt = style.LABEL_PT * (0.68 if compact else 1.0)
    ax.set_xlabel("Geomean Speed-Up", fontsize=label_pt)
    ax.set_ylabel("Token-Cost Ratio, Treated / Control", fontsize=label_pt)
    if compact:
        ax.tick_params(axis="both", labelsize=style.TICK_PT * 0.6)
    style.value_axis(ax, "x")
    style.value_axis(ax, "y", log_base=2.0)
    ax.margins(x=0.22, y=0.22)
    widen_x_axis(ax)
    low, high = ax.get_xlim()
    ax.xaxis.set_major_locator(MultipleLocator(x_tick_step(high - low)))
    ax.xaxis.set_major_formatter(FuncFormatter(log2_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(ratio_tick))
    ax.set_box_aspect(1.0)
    style.despine(ax)


def draw_panel(
    ax: Axes,
    frame: pd.DataFrame,
    stats: pd.DataFrame,
    treatment: str,
    compact: bool = False,
    control_over: Sequence[str] = (),
    control_name: str = "",
    repeats: population.RepeatPolicy = "latest",
) -> list[Line2D]:
    """One comparison: every arm's paired cloud and summary mark, on one panel.

    ``frame`` is the RAW tagged observations (one row per record, ``skills`` True/False for the two
    conditions) -- the per-kernel cloud needs the individual kernels, which an already-reduced table
    cannot give back. Grouped by (model, leg): a leg is the language, unless ``frame`` carries an
    explicit one (:func:`leg_labels`).
    """
    treated_colour = palette.color(treatment)
    shapes = palette.model_markers(sorted(frame.model.unique()))
    significant = significant_flags(stats)
    drawn_models: set[str] = set()
    counts: list[int] = []
    for (model, leg), pair in frame.assign(leg=leg_labels(frame)).groupby(["model", "leg"]):
        series = reduce_pair(pair[~pair.skills], pair[pair.skills], repeats)
        if series is None:
            continue
        drawn_models.add(str(model))
        draw_series(ax, series, treated_colour, shapes[model], True)
        counts.append(series.kernels)
        star = " *" if significant.get((str(model), str(leg)), False) else ""
        ax.annotate(
            f"{leg}{star}",
            (series.x, series.y),
            textcoords="offset points",
            xytext=(13, 0),
            fontsize=style.ANNOTATION_PT * (0.62 if compact else 1.0),
            color=style.MUTED,
            va="center",
            zorder=style.MARK_Z + 2.0,
        )
    style_panel(ax, compact)
    notes = [interval_note(counts, "Speed-Up Geomean"), interval_note(counts, "Token-Cost Geomean")]
    return legend_handles(
        treatment, sorted(drawn_models), notes, list(control_over) or [treatment], family_size(stats), control_name
    )


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
PANEL_MARGINS: dict[str, float] = {"left": 0.135, "right": 0.97, "top": 0.90, "bottom": 0.30}

#: A single panel's side, inches, when several comparisons join in one row at their NATURAL size
#: (no target row width given).
ROW_PANEL_SIDE: float = 3.6
ROW_PANEL_GAP: float = 0.25

#: The fixed chrome around a joined row, in INCHES: a band costs the same inches whether the row's
#: panels are 1.7in or 3.6in on a side, and a constant FRACTION of the figure gives a wide row
#: whitespace it does not need and a narrow one less than it does.
ROW_TITLE_IN: float = 0.60
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
    label: str,
    out: pathlib.Path,
    control_name: str = "",
    repeats: population.RepeatPolicy = "latest",
) -> pathlib.Path:
    """ONE comparison: its square panel under a title and its own legend."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    fig.set_dpi(style.SAVE_DPI)  # measure title/legend fit at the dpi save() actually writes
    handles = draw_panel(ax, frame, stats, treatment, control_name=control_name, repeats=repeats)
    fig.subplots_adjust(**PANEL_MARGINS)
    style.legend_below(fig, handles, ncol=2, y=0.01, fontsize=style.LABEL_PT * 0.8)
    style.title(fig, label)
    untangle_labels(ax)
    return style.save(fig, out.with_suffix(""), fixed=True)


def figure_row(
    panels: Sequence[tuple[str, str, pd.DataFrame, pd.DataFrame]],
    label: str,
    out: pathlib.Path,
    row_width_in: float | None = None,
    repeats: population.RepeatPolicy = "latest",
) -> pathlib.Path:
    """N comparisons as ONE ROW of N square panels, every one against its own control.

    Square and joined side by side rather than stacked: the comparisons are alternatives, not a
    sequence, so a reader compares them by panel shape as well as by content. Each panel is
    ``(title, treatment, stats, frame)`` -- ``title`` is what the panel is CALLED (a caller's own
    "Kernel Formulation" or the packet's own :func:`~hpcagent_bench.packets.label`), ``treatment``
    is the registry key the panel is COLOURED by; they differ whenever a joined figure names its
    panels for something other than the packet itself.
    """
    import matplotlib.pyplot as plt

    n = len(panels)
    side = panel_side(n, row_width_in)
    data_width = side * n + ROW_PANEL_GAP * (n - 1)
    treatments_here = [treatment for panel_title, treatment, treated_arm, control_arm in panels]

    def build(width: float, height: float) -> tuple[Figure, list[Axes], list[Line2D]]:
        fig, axes = plt.subplots(1, n, figsize=(width, height), squeeze=False)
        fig.set_dpi(style.SAVE_DPI)  # measure title/legend/margins at the dpi save() writes
        handles_by_label: dict[str, Line2D] = {}
        for ax, (title, treatment, stats, frame) in zip(axes[0], panels, strict=True):
            for handle in draw_panel(ax, frame, stats, treatment, True, treatments_here, repeats=repeats):
                handles_by_label.setdefault(handle.get_label(), handle)
            ax.text(
                0.5, 0.98, title, transform=ax.transAxes, ha="center", va="top",
                fontsize=style.SUBTITLE_PT * 0.72, color=style.INK, zorder=7,
            )  # fmt: skip
        for ax in axes[0][1:]:
            ax.set_ylabel("")
        return fig, list(axes[0]), list(handles_by_label.values())

    def dress(fig: Figure, handles: list[Line2D]) -> float:
        """Title and legend, drawn once per pass; returns the legend's own measured height (in)."""
        style.title(fig, label)
        return style.legend_below(
            fig, handles, ncol=min(len(handles), 3), y=0.005, fontsize=style.LABEL_PT * 0.55
        )  # fmt: skip

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
        left=min(0.4, left_in / width), right=0.99, top=1.0 - ROW_TITLE_IN / height, bottom=bottom_in / height,
        wspace=0.5,
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
