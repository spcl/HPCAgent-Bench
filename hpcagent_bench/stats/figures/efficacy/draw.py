# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The efficacy figure's DRAWING: every artist on a row, the key, and the figure builders."""

import math
import pathlib
from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Annotation
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator
from matplotlib.transforms import blended_transform_factory

from hpcagent_bench import packets, study_tags
from hpcagent_bench.stats import cost as cost_models
from hpcagent_bench.stats import palette, population, style
from hpcagent_bench.stats.figures.efficacy.layout import (
    BASELINE_BAND_FRACTION,
    BASELINE_BAND_TYPE,
    BASELINE_HEADROOM_FRACTION,
    CATEGORY_TICK_PT,
    CONTROL_MARKER,
    DEFAULT_CONFIG,
    DOT_ROW_HEIGHT_IN,
    FEW_KERNELS_NOTE,
    MEASURE_LABELS,
    MEASURE_PAD_IN,
    MEASURES,
    NAME_CHAR_EM,
    NAME_CLEARANCE_IN,
    NARROW_FIGURE_IN,
    PANEL_NAME_GID,
    PAPER_CONFIG,
    PER_COLUMN_TREATMENTS,
    STAGGER_TICK_SHARE,
    SUCCESS_HEADROOM,
    SUCCESS_TICKS,
    TICK_LABEL_CLEARANCE_IN,
    TREATMENT_NAMES,
    Comparator,
    DifferenceKey,
    DotColumn,
    FigureConfig,
    Measure,
    Panel,
    SetupRow,
    alias_footnotes,
    baseline_text,
    category_tick,
    column_limits,
    column_treatment_name,
    column_x,
    comparator_short_name,
    difference_factor,
    difference_middle,
    dot_columns,
    dot_row_grid,
    dot_row_heights,
    first_per_label,
    group_rules,
    interval_bounds,
    interval_kernels,
    is_pending,
    key_columns,
    leg_centres,
    leg_runs,
    measure_row_config,
    measure_value,
    mode_labels,
    name_layout,
    name_spans,
    near_ticks,
    panel_tag,
    parse_differences,
    placed_spans,
    relative_rows,
    retyped,
    row_label,
    row_label_fit,
    row_verdict,
    setup_rows,
    speedup_row_label,
    success_rate,
    text_band,
    treatment_marker,
    two_line_fold,
    with_comparators,
    wrapped_label,
    x_tick_step,
)
from hpcagent_bench.stats.figures.efficacy.reduce import (
    COST_SIG_LABEL,
    COST_SIG_MARK,
    EQUIVALENT_LABEL,
    EQUIVALENT_MARK,
    JOINT_SIG_LABEL,
    NO_SIGNIFICANCE,
    SCORE_SIG_LABEL,
    SCORE_SIG_MARK,
    SIG_MEASURE_NAMES,
    SPEEDUP_OVER,
    SetupPoint,
    Significance,
    any_significance,
    axis_significance,
    drawn_symbols,
    interval_note,
)
from hpcagent_bench.stats.figures.marks import Estimate, IntervalStyle, Mark, Mode, draw_estimate
from hpcagent_bench.study_tags import Marker

__all__ = [
    "SETUP_DOTS_ROW_HEIGHT_IN",
    "category_x_axis",
    "colour_legend_marks",
    "comparator_legend_marks",
    "control_key_marks",
    "control_legend_mark",
    "dot_row_key_groups",
    "dot_row_legend",
    "dot_rows_key_groups",
    "dot_rows_legend",
    "draw_baseline_band",
    "draw_category_axis",
    "draw_column_names",
    "draw_difference_arrow",
    "draw_grid_rows",
    "draw_measure_marks",
    "draw_measure_row",
    "draw_panel_letter",
    "draw_panel_subtitle",
    "draw_pending",
    "draw_setup",
    "draw_success_row",
    "draw_verdict",
    "even_row_labels",
    "few_kernel_marks",
    "figure_dot_row",
    "figure_setup_dots",
    "finish_row",
    "fit_canvas",
    "fit_column_gaps",
    "fit_legend",
    "fit_panel_names",
    "joint_significance_mark",
    "legend_tail",
    "measured_char_em",
    "minor_grid",
    "model_legend_marks",
    "note_marks",
    "packet_legend_mark",
    "panel_name_artist",
    "place_key",
    "ratio_axis",
    "ratio_ticks",
    "required_left_margin",
    "row_label_text",
    "setup_marks",
    "shape_legend_mark",
    "significance_legend_marks",
    "snap_axis_to_ticks",
    "spaced_grid",
    "stagger_crowded_ticks",
    "thin_rules",
    "token_axis",
    "treatment_legend_marks",
    "widen_y_axis_linear",
]


def significance_legend_marks(score_sig: bool, cost_sig: bool, *equivalent: bool) -> list[Line2D]:
    """One legend row per superscript the panel drew: the symbol as text, then what it means (fixed text; no test
    count or threshold). ``equivalent`` is whether either measure wears :data:`EQUIVALENT_MARK`."""
    rows = (
        (score_sig, SCORE_SIG_MARK, SCORE_SIG_LABEL),
        (cost_sig, COST_SIG_MARK, COST_SIG_LABEL),
        (any(equivalent), EQUIVALENT_MARK, EQUIVALENT_LABEL),
    )
    return [
        # The symbol goes in the TEXT: a mathtext swatch renders the asterisk as a six-pointed star.
        Line2D([], [], linestyle="none", marker="none", label=f"{symbol}  {text}")
        for drawn, symbol, text in rows
        if drawn
    ]


def model_legend_marks(models: Sequence[str], config: FigureConfig = DEFAULT_CONFIG) -> list[Line2D]:
    """One legend row per model: a neutral circle in the model's colour (the mark's shape is the packet's)."""
    return [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=palette.model_color(name),
            markersize=config.legend_marker_pt,
            label=study_tags.model_name(name),
        )
        for name in palette.in_order(models)
    ]


def control_legend_mark(
    control_over: Sequence[str],
    control_name: str = "",
    config: FigureConfig = DEFAULT_CONFIG,
) -> Line2D:
    """The hollow control's one legend row; its text covers every treatment the FIGURE reads against this control
    (``control_over``), unless ``control_name`` overrides it."""
    name = f"Baseline ({control_name or packets.control_label(list(control_over))})"
    if config.mode is Mode.RELATIVE:
        # The control is the 1x line and, on the solved row, a dash: its key swatch is the dash.
        return Line2D(
            [], [], marker="_", linestyle="none", color=palette.control_color(), markeredgewidth=2.0,
            markersize=config.legend_marker_pt, label=f"{name} = 1x",
        )  # fmt: skip
    return Line2D(
        [], [], marker=CONTROL_MARKER, linestyle="none", markerfacecolor="none", markeredgecolor=palette.control_color(),
        markeredgewidth=1.4, markersize=config.legend_marker_pt, label=name,
    )  # fmt: skip


def shape_legend_mark(shape: Marker, label: str, config: FigureConfig) -> Line2D:
    """One treated shape in neutral ink: colour is the model's on the mark, so the swatch carries
    only the shape."""
    return Line2D(
        [],
        [],
        marker=palette.marker_style(shape),
        linestyle="none",
        color=style.MUTED,
        markersize=config.legend_marker_pt,
        label=label,
    )


def packet_legend_mark(treatment: str, config: FigureConfig = DEFAULT_CONFIG) -> Line2D:
    """One packet's shape (:func:`shape_legend_mark`), named by the registry."""
    return shape_legend_mark(
        treatment_marker(treatment), TREATMENT_NAMES.get(treatment, study_tags.packet_name(treatment)), config
    )


def joint_significance_mark(score_sig: bool, cost_sig: bool, *equivalent: bool) -> list[Line2D]:
    """The significance superscripts in ONE key row, listing only the ones the figure drew, then the
    equivalence mark's own row when it drew one."""
    marks = zip((SCORE_SIG_MARK, COST_SIG_MARK), SIG_MEASURE_NAMES, strict=True)
    drawn = [(mark, name) for (mark, name), shown in zip(marks, (score_sig, cost_sig), strict=True) if shown]
    if len(drawn) < 2:
        return significance_legend_marks(score_sig, cost_sig, *equivalent)
    symbols, names = ", ".join(pair[0] for pair in drawn), ", ".join(pair[1] for pair in drawn)
    joint = Line2D([], [], linestyle="none", marker="none", label=f"{symbols}  {JOINT_SIG_LABEL}: {names}")
    return [joint, *significance_legend_marks(False, False, *equivalent)]


def place_key(
    fig: Figure,
    groups: Sequence[Sequence[Line2D]],
    ncol: int,
    *,
    y: float,
    fontsize: float,
    markerscale: float = 1.4,
    spacing: style.KeySpacing | None = None,
) -> float:
    """The key drawn below ``fig`` by meaning (:func:`key_columns`), over the most columns up to ``ncol`` that fit
    the canvas, at :func:`~hpcagent_bench.stats.style.legend_below`'s ``y``, ``fontsize``, ``markerscale`` and
    ``spacing``. Returns its height."""
    if not any(groups):
        return 0.0  # nothing to explain (a figure of stub columns): no key, and no band reserved for one
    for columns in range(max(1, ncol), 0, -1):
        handles, count = key_columns(groups, columns)
        height = style.legend_below(
            fig, handles, ncol=count, y=y, fontsize=fontsize, markerscale=markerscale, exact=True, **(spacing or {})
        )
        if height >= 0.0:
            return height
    return 0.0


def legend_tail(symbols: Significance = NO_SIGNIFICANCE, config: FigureConfig = DEFAULT_CONFIG) -> list[Line2D]:
    """The rows every key ends on: the two interval notes, and one row per significance superscript
    the figure actually drew (``symbols``, from :func:`drawn_symbols`). Every one is FIXED TEXT
    (:func:`interval_note`, :func:`significance_legend_marks`)."""
    handles = [
        Line2D([], [], linestyle="-", linewidth=1.3, color=style.MUTED, label=interval_note("Speedup")),
        Line2D(
            [],
            [],
            linestyle=config.cost_linestyle,
            linewidth=1.3,
            color=style.MUTED,
            label=interval_note("Token Cost"),
        ),
    ]
    return handles + significance_legend_marks(*symbols)


def minor_grid(ax: Axes, axis: Literal["x", "y"], kind: style.MinorKind, config: FigureConfig) -> None:
    """The shared minor ruling (:func:`~hpcagent_bench.stats.style.minor_ticks`) on ``ax``'s value
    axis ``axis`` of ``kind``, in ``config``'s own minor-grid shade and weight: unlabelled ticks
    read off the majors this module sets (:func:`x_tick_step`'s whole exponents, the token subs,
    :data:`SUCCESS_TICKS`)."""
    target = ax.yaxis if axis == "y" else ax.xaxis
    style.minor_ticks(target, kind, config.minor_grid_color, config.minor_grid_width)


def thin_rules(ax: Axes, config: FigureConfig) -> None:
    """Set the major grid and the panel frame to ``config``'s own weights. Applied after the axis
    stylers, which draw both at the weight a full-size figure wants."""
    for line in (*ax.get_xgridlines(), *ax.get_ygridlines()):
        line.set_linewidth(config.grid_width)
    for spine in ax.spines.values():
        spine.set_linewidth(config.spine_width)


def required_left_margin(fig: Figure, ax: Axes) -> float:
    """How far left of ``ax``'s own box its Y ticks and label protrude, in inches, plus
    :data:`~hpcagent_bench.stats.style.PLACED_SIDE_PAD_IN`: the left margin a figure must reserve, and no more."""
    fig.canvas.draw()
    return style.left_protrusion_in(fig, ax) + style.PLACED_SIDE_PAD_IN


def panel_name_artist(ax: Axes) -> Annotation | None:
    """The panel-name annotation :func:`draw_panel_subtitle` drew on ``ax``, if it drew one."""
    found = [text for text in ax.texts if isinstance(text, Annotation) and text.get_gid() == PANEL_NAME_GID]
    return found[-1] if found else None


def measured_char_em(ax: Axes, points: float) -> float:
    """What one character of the drawn name ACTUALLY occupies, as a fraction of its point size (replacing the
    :data:`NAME_CHAR_EM` estimate)."""
    name = panel_name_artist(ax)
    if name is None or points <= 0.0:
        return NAME_CHAR_EM
    line = max(name.get_text().split("\n"), key=len, default="")
    if not line:
        return NAME_CHAR_EM
    width = name.get_window_extent(style.renderer_of(ax.figure)).width / float(ax.figure.dpi)
    return width * style.POINTS_PER_INCH / (len(line) * points)


def draw_category_axis(ax: Axes, rows: Sequence[SetupRow], config: FigureConfig) -> None:
    """The shared categorical X of a dot-row figure: one tick per column naming what it delivered, and a light rule
    between runs; the model is never labelled (it is the colour, named in the key)."""
    ax.set_xticks(leg_centres(rows))
    ax.set_xticklabels(
        [category_tick(leg or row_label(rows, first), config) for leg, first, _ in leg_runs(rows)],
        fontsize=config.type_.tick_pt * config.category_scale,
        color=style.INK,
    )  # fmt: skip
    # Every category gets a tick mark: the row above draws its columns without
    # them, and a name set one line lower by the stagger needs a mark to its column.
    ax.tick_params(axis="x", length=CATEGORY_TICK_PT, width=config.spine_width, color=style.MUTED)


def stagger_crowded_ticks(fig: Figure, axes: Sequence[Axes], config: FigureConfig) -> None:
    """Drop every other category tick one line lower in a column whose tick labels overlap (the category band
    reserves two lines)."""
    renderer = style.renderer_of(fig)
    # Two labels closer than a third of the type size read as one word ("OMPTriton").
    name_pt = config.type_.tick_pt * config.category_scale
    gap = name_pt / 3.0 * fig.dpi / style.POINTS_PER_INCH
    for ax in axes:
        if style.crowded_ticks(ax, renderer, gap):
            for tick in ax.xaxis.get_major_ticks()[1::2]:
                # The name drops one line and its tick mark grows by half the drop: long enough to
                # point at its name, short enough to stay clear of the names on the first line.
                drop = name_pt * 1.15
                tick.set_pad(style.tick_pad(tick) + drop)
                tick.tick1line.set_markersize(tick.tick1line.get_markersize() + STAGGER_TICK_SHARE * drop)
    # Two lines may still touch: the type then steps down, the same in every column.
    style.shrink_crowded_ticks(fig, axes, name_pt, max(style.PRINT_MIN_PT, name_pt * config.category_min_scale))


def draw_panel_subtitle(ax: Axes, index: int, name: str, config: FigureConfig, wrap: int, pad: float) -> None:
    """``i) <name>`` above panel ``index``, folded at ``wrap`` with the tag attached."""
    whole = f"{panel_tag(index)} {name}"
    previous = panel_name_artist(ax)
    if previous is not None:
        previous.remove()
    drawn = ax.annotate(
        wrapped_label(whole, wrap) if wrap else whole, xy=(0.0, 1.0), xycoords="axes fraction",
        xytext=(0.0, pad), textcoords="offset points", ha="left", va="bottom",
        fontsize=config.type_.title_pt, color=style.INK, annotation_clip=False, zorder=style.MARK_Z + 3.0,
    )  # fmt: skip
    drawn.set_gid(PANEL_NAME_GID)


def draw_panel_letter(ax: Axes, index: int, config: FigureConfig) -> None:
    """A bold ``a)`` just above panel ``index``'s left edge, clear of the rotated Y label."""
    ax.text(
        0.0, 1.02, panel_tag(index), transform=ax.transAxes, ha="left", va="bottom",
        fontsize=config.type_.label_pt, fontweight="bold", color=style.INK, clip_on=False,
        zorder=style.MARK_Z + 3.0,
    )  # fmt: skip


def draw_difference_arrow(
    ax: Axes, x: float, control_value: float, treated_value: float, colour: str, measure: Measure,
    config: FigureConfig = DEFAULT_CONFIG, top: float = math.nan,
) -> None:  # fmt: skip
    """A double-headed arrow spanning one comparison's two marks, labelled with the factor between them; the label
    starts above ``top`` and is settled clear at save time (:func:`~hpcagent_bench.stats.style.settle_clear_labels`)."""
    if not (np.isfinite(control_value) and np.isfinite(treated_value)):
        return
    factor = difference_factor(control_value, treated_value, measure)
    middle = difference_middle(control_value, treated_value, measure)
    if not (np.isfinite(factor) and np.isfinite(middle)):
        return
    del colour  # the comparison is not one setup's: it is the span between two, in neutral ink
    low, high = sorted((control_value, treated_value))
    ax.errorbar(
        x, low, yerr=[[0.0], [high - low]], fmt="none", ecolor=style.FAINT,
        elinewidth=config.interval_width, capsize=config.interval_cap_pt * 2.0,
        capthick=config.interval_width, zorder=style.FILL_Z,
    )  # fmt: skip
    ax.annotate(
        style.ratio_label(factor), xy=(x, top if math.isfinite(top) else high), textcoords="offset points",
        xytext=(0.0, config.symbol_offset_pt * 0.5), ha="center", va="bottom", annotation_clip=False,
        fontsize=config.type_.annotation_pt * 0.85, color=style.REFERENCE, zorder=style.FILL_Z, gid=style.CLEAR_GID,
    )  # fmt: skip


def few_kernel_marks(rows: Sequence[SetupRow], config: FigureConfig) -> bool:
    """Whether any drawn speedup or cost mark has too few kernels for its interval."""
    return any(
        0 < interval_kernels(point, measure) < config.min_interval_kernels
        for row in rows
        for point in (row.control, row.treated)
        for measure in (Measure.SPEEDUP, Measure.COST)
    )


def snap_axis_to_ticks(ax: Axes, data_low: float, data_high: float) -> None:
    """Pull the Y limits in to the outermost TICKS that still contain the data, so both borders are labelled."""
    if not (math.isfinite(data_low) and math.isfinite(data_high)):
        return
    # A locator over the data's own range omits the ticks either side, so the range is widened first.
    log = ax.get_yscale() == "log"
    reach = max(data_high - data_low, 1.0)
    span = (data_low / 100.0, data_high * 100.0) if log else (data_low - reach, data_high + reach)
    ticks = np.asarray(ax.yaxis.get_major_locator().tick_values(*span), dtype=float)
    below, above = ticks[ticks <= data_low], ticks[ticks >= data_high]
    # A single value (a stub row holds only its 1x line) lands on one tick: no window to snap to.
    if below.size == 0 or above.size == 0 or below.max() >= above.min():
        return
    low, high = float(below.max()), float(above.min())
    steps = max(1, ticks[(ticks >= low) & (ticks <= high)].size - 1)
    ax.set_ylim(*near_ticks((low, high), (data_low, data_high), steps, log))


def draw_pending(ax: Axes, rows: Sequence[SetupRow], config: FigureConfig) -> None:
    """A :data:`~hpcagent_bench.stats.style.PENDING_MARKER` centred in every pending category, under
    ``config.mark_pending`` only. Placed in axes height so it moves no limit of the row's own scale."""
    if not config.mark_pending:
        return
    across = blended_transform_factory(ax.transData, ax.transAxes)
    xs = column_x(rows)
    for index, row in enumerate(rows):
        if is_pending(row):
            ax.text(
                xs[index], 0.5, "?", transform=across, ha="center", va="center", color=row.colour,
                fontsize=config.type_.annotation_pt * 1.4, fontweight="bold", zorder=style.MARK_Z,
                gid=style.PENDING_GID,
            )  # fmt: skip


def draw_measure_row(
    ax: Axes,
    rows: Sequence[SetupRow],
    measure: Measure,
    shape: Marker,
    significance: dict[tuple[str, str], Significance],
    config: FigureConfig = DEFAULT_CONFIG,
    ylabel: str = "",
    reference_name: str = "",
    differences: frozenset[DifferenceKey] = frozenset(),
) -> None:
    """ONE measure over the shared categorical X: per category the no-packet setup HOLLOW and the packet setup
    FILLED, dodged apart, each with its 95% interval (the success row: :func:`draw_success_row`)."""
    relative = config.mode is Mode.RELATIVE
    # The token row is a count in absolute mode; in relative mode it is a change, on the speedup's own log2 axis.
    count = measure is Measure.COST and not relative
    draw_pending(ax, rows, config)
    if measure is Measure.SUCCESS:
        draw_success_row(ax, rows, shape, config, ylabel, significance)
        return
    span = draw_measure_marks(ax, rows, measure, shape, significance, config, differences)
    if count:
        token_axis(ax, rows, config)
    else:
        # Relative mode's 1x line is named in the key, not on the line.
        ratio_axis(ax, "" if relative else reference_name, config)
    category_x_axis(ax, rows)
    ax.margins(y=config.margin)
    if not count:
        span.append(0.0)  # the 1x reference is drawn, so it is part of what the axis has to hold
        ratio_ticks(ax, span, config)
    if span:
        snap_axis_to_ticks(ax, min(span), max(span))
    if relative:
        draw_baseline_band(ax, rows, measure, span, config)
    if ylabel:
        row_label_text(ax, ylabel, config)
    finish_row(ax, config)


def setup_marks(
    row: SetupRow, shape: Marker, config: FigureConfig
) -> tuple[tuple[SetupPoint, bool, float, Marker, str], ...]:
    """A category's two marks as ``(point, filled, dodge, shape, colour)``: the control hollow, left,
    in its model's lighter shade (one model drawn twice, :data:`~hpcagent_bench.stats.palette.CONTROL_SHADE`);
    the treated setup filled, right, in the column's own shape where it has one. A comparator has one
    mark, centred."""
    if row.comparator:
        return ((row.treated, True, 0.0, row.shape, row.colour),)
    if config.mode is Mode.RELATIVE:
        # The control IS the 1x line: only the change is a mark.
        return ((row.treated, True, 0.0, row.shape or shape, row.colour),)
    return (
        (row.control, False, -config.dodge, CONTROL_MARKER, palette.lighten(row.colour, palette.CONTROL_SHADE)),
        (row.treated, True, config.dodge, row.shape or shape, row.colour),
    )


def draw_setup(
    ax: Axes, x: float, point: SetupPoint, measure: Measure, bounds: tuple[float, float],
    mark: tuple[bool, Marker, str], config: FigureConfig,
) -> tuple[list[float], float]:  # fmt: skip
    """One setup's estimate (``mark`` = filled, shape, colour) drawn as ``config.mark`` and, over enough kernels,
    its interval. Returns the values the axis has to hold and the interval's drawn top (NaN when none is drawn)."""
    filled, shape, colour = mark
    value, low, high = measure_value(point, measure)
    if interval_kernels(point, measure) < config.min_interval_kernels:
        low, high = math.nan, math.nan
    line = IntervalStyle(config.interval_width, config.cost_linestyle if measure is Measure.COST else "-",
                         config.interval_cap_pt)  # fmt: skip
    # A bar stands on the row's reference: 1x (log2 0) on a ratio row, the interval floor on a token count row.
    count = measure is Measure.COST and config.mode is Mode.ABSOLUTE
    base = bounds[0] if count and math.isfinite(bounds[0]) else 0.0
    low, high = draw_estimate(
        ax, Estimate(x, value, low, high), palette.Look(colour, shape), filled, config.mark, line, bounds=bounds,
        size=config.mark_size, width=config.bar_width, base=base,
    )  # fmt: skip
    held = [v for v in (value, low, high) if math.isfinite(v)]
    if config.mark is Mark.BAR and count and math.isfinite(value):
        held.append(base)
    return held, high


def draw_verdict(ax: Axes, x: float, value: float, mark: str, config: FigureConfig) -> None:
    """The measure's superscript ``mark`` (:func:`row_verdict`) just above the treated mark at ``(x, value)``."""
    ax.annotate(
        mark,
        (x, value),
        textcoords="offset points", xytext=(0.0, config.symbol_offset_pt), fontsize=config.type_.annotation_pt,
        color=style.INK, ha="center", va="bottom", zorder=style.MARK_Z + 2.0,
    )  # fmt: skip


def draw_measure_marks(
    ax: Axes,
    rows: Sequence[SetupRow],
    measure: Measure,
    shape: Marker,
    significance: dict[tuple[str, str], Significance],
    config: FigureConfig,
    differences: frozenset[DifferenceKey],
) -> list[float]:
    """Every category's two setups, the faint link between them, its arrow and its star; returns the
    values the Y axis has to hold."""
    span: list[float] = []
    relative = config.mode is Mode.RELATIVE
    bounds = interval_bounds(
        [measure_value(point, measure)[0] for row in rows for point in (row.control, row.treated)],
        measure is Measure.COST and not relative, config,
    )  # fmt: skip
    for x, row in zip(column_x(rows), rows, strict=True):
        if relative:
            # A comparator has no control to be relative to; a category's one mark sits on its own position.
            if not row.comparator:
                span += draw_setup(ax, x, row.treated, measure, bounds, (True, row.shape or shape, row.colour), config)[
                    0
                ]
                if verdict := row_verdict(row, measure, significance):
                    draw_verdict(ax, x, measure_value(row.treated, measure)[0], verdict, config)
            continue
        if row.comparator:
            # A comparator spends no tokens: the cost row keeps its slot empty.
            if measure is not Measure.COST:
                span += draw_setup(ax, x, row.treated, measure, bounds, (True, row.shape, row.colour), config)[0]
            continue
        top = -math.inf
        for point, filled, dodge, mark, colour in setup_marks(row, shape, config):
            held, drawn_top = draw_setup(ax, x + dodge, point, measure, bounds, (filled, mark, colour), config)
            span += held
            top = max(top, drawn_top) if math.isfinite(drawn_top) else top
        control_value = measure_value(row.control, measure)[0]
        treated_value = measure_value(row.treated, measure)[0]
        if config.link_pairs:
            ax.plot(
                [x - config.dodge, x + config.dodge], [control_value, treated_value], color=row.colour,
                linewidth=config.link_width, alpha=config.link_alpha, zorder=style.CONNECTOR_Z - 0.5,
            )  # fmt: skip
        if (row.model, row.leg) in differences:
            arrow_top = top if math.isfinite(top) else math.nan
            draw_difference_arrow(ax, x, control_value, treated_value, row.colour, measure, config, arrow_top)
        if verdict := row_verdict(row, measure, significance):
            draw_verdict(ax, x + config.dodge, treated_value, verdict, config)
    return span


def draw_baseline_band(
    ax: Axes, rows: Sequence[SetupRow], measure: Measure, span: Sequence[float], config: FigureConfig
) -> None:
    """Relative mode's baseline band: the row's foot widened below its lowest value (``span``, what the marks and
    intervals reach) by :data:`BASELINE_BAND_FRACTION`, and under each category the control's own absolute level
    (:func:`baseline_text`) in its model's colour -- the relative mark above says how the treatment moved it, the
    band what it moved from."""
    low, high = ax.get_ylim()
    reach = (max(span) - min(span)) if span else high - low
    reach = max(reach, config.min_span)
    lowest = min(low, *span) if span else low
    highest = max(high, max(span) + BASELINE_HEADROOM_FRACTION * reach) if span else high
    ax.set_ylim(lowest - BASELINE_BAND_FRACTION * reach, highest)
    for x, row in zip(column_x(rows), rows, strict=True):
        text = "" if row.comparator or row.baseline is None else baseline_text(row.baseline, measure)
        if text:
            ax.annotate(
                text, xy=(x, 0.0), xycoords=("data", "axes fraction"), xytext=(0.0, 2.0), textcoords="offset points",
                ha="center", va="bottom", fontsize=config.type_.legend_pt * BASELINE_BAND_TYPE, color=row.colour,
                zorder=style.MARK_Z,
            )  # fmt: skip


def token_axis(ax: Axes, rows: Sequence[SetupRow], config: FigureConfig) -> None:
    """The cost row's log10 token axis, or no ticks at all on a STUB column: an empty token axis
    whose ticks run 1 to 10 names a scale nothing is on."""
    if all(is_pending(row) for row in rows):
        ax.set_yticks([])
        return
    ax.set_yscale("log", base=style.LOG_BASE)
    style.value_axis(ax, "y", log_base=style.LOG_BASE)
    ax.yaxis.set_major_locator(LogLocator(base=style.LOG_BASE, subs=config.token_subs, numticks=style.LOG_NUMTICKS))
    ax.yaxis.set_major_formatter(FuncFormatter(style.decade_label))
    minor_grid(ax, "y", style.MinorKind.TOKEN, config)


def ratio_axis(ax: Axes, reference_name: str, config: FigureConfig) -> None:
    """The speedup row's log2 axis and its 1x reference line, named by ``reference_name``."""
    ax.axhline(0.0, color=style.REFERENCE, linewidth=1.0, zorder=1)
    if reference_name:
        # The denominator, ON the 1x line at its right end, so the Y label need not carry it.
        ax.annotate(
            reference_name, xy=(1.0, 0.0), xycoords=("axes fraction", "data"), xytext=(-3.0, 0.0),
            textcoords="offset points", ha="right", va="center", fontsize=config.type_.annotation_pt,
            color=style.FAINT, zorder=style.MARK_Z + 1.0,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
        )  # fmt: skip
    style.value_axis(ax, "y")
    ax.yaxis.set_major_formatter(FuncFormatter(style.log2_ratio_tick))
    minor_grid(ax, "y", style.MinorKind.LOG2, config)


def ratio_ticks(ax: Axes, span: Sequence[float], config: FigureConfig) -> None:
    """Widen the speedup row to its minimum span and space its ticks by the DATA's own ``span``, not the
    autoscaled window."""
    widen_y_axis_linear(ax, config)
    reach = max(span) - min(span) if span else config.min_span
    ax.yaxis.set_major_locator(MultipleLocator(x_tick_step(max(reach, config.min_span), config.max_ticks)))


def category_x_axis(ax: Axes, rows: Sequence[SetupRow]) -> None:
    """The row's categorical X: its limits, one tick per delivery, and the rules between deliveries."""
    ax.set_xlim(*column_limits(rows))
    ax.set_xticks(leg_centres(rows))
    group_rules(ax, rows)


def finish_row(ax: Axes, config: FigureConfig) -> None:
    """Every row's tick type and frame. The categories are named under the last row; a tick mark on
    every row points at nothing."""
    ax.tick_params(axis="both", labelsize=config.type_.tick_pt)
    ax.tick_params(axis="x", length=0.0)
    style.despine(ax)
    thin_rules(ax, config)


def draw_success_row(
    ax: Axes,
    rows: Sequence[SetupRow],
    shape: Marker,
    config: FigureConfig,
    ylabel: str,
    significance: dict[tuple[str, str], Significance] | None = None,
) -> None:
    """The success row: each setup's solved-over-served RATE on a 0-100% axis, as a mark only (the tag is fixed: a
    census has no sampling interval); the control in its lighter shade."""
    for x, row in zip(column_x(rows), rows, strict=True):
        for point, filled, dodge, mark, colour in setup_marks(row, shape, config):
            if point.served == 0:
                continue
            # A full tag sits on 100%, and the headroom above it is thinner than a mark in a half-height row.
            style.point_mark(
                ax, x + dodge, success_rate(point.solved, point.served), colour, mark, filled,
                size=config.mark_size, clip=False,
            )  # fmt: skip
        relative = config.mode is Mode.RELATIVE
        if relative and not row.comparator and row.control.served:
            # The control's own rate: a short dash in its lighter shade under the change mark, not a second mark.
            ax.hlines(
                success_rate(row.control.solved, row.control.served), x - config.dodge, x + config.dodge,
                color=palette.lighten(row.colour, palette.CONTROL_SHADE), linewidth=2.0, zorder=style.CONNECTOR_Z,
            )  # fmt: skip
        if not row.comparator and (mark := row_verdict(row, Measure.SUCCESS, significance or {})):
            rate = success_rate(row.treated.solved, row.treated.served)
            draw_verdict(ax, x if relative else x + config.dodge, rate, mark, config)
    ax.set_yticks(SUCCESS_TICKS)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100.0 * value:.0f}%"))
    minor_grid(ax, "y", style.MinorKind.COUNT, config)
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    # The dashed rule marks 100%; the headroom above it keeps the rule off the frame.
    ax.axhline(1.0, color=style.MUTED, linestyle="--", linewidth=0.6, zorder=1)
    ax.set_ylim(-SUCCESS_HEADROOM, 1.0 + SUCCESS_HEADROOM)
    category_x_axis(ax, rows)
    if ylabel:
        row_label_text(ax, ylabel, config)
    finish_row(ax, config)


def row_label_text(ax: Axes, text: str, config: FigureConfig) -> None:
    """A row's name, ROTATED along its axis and folded onto at most two lines (:func:`even_row_labels` then sizes
    every row's name alike once the layout is final)."""
    ax.set_ylabel(text if "\n" in text else two_line_fold(text), fontsize=config.type_.tick_pt)


def even_row_labels(axes: Iterable[Axes], config: FigureConfig) -> None:
    """Every row name of the figure at ONE size: the tick type size, or the largest at which every row's name
    still fits its own row -- a short row's name shrinks all of them rather than standing out smaller."""
    named = [ax for ax in axes if ax.get_ylabel()]
    if not named:
        return
    size = min(row_label_fit(ax, config.type_.tick_pt) for ax in named)
    for ax in named:
        ax.yaxis.label.set_fontsize(size)


def widen_y_axis_linear(ax: Axes, config: FigureConfig) -> None:
    """Pad ``ax``'s Y limits to at least :attr:`FigureConfig.min_span`, centred where they already
    are, on a LINEAR log2 Y (the speedup row of a dot-row figure)."""
    low, high = ax.get_ylim()
    if high - low < config.min_span:
        centre = (low + high) / 2.0
        ax.set_ylim(centre - config.min_span / 2.0, centre + config.min_span / 2.0)


def colour_legend_marks(rows: Sequence[SetupRow], config: FigureConfig) -> list[Line2D]:
    """One neutral circle per model ``rows`` drew; a comparator's key row is its own."""
    return model_legend_marks(sorted({row.model for row in rows if not row.comparator}), config)


def dot_rows_legend(
    treatment: str,
    rows: Sequence[SetupRow],
    control_name: str,
    symbols: Significance,
    config: FigureConfig = DEFAULT_CONFIG,
) -> list[Line2D]:
    """A dot-row figure's key: one swatch per colour the figure actually spent, the packet's own
    shape, the control's hollow :data:`CONTROL_MARKER`, and one row per superscript drawn."""
    return [handle for group in dot_rows_key_groups(treatment, rows, control_name, symbols, config) for handle in group]


def dot_rows_key_groups(
    treatment: str,
    rows: Sequence[SetupRow],
    control_name: str,
    symbols: Significance,
    config: FigureConfig = DEFAULT_CONFIG,
) -> list[list[Line2D]]:
    """:func:`dot_rows_legend` as its :func:`key_columns` groups: the models; the treatment and its baseline;
    the interval and band notes; the significance superscripts and alias footnotes."""
    who = colour_legend_marks(rows, config)
    what = [packet_legend_mark(treatment, config), control_legend_mark([treatment], control_name, config)]
    return note_key_groups(who, what, rows, symbols, config)


#: A single comparison's stacked row height, inches: taller than a joined row's (:data:`DOT_ROW_HEIGHT_IN`).
SETUP_DOTS_ROW_HEIGHT_IN: float = 1.5


def figure_setup_dots(
    frame: pd.DataFrame,
    stats: pd.DataFrame,
    treatment: str,
    out: pathlib.Path,
    control_name: str = "",
    config: FigureConfig = DEFAULT_CONFIG,
    measures: Sequence[Measure] = MEASURES,
    width_in: float = style.DOUBLE_COLUMN_WIDTH,
    row_height_in: float = SETUP_DOTS_ROW_HEIGHT_IN,
    labels: dict[Measure, str] | None = None,
    reference_name: str = "",
    differences: str = "",
    over: population.KernelPolicy = SPEEDUP_OVER,
    card: cost_models.CostModel | None = None,
) -> pathlib.Path:
    """ONE comparison as stacked 1-D rows: one panel per measure (``measures`` picks and orders them, ``labels``
    overrides Y labels), one column per (LLM, delivery), two marks per column."""
    import matplotlib.pyplot as plt

    rows = relative_rows(frame, over, card) if config.mode is Mode.RELATIVE else setup_rows(frame, over, card)
    if not rows:
        raise ValueError("no (model, leg) pair draws a point")
    texts = {**MEASURE_LABELS, **mode_labels(config), **(labels or {})}
    significance = axis_significance(stats)
    shape = treatment_marker(treatment)
    control_text = control_name or packets.control_label([treatment])
    # Both placements that draw ABOVE a panel need the band reserved once per row -- once at the
    # top of the figure and once between the rows.
    rows_config = measure_row_config(config, row_height_in)
    # The band above a panel holds its letter and "Higher -> Better"; 0.12in minimum keeps a two-line rotated Y label
    # on the canvas.
    note_pad = config.type_.annotation_pt * 1.7
    band = text_band(config.type_.title_pt) + note_pad / style.POINTS_PER_INCH
    category_band = text_band(config.type_.tick_pt, 2) + text_band(config.type_.label_pt)
    height = row_height_in * len(measures) + category_band + band * len(measures)
    fig, axes = plt.subplots(len(measures), 1, figsize=(width_in, height), squeeze=False, sharex=True)
    fig.set_dpi(style.SAVE_DPI)  # measure the legend and the labels at the dpi save() writes
    for index, (ax, measure) in enumerate(zip(axes[:, 0], measures, strict=True)):
        draw_panel_letter(ax, index, config)
        draw_measure_row(
            ax, rows, measure, shape, significance, rows_config, texts[measure], reference_name,
            differences=parse_differences(differences),
        )  # fmt: skip
    draw_category_axis(axes[-1, 0], rows, config)
    groups = dot_rows_key_groups(treatment, rows, control_text, drawn_symbols(stats), config)
    fig.subplots_adjust(
        left=0.1, right=0.99, top=1.0 - (band + MEASURE_PAD_IN) / height, bottom=0.01,
        hspace=0.14 + band / row_height_in,
    )  # fmt: skip
    # A figure as narrow as a wrap spends a compact key over two columns rather than one tall one.
    narrow = float(fig.get_size_inches()[0]) < NARROW_FIGURE_IN
    legend_h = place_key(
        fig, groups, 2 if narrow else config.legend_ncol, y=0.005, fontsize=config.type_.legend_pt,
        spacing=style.COMPACT_KEY if narrow else None,
    )  # fmt: skip
    # The Y labels are wrapped but still the widest thing left of the panels; reserve what they
    # MEASURE rather than a fraction guessed for one label length.
    left_in = max(required_left_margin(fig, ax) for ax in axes[:, 0])
    fig.subplots_adjust(
        left=min(0.35, left_in / width_in),
        bottom=(legend_h + category_band + MEASURE_PAD_IN) / height,
    )  # fmt: skip
    stagger_crowded_ticks(fig, axes[-1], config)
    even_row_labels(axes[:, 0], config)
    return style.save(fig, out.with_suffix(""), fixed=True, print_size=config.type_ == PAPER_CONFIG.type_)


def treatment_legend_marks(drawn: Sequence[DotColumn], config: FigureConfig) -> list[Line2D]:
    """One shape row per treatment the drawn columns wear: a per-column panel's own shapes, each
    named after its column's packet or harness, else the panel packet's (:func:`packet_legend_mark`)."""
    handles: list[Line2D] = []
    for treatment in dict.fromkeys(column.treatment for column in drawn):
        if treatment not in PER_COLUMN_TREATMENTS:
            handles.append(packet_legend_mark(treatment, config))
            continue
        shapes = {
            column_treatment_name(row.leg): row.shape
            for column in drawn if column.treatment == treatment for row in column.rows
            if row.shape and not row.comparator
        }  # fmt: skip
        handles += [shape_legend_mark(shape, name, config) for name, shape in shapes.items()]
    return handles


def control_key_marks(drawn: Sequence[DotColumn], config: FigureConfig) -> list[Line2D]:
    """The ONE control row: every panel's control wears the hollow circle, so the key names each panel's control
    once."""
    names = list(dict.fromkeys(column.control or packets.control_label([column.treatment]) for column in drawn))
    if not names:
        return []
    # Several controls in a key too narrow to list them: the caption names each panel's.
    label = names[0] if len(names) == 1 else ("Control" if not config.key_notes else f"Control: {', '.join(names)}")
    return [control_legend_mark([drawn[0].treatment], label, config)]


def note_marks(rows: Sequence[SetupRow], config: FigureConfig) -> list[Line2D]:
    """The pending "?" swatch and the few-kernels note, each only where the figure shows one."""
    handles: list[Line2D] = []
    if config.mark_pending and any(is_pending(row) for row in rows):
        handles.append(style.pending_legend_mark(config.legend_marker_pt))
    if config.key_notes and few_kernel_marks(rows, config):
        note = FEW_KERNELS_NOTE.format(config.min_interval_kernels)
        handles.append(Line2D([], [], linestyle="none", marker="none", label=note))
    return handles


def comparator_legend_marks(rows: Sequence[SetupRow], config: FigureConfig) -> list[Line2D]:
    """One key row per comparator drawn: its own shape in its own colour."""
    drawn = {comparator_short_name(row.model): row for row in rows if row.comparator}
    return [
        Line2D(
            [],
            [],
            marker=palette.marker_style(row.shape),
            linestyle="none",
            color=row.colour,
            markersize=config.legend_marker_pt,
            label=label,
        )
        for label, row in drawn.items()
    ]


def dot_row_legend(columns: Sequence[DotColumn], config: FigureConfig = DEFAULT_CONFIG) -> list[Line2D]:
    """One key for the whole row: every colour it spent, then each packet's own filled shape beside
    its control's hollow :data:`CONTROL_MARKER`."""
    return [handle for group in dot_row_key_groups(columns, config) for handle in group]


def dot_row_key_groups(columns: Sequence[DotColumn], config: FigureConfig = DEFAULT_CONFIG) -> list[list[Line2D]]:
    """:func:`dot_row_legend` as its :func:`key_columns` groups: the models (and comparators); each packet's
    shape and the baseline; the interval, band and few-kernel notes; the significance superscripts."""
    rows = [row for column in columns for row in column.rows]
    drawn = [column for column in columns if column.rows]
    # A packet drawn in two panels (a mixed panel's column and a one-packet panel) keeps one row.
    who = first_per_label(colour_legend_marks(rows, config) + comparator_legend_marks(rows, config))
    what = first_per_label(treatment_legend_marks(drawn, config)) + control_key_marks(drawn, config)
    symbols = any_significance(column.symbols for column in columns)
    return note_key_groups(who, what, rows, symbols, config)


def note_key_groups(
    who: list[Line2D], what: list[Line2D], rows: Sequence[SetupRow], symbols: Significance, config: FigureConfig
) -> list[list[Line2D]]:
    """A key's groups after the models ``who`` and the treatments ``what``: the interval, band and few-kernel notes
    and the significance superscripts."""
    if not config.key_notes:
        # The two interval line styles are named in the caption; the key keeps only the significance.
        return [who, what, note_marks(rows, config), joint_significance_mark(*symbols)]
    tail = legend_tail(symbols, config)
    return [
        who,
        what,
        [*tail[:2], *note_marks(rows, config)],
        [*joint_significance_mark(*symbols), *alias_footnotes([row.leg for row in rows])],
    ]


def fit_panel_names(
    fig: Figure,
    top: Sequence[Axes],
    tagged: Sequence[str],
    names: Sequence[str],
    spans: Sequence[float],
    config: FigureConfig,
    drawn_pt: float,
) -> None:
    """Redraw the row's names at the largest type that MEASURES inside each panel's span (the first pass used the
    :data:`NAME_CHAR_EM` estimate)."""
    if not top:
        return
    fig.canvas.draw()
    # Measured at the size the names were ACTUALLY drawn at (the first pass may already have shrunk them);
    # the names are PANEL_NAME_GID annotations, so there is no ax.title to test for.
    em = max((measured_char_em(ax, drawn_pt) for ax in top), default=NAME_CHAR_EM)
    # A name that fills its span to the last glyph runs straight into the next one ("(LLR)ii)").
    spans = [span - NAME_CLEARANCE_IN for span in spans[:-1]] + list(spans[-1:])
    size, folds = name_layout(tagged, spans, config.type_.title_pt, config.max_name_lines, em)
    draw_column_names(top, names, retyped(config, title_pt=size), folds)


def draw_column_names(top: Sequence[Axes], names: Sequence[str], config: FigureConfig, folds: Sequence[int]) -> None:
    """Each column's roman-numbered name above its top panel, folded at ``folds``."""
    for index, ax in enumerate(top):
        draw_panel_subtitle(ax, index, names[index], config, folds[index], MEASURE_PAD_IN * style.POINTS_PER_INCH)


def fit_legend(fig: Figure, groups: Sequence[Sequence[Line2D]], config: FigureConfig) -> float:
    """Draw the key below ``fig`` in the ``legend_chrome_in`` band, shrinking its type where it does not fit (never
    wider than the canvas, centred on it); returns its height, which may exceed the band (the caller grows the
    canvas)."""
    scale = 1.0
    # A wrap-width figure: a compact key, allowed the whole canvas rather than the plot body, so it
    # keeps two columns instead of one tall one.
    narrow = float(fig.get_size_inches()[0]) < NARROW_FIGURE_IN
    while True:
        height = place_key(
            fig, groups, config.legend_ncol, y=0.005, fontsize=config.type_.legend_pt * scale,
            # Centred on the whole canvas, Y-label strip included.
            markerscale=config.legend_marker_scale,
            spacing=style.COMPACT_KEY if narrow or config.compact_key else None,
        )  # fmt: skip
        if height <= config.legend_chrome_in or scale <= config.legend_min_scale:
            return height
        for legend in list(fig.legends):
            legend.remove()
        scale = max(config.legend_min_scale, scale - 0.08)


def figure_dot_row(
    panels: Sequence[Panel],
    out: pathlib.Path,
    config: FigureConfig = PAPER_CONFIG,
    measures: Sequence[Measure] = MEASURES,
    row_width_in: float = style.ACM_TEXT_WIDTH_IN,
    row_height_in: float = DOT_ROW_HEIGHT_IN,
    labels: dict[Measure, str] | None = None,
    references: Sequence[str] = (),
    control_names: Sequence[str] = (),
    differences: Sequence[str] = (),
    placeholders: Sequence[str] = (),
    over: population.KernelPolicy = SPEEDUP_OVER,
    pending: Sequence[str] = (),
    comparators: Sequence[Sequence[Comparator]] = (),
    card: cost_models.CostModel | None = None,
) -> pathlib.Path:
    """N comparisons as a GRID of stacked 1-D panels: one column per comparison (as wide as its categories), one ROW
    per measure; ``comparators[i]`` are drawn in column ``i`` (:func:`with_comparators`)."""
    columns = dot_columns(
        panels, references, control_names, differences, placeholders, over, pending,
        card, config.mode,
    )  # fmt: skip
    # A comparator (a compiler) has no no-packet control to be relative to: relative mode draws LLM pairs only.
    columns = with_comparators(columns, comparators) if config.mode is Mode.ABSOLUTE else columns
    texts = {**MEASURE_LABELS, Measure.SPEEDUP: speedup_row_label(over), **mode_labels(config), **(labels or {})}
    rows_config = measure_row_config(config, row_height_in)
    title_band = text_band(config.type_.title_pt, 2)
    category_band = text_band(config.type_.tick_pt, 2)
    heights, data_height, hspace = dot_row_heights(measures, row_height_in, config)
    # The key gets a fixed band, and grows it only when it cannot fit at its smallest type.
    # MEASURE_PAD_IN counts TWICE, once per margin that spends it (top below the title band, bottom
    # above the category band), so this first canvas already holds the data box at data_height.
    height = data_height + title_band + category_band + config.legend_chrome_in + 2 * MEASURE_PAD_IN
    widths, grid_widths = dot_row_grid(columns, row_width_in, config)
    fig, axes = spaced_grid(heights, grid_widths, (row_width_in, height))
    names = [column.title for column in columns]
    tagged = [f"{panel_tag(index)} {title}" for index, title in enumerate(names)]
    size, folds = name_layout(tagged, name_spans(widths, config), config.type_.title_pt, config.max_name_lines)
    name_config = retyped(config, title_pt=size)
    draw_grid_rows(axes, columns, measures, texts, rows_config)
    draw_column_names(axes[0], names, name_config, folds)
    for ax, column in zip(axes[-1], columns, strict=True):
        draw_category_axis(ax, column.rows, config)
    groups = dot_row_key_groups(columns, config)
    fig.subplots_adjust(
        left=config.left_chrome_in / row_width_in, right=0.995,
        top=1.0 - (title_band + MEASURE_PAD_IN) / height, bottom=0.01, hspace=hspace, wspace=0.0,
    )  # fmt: skip
    # The left margin is what the Y labels measure, not the reserve.
    fig.subplots_adjust(left=max(required_left_margin(fig, ax) for ax in axes[:, 0]) / row_width_in)
    fit_column_gaps(fig, axes)
    # The names are fitted to the columns as finally placed: a widened gap narrows every column.
    fit_panel_names(fig, list(axes[0]), tagged, names, placed_spans(axes, row_width_in), config, size)
    stagger_crowded_ticks(fig, axes[-1], config)
    # The canvas grows by the key's own height, not the reserve.
    legend_in = fit_legend(fig, groups, config)
    fit_canvas(fig, axes, row_width_in, data_height, legend_in)
    even_row_labels(np.asarray(axes).ravel(), config)
    return style.save(fig, out.with_suffix(""), fixed=True, print_size=config.type_ == PAPER_CONFIG.type_)


def draw_grid_rows(
    axes: np.ndarray, columns: Sequence[DotColumn], measures: Sequence[Measure], texts: dict[Measure, str],
    config: FigureConfig,
) -> None:  # fmt: skip
    """Every panel of the grid: one :func:`draw_measure_row` per (measure, column), the Y label on the
    first column only and the baseline's name on every row but cost."""
    for row_index, measure in enumerate(measures):
        for col_index, column in enumerate(columns):
            draw_measure_row(
                axes[row_index][col_index], column.rows, measure, column.shape, column.significance, config,
                texts[measure] if col_index == 0 else "",
                column.reference if measure is not Measure.COST else "", differences=column.differences,
            )  # fmt: skip


def spaced_grid(
    heights: Sequence[float], grid_widths: Sequence[float], size_in: tuple[float, float]
) -> tuple[Figure, np.ndarray]:
    """The dot-row canvas: one axes per (measure, column), the spacer columns between panels removed
    (:func:`dot_row_grid`), each column sharing its X; returns the figure and the panel axes."""
    import matplotlib.pyplot as plt

    fig, cells = plt.subplots(
        len(heights), len(grid_widths), figsize=size_in, squeeze=False, sharex="col",
        gridspec_kw={"width_ratios": grid_widths, "height_ratios": heights},
    )  # fmt: skip
    for gap_ax in cells[:, 1::2].flat:
        gap_ax.remove()
    fig.set_dpi(style.SAVE_DPI)
    return fig, cells[:, ::2]


def fit_canvas(fig: Figure, axes: np.ndarray, row_width_in: float, data_height: float, legend_in: float) -> None:
    """Size the canvas to the data box plus exactly the bands above and below it, measured now that
    the names, the category ticks and the key are final."""
    fig.canvas.draw()
    top_in = max(style.above_protrusion_in(fig, ax) for ax in axes[0]) + MEASURE_PAD_IN
    bottom_in = max(style.below_protrusion_in(fig, ax) for ax in axes[-1]) + legend_in + MEASURE_PAD_IN
    height = data_height + top_in + bottom_in
    fig.set_size_inches(row_width_in, height)
    fig.subplots_adjust(top=1.0 - top_in / height, bottom=bottom_in / height)


def fit_column_gaps(fig: Figure, axes: np.ndarray) -> None:
    """Widen each spacer column of :func:`figure_dot_row` until the Y tick labels of the panel to its right clear
    the panel to its left (each column has its own speedup scale)."""
    grid = axes[0][0].get_subplotspec().get_gridspec()
    ratios = list(grid.get_width_ratios())
    fig.canvas.draw()
    renderer = style.renderer_of(fig)
    span = (axes[0][-1].get_position().x1 - axes[0][0].get_position().x0) * fig.get_figwidth()
    inch = span / sum(ratios)
    needs = [
        max(
            ratios[2 * col + 1] * inch,
            max(
                (
                    (ax.get_window_extent(renderer).x0 - label.get_window_extent(renderer).x0) / fig.dpi
                    for ax in axes[:, col + 1]
                    for label in ax.get_yticklabels()
                    if label.get_text()
                ),
                default=0.0,
            )
            + TICK_LABEL_CLEARANCE_IN,
        )
        for col in range(axes.shape[1] - 1)
    ]
    panels = sum(ratios[::2])
    # Gaps of g_i inches out of a fixed span: the spacers' share G solves G = sum(g) (P + G) / span.
    total = sum(needs) * panels / max(span - sum(needs), 1e-6)
    ratios[1::2] = [need * (panels + total) / span for need in needs]
    grid.set_width_ratios(ratios)
    fig.subplots_adjust()
