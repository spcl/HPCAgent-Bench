# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Did the SKILLS packet buy speed-up, and what did it cost in tokens? One point per model+language.

ONE experiment, split by its own treatment: the skills arms against the no-skills arms of the same
campaign. That is the comparison the campaign was designed to make, and it is paired -- same
kernels, same models, same judge, same week -- where a before/after across two campaigns also
carries every other thing that changed between them.

Two ratios, BEFORE against AFTER, so each axis is a change rather than a level and the two
experiments' absolute scales stop mattering:

    score  rho_S = speed-up(skills) / speed-up(no skills)   -- right is faster
    cost   rho_C = tokens(no skills) / tokens(skills)       -- UP is cheaper

``rho_C`` is inverted on purpose: written the other way round, "up" would mean "spent more" and the
top-right corner -- where a reader's eye goes -- would be the worst outcome rather than the best.
With this orientation the quadrants read directly: up-and-right is better in both, down-and-right
is faster but it costs.

Both are ratios, so both are aggregated in LOG space as a GEOMETRIC MEAN over kernels, with the
t interval and paired t test on that mean log. The sampling unit is the KERNEL and the two sides are
paired on it; a kernel an arm ran more than once is reduced by ``--repeats`` first.

THE MARKS ARE CORRECTED. One figure is not one test: three models x two languages x two axes is
twelve paired tests, and twelve uncorrected 5% thresholds paint at least one star on 46% of
figures where nothing happened. The family is therefore declared once (:func:`points` tests every
(model, language) on both axes), the p values are Benjamini-Hochberg adjusted across it, and the
star is gated on the ADJUSTED value -- an uncorrected star and a corrected star look identical to a
reader, so the legend names the correction and the size of the family. A point whose pairing is
too small for the test to run at all reads ``underpowered`` and is never starred.

"""

import argparse
import math
import pathlib
import sys
from collections.abc import Sequence

import numpy as np
import pandas as pd

from hpcagent_bench import experiment_tags, experiments, packets
from hpcagent_bench.harness import efficacy
from hpcagent_bench.stats import palette, population, rules, summary
from hpcagent_bench.stats import style as plotstyle

plotstyle.apply()
import matplotlib.pyplot as plt  # pyplot must follow plotstyle.apply()
from matplotlib.collections import PathCollection
from matplotlib.text import Annotation

#: A ratio this far from 1.0 is inside the "no change" band for labelling purposes only; the star
#: is decided by the interval, never by this.
NEUTRAL: float = 1.0

#: :func:`points`' row shape, so an empty family is an empty DataFrame carrying these columns
#: rather than one with none at all -- ``pd.DataFrame([])`` has no columns, and ``.dropna(subset=...)``
#: on THAT raises a bare ``KeyError`` instead of reading as "no (model, language) pair to draw".
POINT_COLUMNS: tuple[str, ...] = (
    "model",
    "language",
    "score",
    "score_low",
    "score_high",
    "cost",
    "cost_low",
    "cost_high",
    "kernels",
    "score_p",
    "cost_p",
)

#: Ticks in RATIO units on a log2 axis. Labelled as ratios, not as exponents: a reader wants to see
#: "2x", not "1".
RATIO_TICKS: tuple[float, ...] = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0)

#: Unlabelled minor ratios between the majors above -- powers of 2**(1/16). A log2 axis spanning
#: less than one octave gets no minor lines at all from a LogLocator (its subdivisions are powers
#: of two, and there is at most one inside the window), which left this plot with a four-line grid
#: to interpolate against.
RATIO_MINORS: tuple[float, ...] = tuple(2.0 ** (n / 16.0) for n in range(-32, 33))


def tick_label(value: float) -> str:
    """``1.0`` is the no-change line and says so; everything else is a plain ratio."""
    if value == NEUTRAL:
        return "1x"
    return f"{value:g}x"


#: Powers of sqrt(2), so a narrow window still gets several labelled ticks. A ratio plot whose
#: data spans 0.7x to 1.2x showed exactly two ticks on the decade spacing, and a reader cannot
#: interpolate a value from two ticks.


def ratio_with_ci(before: pd.Series, after: pd.Series, invert: bool) -> tuple[float, float, float, float]:
    """Geomean ratio over the kernels BOTH sides cover, with its t interval and paired t-test p.

    The GEOMETRIC MEAN of the per-kernel ratios, the statistic every overall ratio in this repo is
    reported as. The estimator, its interval and its p value come from
    :func:`hpcagent_bench.stats.summary.paired_geomean`, so all three describe the one mean log --
    which a bootstrap mean beside a separate rank test does not.

    PAIRED, by kernel: both sides ran the same 40 kernels, and the pairing is most of the
    precision here. Mann-Whitney is the unpaired sibling and would throw that away -- with n=40
    kernels and per-kernel spread far larger than the treatment effect, the unpaired test sees
    almost nothing. Restricted to the shared kernels for the same reason as before: a ratio taken
    over two different kernel sets is a change plus whatever the sets differ by.

    Returns ``(ratio, low, high, p)`` on the RATIO scale; all four are NaN when the sides share
    nothing usable.
    """
    shared = before.index.intersection(after.index)
    if len(shared) == 0:
        return (float("nan"),) * 4
    b, a = before.loc[shared].to_numpy(dtype=float), after.loc[shared].to_numpy(dtype=float)
    keep = (b > 0) & (a > 0)
    b, a = b[keep], a[keep]
    if b.size == 0:
        return (float("nan"),) * 4
    change = summary.paired_geomean(np.log(b / a) if invert else np.log(a / b))
    return (
        float(np.exp(change.estimate)),
        float(np.exp(change.low)),
        float(np.exp(change.high)),
        change.pvalue,
    )


def points(before: pd.DataFrame, after: pd.DataFrame, repeats: population.RepeatPolicy = "latest") -> pd.DataFrame:
    """One row per (model, language) present in both experiments, with the flags corrected.

    THE FAMILY IS THIS TABLE: every (model, language) the two sides share, on both axes. It is
    built here, in one place, rather than left to whichever loop a reader of the figure imagines --
    which is how twelve tests came to be thresholded one at a time. ``score_verdict`` and
    ``cost_verdict`` are the only columns a mark or a sentence may be taken from; ``score_p`` is
    the raw test and ``score_p_adjusted`` is it corrected across the family.

    A leg is a (model, language) BOTH sides landed a GRADED answer for. One that appears only on
    the call rows -- an arm that ran and never had a submission persisted -- is not a comparison and
    is absent rather than entered at zero.

    A pair is drawn inside ONE denominator. ``one_denominator`` raises rather than pooling a slice
    whose two sides were divided by different references, because their quotient is not a
    comparison of the two conditions.
    """
    rows = []
    graded_before, graded_after = before[before.record == "submission"], after[after.record == "submission"]
    keys = sorted(
        set(map(tuple, graded_before[["model", "language"]].drop_duplicates().to_numpy()))
        & set(map(tuple, graded_after[["model", "language"]].drop_duplicates().to_numpy()))
    )
    for model, language in keys:
        b = before[(before.model == model) & (before.language == language)]
        a = after[(after.model == model) & (after.language == language)]
        graded = pd.concat([b, a])
        graded = graded[graded.record == "submission"]
        population.one_denominator(graded.baseline.tolist(), label=f"{model}/{language}")
        before_score = population.kernel_answers(b, repeats=repeats).speedup
        after_score = population.kernel_answers(a, repeats=repeats).speedup
        score, s_low, s_high, s_p = ratio_with_ci(before_score, after_score, False)
        before_cost, after_cost = (
            population.kernel_tokens(b, repeats=repeats),
            population.kernel_tokens(a, repeats=repeats),
        )
        cost, c_low, c_high, c_p = ratio_with_ci(before_cost, after_cost, True)
        rows.append(
            {
                "model": model,
                "language": language,
                "score": score,
                "score_low": s_low,
                "score_high": s_high,
                "cost": cost,
                "cost_low": c_low,
                "cost_high": c_high,
                "kernels": len(before_score.index.intersection(after_score.index)),
                # The raw test. The verdict columns below are what may be read as a finding, and
                # they come from the whole family at once -- reading a threshold off one row is the
                # multiplicity error this table exists to avoid.
                "score_p": s_p,
                "cost_p": c_p,
            }
        )
    # ``columns=POINT_COLUMNS`` is the fix: ``rows`` empty (no shared (model, language) at all) must
    # still produce a frame that HAS a "score"/"cost" column to drop NaN out of, or dropna raises a
    # bare KeyError that reads as a crash rather than as "this treatment paired with nothing".
    frame = pd.DataFrame(rows, columns=list(POINT_COLUMNS)).dropna(subset=["score", "cost"])
    if frame.empty:
        return frame
    # Interleaved score, cost, score, cost ... so each row's pair of verdicts comes back adjacent.
    family = [value for row in frame.itertuples(index=False) for value in (row.score_p, row.cost_p)]
    verdicts = efficacy.correct_family(family)
    return frame.assign(
        score_p_adjusted=[v.adjusted for v in verdicts[0::2]],
        cost_p_adjusted=[v.adjusted for v in verdicts[1::2]],
        score_verdict=[v.label for v in verdicts[0::2]],
        cost_verdict=[v.label for v in verdicts[1::2]],
        family_size=sum(1 for v in verdicts if math.isfinite(v.adjusted)),
    )


def draw_interval(ax: plt.Axes, row: pd.Series, colour: str) -> None:
    """One mark's two intervals, as thin crossed whiskers: speed-up across, spend up. A mark too thin
    for an interval on an axis draws none there."""
    x, y = float(row.log2_speedup), float(row.tokens)
    x_low, x_high, y_low, y_high = (
        float(row[c]) for c in ("log2_speedup_low", "log2_speedup_high", "tokens_low", "tokens_high")
    )
    if np.isfinite(x_low) and np.isfinite(x_high):
        ax.hlines(y, x_low, x_high, color=colour, linewidth=1.1, alpha=0.55, zorder=1)
    if np.isfinite(y_low) and np.isfinite(y_high):
        ax.vlines(x, y_low, y_high, color=colour, linewidth=1.1, alpha=0.55, zorder=1)


def draw_absolute(ax, frame: pd.DataFrame, stats: pd.DataFrame, treatment: str, compact: bool = False) -> list:
    """Geomean speed-up against median spend, with each arm's TWO CONDITIONS joined.

    ``treatment`` names the packet on the filled side: the hollow mark's legend text is
    :func:`hpcagent_bench.packets.control_label` for it ("No Skill Packet" only when ``treatment``
    is itself a skill packet, "No Packet" otherwise) and the filled mark's is its own registry
    display name (:func:`hpcagent_bench.experiment_tags.packet_name`) -- never a generic "Skills"
    that misnames a CPF or perf-playbook panel as if it were a skill.

    ABSOLUTE, not a ratio, and that is what makes the connector mean something: a ratio plot has
    one point per arm and nothing to join, so a line drawn on it could only connect two arms --
    which is what it did, joining an arm's C point to its Fortran point and inviting the reading
    that a language change is the treatment. Here each (model, language) has a hollow mark without
    the packet and a filled one with it, and the segment between them IS the treatment effect for
    that arm: its length is the size, its direction the sign, on both axes at once.

    The filled mark is drawn LAST so it sits above the connector and above its own hollow partner
    -- the two land on top of each other whenever the packet changed little, and the "with skills"
    position is the one a reader is looking for.

    ``compact`` is for a SQUARE panel a fraction of :data:`PANEL_SIZE` (:func:`figure_treatments`,
    joining several treatments side by side): the four quadrant captions are fixed-size text sized
    for the full panel and are dropped rather than shrunk into an unreadable smear, and the
    per-point language label shrinks so it does not swallow its neighbour's point.
    """
    hues = palette.model_colors(sorted(frame.model.unique()))
    shapes = palette.model_markers(sorted(frame.model.unique()))
    # Gated on the ADJUSTED verdict, on either axis. An arm the packet moved on tokens alone is a
    # real finding, so the mark fires on either -- which is exactly why both axes are one family.
    significant = {
        (row.model, row.language): efficacy.SIGNIFICANT in (row.score_verdict, row.cost_verdict)
        for row in stats.itertuples(index=False)
    }
    # Only a model that actually lands BOTH marks earns a legend entry: ``hues``/``shapes`` are
    # keyed off every model the control side ran, and a model this treatment never touched (the
    # CPF page figure's control carries Kimi from the campaign's OTHER treatments) fell through the
    # `continue` below with a colour already reserved in ``hues`` -- so the legend named a model the
    # panel never draws a point for.
    drawn_models: set[str] = set()
    for (model, language), pair in frame.groupby(["model", "language"]):
        colour, shape = hues[model], shapes[model]
        off, on = pair[~pair.skills], pair[pair.skills]
        if len(off) != 1 or len(on) != 1:
            continue
        drawn_models.add(model)
        # An ELBOW, not a diagonal. The straight segment between two measured points runs through
        # coordinates that were never measured, and on a plot whose whole subject is where an arm
        # LANDED a reader takes the path for data -- as if the packet moved the arm along it. The
        # right angle is visibly a connector: it says these two marks are one arm and claims
        # nothing about the space between them. Horizontal first, so the corner sits under the
        # "with skills" mark and the vertical leg reads as the change in spend.
        x_off, x_on = float(off.log2_speedup.iloc[0]), float(on.log2_speedup.iloc[0])
        y_off, y_on = float(off.tokens.iloc[0]), float(on.tokens.iloc[0])
        for side in (off, on):
            draw_interval(ax, side.iloc[0], colour)
        ax.plot(
            [x_off, x_on, x_on],
            [y_off, y_off, y_on],
            linestyle=(0, (3, 3)),
            linewidth=0.9,
            color=colour,
            alpha=0.8,
            zorder=2,
        )
        ax.scatter(
            off.log2_speedup,
            off.tokens,
            s=130,
            marker=shape,
            facecolor="none",
            edgecolor=colour,
            linewidth=1.8,
            zorder=3,
        )
        ax.scatter(
            on.log2_speedup,
            on.tokens,
            s=130,
            marker=shape,
            color=colour,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        star = " *" if significant.get((model, language)) else ""
        ax.annotate(
            f"{experiment_tags.language_name(language)}{star}",
            (float(on.log2_speedup.iloc[0]), float(on.tokens.iloc[0])),
            textcoords="offset points",
            xytext=(13, 0),
            fontsize=plotstyle.ANNOTATION_PT * (0.62 if compact else 1.0),
            color=plotstyle.MUTED,
            va="center",
            zorder=6,
        )

    ax.set_yscale("log")
    label_pt = plotstyle.LABEL_PT * (0.68 if compact else 1.0)
    ax.set_xlabel(r"Geomean $\log_2$ Speedup", fontsize=label_pt)
    ax.set_ylabel("Median Tokens per Task", fontsize=label_pt)
    if compact:
        ax.tick_params(axis="both", labelsize=plotstyle.TICK_PT * 0.6)
    plotstyle.value_axis(ax, "x")
    plotstyle.value_axis(ax, "y", log_base=10.0)
    # Breathing room so an annotation at the right-hand point is not cut by the canvas edge.
    ax.margins(x=0.26, y=0.24)
    plotstyle.despine(ax)
    ax.margins(x=0.20, y=0.22)

    # All FOUR corners named, in the same grammar, so the quadrants compare at a glance. Note the
    # vertical sense is the OPPOSITE of the ratio figure's: there the y axis was tokens-saved, so
    # up was cheaper; here it is tokens spent, so up is more expensive.
    #
    # DROPPED in compact mode: this text is fixed-size (ANNOTATION_PT), sized for the full
    # PANEL_SIZE canvas, and a square panel a third that size cannot fit four captions without
    # them running into the axis labels and each other -- the quadrant reading survives without
    # them (up-right is the title's own "faster, cheaper" axes).
    if not compact:
        corners = (
            (0.015, 0.985, "top", "left", "Slower, More Expensive"),
            (0.985, 0.985, "top", "right", "Faster, More Expensive"),
            (0.015, 0.015, "bottom", "left", "Slower, Cheaper"),
            (0.985, 0.015, "bottom", "right", "Faster, Cheaper"),
        )
        for x, y, va, ha, caption in corners:
            ax.text(
                x,
                y,
                caption,
                transform=ax.transAxes,
                fontsize=plotstyle.ANNOTATION_PT - 2.0,
                color=plotstyle.FAINT,
                va=va,
                ha=ha,
                zorder=1,
            )

    handles = [
        plt.Line2D(
            [], [], marker=shapes[n], linestyle="none", color=h, markersize=9, label=experiment_tags.model_name(n)
        )
        for n, h in hues.items()
        if n in drawn_models
    ]
    handles += [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=plotstyle.MUTED,
            markersize=9,
            label=packets.control_label([treatment]),
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=plotstyle.MUTED,
            markersize=9,
            label=experiment_tags.packet_name(treatment),
        ),
        plt.Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            color=plotstyle.MUTED,
            markersize=11,
            # The correction and the family are ON the figure: a reader cannot tell a corrected
            # star from an uncorrected one by looking at it, and this is the only place the two
            # differ visibly.
            label=f"BH q < 0.05 of {family_size(stats)}",
        ),
    ]
    for which, width, style in (("major", 0.7, "-"), ("minor", 0.45, (0, (2, 3)))):
        ax.grid(axis="both", which=which, color=plotstyle.RULE, linewidth=width, linestyle=style, zorder=0)
    ax.set_axisbelow(True)
    return handles


def family_size(stats: pd.DataFrame) -> int:
    """How many tests the figure's marks were corrected over; 0 when the table carries none."""
    if stats.empty or "family_size" not in stats:
        return 0
    return int(stats.family_size.iloc[0])


PANEL_SIZE: tuple[float, float] = (8.4, 5.2)

#: Fixed margins, not ``tight_layout``. This figure is meant to be loaded beside the arm-summary
#: panels, and tight_layout sizes each figure from its own content -- one longer tick label and the
#: pair stops matching. Kept in step with ``plot_arm_summary.PANEL_MARGINS``.
PANEL_MARGINS: dict[str, float] = {"left": 0.175, "right": 0.975, "top": 0.855, "bottom": 0.40}


def write(fig, out: pathlib.Path) -> pathlib.Path:
    """Save at EXACTLY the figure size, with no bbox trimming.

    ``bbox_inches="tight"`` (the repo default) crops to the drawn content, so the saved size is a
    function of how wide the legend and labels happened to be. Two figures meant to be loaded side
    by side must not be sized by their contents, so this overrides it.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    # "standard", NOT None: matplotlib reads None as "fall back to rcParams", and this repo's
    # rcParams set savefig.bbox to "tight" -- so passing None left every figure cropped to its own
    # contents and a two-row legend silently made one figure twice the width of its pair.
    # The WHOLE canvas, explicitly. bbox_inches=None means "use the rcParam" and this repo sets
    # savefig.bbox to "tight"; "standard" is not a value matplotlib still accepts. Passing the
    # figure's own bbox is the only spelling that reliably means "do not crop to content", which
    # is what two figures of matching size require.
    plotstyle.save(fig, out.with_suffix(""), fixed=True)
    return out


def absolute_points(frame: pd.DataFrame, repeats: population.RepeatPolicy = "latest") -> pd.DataFrame:
    """One row per (model, language, condition): geomean log2 speed-up and median tokens per task.

    The same one-value-per-kernel reduction the paired table uses (:func:`population.kernel_medians`
    under ``repeats``), so the two panels of this figure describe one population. The speed-up is the
    GEOMEAN over kernels of the final answer (a ratio's overall value, never a median); the cost is
    the MEDIAN over kernels of the task total (tokens are not a ratio).
    """
    rows = []
    for (model, language, skills), part in frame.groupby(["model", "language", "skills"]):
        point = population.kernel_medians(part, repeats=repeats)
        if point is not None:
            rows.append({"model": model, "language": language, "skills": bool(skills), **point})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    rules.require_costs(table, "log2_speedup", ["baseline_ns", "native_ns"])
    rules.require_interval(table, "log2_speedup", "log2_speedup_low", "log2_speedup_high")
    return rules.require_interval(table, "tokens", "tokens_low", "tokens_high")


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


def untangle_labels(ax: plt.Axes) -> None:
    """Move each point label (an :class:`Annotation`) to the first of :data:`LABEL_PLACES` where its
    RENDERED text touches no mark and no label settled before it; a label with every place taken
    keeps the first. Call once the layout is final: marks move with the axes while a label's offset
    is in points, so a place clear before ``subplots_adjust`` need not be clear after it."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    taken: list[tuple[float, ...]] = []
    for collection in ax.collections:
        if isinstance(collection, PathCollection) and len(collection.get_offsets()):
            half = math.sqrt(float(np.max(collection.get_sizes()))) / 2.0 * fig.dpi / 72.0
            for px, py in collection.get_offset_transform().transform(collection.get_offsets()):
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


def figure_absolute(
    frame: pd.DataFrame, stats: pd.DataFrame, treatment: str, label: str, out: pathlib.Path
) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    handles = draw_absolute(ax, frame, stats, treatment)
    fig.subplots_adjust(**PANEL_MARGINS)
    # Two per row, and the keys kept SHORT. The canvas is fixed, so anything wider than it falls
    # off the edge rather than widening the figure -- and the model names alone ("Kimi-K2.7-Code")
    # are long enough that three columns no longer fit. The test behind the star is named in the
    # caption; what the reader needs AT the mark is that the threshold was corrected and over what.
    plotstyle.legend_below(fig, handles, ncol=2, y=0.015)
    plotstyle.title(fig, label)
    untangle_labels(ax)
    return write(fig, out)


#: A single treatment's SQUARE panel side, inches, when several are joined without ``--double-column``.
SQUARE_PANEL_SIDE: float = 3.6

#: Gap between joined square panels, inches.
SQUARE_PANEL_GAP: float = 0.25


def panel_side(n: int, double_column: bool) -> float:
    """One square panel's side for ``n`` panels joined in a row.

    ``--double-column`` caps the WHOLE row at :data:`~hpcagent_bench.stats.style.DOUBLE_COLUMN_WIDTH`
    inches, the figure's own budget on a paper page; otherwise every panel keeps its natural
    :data:`SQUARE_PANEL_SIDE` and the row grows with ``n``.
    """
    if not double_column:
        return SQUARE_PANEL_SIDE
    side = (plotstyle.DOUBLE_COLUMN_WIDTH - SQUARE_PANEL_GAP * (n - 1)) / n
    return max(1.6, side)


def build_treatments_figure(
    panels: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]], label: str, double_column: bool = False
) -> plt.Figure:
    """N SQUARE efficacy panels side by side, one per (treatment, its stats, its absolute points),
    every one against the SAME control -- see :func:`control_rows` and :func:`treatment_frame`.

    Square, so a reader compares treatments by panel shape as well as by content; joined rather
    than stacked, because the treatments are alternatives against one control, not a sequence.
    Split from :func:`figure_treatments` so a caller (a test, another figure) can inspect the
    figure -- its axes, its size -- before it is saved and closed.
    """
    n = len(panels)
    side = panel_side(n, double_column)
    # Every decorative font in draw_absolute is scaled for compactness at THIS panel size, not
    # the ANNOTATION_PT fixed size a full PANEL_SIZE panel uses -- see draw_absolute(compact=True).
    fig, axes = plt.subplots(1, n, figsize=(side * n + SQUARE_PANEL_GAP * (n - 1), side), squeeze=False)
    handles_by_label: dict[str, object] = {}
    for ax, (treatment, stats, absolute) in zip(axes[0], panels, strict=True):
        for handle in draw_absolute(ax, absolute, stats, treatment, compact=True):
            handles_by_label.setdefault(handle.get_label(), handle)
        # An IN-AXES label, not ax.set_title(): a real title draws ABOVE the axes bounding box, in
        # the same band subplots_adjust(top=...) reserves for the figure's own suptitle -- on a
        # short joined panel that band is thin enough that the two collide. Anchored inside the
        # axes (axes-fraction y=0.98) this cannot run into the suptitle no matter how short the
        # panel is.
        ax.text(
            0.5, 0.98, packets.label(treatment), transform=ax.transAxes,
            ha="center", va="top", fontsize=plotstyle.SUBTITLE_PT * 0.72, color=plotstyle.INK, zorder=7,
        )  # fmt: skip
    for ax in axes[0][1:]:
        ax.set_ylabel("")
    # title()'s return value IS the axes ceiling a figure this short needs: its own margin math is
    # in INCHES, not a guessed fraction, and a fraction picked for the tall PANEL_SIZE figure
    # (0.80-0.86) sits ABOVE a short figure's title text instead of below it.
    top = plotstyle.title(fig, label)
    plotstyle.legend_below(
        fig,
        list(handles_by_label.values()),
        ncol=min(len(handles_by_label), 3),
        y=0.005,
        fontsize=plotstyle.LABEL_PT * 0.55,
    )
    fig.subplots_adjust(left=0.14, right=0.99, top=top, bottom=0.46, wspace=0.45)
    for ax in axes[0]:
        untangle_labels(ax)
    return fig


def figure_treatments(
    panels: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]], label: str, out: pathlib.Path, double_column: bool = False
) -> pathlib.Path:
    return write(build_treatments_figure(panels, label, double_column), out)


def load(path: pathlib.Path, prefix: str) -> pd.DataFrame:
    frame = experiments.read_observations(path)
    if prefix:
        frame = frame[frame["arm"].astype(str).str.startswith(prefix)]
    # NO filter on speedup or tokens here. The two axes come off DIFFERENT record types -- the score
    # from the graded submissions, the cost from the call rows that carry a token count -- and one
    # predicate over both columns keeps only the rows that have both, which is the call rows alone.
    # That silently dropped every graded submission and scored the figure on intermediate rounds.
    #
    # ``packet`` is the row's RECORDED identity (see hpcagent_bench.harness.recording), canonicalized
    # through packets.canonical (aliases included); blank for a row written before that column
    # existed, which reads as the control -- the arm name is provenance, never parsed for this.
    # ``has_part`` catches a composite too (``lang-skills+no-score-tool`` still counts as skilled),
    # which a bare equality check against the canonical key would miss.
    if "packet" not in frame:
        frame = frame.assign(packet="")
    packet = frame["packet"].fillna("").astype(str).map(packets.canonical)
    frame = frame.assign(
        model=frame["arm"].astype(str).map(experiment_tags.model_of),
        packet=packet,
        skills=packet.map(lambda p: packets.has_part(p, "skills")),
    )
    return frame[frame.model != "other"]


def control_rows(frame_all: pd.DataFrame) -> pd.DataFrame:
    """The control side: the arm recording NO packet at all -- canonical packet ``""``.

    NEVER "every arm not carrying a KNOWN treatment": a campaign running several treatments beside
    each other (``cpfsrc``, ``lang-skills``, ``perf-playbook-cpu`` on cpf-llr-focus40) has arms
    whose one treatment is not in whatever list this hard-codes, and a hard-coded triple
    (``skills``, ``cpf``, ``cpfsrc``) let a fourth one -- ``perf-playbook-cpu`` -- read as part of
    the control, scoring the campaign's real control against a mixture instead of the no-packet
    arm. ``packet`` is already :func:`hpcagent_bench.packets.canonical`, which resolves "" for the
    control from the registry itself, so this needs no list of treatment names at all.
    """
    return frame_all[frame_all.packet == ""]


def treatment_frame(frame_all: pd.DataFrame, treatment: str) -> pd.DataFrame:
    """``frame_all``'s control and ``treatment`` rows, tagged ``skills`` True/False for
    :func:`absolute_points` and :func:`draw_absolute` -- which only need an on/off flag, not the
    packet's name.
    """
    control = control_rows(frame_all)
    treated = frame_all[frame_all.packet.map(lambda p: packets.has_part(p, treatment))]
    return pd.concat([control.assign(skills=False), treated.assign(skills=True)], ignore_index=True)


def complete_side_arms(
    control: pd.DataFrame, treated: pd.DataFrame, roster: Sequence[str], treatment: str, include_incomplete: bool
) -> set[str]:
    """The arms of ``control`` and ``treated`` that cover every kernel of ``roster`` -- the SAME
    gate :func:`hpcagent_bench.stats.figures.kernel_comparison` applies, so the two figures never
    disagree about which arms exist. An arm short of the roster is dropped and named on stderr with
    its coverage, never silently: it is the reason ``cpf-llr-focus40-qwen38-c-cpf`` (37/40) used to
    leave its treatment with no shared (model, language) at all, which crashed rather than skipped.
    """
    combined = pd.concat([control, treated], ignore_index=True)
    if include_incomplete:
        return set(combined["arm"].dropna().astype(str).unique())
    kept, dropped = population.complete_arms(combined, roster)
    for arm in sorted(dropped):
        print(f"{treatment}: dropping {arm} ({dropped[arm]}/{len(roster)} roster kernels)", file=sys.stderr)
    return set(kept)


def one_treatment_panel(
    frame_all: pd.DataFrame,
    control: pd.DataFrame,
    treatment: str,
    roster: Sequence[str],
    include_incomplete: bool = False,
    repeats: population.RepeatPolicy = "latest",
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """``(stats, absolute)`` for ONE treatment against ``control``; ``None`` when either side is
    empty (before or after the roster-completeness gate) or the two share no (model, language)."""
    # The two SIDES are the treatment, not two campaigns: an arm CARRYING the recorded packet
    # against one that does not -- never the arm name, which is provenance only. has_part matches a
    # composite too (an arm recording ``lang-skills+no-score-tool`` is still the skills side of
    # this split), which comparing the whole packet for equality would miss.
    treated = frame_all[frame_all.packet.map(lambda p: packets.has_part(p, treatment))]
    if control.empty or treated.empty:
        return None
    keep = complete_side_arms(control, treated, roster, treatment, include_incomplete)
    control = control[control["arm"].astype(str).isin(keep)]
    treated = treated[treated["arm"].astype(str).isin(keep)]
    if control.empty or treated.empty:
        return None
    stats = points(control, treated, repeats)
    if stats.empty:
        return None
    absolute_source = treatment_frame(frame_all, treatment)
    absolute = absolute_points(absolute_source[absolute_source["arm"].astype(str).isin(keep)], repeats)
    return stats, absolute


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=pathlib.Path)
    parser.add_argument("--experiment", required=True, help="arm prefix naming ONE campaign")
    parser.add_argument(
        "--treatment",
        action="append",
        default=[],
        help="packet naming a TREATED side (skills, cpf, cpfsrc, perf-playbook-cpu, ...); "
        "repeatable -- each is read against the SAME no-packet control (control_rows), one at a "
        "time rather than against each other. Default: skills. Two or more join as SQUARE panels "
        "side by side in one figure",
    )
    parser.add_argument(
        "--double-column",
        action="store_true",
        default=False,
        help="cap a joined (2+ treatment) figure's row width at style.DOUBLE_COLUMN_WIDTH",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        default=False,
        help="draw an arm even without a row for every roster kernel (default: dropped, named on stderr)",
    )
    parser.add_argument("--label", default="", help="figure title; defaults to the campaign's display name")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("figures/score_change.pdf"))
    parser.add_argument("--table", type=pathlib.Path, default=pathlib.Path("data/score_change.csv"))
    parser.add_argument(
        "--repeats",
        choices=population.REPEAT_POLICIES,
        default="latest",
        help="a kernel run more than once: latest run counts (reruns, default) or median over runs (designed repeats)",
    )
    args = parser.parse_args()

    treatments = args.treatment or ["skills"]
    frame_all = load(args.observations, args.experiment)
    control = control_rows(frame_all)
    if control.empty:
        raise SystemExit(f"no no-packet control rows for experiment {args.experiment!r}")
    # Every kernel ANY arm of this campaign touched -- the roster :func:`complete_side_arms` gates
    # coverage against, same population :mod:`scripts.plot_kernel_comparison` reads its own from.
    roster = sorted(frame_all["benchmark"].dropna().astype(str).unique())

    args.table.parent.mkdir(parents=True, exist_ok=True)
    panels: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for treatment in treatments:
        built = one_treatment_panel(frame_all, control, treatment, roster, args.include_incomplete, args.repeats)
        if built is None:
            print(f"skipping {treatment!r}: empty side, or no (model, language) shared with control")
            continue
        stats, absolute = built
        # Single treatment keeps the ORIGINAL file names (back-compatible); two or more are
        # suffixed by treatment so nothing overwrites its sibling.
        suffix = "" if len(treatments) == 1 else f"-{treatment}"
        stats.to_csv(args.table.with_name(f"{args.table.stem}{suffix}{args.table.suffix}"), index=False)
        absolute.to_csv(args.table.with_name(f"{args.table.stem}{suffix}-absolute{args.table.suffix}"), index=False)
        panels.append((treatment, stats, absolute))
    if not panels:
        raise SystemExit(f"no treatment of {treatments} produced a comparison for experiment {args.experiment!r}")

    label = args.label or experiment_tags.display_name(args.experiment)
    if len(panels) == 1:
        treatment, stats, absolute = panels[0]
        written = figure_absolute(absolute, stats, treatment, label, args.out)
    else:
        written = figure_treatments(panels, label, args.out, double_column=args.double_column)

    for treatment, stats, _ in panels:
        score_hits = int((stats.score_verdict == efficacy.SIGNIFICANT).sum())
        cost_hits = int((stats.cost_verdict == efficacy.SIGNIFICANT).sum())
        withheld = int((stats.score_verdict == efficacy.UNDERPOWERED).sum())
        print(
            f"{treatment}: {len(stats)} points; BH over {family_size(stats)} tests: "
            f"{score_hits} score-significant, {cost_hits} cost-significant, {withheld} underpowered"
        )
    print(f"table  -> {args.table}")
    print(f"figure -> {written} (+ .png)")


if __name__ == "__main__":
    main()
