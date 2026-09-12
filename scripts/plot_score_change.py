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

Both are ratios, so both are aggregated in LOG space, by the Hodges-Lehmann estimator with a
distribution-free interval from the Wilcoxon signed-rank test. The sampling unit is the KERNEL and
the two sides are paired on it.

THE MARKS ARE CORRECTED. One figure is not one test: three models x two languages x two axes is
twelve signed-rank tests, and twelve uncorrected 5% thresholds paint at least one star on 46% of
figures where nothing happened. The family is therefore declared once (:func:`points` tests every
(model, language) on both axes), the p values are Benjamini-Hochberg adjusted across it, and the
star is gated on the ADJUSTED value -- an uncorrected star and a corrected star look identical to a
reader, so the legend names the correction and the size of the family. A point whose pairing is
too small for the test to run at all reads ``underpowered`` and is never starred.

"""

from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
import pandas as pd

from hpcagent_bench import experiment_tags
from hpcagent_bench.harness import efficacy
from hpcagent_bench.stats import palette, population, summary
from hpcagent_bench.stats import style as plotstyle

plotstyle.apply()
import matplotlib.pyplot as plt  # pyplot must follow plotstyle.apply()

#: A ratio this far from 1.0 is inside the "no change" band for labelling purposes only; the star
#: is decided by the interval, never by this.
NEUTRAL: float = 1.0

#: Order an episode's graded rows are read in. ``ts_ms`` ties when two land in the same
#: millisecond; ``attempt_index`` breaks it in the order the agent made them.
SUBMISSION_ORDER: tuple[str, str] = ("ts_ms", "attempt_index")

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
    """Hodges-Lehmann ratio over the kernels BOTH sides cover, with a distribution-free CI and p.

    RANK-BASED, not a bootstrap of the mean, and the reason is the shape of this data: a per-kernel
    speed-up ratio is heavy-tailed (one kernel at 40x against a median near 2x), and a mean in log
    space still lets that kernel carry the estimate. The estimator, its interval and its p value
    come from :func:`hpcagent_bench.stats.summary.paired_change`, so all three describe one
    quantity -- which a bootstrap mean beside a separate test does not.

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
    change = summary.paired_change(np.log(b / a) if invert else np.log(a / b))
    return (
        float(np.exp(change.estimate)),
        float(np.exp(change.low)),
        float(np.exp(change.high)),
        change.pvalue,
    )


def scores(frame: pd.DataFrame) -> pd.Series:
    """One speed-up per kernel for one side of the pair: the best FINAL answer.

    Read off the GRADED rows and reduced by the scoring policy
    (:func:`hpcagent_bench.stats.population.final_answers`): within an episode the last verified
    submission counts, and the maximum is kept across the episodes of that side. A ``call`` row
    carries a speed-up for a round the judge never persisted, and a median over those rows -- which
    this figure used to take -- weights a kernel by how many rounds the agent spent on it.
    """
    graded = frame[frame.record == "submission"]
    if graded.empty:
        return pd.Series(dtype=float)
    best = population.final_answers(graded, SUBMISSION_ORDER, ("arm", "benchmark"))
    return best.groupby("benchmark").speedup.max()


def costs(frame: pd.DataFrame) -> pd.Series:
    """One token spend per kernel for one side of the pair.

    ``calls.tokens`` is CUMULATIVE through a call, so an episode's spend is its own maximum and a
    kernel's is the SUM over its episodes. Summing the rows instead counts every earlier call once
    per later one, and a median over them is a median of running totals.
    """
    calls = frame[frame.record == "call"].copy()
    if calls.empty:
        return pd.Series(dtype=float)
    calls["tokens"] = pd.to_numeric(calls.tokens, errors="coerce")
    calls = calls.dropna(subset=["tokens", "benchmark"])
    if calls.empty:
        return pd.Series(dtype=float)
    per_episode = population.per_episode_max(calls, "tokens")
    totals = per_episode.groupby("benchmark").tokens.sum()
    return totals[totals > 0]


def points(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
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
        before_score, after_score = scores(b), scores(a)
        score, s_low, s_high, s_p = ratio_with_ci(before_score, after_score, False)
        cost, c_low, c_high, c_p = ratio_with_ci(costs(b), costs(a), True)
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
    frame = pd.DataFrame(rows).dropna(subset=["score", "cost"])
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


def draw_absolute(ax, frame: pd.DataFrame, stats: pd.DataFrame) -> list:
    """Median speed-up against median spend, with each arm's TWO CONDITIONS joined.

    ABSOLUTE, not a ratio, and that is what makes the connector mean something: a ratio plot has
    one point per arm and nothing to join, so a line drawn on it could only connect two arms --
    which is what it did, joining an arm's C point to its Fortran point and inviting the reading
    that a language change is the treatment. Here each (model, language) has a hollow mark without
    the packet and a filled one with it, and the segment between them IS the treatment effect for
    that arm: its length is the size, its direction the sign, on both axes at once.

    The filled mark is drawn LAST so it sits above the connector and above its own hollow partner
    -- the two land on top of each other whenever the packet changed little, and the "with skills"
    position is the one a reader is looking for.
    """
    hues = palette.model_colors(sorted(frame.model.unique()))
    shapes = palette.model_markers(sorted(frame.model.unique()))
    # Gated on the ADJUSTED verdict, on either axis. An arm the packet moved on tokens alone is a
    # real finding, so the mark fires on either -- which is exactly why both axes are one family.
    significant = {
        (row.model, row.language): efficacy.SIGNIFICANT in (row.score_verdict, row.cost_verdict)
        for row in stats.itertuples(index=False)
    }
    for (model, language), pair in frame.groupby(["model", "language"]):
        colour, shape = hues[model], shapes[model]
        off, on = pair[~pair.skills], pair[pair.skills]
        if len(off) != 1 or len(on) != 1:
            continue
        # An ELBOW, not a diagonal. The straight segment between two measured points runs through
        # coordinates that were never measured, and on a plot whose whole subject is where an arm
        # LANDED a reader takes the path for data -- as if the packet moved the arm along it. The
        # right angle is visibly a connector: it says these two marks are one arm and claims
        # nothing about the space between them. Horizontal first, so the corner sits under the
        # "with skills" mark and the vertical leg reads as the change in spend.
        x_off, x_on = float(off.log2_speedup.iloc[0]), float(on.log2_speedup.iloc[0])
        y_off, y_on = float(off.tokens.iloc[0]), float(on.tokens.iloc[0])
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
            fontsize=plotstyle.ANNOTATION_PT,
            color=plotstyle.MUTED,
            va="center",
            zorder=6,
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"Median $\log_2$ Speedup")
    ax.set_ylabel("Median Tokens per Task")
    plotstyle.value_axis(ax, "x")
    plotstyle.value_axis(ax, "y", log_base=10.0)
    # Breathing room so an annotation at the right-hand point is not cut by the canvas edge.
    ax.margins(x=0.26, y=0.24)
    plotstyle.despine(ax)
    ax.margins(x=0.20, y=0.22)

    # All FOUR corners named, in the same grammar, so the quadrants compare at a glance. Note the
    # vertical sense is the OPPOSITE of the ratio figure's: there the y axis was tokens-saved, so
    # up was cheaper; here it is tokens spent, so up is more expensive.
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
            label="No Skills",
        ),
        plt.Line2D([], [], marker="o", linestyle="none", color=plotstyle.MUTED, markersize=9, label="Skills"),
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
    fig.savefig(out, bbox_inches=fig.bbox_inches)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches=fig.bbox_inches)
    plt.close(fig)
    return out


def absolute_points(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, language, condition): median log2 speed-up and median tokens per task.

    Medians over KERNELS, over the same one-value-per-kernel reduction the paired table uses, so
    the two panels of this figure describe one population. The speed-up is the best final answer
    and the cost is the kernel's episode total.
    """
    rows = []
    for (model, language, skills), part in frame.groupby(["model", "language", "skills"]):
        speed = scores(part)
        speed = speed[speed > 0]
        tokens = costs(part)
        tokens = tokens[tokens > 0]
        if speed.empty or tokens.empty:
            continue
        rows.append(
            {
                "model": model,
                "language": language,
                "skills": bool(skills),
                "log2_speedup": float(np.median(np.log2(speed.to_numpy(dtype=float)))),
                "tokens": float(np.median(tokens.to_numpy(dtype=float))),
                "kernels": int(speed.size),
            }
        )
    return pd.DataFrame(rows)


def figure_absolute(frame: pd.DataFrame, stats: pd.DataFrame, label: str, out: pathlib.Path) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    handles = draw_absolute(ax, frame, stats)
    fig.subplots_adjust(**PANEL_MARGINS)
    # Two per row, and the keys kept SHORT. The canvas is fixed, so anything wider than it falls
    # off the edge rather than widening the figure -- and the model names alone ("Kimi-K2.7-Code")
    # are long enough that three columns no longer fit. The test behind the star is named in the
    # caption; what the reader needs AT the mark is that the threshold was corrected and over what.
    plotstyle.legend_below(fig, handles, ncol=2, y=0.015)
    plotstyle.title(fig, label)
    return write(fig, out)


def load(path: pathlib.Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if prefix:
        frame = frame[frame["arm"].astype(str).str.startswith(prefix)]
    # NO filter on speedup or tokens here. The two axes come off DIFFERENT record types -- the score
    # from the graded submissions, the cost from the call rows that carry a token count -- and one
    # predicate over both columns keeps only the rows that have both, which is the call rows alone.
    # That silently dropped every graded submission and scored the figure on intermediate rounds.
    frame = frame.assign(
        model=frame["arm"].astype(str).map(experiment_tags.model_of),
        skills=frame["arm"].astype(str).str.endswith("-skills"),
    )
    return frame[frame.model != "other"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=pathlib.Path)
    parser.add_argument("--experiment", required=True, help="arm prefix naming ONE campaign")
    parser.add_argument(
        "--treatment",
        default="skills",
        help="arm suffix marking the TREATED side (skills, cpf, cpfsrc); the control is the arm "
        "without any suffix, so a campaign carrying several treatments is read one at a time "
        "against the same control rather than against each other",
    )
    parser.add_argument("--label", default="", help="figure title; defaults to the campaign's display name")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("figures/score_change.pdf"))
    parser.add_argument("--table", type=pathlib.Path, default=pathlib.Path("data/score_change.csv"))
    args = parser.parse_args()

    frame_all = load(args.observations, args.experiment)
    # The two SIDES are the treatment, not two campaigns: an arm carrying the packet against the
    # arm that did not. Split on the suffix the submit scripts already use.
    suffix = f"-{args.treatment}"
    # The CONTROL is the arm with no suffix at all, never "everything that is not the treatment":
    # a campaign carrying skills, cpf and cpfsrc would otherwise put two other treatments into the
    # control side and report a contrast against a mixture.
    names = frame_all["arm"].astype(str)
    treated = frame_all[names.str.endswith(suffix)]
    control = frame_all[~names.str.contains(r"-(?:skills|cpf|cpfsrc)$", regex=True)]
    before, after = control, treated
    if before.empty or after.empty:
        raise SystemExit(f"empty side: control={len(before)} {args.treatment}={len(after)}")
    frame = points(before, after)
    if frame.empty:
        raise SystemExit("no (model, language) appears in both experiments")
    args.table.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.table, index=False)
    label = args.label or experiment_tags.display_name(args.experiment)
    absolute = absolute_points(frame_all)
    absolute.to_csv(args.table.with_name(args.table.stem + "-absolute" + args.table.suffix), index=False)
    written = figure_absolute(absolute, frame, label, args.out)
    score_hits = int((frame.score_verdict == efficacy.SIGNIFICANT).sum())
    cost_hits = int((frame.cost_verdict == efficacy.SIGNIFICANT).sum())
    withheld = int((frame.score_verdict == efficacy.UNDERPOWERED).sum())
    print(
        f"{len(frame)} points; BH over {family_size(frame)} tests: {score_hits} score-significant, "
        f"{cost_hits} cost-significant, {withheld} score pairings too small to test"
    )
    print(f"table  -> {args.table}")
    print(f"figure -> {written} (+ .png)")


if __name__ == "__main__":
    main()
