"""What canonicalization buys, measured PAIRWISE against dace main and against llvm + polly.

The companion figure (``plot_tsvc_speedup.py``) puts three arms on a common serial-gcc denominator,
which answers "how fast is each arm". This one answers the different question the ablation is about:
on a kernel BOTH tools compiled, is canon the faster of the two? So the denominator here is the other
TOOL, per kernel, and the two rows are two paired comparisons rather than two independent arms.

WHY PAIRED. Dividing two geomeans taken over different kernel sets is not a speed-up of anything: an
arm that fails on the kernels it is bad at comes out ahead by attrition. Each row here is restricted
to the intersection -- kernels canon AND the comparison arm both timed and validated -- so every
point is one kernel measured twice, and the row's n is reported because it is not the corpus size.

The signed axis, the exclusion rules and the log-space geomean interval are the companion's, imported
rather than restated, so the two figures cannot drift apart.

Usage:  python3 plot_canon_vs.py <sweep-directory>
"""

from __future__ import annotations

import argparse
import pathlib
import random
import statistics
import sys

import matplotlib
import matplotlib.lines

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  -- must follow the Agg backend selection

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # noqa: E402

from benchlib import style as theme  # noqa: E402  -- ditto
import plot_tsvc_speedup  # noqa: E402  -- ditto, it selects the backend itself

#: The NUMERATOR of every ratio on this figure: the arm the ablation is about.
REFERENCE = "dace_cpu_canonicalize"

#: Denominator -> the row label. Insertion order is the order on the axis.
COMPARISONS = {"dace_cpu": "vs dace main", "cc_llvm_autopar": "vs llvm + polly"}

#: Each row keeps the hue its arm carries on the companion figure, so a reader moving between the
#: two reads "llvm + polly" as the same colour in both.
ROW_COLOR = {
    "dace_cpu": plot_tsvc_speedup.ARM_COLOR["dace_cpu"],
    "cc_llvm_autopar": plot_tsvc_speedup.ARM_COLOR["cc_llvm_autopar"],
}


def paired(reference: dict[str, float], other: dict[str, float]) -> dict[str, float]:
    """``kernel -> other_ms / canon_ms`` over the kernels BOTH timed; above 1 means canon is faster."""
    return {k: other[k] / ms for k, ms in sorted(reference.items()) if k in other}


def sign_test(ratios: dict[str, float]) -> tuple[int, int]:
    """Kernels canon wins and loses, at a 1% dead band -- below that the two are the same code."""
    wins = sum(1 for v in ratios.values() if v > 1.01)
    losses = sum(1 for v in ratios.values() if v < 1.0 / 1.01)
    return wins, losses


def render(root: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    """Draw the paired figure for one sweep directory; returns the stem both files were written to."""
    canon = plot_tsvc_speedup.read_arm(root, REFERENCE)
    if not canon.times:
        raise SystemExit(
            f"no {REFERENCE} arm in {root}: looked for {REFERENCE}.csv and {REFERENCE}.rank*.csv. "
            f"It is the numerator of every ratio here, so there is nothing to plot without it."
        )

    tall = 1.35 + 0.62 * len(COMPARISONS)
    fig, ax = plt.subplots(figsize=(6.8, tall))
    fig.subplots_adjust(left=0.20, right=0.885, top=1.0 - 0.86 / tall, bottom=0.86 / tall)
    jitter = random.Random(0)
    notes: list[str] = []
    theme.axis_title(ax, "TSVC kernels, canonicalized dace against each tool it is paired with", pad=18.0)
    ax.set_xlabel(
        "signed relative speed-up of dace canon\n$+1$ = 2$\\times$ faster, 0 = no change, $-1$ = 2$\\times$ slower"
    )
    theme.row_axis(ax, list(COMPARISONS.values()))
    for index, (framework, name) in enumerate(COMPARISONS.items()):
        arm = plot_tsvc_speedup.read_arm(root, framework)
        ratios = paired(canon.times, arm.times)
        colour = ROW_COLOR[framework]
        tally = ", ".join(f"{n} {why}" for why, n in sorted(arm.rejected.items()))
        if not ratios:
            theme.right_label(ax, index, "no data", theme.MUTED)
            notes.append(f"{name}: no kernel timed by both" + (f" ({tally})" if tally else ""))
            continue
        ax.scatter(
            [plot_tsvc_speedup.signed(v) for v in ratios.values()],
            [index - 0.22 + jitter.uniform(-0.10, 0.10) for _ in ratios],
            s=11,
            color=colour,
            alpha=0.45,
            linewidth=0,
            zorder=2,
        )
        centre, low, high = plot_tsvc_speedup.geomean_interval(list(ratios.values()))
        ax.errorbar(
            plot_tsvc_speedup.signed(centre),
            index + 0.24,
            xerr=[
                [plot_tsvc_speedup.signed(centre) - plot_tsvc_speedup.signed(low)],
                [plot_tsvc_speedup.signed(high) - plot_tsvc_speedup.signed(centre)],
            ],
            fmt="o",
            markersize=5.5,
            color=colour,
            ecolor=theme.INK2,
            elinewidth=1.1,
            capsize=3.5,
            zorder=4,
        )
        middle = statistics.median(list(ratios.values()))
        ax.plot(
            [plot_tsvc_speedup.signed(middle)] * 2,
            [index + 0.10, index + 0.38],
            color=theme.INK,
            linewidth=1.0,
            zorder=3,
        )
        theme.right_label(ax, index, f"n={len(ratios)}")
        wins, losses = sign_test(ratios)
        unpaired = len(canon.times) - len(ratios)
        notes.append(
            f"{name}: geomean {centre:.2f}x, median {middle:.2f}x, "
            f"canon faster on {wins} of {len(ratios)}, slower on {losses}, "
            f"{unpaired} canon kernels unpaired" + (f"; comparison arm lost {tally}" if tally else "")
        )

    ax.axvline(0.0, color=theme.INK2, linewidth=1.0, zorder=1)
    ax.margins(x=0.08)
    handles = [
        matplotlib.lines.Line2D([], [], marker="o", linestyle="none", color=theme.MUTED, markersize=3.5, alpha=0.5),
        matplotlib.lines.Line2D([], [], marker="o", linestyle="none", color=theme.MUTED, markersize=5.5),
        matplotlib.lines.Line2D([], [], color=theme.INK, linewidth=1.0),
    ]
    theme.key(
        ax,
        list(
            zip(
                ["one kernel", f"geomean, {plot_tsvc_speedup.CONFIDENCE:.0%} t-interval", "median"],
                handles,
                strict=True,
            )
        ),
        anchor=(1.0, 1.005),
    )
    theme.save(fig, out)
    for note in notes:
        print(f"  {note}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=pathlib.Path, help="directory of <framework>[.rank<N>].csv files")
    args = parser.parse_args()
    plot_tsvc_speedup.style()
    figures = pathlib.Path(__file__).resolve().parent / "figures"
    render(args.sweep, figures / f"tsvc_canon_paired_{args.sweep.resolve().name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
