# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tokens per KERNEL, per model, for one experiment.

Tokens are the other half of a result. A model that reaches the same speed-up for a third of the
spend is a different proposition from one that does not, and a speed-up chart alone cannot say so.

THE SAME QUANTITY every other figure here costs a kernel at: what that kernel cost this model, the
sum over the episodes that ran it (:func:`hpcagent_bench.stats.population.kernel_tokens`), reruns
reduced to the latest. It used to plot a MEDIAN over the episodes inside each cell instead, which is
a different number with a different unit -- one cell held a handful of episodes from several arms --
so a kernel's cost read one way here and another in `plot_arm_summary.py`, `plot_score_change.py`
and `paired_arms.py`, which all reduce a kernel to one value and take the median over KERNELS.

Drawn on a log y axis because spend spans decades -- a linear axis puts every ordinary kernel in
the first tenth of the panel and gives the runaway the other nine.
"""

import argparse
import pathlib

import numpy as np
import pandas as pd

from hpcagent_bench import experiment_tags
from hpcagent_bench.stats import palette, population, summary
from hpcagent_bench.stats import style as plotstyle

plotstyle.apply()
import matplotlib.pyplot as plt

#: Seed kept for the episode ordering in the CSV, so the published table is reproducible.
SEED: int = 0


def cells(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, kernel): what that kernel cost, and the episodes the cost is made of.

    ``kernel_tokens`` is the published number and comes from the one shared definition. ``episodes``
    is the same spend broken out per task, kept so the figure can draw what was measured and the CSV
    can be re-analysed without going back to the observations file; it sums to ``kernel_tokens``.
    """
    spend = population.kernel_tokens(frame, ("model", "benchmark"))
    episodes = population.episode_tokens(population.latest_runs(frame), by=("model", "benchmark"))
    per_cell = {key: group["tokens"].to_numpy(dtype=float) for key, group in episodes.groupby(["model", "benchmark"])}
    rows = []
    for (model, kernel), total in spend.items():
        tokens = per_cell.get((model, kernel), np.zeros(0))
        rows.append(
            {
                "model": model,
                "benchmark": kernel,
                "n": int(tokens.size),
                "kernel_tokens": float(total),
                "episodes": sorted(float(v) for v in tokens),
            }
        )
    return pd.DataFrame(rows)


def draw(cell_frame: pd.DataFrame, experiment: str, out: pathlib.Path, unit: str = "tokens") -> pathlib.Path:
    """Kernels down the y axis so their names read horizontally; one coloured mark per model."""
    order = summary.median_per_kernel(cell_frame, "kernel_tokens").sort_values().index.tolist()
    models = [m for m in palette.order("models") if m in set(cell_frame["model"])]
    hues = palette.model_colors(models)
    positions = {kernel: i for i, kernel in enumerate(order)}
    offsets = np.linspace(-0.26, 0.26, len(models)) if len(models) > 1 else [0.0]

    # Kernels along X, the measured quantity up Y. The measured axis is the one a reader compares
    # across series, and comparing along a shared vertical is what every other chart in the report
    # asks of them; a horizontal value axis made this the one figure read sideways.
    fig, ax = plt.subplots(figsize=(max(7.4, 0.26 * len(order) + 2.0), 6.0))
    shapes = palette.model_markers(models)
    for model, offset in zip(models, offsets, strict=True):
        part = cell_frame[cell_frame["model"] == model]
        if part.empty:
            continue
        x = np.array([positions[k] for k in part["benchmark"]], dtype=float) + offset
        # One mark per cell and nothing else. The cell holds ~4 episodes from ~4 different arms --
        # on v11, every one of its 120 cells mixed the skills and no-skills arms -- so neither a
        # bootstrap interval nor a scatter of the episodes says anything about uncertainty: the
        # spread is mostly the treatment, and the treatment has its own figure. Drawing it here put
        # four indistinguishable faint marks per cell on the chart and invited the reader to read
        # them as noise.
        ax.scatter(
            x,
            part["kernel_tokens"].to_numpy(),
            s=52,
            color=hues[model],
            marker=shapes[model],
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
            label=experiment_tags.model_name(model),
        )
    ax.set_yscale("log")
    # Short: a long y label on a wide short figure runs up into the title.
    ax.set_ylabel(f"{unit.title()} per Kernel")
    ax.set_xlabel("")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=plotstyle.ANNOTATION_PT, rotation=90)
    ax.set_xlim(-0.8, len(order) - 0.2)
    plotstyle.value_axis(ax, "y", log_base=10.0)
    plotstyle.despine(ax)
    # No "one episode" key: the episodes are not drawn (see above), and a legend entry for a mark
    # that is not on the chart is a legend that describes a different figure.
    plotstyle.legend_below(fig, ax.get_legend_handles_labels()[0])
    top = plotstyle.title(fig, experiment)
    fig.tight_layout(rect=(0, 0, 1, top))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=pathlib.Path, help="llr40_observations.csv or the same schema")
    parser.add_argument("--experiment", default="", help="arm prefix selecting one experiment; blank takes all")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("figures/tokens_per_kernel.pdf"))
    parser.add_argument("--table", type=pathlib.Path, default=pathlib.Path("data/tokens_per_kernel.csv"))
    args = parser.parse_args()

    frame = pd.read_csv(args.observations, low_memory=False)
    if args.experiment:
        frame = frame[frame["arm"].astype(str).str.startswith(args.experiment)]
    frame = frame.assign(model=frame["arm"].astype(str).map(experiment_tags.model_of))
    frame = frame[frame["model"] != "other"]
    cell_frame = cells(frame)
    if cell_frame.empty:
        raise SystemExit(f"no task token rows for experiment {args.experiment!r}")
    args.table.parent.mkdir(parents=True, exist_ok=True)
    cell_frame.to_csv(args.table, index=False)
    written = draw(cell_frame, experiment_tags.display_name(args.experiment) or "all arms", args.out)
    print(f"{int(cell_frame['n'].sum())} episodes -> {len(cell_frame)} cells")
    print(f"table  -> {args.table}")
    print(f"figure -> {written} (+ .png)")


if __name__ == "__main__":
    main()
