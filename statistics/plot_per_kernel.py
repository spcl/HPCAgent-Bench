# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-kernel speed-up and per-kernel tokens, compact enough for a paper column.

Reads one selection of observations (``--experiment`` for an arm prefix, ``--arm`` for a further
regex) and draws its per-kernel speed-up and its per-kernel tokens
(:mod:`hpcagent_bench.stats.figures.per_kernel`). By default every episode matching the selection is
pooled into one series per kernel; ``--series arm`` draws one series per arm instead (colour = model,
shape = packet, a control hollow), each with its own geomean slot in the summary column.

Usage::

    python statistics/plot_per_kernel.py obs.csv --experiment cpf-llr-focus40-qwen38-c
    python statistics/plot_per_kernel.py obs.csv --experiment cpf-llr-focus40-qwen38-c --style box
    python statistics/plot_per_kernel.py obs.csv --experiment git-scicomp --style box --summary --layout stacked
    python statistics/plot_per_kernel.py obs.csv --arm 'llr-focus40-.*-c-cpf' --series arm --summary \
        --layout stacked --width 5.5
"""

import argparse
import pathlib

import pandas as pd

from hpcagent_bench import experiments
from hpcagent_bench.stats import palette
from hpcagent_bench.stats.figures import per_kernel


def load(path: pathlib.Path, prefix: str, arm: str) -> pd.DataFrame:
    frame = experiments.read_observations(path)
    if prefix:
        frame = frame[frame["arm"].astype(str).str.startswith(prefix)]
    if arm:
        frame = frame[frame["arm"].astype(str).str.fullmatch(arm)]
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=pathlib.Path, help="observations CSV or extracted .db")
    parser.add_argument("--experiment", default="", help="arm prefix selecting one experiment; blank keeps all")
    parser.add_argument("--arm", default="", help="regex; keep only arms whose full name matches")
    parser.add_argument("--style", choices=("ci", "box"), default="ci", help="median+CI (default) or a boxplot")
    parser.add_argument(
        "--summary",
        action="store_true",
        default=False,
        help="append a summary column over the solved kernels: geomean speed-up, median tokens",
    )
    parser.add_argument(
        "--layout",
        choices=("separate", "stacked"),
        default="separate",
        help="two files (default) or one figure with speed-up over tokens, sharing the kernel axis",
    )
    parser.add_argument(
        "--series",
        choices=("pooled", "arm"),
        default="pooled",
        help="one series pooling every selected arm (default), or one series per arm",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=0.0,
        help="draw at this width in inches at print type size (e.g. 5.5 for ICLR); default: authoring size",
    )
    parser.add_argument("--label", default="", help="draw this title above the figure; default: no title")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("figures/per_kernel.pdf"))
    parser.add_argument("--table", type=pathlib.Path, default=pathlib.Path("data/per_kernel.csv"))
    return parser


def write_tables(
    speed_cells: list[per_kernel.KernelCell], token_cells: list[per_kernel.KernelCell], table: pathlib.Path
) -> None:
    table.parent.mkdir(parents=True, exist_ok=True)
    if speed_cells:
        per_kernel.cells_table(speed_cells, "speedup").to_csv(
            table.with_name(f"{table.stem}-speedup{table.suffix}"), index=False
        )
    if token_cells:
        per_kernel.cells_table(token_cells, "tokens").to_csv(
            table.with_name(f"{table.stem}-tokens{table.suffix}"), index=False
        )


def render(
    speed: per_kernel.Metric,
    tokens: per_kernel.Metric,
    speed_cells: list[per_kernel.KernelCell],
    token_cells: list[per_kernel.KernelCell],
    args: argparse.Namespace,
    label: str,
) -> list[pathlib.Path]:
    stem, suffix = args.out.stem, args.out.suffix
    written: list[pathlib.Path] = []
    width = args.width or None
    # A pooled series is one unlabelled colour: no key. Per-arm series need theirs.
    legend = []
    if args.series == "arm":
        legend = [*per_kernel.series_handles(speed.series), *per_kernel.status_handles([speed, tokens])]
    if args.layout == "stacked":
        if not (speed_cells and token_cells):
            raise SystemExit("--layout stacked needs both a speed-up and a tokens cell to share the kernel axis")
        kernels = per_kernel.shared_kernel_order(speed_cells, token_cells)
        fig = per_kernel.figure_panels([speed, tokens], kernels, args.style, args.summary, label, width, legend)
        written.append(per_kernel.save(fig, args.out))
        return written
    if speed_cells:
        kernels = per_kernel.ordered_kernels(speed_cells)
        title = f"{label}: Speed-Up" if label else ""
        fig = per_kernel.figure_one(speed, kernels, args.style, args.summary, title, width, legend)
        written.append(per_kernel.save(fig, args.out.with_name(f"{stem}-speedup{suffix}")))
    if token_cells:
        kernels = per_kernel.ordered_kernels(token_cells)
        title = f"{label}: Tokens" if label else ""
        fig = per_kernel.figure_one(tokens, kernels, args.style, args.summary, title, width, legend)
        written.append(per_kernel.save(fig, args.out.with_name(f"{stem}-tokens{suffix}")))
    return written


def main() -> None:
    args = build_parser().parse_args()
    frame = load(args.observations, args.experiment, args.arm)
    speed_cells = per_kernel.speedup_cells(frame)
    token_cells = per_kernel.token_cells(frame)
    if not speed_cells and not token_cells:
        raise SystemExit(f"no per-kernel speed-up or tokens for experiment={args.experiment!r} arm={args.arm!r}")

    # NO TITLE by default: a paper's caption is the title, and "all arms" over a panel naming one
    # arm's kernels was a caption that said nothing. --label draws one for a standalone render.
    label = args.label
    if args.series == "arm":
        speed_series = per_kernel.arm_series(frame, "speedup")
        token_series = per_kernel.arm_series(frame, "tokens")
    else:
        hues = palette.hues()
        speed_series = [per_kernel.Series("", tuple(speed_cells), hues[0])]
        token_series = [per_kernel.Series("", tuple(token_cells), hues[1])]
    speed = per_kernel.speedup_series_metric(speed_series, "Speed-Up")
    tokens = per_kernel.token_series_metric(token_series, "Tokens per Episode")

    write_tables(speed_cells, token_cells, args.table)
    written = render(speed, tokens, speed_cells, token_cells, args, label)

    print(f"{len(speed_cells)} speed-up kernel(s), {len(token_cells)} token kernel(s)")
    for path in written:
        print(f"figure -> {path} (+ .png)")


if __name__ == "__main__":
    main()
