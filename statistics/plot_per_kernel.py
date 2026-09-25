# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-kernel speedup and per-kernel tokens, compact enough for a paper column.

Reads one selection of observations (``--experiment`` for an arm prefix, ``--arm`` for a further
regex) and draws its per-kernel speedup and its per-kernel tokens
(:mod:`hpcagent_bench.stats.figures.per_kernel`). Every episode matching the selection is pooled
into one series per kernel -- this script draws ONE condition at a time; compare two conditions
(a model, a packet) by rendering it once per ``--arm`` selection.

Usage::

    python statistics/plot_per_kernel.py obs.csv --experiment cpf-llr-focus40-qwen38-c
    python statistics/plot_per_kernel.py obs.csv --experiment cpf-llr-focus40-qwen38-c --style box
    python statistics/plot_per_kernel.py obs.csv --experiment git-scicomp --style box --summary --layout stacked
"""

import argparse
import pathlib

import pandas as pd

from hpcagent_bench import experiments
from hpcagent_bench.stats import cost, palette
from hpcagent_bench.stats.figures import per_kernel


def load(path: pathlib.Path, prefix: str, arm: str, card: cost.CostModel = cost.resolve()) -> pd.DataFrame:
    frame = cost.priced(experiments.read_observations(path), card)
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
        help="append a summary column over the solved kernels: geomean speedup, median tokens",
    )
    parser.add_argument(
        "--layout",
        choices=("separate", "stacked"),
        default="separate",
        help="two files (default) or one figure with speedup over tokens, sharing the kernel axis",
    )
    parser.add_argument("--label", default="", help="draw this title above the figure; default: no title")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("figures/per_kernel.pdf"))
    parser.add_argument("--table", type=pathlib.Path, default=pathlib.Path("data/per_kernel.csv"))
    cost.add_arguments(parser)
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
    if args.layout == "stacked":
        if not (speed_cells and token_cells):
            raise SystemExit("--layout stacked needs both a speedup and a tokens cell to share the kernel axis")
        kernels = per_kernel.shared_kernel_order(speed_cells, token_cells)
        fig = per_kernel.figure_panels([speed, tokens], kernels, args.style, args.summary, label)
        written.append(per_kernel.save(fig, args.out))
        return written
    if speed_cells:
        kernels = per_kernel.ordered_kernels(speed_cells)
        fig = per_kernel.figure_one(speed, kernels, args.style, args.summary, f"{label}: Speedup" if label else "")
        written.append(per_kernel.save(fig, args.out.with_name(f"{stem}-speedup{suffix}")))
    if token_cells:
        kernels = per_kernel.ordered_kernels(token_cells)
        fig = per_kernel.figure_one(tokens, kernels, args.style, args.summary, f"{label}: Tokens" if label else "")
        written.append(per_kernel.save(fig, args.out.with_name(f"{stem}-tokens{suffix}")))
    return written


def main() -> None:
    args = build_parser().parse_args()
    frame = load(args.observations, args.experiment, args.arm, cost.resolve(args.cost_model, args.cost_models))
    speed_cells = per_kernel.speedup_cells(frame)
    token_cells = per_kernel.token_cells(frame)
    if not speed_cells and not token_cells:
        raise SystemExit(f"no per-kernel speedup or tokens for experiment={args.experiment!r} arm={args.arm!r}")

    # NO TITLE by default: a paper's caption is the title, and "all arms" over a panel naming one
    # arm's kernels was a caption that said nothing. --label draws one for a standalone render.
    label = args.label
    hues = palette.hues()
    speed = per_kernel.speedup_series_metric([per_kernel.Series("", tuple(speed_cells), hues[0])], "Speedup")
    tokens = per_kernel.token_series_metric([per_kernel.Series("", tuple(token_cells), hues[1])], "Tokens per Episode")

    write_tables(speed_cells, token_cells, args.table)
    written = render(speed, tokens, speed_cells, token_cells, args, label)

    print(f"{len(speed_cells)} speedup kernel(s), {len(token_cells)} token kernel(s)")
    for path in written:
        print(f"figure -> {path} (+ .png)")


if __name__ == "__main__":
    main()
