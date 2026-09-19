# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-kernel speed-up and per-kernel tokens, compact enough for a paper column.

Reads one selection of observations (``--experiment`` for an arm prefix, ``--arm`` for a further
regex) and draws its per-kernel speed-up and its per-kernel tokens
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

from hpcagent_bench import experiment_tags, experiments
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
    parser.add_argument("--summary", action="store_true", default=False, help="append a median-over-kernels column")
    parser.add_argument(
        "--layout",
        choices=("separate", "stacked"),
        default="separate",
        help="two files (default) or one figure with speed-up over tokens, sharing the kernel axis",
    )
    parser.add_argument("--label", default="", help="figure title; defaults to the experiment's display name")
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
    if args.layout == "stacked":
        if not (speed_cells and token_cells):
            raise SystemExit("--layout stacked needs both a speed-up and a tokens cell to share the kernel axis")
        kernels = per_kernel.shared_kernel_order(speed_cells, token_cells)
        fig = per_kernel.figure_stacked(speed, tokens, kernels, args.style, args.summary, label)
        written.append(per_kernel.save(fig, args.out))
        return written
    if speed_cells:
        kernels = per_kernel.ordered_kernels(speed_cells)
        fig = per_kernel.figure_one(speed, kernels, args.style, args.summary, f"{label}: Speed-Up")
        written.append(per_kernel.save(fig, args.out.with_name(f"{stem}-speedup{suffix}")))
    if token_cells:
        kernels = per_kernel.ordered_kernels(token_cells)
        fig = per_kernel.figure_one(tokens, kernels, args.style, args.summary, f"{label}: Tokens")
        written.append(per_kernel.save(fig, args.out.with_name(f"{stem}-tokens{suffix}")))
    return written


def main() -> None:
    args = build_parser().parse_args()
    frame = load(args.observations, args.experiment, args.arm)
    speed_cells = per_kernel.speedup_cells(frame)
    token_cells = per_kernel.token_cells(frame)
    if not speed_cells and not token_cells:
        raise SystemExit(f"no per-kernel speed-up or tokens for experiment={args.experiment!r} arm={args.arm!r}")

    label = args.label or experiment_tags.display_name(args.experiment) or "all arms"
    hues = palette.hues()
    speed = per_kernel.speedup_metric(speed_cells, "Speed-Up", hues[0])
    tokens = per_kernel.token_metric(token_cells, "Tokens per Episode", hues[1])

    write_tables(speed_cells, token_cells, args.table)
    written = render(speed, tokens, speed_cells, token_cells, args, label)

    print(f"{len(speed_cells)} speed-up kernel(s), {len(token_cells)} token kernel(s)")
    for path in written:
        print(f"figure -> {path} (+ .png)")


if __name__ == "__main__":
    main()
