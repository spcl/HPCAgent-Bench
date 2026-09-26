# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Weak- and strong-scaling figures for the distributed ML-op track.

Reads one selection of observations (``--experiment`` for an arm prefix, ``--arm`` for a further
regex) and draws its scaling curves (:mod:`hpcagent_bench.stats.figures.scaling`): parallel
efficiency eta(P), speedup sigma(P), the per-kernel small multiples, and the per-arm geomean eta
with its interval. Every figure writes a PDF, a PNG and the CSV behind the marks, plus a second CSV
naming every point the sweep did not measure and every curve too short to draw.

Usage::

    python statistics/plot_scaling.py obs.csv --experiment mlscale
    python statistics/plot_scaling.py obs.csv --experiment mlscale --figure efficiency
    python statistics/plot_scaling.py obs.csv --experiment mlscale --figure per-kernel --mode weak
    python statistics/plot_scaling.py obs.csv --arm 'mlscale-(weak|strong)-qwen38-hip'

The torch.distributed baseline curve (arm ``torch_dist``, read off the grade job's
``baseline_points`` rows) is drawn beside the selected arms; ``--no-torch-dist`` leaves it out.
"""

import argparse
import pathlib
import sys

import pandas as pd

from hpcagent_bench import experiments
from hpcagent_bench.stats import style as plotstyle
from hpcagent_bench.stats.figures import scaling

#: ``--figure`` choices. ``all`` draws every one of them in a single pass over the frame.
FIGURES: tuple[str, ...] = (
    "all",
    "efficiency",
    "speedup",
    "per-kernel",
    "summary",
    "mode-grid",
)


def load(path: pathlib.Path, prefix: str, arm: str, torch_dist: bool = True) -> pd.DataFrame:
    """The observations frame, narrowed to one experiment prefix and one arm regex. The
    torch.distributed baseline curve's rows (arm ``torch_dist``: no experiment's arm) are kept
    beside the selection unless ``torch_dist`` is False."""
    frame = experiments.read_observations(path)
    names = frame["arm"].astype(str)
    keep = pd.Series(True, index=frame.index)
    if prefix:
        keep &= names.str.startswith(prefix)
    if arm:
        keep &= names.str.fullmatch(arm)
    baseline = names == scaling.TORCH_DIST_ARM
    return frame.loc[(keep & ~baseline) | (baseline & torch_dist)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("observations", type=pathlib.Path, help="observations CSV or extracted .db")
    parser.add_argument("--experiment", default="", help="arm prefix selecting one experiment; blank keeps all")
    parser.add_argument("--arm", default="", help="regex; keep only arms whose full name matches")
    parser.add_argument("--figure", choices=FIGURES, default="all", help="which figure to draw (default: all)")
    parser.add_argument(
        "--mode",
        choices=scaling.MODES,
        default="weak",
        help="which scaling law the per-kernel small multiples draw (default: weak)",
    )
    parser.add_argument(
        "--quantity",
        type=scaling.Quantity,
        choices=tuple(scaling.Quantity),
        default=scaling.Quantity.EFFICIENCY,
        help="what the per-kernel small multiples put on Y (default: efficiency)",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=0.0,
        help="figure width in inches; default is the double-column paper width",
    )
    parser.add_argument(
        "--print-width",
        type=float,
        default=0.0,
        help="draw at PRINT size for a paper that places the figure at exactly this width in inches "
        "(e.g. 2.475 for an ICLR wrap figure); overrides --width",
    )
    parser.add_argument(
        "--kernels", nargs="+", default=[], help="per-kernel figure: the kernels that get a panel, in order"
    )
    parser.add_argument(
        "--no-torch-dist", action="store_true", help="leave out the torch.distributed baseline curve (arm torch_dist)"
    )
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("figures/scaling"))
    parser.add_argument("--table", type=pathlib.Path, default=pathlib.Path("data/scaling.csv"))
    return parser


def write_tables(curves: list[scaling.Curve], table: pathlib.Path) -> None:
    """The numbers behind the marks, and everything that was refused, beside the figures."""
    table.parent.mkdir(parents=True, exist_ok=True)
    scaling.points_table(curves).to_csv(table, index=False)
    dropped = scaling.dropped_table(curves)
    if not dropped.empty:
        dropped.to_csv(table.with_name(f"{table.stem}-dropped{table.suffix}"), index=False)


def report(curves: list[scaling.Curve]) -> None:
    """What was drawn and, in full, what was not -- on stderr, so a redirected table stays clean."""
    drawn = scaling.drawable(curves)
    short = scaling.single_point_curves(curves)
    missing = scaling.dropped_points(curves)
    print(f"{len(drawn)} curve(s) drawable of {len(curves)}", file=sys.stderr)
    for curve in short:
        print(
            f"  not drawn: {curve.arm} / {curve.kernel} / {curve.mode}: {len(curve.points)} point(s)", file=sys.stderr
        )
    for arm, kernel, mode, ranks, reason in missing:
        print(f"  no point: {arm} / {kernel} / {mode} at P={ranks}: {reason}", file=sys.stderr)
    for mode in scaling.MODES:
        common = scaling.common_kernels(curves, mode)
        all_kernels = {curve.kernel for curve in drawn if curve.mode == mode}
        if all_kernels:
            solo = sorted(all_kernels - common)
            print(
                f"  {mode}: {len(common)} kernel(s) every arm has"
                + (f"; not on every arm: {', '.join(solo)}" if solo else ""),
                file=sys.stderr,
            )


def draw(curves: list[scaling.Curve], args: argparse.Namespace) -> list[pathlib.Path]:
    """Every requested figure, saved under ``--out``. A figure with nothing to draw is skipped."""
    type_ = plotstyle.PRINT_SCALE if args.print_width else plotstyle.AUTHOR_SCALE
    width = args.print_width or args.width or plotstyle.DOUBLE_COLUMN_WIDTH
    written: list[pathlib.Path] = []
    wanted = FIGURES[1:] if args.figure == "all" else (args.figure,)
    for name in wanted:
        if name == "per-kernel":
            fig = scaling.figure_per_kernel(
                curves, args.mode, args.quantity, width=width, type_=type_, kernels=args.kernels
            )  # fmt: skip
            stem = args.out.with_name(f"{args.out.name}-per-kernel-{args.mode}")
        elif name == "mode-grid":
            fig = scaling.figure_mode_grid(curves, args.kernels, args.quantity, width=width, type_=type_)
            stem = args.out.with_name(f"{args.out.name}-{name}")
        else:
            fig = scaling.BUILDERS[name](curves, width=width, type_=type_)
            stem = args.out.with_name(f"{args.out.name}-{name}")
        if fig is None:
            print(f"nothing drawable for --figure {name}", file=sys.stderr)
            continue
        written.append(scaling.save(fig, stem, width_in=args.print_width))
    return written


def main() -> None:
    args = build_parser().parse_args()
    frame = load(args.observations, args.experiment, args.arm, not args.no_torch_dist)
    curves = scaling.curves(frame)
    if not curves:
        raise SystemExit(f"no scaling rows for experiment={args.experiment!r} arm={args.arm!r}")

    # A recorded eta that disagrees with the formula behind it means one of the two is wrong, and
    # no figure drawn from either is worth reading -- so this refuses rather than drawing it.
    mismatched = scaling.disagreements(frame)
    if mismatched:
        lines = "\n".join(
            f"  {arm} / {kernel} P={p}: recorded {a:.6g}, recomputed {b:.6g}" for arm, kernel, p, a, b in mismatched
        )
        raise SystemExit(f"{len(mismatched)} row(s) record an efficiency the times do not give:\n{lines}")

    write_tables(curves, args.table)
    report(curves)
    for path in draw(curves, args):
        print(f"figure -> {path}.pdf (+ .png)")


if __name__ == "__main__":
    main()
