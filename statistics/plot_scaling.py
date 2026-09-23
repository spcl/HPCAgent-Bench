# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Weak- and strong-scaling figures for the distributed ML-op track.

Reads one selection of observations (``--experiment`` for an arm prefix, ``--arm`` for a further
regex) and draws its scaling curves (:mod:`hpcagent_bench.stats.figures.scaling`): parallel
efficiency eta(P), speed-up sigma(P), the per-kernel small multiples, and the per-arm geomean eta
with its interval. Every figure writes a PDF, a PNG and the CSV behind the marks, plus a second CSV
naming every point the sweep did not measure and every curve too short to draw.

Usage::

    python statistics/plot_scaling.py obs.csv --experiment mlscale
    python statistics/plot_scaling.py obs.csv --experiment mlscale --figure efficiency
    python statistics/plot_scaling.py obs.csv --experiment mlscale --figure per-kernel --mode weak
    python statistics/plot_scaling.py obs.csv --arm 'mlscale-qwen38-hip.*'
"""

import argparse
import pathlib
import sys

import pandas as pd

from hpcagent_bench import experiments
from hpcagent_bench.stats.figures import scaling

#: ``--figure`` choices. ``all`` draws every one of them in a single pass over the frame.
FIGURES: tuple[str, ...] = ("all", "efficiency", "speedup", "per-kernel", "summary")


def load(path: pathlib.Path, prefix: str, arm: str) -> pd.DataFrame:
    """The observations frame, narrowed to one experiment prefix and one arm regex."""
    frame = experiments.read_observations(path)
    if prefix:
        frame = frame[frame["arm"].astype(str).str.startswith(prefix)]
    if arm:
        frame = frame[frame["arm"].astype(str).str.fullmatch(arm)]
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("observations", type=pathlib.Path, help="observations CSV or extracted .db")
    parser.add_argument("--experiment", default="", help="arm prefix selecting one experiment; blank keeps all")
    parser.add_argument("--arm", default="", help="regex; keep only arms whose full name matches")
    parser.add_argument("--figure", choices=FIGURES, default="all", help="which figure to draw (default: all)")
    parser.add_argument(
        "--mode",
        choices=scaling.MODES,
        default=None,
        help="which scaling law the per-kernel small multiples draw (default: both, one figure each)",
    )
    parser.add_argument(
        "--quantity",
        choices=("efficiency", "speedup"),
        default="efficiency",
        help="what the per-kernel small multiples put on Y (default: efficiency)",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=0.0,
        help="figure width in inches; default is the ICLR text width (scaling.DEFAULT_WIDTH_IN)",
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
    width = args.width or None
    written: list[pathlib.Path] = []
    wanted = FIGURES[1:] if args.figure == "all" else (args.figure,)
    sized = {"width": width} if width else {}
    for name in wanted:
        if name == "per-kernel":
            modes = (args.mode,) if args.mode else scaling.MODES
            drawn = [
                (scaling.figure_per_kernel(curves, mode, args.quantity, **sized), f"per-kernel-{mode}")
                for mode in modes
            ]
        else:
            drawn = [(scaling.BUILDERS[name](curves, **sized), name)]
        for fig, suffix in drawn:
            if fig is None:
                print(f"nothing drawable for --figure {name} ({suffix})", file=sys.stderr)
                continue
            written.append(scaling.save(fig, args.out.with_name(f"{args.out.name}-{suffix}")))
    return written


def main() -> None:
    args = build_parser().parse_args()
    frame = load(args.observations, args.experiment, args.arm)
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
