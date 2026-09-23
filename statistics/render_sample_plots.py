# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render one sample of every figure this repo currently maintains, from STUB-RANDOM data.

No cluster run is needed: :mod:`hpcagent_bench.stats.stub_data` writes a seeded, deterministic
observations CSV in the exact schema :func:`hpcagent_bench.experiments.read_observations` reads
(the one every real figure reads), and this script drives the SAME entry points a real campaign
would -- ``statistics/plot_per_kernel.py`` and ``statistics/plot_scaling.py`` as subprocesses (so a
change to either script's CLI is exercised here too, not bypassed), plus the library call for the
new 3D stacked-bar figure (:mod:`hpcagent_bench.stats.figures.stack3d`), which has no CLI script of
its own yet.

Three figure families, matching ``make sample-plots``:

* per-kernel + geomean (``--summary``): every stub kernel's speed-up, with the geomean column.
* weak- and strong-scaling (``--figure all``): efficiency, speed-up, per-kernel small multiples,
  and the per-arm geomean summary, both scaling laws.
* the 3D stacked-bar figure: a representative kernel subset, extruded per arm.

THE 2D EFFICACY/PARETO SCATTER (:mod:`hpcagent_bench.stats.figures.efficacy`'s ``figure_one`` /
``figure_row``) IS DELIBERATELY NOT RENDERED HERE. It is superseded by the tuple-efficacy stacked
row (2026-09-21 decision, ``project_efficacy_tuple_pareto_20260921``: "2-D plot: ignore for now")
and the user asked for it dropped from new figure work. Its code still backs the still-used
``figure_dot_row`` stacked-tuple figure in the same module, so it was NOT deleted here -- that is a
separate, larger change (touching ``docs/plotting.md``'s numbered rules, ``scripts/plot_score_
change.py``, and every pinned test in ``tests/test_plot_score_change.py``) that needs its own
reviewed change, not a side effect of a sample-data script.

Usage::

    python statistics/render_sample_plots.py
    python statistics/render_sample_plots.py --out-dir /path/to/sample-plots --seed 7
"""

import argparse
import pathlib
import subprocess
import sys

from hpcagent_bench import paths
from hpcagent_bench.stats import stub_data
from hpcagent_bench.stats.figures import stack3d

REPO = pathlib.Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=20260923, help="stub data seed (deterministic)")
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="where to write data/ and figures/ (default: $SCRATCH/sample-plots-0923, or ./.cache/sample-plots-0923)",
    )
    parser.add_argument(
        "--kernel-count", type=int, default=stack3d.DEFAULT_KERNEL_COUNT, help="kernels drawn on the 3D figure"
    )
    return parser


def write_observations(out_dir: pathlib.Path, seed: int) -> pathlib.Path:
    """The one stub CSV every figure below reads, in the real observations schema."""
    dataset = stub_data.generate(seed=seed)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "stub_observations.csv"
    dataset.combined().to_csv(csv_path, index=False)
    return csv_path


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(part) for part in cmd), file=sys.stderr)
    subprocess.run(cmd, check=True, cwd=REPO)


def render_per_kernel(csv_path: pathlib.Path, figures_dir: pathlib.Path, tables_dir: pathlib.Path) -> None:
    """The 40-kernel + geomean figure: one treated arm (a model x the ``cpf`` packet), ``--summary``
    appends the geomean column the user asked to see "on the right"."""
    run(
        [
            sys.executable,
            "statistics/plot_per_kernel.py",
            str(csv_path),
            "--arm",
            r"sample-plots-qwen38-hip-cpf",
            "--summary",
            "--label",
            "Sample: 40 Kernels + Geomean",
            "--out",
            str(figures_dir / "40-kernels-geomean.pdf"),
            "--table",
            str(tables_dir / "40-kernels-geomean.csv"),
        ]
    )


def render_scaling(csv_path: pathlib.Path, figures_dir: pathlib.Path, tables_dir: pathlib.Path) -> None:
    """Every scaling figure (efficiency, speed-up, per-kernel, summary), both laws pooled: the
    script itself draws weak and strong as separate panels/curves per
    :mod:`hpcagent_bench.stats.figures.scaling`."""
    run(
        [
            sys.executable,
            "statistics/plot_scaling.py",
            str(csv_path),
            "--experiment",
            "sample-plots",
            "--figure",
            "all",
            "--out",
            str(figures_dir / "scaling"),
            "--table",
            str(tables_dir / "scaling.csv"),
        ]
    )


def render_stack3d(
    csv_path: pathlib.Path, figures_dir: pathlib.Path, tables_dir: pathlib.Path, kernel_count: int
) -> None:
    """The 3D stacked-bar figure, in-process (no CLI script of its own -- see the module
    docstring)."""
    import pandas as pd

    frame = pd.read_csv(csv_path, low_memory=False)
    arms, kernels, drawn = stack3d.drawn_bars(frame, kernel_count=kernel_count)
    if not drawn:
        raise SystemExit("stub data produced nothing drawable for the 3D stack figure")
    fig = stack3d.figure_stack3d(frame, arms=arms, kernels=kernels, title="Sample: 3D Stacked Speed-Up")
    assert fig is not None  # drawn is non-empty, so figure_stack3d cannot refuse this selection
    stack3d.save(fig, figures_dir / "3d-stack")
    tables_dir.mkdir(parents=True, exist_ok=True)
    stack3d.bars_table(drawn).to_csv(tables_dir / "3d-stack.csv", index=False)


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.out_dir or paths.scratch_root("sample-plots-0923")
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    csv_path = write_observations(out_dir, args.seed)
    render_per_kernel(csv_path, figures_dir, tables_dir)
    render_scaling(csv_path, figures_dir, tables_dir)
    render_stack3d(csv_path, figures_dir, tables_dir, args.kernel_count)

    written = sorted(figures_dir.glob("*"))
    print(f"wrote {len(written)} file(s) under {figures_dir}", file=sys.stderr)
    for path in written:
        print(f"  {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
