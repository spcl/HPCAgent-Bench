# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoke test: ``statistics/render_sample_plots.py`` renders every sample figure from stub-random
data end to end -- the per-kernel + geomean figure, every scaling figure, and the 3D stack figure --
each as a non-empty PDF and PNG under ``--out-dir``.

Runs the script as a real subprocess (its own ``sys.executable`` invocation of
``statistics/plot_per_kernel.py`` and ``statistics/plot_scaling.py``, exactly as a person would),
so this test exercises the actual CLI surface rather than importing around it.
"""

import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Every file the script must have written by the time it exits 0.
EXPECTED_SUFFIXES: tuple[str, ...] = (
    "40-kernels-geomean-speedup.pdf",
    "40-kernels-geomean-speedup.png",
    "40-kernels-geomean-tokens.pdf",
    "scaling-efficiency.pdf",
    "scaling-speedup.pdf",
    "scaling-summary.pdf",
    "3d-stack.pdf",
    "3d-stack.png",
)


def test_render_sample_plots_writes_every_figure_from_stub_data(tmp_path: pathlib.Path) -> None:
    out_dir = tmp_path / "sample-plots"
    result = subprocess.run(
        [
            sys.executable,
            "statistics/render_sample_plots.py",
            "--out-dir",
            str(out_dir),
            "--seed",
            "3",
            "--kernel-count",
            "6",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    figures_dir = out_dir / "figures"
    for name in EXPECTED_SUFFIXES:
        path = figures_dir / name
        assert path.is_file(), f"missing {path}\nstderr:\n{result.stderr}"
        assert path.stat().st_size > 0, f"{path} is empty"

    assert (out_dir / "data" / "stub_observations.csv").is_file()
    assert (out_dir / "tables" / "40-kernels-geomean-speedup.csv").is_file()
    assert (out_dir / "tables" / "3d-stack.csv").is_file()


def test_render_sample_plots_is_deterministic(tmp_path: pathlib.Path) -> None:
    """The same seed writes byte-identical figures on two independent runs (matplotlib's
    ``UNDATED`` metadata plus :func:`hpcagent_bench.stats.stub_data.generate`'s determinism)."""
    outputs = []
    for label in ("a", "b"):
        out_dir = tmp_path / label
        result = subprocess.run(
            [sys.executable, "statistics/render_sample_plots.py", "--out-dir", str(out_dir), "--seed", "11"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        outputs.append((out_dir / "data" / "stub_observations.csv").read_bytes())
    assert outputs[0] == outputs[1]
