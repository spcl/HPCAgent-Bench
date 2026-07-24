# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end reporting test: produce a results DB, then drive the WHOLE reporting path over it
-- ``plot_heatmap`` + ``plot_distribution_grid`` + ``stats.median_ci`` -- and assert the PDFs come
out non-empty and the stats are finite.

The DB is produced by a REAL native (no-container) run when that is feasible on the box: numpy +
dace_cpu over one simple, known-good stencil (``heat_3d``) at preset S with a few reps, forked into
its own process under an 8GB memory cap. When the native build is not feasible here (no compiler /
memory / a per-kernel dace-emit bug), the test falls back to a small SYNTHETIC DB that mimics the
real ``results`` schema so the reporting path is always covered rather than skipped (repo rule: no
skipped tests). Which path ran is logged.

Rendered with ``usetex=False`` so the figures build without a LaTeX install (mathtext still renders
the CI superscripts)."""
import os
import pathlib
import shutil
import subprocess
import sqlite3
import sys
import textwrap
from typing import List

import numpy as np
import pytest

from hpcagent_bench import stats
from hpcagent_bench.plotting import cell_summary, load_results, plot_distribution_grid, plot_heatmap

#: Known-good simple stencil that builds + validates under native dace (verified on this box).
_NATIVE_KERNEL_STEM = "heat_3d"  # directory stem; recorded under short_name "heat3d"


def _native_env(cwd: pathlib.Path) -> dict:
    """Environment for the forked native sweep: isolated dace cache, single-threaded, MPI
    anti-hang vars, and the repo on PYTHONPATH so the child can import hpcagent_bench."""
    import hpcagent_bench
    repo_root = pathlib.Path(hpcagent_bench.__file__).resolve().parents[1]
    cache = cwd / "dacecache"
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(repo_root) + os.pathsep + env.get("PYTHONPATH", ""),
        DACE_default_build_folder=str(cache / ".dacecache"),
        OMP_NUM_THREADS="1",
        MPLBACKEND="Agg",
        OMPI_MCA_rmaps_base_oversubscribe="1",
        OMPI_MCA_pml="ob1",
        OMPI_MCA_btl="^openib",
        MPICH_NO_LOCAL="1",
        PMIX_MCA_gds="hash",
    )
    return env


def _capped(argv: List[str]) -> List[str]:
    """Wrap ``argv`` in an 8GB systemd memory cap when systemd-run is available (the run
    discipline for native builds); otherwise run uncapped."""
    if shutil.which("systemd-run"):
        return ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=8G", "--"] + argv
    return argv


def _db_has_validated(db: pathlib.Path, frameworks: List[str]) -> bool:
    """True iff ``db`` holds >= 1 validated preset-S row for every named framework."""
    if not db.exists():
        return False
    con = sqlite3.connect(db)
    try:
        for fw in frameworks:
            n = con.execute("SELECT COUNT(*) FROM results WHERE framework=? AND validated=1 AND preset='S'",
                            (fw, )).fetchone()[0]
            if not n:
                return False
    except sqlite3.Error:
        return False
    finally:
        con.close()
    return True


def _attempt_native(work: pathlib.Path) -> bool:
    """Run numpy + dace_cpu over the simple stencil into ``work/hpcagent_bench.db`` (forked, memory
    capped, timed out). Returns True iff both frameworks produced validated rows."""
    db = work / "hpcagent_bench.db"
    script = textwrap.dedent(f"""
        from hpcagent_bench.support.collect.sweep import run_benchmark_sweep
        # numpy first (the required denominator), then the dace_cpu optimization.
        run_benchmark_sweep({_NATIVE_KERNEL_STEM!r}, "numpy",    "S", True, 5, 120.0, False, False, "float64")
        run_benchmark_sweep({_NATIVE_KERNEL_STEM!r}, "dace_cpu", "S", True, 5, 120.0, False, False, "float64")
    """)
    argv = _capped([sys.executable, "-c", script])
    try:
        subprocess.run(argv, cwd=str(work), env=_native_env(work), timeout=360, capture_output=True, text=True)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return _db_has_validated(db, ["numpy", "dace_cpu"])


def _build_synthetic(db: pathlib.Path) -> None:
    """Write a small SYNTHETIC results DB with the real schema: two structured-grid kernels x
    (numpy, dace_cpu, numba), a handful of jittered samples each plus a slow outlier on one cell
    so the outlier-drop + bootstrap-CI path is genuinely exercised."""
    from sqlmodel import Session

    from hpcagent_bench.frameworks.schema import Result, results_engine

    rng = np.random.default_rng(0)
    engine = results_engine(str(db))
    # (kernel short_name, domain) -- real short_names so the ordering resolves them to HPC.
    kernels = [("heat3d", "Physics"), ("jacobi2d", "Physics")]
    # framework -> baseline runtime (ms); numpy is the slow reference.
    fw_base = {"numpy": 10.0, "dace_cpu": 4.0, "numba": 6.0}
    with Session(engine) as session:
        for kernel, domain in kernels:
            for fw, base in fw_base.items():
                samples = (base + rng.normal(0.0, base * 0.03, 8)).tolist()
                if fw == "numpy":
                    samples.append(base * 9.0)  # a slow OS-hiccup outlier: drop_outliers must catch it
                for t in samples:
                    session.add(
                        Result(timestamp=1_700_000_000,
                               benchmark=kernel,
                               domain=domain,
                               preset="S",
                               framework=fw,
                               agent=None,
                               validated=True,
                               time=float(t),
                               native_time=None,
                               datatype="float64",
                               variant=None,
                               prompt_hash=None,
                               execution="native"))
        session.commit()


@pytest.mark.integration
def test_reporting_pipeline_end_to_end(tmp_path, capsys) -> None:
    work = tmp_path
    db = work / "hpcagent_bench.db"

    ran_native = _attempt_native(work)
    if ran_native:
        source = "native-dace"
    else:
        _build_synthetic(db)
        source = "synthetic"
    # Surface which path ran (repo rule: never silently skip; log the branch taken).
    with capsys.disabled():
        print(f"\n[reporting-e2e] data source: {source}")

    assert db.exists() and db.stat().st_size > 0

    # --- the reporting path: heatmap + distribution grid + median CI -------------------
    heatmap = work / "heatmap.pdf"
    violin = work / "dist_violin.pdf"
    box = work / "dist_box.pdf"

    plot_heatmap(benchmark="all", preset="S", datatype="float64", db=str(db), output=str(heatmap), usetex=False)
    plot_distribution_grid(benchmark="all",
                           preset="S",
                           datatype="float64",
                           kind="violin",
                           db=str(db),
                           output=str(violin),
                           usetex=False)
    plot_distribution_grid(benchmark="all",
                           preset="S",
                           datatype="float64",
                           kind="box",
                           db=str(db),
                           output=str(box),
                           usetex=False)

    for pdf in (heatmap, violin, box):
        assert pdf.exists(), pdf
        assert pdf.stat().st_size > 1000, f"{pdf} looks empty ({pdf.stat().st_size} bytes)"
        assert pdf.read_bytes()[:4] == b"%PDF"

    # --- the stats path: cleaned median + finite bootstrap CI --------------------------
    data = load_results(str(db), "all", "S", "float64")
    assert not data.empty and "numpy" in set(data["framework"])

    summary = cell_summary(data)
    assert len(summary) >= 2
    assert np.isfinite(summary["time"].to_numpy()).all()
    assert np.isfinite(summary["ci_perc"].to_numpy()).all()

    # median_ci over a real cell's samples: median finite and bracketed by its CI.
    numpy_cell = data[(data["framework"] == "numpy")]["time"].to_numpy()
    med, lo, hi, _n = stats.median_ci(numpy_cell, warn=False)
    assert np.isfinite([med, lo, hi]).all()
    assert lo <= med <= hi
