"""End-to-end smoke test for the `extended` branch.

Exercises:
- numpy / S preset / float32 (run_benchmark.py)
- numpy / S preset / float64 (run_benchmark.py)
- dace_cpu / S preset / float64 (run_benchmark.py)
- legacy NULL-datatype row migration path
- plot_grid.py with --datatype float32 and float64
- plot_results.py with --datatype float32 and float64

Runs in a scratch directory so it never touches the user's working
optarena.db. Exits non-zero on the first failure.
"""

import argparse
import os
import pathlib
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PYTHON = os.environ.get("OPTARENA_TEST_PYTHON", sys.executable)
NUMPY_KERNELS = ["atax", "mvt", "gemver"]
DACE_KERNELS = ["atax", "mvt"]


def _run(cmd, cwd, env, label, timeout):
    """Run a command, stream output to a log file, raise on non-zero exit."""
    print(f"[smoke] >>> {label}: {' '.join(cmd)}", flush=True)
    log = pathlib.Path(cwd) / f"{label.replace('/', '_')}.log"
    with log.open("w") as fp:
        rc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=fp,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        ).returncode
    if rc != 0:
        tail = log.read_text().splitlines()[-30:]
        raise RuntimeError(f"[smoke] {label} failed (rc={rc}); last 30 lines:\n" + "\n".join(tail))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true", help="Keep the scratch directory after the run.")
    ap.add_argument("--skip-dace", action="store_true", help="Skip the dace_cpu kernels (faster local sanity).")
    ap.add_argument("--skip-tvm",
                    action="store_true",
                    help="Skip the tvm_cpu kernel (faster local sanity; the "
                    "meta_schedule warmup is ~30s even at 4 trials).")
    ap.add_argument("--kernel-timeout", type=int, default=180, help="Per-kernel subprocess timeout (s, default 180).")
    args = ap.parse_args(argv)

    scratch = pathlib.Path(tempfile.mkdtemp(prefix="optarena-smoke-"))
    print(f"[smoke] scratch: {scratch}", flush=True)
    env = os.environ.copy()
    # ensure imports resolve against the source tree, not a stale install
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # TVM meta_schedule trials: keep the smoke fast — the kernel-author
    # tuned defaults (64-1024) take 30s-many-minutes each. 4 trials is
    # enough to verify the autotune pipeline runs end-to-end.
    env.setdefault("OPTARENA_TVM_METASCHEDULE_TRIALS", "4")

    try:
        # 0) theoretical hardware-info: call get_cpu_flops and the two
        # get_theoretical_bandwidth paths. The bandwidth paths depend on
        # dmidecode and silently return 0 without sudo; that's fine —
        # we assert only that the calls don't crash and that the FLOPS
        # tuple has the expected (fp32, fp64) shape with positive values.
        # plot_roofline.py also gets a smoke run so the rooflines render.
        _run(
            [
                PYTHON, "-c",
                textwrap.dedent("""
                import psutil, sys
                from optarena.hardware_info.theoretical.cpu_gpu_info import (
                    get_cpu_flops, get_theoretical_bandwidth as cpu_bw,
                )
                from optarena.hardware_info.theoretical.memory_info import (
                    get_theoretical_bandwidth as mem_bw,
                )
                n = psutil.cpu_count(logical=False) or 1
                fp32, fp64 = get_cpu_flops(n)
                assert fp32 > 0 and fp64 > 0, f"unexpected flops {(fp32, fp64)}"
                # bandwidth fallbacks are sudo-gated; should return 0 or
                # a positive number without raising
                cpu_bw_v = cpu_bw(n)
                mem_bw_v = mem_bw()
                print(f"[hardware_info] cores={n} fp32={fp32:.0f} "
                      f"fp64={fp64:.0f} GFLOP/s; "
                      f"cpu_bw={cpu_bw_v} mem_bw={mem_bw_v} GB/s")
            """).strip()
            ],
            cwd=scratch,
            env=env,
            label="theoretical_hwinfo",
            timeout=30,
        )

        # 1) seed a legacy NULL-datatype row so the migration + filter path
        # is exercised. The harness will run ALTER TABLE on first write.
        legacy_db = scratch / "optarena.db"
        with sqlite3.connect(legacy_db) as conn:
            conn.execute("""
                CREATE TABLE results (
                    id integer PRIMARY KEY, timestamp integer NOT NULL,
                    benchmark text NOT NULL, kind text, domain text,
                    dwarf text, preset text NOT NULL, mode text NOT NULL,
                    framework text NOT NULL, version text NOT NULL,
                    details text, validated integer, time real
                )
            """)
            conn.execute("INSERT INTO results VALUES "
                         "(NULL, 1, 'gemm', 'microbench', 'LinAlg', 'dense', "
                         "'S', 'main', 'numpy', '0.1', 'default', 1, 0.05)")

        # 2) run kernels at both precisions
        for dt in ("float32", "float64"):
            for k in NUMPY_KERNELS:
                _run(
                    [
                        PYTHON,
                        str(REPO_ROOT / "run_benchmark.py"), "-b", k, "-f", "numpy", "-p", "S", "-v", "True", "-r", "2",
                        "-d", dt
                    ],
                    cwd=scratch,
                    env=env,
                    label=f"numpy_{k}_{dt}",
                    timeout=args.kernel_timeout,
                )

        if not args.skip_dace:
            for k in DACE_KERNELS:
                for dt in ("float32", "float64"):
                    _run(
                        [
                            PYTHON,
                            str(REPO_ROOT / "run_benchmark.py"), "-b", k, "-f", "dace_cpu", "-p", "S", "-v", "True",
                            "-r", "2", "-d", dt
                        ],
                        cwd=scratch,
                        env=env,
                        label=f"dace_cpu_{k}_{dt}",
                        timeout=args.kernel_timeout,
                    )

        # TVM smoke (CPU) — exercise the meta_schedule autotune path
        # end-to-end. The 4-trial cap from env keeps this under a
        # minute. permute_3d is the cheapest TVM kernel we have today.
        if not args.skip_tvm:
            _run(
                [
                    PYTHON,
                    str(REPO_ROOT / "run_benchmark.py"), "-b", "permute_3d", "-f", "tvm_cpu", "-p", "S", "-v", "True",
                    "-r", "2", "-d", "float64"
                ],
                cwd=scratch,
                env=env,
                label="tvm_cpu_permute_3d_float64",
                timeout=max(args.kernel_timeout, 180),
            )

        # 2.5) sparse sweep — exercise run_sparse_benchmark.py on a small
        # subset of sparse benchmarks × variants. This validates the
        # variant-CLI threading, the new `variant` DB column, and the
        # _generators.build_sparse() path. SuiteSparse pulls are
        # disabled here to keep the smoke offline; the harness covers
        # them when run by hand.
        _run(
            [
                PYTHON,
                str(REPO_ROOT / "run_sparse_benchmark.py"), "-f", "numpy", "-p", "S", "-r", "1", "-d", "float64", "-b",
                "sp_cg", "sp_minres", "-V", "csr_uniform", "csc_uniform", "csr_banded", "--ignore-errors"
            ],
            cwd=scratch,
            env=env,
            label="run_sparse_benchmark",
            timeout=300,
        )

        # 3) assertions on the DB
        with sqlite3.connect(legacy_db) as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(results)")}
            assert "datatype" in cols, \
                f"datatype column missing post-migration; cols={cols}"
            assert "variant" in cols, \
                f"variant column missing post-migration; cols={cols}"

            counts = dict(
                conn.execute("SELECT COALESCE(datatype, 'NULL_LEGACY'), COUNT(*) "
                             "FROM results GROUP BY datatype"))
            print(f"[smoke] datatype breakdown: {counts}", flush=True)
            assert counts.get("float32", 0) >= len(NUMPY_KERNELS) * 2, \
                f"missing or insufficient fp32 rows: {counts}"
            assert counts.get("float64", 0) >= len(NUMPY_KERNELS) * 2, \
                f"missing or insufficient fp64 rows: {counts}"
            assert counts.get("NULL_LEGACY", 0) >= 1, \
                f"legacy NULL row was not preserved: {counts}"

            variant_counts = dict(
                conn.execute("SELECT COALESCE(variant, 'NULL'), COUNT(*) FROM results "
                             "WHERE benchmark IN ('sp_cg', 'sp_minres') "
                             "GROUP BY variant"))
            print(f"[smoke] sparse variant breakdown: {variant_counts}", flush=True)
            for v in ("csr_uniform", "csc_uniform", "csr_banded"):
                assert variant_counts.get(v, 0) >= 2, \
                    f"expected >= 2 rows for variant {v} (sp_cg + sp_minres); " \
                    f"got {variant_counts}"

        # plot_grid is "speedup over numpy", so it needs at least one
        # non-numpy framework to render anything. With --skip-dace there
        # is no such framework, so the script's clean-exit path is what
        # we verify there.
        expect_grid_png = not args.skip_dace

        # 4) plot_grid.py for both precisions
        for dt in ("float32", "float64"):
            # plot_grid.py filters on preset='paper'; re-tag a copy of the
            # rows as 'paper' so the script has data to plot.
            with sqlite3.connect(legacy_db) as conn:
                conn.execute(
                    "INSERT INTO results "
                    "(timestamp, benchmark, kind, domain, dwarf, preset, "
                    "mode, framework, version, details, validated, time, "
                    "datatype) "
                    "SELECT timestamp, benchmark, kind, domain, dwarf, "
                    "'paper', mode, framework, version, details, validated, "
                    "time, datatype FROM results WHERE preset='S' AND "
                    "COALESCE(datatype, 'float64')=?", (dt, ))
            # Override plot_grid's default GPU-only framework exclusion so
            # the dace_cpu test data is plotted. When --skip-dace is set,
            # the only non-numpy framework is missing and plot_grid prints
            # "nothing to plot" and exits 0 — that's the contract we check.
            _run(
                [
                    PYTHON,
                    str(REPO_ROOT / "plot_grid.py"), "--rows", "1", "--cols",
                    str(len(NUMPY_KERNELS)), "--datatype", dt, "--plot-kind", "violin", "--exclude-frameworks", ""
                ],
                cwd=scratch,
                env=env,
                label=f"plot_grid_{dt}",
                timeout=60,
            )
            grid_png = scratch / f"benchmark_grid_1x{len(NUMPY_KERNELS)}.png"
            if expect_grid_png:
                assert grid_png.exists(), f"missing {grid_png}"
                (scratch / f"grid_{dt}.png").write_bytes(grid_png.read_bytes())

        # 5) plot_results.py for both precisions (writes heatmap.pdf each
        # run; rename so both survive). plot_results.py has no framework
        # exclusion so dace_cpu data plots directly.
        if expect_grid_png:
            for dt in ("float32", "float64"):
                _run(
                    [PYTHON, str(REPO_ROOT / "plot_results.py"), "-p", "S", "-d", dt],
                    cwd=scratch,
                    env=env,
                    label=f"plot_results_{dt}",
                    timeout=60,
                )
                heat = scratch / "heatmap.pdf"
                assert heat.exists(), f"missing {heat}"
                heat.rename(scratch / f"heatmap_{dt}.pdf")

        # 6) plot_roofline.py — the theoretical-compute call exercised in
        # step 0 feeds straight into this. A bandwidth override is
        # passed so the memory roof always renders (without it the
        # dmidecode path silently returns 0 and the roof disappears).
        _run(
            [
                PYTHON,
                str(REPO_ROOT / "plot_roofline.py"), "-f", "numpy", "-p", "S", "-d", "float64", "--bandwidth-gb-s", "40"
            ],
            cwd=scratch,
            env=env,
            label="plot_roofline",
            timeout=60,
        )
        roof_pdf = scratch / "roofline.pdf"
        assert roof_pdf.exists(), f"missing {roof_pdf}"

        print(f"[smoke] OK — artifacts kept at {scratch}" if args.keep else "[smoke] OK", flush=True)
        if not args.keep:
            shutil.rmtree(scratch, ignore_errors=True)
        return 0

    except Exception as e:
        print(f"[smoke] FAIL: {e}", file=sys.stderr, flush=True)
        print(f"[smoke] scratch (kept on failure): {scratch}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
