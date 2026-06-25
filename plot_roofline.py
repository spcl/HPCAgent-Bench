"""Roofline plot.

Combines:
- Theoretical CPU peak FLOPS + memory bandwidth from
  ``optarena/hardware_info/theoretical/`` (no sudo required for the
  theoretical numbers; the practical HPL/STREAM path is separate).
- Measured runtimes from ``optarena.db`` for the chosen framework /
  preset / datatype.
- Per-benchmark FLOP and byte estimates from
  ``bench_info[<short_name>]["flops"]`` and ``["bytes"]`` — when both
  are present and a parameters preset is selected, the benchmark's
  arithmetic intensity (FLOPs / byte) and measured performance
  (GFLOP/s = flops / 1e9 / median_time) are plotted as a scatter point.

Benchmarks without flops/bytes entries are listed under "Skipped:" in
stdout so it's obvious what's missing. The rooflines themselves render
even with zero data, so the script doubles as "show me what this CPU
is theoretically capable of".

Outputs: ``roofline.pdf`` + ``roofline.png`` in the current directory.

Example::

    python plot_roofline.py -f numpy -p paper -d float64
    python plot_roofline.py --dtype-peak float32  # use the fp32 peak line
"""

import argparse
import json
import pathlib
import sqlite3
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = pathlib.Path(__file__).parent.resolve()
BENCH_DIR = REPO_ROOT / "bench_info"


def _safe_call(func, *args, **kwargs):
    """Catch the typical hardware_info hiccups (psutil/GPUtil/sudo) and
    return None instead of crashing the whole plot."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"[roofline] WARNING: {func.__name__} failed: {e}",
              file=sys.stderr)
        return None


def get_theoretical_caps(num_cores=None, dtype_peak="float64",
                          bandwidth_override=None):
    """Return (peak_gflops, bandwidth_gb_s)."""
    from optarena.hardware_info.theoretical.cpu_gpu_info import (
        get_cpu_flops, get_theoretical_bandwidth,
    )
    import psutil
    if num_cores is None:
        num_cores = psutil.cpu_count(logical=False) or 1
    flops = _safe_call(get_cpu_flops, num_cores) or (None, None)
    # (fp32_gflops, fp64_gflops). Default to fp64 row.
    peak = flops[0] if dtype_peak == "float32" else flops[1]
    if bandwidth_override is not None:
        bw = bandwidth_override
    else:
        bw = _safe_call(get_theoretical_bandwidth, num_cores)
        # Both theoretical-bandwidth paths in hardware_info depend on
        # dmidecode and silently return 0 without sudo. Treat 0 as "not
        # available" so the roof line is omitted rather than drawn at
        # zero.
        if not bw:
            bw = None
            print("[roofline] memory bandwidth unavailable (likely needs "
                  "sudo for dmidecode); pass --bandwidth-gb-s to override.",
                  file=sys.stderr)
    return peak, bw, num_cores


def load_bench_estimates(preset: str, datatype: str,
                         use_dace_analysis: bool = True):
    """Yield (short_name, flops, bytes, source) for every bench_info.

    ``source`` is "json" when the bench_info JSON statically declares
    ``flops``/``bytes`` (per-preset or per-(preset, dtype) dicts, or a
    single scalar). When neither is present and ``use_dace_analysis``
    is true, the dace work_depth + total_volume passes are run lazily
    via :mod:`optarena.infrastructure.dace_analysis`; ``source`` is
    then "dace". When neither path produces a value the bench is
    skipped silently.
    """
    dace_cache: dict = {}
    for path in sorted(BENCH_DIR.glob("*.json")):
        with path.open() as fp:
            cfg = json.load(fp).get("benchmark", {})
        short_name = cfg.get("short_name")
        if short_name is None:
            continue
        flops = cfg.get("flops")
        bytes_ = cfg.get("bytes")
        f = _resolve_estimate(flops, preset, datatype) if flops else None
        b = _resolve_estimate(bytes_, preset, datatype) if bytes_ else None
        if f is not None and b is not None:
            yield short_name, float(f), float(b), "json"
            continue
        if not use_dace_analysis:
            continue
        # Lazy fallback: dace work_depth + total_volume passes. Only
        # available on the modernize-perf-analysis-and-memory-volume
        # branch (PR pending); on yakup/dev / main this is a no-op.
        try:
            from optarena.infrastructure.dace_analysis import get_flops_bytes
        except Exception:
            return
        df, dbytes = get_flops_bytes(short_name, preset, datatype, dace_cache)
        if df is not None and dbytes is not None:
            yield short_name, float(df), float(dbytes), "dace"


def _resolve_estimate(node, preset, datatype):
    if isinstance(node, (int, float)):
        return node
    if isinstance(node, dict):
        if preset in node:
            child = node[preset]
        else:
            return None
        if isinstance(child, (int, float)):
            return child
        if isinstance(child, dict) and datatype in child:
            return child[datatype]
    return None


def load_runtimes(conn, framework, preset, datatype):
    """Return ``{short_name: median_time_ms}`` from the DB (times are ms)."""
    rows = conn.execute(
        "SELECT benchmark, time FROM results WHERE framework=? AND preset=? "
        "AND COALESCE(datatype, 'float64')=? AND validated=1",
        (framework, preset, datatype),
    ).fetchall()
    if not rows:
        return {}
    # Aggregate to median per benchmark.
    grouped = {}
    for bench, t in rows:
        grouped.setdefault(bench, []).append(t)
    return {b: float(np.median(ts)) for b, ts in grouped.items()}


def plot(peak_gflops, bw_gb_s, points, output_base, dtype_peak,
         num_cores, preset, framework):
    """Render the roofline."""
    fig, ax = plt.subplots(figsize=(8, 6))
    # Axis ranges. Pick a sensible default if no points yet.
    intensities = [p[1] for p in points] if points else [0.1, 100.0]
    perfs = [p[2] for p in points] if points else []
    xmin = max(1e-3, min(intensities) * 0.5)
    xmax = max(intensities) * 2 if intensities else 100.0
    if peak_gflops and bw_gb_s:
        ridge_intensity = peak_gflops / bw_gb_s
        xmax = max(xmax, ridge_intensity * 4)

    xs = np.geomspace(xmin, xmax, 200)

    if bw_gb_s:
        mem_roof = bw_gb_s * xs  # GFLOPS = (GB/s) * (FLOP/B)
        if peak_gflops:
            mem_roof = np.minimum(mem_roof, peak_gflops)
        ax.plot(xs, mem_roof, lw=2.0, color="C0",
                label=f"memory roof ({bw_gb_s:.0f} GB/s)")
    if peak_gflops:
        ax.axhline(peak_gflops, color="C3", lw=2.0, ls="--",
                   label=f"compute roof ({peak_gflops:.0f} GFLOP/s, {dtype_peak})")

    # Scatter benchmarks.
    for name, intensity, perf in points:
        ax.scatter(intensity, perf, s=80, alpha=0.85, edgecolor="k",
                   zorder=5)
        ax.annotate(name, (intensity, perf), xytext=(4, 4),
                    textcoords="offset points", fontsize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    if peak_gflops:
        ax.set_ylim(max(1e-1, perfs and min(perfs) * 0.3 or 1), peak_gflops * 1.5)
    ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    title = f"{framework} @ {preset} ({dtype_peak}); {num_cores} cores"
    ax.set_title(f"Roofline — {title}", fontsize=11)
    ax.grid(which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    pdf = output_base.with_suffix(".pdf")
    png = output_base.with_suffix(".png")
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=200)
    print(f"[roofline] wrote {pdf} and {png}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-f", "--framework", default="numpy")
    ap.add_argument("-p", "--preset", choices=["S", "M", "L", "paper"],
                    default="paper")
    ap.add_argument("-d", "--datatype", choices=["float32", "float64"],
                    default="float64")
    ap.add_argument("--dtype-peak", choices=["float32", "float64"],
                    default=None,
                    help="Which CPU peak to draw the compute roof at. "
                         "Defaults to --datatype.")
    ap.add_argument("--num-cores", type=int, default=None,
                    help="Override the physical core count psutil reports.")
    ap.add_argument("--bandwidth-gb-s", type=float, default=None,
                    help="Override the memory bandwidth (GB/s). Useful when "
                         "the dmidecode-based detection requires sudo.")
    ap.add_argument("--db", default="optarena.db",
                    help="Path to the optarena results SQLite database.")
    ap.add_argument("--output", default="roofline",
                    help="Output file base (without extension; .pdf and "
                         ".png are produced).")
    args = ap.parse_args(argv)

    dtype_peak = args.dtype_peak or args.datatype
    peak, bw, num_cores = get_theoretical_caps(args.num_cores, dtype_peak,
                                                args.bandwidth_gb_s)
    print(f"[roofline] CPU peak {dtype_peak} = {peak} GFLOP/s, "
          f"memory BW = {bw} GB/s, cores = {num_cores}")

    points = []
    skipped = []
    db_path = pathlib.Path(args.db)
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            runtimes = load_runtimes(conn, args.framework, args.preset,
                                     args.datatype)
        sources = {}
        for short_name, flops, byts, src in load_bench_estimates(
                args.preset, args.datatype):
            sources[short_name] = src
            if short_name not in runtimes:
                skipped.append((short_name, "no DB row"))
                continue
            median_ms = runtimes[short_name]
            perf = flops / 1e9 / (median_ms / 1e3)  # DB times are milliseconds
            intensity = flops / byts
            points.append((short_name, intensity, perf))
        if sources:
            from collections import Counter
            tally = Counter(sources.values())
            print(f"[roofline] flops/bytes sources: "
                  f"{dict(tally)}")
    else:
        print(f"[roofline] {args.db} not present — rendering rooflines only.")

    # Also report which benches had no flops/bytes annotation.
    annotated = {n for n, _, _, _ in load_bench_estimates(args.preset, args.datatype,
                                                          use_dace_analysis=False)}
    all_benches = {json.load(p.open())["benchmark"]["short_name"]
                   for p in BENCH_DIR.glob("*.json")}
    for n in sorted(all_benches - annotated):
        skipped.append((n, "no flops/bytes in bench_info"))

    plot(peak, bw, points, pathlib.Path(args.output),
         dtype_peak, num_cores, args.preset, args.framework)

    if skipped:
        print(f"[roofline] {len(skipped)} bench(es) skipped:")
        for n, why in skipped[:20]:
            print(f"    {n:<30} {why}")
        if len(skipped) > 20:
            print(f"    ... and {len(skipped) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
