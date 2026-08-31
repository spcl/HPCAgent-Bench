# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Choose the smallest kernel set that preserves the translators' EMIT coverage.

tests/test_e2e_numerical.py collects 2848 cases -- the corpus crossed with every backend, and about
two thirds of the whole suite. Most of that is not coverage: the corpus holds 151 tsvc_2_s* kernels,
27 matmul, 22 gemm, 20 conv2d. Each is a distinct BENCHMARK (that is the point of TSVC) but as a
TRANSLATOR test the 140th exercises the same emitter lines as the 10th.

So measure it rather than guess from names. For every kernel, emit it to every target under
coverage and record which lines of numpy_translators/src it touched; then greedily pick kernels
until the union is covered. What comes out is a per-push set whose emit coverage equals the full
corpus, with the remainder left for a scheduled run.

The measurement is EMIT coverage, not runtime numerics: it answers "which kernels drive different
translation logic", which is the question kernel selection asks. It does NOT claim two kernels
covering the same lines must produce the same numbers -- a per-kernel numerical bug still needs the
full sweep, which is why the remainder moves to a schedule instead of being deleted.
"""
import argparse
import json
import multiprocessing as mp
import pathlib
import sys
import traceback
from typing import Dict, List, Set, Tuple

CHUNK = 20
TIMEOUT_S = 180.0


def emit_under_coverage(key: str) -> Tuple[str, List[Tuple[str, int]], str]:
    """Emit one kernel to every target; return the numpy_translators/src lines it executed."""
    import coverage
    from hpcagent_bench import paths
    from hpcagent_bench.emit_bridge import bench_info_tempfile
    from hpcagent_bench.spec import BenchSpec

    root = str(pathlib.Path(paths.BENCHMARKS).parent / "numpy_translators" / "src")
    # source=, not include=: the repo's own coverage config sets source, and coverage then drops an
    # include= as redundant ("--include is ignored because --source is set") -- measuring the whole
    # package instead of the translators.
    cov = coverage.Coverage(source=[root], data_file=None, branch=False, config_file=False)
    status = "ok"
    cov.start()
    try:
        from numpyto_c import dace_emit, emit as c_emit  # noqa: F401 -- imported for its side effects
        from numpyto_common.frontend import emit_with_inline_fallback, parse_kernel
        spec = BenchSpec.load(key)
        kdir = paths.BENCHMARKS / spec.relative_path
        numpy_py = kdir / f"{spec.module_name}_numpy.py"
        with bench_info_tempfile(spec) as info:
            for emitter in (dace_emit.emit_dace, ):
                try:
                    emit_with_inline_fallback(lambda fn=emitter: fn(parse_kernel(numpy_py, info)))
                except Exception:  # noqa: BLE001 -- a refusal still covers the lines that decided it
                    status = "partial"
    except Exception as exc:  # noqa: BLE001
        status = f"error: {type(exc).__name__}: {exc}"[:120]
    finally:
        cov.stop()
    data = cov.get_data()
    lines: List[Tuple[str, int]] = []
    for filename in data.measured_files():
        if not filename.startswith(root):
            continue  # a line outside the translators says nothing about which kernels to keep
        rel = filename[len(root) + 1:]
        lines.extend((rel, n) for n in (data.lines(filename) or ()))
    return key, lines, status


def worker(keys: List[str], out: "mp.Queue") -> None:
    for key in keys:
        try:
            name, lines, status = emit_under_coverage(key)
            out.put({"kernel": name, "lines": lines, "status": status})
        except BaseException as exc:  # noqa: BLE001
            out.put({
                "kernel": key,
                "lines": [],
                "status": f"crash: {type(exc).__name__}",
                "frame": traceback.format_exc().strip().splitlines()[-1][:160],
            })
    out.put(None)


def measure(keys: List[str], destination: pathlib.Path) -> None:
    ctx = mp.get_context("spawn")
    with destination.open("w") as handle:
        for start in range(0, len(keys), CHUNK):
            chunk = keys[start:start + CHUNK]
            queue = ctx.Queue()
            proc = ctx.Process(target=worker, args=(chunk, queue))
            proc.start()
            seen = 0
            while seen < len(chunk):
                try:
                    record = queue.get(timeout=TIMEOUT_S)
                except Exception:  # noqa: BLE001 -- a wedged chunk costs the chunk, not the run
                    break
                if record is None:
                    break
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                seen += 1
            proc.terminate()
            proc.join()
            print(f"{start + seen}/{len(keys)}", flush=True)


def select(records: List[dict]) -> Tuple[List[str], int, int]:
    """Greedy set cover: repeatedly take the kernel adding the most uncovered lines."""
    coverage_by_kernel: Dict[str, Set[Tuple[str, int]]] = {
        r["kernel"]: {tuple(line)
                      for line in r["lines"]}
        for r in records if r["lines"]
    }
    universe: Set[Tuple[str, int]] = set().union(*coverage_by_kernel.values()) if coverage_by_kernel else set()
    remaining = set(universe)
    chosen: List[str] = []
    while remaining:
        best = max(coverage_by_kernel, key=lambda k: len(coverage_by_kernel[k] & remaining))
        gain = coverage_by_kernel[best] & remaining
        if not gain:
            break
        chosen.append(best)
        remaining -= gain
        del coverage_by_kernel[best]
    return chosen, len(universe), len(universe) - len(remaining)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--select-only", action="store_true", help="skip measuring; re-select from --out")
    args = parser.parse_args()

    if not args.select_only:
        from hpcagent_bench.spec import KERNELS
        # The same enumeration test_e2e_numerical.py parametrizes over, so the subset is chosen from
        # exactly the set the sweep runs.
        keys = sorted({key.rsplit("/", 1)[-1] for key in KERNELS})
        print(f"measuring {len(keys)} kernels", flush=True)
        measure(keys, args.out)

    records = [json.loads(line) for line in args.out.read_text().splitlines() if line.strip()]
    chosen, total, covered = select(records)
    print(f"\nkernels measured : {len(records)}")
    print(f"emit lines total : {total}")
    print(f"covered by subset: {covered} ({100.0 * covered / total:.2f}%)" if total else "no coverage")
    print(f"subset size      : {len(chosen)}")
    print("\n".join(f"  {k}" for k in chosen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
