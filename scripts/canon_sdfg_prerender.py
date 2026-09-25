# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Warm (or just report on) the DaCe base-SDFG cache a canon column reads from.

A ``dace_cpu``/``dace_cpu_canonicalize``/``dace_cpu_parallel`` run all read the SAME cached parse,
``<kernel_dir>/.cache/<module>_cpu.sdfgz`` (:mod:`hpcagent_bench.framework_cache`); the three
``dace_gpu*`` flavors likewise share ``<module>_gpu.sdfgz``. The parse itself (``program.to_sdfg``)
does not touch a device either way -- ``dace_framework.DaceFramework._build_sdfgs`` calls the exact
same builder for both tags, so the two cache files hold byte-identical graphs and this tool builds
ONCE per kernel and saves it under both tags, never spending a second parse on the one that already
matches. A HIT on either tag short-circuits the other's build too (load, don't reparse).

The cache key (:meth:`DaceFramework._sdfg_fingerprint`) is the kernel's numpy reference + its
generated ``<module>_dace.py`` + the RESOLVED datatype string + which DaCe tree parsed it. A canon
column never passes ``--datatype`` (``canon_column.sh``'s ``run-framework`` call has no ``-d``), so
the datatype a real run binds to is whatever :meth:`hpcagent_bench.frameworks.test.Test.run`
resolves for ``None``: fp64 by default, RESYNCED to the fuzzed data's own dtype when that differs
(``float_scalar_of`` over the materialized values) -- replicated here via :func:`resolve_datatype`
so a warm computes the SAME fingerprint a real run will look up, not a guess.

Two subcommands:

  single <kernel>   One kernel, in-process: resolve datatype, ensure the generated dace sibling,
                     check/build/save both tags. Prints one JSON line to stdout. This is the unit a
                     subprocess boundary should isolate -- ``set_datatype`` mutates process-global
                     state (:mod:`hpcagent_bench.frameworks.framework`'s ``np_float``/``np_complex``),
                     so two kernels sharing a process is a state-leak risk `sweep` avoids by giving
                     each kernel its own child.

  sweep             Drive `single` for a whole roster across a thread pool of subprocess workers,
                     one kernel per subprocess, each under its own wall timeout (a kernel that hangs
                     must cost its own budget, not the sweep's -- the same failure `canon_column.sh`'s
                     per-kernel `timeout` wrapper now guards against downstream). Writes one JSON
                     result per kernel under ``--out-dir`` and prints a fresh/warmed/failed coverage
                     table per device tag.

Usage:
    python3 scripts/canon_sdfg_prerender.py single tsvc_2_s315 --preset fuzzed
    python3 scripts/canon_sdfg_prerender.py sweep --roster experiments/kernels-llr248.txt \\
        --out-dir "$SCRATCH/audit-20260918/prerender/llr" --workers 16 --timeout 3600
"""

import argparse
import concurrent.futures
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import time

from hpcagent_bench import paths

DEVICE_TAGS = ("cpu", "gpu")


def parse_roster(spec: str) -> list[str]:
    """A roster from a comma-separated list, or (when ``spec`` names an existing file) one kernel
    per line -- each line ALSO comma-split, so a file holding ``roster_for``'s own comma-joined
    output (as opposed to ``experiments/prerender_cpf.sbatch``'s one-name-per-line convention) is
    read the same way instead of becoming one giant bogus "kernel" -- an earlier version of this
    tool did exactly that and blew up with ENAMETOOLONG writing that "kernel"'s result file.
    Lines may carry a trailing ``# note``. Sorted and de-duplicated so two callers naming the same
    roster differently still shard it identically."""
    path = pathlib.Path(spec)
    if path.is_file():
        raw = path.read_text().splitlines()
    else:
        raw = [spec]
    names: list[str] = []
    for line in raw:
        for part in line.split("#", 1)[0].split(","):
            stripped = part.strip()
            if stripped:
                names.append(stripped)
    return sorted(set(names))


@dataclasses.dataclass(frozen=True)
class KernelResult:
    """One kernel's outcome: a status per device tag (``fresh`` / ``warmed`` / ``missing`` /
    ``failed``) plus the datatype the fingerprint resolved to, and an error string when the kernel
    could not be parsed at all (both tags then read ``failed``)."""

    kernel: str
    cpu: str
    gpu: str
    datatype: str = ""
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))


def parse_result_line(line: str) -> KernelResult:
    """The inverse of :meth:`KernelResult.to_json`, for reading a `single` child's stdout back."""
    data = json.loads(line)
    return KernelResult(
        kernel=data["kernel"],
        cpu=data.get("cpu", "failed"),
        gpu=data.get("gpu", "failed"),
        datatype=data.get("datatype", ""),
        error=data.get("error", ""),
    )


def resolve_datatype(bench: object, preset: str) -> "str | None":
    """The datatype string a real (no ``--datatype``) canon run binds ``DaceFramework`` to for
    ``bench``: fp64 (``None``) unless the fuzzed data itself is materialized at a single other float
    dtype, mirroring :meth:`hpcagent_bench.frameworks.test.Test.run`'s resync (test.py, the
    "No --datatype was requested" branch) so the fingerprint computed here is the one that run
    would compute too."""
    from hpcagent_bench.frameworks.test import float_scalar_of

    bdata = bench.get_data(preset, None, fuzz_iteration=0)
    dtypes = {scalar for value in bdata.values() if (scalar := float_scalar_of(value)) is not None}
    if len(dtypes) > 1:
        raise ValueError(f"{bench.bname}: mixed float32/float64 values in fuzzed data")
    return dtypes.pop().__name__ if dtypes else None


def warm_kernel(kernel: str, preset: str, check_only: bool) -> KernelResult:
    """Resolve ``kernel``'s datatype, ensure its generated dace sibling, and check/build/save the
    base SDFG for both device tags -- ONE parse total, reused for whichever tag(s) were missing.
    ``check_only`` skips the build: a coverage read that never touches ``to_sdfg``."""
    from hpcagent_bench import framework_cache, paths
    from hpcagent_bench.frameworks import Benchmark, dace_framework

    fw = dace_framework.DaceFramework("dace_cpu")  # tag-independent: only used to parse + fingerprint
    fw.set_datatype(None)
    bench = Benchmark(kernel)
    datatype = resolve_datatype(bench, preset)
    fw.set_datatype(datatype)

    program = fw._import_kernel(bench)  # generates/loads <module>_dace.py before the fingerprint reads it
    kdir = paths.BENCHMARKS / bench.info["relative_path"]
    module_name = bench.info["module_name"]
    cache_dir = framework_cache.kernel_cache_dir(kdir)
    fingerprint = fw._sdfg_fingerprint(bench)

    statuses: dict[str, str] = {}
    built_sdfg = None
    for tag in DEVICE_TAGS:
        cached = framework_cache.load_sdfg(cache_dir, module_name, tag, fingerprint)
        if cached is not None:
            statuses[tag] = "fresh"
            built_sdfg = built_sdfg if built_sdfg is not None else cached
            continue
        if check_only:
            statuses[tag] = "missing"
            continue
        if built_sdfg is None:
            built_sdfg = program.to_sdfg(simplify=False)
        framework_cache.save_sdfg(cache_dir, module_name, tag, fingerprint, built_sdfg)
        statuses[tag] = "warmed"
    return KernelResult(kernel=kernel, cpu=statuses["cpu"], gpu=statuses["gpu"], datatype=str(datatype))


def build_coverage_table(results: list[KernelResult]) -> dict[str, dict[str, int]]:
    """Per-tag counts of fresh/warmed/missing/failed, from a list of :class:`KernelResult`."""
    table: dict[str, dict[str, int]] = {tag: {} for tag in DEVICE_TAGS}
    for result in results:
        for tag in DEVICE_TAGS:
            status = getattr(result, tag)
            table[tag][status] = table[tag].get(status, 0) + 1
    return table


def format_coverage_table(table: dict[str, dict[str, int]]) -> str:
    lines = []
    for tag in DEVICE_TAGS:
        counts = table.get(tag, {})
        total = sum(counts.values())
        parts = ", ".join(f"{status}={n}" for status, n in sorted(counts.items()))
        lines.append(f"  {tag}: {total} kernel(s) -- {parts or 'none'}")
    return "\n".join(lines)


def cmd_single(args: argparse.Namespace) -> int:
    try:
        result = warm_kernel(args.kernel, args.preset, args.check_only)
    except Exception as exc:  # noqa: BLE001 - a kernel that cannot parse is a reported result, not a crash
        result = KernelResult(
            kernel=args.kernel, cpu="failed", gpu="failed", error=f"{type(exc).__name__}: {exc}"[:500]
        )
    print(result.to_json(), flush=True)
    return 0 if not result.error else 1


def run_one(kernel: str, args: argparse.Namespace, child_env: dict[str, str], this_file: str) -> KernelResult:
    repo_python = str(pathlib.Path(args.opt) / "scripts" / "repo_python")
    cmd = [repo_python, "-u", this_file, "single", kernel, "--preset", args.preset]
    if args.check_only:
        cmd.append("--check-only")
    try:
        completed = subprocess.run(cmd, env=child_env, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return KernelResult(kernel=kernel, cpu="failed", gpu="failed", error=f"timeout after {args.timeout}s")
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if stdout_lines:
        try:
            return parse_result_line(stdout_lines[-1])
        except (json.JSONDecodeError, KeyError):
            pass  # fall through to the stderr-tail report below
    tail = (completed.stderr or completed.stdout or "no output")[-500:]
    return KernelResult(kernel=kernel, cpu="failed", gpu="failed", error=f"rc={completed.returncode}: {tail}")


def cmd_sweep(args: argparse.Namespace) -> int:
    kernels = parse_roster(args.roster)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    this_file = str(pathlib.Path(__file__).resolve())

    # --opt's scripts/repo_python puts that checkout (and DACE_TREE ahead of it) on the child's path.
    child_env = {**os.environ, "REPO_PYTHON": sys.executable}
    child_env.pop("PYTHONPATH", None)
    if args.dace_tree:
        child_env["DACE_TREE"] = args.dace_tree

    results: list[KernelResult] = []
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, kernel, args, child_env, this_file): kernel for kernel in kernels}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            kernel = futures[future]
            result = future.result()
            results.append(result)
            (out_dir / f"{kernel}.json").write_text(result.to_json())
            elapsed = time.monotonic() - start
            print(
                f"[{i}/{len(kernels)}] {kernel}: cpu={result.cpu} gpu={result.gpu} "
                f"dtype={result.datatype or 'default'} ({elapsed:.0f}s elapsed)"
                + (f" ERROR: {result.error}" if result.error else ""),
                flush=True,
            )

    table = build_coverage_table(results)
    summary = {
        "roster": args.roster,
        "kernel_count": len(kernels),
        "coverage": table,
        "failures": [{"kernel": r.kernel, "error": r.error} for r in results if r.cpu == "failed" or r.gpu == "failed"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\ncoverage for {len(kernels)} kernel(s):")
    print(format_coverage_table(table))
    if summary["failures"]:
        print(f"\n{len(summary['failures'])} kernel(s) failed to parse:")
        for f in summary["failures"]:
            print(f"  {f['kernel']}: {f['error']}")
    return 0


def default_opt() -> str:
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return str(pathlib.Path(scratch) / "hpcagent-bench")
    return str(paths.repo_root())


def default_dace_tree() -> str:
    return os.environ.get("DACE_TREE", "")


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    single = sub.add_parser("single", help="warm/check one kernel, in-process")
    single.add_argument("kernel")
    single.add_argument("--preset", default="fuzzed")
    single.add_argument("--check-only", action="store_true", help="report freshness only, never build")
    single.set_defaults(func=cmd_single)

    sweep = sub.add_parser("sweep", help="drive `single` over a roster, one subprocess per kernel")
    sweep.add_argument("--roster", required=True, help="comma-separated kernel names, or a roster file path")
    sweep.add_argument("--out-dir", required=True, help="per-kernel JSON results + summary.json land here")
    sweep.add_argument("--preset", default="fuzzed")
    sweep.add_argument("--check-only", action="store_true", help="report freshness only, never build")
    sweep.add_argument("--workers", type=int, default=16)
    sweep.add_argument("--timeout", type=float, default=3600.0, help="per-kernel wall cap, seconds")
    sweep.add_argument("--opt", default=default_opt(), help="repo root the child processes import from")
    sweep.add_argument("--dace-tree", default=default_dace_tree(), help="DACE_TREE for the child processes")
    sweep.set_defaults(func=cmd_sweep)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
