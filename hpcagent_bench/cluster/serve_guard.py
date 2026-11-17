# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The KV-pool guard of a served model: the pool the server allocated against what its agents need.

A pipeline serves ``AGENTS_PER_PIPELINE`` agents, each holding a conversation of about ``KV_TOKENS_PER_AGENT`` tokens
that the radix cache must keep (docs/serving/kimi27sglang.md, "Tuning record"): a smaller pool evicts the waiting
conversations and every turn re-prefills. ``--mem-fraction-static`` 0.48 gave 0.15 to 1.0 M tokens, different on every
start, and no log line said so. The guard reads the pool from the server logs and fails the job (``KV_GUARD=warn``:
reports it) when any pipeline's is below the floor. A model that sets no ``KV_TOKENS_PER_AGENT`` has no guard.

    python -m hpcagent_bench.cluster.serve_guard watch --env ENV --run-dir DIR --job ID   # run_cluster.sh, in the batch shell
    python -m hpcagent_bench.cluster.serve_guard check --env ENV --run-dir DIR            # serve-only.sbatch, after the API is up
"""

import argparse
import dataclasses
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence

from hpcagent_bench.cluster import env_spec

__all__ = ["REPORT", "Verdict", "check", "floor_tokens", "main", "pool_tokens", "verdict"]

#: Written into the run directory.
REPORT = "kv-guard.json"
#: ``KV_GUARD`` value that reports instead of failing.
WARN = "warn"
#: What the servers print: SGLang's scheduler line, its allocator's per-rank line, vLLM's.
POOL = re.compile(
    r"max_total_num_tokens=(\d+)|KV Cache is allocated\. .*?#tokens: (\d+)|GPU KV cache size: ([\d,]+) tokens"
)
POLL_SECONDS = 30.0


@dataclasses.dataclass(frozen=True, slots=True)
class Verdict:
    ok: bool
    pool: int | None
    floor: int | None
    message: str


def pool_tokens(lines: Iterable[str]) -> int | None:
    """The smallest KV pool the server lines state (every pipeline and rank must hold the floor); None when none does."""
    found = [
        int(next(group for group in match.groups() if group).replace(",", ""))
        for line in lines
        if (match := POOL.search(line))
    ]
    return min(found) if found else None


def floor_tokens(env: Mapping[str, str]) -> int | None:
    """``AGENTS_PER_PIPELINE`` x ``KV_TOKENS_PER_AGENT``; None when the model sets no per-agent figure."""
    per_agent = int(env.get("KV_TOKENS_PER_AGENT") or 0)
    return per_agent * int(env.get("AGENTS_PER_PIPELINE") or 1) if per_agent else None


def verdict(pool: int | None, floor: int | None) -> Verdict:
    if floor is None:
        return Verdict(True, pool, None, "no KV_TOKENS_PER_AGENT: no guard")
    if pool is None:
        return Verdict(False, None, floor, "the server logs state no KV pool yet")
    if pool < floor:
        return Verdict(
            False,
            pool,
            floor,
            f"KV pool {pool} tokens is below the floor {floor} (AGENTS_PER_PIPELINE x KV_TOKENS_PER_AGENT): conversations are evicted and every turn re-prefills; raise --mem-fraction-static",
        )
    return Verdict(True, pool, floor, f"KV pool {pool} tokens holds the floor {floor}")


def read_pool(logs: Iterable[pathlib.Path]) -> int | None:
    """:func:`pool_tokens` of the log files that exist."""

    def lines() -> Iterable[str]:
        for log in logs:
            if log.is_file():
                with log.open(encoding="utf-8", errors="replace") as handle:
                    yield from handle

    return pool_tokens(lines())


def check(env: Mapping[str, str], logs: Iterable[pathlib.Path], run_dir: pathlib.Path) -> Verdict:
    """:func:`verdict` of the pool in ``logs``, written to :data:`REPORT` in ``run_dir``."""
    result = verdict(read_pool(logs), floor_tokens(env))
    (run_dir / REPORT).write_text(json.dumps(dataclasses.asdict(result), indent=1) + "\n", encoding="utf-8")
    return result


def job_stdout(job: str) -> pathlib.Path | None:
    """The file Slurm writes the batch step's output to (the experiment's server steps print there too)."""
    shown = subprocess.run(["scontrol", "show", "job", "-o", job], capture_output=True, text=True, check=False).stdout
    found = re.search(r"StdOut=(\S+)", shown)
    return pathlib.Path(found[1]) if found else None


def watch(env: Mapping[str, str], run_dir: pathlib.Path, job: str, timeout: float) -> int:
    """Poll the server logs until a pool is stated, then judge it; a pool below the floor cancels the job unless
    ``KV_GUARD=warn``. 0 when it holds, was only reported, or none was stated within ``timeout``."""
    if floor_tokens(env) is None:
        return 0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        stdout = job_stdout(job)
        logs = [*sorted(run_dir.glob("server-*.log")), *([stdout] if stdout else [])]
        if read_pool(logs) is not None:
            result = check(env, logs, run_dir)
            print(f"kv-guard: {result.message}", flush=True)
            if result.ok or env.get("KV_GUARD") == WARN:
                return 0
            print(f"FAIL kv-guard: cancelling job {job}", file=sys.stderr, flush=True)
            subprocess.run(["scancel", job], check=False)
            return 1
        time.sleep(POLL_SECONDS)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="serve_guard", description=__doc__.split("\n\n")[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("watch", "check"):
        command = sub.add_parser(name)
        command.add_argument("--env", type=pathlib.Path, required=True)
        command.add_argument("--run-dir", type=pathlib.Path, required=True)
        if name == "watch":
            command.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    env = env_spec.assignments(args.env) | {key: value for key, value in os.environ.items() if key == "KV_GUARD"}
    if args.command == "watch":
        return watch(env, args.run_dir, args.job, float(env.get("AGENT_READY_TIMEOUT_SECONDS") or 7200))
    result = check(env, sorted(args.run_dir.glob("server-*.log")), args.run_dir)
    print(f"kv-guard: {result.message}")
    return 0 if result.ok or env.get("KV_GUARD") == WARN else 1


if __name__ == "__main__":
    raise SystemExit(main())
