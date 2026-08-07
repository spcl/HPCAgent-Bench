#!/usr/bin/env python3
"""Summarize node_monitor.sh CSVs: per-node cpu/gpu/mem stats plus balance verdicts.

Usage: monitor_report.py <monitor_dir_or_run_dir>
Reads *.csv recursively (files are named "<role>-<hostname>.csv") and prints a
plain-text table and a verdicts section. Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

ROLES = ("vllm", "judge", "agent")

# thresholds for the verdict rules; picked to flag clear imbalance, not noise
IDLE_CPU_PCT = 20.0
SATURATED_CPU_PCT = 80.0
LOW_GPU_PCT = 40.0
HIGH_CPU_P95_PCT = 90.0

COLUMNS = ("ts", "cpu_pct", "load1", "mem_used_mib", "mem_total_mib", "gpu_pct", "vram_used_mib", "vram_total_mib")


def percentile(values: list[float], p: float) -> float | None:
    # linear interpolation between closest ranks, same convention as numpy's default
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def parse_role_host(path: Path) -> tuple[str, str]:
    stem = path.stem
    for role in ROLES:
        prefix = role + "-"
        if stem.startswith(prefix):
            return role, stem[len(prefix):]
    return "unknown", stem


def read_columns(path: Path) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {name: [] for name in COLUMNS if name != "ts"}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for name in values:
                raw = row.get(name, "")
                if raw == "" or raw is None:
                    continue
                try:
                    values[name].append(float(raw))
                except ValueError:
                    continue
    return values


@dataclass
class NodeStats:
    role: str
    host: str
    samples: int
    cpu_mean: float | None
    cpu_p95: float | None
    gpu_mean: float | None
    gpu_p95: float | None
    mem_peak_mib: float | None
    vram_peak_mib: float | None


def compute_node_stats(path: Path) -> NodeStats:
    role, host = parse_role_host(path)
    cols = read_columns(path)
    cpu = cols["cpu_pct"]
    gpu = cols["gpu_pct"]
    mem = cols["mem_used_mib"]
    vram = cols["vram_used_mib"]
    return NodeStats(
        role=role,
        host=host,
        samples=len(cpu),
        cpu_mean=statistics.mean(cpu) if cpu else None,
        cpu_p95=percentile(cpu, 95),
        gpu_mean=statistics.mean(gpu) if gpu else None,
        gpu_p95=percentile(gpu, 95),
        mem_peak_mib=max(mem) if mem else None,
        vram_peak_mib=max(vram) if vram else None,
    )


def fmt(value: float | None, unit: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:.1f}{unit}"


def print_table(nodes: list[NodeStats]) -> None:
    header = (f"{'role':<7} {'host':<20} {'n':>5} {'cpu_mean':>9} {'cpu_p95':>8} "
              f"{'gpu_mean':>9} {'gpu_p95':>8} {'mem_peak':>10} {'vram_peak':>10}")
    print(header)
    print("-" * len(header))
    for node in sorted(nodes, key=lambda n: (n.role, n.host)):
        print(f"{node.role:<7} {node.host:<20} {node.samples:>5} "
              f"{fmt(node.cpu_mean, '%'):>9} {fmt(node.cpu_p95, '%'):>8} "
              f"{fmt(node.gpu_mean, '%'):>9} {fmt(node.gpu_p95, '%'):>8} "
              f"{fmt(node.mem_peak_mib, 'M'):>10} {fmt(node.vram_peak_mib, 'M'):>10}")


def print_role_summary(groups: dict[str, list[NodeStats]]) -> None:
    print()
    print(f"{'role':<7} {'nodes':>5} {'cpu_mean':>9} {'gpu_mean':>9}")
    for role in ROLES:
        nodes = groups.get(role, [])
        if not nodes:
            continue
        cpu_vals = [n.cpu_mean for n in nodes if n.cpu_mean is not None]
        gpu_vals = [n.gpu_mean for n in nodes if n.gpu_mean is not None]
        cpu_mean = statistics.mean(cpu_vals) if cpu_vals else None
        gpu_mean = statistics.mean(gpu_vals) if gpu_vals else None
        print(f"{role:<7} {len(nodes):>5} {fmt(cpu_mean, '%'):>9} {fmt(gpu_mean, '%'):>9}")


def print_verdicts(groups: dict[str, list[NodeStats]]) -> None:
    print()
    print("verdicts:")
    judge_nodes = groups.get("judge", [])
    agent_nodes = groups.get("agent", [])
    vllm_nodes = groups.get("vllm", [])
    said_something = False

    if judge_nodes and agent_nodes:
        judge_cpu = [n.cpu_mean for n in judge_nodes if n.cpu_mean is not None]
        agent_cpu = [n.cpu_mean for n in agent_nodes if n.cpu_mean is not None]
        if judge_cpu and agent_cpu:
            judge_mean = statistics.mean(judge_cpu)
            agent_mean = statistics.mean(agent_cpu)
            if judge_mean < IDLE_CPU_PCT and agent_mean > SATURATED_CPU_PCT:
                print("  judge nodes idle while agent node saturated -> raise AGENTS_PER_JUDGE")
                said_something = True

    if vllm_nodes:
        gpu_means = [n.gpu_mean for n in vllm_nodes]
        if all(m is not None and m < LOW_GPU_PCT for m in gpu_means):
            print("  inference gpu mean < 40% on all replicas -> drop to 1 inference node")
            said_something = True

    for node in agent_nodes:
        if node.cpu_p95 is not None and node.cpu_p95 > HIGH_CPU_P95_PCT:
            print(f"  agent node cpu > 90% p95 -> lower AGENTS_PER_NODE ({node.host})")
            said_something = True

    if not said_something:
        print("  none")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("monitor_dir", type=Path, help="monitor dir or run dir to scan")
    args = parser.parse_args()

    csv_paths = sorted(args.monitor_dir.rglob("*.csv"))
    if not csv_paths:
        print(f"no CSV files found under {args.monitor_dir}")
        return

    nodes = [compute_node_stats(path) for path in csv_paths]
    groups: dict[str, list[NodeStats]] = {}
    for node in nodes:
        groups.setdefault(node.role, []).append(node)

    print_table(nodes)
    print_role_summary(groups)
    print_verdicts(groups)


if __name__ == "__main__":
    main()
