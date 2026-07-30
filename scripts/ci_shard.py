#!/usr/bin/env python3
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Split test files across the CI runners we already have, balanced by COST rather than count.

`ls | awk 'NR % 3 == shard'` deals files round-robin, which balances the number of files and
nothing else. Cost here spans orders of magnitude -- a level-3 kernel test compiles and runs a
fused conv stack, a level-1 one is a loop -- so the shard that draws the heavy files decides the
job's wall clock while its siblings idle.

Weights come from the kernel level a test file exercises: level 1 = 1, level 2 = 3, level 3 = 12.
Deliberately a guess, not a measurement -- the ratios only have to be right enough to keep the
heavy files apart, and a wrong guess costs balance, never correctness. A file naming no level
falls back to its collected test count, and one we cannot size at all gets the median weight so it
is neither hoarded nor ignored.

LPT: sort descending by weight, give each file to the least-loaded shard. Deterministic, so every
runner computes the identical split without coordinating.

    python scripts/ci_shard.py --shard 1/3 --files "$(ls tests/test_*.py)"
"""
import argparse
import pathlib
import re
import sys
from typing import Dict, List, Sequence, Tuple

#: Cost of one kernel at each KernelBench level, in arbitrary units. A level-3 kernel is a fused
#: multi-op stack; a level-1 kernel is a single op. Ratios, not seconds.
LEVEL_WEIGHT: Dict[int, float] = {1: 1.0, 2: 3.0, 3: 12.0}

#: What a file with no level marker and no collected count is worth. The median of the weights we
#: do know, so an unsizable file cannot dominate a shard or vanish from one.
DEFAULT_WEIGHT = 3.0

LEVEL_RE = re.compile(r"level(\d)")


def weight_of(path: pathlib.Path, counts: Dict[str, int]) -> float:
    """Cost estimate for one test file: its levels if it names any, else its collected test count."""
    name = str(path)
    if name in counts and counts[name] > 0:
        base = float(counts[name])
    else:
        base = DEFAULT_WEIGHT
    try:
        text = path.read_text()
    except OSError:
        return base
    levels = [int(m) for m in LEVEL_RE.findall(text) if int(m) in LEVEL_WEIGHT]
    if not levels:
        return base
    # A file that drives kernels is priced by the levels it names, scaled by how much of it does
    # so -- a passing mention of "level3" in a comment should not price the whole file as heavy.
    return base * (sum(LEVEL_WEIGHT[lv] for lv in levels) / len(levels))


def pack(files: Sequence[pathlib.Path], shards: int, counts: Dict[str, int]) -> List[List[pathlib.Path]]:
    """LPT bin-pack: heaviest first, each to the least-loaded shard. Pure function of its inputs."""
    weighted: List[Tuple[float, pathlib.Path]] = sorted(((weight_of(f, counts), f) for f in files),
                                                        key=lambda wf: (-wf[0], str(wf[1])))
    load = [0.0] * shards
    out: List[List[pathlib.Path]] = [[] for _ in range(shards)]
    for weight, path in weighted:
        target = min(range(shards), key=lambda i: (load[i], i))
        out[target].append(path)
        load[target] += weight
    return out


def read_counts(path: str) -> Dict[str, int]:
    """`<count> <file>` lines, as `pytest --collect-only` output can be reduced to. Missing is fine."""
    counts: Dict[str, int] = {}
    if not path:
        return counts
    p = pathlib.Path(path)
    if not p.is_file():
        return counts
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            counts[parts[1]] = int(parts[0])
    return counts


def main(argv: Sequence[str] = ()) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", required=True, help="i/N -- this runner's index and the total")
    ap.add_argument("--files", required=True, help="whitespace-separated file list (from ls)")
    ap.add_argument("--counts", default="", help="optional '<count> <file>' weight table")
    args = ap.parse_args(list(argv) or None)

    index_s, _, total_s = args.shard.partition("/")
    index, total = int(index_s), int(total_s)
    if not 0 < total or not 0 <= index % total < total:
        raise SystemExit(f"bad --shard {args.shard!r}: expected i/N with N > 0")
    files = [pathlib.Path(f) for f in args.files.split() if f]
    if not files:
        raise SystemExit("no files given; an empty shard would make pytest collect the whole repo")
    for path in pack(files, total, read_counts(args.counts))[index % total]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
