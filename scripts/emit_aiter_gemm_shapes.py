# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Emit the aiter untuned-GEMM CSV for the shapes our SGLang arms actually execute.

Every aiter GEMM lookup in every kimi arm on record misses: 2,075,376 lookups, zero hits, each one
logging `not found tuned config ... using torch solution:0`. The tuned table ships with aiter and is
keyed on the EXACT (M, N, K), so it only ever hits shapes AMD happened to tune -- and M is the batch
token count, which our workload never repeats.

Only 17 (N, K) pairs exist and 8 carry 99.6% of the traffic, so the N/K side is tiny. M is the
problem: 7,591 distinct values, near-uniform. Tuning all of them is ~39k shapes and not worth a
node-week, so this emits the bounded part:

  * the decode ladder -- M values a captured CUDA graph can actually run, which is a fixed list;
  * a prefill ladder up to chunked_prefill_size, where a miss costs one untuned call per chunk.

Collect from real logs rather than guessing the ladder:

    python3 scripts/emit_aiter_gemm_shapes.py --logs 'results/beverin-services-*.err' \
        --out bf16_untuned_gemm.csv

Then, inside the SGLang image (aiter checkout at /sgl-workspace/aiter):

    python3 csrc/gemm_a16w16/gemm_a16w16_tune.py \
        --input_file bf16_untuned_gemm.csv --tuned_file aiter/configs/bf16_tuned_gemm.csv

and bake the result back to aiter/configs/bf16_tuned_gemm.csv in the image, so nothing JITs or
tunes at request time -- the failure mode that wedged six earlier aiter attempts.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import pathlib
import re
import sys

SHAPE = re.compile(r"shape is M:(\d+), N:(\d+), K:(\d+) dtype='([^']+)' otype='([^']+)' "
                   r"bias=(\w+), scaleAB=(\w+), bpreshuffle=(\w+)")
# aiter's own column order for an untuned list; the tuner appends the solution columns.
COLUMNS = ["cu_num", "M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle"]
TORCH_TO_AITER = {"torch.bfloat16": "bf16", "torch.float16": "fp16", "torch.float32": "fp32"}


def ladder(limit: int) -> list[int]:
    """Powers of two plus the eighths between them: dense where M is small and the relative gap
    between neighbouring sizes is large, sparse where it is not."""
    out = {1, 2, 4, 8}
    step = 8
    while step < limit:
        out.update(range(step, min(step * 2, limit) + 1, max(1, step // 8)))
        step *= 2
    return sorted(m for m in out if m <= limit)


def collect(patterns: list[str]) -> collections.Counter:
    seen: collections.Counter = collections.Counter()
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                with open(path, errors="ignore") as handle:
                    for line in handle:
                        if "not found tuned config" not in line:
                            continue
                        found = SHAPE.search(line)
                        if found:
                            seen[found.groups()] += 1
            except OSError as exc:
                print(f"skipped {path}: {exc}", file=sys.stderr)
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", nargs="+", required=True, help="glob(s) over SGLang stderr logs")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--cu-num", type=int, default=304, help="CUs per device (gfx942 MI300A: 304)")
    parser.add_argument("--max-m", type=int, default=8192, help="chunked_prefill_size: M never exceeds it")
    parser.add_argument("--top-nk", type=int, default=8, help="how many (N, K) pairs to tune")
    args = parser.parse_args()

    seen = collect(args.logs)
    if not seen:
        print("no aiter shape misses found in those logs", file=sys.stderr)
        return 1
    # Rank (N, K) by how much traffic each carries, and keep the head. The tail here is 4 orders of
    # magnitude smaller -- tuning it costs the same per shape and buys nothing measurable.
    traffic: collections.Counter = collections.Counter()
    meta: dict[tuple[str, str], tuple[str, str, str, str, str]] = {}
    for (m, n, k, dtype, otype, bias, scale, preshuffle), count in seen.items():
        traffic[(n, k)] += count
        meta[(n, k)] = (dtype, otype, bias, scale, preshuffle)
    hot = [nk for nk, _ in traffic.most_common(args.top_nk)]

    rows = []
    for n, k in hot:
        dtype, otype, bias, scale, preshuffle = meta[(n, k)]
        for m in ladder(args.max_m):
            rows.append({
                "cu_num": args.cu_num,
                "M": m,
                "N": int(n),
                "K": int(k),
                "bias": bias,
                "dtype": TORCH_TO_AITER.get(dtype, dtype),
                "outdtype": TORCH_TO_AITER.get(otype, otype),
                "scaleAB": scale,
                "bpreshuffle": preshuffle,
            })
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{sum(seen.values())} misses over {len(seen)} distinct shapes, {len(traffic)} (N, K) pairs")
    print(f"tuning the top {len(hot)} pairs x {len(ladder(args.max_m))} M values = {len(rows)} shapes -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
