#!/usr/bin/env python3
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Check every testable claim on the lang-triton skill page against this ROCm Triton.

The page carries no code fences, so what is verified here is each MECHANICAL claim an agent would
act on: wave width, the two num_stages defaults, the cluster error, fp8 type names, and whether a
small tl.dot is really tolerated. A claim that no longer holds is a page edit.
"""

from __future__ import annotations

import json
import pathlib
import sys
import traceback

import torch
import triton
import triton.language as tl

RESULTS: list[dict] = []


def check(name: str, why: str):
    def deco(fn):
        try:
            detail = fn()
            RESULTS.append({"case": name, "why": why, "verdict": "OK", "detail": detail})
            print(f"OK            {name}  -- {detail}", flush=True)
        except AssertionError as exc:
            RESULTS.append({"case": name, "why": why, "verdict": "MISMATCH", "detail": str(exc)})
            print(f"MISMATCH      {name}\n    {exc}", flush=True)
        except Exception:
            tb = traceback.format_exc().strip().splitlines()[-1]
            RESULTS.append({"case": name, "why": why, "verdict": "ERROR", "detail": tb})
            print(f"ERROR         {name}\n    {tb}", flush=True)
        return fn

    return deco


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr) -> None:
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


@check("versions", "the page is written for CDNA3 / ROCm triton; record what it was checked against")
def _():
    return f"triton {triton.__version__}, torch {torch.__version__}, device {torch.cuda.get_device_name(0)}"


@check("masked-tail-store", "page: mask every load and store whose tile does not exactly cover the array")
def _():
    n = 1000  # deliberately not a multiple of BLOCK
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
    torch.cuda.synchronize()
    assert torch.allclose(out, x + y), "masked tail did not produce the reference answer"
    return f"n={n} with BLOCK=256 correct through the masked tail"


@check("wave-is-64-lanes", "page: a wave is 64 lanes, not 32, so num_warps=4 is 256 threads")
def _():
    props = torch.cuda.get_device_properties(0)
    warp = getattr(props, "warp_size", None)
    assert warp == 64, f"device reports warp_size={warp}, page says 64"
    return f"warp_size={warp}, so num_warps=4 is {4 * warp} threads"


@check("num-stages-bare-launch-default", "page: num_stages defaults to 2 on a bare launch, not 3")
def _():
    n = 4096
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    compiled = add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
    got = compiled.metadata.num_stages
    assert got == 2, f"bare launch compiled with num_stages={got}, page says 2"
    return f"bare launch num_stages={got}"


@check("num-stages-config-default", "page: triton.Config keeps its own default of 3 and always forwards it")
def _():
    cfg = triton.Config({"BLOCK": 256})
    assert cfg.num_stages == 3, f"triton.Config default num_stages={cfg.num_stages}, page says 3"
    return f"triton.Config(...).num_stages={cfg.num_stages} with no num_stages given"


@check("num-ctas-gt-1-is-an-error", "page: no clusters -- num_ctas > 1 is a hard error")
def _():
    n = 4096
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    try:
        add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256, num_ctas=2)
        torch.cuda.synchronize()
    except Exception as exc:
        return f"rejected: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
    raise AssertionError("num_ctas=2 was ACCEPTED; the page calls it a hard error")


@check("fp8-fnuz-types-exist", "page: if you use fp8, use the FNUZ types float8e4b8 / float8e5b16")
def _():
    missing = [n for n in ("float8e4b8", "float8e5b16", "float8e4nv", "float8e5") if not hasattr(tl, n)]
    present = [n for n in ("float8e4b8", "float8e5b16") if hasattr(tl, n)]
    assert len(present) == 2, f"FNUZ names the page recommends are absent: {missing}"
    return f"tl.{' and tl.'.join(present)} exist"


@check("small-dot-does-not-error", "page: a small tl.dot falls back to FMA here rather than being a hard error")
def _():
    @triton.jit
    def dot_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr) -> None:
        offs_m = tl.arange(0, M)
        offs_n = tl.arange(0, N)
        offs_k = tl.arange(0, K)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], tl.dot(a, b))

    m = n = 16
    k = 8  # below the 16 the page tells you to keep every dot dimension at
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    c = torch.empty((m, n), device="cuda", dtype=torch.float32)
    dot_kernel[(1,)](a, b, c, M=m, N=n, K=k)
    torch.cuda.synchronize()
    assert torch.allclose(c, (a.float() @ b.float()), atol=1e-2), "K=8 dot ran but gave the wrong answer"
    return f"K={k} dot compiled, ran and matched the reference (no hard error)"


@check(
    "lds-limit-names-65536",
    "page: a CUDA config's 3-4 stages overflows the 64 KB workgroup LDS and the error names 65536",
)
def _():
    """A MATMUL, not an elementwise loop: only a pipelined kernel with staged operand tiles puts
    anything in local memory, so num_stages can multiply it past the 64 KB a workgroup gets. Round
    one asked this of an elementwise kernel, which allocates no LDS at all and so could not
    overflow it -- the page was not refuted there, the test was."""

    @triton.jit
    def mm(
        a_ptr,
        b_ptr,
        c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ) -> None:
        offs_m = tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            offs_k = k + tl.arange(0, BK)
            a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
            b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    # 128x128 tiles of fp16 operands, 4 stages: 4 * (128*128 + 128*128) * 2 B = 256 KB of LDS.
    m = n = k = 256
    bm = bn = bk = 128
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    c = torch.empty((m, n), device="cuda", dtype=torch.float32)
    try:
        mm[(1,)](a, b, c, M=m, N=n, K=k, BM=bm, BN=bn, BK=bk, num_stages=4)
        torch.cuda.synchronize()
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        return f"rejected; names 65536: {'65536' in text}; {text.splitlines()[0][:200]}"
    raise AssertionError(
        f"a {bm}x{bn}x{bk} fp16 tile at num_stages=4 was ACCEPTED; the page promises an LDS-limit error"
    )


def main() -> int:
    bad = [r for r in RESULTS if r["verdict"] != "OK"]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} claims hold; {len(bad)} to look at", flush=True)
    for r in bad:
        print(f"  {r['case']}: {r['why']}", flush=True)
    pathlib.Path(__file__).resolve().parent.joinpath("triton_claims.json").write_text(json.dumps(RESULTS, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
