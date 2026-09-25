#!/usr/bin/env python3
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Launch the kernels an SGLang serve reaches on the visible GPU and compare them with torch.

verify_image.py only imports, and importing sgl_kernel loads common_ops.so without launching a
kernel: an image whose device code targets another gfx arch passes it and fails at the first forward.
Run inside the image on a GPU node:

    python3 sglang_kernel_launch_check.py

Checks sgl_kernel.silu_and_mul and, when importable, the triton causal_conv1d_fn the GDN backend uses,
each at bf16 and fp32 against an fp32 torch reference. Exit status is the number of failed checks;
every failure names its op.
"""

import dataclasses
import functools
import sys
from collections.abc import Callable

import torch

#: Qwen3.8-27B MLP half-width at tp4: intermediate_size 17408 / 4.
MLP_HALF_WIDTH = 4352
#: Conv shape crossing the kernel's BLOCK_N=256 channels and BLOCK_M=8 tokens; GDN kernel width 4.
CONV_DIM, CONV_TOKENS, CONV_WIDTH = 320, 67, 4
DTYPES: tuple[torch.dtype, ...] = (torch.bfloat16, torch.float32)
#: (rtol, atol) against the fp32 reference. bf16 rounds every intermediate product.
TOLERANCE: dict[torch.dtype, tuple[float, float]] = {torch.bfloat16: (1e-2, 1e-2), torch.float32: (1e-5, 1e-5)}

ConvFn = Callable[..., torch.Tensor]


@dataclasses.dataclass(frozen=True, slots=True)
class Outcome:
    op: str
    ok: bool
    detail: str


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def device_outcome() -> Outcome:
    if not torch.cuda.is_available():
        return Outcome("device", False, "torch sees no GPU")
    props = torch.cuda.get_device_properties(0)
    return Outcome("device", True, f"{props.name} gcnArchName={props.gcnArchName} torch={torch.__version__}")


def compare(op: str, got: torch.Tensor, want: torch.Tensor, dtype: torch.dtype) -> Outcome:
    rtol, atol = TOLERANCE[dtype]
    got32 = got.float()
    if not bool(torch.isfinite(got32).all()):
        return Outcome(op, False, "non-finite output")
    diff = (got32 - want).abs()
    ok = bool((diff <= atol + rtol * want.abs()).all())
    return Outcome(op, ok, f"max|d|={float(diff.max()):.3g} rtol={rtol} atol={atol}")


def guarded(op: str, check: Callable[[], Outcome]) -> Outcome:
    try:
        return check()
    except Exception as exc:  # noqa: BLE001 -- a launch that raises is the failure this looks for
        return Outcome(op, False, f"{type(exc).__name__}: {str(exc)[:200]}")


def silu_and_mul_outcome(op: str, dtype: torch.dtype) -> Outcome:
    # Imported here so a missing or unloadable common_ops is reported as this op failing.
    from sgl_kernel import silu_and_mul

    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(5, 2 * MLP_HALF_WIDTH, device="cuda", dtype=dtype, generator=gen)
    x32 = x.float()
    want = torch.nn.functional.silu(x32[..., :MLP_HALF_WIDTH]) * x32[..., MLP_HALF_WIDTH:]
    got = silu_and_mul(x)
    torch.cuda.synchronize()
    return compare(op, got, want, dtype)


def load_causal_conv1d() -> ConvFn | None:
    try:
        from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
    except ImportError as exc:
        print(f"--   causal_conv1d_fn not importable, not launched: {exc}", flush=True)
        return None
    return causal_conv1d_fn


def causal_conv1d_outcome(op: str, conv: ConvFn, dtype: torch.dtype) -> Outcome:
    gen = torch.Generator(device="cuda").manual_seed(1)
    # (tokens, dim) transposed to (dim, tokens): the channel-last layout the GDN backend passes.
    x = torch.randn(CONV_TOKENS, CONV_DIM, device="cuda", dtype=dtype, generator=gen).t()
    weight = 0.5 * torch.randn(CONV_DIM, CONV_WIDTH, device="cuda", dtype=dtype, generator=gen)
    bias = 0.1 * torch.randn(CONV_DIM, device="cuda", dtype=dtype, generator=gen)
    states = torch.zeros(2, CONV_DIM, CONV_WIDTH - 1, device="cuda", dtype=dtype)
    got = conv(
        x,
        weight,
        bias,
        states,
        torch.tensor([0, CONV_TOKENS], device="cuda", dtype=torch.int32),
        [CONV_TOKENS],
        cache_indices=torch.tensor([0], device="cuda", dtype=torch.int32),
        has_initial_state=torch.tensor([False], device="cuda"),
        activation="silu",
    )
    torch.cuda.synchronize()
    # Zero initial state: the causal conv is the left part of a (width-1)-padded grouped conv1d.
    padded = torch.nn.functional.conv1d(
        x.float().unsqueeze(0), weight.float().unsqueeze(1), bias.float(), padding=CONV_WIDTH - 1, groups=CONV_DIM
    )
    want = torch.nn.functional.silu(padded[0, :, :CONV_TOKENS])
    return compare(op, got, want, dtype)


def launch_outcomes() -> list[Outcome]:
    outcomes: list[Outcome] = []
    for dtype in DTYPES:
        op = f"sgl_kernel.silu_and_mul[{dtype_name(dtype)}]"
        outcomes.append(guarded(op, functools.partial(silu_and_mul_outcome, op, dtype)))
    conv = load_causal_conv1d()
    if conv is None:
        return outcomes
    for dtype in DTYPES:
        op = f"triton.causal_conv1d_fn[{dtype_name(dtype)}]"
        outcomes.append(guarded(op, functools.partial(causal_conv1d_outcome, op, conv, dtype)))
    return outcomes


def main() -> int:
    device = device_outcome()
    outcomes = [device, *launch_outcomes()] if device.ok else [device]
    for outcome in outcomes:
        print(f"{'PASS' if outcome.ok else 'FAIL'} {outcome.op}: {outcome.detail}", flush=True)
    failed = [outcome.op for outcome in outcomes if not outcome.ok]
    if failed:
        print(f"FAILED: {', '.join(failed)}", file=sys.stderr)
    return len(failed)


if __name__ == "__main__":
    raise SystemExit(main())
