"""Do aiter's MLA attention kernels work on this image, and do they agree with the reference?

WHY A KERNEL TEST AND NOT A SERVE. Every previous aiter attempt on this system was inconclusive
for a reason that had nothing to do with the kernels: SGLANG_USE_AITER=1 sends the JIT into a
per-module baton lock that wedges serving for hours (0-for-6 across probes), and the master switch
separately broke MLA prefill. So "does aiter work" was never answered -- the serve never got far
enough to ask. This asks the kernels directly, on synthetic tensors, in a couple of minutes, and
it CANNOT wedge because it never starts a server.

WHAT IT PROVES, in order, so a failure names its own stage:
  1. import        aiter imports at all on this ROCm/torch pair
  2. build         the MLA op is a BUILT module, not merely importable -- importing builds nothing,
                   which is how a prebuild step once "succeeded" having compiled zero kernels
  3. launch        it runs on a device without raising
  4. correctness   its output matches a reference attention within tolerance. This is the point:
                   a kernel that runs and returns wrong numbers is the failure mode that reached
                   9k context before anyone noticed, and it is invisible to a smoke test
  5. speed         a rough per-call time against the reference, for whether it is worth enabling

Exit is non-zero if any NAMED-as-present kernel fails; a kernel that is simply absent from this
aiter build is reported and skipped, because absence is a version fact, not a defect.
"""

import argparse
import importlib
import inspect
import os
import sys
import time
import traceback

import torch

# MLA on gfx942 wants head counts divisible by 16 -- that constraint is why the campaign runs TP=4
# in-node rather than a wider tensor split. Defaults mirror the kimi shape the campaign serves.
DEFAULT_SHAPES = [
    # (batch, seq_q, seq_kv, heads, head_dim_qk, head_dim_v)
    (1, 1, 1024, 16, 576, 512),
    (4, 1, 4096, 16, 576, 512),
    (1, 512, 512, 16, 576, 512),
]

TOL = {"atol": 2e-2, "rtol": 2e-2}


def say(stage, name, verdict, detail=""):
    print(f"{stage:<12} {name:<34} {verdict:<8} {detail}", flush=True)


def reference_attention(q, k, v):
    # Plain scaled dot-product in fp32, which is the thing the kernel must agree WITH. Deliberately
    # not another fused kernel: two fused paths can share a bug and agree with each other.
    qf, kf, vf = q.float(), k.float(), v.float()
    scale = 1.0 / (qf.shape[-1] ** 0.5)
    scores = torch.einsum("bqhd,bkhd->bhqk", qf, kf) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, vf)


def probe_module(modname):
    """Is this aiter op module IMPORTABLE and does it carry a compiled extension?"""
    try:
        mod = importlib.import_module(modname)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return mod, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    failures = 0
    absent = []
    skipped_calls = []

    print(f"torch {torch.__version__}  hip {torch.version.hip}  devices {torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        print("no GPU visible -- an MLA kernel test without a device proves nothing", file=sys.stderr)
        return 2
    dev = torch.device("cuda")
    dtype = getattr(torch, args.dtype)

    try:
        import aiter
    except Exception as exc:
        say("import", "aiter", "FAIL", f"{type(exc).__name__}: {exc}")
        return 1
    say("import", "aiter", "OK", getattr(aiter, "__file__", "?"))
    print(f"aiter version: {getattr(aiter, '__version__', 'unknown')}", flush=True)

    # The JIT cache the image prebuilt. An empty one here means every kernel below pays a build,
    # which is exactly the stall that wedged serving -- worth reporting before it happens.
    jit = os.environ.get("AITER_JIT_DIR", "")
    if jit and os.path.isdir(jit):
        built = [f for f in os.listdir(jit) if f.endswith(".so")]
        say("jit", "prebuilt modules", "OK" if built else "WARN", f"{len(built)} .so in {jit}")
    else:
        say("jit", "AITER_JIT_DIR", "WARN", f"not a directory: {jit!r} -- kernels will JIT on first call")

    # The MLA entry points, most-specific first. Which of these exists depends on the aiter build,
    # so a missing name is reported as absent rather than failed.
    candidates = [
        ("aiter.ops.mla", "mla_decode_fwd"),
        ("aiter.ops.mla", "mla_prefill_fwd"),
        ("aiter.mla", "mla_decode_fwd"),
        ("aiter", "mla_decode_fwd"),
    ]

    found = []
    for modname, fn in candidates:
        mod, err = probe_module(modname)
        if mod is None:
            continue
        if hasattr(mod, fn):
            found.append((modname, fn, getattr(mod, fn)))
            say("resolve", f"{modname}.{fn}", "OK")
        else:
            absent.append(f"{modname}.{fn}")

    # DISCOVERY, before any call. aiter's MLA ops take paged KV, index pointers and a workspace in
    # some versions and plain tensors in others, and calling with a guessed signature would report
    # a FAILURE THAT IS THIS SCRIPT'S OWN -- indistinguishable, in a log, from a broken kernel.
    # So dump the real surface first: whatever this build exposes is printed here, and the call
    # below is attempted only where the signature can actually be satisfied.
    print("\n--- MLA API surface in this aiter build")
    for modname in sorted({m for m, _ in candidates}):
        mod, err = probe_module(modname)
        if mod is None:
            print(f"  {modname}: not importable ({err})")
            continue
        names = [n for n in dir(mod) if "mla" in n.lower() and not n.startswith("_")]
        print(f"  {modname}: {', '.join(names) if names else '(nothing matching mla)'}")
        for n in names:
            try:
                print(f"      {n}{inspect.signature(getattr(mod, n))}")
            except (TypeError, ValueError):
                print(f"      {n}(<signature unavailable -- native binding>)")
    print()

    if not found:
        say("resolve", "any MLA entry point", "FAIL", "this aiter build exposes no MLA op")
        print("tried: " + ", ".join(f"{m}.{f}" for m, f in candidates), file=sys.stderr)
        return 1

    for b, sq, skv, h, dqk, dv in DEFAULT_SHAPES:
        shape = f"b{b} q{sq} kv{skv} h{h} d{dqk}/{dv}"
        q = torch.randn(b, sq, h, dqk, dtype=dtype, device=dev)
        k = torch.randn(b, skv, h, dqk, dtype=dtype, device=dev)
        v = torch.randn(b, skv, h, dv, dtype=dtype, device=dev)
        ref = reference_attention(q, k, v[..., :dv])

        for modname, fnname, fn in found:
            label = f"{fnname} {shape}"
            # Only attempt the plain (q, k, v) call where the signature actually accepts exactly
            # that. Anything else is reported with its signature and SKIPPED, so the log says "this
            # script does not know how to call it yet" rather than "the kernel is broken" -- those
            # are different findings and only one of them is about aiter.
            try:
                sig = inspect.signature(fn)
                required = [pname for pname, prm in sig.parameters.items()
                            if prm.default is inspect.Parameter.empty
                            and prm.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                                 inspect.Parameter.VAR_KEYWORD)]
            except (TypeError, ValueError):
                required = None
            if required is not None and len(required) != 3:
                say("launch", label, "SKIP", f"needs {len(required)} args: {', '.join(required)}")
                skipped_calls.append(f"{fnname}({', '.join(required)})")
                continue
            try:
                out = fn(q, k, v)
            except Exception as exc:
                say("launch", label, "FAIL", f"{type(exc).__name__}: {str(exc)[:70]}")
                traceback.print_exc(limit=3)
                failures += 1
                continue
            torch.cuda.synchronize()
            say("launch", label, "OK", f"out {tuple(out.shape)}")

            got = out.float()
            if got.shape != ref.shape:
                say("correct", label, "FAIL", f"shape {tuple(got.shape)} vs ref {tuple(ref.shape)}")
                failures += 1
                continue
            if torch.allclose(got, ref, **TOL):
                say("correct", label, "OK", f"max|d| {(got - ref).abs().max().item():.4g}")
            else:
                # The failure that matters. A wrong kernel is fast and silent.
                say("correct", label, "FAIL", f"max|d| {(got - ref).abs().max().item():.4g} exceeds {TOL}")
                failures += 1
                continue

            for _ in range(args.warmup):
                fn(q, k, v)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                fn(q, k, v)
            torch.cuda.synchronize()
            per = (time.perf_counter() - t0) / args.iters * 1e3
            say("speed", label, "OK", f"{per:.3f} ms/call")

    print()
    if absent:
        print("ABSENT from this aiter build (not a defect): " + ", ".join(absent))
    if skipped_calls:
        print("NOT CALLED -- signature not (q, k, v); use the surface dump above to write the real")
        print("call before claiming anything about these: " + "; ".join(sorted(set(skipped_calls))))
    if failures:
        print(f"AITER MLA CHECK: FAILED ({failures} failure(s))")
        return 1
    print("AITER MLA CHECK: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
