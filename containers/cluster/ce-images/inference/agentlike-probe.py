"""Sustained agent-shaped load against a served vLLM endpoint.

The saturation probe in smoke-kimi-eager-pg.sbatch sends "Explain loop tiling, variation N." --
about 8 tokens -- and measures one burst. Every kimi figure we have came from it, which is why
601653 read 294-307 tok/s while the campaign decoded at 0.2: that probe did almost no prefill
(24 of its 28 samples showed ZERO prompt throughput) and this workload is prefill-bound at
roughly 25 tokens in per token out.

This sends what an agent turn actually looks like -- a long prefix shared by every stream, as the
skills packet and system prompt are, plus a unique tail -- and keeps sending for a set duration,
because the EngineCore death we are chasing arrives at ~90 minutes, not inside a 6-minute burst.
"""

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

STOP = threading.Event()
LOCK = threading.Lock()
STATS = {"ok": 0, "err": 0, "out_tokens": 0, "first_err": ""}


def build_prompt(shared_words: int, stream: int) -> str:
    # One prefix for every stream, so the prefix cache behaves as it does under real agents.
    shared = "loop tiling and unrolling analysis for a stencil kernel on a cache hierarchy. " * shared_words
    return f"{shared}\nStream {stream} unique tail. Summarise the tradeoffs."


def worker(base: str, model: str, prompt: str, max_tokens: int, timeout: int) -> None:
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "ignore_eos": True}).encode()
    while not STOP.is_set():
        req = urllib.request.Request(f"{base}/v1/completions", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.load(resp)
            n = payload.get("usage", {}).get("completion_tokens", 0)
            with LOCK:
                STATS["ok"] += 1
                STATS["out_tokens"] += n
        except Exception as exc:  # noqa: BLE001 -- any failure is a datapoint, not a crash
            with LOCK:
                STATS["err"] += 1
                if not STATS["first_err"]:
                    STATS["first_err"] = f"{type(exc).__name__}: {exc}"


def metric(base: str, name: str) -> float:
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=10) as resp:
            for line in resp.read().decode().splitlines():
                if line.startswith(name) and not line.startswith("#"):
                    return float(line.rsplit(" ", 1)[1])
    except Exception:  # noqa: BLE001
        pass
    return -1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--streams", type=int, default=64)
    ap.add_argument("--shared-words", type=int, default=2000)  # ~25k tokens at ~12 tok/repeat
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--duration", type=int, default=7200)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    prompts = [build_prompt(args.shared_words, i) for i in range(args.streams)]
    approx_tokens = len(prompts[0].split()) * 4 // 3
    print(
        f"streams={args.streams} approx_prompt_tokens={approx_tokens} "
        f"max_tokens={args.max_tokens} duration={args.duration}s",
        flush=True,
    )

    threads = [
        threading.Thread(
            target=worker, args=(args.base, args.model, prompts[i], args.max_tokens, args.timeout), daemon=True
        )
        for i in range(args.streams)
    ]
    for t in threads:
        t.start()

    start = time.time()
    while time.time() - start < args.duration:
        time.sleep(60)
        with LOCK:
            ok, err, toks, first = STATS["ok"], STATS["err"], STATS["out_tokens"], STATS["first_err"]
        elapsed = time.time() - start
        print(
            f"t={elapsed:6.0f}s ok={ok:<6} err={err:<5} out_tok={toks:<9} "
            f"decode={toks / elapsed:6.1f} tok/s running={metric(args.base, 'vllm:num_requests_running'):.0f} "
            f"waiting={metric(args.base, 'vllm:num_requests_waiting'):.0f}" + (f" first_err={first}" if first else ""),
            flush=True,
        )
        if err and ok == 0:
            print("FAIL: every request errored", flush=True)
            STOP.set()
            return 1
    STOP.set()
    with LOCK:
        print(
            f"FINAL ok={STATS['ok']} err={STATS['err']} out_tokens={STATS['out_tokens']} "
            f"mean_decode={STATS['out_tokens'] / max(1, time.time() - start):.1f} tok/s",
            flush=True,
        )
        return 1 if STATS["ok"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
