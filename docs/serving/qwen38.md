# Serving Qwen3.8 on MI300A

`Qwen/Qwen3.8-27B-FP8`. One node, four GPUs. The cheapest useful endpoint on this machine and the
one to start from.

Authoritative source: `experiments/.env.base-qwen38`. If this page and that file disagree, the file
is right. Cross-model background is in [`knobs.md`](knobs.md).

---

## Configuration

| | |
|---|---|
| Engine | SGLang |
| EDF | `sglang-latest` |
| Nodes | **1**, `tp=4`, no pipeline stage |
| Port | 8000, served name `optarena-vllm` |
| KV pool | **3,318,498 tokens** |
| Mamba slots | **704** |
| Host memory left free | about 77 GB of 513 GB |

```
--tp-size 4 --host 0.0.0.0 --port 8000
--attention-backend aiter
--chat-template experiments/chat-template-qwen38.jinja
--trust-remote-code
--language-only
--watchdog-timeout 1800
--context-length 262144
--mem-fraction-static 0.306
--mamba-full-memory-ratio 0.5
--max-running-requests 128
--enable-metrics
--reasoning-parser qwen3
--tool-call-parser qwen3_coder
--enable-cache-report
```

Environment: `SGLANG_USE_AITER=1`, `SGLANG_SET_CPU_AFFINITY=0`.

```bash
cd experiments
./serve-only.sbatch          # qwen38 is the default MODEL
```

---

## The one mechanism behind every memory knob here

The KV pool must **exceed the working set**, where the working set is the number of concurrent
conversations times the largest prompt each of them sends. Crossing that line is worth about **8x**;
moving around inside either regime is worth nothing. On this model the crossing sits near a pool to
working-set ratio of **1.0**.

At 40 concurrent conversations, the load this endpoint is sized for:

| Prompt each | Working set | Pool / working set | Prefix-cache hit | Aggregate |
|---|---|---|---|---|
| 33k tokens | 1.93 M | 1.72 | 0.974 | 319 tok/s |
| 60k tokens | 3.18 M | **1.04** | 0.984 | 254 tok/s |

Both are above the line, which is the point of the configuration. Serve more conversations, or
longer ones, and you fall off it: **measure your own working set and keep the pool above it.**

---

## DO

- **Serve this on SGLang, not vLLM.** Roughly **19x** the throughput for this model. There is no
  configuration of vLLM here that recovers it.
- **Read the two allocation lines out of the server log before believing any tuning.**
  ```bash
  grep -a "Mamba Cache is allocated\|max_total_num_tokens" server-0.log
  ```
  They report the mamba slot count and the KV pool, the two numbers every knob on this page moves.
  **Read the slot count, never infer it from the ratio** -- the same ratio gives a different count at
  a different `--mem-fraction-static`.
- **Keep the KV pool above the working set.** Concurrent conversations times their largest prompt.
  That is the whole tuning problem; everything else on this page is a detail.
- **Move `--mem-fraction-static` and `--mamba-full-memory-ratio` as a pair.** They divide **one**
  budget: with ratio `r` the mamba state cache gets `R*r/(1+r)` and the KV cache `R/(1+r)`. Every
  byte one gets, the other does not, so neither value means anything alone. Changing only the
  fraction leaves 1002 slots and too small a pool; changing only the ratio starves the state cache.
- **Move `--mem-fraction-static` and `--attention-backend aiter` together, never separately.** aiter
  derates the fraction by 0.85 internally, so 0.306 is an effective 0.26. Change one alone and you
  land either below the startup floor or in the host OOM killer.
- **Name `--attention-backend aiter` explicitly.** `SGLANG_USE_AITER=1` alone switches aiter *ops*
  and leaves SGLang to pick the attention backend, which on ROCm is triton. aiter is worth about
  **+5%** at 40 concurrent streams.
- **Pass the chat template.** `--chat-template experiments/chat-template-qwen38.jinja` is the stock
  template plus three lines this model needs; without it tool-call round-trips do not render. The
  configuration file writes the path as `${SCRIPT_DIR}/...`, expanded when sourced, so the path
  travels with the launch line or it stops working.
- **Pass both parsers**, `--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder`. With one
  missing the server starts fine and fails at the first tool call in a way that reads as a client
  bug.
- **Keep at least 704 mamba slots.** The live peak is **315-322**, so that is a 2.2x margin, and it
  is also the point at which captured decode batches reach the full `--max-running-requests 128`.
- **Measure with 40 long-lived streams that re-send a growing conversation**, at both the p50 and
  the p90 prompt size you actually serve. A cold one-shot smoke sits above the threshold at every
  setting and reports every knob here as null.
- **Re-measure back to back on one node**, with the same configuration run first and last as a drift
  control. Node-to-node spread is about 30%, larger than most effects on this page.

## DO NOT

- **Do not serve this on vLLM.** See above.
- **Do not trust a cold smoke that reports a memory knob as null.** That is the threshold's flat
  side, not evidence. It is also why a comparison can tie: cells on the same side of the line score
  the same whatever their pool.
- **Do not drop `--mamba-full-memory-ratio` from a configuration.** Left out, the engine default of
  0.9 spends 47.4% of the budget on a state cache that never fills, and the KV pool falls to 1.91 M
  tokens. At 40 conversations of 60k that is hit 0.20 and 25 tok/s against 0.984 and 254.
- **Do not lower the ratio to 0.25 to buy a bigger pool.** It ties on throughput, cuts the slot count
  to 427, and caps captured decode batches at 85 -- below the 128 the server advertises. The pool is
  already past the threshold, so the extra tokens buy nothing and the margin is real.
- **Do not raise `--mem-fraction-static` further.** 0.306 already leaves only about 77 GB of host
  memory free, and on an APU an out-of-memory death is the host OOM killer taking the process with
  no traceback. Raise it only if the pool is below your working set, and re-read the free-memory
  line when you do.
- **Do not pass `--page-size 64` on this model.** Null to slightly negative, and a *smaller* KV pool.
  It is in the Kimi K2.7 recipe, where it belongs; see [`kimi27sglang.md`](kimi27sglang.md).
  Carrying it across is a common copy-paste.
- **Do not set `--chunked-prefill-size 16384`.** Null, inside noise.
- **Do not enable HiCache (`--enable-hierarchical-cache`, `--hicache-ratio`).** On an APU the host
  tier is the same physical memory, so it allocates a second copy of the KV cache rather than
  offloading anything; the server dies to the host OOM killer with no traceback.
- **Do not assume `--language-only` means "text only".** It is accepted by this architecture, but in
  current SGLang builds it selects the vision-encoder-disaggregation *receiver* role. GLM-5.3
  refuses to start with it. Do not reason from this model to that one.
- **Do not let the serving step run without `--cpus-per-task`.** A step that does not ask gets one
  core of 192, and the server degrades with load rather than failing. See [`README.md`](README.md).

---

## Open

- **`--schedule-policy lpm` on top of this configuration.** It groups prefix-sharing requests into
  the same batch, so it substitutes for pool size rather than adding to it. On a 2.42 M pool it is
  worth 2.3x at 40 conversations of 60k, and null at 33k. On this pool the prefix-cache hit rate is
  already 0.984, so there should be no headroom left for it to recover, but the two have never been
  measured together and the same reasoning predicted a null on the smaller pool and was wrong.
- **`--moe-runner-backend aiter` and `--fp8-gemm-backend aiter`.** Both accepted, neither measured.
  This model is MoE and fp8, so that is the larger remaining surface.
