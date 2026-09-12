# Serving Qwen3.8 on MI300A

`Qwen/Qwen3.8-27B-FP8`. One node, four GPUs. The cheapest useful endpoint on this machine and the
one to start from.

Authoritative source: `experiments/.env.base-qwen38`. If this page and that file disagree, the file
is right. Cross-model background is in [`knobs.md`](knobs.md).

---

## Best known configuration (2026-09-11)

| | |
|---|---|
| Engine | SGLang |
| EDF | `sglang-latest` |
| Nodes | **1**, `tp=4`, no pipeline stage |
| Port | 8000, served name `optarena-vllm` |
| KV pool it produces | **2,426,200 tokens** |
| Weight load | a few minutes |

```
--tp-size 4 --host 0.0.0.0 --port 8000
--attention-backend aiter
--chat-template experiments/chat-template-qwen38.jinja
--trust-remote-code
--language-only
--watchdog-timeout 1800
--context-length 262144
--mem-fraction-static 0.247
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

### Where this configuration sits against the KV pool threshold

[`knobs.md`](knobs.md) explains the threshold: above a pool-to-working-set ratio of about **1.15**
every configuration reaches a prefix-cache hit rate of 0.984-0.988; below it, every configuration
thrashes.

| Load | Working set | Pool / working set | Result |
|---|---|---|---|
| 20 long-lived streams | about 2.06 M tokens | **1.18** | just above the threshold: hit 0.988, 172 tok/s |
| 40 long-lived streams | about 3.18 M tokens | **0.76** | below: hit 0.27, 27 tok/s |

**The shipped configuration is sized for about 20 concurrent conversations and no more.** Measure
your own working set (concurrent conversations times their largest prompt) and aim the pool at about
1.3x it. At 40 streams the configuration that clears the threshold is
`--mem-fraction-static 0.306 --mamba-full-memory-ratio 0.25` (pool 4.03 M, ratio 1.27, 271 tok/s),
but read the DON'T list before copying that: it changes the mamba slot count too.

---

## DO

- **Serve this on SGLang, not vLLM.** Roughly **19x** the throughput for this model
  (measured 2026-08-27). This figure is corroborated only by a comment in `.env.base-qwen38`
  ("vLLM cannot serve this hybrid backbone at usable throughput"); no artefact of the run survives
  on disk. The direction is not in doubt.
- **Set `--mamba-full-memory-ratio 0.5`.** The engine default of 0.9 spends 47.4% of the budget on a
  state cache that never fills. Moving to 0.5 takes the KV pool from 1,915,229 to 2,426,200 tokens
  at **zero extra memory**, and takes 92 tok/s to 172 at 20 streams (measured 2026-09-11).
- **Name `--attention-backend aiter` explicitly.** `SGLANG_USE_AITER=1` alone switches aiter *ops*
  and leaves SGLang to pick the attention backend, which on ROCm is triton. aiter is worth about
  **+5% at 40 concurrent streams** (measured 2026-09-06; corroborated by a configuration comment,
  no surviving artefact).
- **Move `--mem-fraction-static` and `--attention-backend aiter` together, never separately.** aiter
  derates the fraction by 0.85 internally, so 0.247 is an effective 0.21. Change one alone and you
  land either below the startup floor or in the host OOM killer.
- **Pass the chat template.** `--chat-template experiments/chat-template-qwen38.jinja` is the stock
  template plus three lines this model needs; without it tool-call round-trips do not render. The
  configuration file writes the path as `${SCRIPT_DIR}/...`, expanded when sourced, so the path
  travels with the launch line or it stops working.
- **Pass both parsers**, `--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder`. With one
  missing the server starts fine and fails at the first tool call in a way that reads as a client
  bug.
- **Read the two allocation lines out of the log before believing any tuning.**
  ```bash
  grep -a "Mamba Cache is allocated\|max_total_num_tokens" server-0.log
  ```
  They report the mamba slot count and the KV pool, which are the two numbers every knob on this
  page moves.
- **Measure with 20 or more long-lived streams that re-send a growing conversation.** A cold
  one-shot smoke sits above the threshold at every setting and reports every knob here as null.
- **Re-measure back to back on one node.** Node-to-node spread is about 30% (measured 2026-09-04),
  which is larger than most effects on this page.

## DO NOT

- **Do not serve this on vLLM.** See above: about 19x slower. There is no configuration of vLLM here
  that recovers it.
- **Do not set `--mamba-full-memory-ratio 0.25` at `--mem-fraction-static 0.247`.** It yields **308**
  mamba slots against a **315-322** observed peak, so a busy server begins evicting state. It
  measured fine at 20 streams only because 20 streams never reach the peak (measured 2026-09-11).
  The rule is about **slots, not the ratio**: at `--mem-fraction-static 0.306` the same 0.25 gives
  427 slots and is safe. Read the slot count, do not reason from the ratio.
- **Do not pass `--page-size 64` on this model.** It measured **null to slightly negative**: 82 tok/s
  against baseline legs of 92 and 86, and a *smaller* KV pool, 1,905,280 tokens against 1,915,229
  (measured 2026-09-11). It is in the Kimi K2.7 recipe, where it is part of the vendor `gfx942`
  recipe and belongs; see [`kimi27sglang.md`](kimi27sglang.md). Carrying it across is a common
  copy-paste. It costs about 0.5% of the pool, so it is not what is hurting you either.
- **Do not raise `--mem-fraction-static` expecting a gain while you are above the threshold.** At 20
  streams, 0.247 to 0.306 changed throughput by less than run-to-run noise (172 against 164) and
  halved the host-memory safety margin (measured 2026-09-11). Raise it when the pool is *below* your
  working set, not otherwise.
- **Do not set `--chunked-prefill-size 16384`.** Null, inside noise: 88 tok/s against baseline legs
  of 92 and 86 (measured 2026-09-11).
- **Do not enable HiCache (`--enable-hierarchical-cache`, `--hicache-ratio`).** On an APU the host
  tier is the same physical memory, so it allocates a second copy of the KV cache rather than
  offloading anything; the server dies to the host OOM killer with no traceback (measured
  2026-09-03).
- **Do not assume `--language-only` means "text only".** It is accepted by this architecture, but in
  current SGLang builds it selects the vision-encoder-disaggregation *receiver* role. GLM-5.3
  refuses to start with it. Do not reason from this model to that one.
- **Do not let the serving step run without `--cpus-per-task`.** A step that does not ask gets one
  core of 192, and the server degrades with load rather than failing. See [`README.md`](README.md).
- **Do not trust a cold smoke that reports a knob as null.** That is the threshold's flat side, not
  evidence.

## Open questions

- **Whether the larger KV pool beats the landed configuration at the campaign's real prompt sizes.**
  A probe is in flight. The 40-stream numbers below come from a single warm round per configuration
  and point at `--mem-fraction-static 0.306` plus a lower mamba ratio; whether that holds at the
  prompt sizes actually served is exactly what is being settled. **Treat the 40-stream row of this
  page as provisional until that lands.**
- **`--schedule-policy lpm`** measured 115 tok/s at 20 streams and 50 at 40, against baselines of
  92/86 and 25 (measured 2026-09-11). That is a real-looking gain at 40 that the mamba-ratio change
  dwarfs, and the two were never measured **in combination**. Not shipped on this evidence. Worth a
  sweep.

---

## The data behind the instructions

All from one sweep on 2026-09-11: one node, legs back to back, with two baseline legs run before and
after as a drift control (92 and 86 tok/s at 20 streams).

### `--mamba-full-memory-ratio`, 20 streams

This model is hybrid, so SGLang keeps a **mamba state cache** alongside the KV cache and sizes both
out of **one** budget: with ratio `r` the state cache gets `R*r/(1+r)` and the KV cache `R/(1+r)`.
Every byte one gets, the other does not. The state cache is sized in **slots**, one per concurrent
sequence.

| ratio | mamba slots | KV pool (tokens) | prefix-cache hit | aggregate |
|---|---|---|---|---|
| 0.9 (engine default) | 731 | 1,915,229 | 0.92 | 92 tok/s |
| **0.5 (shipped)** | **514** | **2,426,200** | **0.99** | **172 tok/s** |
| 0.25 | 308 | 2,910,904 | 0.99 | 165 tok/s |

514 slots is about 1.6x the 315-322 observed peak, which is the margin worth keeping.

### Both knobs at 40 streams (provisional: one warm round each)

| configuration | KV pool | pool / working set | hit | aggregate |
|---|---|---|---|---|
| `0.247`, ratio 0.9 | 1.92 M | 0.60 | 0.20 | 25 tok/s |
| `0.247`, ratio 0.5 (shipped) | 2.43 M | 0.76 | 0.27 | 27 tok/s |
| `0.306`, ratio 0.9 | 2.65 M | 0.84 | 0.60 | 43 tok/s |
| `0.306`, ratio 0.25 | 4.03 M | 1.27 | 0.98 | 271 tok/s |

Only the last row clears the 1.15 threshold, and only the last row performs. That is the threshold,
not a slope.

### Dead ends

| Change | Result at 20 streams | Verdict |
|---|---|---|
| `--page-size 64` | 82 tok/s, pool 1,905,280 | null to slightly negative |
| `--chunked-prefill-size 16384` | 88 tok/s | null, inside noise |
| `--mem-fraction-static 0.306` alone | 164 tok/s | no gain above the threshold; halves the host-memory margin |
| `--schedule-policy lpm` | 115 tok/s | see Open questions |

### An older instance of the same threshold

One small increase in pool size took the prefix-cache hit rate from 8% to 99.9% and throughput from
26 to 360 tok/s (measured 2026-09-06). That number is corroborated only by project notes; no
artefact survives on disk. The shape is reproduced by the tables above, and the threshold is the
explanation for both.
