# Serving Kimi K2.7 on MI300A

`moonshotai/Kimi-K2.7-Code`. Four nodes, sixteen GPUs. This model does not fit in one node, so the
four-node layout is not a throughput choice: it is the only way to serve it here.

Authoritative source: `experiments/.env.base-kimi27sglang`. If this page and that file disagree, the
file is right. Cross-model background is in [`knobs.md`](knobs.md).

## Current configuration

| | |
|---|---|
| Engine | SGLang |
| EDF | `sglang-latest` |
| Nodes | **4**, `tp=4` inside each node, `pp=4` across them |
| Port | 8000 on **rank 0 only**, served name `optarena-vllm` |
| Weight load | 30-40 minutes before the API answers |

```
--tp-size 4 --pp-size 4 --nnodes 4 --node-rank <rank> --dist-init-addr <rank0>:29500
--host 0.0.0.0 --port 8000
--attention-backend aiter
--trust-remote-code
--language-only
--watchdog-timeout 1800
--kv-cache-dtype fp8_e4m3
--page-size 64
--context-length 262144
--mem-fraction-static 0.588
--cuda-graph-max-bs-decode 64
--max-running-requests 128
--enable-metrics
--pre-warm-nccl
--reasoning-parser kimi_k2
--tool-call-parser kimi_k2
--enable-cache-report
```

Environment: `SGLANG_USE_AITER=1`, `SGLANG_ROCM_FUSED_DECODE_MLA=0`, `SGLANG_SET_CPU_AFFINITY=0`,
`NCCL_NET_GDR_LEVEL=0`, `AITER_USE_FLYDSL_MOE_SORTING=1`.

```bash
cd experiments
MODEL=kimi27sglang ./serve-only.sbatch
```

### Where this configuration sits against the KV pool threshold

[`knobs.md`](knobs.md) explains the threshold: above the crossing ratio every configuration reaches
a prefix-cache hit rate of 0.984-0.988; below it, every configuration thrashes. The crossing ratio
is model-specific and was measured near 1.0 on Qwen3.8. This model's own crossing has not been
measured, so read the hit rate directly instead of assuming a ratio from another model:

```bash
grep -a "max_total_num_tokens\|KV Cache is allocated" server-0.log     # your pool
grep -a "cached_tokens"                                server-0.log   # your hit rate
```

Estimate your working set as concurrent conversations times their largest prompt, and aim the pool
at about 1.3x it. `--mem-fraction-static 0.588` is already at this model's ceiling, so if the hit
rate is falling, the lever is fewer concurrent conversations, not a bigger fraction.

## DO

- **Serve this on SGLang, not vLLM.** vLLM stalls above concurrency 1 on this topology. Aggregate
  tok/s at concurrency 1 / 2 / 4 / 6 (measured 2026-08-27): SGLang 13.7 / 17.6 / 38.1 / 46.8; vLLM
  20.6 / 6.4 / 7.0 / 6.6, with 42-43% of its samples generating nothing at all. This comparison is
  recorded in `experiments/run_cluster.sh` beside the code that acts on it.
- **Allocate four nodes.** At `pp=2` each pipeline stage holds twice the weights and SGLang refuses
  at startup with "minimum viable = 0.7525" (measured 2026-09-02). When you see that message, raise
  the node count.
- **Keep `--mem-fraction-static 0.588` paired with `--attention-backend aiter`.** 0.588 configured
  is **0.50 effective** under aiter's 0.85 derate, and 0.50 is the ceiling: higher runs out of
  memory. The floor at `pp=4` is near 0.415, because the model loads about 171 GB of weights per
  stage against roughly 412 GB free before load.
- **Keep `--cuda-graph-max-bs-decode 64`.** Graph capture happens *after* the KV cache is sized and
  takes its memory from what is left. Without the cap, `--mem-fraction-static 0.42` on four nodes
  lands at an effective 0.357 against a 0.3988 floor and the server refuses to start. An apparent
  4.7x KV shortfall on this model was read as a memory-fraction problem for weeks and was this
  (measured 2026-09-05, confirmed on the live image 2026-09-08).
- **Keep `--page-size 64`.** Part of the vendor `gfx942` recipe; it avoids decode faults seen at
  `page_size=1` on this part. Note that the same flag measured **null to slightly negative** on
  Qwen3.8 -- see [`qwen38.md`](qwen38.md). It is a per-model answer, not a general one.
- **Set `--context-length 262144`.** 256k context costs nothing measurable against a shorter window
  on this model (measured 2026-08-29). The KV pool, not the declared window, is what constrains a
  real workload.
- **Set `NCCL_NET_GDR_LEVEL=0`.** RCCL enables GPU-direct RDMA by itself and on this fabric it
  silently corrupts cross-node collectives: wrong sums, not an error (measured 2026-08-25). Note the
  variable name; it is not `NCCL_GDR_LEVEL`.
- **Confirm the fabric came up before trusting any throughput number.**
  ```bash
  grep -a "Using network" server-0.log        # want: AWS Libfabric
  ```
  A `NET/Plugin: Could not find: libnccl-net.so` line means RCCL fell back to TCP: correct answers,
  several times slower, no error.
- **Pass both parsers**, `--reasoning-parser kimi_k2` and `--tool-call-parser kimi_k2`.
- **Size against the heaviest pipeline stage and read `avail mem=` on every rank.** The four stages
  are not equal consumers of memory; a change that fits on rank 0 can still take rank 3 to the host
  OOM killer.
- **Keep `HF_HOME` on `iopsstor` and stripe the hub directory wide.** 9.45 GB/s against 0.83 at 16
  concurrent readers (measured 2026-08-26). A checkpoint downloaded into a narrow-striped directory
  reads back at one storage target's bandwidth.
- **Keep the aiter JIT cache warm.** The EDF points at the image's baked `/opt/aiter-jit`. aiter
  JIT-builds on first **use**, not on import, behind a lock; cold, that build can outrun the
  engine's watchdog and no token is ever decoded.

## DO NOT

- **Do not serve this on vLLM.** See above: it collapses above concurrency 1, and nearly half its
  samples generate nothing.
- **Do not copy this pipeline layout onto a model that fits in one node.** `pp=4` costs about **42%**
  of engine time to stalls (measured 2026-08-30). Here that is the price of serving the model at
  all; on Qwen3.8 or gpt-oss-120b it is pure loss.
- **Do not raise `--mem-fraction-static` above 0.588.** 0.50 effective is the ceiling; above it the
  host OOM killer takes the process, and on an APU that arrives with no traceback -- the log simply
  stops after a `Load weight` or `Memory pool` line.
- **Do not drop `--cuda-graph-max-bs-decode` while changing `--mem-fraction-static`.** The residual
  comes out of the KV pool and the failure reads as a memory-fraction problem that it is not.
- **Do not change `--kv-cache-dtype` on the strength of a short accuracy check.** An fp8 checkpoint
  ships **no calibrated KV scales**, so the engine quantizes at runtime against scale 1.0. A corrupt
  attention path answers short prompts correctly. Gate any change on long context:
  `containers/cluster/ce-images/inference/accuracy-gate.py` asks at about 10k tokens of **varied**
  filler for exactly this reason (in use since 2026-09-02). Varied, not repeated: literal repetition
  creates an echo attractor at temperature 0 that even a correct backend falls into.
- **Do not enable HiCache (`--enable-hierarchical-cache`, `--hicache-ratio`).** On an APU the host
  tier is the same physical memory: it allocates a second copy of the KV cache rather than
  offloading anything, and the server dies to the host OOM killer with no traceback (measured
  2026-09-03).
- **Do not disable PyNCCL to work around a collective problem.** About **20x**: graph capture
  stalls, eager mode goes on, and decode drops from about 17 tok/s per request to 1.4 on this exact
  four-node shape (measured 2026-08-24).
- **Do not point four clients at one endpoint.** Four clients produced less useful work than one
  (measured 2026-08-28): they compete for the same KV pool and evict each other's prefixes. Four
  users are better served by four one-node servers -- of a model that fits.
- **Do not turn on `SGLANG_ROCM_FUSED_DECODE_MLA`.** Off, because the vendor recipe says so. The
  kernel it would route decode to verifies against an fp32 reference (max absolute difference 6e-4
  at 1k KV, 7e-4 at 4k, measured 2026-09-09), but that is the prerequisite for *trying* it, not
  evidence that serving with it on is correct.
- **Do not carry `AITER_USE_FLYDSL_MOE_SORTING=1` to another model.** It is specific to this
  checkpoint's pack-quantized int4 weights. GLM-5.3's fp8 weights were never measured with it.
- **Do not conclude the server is wedged before `KV Cache is allocated` appears.** 30-40 minutes of
  weight loading is normal on this model.

## Open questions

- **Where this model's pool sits against the KV pool threshold.** Not measured. The threshold was
  established on Qwen3.8, and whether the crossing ratio carries to this model is unknown. Measure
  this model's own pool against a realistic working set, and read the prefix-cache hit rate.
- **Whether `--context-length` below 262144 would free useful pool.** The 2026-08-29 result says the
  window is free, which implies not, but that was measured on the declared window rather than on
  pool size directly.

## The data behind the instructions

### Engine comparison, aggregate tok/s by concurrency (2026-08-27)

| Engine | 1 | 2 | 4 | 6 |
|---|---|---|---|---|
| SGLang | 13.7 | 17.6 | 38.1 | 46.8 |
| vLLM | 20.6 | 6.4 | 7.0 | 6.6 |

vLLM wins the single-stream case and loses every other one.

### Memory budget at `pp=4`

| Quantity | Value |
|---|---|
| Free memory per rank before weight load | about 412 GB |
| Weights per pipeline stage | about 171 GB |
| Resulting `--mem-fraction-static` floor | about 0.415 |
| Ceiling (higher OOMs) | 0.50 effective, 0.588 configured |
| At `pp=2` | SGLang refuses: "minimum viable = 0.7525" |

Measured 2026-09-02.

### Arguments and what each buys

| Argument | What it buys |
|---|---|
| `--attention-backend aiter` | The vendor `gfx942` backend. Applies the 0.85 derate, so it and the fraction move together. |
| `--kv-cache-dtype fp8_e4m3` | Halves KV cost per token. Not free; see the DON'T list. |
| `--page-size 64` | Avoids `page_size=1` decode faults on this part. |
| `--mem-fraction-static 0.588` | Effective 0.50, this model's ceiling. |
| `--cuda-graph-max-bs-decode 64` | Stops graph-capture residual eating the KV pool. |
| `--context-length 262144` | 256k window, measured free. |
| `--pre-warm-nccl` | Moves collective init out of the first user's latency. |
| `--language-only` | Accepted by this architecture. Not a generic text-only switch; GLM-5.3 refuses to start with it. |
| `AITER_USE_FLYDSL_MOE_SORTING=1` | Specific to this checkpoint's int4 weights. |
