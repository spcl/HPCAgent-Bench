# Cross-model serving knobs on MI300A

This page holds only what applies to every model. Per-model numbers live on
[`qwen38.md`](qwen38.md), [`kimi27sglang.md`](kimi27sglang.md), [`glm53.md`](glm53.md) and
[`oss120b.md`](oss120b.md). Node-to-node throughput spread is about 30%, so every ratio quoted here
was measured back to back on one node, under 20 to 40 long-lived streams that re-send a growing
conversation.

## The MI300A memory model

MI300A is an APU: GPUs and host share one physical memory pool.

1. **`--mem-fraction-static` is node-wide.** SGLang sees `is_integrated` and sizes its static budget
   against the whole node's memory, per rank. Safe values are far below upstream's 0.8-0.9 and differ
   per model and node count. vLLM's `--gpu-memory-utilization` behaves the same way.
2. **The fraction is a floor as well as a ceiling.** It bounds weights and KV together, measured
   against free memory before weights load. At pipeline depth `pp` each stage holds `1/pp` of the
   weights; fewer nodes raise the floor until SGLang refuses with `minimum viable = ...`. The fix is
   **more nodes**, not a bigger fraction.
3. **There is no second tier.** Spilling KV to "host memory" spills into the same pool.

An out-of-memory death is the host OOM killer: the log stops, no Python traceback, exit by signal.
Read the last `avail mem=` line.

## The KV pool threshold

Every cache knob acts through one ratio:

```
pool / working set     pool        = max_total_num_tokens (printed at startup)
                       working set = prompt tokens of all concurrent conversations at their largest
```

Above the crossing, every configuration measured reaches a prefix-cache hit rate of 0.984-0.988.
Below it, every configuration thrashes. On Qwen3.8 (14 leg-concurrency cells, no exceptions) the
crossing lies between ratio 0.90 (hit 0.601) and 1.04 (hit 0.984). Consequences:

1. Raising the pool buys nothing above the threshold and up to 10x below it (Qwen3.8: 25 against
   271 tok/s at the largest measured gap).
2. A cold one-shot smoke has almost no working set and sits above the threshold at every setting,
   so it reports every cache knob as null. Load 20+ long-lived streams instead.
3. Tune the ratio, not a particular flag: two flags that land the same pool land the same throughput.

Find the pool with `grep -a "max_total_num_tokens\|KV Cache is allocated" server-0.log`, estimate
the working set as conversations times largest prompt, and aim the pool at about 1.3x it. With
`--enable-cache-report`, a hit rate that falls as conversations grow is the ratio crossing below 1.

## `--mem-fraction-static` and `--attention-backend aiter` are one decision

With the aiter attention backend and a context above 8192 tokens, SGLang multiplies
`--mem-fraction-static` by **0.85**: a configured 0.55 is an effective 0.4675. On any other backend,
or a shorter context, the configured value is the effective one.

- Never move one without the other: dropping aiter at the same number overshoots into the OOM
  killer; raising the number while keeping aiter scales differently than you expect.
- Read the resulting pool from the allocator's `KV size: X GB` line, never infer it from the flag.
- `SGLANG_USE_AITER=1` switches aiter **ops**, not the attention backend. Keep it on: without it the
  ROCm path loses aiter's preshuffled paged-MQA kernel and forces `page_size` 1.
- `run_cluster.sh` appends `--attention-backend aiter` unless `SGLANG_ATTENTION_BACKEND` is assigned
  (see [README](README.md#6-how-configuration-becomes-flags)). An explicit backend **suppresses** a
  model's own choice; that is how GLM-5.3 would lose `dsa`. Check a value against
  `python3 -m sglang.launch_server --help` inside the image first (an unknown value kills every rank),
  and read `attention_backend=` back from the server log.

## HiCache: never

Do not set `--enable-hierarchical-cache` or `--hicache-ratio`. On an APU the "host" tier is the pool
the KV cache already lives in, so it allocates a second KV copy in the same memory and the server
dies to the OOM killer with no traceback. No `experiments/` env sets it.

## Both parsers, always

Name `--reasoning-parser` **and** `--tool-call-parser`. With one missing the server starts normally,
then the first request using the other feature fails: a 400 the client logs as success, or a tool
call returned as prose instead of `tool_calls`. Nothing in the log says "parser". Verify with
`containers/cluster/ce-images/inference/verify-tools-reasoning.py`, which asserts
`choices[0].message.tool_calls[0]` and non-empty reasoning. Parser names are on each model page.

## Topology

- **Pipeline parallelism only when the model does not fit one node.** `pp=4` spends about 42% of
  engine time in stalls.
- **Several nodes for a model that fits = independent replicas**, each on its own hostname with its
  own KV cache. More aggregate throughput, nothing for one conversation.
- **One client per large-model server.** Four clients against one Kimi K2.7 endpoint did less useful
  work than one: they evict each other's prefixes.

## Multi-node fabric

- **`NCCL_NET_GDR_LEVEL=0`** (not `NCCL_GDR_LEVEL`). RCCL enables GPU-direct RDMA by itself, and on
  this fabric it silently corrupts cross-node sums.
- **Keep PyNCCL.** Disabling it breaks graph capture and drops decode from about 17 to 1.4 tok/s per
  request on the four-node shape.
- **Check the plugin:** `grep -a "Using network" server-0.log` must say `AWS Libfabric`.
  `NET/Plugin: Could not find ...` means TCP fallback: correct, several times slower, no error.
- **`--pre-warm-nccl`** moves collective init out of the first request's latency.

## Environment variables

| Variable | Value | Why |
|---|---|---|
| `SGLANG_USE_AITER` | `1` | aiter ops; does not pick the attention backend |
| `SGLANG_SET_CPU_AFFINITY` | `0` | SGLang's own pinning is rejected by the Slurm cgroup; dies on a `psutil` error |
| `AITER_JIT_DIR` | the image's `/opt/aiter-jit` (set by the EDF) | aiter JIT-builds on first use behind a lock; cold, the build can outrun the engine watchdog. Some aiter ops ignore it and use `$HOME/.aiter`; `run_cluster.sh` points `HOME` at a persistent cache |
| `TRITON_CACHE_DIR` | persistent (`run_cluster.sh` derives it from `JIT_CACHE_ROOT`) | unset, every job re-JITs kernels during inference and generation stalls in bursts |
| `HF_HOME` | on `iopsstor` (`$FAST_SCRATCH`, default from `scripts/cache_env.sh`) | 11x faster than general scratch at 16 concurrent readers; `run_cluster.sh` stripes `$HF_HOME/hub` wide |
| `NCCL_NET_GDR_LEVEL` | `0` | multi-node only |
| `TOKENIZERS_PARALLELISM` | `false` | silences a fork warning |

## Cheap flags worth setting everywhere (SGLang)

| Flag | Why |
|---|---|
| `--enable-metrics` | Prometheus at `/metrics`: throughput, running and waiting counts |
| `--enable-cache-report` | `cached_tokens` per response: the per-request view of the threshold above |
| `--watchdog-timeout 1800` | a wedged engine dies instead of hanging to the wall clock |
| `--max-running-requests` | scheduler concurrency cap; reserves no memory |

Slurm flags that act as serving knobs (`--mem=0`, `--cpus-per-task`, `--partition=mi300`, no `-A`,
`ulimit -c 0`) are in [`README.md`](README.md#3-slurm-shape).
