# Serving GLM-5.3 on MI300A

`zai-org/GLM-5.3`, fp8, about 755 GB of weights, on SGLang across four nodes (`tp=4`, `pp=4`).
Source of truth: `[model.glm53]` plus the `glm53` entries of
`experiments/experiments.toml`; render with `hpcagent_bench/cluster/env_layers.sh render experiment:glm53`.
Background: [`knobs.md`](knobs.md).

```bash
MODEL=glm53 hpcagent_bench/cluster/serve-only.sbatch
```

**Image.** The same `hpcagent-bench-sglang-mi300-latest` as Qwen3.8 and Kimi K2.7: the sglang image
bakes in what GLM-5.3 needs (`torch.Tensor.format_ue8m0` kept false, and
`HIPCC_COMPILE_FLAGS_APPEND=-U__HIP_NO_HALF_CONVERSIONS__ -U__HIP_NO_HALF_OPERATORS__`).

## Configuration

`run_cluster.sh` adds `--tp-size 4 --pp-size 4 --nnodes 4 --node-rank <r> --dist-init-addr <rank0>:29500
--host 0.0.0.0 --port 8000` and **no** `--attention-backend` (the model selects `dsa`). The env adds:

```
--trust-remote-code --watchdog-timeout 1800
--kv-cache-dtype fp8_e4m3 --page-size 64 --context-length 262144
--mem-fraction-static 0.57 --max-total-tokens 1500000
--chunked-prefill-size 4096 --max-running-requests 24 --cuda-graph-max-bs-decode 24
--enable-metrics --pre-warm-nccl
--reasoning-parser glm45 --tool-call-parser glm47
--dsa-prefill-backend tilelang --dsa-decode-backend tilelang --enable-cache-report
```

Environment: `SGLANG_ATTENTION_BACKEND=` (assigned empty), `SGLANG_USE_AITER=1`,
`SGLANG_ROCM_FUSED_DECODE_MLA=0`, `SGLANG_SET_CPU_AFFINITY=0`, `NCCL_NET_GDR_LEVEL=LOC`,
`AITER_LOG_TUNED_CONFIG=1`, and `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` on the server alone
(`SGLANG_PYTORCH_HIP_ALLOC_CONF`; the judge must not inherit an allocator setting).

Twenty agents are live at once (`AGENTS_PER_NODE: 20`). `--max-running-requests 24` and
`--cuda-graph-max-bs-decode 24` size the request slots and decode graphs for them plus slack, not for
48/64 nobody reaches. Twelve agents peaked at 0.92 M pool tokens; a full pool queues requests, it does not crash. The
first 20-agent runs at a 1.8 M pool broke the rule that the heaviest stage stays under about 480 of
513 GB (677837 died at 508 GB, 677838 sat at 493 GB), so the pool is 1.5 M.

The fraction caps weights plus KV, so `pool(f) = 39.0M * (f - 0.4838)` tokens:

| `f` | Outcome |
|---|---|
| below 0.486 | refuses: weights alone exceed the budget |
| 0.50 | 632,384-token pool |
| 0.55 | 2.58 M pool; OOM-killed on a PP node at concurrency 20 |
| 0.57 + `--max-total-tokens 2800000` | pool pinned at 2.8 M; a 40-request leg peaked at 485 of 501 GiB, but hours of 12-agent load OOM-kill the heaviest stage |
| 0.57 + `--max-total-tokens 1800000` | pool pinned at 1.8 M; with 20 agents the heaviest stage reached 508 of 513 GB after 3 h and died (677837) |
| **0.57 + `--max-total-tokens 1500000`** | pool pinned at 1.5 M, about 14 GB below 1.8 M on the heaviest stage |
| 0.62 | OOM killer takes the heaviest stage |

Host memory, not the pool, kills a stage; `--chunked-prefill-size` and `--max-running-requests` bound
what grows with load. The dynamic (non-pool) allocations ratchet up as agent contexts grow: in 676681
(12 agents, pool 2.8 M, peak pool use 0.33) the heaviest stage's GPU memory rose from 401 to 446 GB in
3 h of serving, the node reached 504-508 of 513 GB, and Slurm OOM-killed it. The 1.5 M pool still holds
the observed 0.92 M working set and leaves about 14 GB more headroom than 1.8 M on the heaviest stage. Stages are uneven (172.4 / 197.2 / 203.8 / 206.1 GB). Time to a live API is the
slowest stage's weight load (up to 5400 s) plus about 130 s KV allocation and 1080 s graph capture;
`AGENT_READY_TIMEOUT_SECONDS=10800` covers it.

## Rules

- Assign `SGLANG_ATTENTION_BACKEND=` empty, never `${VAR:-default}`: an absent key appends
  `--attention-backend aiter` ([README](README.md#6-how-configuration-becomes-flags)), and any explicit
  backend suppresses `dsa`.
- Keep `SGLANG_USE_AITER=1` (without it the DSA path forces `page_size` 1); do not set
  `SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION`.
- Do not pass `--language-only` (the server refuses to start) or enable HiCache. Do not carry
  `AITER_USE_FLYDSL_MOE_SORTING` over from Kimi K2.7. GLM-5.3-Flash is CUDA-only (`IndexerKPool`).
- Pass both parsers, `glm45` and `glm47`. Give an accuracy gate 2048 tokens (the model reasons first),
  and gate a `--kv-cache-dtype` change on long context.
- Read back every launch; `Using network` must say `AWS Libfabric`:
  `grep -aiE "attention.backend|Use dsa attention|max_total_num_tokens|Using network" server-0.log`.

## aiter DSA kernels (`--dsa-prefill-backend aiter --dsa-decode-backend aiter`)

GLM-5.3's sparse attention runs through SGLang's `dsa` backend; `--attention-backend aiter` is not a
substitute (`AiterAttnBackend.forward_decode() got an unexpected keyword argument 'topk_indices'`). The aiter route is the two `--dsa-*-backend aiter` flags. Both aiter paths call
`aiter.mla_decode_fwd` in persistent mode (kernel `mla_a16w8_qh16_m16x4_n16x1_coex0_mask1_ps`: bf16 q,
fp8 KV, 16 heads per rank) with metadata from `get_mla_metadata_v1`, and each faults for a
different reason:

| path | symptom | cause |
|---|---|---|
| decode | illegal memory access in CUDA-graph capture at bs 40, right after bs 48; a scheduler rank aborts (-6) | `_forward_aiter` passes the whole `self.kv_indptr` buffer (`max_bs + 1` entries) and `get_mla_metadata_v1` sets `num_batches = kv_indptr.size(0) - 1`. Below `max_bs` the planner schedules the stale tail batches; stage 1 and `mla_reduce_v1` write output rows past `bs`. Where that lands in mapped memory the result is wrong (NaN) instead of a fault, so a run that does not crash is not correct either. |
| prefill | illegal memory access on the first 4096-token chunk (the 12.8k-token accuracy step) | `get_mla_metadata_v1` (sparse, `intra_batch_mode`) plans garbage for 4096 or more batches; the extend path makes every token a batch. 4095 tokens is correct, 4096 and 4100 fault, 8192 returns wrong values with `kv_end <= 1` in the work list, independent of KV length. |

Neither is MoE sorting, CUDA-graph capture, RCCL or PP: a single-GPU replay of the two paths, with no
model, no graph and no collectives, faults and fixes the same way (`moe_sorting` and `fused_qk_rmsnorm`
in server stacks are only the next HIP call after the faulting kernel).

**Fix.** Decode: the SGLang recipe (`containers/images/sglang/Dockerfile`) edits
`kv_indptr = self.kv_indptr` to `self.kv_indptr[: bs + 1]` in `_forward_aiter`; the slice is a view, so
graph capture and the in-place cumsum are unchanged. Prefill: `--chunked-prefill-size 2048` keeps every
extend batch below 4096 tokens (the scheduler debits `rem_chunk_tokens` per request) and halves the
transient fp32 split buffer aiter allocates per layer (8 GiB at 4096 tokens). Both are needed for the
aiter pair; no path falls back.

No throughput is measured for this configuration yet.
