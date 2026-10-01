# Serving GLM-5.3 on MI300A

`zai-org/GLM-5.3`, fp8, about 755 GB of weights, on SGLang across four nodes (`tp=4`, `pp=4`).
Source of truth: `experiments/layers/model-glm53.env` plus the `glm53` entries of
`experiments/arms.yaml`; render with `hpcagent_bench/cluster/env_layers.sh render experiment:glm53`.
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
--mem-fraction-static 0.57 --max-total-tokens 2800000
--chunked-prefill-size 4096 --max-running-requests 48 --cuda-graph-max-bs-decode 64
--enable-metrics --pre-warm-nccl
--reasoning-parser glm45 --tool-call-parser glm47
--dsa-prefill-backend tilelang --dsa-decode-backend tilelang --enable-cache-report
```

Environment: `SGLANG_ATTENTION_BACKEND=` (assigned empty), `SGLANG_USE_AITER=1`,
`SGLANG_ROCM_FUSED_DECODE_MLA=0`, `SGLANG_SET_CPU_AFFINITY=0`, `NCCL_NET_GDR_LEVEL=0`,
`AITER_LOG_TUNED_CONFIG=1`.

The fraction caps weights plus KV, so `pool(f) = 39.0M * (f - 0.4838)` tokens:

| `f` | Outcome |
|---|---|
| below 0.486 | refuses: weights alone exceed the budget |
| 0.50 | 632,384-token pool |
| 0.55 | 2.58 M pool; OOM-killed on a PP node at concurrency 20 |
| **0.57 + `--max-total-tokens 2800000`** | pool pinned at 2.8 M; concurrency 40, peak 485 of 501 GiB step cgroup |
| 0.62 | OOM killer takes the heaviest stage |

Host memory, not the pool, kills a stage; `--chunked-prefill-size` and `--max-running-requests` bound
what grows with load. Stages are uneven (172.4 / 197.2 / 203.8 / 206.1 GB). Time to a live API is the
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
