# Serving Kimi K2.7 on MI300A

`moonshotai/Kimi-K2.7-Code` on SGLang, four nodes, `tp=4` per node and `pp=4` across them (it does
not fit fewer). Source of truth: `[model.kimi27sglang]` (render:
`experiment:kimi27sglang`). Background: [`knobs.md`](knobs.md).

```bash
MODEL=kimi27sglang hpcagent_bench/cluster/serve-only.sbatch
```

## Configuration

EDF `hpcagent-bench-sglang-mi300-latest`. Port 8000 on rank 0 only, served name
`hpcagent-bench-vllm`. Weight load takes 30-40 minutes before `KV Cache is allocated`; the readiness wait is
`AGENT_READY_TIMEOUT_SECONDS=7200` (serve-only.sbatch and the agent driver read it before `VLLM_READY_TIMEOUT_SECONDS`).

`run_cluster.sh` adds `--tp-size 4 --pp-size 4 --nnodes 4 --node-rank <r> --dist-init-addr <rank0>:29500
--host 0.0.0.0 --port 8000 --attention-backend aiter`. The env adds:

```
--trust-remote-code --language-only --watchdog-timeout 1800
--kv-cache-dtype fp8_e4m3 --page-size 64 --context-length 262144
--mem-fraction-static 0.55 --max-running-requests 24 --cuda-graph-max-bs-decode 24
--enable-metrics --pre-warm-nccl
--reasoning-parser kimi_k2 --tool-call-parser kimi_k2 --enable-cache-report
```

Environment: `SGLANG_USE_AITER=1`, `SGLANG_ROCM_FUSED_DECODE_MLA=0`, `SGLANG_SET_CPU_AFFINITY=0`,
`NCCL_NET_GDR_LEVEL=LOC`, `AITER_USE_FLYDSL_MOE_SORTING=1`, `AITER_LOG_TUNED_CONFIG=1`, and
`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` on the server alone (`SGLANG_PYTORCH_HIP_ALLOC_CONF`).

| Quantity at `pp=4` | Value |
|---|---|
| Free per rank before weight load | about 412 GB |
| Weights per stage | about 171 GB (unequal: read `avail mem=` on every rank) |
| `--mem-fraction-static` 0.55 | 0.4675 effective (aiter x0.85); throughput-neutral across 0.408-0.4675 |
| 0.588 (0.4998 effective) | heaviest stage at 513.2 of 513.5 GB: too close to OOM |
| `pp=2` | refuses at startup: `minimum viable = 0.7525` |

The pool has not been measured against a working set; if the hit rate falls under load, serve fewer
conversations, since the fraction has no headroom left.

## Rules

- Serve on SGLang (engine numbers are re-measured when agent tuning is done).
- Keep `--cuda-graph-max-bs-decode` at the live agent count plus slack (24 for 20 agents; graph capture
  takes memory after KV sizing; without the cap the failure looks like a fraction problem),
  `--max-running-requests` the same, `--page-size 64` (vendor `gfx942` recipe) and the
  fraction at or below 0.55.
- Set `NCCL_NET_GDR_LEVEL=LOC`, keep PyNCCL on (about 20x slower decode without it), and confirm
  `grep -a "Using network" server-0.log` says `AWS Libfabric` ([`knobs.md`](knobs.md#multi-node-fabric)).
- Gate a `--kv-cache-dtype` change on long context: fp8 checkpoints ship no calibrated KV scales, and
  a broken attention path still answers short prompts (`containers/inference/accuracy-gate.py`, about
  10k tokens of varied filler).
- One client per endpoint; no HiCache ([`knobs.md`](knobs.md#hicache-never)); leave
  `SGLANG_ROCM_FUSED_DECODE_MLA` off. `AITER_USE_FLYDSL_MOE_SORTING=1` and `pp=4` are for this
  checkpoint only.
