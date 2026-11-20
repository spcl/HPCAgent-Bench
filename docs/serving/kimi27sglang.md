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
--trust-remote-code --attention-backend aiter --language-only --watchdog-timeout 1800
--kv-cache-dtype fp8_e4m3 --page-size 64 --context-length 262144
--mem-fraction-static 0.52 --max-running-requests 24 --cuda-graph-max-bs-decode 24
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
| `--mem-fraction-static` 0.52 | 0.442 effective (aiter x0.85): about 16 GB per rank above the heaviest stage, KV pool about 2.2M tokens |
| 0.48 (0.408 effective) | 2-4 GB left above PP3's weights: KV pool 0.15-1.0M tokens, different every run; 20 agents queue and re-prefill |
| 0.55 (0.4675 effective) | host memory grew from 415 to 501 GiB over the last 40 min of a 4 h run, then OOM (job 746293) |
| 0.588 (0.4998 effective) | heaviest stage at 513.2 of 513.5 GB: too close to OOM |
| `pp=2` | refuses at startup: `minimum viable = 0.7525` |

The pool has not been measured against a working set; if the hit rate falls under load, serve fewer
conversations, since the fraction has no headroom left.

One pipeline serves 20 agents (24 request slots and decode graphs; the fraction leaves no room for a larger
pool). `submit.sh` splits a setup of more problems into one job per 20 (`-p1`, `-p2`, ...; [LAUNCH.md](../../experiments/LAUNCH.md#sizing-agents-and-walltime)).
By hand, more pipelines serve in one job: `INFERENCE_NODES=8 INFERENCE_REPLICAS=2` runs two `pp=4` pipelines on consecutive
nodes, each answering on its first node, and the driver stripes the agents over them by problem index (`AGENTS_PER_NODE=40`
on one agent node: 20 per pipeline).

## Tuning record (2026-11-17)

Measured with `hpcagent-bench serve-bench` ([README](README.md#8-benchmarking-a-configuration-and-the-guards); the
`[bench.kimi-*]` tables of `experiments.toml` are the configurations): one `pp=4` server per job and an agent-shaped load
(per agent a 30k-token prefix, then turns of 1.5k new tokens and 400 output tokens, 15 min at 20 agents, then 15 min at
40). Server logs: `$SCRATCH/inference-server/<job>/`. Read this before running a new sweep; every row below has been
measured once. The pool floor below is `KV_TOKENS_PER_AGENT = 80000` x 20 agents = 1.6M tokens; 0.52 (1.85M) passes, 0.48
(0.15 to 1.0M) fails.

| Setting | Result | Job |
|---|---|---|
| fraction 0.48 (0.408 effective) | KV pool 0.15-1.0M tokens, different every run: PP3 holds about 165 GB of weights against a 167 GB budget. At 0.21M: cache hit 0%, TTFT p50 152 s, 1.9 tok/s per agent | 747586 |
| same, pool 0.98M, 48 slots | cache hit 88%, TTFT p50 10 s, 4.3 tok/s per agent, decode 5.8 tok/s | 747584 |
| fraction 0.55 | pool 3.7M, cache hit 98% at 20 agents; host memory ran out once after 4 h (746293) | 742817 |
| 40 agents on one pipeline (24 slots) | about 13 requests always queued; waiting conversations are evicted and re-prefilled: 46% cache hit, 133M fresh prefill tokens in one job | 742706 |
| `--max-running-requests` | SGLang splits it over the `pp` micro-batches (`pp_max_micro_batch_size = max_running_requests // pp_size`): 24 is 4 x 6 | source |
| HiCache `--hicache-size 32` | slower: see [`knobs.md`](knobs.md#hicache-never) | 747589 |
| `--attention-backend triton` | no aiter x0.85 on the fraction: 0.48 gives a 3.57M pool (more host memory used than aiter at 0.55) | 747585 |
| `--ep-size 4` (auto MoE runner) | `illegal memory access` at CUDA-graph capture. The dispatcher keeps global expert ids for the aiter runner, while the int4 MoE on ROCm runs the Triton runner (`CompressedTensorsWNA16TritonMoE`), which indexes local ids | 747587 |
| `--ep-size 4 --moe-runner-backend triton` | serves; API up in 1098 s against about 1490 s for plain `pp=4` | 747607 |
| `--enable-dp-attention --dp-size 4` (with `pp=4`, EP4) | needs fraction above 0.4874 x 0.85; at 0.55 it serves, then the first gathered prefill faults (`HSA_STATUS_ERROR_EXCEPTION`, `unspecified launch failure`) | 747588, 747608 |
| `tp16 ep16 dp-attention`, `pp=1` | needs fraction above 0.4874 at 0.85 (each rank keeps the attention weights); weights load in about 10 min | 747591 |
| **fraction 0.52 (the setting)** | pool 1.85M; 20 agents: 5.1 tok/s per agent, cache hit 90%, TTFT p50/p90 10/16 s. 40 agents: 105 tok/s total (2.6 per agent), hit 83% | 747672 |
| `--weight-loader-prefetch-checkpoints` (+ drop cache) | API up after 2435 s against 1789 s without: slower on this storage | 747673 |
| `--attention-backend triton` under load | the server's own warmup request timed out at 1800 s; no request answered | 747585 |
| `--ep-size 4 --moe-runner-backend triton`, pool 0.35M | decode 5.0 tok/s median against 7.4 for the aiter MoE at the same starved pool | 747607 |
| DP attention, any layout (`pp=4` or `pp=1`, mask of pad rows on or off) | `HSA_STATUS_ERROR_EXCEPTION` on the first real prefill in SGLang 0.5.20 | 747608, 747609, 747722 |
| `tp16 ep16`, `pp=1`, no DP attention, fraction 0.55 | serves; 20 agents: 3.4 tok/s per agent, hit 85%, TTFT p90 128 s. Slower than `pp=4` at 0.52; at 40 agents (fraction 0.48) 0.8 tok/s | 747833, 747610 |

Conclusion: `pp=4`, fraction 0.52, 20 agents per pipeline. More agents per server lose prefix hits whatever the
layout; split larger setups into separate jobs of 20.

## Rules

- Serve on SGLang (engine numbers are re-measured when agent tuning is done).
- Keep `--cuda-graph-max-bs-decode` at the live agent count plus slack (24 for 20 agents; graph capture
  takes memory after KV sizing; without the cap the failure looks like a fraction problem),
  `--max-running-requests` the same, `--page-size 64` (vendor `gfx942` recipe) and the
  fraction at or below 0.55. `--ep-size` above 1 needs `--moe-runner-backend triton` (`MOE_CHECKPOINT`, refused otherwise).
- Set `NCCL_NET_GDR_LEVEL=LOC`, keep PyNCCL on (about 20x slower decode without it), and confirm
  `grep -a "Using network" server-0.log` says `AWS Libfabric` ([`knobs.md`](knobs.md#multi-node-fabric)).
- Gate a `--kv-cache-dtype` change on long context: fp8 checkpoints ship no calibrated KV scales, and
  a broken attention path still answers short prompts (`containers/inference/accuracy-gate.py`, about
  10k tokens of varied filler).
- One client per endpoint; no HiCache ([`knobs.md`](knobs.md#hicache-never), measured slower); leave
  `SGLANG_ROCM_FUSED_DECODE_MLA` off. `AITER_USE_FLYDSL_MOE_SORTING=1` and `pp=4` are for this
  checkpoint only.
