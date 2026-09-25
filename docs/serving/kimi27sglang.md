# Serving Kimi K2.7 on MI300A

`moonshotai/Kimi-K2.7-Code` on SGLang, four nodes, `tp=4` per node and `pp=4` across them. The model
does not fit one node, so four nodes is the only way to serve it. Source of truth:
`experiments/layers/model-kimi27sglang.env` (rendered by `.env.base-kimi27sglang`). Background:
[`knobs.md`](knobs.md).

```bash
cd experiments && MODEL=kimi27sglang ./serve-only.sbatch
```

## Configuration

EDF `hpcagent-bench-sglang-mi300-latest`. Port 8000 on rank 0 only, served name
`hpcagent-bench-vllm`. Weight load takes 30-40 minutes before the API answers.

`run_cluster.sh` adds `--tp-size 4 --pp-size 4 --nnodes 4 --node-rank <r> --dist-init-addr <rank0>:29500
--host 0.0.0.0 --port 8000 --attention-backend aiter`. The env adds:

```
--trust-remote-code --language-only --watchdog-timeout 1800
--kv-cache-dtype fp8_e4m3 --page-size 64 --context-length 262144
--mem-fraction-static 0.55 --cuda-graph-max-bs-decode 64
--enable-metrics --pre-warm-nccl
--reasoning-parser kimi_k2 --tool-call-parser kimi_k2 --enable-cache-report
```

Environment: `SGLANG_USE_AITER=1`, `SGLANG_ROCM_FUSED_DECODE_MLA=0`, `SGLANG_SET_CPU_AFFINITY=0`,
`NCCL_NET_GDR_LEVEL=0`, `AITER_USE_FLYDSL_MOE_SORTING=1`, `AITER_LOG_TUNED_CONFIG=1`.

## Memory budget at `pp=4`

| Quantity | Value |
|---|---|
| Free per rank before weight load | about 412 GB |
| Weights per stage | about 171 GB |
| `--mem-fraction-static` 0.55 | 0.4675 effective (aiter x0.85); throughput-neutral across 0.408-0.4675 effective |
| 0.588 (0.4998 effective) | heaviest stage at 513.2 of 513.5 GB host memory: too close to OOM |
| `pp=2` | refuses at startup: `minimum viable = 0.7525` |

The crossing of this model's pool against its working set has not been measured. Read the hit rate
under your own load; if it falls, serve fewer concurrent conversations, since the fraction has no
headroom left.

## DO

- **Serve on SGLang.** Aggregate tok/s at concurrency 1 / 2 / 4 / 6: SGLang 13.7 / 17.6 / 38.1 /
  46.8; vLLM 20.6 / 6.4 / 7.0 / 6.6, with 42-43% of vLLM samples generating nothing.
- **Allocate four nodes.** On `minimum viable = ...`, add nodes.
- **Keep `--cuda-graph-max-bs-decode 64`.** Graph capture takes memory after the KV cache is sized;
  without the cap the residual pushes the effective fraction under the startup floor.
- **Keep `--page-size 64`.** Vendor `gfx942` recipe; avoids decode faults at `page_size=1`.
- **Keep `--context-length 262144`.** 256k costs nothing measurable against a shorter window.
- **Set `NCCL_NET_GDR_LEVEL=0`** and confirm `grep -a "Using network" server-0.log` says
  `AWS Libfabric` before trusting any number.
- **Read `avail mem=` on every rank.** Stages are unequal; a change that fits rank 0 can OOM rank 3.
- **Gate a `--kv-cache-dtype` change on long context.** fp8 checkpoints ship no calibrated KV scales
  (runtime scale 1.0), and a broken attention path still answers short prompts.
  `containers/inference/accuracy-gate.py` asks at about 10k tokens of varied filler;
  repeated filler causes an echo attractor at temperature 0.

## DO NOT

- **Do not serve on vLLM.** See above.
- **Do not raise `--mem-fraction-static` above 0.55.**
- **Do not drop `--cuda-graph-max-bs-decode`** when changing the fraction; the failure looks like a
  fraction problem.
- **Do not disable PyNCCL.** About 20x slower decode on this shape ([`knobs.md`](knobs.md#multi-node-fabric)).
- **Do not point several clients at one endpoint.** Four clients did less useful work than one.
- **Do not enable HiCache** ([`knobs.md`](knobs.md#hicache-never)).
- **Do not turn on `SGLANG_ROCM_FUSED_DECODE_MLA`.** Off per the vendor recipe; its kernel matches an
  fp32 reference (max abs diff 6e-4 at 1k KV, 7e-4 at 4k), which is not evidence that serving with
  it is correct.
- **Do not carry `AITER_USE_FLYDSL_MOE_SORTING=1` or `pp=4` to another model.** The first is for
  this checkpoint's pack-quantized int4 weights; the second costs about 42% of engine time to stalls
  and only pays off when the model does not fit one node.
- **Do not call the server wedged before `KV Cache is allocated`.** 30-40 minutes of weight load is
  normal; `VLLM_READY_TIMEOUT_SECONDS=3600` covers it.
