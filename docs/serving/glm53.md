# Serving GLM-5.3 on MI300A

`zai-org/GLM-5.3`, fp8, about 755 GB of weights, on SGLang across four nodes (`tp=4`, `pp=4`).
Source of truth: `experiments/layers/model-glm53.env` plus `SGLANG_EXTRA_ARGS` in
`experiments/.env.base-glm53`; render with `experiments/env_layers.sh render .env.base-glm53`.
Background: [`knobs.md`](knobs.md).

```bash
cd experiments && MODEL=glm53 ./serve-only.sbatch
```

**Image prerequisite.** GLM-5.3 needs the `sglang-candidate` EDF, and that EDF currently has no
image: `install_edfs.sh` does not render it. Rebuild the sglang role
(`containers/images/sglang/build.sbatch`, output `hpcagent-bench-sglang-candidate.sqsh`),
then render the EDF. See "Known traps" in [`SUBMITTING.md`](../../SUBMITTING.md#known-traps).
The image must bake in a guard keeping `torch.Tensor.format_ue8m0` false and
`HIPCC_COMPILE_FLAGS_APPEND=-U__HIP_NO_HALF_CONVERSIONS__ -U__HIP_NO_HALF_OPERATORS__`. The other
sglang EDFs reach that patch through a `PYTHONPATH` under `$SCRATCH`, which the inference role's
mount policy drops, so they fail to load the model.

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

## Memory

`--mem-fraction-static` is a ceiling on weights plus KV, so the pool is what the fraction leaves over
the weights:

```
pool(f) = 39.0M * (f - 0.4838) tokens      (tp4 x pp4)
```

| `f` | Outcome |
|---|---|
| below 0.486 | refuses: weights alone exceed the budget |
| 0.50 | 632,384-token pool |
| 0.55 | 2.58 M pool; OOM-killed on a PP node at concurrency 20 |
| **0.57 + `--max-total-tokens 2800000`** | pool pinned at 2.8 M; served concurrency 40, peak 485 of 501 GiB step cgroup |
| 0.62 | OOM killer takes the heaviest stage |

Host memory, not the pool, is what kills a stage. `--chunked-prefill-size 4096` and
`--max-running-requests 48` bound the prefill buffers and `req_to_token` that grow with load.
Stage weights are uneven (172.4 / 197.2 / 203.8 / 206.1 GB): size against the heaviest and read
`avail mem=` on every rank.

## Startup

Time to a live API = slowest stage's weight load (up to 5400 s) + about 130 s KV allocation + about
1080 s graph capture. `AGENT_READY_TIMEOUT_SECONDS=10800` in the layer covers it; judge readiness by
the slowest stage, never the first to report.

## DO

- **Assign `SGLANG_ATTENTION_BACKEND=` empty.** An absent key makes `run_cluster.sh` append
  `--attention-backend aiter` ([README](README.md#6-how-configuration-becomes-flags)).
- **Allocate four nodes.** At `pp=2` each stage holds about 378 GB of the roughly 412 GB free.
- **Keep `SGLANG_USE_AITER=1`.** Without aiter ops the ROCm DSA path forces `page_size` 1.
- **Pass both parsers**, `glm45` and `glm47`; the mismatched versions are correct.
- **Give an accuracy gate 2048 tokens.** The model reasons first; below about 512 the gate reports
  truncation as corruption.
- **Read back every launch:**
  ```bash
  grep -aiE "attention.backend|Use dsa attention" server-0.log
  grep -a "max_total_num_tokens\|KV Cache is allocated\|Using network" server-0.log
  ```
  `Using network` must say `AWS Libfabric`.

## DO NOT

- **Do not pass `--attention-backend`.** Any explicit value suppresses `dsa`; `aiter` also derates
  the fraction by 0.85 (0.588 reads back as 0.4998).
- **Do not template the key with `${VAR:-default}`**; `:-` reverts the deliberate empty value.
- **Do not set `SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION`** to escape the derate; that reserve is
  workspace long-context serving needs.
- **Do not pass `--language-only`.** It selects the vision-encoder receiver role, and this
  architecture is off its allowlist: the server refuses to start.
- **Do not enable HiCache** ([`knobs.md`](knobs.md#hicache-never)).
- **Do not carry `AITER_USE_FLYDSL_MOE_SORTING` over from Kimi K2.7**; those weights are int4, these fp8.
- **Do not change `--kv-cache-dtype` on a short accuracy check.** No calibrated KV scales ship with
  the checkpoint; gate on long context.
- **Do not decide a KV knob from a cold smoke**; use per-stream distinct prefixes re-sent across
  rounds and report the hit rate ([`knobs.md`](knobs.md#the-kv-pool-threshold)).
- **Do not serve GLM-5.3-Flash.** Its `index_kpool` forces `IndexerKPool`, which is CUDA-only.
