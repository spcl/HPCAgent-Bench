# Serving Qwen3.8 on MI300A

`Qwen/Qwen3.8-27B-FP8` on SGLang, one node, `tp=4`. The cheapest useful endpoint here. Source of
truth: `experiment:qwen38` in `experiments/setups.yaml` over `layers/model-qwen38.env`. Background:
[`knobs.md`](knobs.md).

```bash
hpcagent_bench/cluster/serve-only.sbatch     # qwen38 is the default MODEL
```

## Configuration

EDF `hpcagent-bench-sglang-mi300-latest`; env `SGLANG_USE_AITER=1`, `SGLANG_SET_CPU_AFFINITY=0`.
`run_cluster.sh` adds `--tp-size 4 --host 0.0.0.0 --port 8000 --served-model-name hpcagent-bench-vllm`.

```
--chat-template containers/inference/chat-template-qwen38.jinja --trust-remote-code
--attention-backend aiter --kv-cache-dtype fp8_e4m3 --fp8-gemm-backend triton --language-only
--watchdog-timeout 1800
--context-length 262144 --mem-fraction-static 0.306 --mamba-full-memory-ratio 0.25
--max-running-requests 128 --enable-metrics
--reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-cache-report
```

| Mamba ratio | KV pool | Mamba slots | Captured decode batch | Used by |
|---|---|---|---|---|
| **0.25** | **4.03 M tokens** | 427 | up to 85 | `experiment:qwen38`, `scicomp:qwen38`, `serve-only.sbatch` |
| 0.5 | 3.32-3.36 M tokens | 704 | full 128 | `llrbase-c:qwen38`, private `mi300` preset |
| 0.9 (engine default) | 1.91 M tokens | | | never: hit rate 0.20, 25 tok/s at 40 x 60k |

Keep the pool above the working set (concurrent conversations x largest prompt): crossing it is worth
about 8x (40 x 60k on 3.32 M: hit 0.984, 254 tok/s; on 2.42 M: hit 0.35-0.45, 30-34 tok/s). Live
Mamba slot peak is 315-322, so 427 covers 40 streams. Read slots and pool from the log
(`grep -a "Mamba Cache is allocated\|max_total_num_tokens" server-0.log`), never infer them.

## Measured: 40 agents, one node

Load: 40 long-lived conversations, 16 turns, 12k shared + ~22k unique prefix, prompts 35k-50k (p50
~38k), 300-500 output tokens per turn; one server per leg, back to back on one node, tool and
long-context gates on every leg.

These legs run the Gated-DeltaNet `ba` projection on torch's GEMM (aiter has no gfx942 row for it; 700-930
log lines), which is the shipped configuration ([knobs.md](knobs.md#aiter-correctness-first-fallbacks-accepted)).

| Leg | out tok/s | tok/s per agent | TTFT p50/p90 s | ITL p50 ms | gates |
|---|---|---|---|---|---|
| base, 3 nodes | 241-272 | 6.5-7.4 | 1.2-2.7 / 7.8-19 | 130-146 | pass |
| `--kv-cache-dtype fp8_e4m3` | 284.9, 283.5 | 7.84, 7.72 | 1.2 / 8.6-10 | 123-125 | pass |

- **fp8 KV: +11% aggregate, +18% per agent** against the same node's base and base2 (257.5/256.6,
  drift under 1%), reproduced on a second node; gates pass. Shipped: it doubles the KV pool, which the
  prompt study's ~115k-token conversations need (40 x 115k is past the 4.03 M bf16 pool).
- Rejected: MTP/NEXTN 3-4 (-36%), `--mamba-full-memory-ratio 0.5` (233 vs 241-255), `--schedule-policy
  lpm` (inside the 6% same-node drift), `--tp-size 2 --dp-size 2` (78 tok/s: DP round-robin splits a
  conversation, hit 0.48), `--mamba-radix-cache-strategy no_buffer --disable-overlap-schedule` (132 tok/s,
  hit 0.79; ReplaySSM needs it), `--mamba-ssm-dtype bfloat16` (263, within noise).

## Measured: decode, one replica (10-29)

Distinct random prompts, steady-state decode while every stream decodes (`jobs/qwen-serve-bench/decode_bench.py`).

| Leg (aggregate tok/s) | 1 stream, 1k | 8 streams, 1k | 20 streams, 1k | 20 streams, 100k |
|---|---|---|---|---|
| bf16 KV, aiter FP8 GEMM | | | | 297 |
| fp8 KV, aiter FP8 GEMM | 41 | 169 | 403 | 341 |
| fp8 KV, `--fp8-gemm-backend triton` | 66 | 410 | 1078 | 725 |

- On gfx942 SGLang sends every block-scaled FP8 linear to aiter's Triton `gemm_a8w8_blockscale` (the CK path is
  gfx950 only); a decode step was ~44 ms of GPU time with the GPU otherwise idle. SGLang's own Triton block GEMM
  is 2.7x at 20 streams (2.1x at 100k context). Shipped.
- Context length barely moves the step (1k and 100k within 15%): attention is not the decode cost.
- No gain: `--enable-aiter-allreduce-fusion`, triton attention with 16 KV splits. Worse: MTP (EAGLE 3/1/4, -60% at
  8-20 streams), TP2.
- Gates (serve-only PREWARM=1, job 704847): tool-call, reasoning and ACCURACY 9/9 pass with the Triton GEMM.
- An experiment with two replicas pins each agent node to its own replica (`run_cluster.sh`), so a conversation
  never splits across prefix caches.

## Rules

- Serve on SGLang on MI300A. vLLM with `ROCM_AITER_FA` garbled qwen3.8 under agent load (random tokens
  in the reasoning, invented tool names), so qwen3.8 has no MI300A vLLM line. MI250X serves it on
  vLLM with `TRITON_ATTN` (`layers/hardware-mi200-qwen38.env`).
- Move `--mem-fraction-static` together with `--mamba-full-memory-ratio` (with ratio `r` the budget
  `R` splits mamba `R*r/(1+r)`, KV `R/(1+r)`) and with `--attention-backend aiter` (aiter derates by
  0.85, so 0.306 is an effective 0.26). Never above 0.306: the OOM killer takes the process without a
  traceback.
- Name `--attention-backend aiter` (ROCm default is triton; aiter is about +5%). Pass the chat template
  and both parsers.
- `--page-size 64`, `--chunked-prefill-size 16384` and HiCache do not help
  ([`knobs.md`](knobs.md#hicache-never)). `--language-only` selects SGLang's vision-encoder receiver
  role, not "text only".
- Measure with 40 long-lived streams at your p50 and p90 prompt sizes, back to back on one node; never
  trust a cold smoke ([`knobs.md`](knobs.md#the-kv-pool-threshold)).

## GH200 (Daint)

Served on Daint (TP4, vLLM): how to run it is in [`qwen38-gh200.md`](qwen38-gh200.md).
`Qwen/Qwen3.8-27B-FP8` (about 28.5 GB) fits one 96 GB GPU;
native context is 262144 tokens. Parsers: `--reasoning-parser qwen3`, plus `--tool-call-parser qwen3_coder`
on SGLang or `--enable-auto-tool-choice --tool-call-parser qwen3_xml` on vLLM. A CUDA 13 image needs
`com.hooks.aws_ofi_nccl.variant = "cuda13"` ([IMAGE_REQUIREMENTS.md](../../containers/images/IMAGE_REQUIREMENTS.md)).

Starting point for one node (an estimate): TP4 on one port (`run_cluster.sh` serves one port per node),
SGLang with `--kv-cache-dtype fp8_e4m3 --attention-backend flashinfer --mm-attention-backend triton_attn
--max-running-requests 128` (the aarch64 `sgl-kernel` has no FA3, FA4 crashes on head_dim 256, and the
vision tower imports FA3 unless `--mm-attention-backend` is set). Measure with 40 long-lived
conversations at 35k-50k prompts before any setup uses it.
