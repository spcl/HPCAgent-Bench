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
--attention-backend aiter --language-only --watchdog-timeout 1800
--context-length 262144 --mem-fraction-static 0.306 --mamba-full-memory-ratio 0.25
--max-running-requests 128 --enable-metrics
--reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-cache-report
```

| Mamba ratio | KV pool | Mamba slots | Captured decode batch | Used by |
|---|---|---|---|---|
| **0.25** | **4.03 M tokens** | 427 | up to 85 | `experiment:qwen38`, `serve-only.sbatch` |
| 0.5 | 3.32-3.36 M tokens | 704 | full 128 | `llrbase-c:qwen38`, private `mi300` preset |
| 0.9 (engine default) | 1.91 M tokens | | | never: hit rate 0.20, 25 tok/s at 40 x 60k |

Keep the pool above the working set (concurrent conversations x largest prompt): crossing it is worth
about 8x (40 x 60k on 3.32 M: hit 0.984, 254 tok/s; on 2.42 M: hit 0.35-0.45, 30-34 tok/s). Live
Mamba slot peak is 315-322, so 427 covers 40 streams. Read slots and pool from the log
(`grep -a "Mamba Cache is allocated\|max_total_num_tokens" server-0.log`), never infer them.

## Measured 2026-10-05: 40 agents, one node (pre-tuning controls)

Load: 40 long-lived conversations, 16 turns, 12k shared + ~22k unique prefix, prompts 35k-50k (p50
~38k), 300-500 output tokens per turn; one server per leg, back to back on one node, tool and
long-context gates on every leg. Harness and raw numbers:
`$SCRATCH/hpcagent-bench-runs/inference-tuning-20261005/` (`bench.sbatch`, `summarize.py`).

These legs run the Gated-DeltaNet `ba` projection on torch's GEMM (aiter has no gfx942 row for it; 700-930
log lines), which is the shipped configuration ([knobs.md](knobs.md#aiter-correctness-first-fallbacks-accepted)).

| Leg (job) | out tok/s | tok/s per agent | TTFT p50/p90 s | ITL p50 ms | gates |
|---|---|---|---|---|---|
| base, 3 nodes (668269/668270/668364/668567) | 241-272 | 6.5-7.4 | 1.2-2.7 / 7.8-19 | 130-146 | pass |
| `--kv-cache-dtype fp8_e4m3` (668269, 668567) | 284.9, 283.5 | 7.84, 7.72 | 1.2 / 8.6-10 | 123-125 | pass |

- **fp8 KV: +11% aggregate, +18% per agent** against the same node's base and base2 (257.5/256.6,
  drift under 1%), reproduced on a second node; gates pass. Candidate for the shipped line.
- Rejected: MTP/NEXTN 3-4 (-36%), `--mamba-full-memory-ratio 0.5` (233 vs 241-255), `--schedule-policy
  lpm` (inside the 6% same-node drift), `--tp-size 2 --dp-size 2` (78 tok/s: DP round-robin splits a
  conversation, hit 0.48), `--mamba-radix-cache-strategy no_buffer --disable-overlap-schedule` (132 tok/s,
  hit 0.79; ReplaySSM needs it), `--mamba-ssm-dtype bfloat16` (263, within noise).

## Measured 2026-10-06: vLLM against SGLang, current images

Same load and harness as above (40 agents, `agentic-c40.json`, tool and long-context gates per leg),
one node per job, `$SCRATCH/hpcagent-bench-runs/inference-tuning-20261005/` (`q38eng-*.legs`,
`q38abba-mi300.legs`). SGLang runs the `experiment:qwen38` line with a tuned bf16 GEMM table that has since been removed
(it selected inexact kernels; [knobs.md](knobs.md#aiter-correctness-first-fallbacks-accepted)). vLLM 0.28.0 (`hpcagent-bench-vllm-mi300-latest`) runs the
same FP8 checkpoint, chat template and parsers with `VLLM_ROCM_USE_AITER=1 --max-model-len 262144
--gpu-memory-utilization 0.70 --max-num-seqs 128 --language-model-only --enable-prefix-caching
--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`.

| Leg (job, node) | out tok/s | tok/s per agent | TTFT p50/p90 s | ITL p50/p90 ms | hit | gates | fallback lines |
|---|---|---|---|---|---|---|---|
| SGLang base / base2 (669808) | 283.6 / 292.9 | 7.93 / 7.83 | 1.0 / 12.5, 0.9 / 9.5 | 123 / 161 | 0.87 | pass | 0 |
| SGLang `--kv-cache-dtype fp8_e4m3` (669808) | 310.3 | 8.38 | 0.9 / 8.4 | 116 / 144 | 0.87 | pass | 0 |
| vLLM `ROCM_AITER_FA` / repeat (669809) | 541.6 / 472.3 | 13.44 / 12.86 | 1.4 / 7.4, 1.3 / 9.6 | 69-70 / 76-117 | 0.86 | pass | 3 |
| vLLM `TRITON_ATTN` (669809) | 300.3 | 7.59 | 2.0 / 14.3 | 121 / 149 | 0.87 | pass | 3 |
| vLLM `ROCM_AITER_FA --kv-cache-dtype fp8` (669809) | 548.5 | 13.60 | 1.6 / 9.0 | 69 / 76 | 0.85 | pass | 3 |
| **same node, A-B-B-A (669893)**: SGLang fp8 KV | 306.1, 302.7 | 8.18, 8.23 | 1.0 / 9.3, 1.0 / 8.6 | 117 / 150 | 0.87 | pass | 0 |
| **same node, A-B-B-A (669893)**: vLLM `ROCM_AITER_FA` | 542.4, 537.7 | 13.57, 13.21 | 1.3 / 8.8 | 69 / 76-80 | 0.86 | pass | 3 |

- **vLLM is the faster engine: 1.77x SGLang's best leg on one node** (540 vs 304 tok/s, drift under
  1.5% within each engine), from decode: ITL 69 ms against 117 ms at equal hit rate, with TTFT about
  equal. `ROCM_AITER_FA` is 1.8x vLLM's `TRITON_ATTN`; fp8 KV adds nothing on vLLM (548 vs 542) and
  9% on SGLang.
- vLLM's three fallback lines, every leg: the Gated-DeltaNet decode runs the Triton kernel
  (`fused_gdn_decode_post_conv_mtp is not built`, the MTP-fused variant), the aiter sampler hands
  requests that carry a seed to PyTorch, and an unused GELU-tanh. vLLM also logs about 5300 aiter
  `a8w8_blockscale` shapes with no tuned row, which run aiter's default kernel, not torch.
- Not yet run on vLLM: MTP, `--max-num-batched-tokens`, the agent harness end to end.

## Rules

- Serve on vLLM with `--attention-backend ROCM_AITER_FA` and `VLLM_ROCM_USE_AITER=1`: 1.77x SGLang's
  best leg on the same node (below).
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

## GH200 (Daint): published recipes, not yet measured here

No Qwen3.8 GH200 number on this page comes from this repo; nothing here has served it on Daint.
`containers/images/vllm-cuda` (vLLM 0.28.0) gates the `qwen3` / `qwen3_coder` parsers but has no
qwen38 serving layer. The engine ranking above is a ROCm result and does not carry to
CUDA. Collected 2026-10-05 (registry digests read from the registries, nothing pulled).

**Model.** `Qwen/Qwen3.8-27B-FP8` (fine-grained FP8, block 128, about 28.5 GB) or `Qwen/Qwen3.8-27B`
(BF16, about 54 GB). The architecture is `Qwen3_5ForConditionalGeneration`, a VL model with 48 Gated
DeltaNet layers, 16 attention layers and one in-checkpoint MTP layer. Native context is 262144 tokens.
Parsers: `--reasoning-parser qwen3`, plus `--tool-call-parser qwen3_coder` on SGLang or
`--enable-auto-tool-choice --tool-call-parser qwen3_xml` on vLLM (the vLLM recipe's choice, and
`llrbase-c:qwen38`'s). The model card's thinking-mode sampling is `temperature=1.0 top_p=0.95 top_k=20`.
Both FP8 and BF16 fit one 96 GB GH200 GPU.

**Images** (all ship an arm64 manifest):

| Image | Engine | CUDA | Pushed | Digest |
|---|---|---|---|---|
| `vllm/vllm-openai:v0.28.0-aarch64-cu129` (our `vllm-cuda` base) | vLLM 0.28.0 | 12.9 | 2026-08-26 | `sha256:60fa2715937e604931086a790fff2978c09995eff93439261ba09a79f02e9e68` |
| `vllm/vllm-openai:v0.31.0-aarch64-cu129` | vLLM 0.31.0 (latest release) | 12.9 | 2026-10-04 | `sha256:ff7c43dcb2059e8f3c3235a09b0963cc241c3a05806377f79e089ea2a8eac194` |
| `vllm/vllm-openai:v0.31.0-aarch64` | vLLM 0.31.0 | 13 | 2026-10-04 | `sha256:3f7dd5b777d34d1724456ce71f87385dca288c3bb23029ab27dee358f5d2b971` |
| `lmsysorg/sglang:v0.5.21` (multi-arch index) | SGLang 0.5.21 (latest) | 13.0 | 2026-10-01 | `sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469` |
| `lmsysorg/sglang:v0.5.19-cu129` (newest CUDA 12.9 build) | SGLang 0.5.19 | 12.9 | 2026-09-04 | `sha256:59e11312666e1c5c155210ea335589b91daa0d70848521b390b93b1b1e8fb0ef` |
| `lmsysorg/sglang:qwen38-27b-cu129` (model-pinned) | SGLang | 12.9 | 2026-08-14 | `sha256:e35dfb0beaf6b1fb6619ae0dac9474b5cdda24b81cee7202316e371301425e46` |
| `nvcr.io/nvidia/vllm:26.09-py3` (index; arm64 `sha256:fa68ef92...`) | vLLM 0.29.0 | 13.4.1 | 2026-09-23 | `sha256:557747337846eecf34486866f3d992383c6f8af869fdb0e365b5093fcaee02c2` |
| `nvcr.io/nvidia/sglang:26.09-py3` (index; arm64 `sha256:5f92f380...`) | SGLang 0.5.19 | 13.4.1 | 2026-09-17 | `sha256:edb36119c3c7568692cff32e8c1240cabea9810be2664671c6f88d1ae654b514` |

The NGC pair shares CUDA 13.4.1 with `judge-agent-cuda`'s NGC PyTorch 26.09 base. A CUDA 13 image
needs `com.hooks.aws_ofi_nccl.variant = "cuda13"`, while `vllm-cuda` uses `"cuda12"`
([IMAGE_REQUIREMENTS.md](../../containers/images/IMAGE_REQUIREMENTS.md)). On Clariden (GH200),
swiss-ai's `model-launch` builds its own CUDA 13.0 images from source (`images/sglang_0.5.20`,
`images/vllm_cuda13`) with `variant = "cuda13"` and both Alps scratch filesystems mounted.

**Published launch lines.**

- swiss-ai `model-launch`, Clariden GH200, one node, SGLang 0.5.20, gate passed 2026-09-24
  ([recipe](https://github.com/swiss-ai/model-launch/blob/main/mfa_examples/clariden/Qwen/Qwen3.8-27B/sglang/Qwen3.8-27B-sglang.sh)):
  `--tp-size 4 --context-length 131072 --reasoning-parser qwen3 --tool-call-parser qwen3_coder
  --attention-backend flashinfer --mm-attention-backend triton_attn`. The aarch64 `sgl-kernel` ships
  without FA3, which is SGLang's Hopper default, and FA4 crashes on this model's head_dim 256. The
  vision tower imports FA3 unless `--mm-attention-backend` is set.
- `model-launch`, Clariden, vLLM nightly (CUDA 13), gate passed 2026-09-22
  ([recipe](https://github.com/swiss-ai/model-launch/blob/main/mfa_examples/clariden/Qwen/Qwen3.8-27B/vllm/Qwen3.8-27B-vllm.sh)):
  `--tensor-parallel-size 4 --max-model-len 131072 --max-num-seqs 16 --gpu-memory-utilization 0.85
  --reasoning-parser qwen3`. It names no tool parser, and `max-num-seqs 16` is too low for 40 agents.
- SGLang cookbook, H200 (SM90), one GPU, FP8
  ([cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx)):
  `--kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 --attention-backend flashinfer
  --chunked-prefill-size 32768 --max-prefill-tokens 32768 --reasoning-parser qwen3
  --tool-call-parser qwen3_coder`. MTP: `--speculative-algorithm EAGLE --speculative-num-steps 3
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`. Size `--mamba-full-memory-ratio` as
  `(S + D) x state_bytes / (L x kv_bytes_per_token)`: S state slots per request (no_buffer 3,
  extra_buffer 5), D = 4 with MTP, a 153.9 MB fp32 state slot, 32.8 KB per token of fp8 KV, and L the
  average request length.
- vLLM recipe ([recipes.vllm.ai](https://recipes.vllm.ai/Qwen/Qwen3.8-27B), updated 2026-10-02):
  `vllm serve Qwen/Qwen3.8-27B-FP8 --tensor-parallel-size 4 --max-model-len 262144 --kv-cache-dtype fp8
  --reasoning-parser qwen3`. MTP: `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`.
  The recipe has no Hopper-specific line.

**Published speed numbers.** None for GH200 or for an agentic long-context load. model-launch's
Clariden `benchmarks/summary.txt` says "perf: skipped". The nearest numbers:

| Source | Hardware, engine | Load | Result |
|---|---|---|---|
| [g factor on dev.to](https://dev.to/g_factor/benchmarking-qwen-38-27b-across-inference-providers-together-fireworks-doubleword-and-g-factor-4c1i) | 2x H100 SXM, vLLM, FP8, DP2 | ~564 input / 128 output tokens | aggregate 69 tok/s at c1, 459 at c8, 1601 at c32, 2462 at c64 |
| [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) | 2x RTX 5090, FP8 | 262K context | MTP acceptance 0.771 |
| [SGLang cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx) | SM120 card, FP8 + EAGLE | ISL 8192 / OSL 1024, c1 | 106.3 (fp32 state) vs 116.1 (bf16 state) tok/s per user |

**Starting point for one GH200 node: an estimate, unmeasured.** Each GPU holds the FP8 weights with
about 50 GB left for KV and state at 0.85. Four TP1 replicas with sticky agents need no all-reduce
and give each replica a full prefix cache. This repo has no in-node replica router yet:
`run_cluster.sh` serves one port per node. Until it does, use TP4 on one port, with the cookbook's
flags plus `--attention-backend flashinfer --mm-attention-backend triton_attn` on SGLang (or the
vLLM recipe line), `--max-running-requests` / `--max-num-seqs` 128, and MTP as the first candidate
to measure. Measure it with a multi-turn load (40 long-lived conversations, 35k-50k prompts) before any setup uses it.
