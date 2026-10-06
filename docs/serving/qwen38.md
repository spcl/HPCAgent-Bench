# Serving Qwen3.8 on MI300A

`Qwen/Qwen3.8-27B-FP8` on vLLM, one node, `tp=4`. The cheapest useful endpoint here. Source of
truth: `layers/model-qwen38.env` (rendered by `experiment:qwen38` in `experiments/setups.yaml`).
Background: [`knobs.md`](knobs.md).

```bash
hpcagent_bench/cluster/serve-only.sbatch     # qwen38 is the default MODEL
```

## Configuration

EDF `hpcagent-bench-vllm-mi300-latest` (vLLM 0.28.0); env `VLLM_ROCM_USE_AITER=1`. `run_cluster.sh`
adds `--tensor-parallel-size 4 --host 0.0.0.0 --port 8000 --served-model-name hpcagent-bench-vllm`.

```
--dtype auto --attention-backend ROCM_AITER_FA
--chat-template containers/inference/chat-template-qwen38.jinja --trust-remote-code
--max-model-len 262144 --gpu-memory-utilization 0.70 --max-num-seqs 128
--language-model-only --enable-prefix-caching
--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

On mi200 (MI250X, BF16) `layers/hardware-mi200-qwen38.env` serves the BF16 checkpoint on the same
vLLM image with `--attention-backend TRITON_ATTN` and `VLLM_ROCM_USE_AITER=0`: aiter's attention has
no gfx90a kernels.

## vLLM against SGLang, measured 2026-10-06

Load: 40 long-lived conversations (`agentic-c40.json`), 16 turns, 12k shared + ~22k unique prefix,
prompts 35k-50k (p50 ~38k), 300-500 output tokens per turn; one server per leg, back to back on one
node, tool and long-context gates on every leg. Harness and raw numbers:
`$SCRATCH/hpcagent-bench-runs/inference-tuning-20261005/` (`bench.sbatch`, `summarize.py`,
`q38eng-*.legs`, `q38abba-mi300.legs`). SGLang 0.5.20 ran its best line: `--attention-backend aiter`,
`SGLANG_USE_AITER=1`, `--mem-fraction-static 0.306 --mamba-full-memory-ratio 0.25` (a 4.03 M-token KV
pool) and `--kv-cache-dtype fp8_e4m3`.

| Leg (job) | out tok/s | tok/s per agent | TTFT p50/p90 s | ITL p50/p90 ms | hit | gates | fallback lines |
|---|---|---|---|---|---|---|---|
| SGLang, bf16 KV / repeat (669808) | 283.6 / 292.9 | 7.93 / 7.83 | 1.0 / 12.5, 0.9 / 9.5 | 123 / 161 | 0.87 | pass | 0 |
| SGLang, fp8 KV (669808) | 310.3 | 8.38 | 0.9 / 8.4 | 116 / 144 | 0.87 | pass | 0 |
| vLLM `ROCM_AITER_FA` / repeat (669809) | 541.6 / 472.3 | 13.44 / 12.86 | 1.4 / 7.4, 1.3 / 9.6 | 69-70 / 76-117 | 0.86 | pass | 3 |
| vLLM `TRITON_ATTN` (669809) | 300.3 | 7.59 | 2.0 / 14.3 | 121 / 149 | 0.87 | pass | 3 |
| vLLM `ROCM_AITER_FA`, fp8 KV (669809) | 548.5 | 13.60 | 1.6 / 9.0 | 69 / 76 | 0.85 | pass | 3 |
| **same node, A-B-B-A (669893)**: SGLang fp8 KV | 306.1, 302.7 | 8.18, 8.23 | 1.0 / 9.3, 1.0 / 8.6 | 117 / 150 | 0.87 | pass | 0 |
| **same node, A-B-B-A (669893)**: vLLM `ROCM_AITER_FA` | 542.4, 537.7 | 13.57, 13.21 | 1.3 / 8.8 | 69 / 76-80 | 0.86 | pass | 3 |

- **vLLM is 1.77x SGLang's best line on one node** (540 vs 304 tok/s, under 1.5% drift within each
  engine), all of it decode: ITL 69 ms against 117 ms at equal hit rate, TTFT about equal.
  `ROCM_AITER_FA` is 1.8x vLLM's `TRITON_ATTN`; fp8 KV adds nothing on vLLM (548 vs 542).
- Rejected on SGLang (2026-10-05, same load): MTP/NEXTN 3-4 (-36%), `--mamba-full-memory-ratio 0.5`,
  `--schedule-policy lpm`, `--tp-size 2 --dp-size 2` (DP round-robin splits a conversation, hit 0.48),
  `--mamba-radix-cache-strategy no_buffer`, `--mamba-ssm-dtype bfloat16`.
- Not yet run on vLLM: MTP, `--max-num-batched-tokens`, the agent harness end to end.

**vLLM's three fallback lines.** None is an aiter-to-torch GEMM or attention fallback.

| Line | What runs instead | Fixable here |
|---|---|---|
| `Falling back to the Triton GDN decode path: torch.ops._C.fused_gdn_decode_post_conv_mtp is not built` | vLLM's Triton Gated-DeltaNet decode, the kernel SGLang also runs | no: the fused op is CUDA C++ that the ROCm wheel does not build; it would need a HIP port of vLLM's `csrc` kernel |
| `aiter sampler does not support per-request generators; falling back to PyTorch-native` | logged once, at the startup sampler warmup, which passes a generator; a request without `seed` samples on aiter | only by not sending `seed` (our harnesses do not) |
| `[ROCm] PyTorch's native GELU with tanh approximation is unstable. Falling back to GELU(approximate='none')` | the vision tower's GELU, built and never run under `--language-model-only` | not needed: the language model uses SiLU |

vLLM also logs about 5300 aiter `a8w8_blockscale` shapes without a tuned row; they run aiter's default
blockscale kernel, not torch.

## Rules

- Serve on vLLM with `--attention-backend ROCM_AITER_FA` and `VLLM_ROCM_USE_AITER=1`; pass the chat
  template and both parsers.
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
