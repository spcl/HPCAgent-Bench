# Qwen3.8-27B (BF16) on an mi200 node, checked from anywhere

A keyed, OpenAI-compatible Qwen3.8-27B server on one Beverin mi200 node (8x MI250X), started with
`containers/inference/serve-private.sbatch` and checked with `containers/inference/ping-endpoint.sh`.
Security design, key handling and every launcher variable: [`private-endpoint.md`](private-endpoint.md).

| | |
|---|---|
| Image | `hpcagent-bench-vllm-mi200-latest`: the AMD vLLM 0.28 image ([`containers/images/vllm/`](../../containers/images/vllm/Dockerfile)), the one that serves oss120b on mi300 |
| Weights | `Qwen/Qwen3.8-27B`, BF16 (MI250X has no FP8), in `$HF_HOME/hub` |
| Shape | `tp8:0.85`: tensor parallel over all 8 GCDs, 0.85 of each GCD's 64 GiB (`--gpu-memory-utilization`) |
| Attention | `--attention-backend TRITON_ATTN` (3.4x the default backend; aiter has no gfx90a kernels, so it cannot run here) |
| Measured | 40 agents (the experiment load, below): 345 tok/s aggregate, 8.8 tok/s per agent, both gates pass; 128 concurrent requests fit |
| Binds | `127.0.0.1:30000` on the node; every `/v1` request needs the key (vLLM leaves `/health` and `/metrics` open, so this endpoint is tunnel-only) |

## Measured at the experiment load

40 concurrent agentic conversations, 16 turns each, 35k-50k-token prompts (p50 about 38k), 300-500 output
tokens per turn, on one node. Every row passed the long-context accuracy gate
and the tool-call gate.

| Configuration | tok/s aggregate | tok/s per agent | TTFT p50 / p90 | ITL p50 | prefix-cache hit |
|---|---|---|---|---|---|
| `--attention-backend TRITON_ATTN` (shipped) | 345.2 | 8.78 | 1.4 s / 10.6 s | 108 ms | 0.86 |
| default backend, tp8 | 101.7-107.3 | 2.70-2.90 | 2.6 s / 31-41 s | 328-344 ms | 0.85-0.87 |
| tp4 x dp2 | 94.6 | 2.88 | 3.6 s / 85 s | 334 ms | 0.82 |
| tp2 x dp4 | 94.7 | 2.67 | 4.3 s / 47 s | 322 ms | 0.68 |
| `--max-num-batched-tokens 8192` | 89.8 | 2.50 | 2.4 s / 19 s | 391 ms | 0.88 |
| MTP speculative decoding (1 token) | 24.7 | 0.60 | 29 s / 62 s | 1589 ms | 0.87 |

Data parallelism splits each conversation across replicas, so the prefix cache stops hitting; one tp8
replica is the shape. vLLM logs one fallback of its own here ("Falling back to the Triton GDN decode path:
`fused_gdn_decode_post_conv_mtp` is not built"): the Gated DeltaNet decode runs Triton on gfx90a.

aiter cannot run on gfx90a: it ships asm kernels for gfx942, gfx950 and gfx1250 only, its
`get_device_name()` raises for gfx90a, and vLLM refuses a forced `ROCM_AITER_FA` with "compute capability not
supported" (668579). So mi200 is the one AMD platform that serves qwen3.8 on vLLM rather than SGLang (the
mi300 engine; SGLang's `sgl_kernel` in the current image carries gfx942 code only and segfaults here).

## 1. One-time setup

On Beverin, a key only you can read:

```bash
umask 077; mkdir -p ~/.config/hpcagent-bench
openssl rand -hex 32 > ~/.config/hpcagent-bench/mi200-endpoint.key
```

The weights must already be in `$HF_HOME/hub` (`helpers/scripts/cache_env.sh` sets `HF_HOME`; fetch with
`containers/inference/fetch_weights.sbatch`).

## 2. Start the server

From a checkout on Beverin:

```bash
PRESET=mi200 MODE=serve sbatch --partition=mi200 --gpus-per-node=8 --time=08:00:00 \
    containers/inference/serve-private.sbatch
```

Loading takes a few minutes. The job output ends with `private endpoint is live: 127.0.0.1:30000 on
<node>` once a request without the key has returned 401 and one with it 200. `squeue -j <jobid> -o %N`
names the node too.

## 3. Check it (the ping test)

One command, from wherever you are. It runs on the serving node over ssh, so nothing listens beyond
`127.0.0.1` and the key never leaves your home directory:

```bash
containers/inference/ping-endpoint.sh <node> mi200                                  # Beverin login node
JUMP=beverin.alps.cscs.ch containers/inference/ping-endpoint.sh <node> mi200        # ela, Daint login
JUMP=ela.cscs.ch,beverin.alps.cscs.ch containers/inference/ping-endpoint.sh <node> mi200   # laptop
```

It checks that `/health` answers 200, that a request without the key answers 401, and that a chat
completion with the key answers 200 with text; then it prints the completion and its tokens per
second. Exit 0 means the endpoint works:

```text
health                     200 (want 200)
POST without the key       401 (want 401)
chat completion with key   200 (want 200)
completion                 63 tokens in 1.00 s = 62.7 tok/s
A stencil kernel is ...
```

The jump hosts are the public names: `beverin.alps.cscs.ch` does not resolve from inside Beverin, and
compute node names (`nid...`) resolve only from Beverin. `PORT` selects another leg,
`KEY_FILE` another key file on the node.

## 4. Use it

Over a tunnel from your laptop or ela (section 4 of [`private-endpoint.md`](private-endpoint.md)):

```bash
ssh -N -J ela.cscs.ch,beverin.alps.cscs.ch -L 127.0.0.1:30000:127.0.0.1:30000 <node>
```

then any OpenAI client at `http://127.0.0.1:30000/v1`, model `hpcagent-bench-vllm`, with the header
`Authorization: Bearer <key>`. Your own Daint jobs connect without ssh when the server starts with
`ACCESS=alps` (section 5 there).

## 5. Stop

`scancel <jobid>`. The launcher deletes the key copies and the config it wrote.
