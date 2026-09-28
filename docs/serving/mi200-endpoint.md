# Qwen3.8-27B (BF16) on an mi200 node, checked from anywhere

A keyed, OpenAI-compatible Qwen3.8-27B server on one Beverin mi200 node (8x MI250X), started with
`containers/inference/serve-private.sbatch` and checked with `containers/inference/ping-endpoint.sh`.
Security design, key handling and every launcher variable: [`private-endpoint.md`](private-endpoint.md).

| | |
|---|---|
| Image | `hpcagent-bench-vllm-mi200-latest`: the AMD vLLM 0.28 image ([`containers/images/vllm/`](../../containers/images/vllm/Dockerfile)), the one that serves oss120b on mi300 |
| Weights | `Qwen/Qwen3.8-27B`, BF16 (MI250X has no FP8), in `$HF_HOME/hub` |
| Shape | `tp8:0.85`: tensor parallel over all 8 GCDs, 0.85 of each GCD's 64 GiB (`--gpu-memory-utilization`) |
| Measured | 128 concurrent requests, 421 tok/s on 16 concurrent 256-token requests, about 26 tok/s for one request |
| Binds | `127.0.0.1:30000` on the node; every `/v1` request needs the key (vLLM leaves `/health` and `/metrics` open, so this endpoint is tunnel-only) |

## 1. One-time setup

On Beverin, a key only you can read:

```bash
umask 077; mkdir -p ~/.config/hpcagent-bench
openssl rand -hex 32 > ~/.config/hpcagent-bench/mi200-endpoint.key
```

The weights must already be in `$HF_HOME/hub` (`scripts/cache_env.sh` sets `HF_HOME`; fetch with
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
completion                 56 tokens in 1.59 s = 35.1 tok/s
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
