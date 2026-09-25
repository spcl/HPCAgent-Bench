# Running an inference server on Beverin (AMD MI300A)

This folder is for people who want **only a model endpoint**: an OpenAI-compatible HTTP server on
Beverin compute nodes (AMD MI300A `gfx942` APUs, 4 GPUs per node, Slurm, CSCS Container Engine). No
benchmark, no grading, no agents. Numbers here do not carry to discrete GPUs, and several do not
carry to MI300X.

| Page | Covers |
|---|---|
| [`qwen38.md`](qwen38.md), [`kimi27sglang.md`](kimi27sglang.md), [`glm53.md`](glm53.md), [`oss120b.md`](oss120b.md) | one model each: configuration, DO / DO NOT, the numbers behind them |
| [`knobs.md`](knobs.md) | cross-model: APU memory model, KV pool threshold, aiter derate, HiCache, fabric, Slurm shape |
| [`private-endpoint.md`](private-endpoint.md) | a keyed Qwen3.8 server only you can use, from your laptop or your own Daint jobs |
| [`extending-private-inference.md`](extending-private-inference.md) | contributors: the private launcher's security contract, new presets, access paths, engines |

The authoritative launch line per model is the rendered `experiments/.env.base-<model>`. If a page
here and that file disagree, the file wins.

## 1. Shortest path

Once per account, register the EDFs (container definitions) and resolve your Slurm account:

```bash
cd "$REPO"
containers/images/install_edfs.sh          # renders into ~/.edf
sbatch containers/images/pull_images.sbatch  # only if install_edfs.sh reports a missing image
. scripts/cscs/account_env.sh                          # Beverin rejects jobs without an account
```

Then, from `experiments/`:

```bash
SUBMIT=0 ./serve-only.sbatch                  # print what would be submitted
./serve-only.sbatch                           # Qwen3.8 (default MODEL)
MODEL=kimi27sglang ./serve-only.sbatch        # any .env.base-<MODEL>: qwen38, kimi27sglang, glm53, oss120b
SBATCH_TIMELIMIT=08:00:00 ./serve-only.sbatch # longer than the 4 h default
```

`serve-only.sbatch` renders `.env.base-<MODEL>`, reads the node count from it, submits itself with
that `--nodes`, starts the server, polls `/v1/models`, then prints:

```
===== endpoint is live =====
base URL:   http://nid002968:8000/v1
model name: hpcagent-bench-vllm
...
The endpoint takes no API key. It stays up until this job ends; scancel <jobid> to stop it.
```

followed by a ready-to-paste `curl`. The job output is `serve-only-<jobid>.out`; server logs are
`$SCRATCH/inference-server/<jobid>/server-<rank>.log`, and `serve.env` there is the merged env that
actually ran.

**Runtime caveat.** `serve-only.sbatch` always launches through the CE (`srun --environment=`). Under
the CE, a single-node server (qwen38, oss120b) can fail building its tensor-parallel group with
`Failed to initialize any NET plugin`, and pyxis needs the site `ENROOT_CACHE_PATH` to be creatable.
The campaign launcher avoids both through `enroot`; see
[`experiments/README.md`](../../experiments/README.md#container-runtimes).

## 2. Images (EDFs)

An EDF is a TOML file in `~/.edf` naming the image, bind mounts, environment and CE hooks. The CE does
not reliably keep the image's own `ENV`, so the EDFs re-declare `PATH`, `LD_LIBRARY_PATH` and cache
dirs. The fabric comes from three pinned hooks (`netstack`, `cxi`, `aws_ofi_nccl`); without the RCCL
plugin, multi-node RCCL silently falls back to TCP.

| EDF | Engine | Models | Rendered by `install_edfs.sh` |
|---|---|---|---|
| `hpcagent-bench-sglang-mi300-latest` | SGLang 0.5.19 | Qwen3.8, Kimi K2.7 | yes |
| `hpcagent-bench-vllm-mi300-latest` | vLLM 0.23.0 | gpt-oss-120b | yes |
| `sglang-candidate` | SGLang | GLM-5.3 only | **no**; needs a rebuilt image, see [`glm53.md`](glm53.md) |

## 3. Slurm shape

`serve-only.sbatch` already sets all of this. Copy it if you write your own launcher.

| Setting | Why |
|---|---|
| `--partition=mi300` | default partition is `mi200`, different hardware |
| no `-A` | `scripts/cscs/account_env.sh` exports `SBATCH_ACCOUNT`; naming one yourself splits identical jobs across accounts |
| `--mem=0` | otherwise the step's memory cgroup follows its CPU share and the server dies in weight load |
| `--gpus-per-node=4`, `--ntasks-per-node=1` | every recipe is `tp=4` inside a node |
| `--cpus-per-task="${SLURM_CPUS_ON_NODE}"` on the server step | see below |
| `ulimit -c 0` | machine-global `core_pattern` drops multi-GB core files in the CWD; `scripts/checks/check_core_dumps.py` enforces it |

**The CPU trap.** `--exclusive` gives the job the node, not the step its CPUs. A step without
`--cpus-per-task` gets one core plus its SMT sibling (2 of 192). A starved server does not crash, it
degrades with load: 147 s per decode step after half an hour, against 88-91 tok/s for the same model
with `--cpus-per-task=32`. It can also hang in Triton JIT until the 600 s RCCL watchdog kills every
rank. Give a client or probe running alongside one socket: `--cpus-per-task=24 --hint=nomultithread`.

## 4. Talking to the endpoint

The server binds `0.0.0.0:8000` on the first node of the allocation. No gateway, **no API key**:
every Alps user can reach it. For a private server use [`private-endpoint.md`](private-endpoint.md).

```bash
squeue -u "$USER" -n serve-only -o '%i %T %N'
BASE=http://nid002968:8000
curl -s "$BASE/v1/models"
curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"hpcagent-bench-vllm","max_tokens":128,"messages":[{"role":"user","content":"Say hi."}]}'
curl -s "$BASE/metrics"      # Prometheus: throughput, running/waiting requests, cache hits
```

The served name is `hpcagent-bench-vllm` for every model, not the HuggingFace id.

- **Multi-node (Kimi K2.7, GLM-5.3, `pp=4`):** only rank 0 binds the port; talk to the first node.
- **Single-node models on several nodes:** independent replicas, one per hostname, each with its own
  KV cache. The client spreads load itself.

| Model | `MODEL=` | Engine | Nodes | Page |
|---|---|---|---|---|
| `Qwen/Qwen3.8-27B-FP8` | `qwen38` | SGLang | 1 | [`qwen38.md`](qwen38.md) |
| `moonshotai/Kimi-K2.7-Code` | `kimi27sglang` | SGLang | 4 (`pp=4`) | [`kimi27sglang.md`](kimi27sglang.md) |
| `zai-org/GLM-5.3` | `glm53` | SGLang | 4 (`pp=4`) | [`glm53.md`](glm53.md) |
| `openai/gpt-oss-120b` | `oss120b` | vLLM | 1 | [`oss120b.md`](oss120b.md) |

The engine is per model: Qwen3.8 on vLLM is about 19x slower than on SGLang; Kimi K2.7 on vLLM
collapses above concurrency 1.

## 5. Healthy or sick

**Readiness.** `serve-only.sbatch` waits `VLLM_READY_TIMEOUT_SECONDS`, else
`AGENT_READY_TIMEOUT_SECONDS`, else 7200 s. Each `.env.base-*` sets one high enough for that model
(GLM-5.3: 10800 s). Both keys are set inside the model file, so a value on the command line is
overwritten when the file is sourced. To override, copy `experiments/serve-only.env`, add the key,
and pass `SERVE_ENV_FILE=<copy>` (sourced last, so it wins). The same applies to
`VLLM_SERVED_MODEL`.

**SGLang startup, in order:**

```bash
grep -aE "Load weight (begin|end)|Cache is allocated|max_total_num_tokens|Capture|fired up" server-0.log
```

| Line | Meaning |
|---|---|
| `Load weight begin. avail mem=...` | free memory before weights (host memory on an APU) |
| `Load weight end. elapsed=... mem usage=...` | this rank's weight share |
| `Mamba Cache is allocated. max_mamba_cache_size: N` | Qwen3.8 state slots; below planned concurrency is a problem |
| `KV Cache is allocated. #tokens: N` / `max_total_num_tokens=N` | the whole prefix-cache budget: **the number that matters** |
| `Capture ... CUDA graph` | graph capture, takes memory after KV sizing |
| `The server is fired up` | answering HTTP |

Under load, a prefix-cache hit rate that **falls** as conversations grow means the KV pool is
thrashing ([`knobs.md`](knobs.md#the-kv-pool-threshold)). For vLLM, grep
`Loading|KV cache|Capturing|Application startup complete`; `Available KV cache memory` is the pool.

| Symptom | Cause | Fix |
|---|---|---|
| log stops after `Load weight begin`, no traceback | host OOM; KV cache is host memory | lower `--mem-fraction-static`, or more nodes |
| `minimum viable = ...` at startup | weights alone exceed the fraction at this node count | more nodes, not a bigger fraction |
| API never answers, log ends mid-JIT | CPU trap | `--cpus-per-task` |
| tool calls arrive as prose, or a 400 logged as success | only one parser named | pass `--reasoning-parser` **and** `--tool-call-parser` |
| RCCL watchdog abort after ~600 s | peer blocked: CPU trap or fabric fallback | check CPUs, then `Using network` |
| wrong numbers across nodes, no error | GPU-direct RDMA on this fabric | `NCCL_NET_GDR_LEVEL=0` |
| multi-node throughput several times low | RCCL on TCP | `grep -a "Using network" server-0.log` must say `AWS Libfabric` |

## 6. How configuration becomes flags

**Absent key vs empty key.** `run_cluster.sh` reads `${SGLANG_ATTENTION_BACKEND-aiter}` (dash, not
colon-dash):

| Env file says | `${VAR-default}` | `${VAR:-default}` |
|---|---|---|
| nothing | `default` | `default` |
| `VAR=` | empty: flag omitted | `default` |
| `VAR=x` | `x` | `x` |

Deleting a key turns the default **on**. To omit a flag, assign it empty (GLM-5.3 does this). When
templating a flag, use the dash form.

**Layers, last assignment wins.** `layers/common.env` < `layers/model-<m>.env` < `.env.base-<m>`
([Env layers](../../experiments/README.md#env-layers)); a layer can override a key, never unset it.
Render one with `./env_layers.sh render .env.base-<m>`. `serve-only.sbatch` sources the render, then
`serve-only.env` (zero judge and agent nodes, `RUN_ROOT`), under `set -a`.

**Arm `.env.<arm>` files are renders.** Fix the layer that owns a key, never the render.

**Mounts.** A campaign job narrows the inference container's mounts; `serve-only.sbatch` uses the
registered EDF as-is. A model that serves here and fails in a campaign run: suspect the mounts first.

## 7. Where the numbers live

- `experiments/serve-only.sbatch`, `experiments/serve-only.env`: the launcher on this page.
- `experiments/layers/model-<m>.env`, `experiments/.env.base-<m>`: per-model launch lines with inline reasons.
- `containers/inference/`: `smoke-kimi-sglang.sbatch` (serving smoke with accuracy
  gate and concurrency sweep), `agentlike-probe.py` (multi-stream load), `accuracy-gate.py`,
  `verify-tools-reasoning.py`.

Node-to-node spread is about 30%. Re-measure a flag change **on one node, back to back**.
