# Running an inference server on Beverin (AMD MI300A)

This folder is for people who want **only a model endpoint** on these nodes: an OpenAI-compatible
HTTP server their own client can talk to. No benchmark, no grading, no agents. You do not need to
understand the rest of this repository to use it.

This page is the entry point: how to start a server, find it, talk to it, and tell a healthy one
from a sick one. Then:

- **one page per model** -- [`qwen38.md`](qwen38.md), [`kimi27sglang.md`](kimi27sglang.md),
  [`glm53.md`](glm53.md), [`oss120b.md`](oss120b.md). Each is self-contained: best known
  configuration, what to do, what not to do, and the measurements behind both. If you only care
  about one model, that is the only other file you need.
- [`knobs.md`](knobs.md) -- what is genuinely cross-model: the APU memory model, the KV pool
  threshold, the aiter derate, HiCache, the multi-node fabric and the Slurm shape.

Measurements carry the date they were taken, because a serving number ages.

Everything here was measured on **Beverin**: AMD MI300A (`gfx942`) APU nodes, 4 GPUs per node,
Slurm, CSCS Container Engine. Numbers do not carry to a discrete-GPU cluster; several of them do
not even carry to MI300X.

## 1. The shortest path

```bash
cd experiments
SUBMIT=0 ./serve-only.sbatch        # see what it would do
./serve-only.sbatch                 # start a Qwen3.8 server
```

`serve-only.sbatch` reads a model's configuration file, works out how many nodes that model needs,
submits itself with that node count, starts the server, waits for it to answer, and then prints the
endpoint URL and a ready-to-paste `curl`. Watch the job's output file for that block:

```
===== endpoint is live =====
base URL:   http://nid002968:8000/v1
model name: optarena-vllm
health:     curl -s http://nid002968:8000/v1/models
server log: /capstor/scratch/cscs/<you>/x86_64/inference-server/<jobid>/server-0.log
```

Pick another model with `MODEL=`:

```bash
MODEL=kimi27sglang ./serve-only.sbatch
MODEL=oss120b      ./serve-only.sbatch
MODEL=glm53        ./serve-only.sbatch
```

The name after `MODEL=` is the suffix of a file in `experiments/`: `MODEL=qwen38` reads
`.env.base-qwen38`. Those are the same files the benchmark campaigns serve from, so the endpoint
you get is the endpoint they get. The launcher layers `experiments/.env.serve-only` on top, which
does one thing: sets the judge and agent node counts to zero.

The server stays up until the job's wall clock expires (4 h by default, `--time` to change it) or
until you `scancel` it.

## 2. What a container environment is here, and which one to use

Beverin runs jobs through the **CSCS Container Engine (CE)**. You do not run `docker` or
`podman`; you add `--environment=<name>` to an `srun` and Slurm starts your command inside a
container image.

The `<name>` is an **EDF** -- an Environment Definition File, a small TOML file in `~/.edf/`. It
names the image (a SquashFS file on scratch), the host directories to bind-mount, and a block of
environment variables the image needs but cannot set for itself. Two things about EDFs surprise
people:

- **The CE does not reliably preserve the image's own `ENV`.** That is why the EDFs here re-declare
  `PATH`, `LD_LIBRARY_PATH` and cache directories absolutely. Do not assume a variable baked into
  the Dockerfile arrives.
- **The network fabric comes from "hooks", not from the image.** The `[annotations]` block enables
  three of them: `netstack` (the pinned Slingshot software stack), `cxi` (the Cassini provider and
  the `/dev/cxi*` devices) and `aws_ofi_nccl` (the plugin that lets RCCL use Slingshot). With the
  plugin missing, RCCL silently falls back to TCP: numerically correct, several times slower, no
  error message. The EDFs in this repo pin the hook version so a site-side upgrade cannot change
  the fabric under a running job.

`ls ~/.edf` shows what is registered for you. The ones that matter:

| EDF | Engine | Use it for |
|---|---|---|
| `sglang-latest` | SGLang | Qwen3.8, Kimi K2.7 |
| `sglang-glm-halfconv` | SGLang | GLM-5.3 only |
| `vllm-latest` | vLLM | gpt-oss-120b |

`sglang-glm-halfconv` points at the **same image** as `sglang-latest`. It differs by exactly two
environment variables, both needed to get GLM-5.3 to load at all -- see the per-model recipes
below. There is no separate GLM build.

If `~/.edf` is empty, `containers/cluster/ce-images/install_edfs.sh` registers the repo's copies
against the images named in `containers/cluster/ce-images/images.env`.

## 3. Submitting: the Slurm flags, and why each one

```
#SBATCH --partition=mi300
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
```

- **`--partition=mi300` always.** The default partition is `mi200`. That is different hardware and
  the configurations here are not valid on it.
- **Do not pass `--account`.** The default association works; naming an account is how identical
  jobs end up split across two project accounts depending on which command line was typed.
- **`--mem=0`.** A step's memory cgroup is sized from its share of the node's CPUs. Without
  `--mem=0` the server is capped far below the node's memory and dies during weight load with no
  useful message.
- **`--gpus-per-node=4`.** All four GPUs; tensor parallelism is 4 in every recipe here.
- **`ulimit -c 0` in the script.** Beverin's `core_pattern` is machine-global and a crash drops a
  zero-byte `core_nid<node>_<pid>` stub into the process's working directory. Slurm propagates the
  limit to job steps, so setting it once at the top of the batch script is enough.

### The CPU trap: set `--cpus-per-task` explicitly

This one is worth its own heading because it is silent and it has cost real runs.

A node has 192 logical CPUs (4 sockets x 24 physical cores x 2 SMT threads). `--exclusive` gives
the **job** the node; it does **not** give a **step** the node's CPUs. An `srun` step that does not
say `--cpus-per-task` gets **one** core plus its SMT sibling -- two CPUs out of 192 -- and every
process in the step shares them.

An inference server does a great deal of host-side work: scheduling, KV block management,
prefix-cache hashing, detokenization, sampling. Starved of CPU it does not crash; it **degrades
with load**. The observed shape is a server that looks fine for ten minutes and then spends 147 s
per decode step with nothing queued, nothing preempted, and a 99% prefix-cache hit rate. The same
model on the same four nodes with `--cpus-per-task=32` served at 88-91 tok/s.

It can also hang outright: a serving step on 2 CPUs across 2 nodes never finished JIT-compiling a
Triton kernel, its peer blocked in a pipeline send, and the 600 s RCCL watchdog aborted every rank.

So: **one task per node, and that task takes the whole node**:

```bash
srun --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_ON_NODE}" ...
```

`serve-only.sbatch` does this for you. If you write your own launcher, copy that line first.

The same bug class hits any step you run *alongside* the server (a benchmark client, a probe). Give
those one socket's physical cores -- `--cpus-per-task=24 --hint=nomultithread` -- enough not to be
the bottleneck, not so much that the measurement contends with what it is measuring.

## 4. Finding the endpoint and talking to it

The server binds `0.0.0.0` on port 8000 of its node. There is no gateway and no proxy: the URL is
the compute node's hostname.

```bash
squeue -u "$USER" -n serve-only -o '%i %T %N'      # job id, state, node list
```

`serve-only.sbatch` prints the URL once the API answers. To find it by hand, the first node of the
allocation is the one that serves:

```bash
scontrol show hostnames "$(squeue -j <jobid> -h -o '%N')" | head -1
```

There is **no API key**. Any OpenAI-compatible client works against `http://<node>:8000/v1`; pass a
dummy key if your client insists on one.

```bash
BASE=http://nid002968:8000

curl -s "$BASE/v1/models"

curl -s "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"optarena-vllm","max_tokens":128,
       "messages":[{"role":"user","content":"Say hi."}]}'
```

The model name in the request body is the **served name**, not the HuggingFace repo id. Every
recipe here serves under `optarena-vllm`; `/v1/models` tells you for certain. Change it with
`VLLM_SERVED_MODEL` if a client hard-codes something else.

Prometheus metrics are at `$BASE/metrics` (both engines, because every recipe passes
`--enable-metrics`). They are the cheapest way to watch a live server: token throughput, running
and waiting request counts, and cache hit rate.

### Multi-node servers

Kimi K2.7 and GLM-5.3 are split across 4 nodes with pipeline parallelism. Only **rank 0 binds the
HTTP port**; the other three are members of its pipeline and answer nothing. Always talk to the
first node of the allocation.

Qwen3.8 and gpt-oss-120b fit in one node. If you allocate several, you get **independent replicas**
rather than one bigger server -- each binds port 8000 on its own hostname and holds its own KV
cache. That multiplies throughput but does not raise the ceiling for a single conversation, and a
client must spread its requests itself.

## 5. The model pages

One page per model. Each leads with the best configuration we currently know, then a list of what to
DO and a list of what NOT to do with the reason and the measurement behind each, then the data those
instructions rest on.

| Model | `MODEL=` | Engine | Nodes | Page |
|---|---|---|---|---|
| Qwen3.8 (`Qwen/Qwen3.8-27B-FP8`) | `qwen38` | SGLang | 1 | [`qwen38.md`](qwen38.md) |
| Kimi K2.7 (`moonshotai/Kimi-K2.7-Code`) | `kimi27sglang` | SGLang | 4 (`pp=4`) | [`kimi27sglang.md`](kimi27sglang.md) |
| GLM-5.3 (`zai-org/GLM-5.3`) | `glm53` | SGLang | 4 (`pp=4`) | [`glm53.md`](glm53.md) |
| gpt-oss-120b (`openai/gpt-oss-120b`) | `oss120b` | vLLM | 1 | [`oss120b.md`](oss120b.md) |

**The engine is a per-model decision and the wrong one is expensive.** Qwen3.8 on vLLM is roughly
19x slower than on SGLang; Kimi K2.7 on vLLM collapses above concurrency 1. gpt-oss-120b is the one
model here served by vLLM. Each page has the numbers.

[`knobs.md`](knobs.md) holds only what is genuinely cross-model: the APU memory model, the KV pool
threshold, the aiter derate, HiCache, the fabric and the Slurm shape. Anything measured on one model
lives on that model's page, and where two models disagree, both pages say so.

## 6. Healthy or sick: what to read in the log

Each node writes `server-<rank>.log` in the run directory the launcher prints. Below, `grep -a`
because these logs contain progress bars and other binary noise.

### SGLang, a healthy startup, in order

```bash
grep -aE "Load weight (begin|end)|Cache is allocated|Memory pool end|Capture|fired up" server-0.log
```

You want to see all of these, and you want the numbers to be sane:

| Line | What it tells you |
|---|---|
| `Load weight begin. avail mem=405.02 GB` | free memory **before** the weights. On an APU this is host memory. |
| `Load weight end. elapsed=878 s, ... mem usage=167.32 GB` | weights actually landed, and how big this rank's share is. |
| `Mamba Cache is allocated. max_mamba_cache_size: 514, ...` | state-cache slots (Qwen3.8 only). A number **below** the concurrency you plan to run is a problem. |
| `KV Cache is allocated. ... #tokens: 2426200, K size: 18.51 GB` | **the number that matters.** This is your whole prefix-cache budget in tokens. |
| `max_total_num_tokens=2426200` | the same figure, restated. |
| `Capture target decode CUDA graph begin. ... bs=[1, 2, ... 64]` | graph capture. It consumes memory *after* the KV cache is sized. |
| `The server is fired up` | it will now answer HTTP. |

Under load, SGLang prints a running status line. Read `token usage` (fraction of the KV pool in
use) and, with `--enable-cache-report`, the prefix-cache hit rate. A hit rate that **falls** as a
conversation grows is the pool thrashing and is the single most useful sickness signal on this
hardware -- see `knobs.md` section on the KV pool.

### The failures you will actually hit

| Symptom | Cause | Fix |
|---|---|---|
| Log stops after `Load weight begin`, job exits non-zero, no traceback | host OOM. On an APU the KV cache is host memory. | lower `--mem-fraction-static`, or add nodes so each pipeline stage holds less |
| `minimum viable = 0.75...` at startup, server refuses | `--mem-fraction-static` is below what the weights alone need at this node count | raise the **node count**, not the fraction |
| API never answers, log ends mid-JIT | the CPU trap: step running on 2 CPUs | set `--cpus-per-task` |
| Server answers, but tool calls come back as prose | only one of the two parsers named | pass **both** `--reasoning-parser` and `--tool-call-parser` |
| 400 on the first request, logged upstream as success | same as above | same as above |
| RCCL watchdog abort after ~600 s, every rank | a peer blocked; usually the CPU trap or a fabric fallback | check `--cpus-per-task`, then `NET/Plugin` below |
| Wrong numbers across nodes, no error | GPU-direct RDMA over this fabric | `NCCL_NET_GDR_LEVEL=0` |
| Throughput several times lower than expected on a multi-node job | RCCL fell back to TCP | `grep -a "NET/Plugin\|Using network" server-0.log`; you want `Using network AWS Libfabric`, not a "Could not find libnccl-net.so" line |

### vLLM

`grep -aE "Loading|KV cache|Capturing|Application startup complete" server-0.log`. The same shape:
weights, then a KV cache size, then graph capture, then the HTTP server. The `Available KV cache
memory` line plays the role that `KV Cache is allocated` plays in SGLang.

## 7. Mechanism: how a configuration key becomes a command-line flag

Three pieces of machinery here behave in ways a reader outside the project will not guess. Each has
cost a real run.

### An ABSENT key takes the default. Only an EMPTY ASSIGNED key suppresses the flag.

The launcher builds some flags from shell parameter expansion, and it deliberately uses
`${VAR-default}` rather than `${VAR:-default}`:

```bash
backend="${SGLANG_ATTENTION_BACKEND-aiter}"     # dash, not colon-dash
[[ -n "${backend}" ]] && command+=(--attention-backend "${backend}")
```

The two forms differ only on the empty string, and that difference is the whole point:

| Configuration file says | `${VAR-default}` gives | `${VAR:-default}` gives |
|---|---|---|
| nothing at all (key absent) | `default` | `default` |
| `VAR=` (assigned, empty) | **empty, so the flag is omitted** | `default` |
| `VAR=x` | `x` | `x` |

So **deleting a key does not turn a flag off.** It turns the default on. To omit a flag you must
assign the key and leave it empty, which looks like a mistake and is not. A model that must choose
its own attention backend needs `SGLANG_ATTENTION_BACKEND=` in its file; with the key simply absent
it gets `--attention-backend aiter` appended, which is the opposite of the intent.

This is not hypothetical in both directions. Writing `${VAR:-default}` where `${VAR-default}` was
meant re-enabled a flag that had just been removed, and killed two launches after the fix that was
supposed to prevent exactly that. If you template a flag, use the dash form.

### Configuration files are layered, last assignment wins

`serve-only.sbatch` sources the model's file first and `experiments/.env.serve-only` second, under
`set -a`, so every value is exported and a later assignment overrides an earlier one. The override
file sets the judge and agent node counts to zero and redirects the run root; everything else comes
from the model's own file, unchanged. That is what keeps this documentation and a real deployment
from drifting apart.

The job writes the merged result to `serve.env` in its run directory. That file is the record of
what actually ran; read it rather than re-deriving the layering by hand.

### A GENERATED configuration freezes the decision it was generated from

Some models' per-run configurations are produced by a generator script from a base file. When a
setting has to change, it must change in the base **and** in the generator. Change one and the two
disagree, and which value a server gets depends on which file it was launched from.

This is live, not theoretical. On 2026-09-11 GLM-5.3's base file named one container image while its
generator named another; both were checked in and both were plausible. The two have since been
converged. The rule that follows: **never fix a setting in a generated file** -- it will be
regenerated over -- and when you find a base and a generator disagreeing, treat it as the bug rather
than picking the value you prefer.

### The campaign launcher narrows container mounts; this one does not

When a benchmark run starts a server, it rewrites the container definition to mount only what the
server needs, because other roles in that job must not see the graded material. An inference-only
job has no other roles, so `serve-only.sbatch` uses the registered EDF as-is, with its wider mounts.

The difference is usually invisible and once was not: a model whose loader patch arrives through a
path under `/capstor` loads fine under the wide mounts and dies under the narrow ones. If a model
serves for you here and fails inside a benchmark run, suspect the mounts before the model.

## 8. Where the real numbers live

- the model pages above -- per-model instructions with their evidence.
- [`knobs.md`](knobs.md) -- the cross-model knobs, dated.
- `experiments/serve-only.sbatch` and `experiments/.env.serve-only` -- the launcher this page
  describes. The first is the job; the second is the three-line override that removes the
  benchmark roles from a model's own configuration.
- `experiments/.env.base-<model>` -- the authoritative launch line per model, with its own inline
  reasons. If this folder and one of those files disagree, that file wins.
- `containers/cluster/ce-images/inference/` -- the serving smokes and probes these numbers come
  from: `smoke-kimi-sglang.sbatch` (a serving smoke with an accuracy gate and a concurrency sweep),
  `agentlike-probe.py` (throughput under a realistic multi-stream load) and `accuracy-gate.py`.

If you change a serving flag and want to believe the result, re-measure **on one node, back to
back**. Node-to-node spread on this machine is about 30%, which is larger than most of the effects
worth chasing.
