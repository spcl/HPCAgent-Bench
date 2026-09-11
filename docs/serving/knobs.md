# Cross-model serving knobs on MI300A

What is on this page applies to **every** model served on these nodes. Anything whose number was
measured on one model lives on that model's page instead:

| Model | Page |
|---|---|
| Qwen3.8 | [`qwen38.md`](qwen38.md) |
| Kimi K2.7 | [`kimi27sglang.md`](kimi27sglang.md) |
| GLM-5.3 | [`glm53.md`](glm53.md) |
| gpt-oss-120b | [`oss120b.md`](oss120b.md) |

A knob measured on two models with different answers appears on both pages with both numbers. It is
not averaged into one claim here.

**Every number carries the date it was measured.** A serving number ages: the engine, the ROCm
build and the image all move. A number from three months ago is a hypothesis, not a fact. Re-measure
before you build a decision on anything below that is more than a couple of months old. Where a
claim survives only as a comment in a configuration file, with no measurement artefact left on
disk, this documentation says so.

**How to read a measurement.** Node-to-node throughput spread on Beverin is about **30%** (measured
2026-09-04), so a single-shot A/B across two nodes cannot resolve anything under about 1.3x. Every
comparison in this folder that quotes a ratio was run **back to back on one node**. Throughput
numbers come from a load of 20 or 40 long-lived streams that re-send a growing conversation, because
that is the only regime in which the cache knobs are visible at all: a cold one-shot smoke reports
them as null.

---

## The MI300A memory model -- read this before any knob

MI300A is an **APU**. The GPUs and the host share one physical pool of memory. Three consequences
that break the intuition from discrete GPUs:

1. **`--mem-fraction-static` is accounted node-wide.** SGLang sees `is_integrated` and sizes its
   static budget against the whole node's memory, per rank. The usable range is nothing like the
   0.8-0.9 that upstream documentation suggests, and the safe value is different for every model
   and every node count. Do not "fix" a low-looking value upward. vLLM's `--gpu-memory-utilization`
   is the same knob under another name and carries the same caveat.
2. **The fraction is a floor as well as a ceiling.** It bounds weights **and** KV cache together,
   and it applies to free memory measured *before* the weights load. So the minimum viable value is
   set by the weights, and at pipeline depth `pp` each stage holds `1/pp` of them. Halve the node
   count and each stage holds twice as much; the floor can then exceed the value you set, and
   SGLang refuses at startup with a "minimum viable = ..." message. **The fix for that message is
   more nodes, not a bigger fraction** (measured 2026-09-02). Per-model floors and the exact values
   in use are on the model pages.
3. **Spilling KV to "host memory" spills it into the same pool.** See HiCache below. There is no
   second tier here.

Because the budget is shared, an out-of-memory death arrives as the host OOM killer taking the
process: the log stops, there is no Python traceback, and the exit status is a signal. Do not look
for a stack trace that does not exist; look at the last `avail mem=` line.

---

## The KV pool threshold: pool divided by working set, near 1.15

**This is the mechanism behind every cache knob in this folder. Read it before tuning any of them.**

The prefix cache does not degrade gradually as the KV pool shrinks. It is a **threshold** on one
ratio:

```
pool / working set        pool         = max_total_num_tokens, printed at startup
                          working set  = the prompt tokens of all concurrent conversations,
                                         at their largest
```

**Above about 1.15, every configuration lands at a prefix-cache hit rate of 0.984 to 0.988. Below
it, every configuration thrashes.** Measured on Qwen3.8 across 14 leg-concurrency cells with no
exceptions (2026-09-11).

Three things follow, and they matter more than any individual flag:

1. **Raising `--mem-fraction-static` buys nothing when you are already above the threshold, and up
   to 10x when you are below it.** That is one mechanism, not two regimes. On Qwen3.8 the same knob
   change was worth nothing at 20 streams (already above) and about 10x at 40 streams (below);
   the largest measured gap on that model was 25 tok/s against 271.
2. **A cold one-shot smoke cannot see any cache knob.** Such probes sit above the threshold at every
   setting, because a single short prompt has almost no working set. This is why three separate
   models measured every memory and backend knob as null on a cold smoke: the measurement was taken
   entirely on the flat side of the threshold. Load 20 or more long-lived streams that re-send a
   growing conversation, or you are measuring nothing.
3. **Tune the ratio, not the knob.** Two different flags that land the same pool size land the same
   throughput. Which flag you move is a question of what else it costs.

**How to find where you are.** The pool is printed once at startup:

```bash
grep -a "max_total_num_tokens\|KV Cache is allocated" server-0.log
```

The working set is your own load: concurrent conversations times their largest prompt. Estimate it
before you tune, then aim the pool at about 1.3x it to leave margin. Each model page states where
that model's shipped configuration sits.

**Watch it live** with `--enable-cache-report`: a hit rate that **falls** as conversations grow is
the ratio crossing below the threshold in front of you.

---

## `--mem-fraction-static` and `--attention-backend aiter` are one decision

With the **aiter** attention backend, the engine multiplies `--mem-fraction-static` by **0.85**
internally before using it. A configured 0.588 is an effective 0.50; a configured 0.247 is an
effective 0.21.

The derate is conditional. `arg_groups/attention_hook.py` applies it only when
`attention_backend == "aiter"` **and** `context_len > 8192`. On any other backend, or at a short
context, the configured fraction is the effective fraction.

**Never move one of these two without the other.** Dropping the backend while leaving the number
alone leaves a KV pool too small to hold a working set. Raising the number while dropping the
backend overshoots into the host OOM killer.

Read the resulting pool from the allocator's own `KV size: X GB` line. Never infer it from the
flag: which backend carries the number decides whether the 0.85 applies, so one flag value means
two different pools.

**`SGLANG_USE_AITER=1` does not select the attention backend.** It switches aiter **ops**, and it
is worth keeping on: without it the ROCm path loses aiter's preshuffled paged-MQA kernel and forces
`page_size` to 1 whatever the flag says.

**The default backend is per-model, not per-platform.** With `--attention-backend` unset, a model's
own override picks it: kimi gets aiter, GLM-5.3 gets `dsa` from
`arg_groups/model_overrides/deepseek_v2.py`. There is no single ROCm default to reason from.

**Naming the backend explicitly is not always the safe choice.** An explicit value SUPPRESSES the
model's own override, which is how GLM-5.3 loses `dsa`. Omit the flag where the model selects
correctly for itself; name it only where the model's own choice is wrong, and check the value
against `python3 -m sglang.launch_server --help` **inside the image** first, since an unrecognised
value is an argparse error that takes down every rank at launch.

Confirm what the engine actually chose by reading `attention_backend=` back out of the server log.

Which backend a given model should use, and what it measured, is on that model's page. It is not the
same answer for every model.

---

## HiCache is wrong on this hardware

**Do not set `--enable-hierarchical-cache` or `--hicache-ratio`.**

Elsewhere, hierarchical caching mirrors the KV cache into host RAM so evicted prefixes can be pulled
back instead of recomputed -- trading cheap host memory for expensive device memory. On an APU,
host memory **is** the pool the KV cache and the weights already live in. `--hicache-ratio 2.0`
therefore offloads nothing: it allocates a second copy of the KV cache in the same physical memory,
so every cached token costs twice. The server dies to the host OOM killer with no traceback
(measured 2026-09-03).

Nothing in the repository's current configurations sets it. Older launch lines that do -- including
one snapshot in `containers/cluster/ce-images/IMAGE_REQUIREMENTS.md` -- are superseded.

---

## Both parsers, always

Name **`--reasoning-parser` and `--tool-call-parser` together**, for every model. This is the single
most common way to get a server that looks healthy and is not.

Name only one and the server starts normally. Then the first request that uses the other feature
fails -- in some builds with a 400 that the calling client records as a success, in others with the
model's tool call arriving as prose describing the call rather than a structured `tool_calls` entry.
Nothing in the server log says "parser". Observed repeatedly since 2026-08-20.

**Verify rather than assume.** Send one request carrying a tool schema and assert that
`choices[0].message.tool_calls[0]` exists and that `reasoning_content` is non-empty. A server that
answers with prose *about* the call passes every throughput check ever written.
`containers/cluster/ce-images/inference/verify-tools-reasoning.py` does exactly this.

The parser names are per model and are listed on each model's page.

---

## Topology

**Use pipeline parallelism only when the model does not fit in one node.**

**`pp=4` costs about 42% of engine time to stalls** (measured 2026-08-30). For a model that already
fits in a node, splitting it adds a network hop per token and buys nothing. For a model that does
not fit, that cost is the price of serving it at all.

**Several nodes serving a model that fits gives you independent replicas, not one bigger server.**
Each replica binds the same port on its own hostname and holds its own KV cache. That multiplies
aggregate throughput and does nothing for a single conversation; the client must spread load itself.

**One client per server beats several** on the large models (measured 2026-08-28): four clients
against one Kimi K2.7 endpoint produced less useful work than one, because they compete for the same
KV pool and evict each other's prefixes. If you have four nodes and four users, four one-node
servers beat one four-node server -- when the model fits.

---

## Multi-node fabric

These apply only to a server split across nodes.

- **`NCCL_NET_GDR_LEVEL=0`.** RCCL enables GPU-direct RDMA by itself. On this fabric it silently
  corrupts cross-node collectives -- wrong sums, not an error (measured 2026-08-25). Note the
  variable name: `NCCL_NET_GDR_LEVEL`, not `NCCL_GDR_LEVEL`.
- **Do not disable PyNCCL to work around a collective problem.** It costs about 20x: without it
  every collective goes through a path that is not graph-capturable, so graph capture stalls, eager
  mode goes on, and decode drops from about 17 tok/s per request to 1.4 on the same topology
  (measured 2026-08-24).
- **Confirm the fabric is actually being used.** `grep -a "Using network" server-0.log` should say
  `AWS Libfabric`. A line like `NET/Plugin: Could not find: libnccl-net.so` means RCCL fell back to
  TCP: correct answers, several times slower, no error. The EDFs in this repository enable the three
  CE hooks that prevent this, verified on an unmodified image 2026-09-09. Before that date the
  SGLang EDF carried only one of them and could not serve multi-node at all -- if you are reading an
  older note that says so, it is out of date.
- **`--pre-warm-nccl`** initialises the collective library at startup instead of on the first
  request. Set on the multi-node recipes. It moves a cost rather than removing one, but it moves it
  out of the first user's latency.
- **MPI images are a separate problem.** If your container also runs MPI, MPICH's build can hide the
  host `hwloc` that the fabric hook injects, and the documented workaround is
  `LD_PRELOAD=/opt/cscs/netstack/libhwloc.so.15`. The serving images here ship no MPI and need
  nothing of the kind.

---

## Cross-model environment variables

| Variable | Value | Why |
|---|---|---|
| `SGLANG_USE_AITER` | `1` | Switches aiter **ops**. Does **not** select the attention backend, which defaults per model. |
| `SGLANG_SET_CPU_AFFINITY` | `0` | SGLang's own pinning is rejected by the Slurm cgroup here and the process dies on a `psutil` error. |
| `AITER_JIT_DIR`, `AITER_ROOT_DIR` | a persistent path, or the image's baked one | aiter ships no prebuilt objects and JIT-builds on first **use**, not on import, behind a lock. Cold, that build can outrun the engine's watchdog and the server never serves a token. Warm, it costs nothing. Some aiter code paths ignore `AITER_JIT_DIR` and use `$HOME` instead, so point `HOME` somewhere persistent too. |
| `TRITON_CACHE_DIR` | a persistent path | Unset, it defaults under `$HOME` and every job re-JITs every kernel -- *during inference*, not at startup. Generation then arrives in bursts between total stalls. |
| `HF_HOME` | on `iopsstor`, not `capstor` | Checkpoint loading is many concurrent large reads: 9.45 GB/s against 0.83 at 16 readers (measured 2026-08-26). Also set a wide Lustre stripe on the hub directory, or a download lands on one storage target and reads back at that one target's bandwidth. |
| `NCCL_NET_GDR_LEVEL` | `0` | Multi-node only; see above. |
| `TOKENIZERS_PARALLELISM` | `false` | Silences a fork warning; no measured effect. |

---

## Flags that are cheap and worth setting everywhere

| Flag | Why |
|---|---|
| `--enable-metrics` | Prometheus metrics at `/metrics`: token throughput, running and waiting counts. The cheapest way to watch a live server. |
| `--enable-cache-report` | Puts `cached_tokens` in each response's usage block, so you see the prefix-cache hit rate **per request** rather than in aggregate. This is the diagnostic that matters on this hardware: it is how you watch the pool-to-working-set ratio cross the 1.15 threshold described above. |
| `--watchdog-timeout 1800` | A genuinely wedged engine still surfaces as a dead server rather than a job that hangs to its wall clock. |
| `--max-running-requests 128` | Caps concurrency at the scheduler. It does not reserve memory. |

---

## Slurm flags that are serving knobs in disguise

| Flag | Value | Why |
|---|---|---|
| `--partition` | `mi300` | The default partition is `mi200`: different hardware, none of this applies. |
| `--account` | **omit** | The default association works. Naming one splits otherwise identical jobs across project accounts. |
| `--mem=0` | always | A step's memory cgroup is sized from its CPU share. Without this the server is capped far below the node and dies during weight load. |
| `--cpus-per-task` | `${SLURM_CPUS_ON_NODE}` for the server | A step that does not ask gets **one** core of 192. The server then degrades with load rather than failing: 2 s per decode step early, 147 s after half an hour, with nothing queued. Measured against 88-91 tok/s for the same model with the CPUs it needs (2026-08-31). Give a client or probe running alongside one socket instead: `--cpus-per-task=24 --hint=nomultithread`. See the README for the full account. |
| `--gpus-per-node` | `4` | Every recipe here is tensor-parallel 4 inside a node. |
| `ulimit -c 0` | in the script | Beverin's `core_pattern` is machine-global; a crash otherwise drops a zero-byte stub in the working directory. Slurm propagates the limit to steps. |

---

For how to start a server, find it and read its log, see [`README.md`](README.md) in this folder.
