# Known issues serving on Beverin (MI300A)

Problems of LLM endpoints and agent fleets on Beverin, each with its symptoms and the configuration that
avoids it. Most of them produce no error, so a job looks healthy while its results are wrong. The
per-model pages and [`knobs.md`](knobs.md) carry the configuration that avoids them.

| # | Issue | Visible as | Status |
|---|---|---|---|
| 1 | RCCL internal error on cross-node sends and all-reduces (GLM-5.3, 4 nodes) | server dies after 1-4 h of agent load (tp=4 pp=4), or at graph capture (tp=8 pp=2) | fixed for tp=4 pp=4: a stage OOM-killed mid-send, or #5; the 1.5 M pool and `LOC` serve 15-24 h. tp=8 pp=2 is not used |
| 2 | A writable CE container freezes on fuse-overlayfs | every agent on the node stalls at once, no error | fixed in hb: the agent container is read-only |
| 3 | vLLM `ROCM_AITER_FA` garbles Qwen3.8 under agent load | random tokens in reasoning, invented tool names | avoided: Qwen3.8 serves on SGLang |
| 4 | Self-tuned GEMM kernel tables break Qwen3.8 numerics | agents loop on degenerate reasoning | avoided: stock aiter tables only |
| 5 | RCCL GPU-direct RDMA corrupts cross-node sums; its connect can fail registering GPU memory | wrong outputs, no error; or `NET/OFI Couldn't register memory region with regattr. RC: -22` and the server dies at the first cross-node send | avoided: `NCCL_NET_GDR_LEVEL=LOC` on the server command (`run_cluster.sh`); the CSCS netstack hook overwrites a value set in the step environment |
| 6 | RCCL silently falls back to TCP | several times slower, correct | check the `Using network` log line |
| 7 | SGLang split groups hang on the RCCL/ROCm build | init hang | avoided: no serving setup enables them (SGLang has no environment switch for them) |
| 8 | APU out-of-memory kills without a traceback | log stops, exit by signal | fraction rules in [`knobs.md`](knobs.md#the-mi300a-memory-model) |
| 9 | gpt-oss-120b vLLM EngineCore segfaults minutes after its first requests | `!!!!!!! Segfault encountered !!!!!!!`, then `EngineDeadError` and HTTP 500 on every request; the run ends with no grades | fixed with issue 10: 4 of 4 runs with the inherited binding crashed, 0 of 6 without it |
| 10 | A fused launcher started as a one-CPU step passed its CPU binding to every run | inference servers and agents of the run pinned to 2 of 192 cores; vLLM `No available shared memory broadcast block found in 60 seconds` | fixed: `launch_fused.sbatch` unsets `SLURM_CPU_BIND*` |

## 1. GLM-5.3: RCCL `ncclInternalError` on the pipeline send

**Setup.** `zai-org/GLM-5.3` fp8 on SGLang, `tp=4 pp=4` over four MI300A nodes, RCCL over Slingshot through
the `aws-ofi-nccl` plugin (`NCCL_NET="AWS Libfabric"`, `FI_PROVIDER=cxi`, `FI_CXI_RX_MATCH_MODE=software`,
`FI_MR_CACHE_MONITOR=disabled`; GPU-direct RDMA on until 2026-10-13, see #5). 12 to 48 concurrent agent
conversations.

**What happens.** After 1 to 4 hours of steady decode, one pipeline stage's scheduler raises during the
send of a step's tensors to the next stage:

```
scheduler_pp_mixin.py, in _pp_send_dict_to_next_stage
torch.distributed.DistBackendError: NCCL error in: .../NCCLUtils.cpp:93, internal error - please report this
issue to the NCCL developers, NCCL version 2.27.7
ncclInternalError: Internal check failed.
```

The other stages then fail with `gloo ... Connection closed by peer`, and teardown logs
`Failed to destroy CXI Service ID 5 (cxi0): Device or resource busy` on every node. Observed times to failure:
18 min to 3 h 07 min of serving, or `NODE_FAIL`; the same shape has also served for 14 h. The error comes from
RCCL's own internal check, not from SGLang.

**A second shape fails the same way.** On 4 nodes as `tp=8 pp=2` (`SGLANG_PP_SIZE=2`), each tensor-parallel
group spans two nodes, and the server dies before serving a request: the cross-node all-reduce raises the same
RCCL internal error while SGLang captures CUDA graphs. SGLang's post-load barrier is fixed at 480 s, and the two
nodes of a group can finish loading weights up to 18 minutes apart; the sglang image reads
`SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S` to lengthen it. Both a pipeline send and a tensor-parallel all-reduce
fail inside RCCL when they cross nodes through the plugin.

**CSCS's CXI settings.** The serving EDFs set what CSCS documents for NCCL over Slingshot:
`FI_MR_CACHE_MONITOR=userfaultfd`, `FI_CXI_DISABLE_HOST_REGISTER=1`, `FI_CXI_DEFAULT_CQ_SIZE=131072`,
`FI_CXI_DEFAULT_TX_SIZE=32768` and `NCCL_CROSS_NIC=1`. With them:

- `tp=8 pp=2` raises no RCCL error through graph capture, but its warmup request times out after 1800 s; that
  shape is not used.
- `tp=4 pp=4` serves: API up after 55 min of weight loading; tool-call, reasoning and long-context recall (9/9)
  checks pass.

**Cause: a stage dies first, or #5.** Slurm's accounting names the trigger in each of the four jobs that logged the
internal error (2026-10-10/11), and the jobs after them ended the same way:

| Job | KV pool | Slurm record of the inference step | Reading |
|---|---|---|---|
| 675850, 675956 | 2.6-2.8 M | `OUT_OF_MEMORY` | a stage OOM-killed; its peer's send fails with the internal error |
| 675957, 676621 | 2.6-2.8 M | job `NODE_FAIL` | a node lost under load |
| 676681 | 2.8 M | `OUT_OF_MEMORY` | heaviest stage at 504-508 of 513 GB |
| 677837 | 1.8 M | job `FAILED` after 3 h 24 min of the step | heaviest stage at 508 of 513 GB |
| 675958, 677231, 678947 | | ended after 20-24 min | GPU-direct registration failure of [#5](#5-8-fabric-and-memory) (`regattr RC -22`, `cxil_map: write error`) |

The surviving stages of a killed peer fail with `gloo ... Connection closed by peer`. Every one of these runs had
GPU-direct RDMA on: until 2026-10-13 the netstack hook's `PHB` replaced the layer's value (#5). In the same period
one run, 677838 (1.8 M), served past 4 h (15 h 27 min).

**Fix (2026-10-13).** Two changes went in together: the KV pool is capped at 1.5 M tokens with
`--max-total-tokens` ([glm53.md](glm53.md)), and `run_cluster.sh` puts `NCCL_NET_GDR_LEVEL=LOC` on the server
command. Every `tp=4 pp=4` run since then served until its job ended, with no RCCL error and no OOM kill:

| Job | Inference step | Ended by |
|---|---|---|
| 679141 | 14 h 52 min | the experiment finished |
| 716131, 716132, 716221, 716222 | 23 h 45 min each | the 24 h time limit |
| 731329 | 22 h 16 min, 205 k decode batches, pool full at peak | the experiment finished |

No run separates the two changes, so serving keeps both. Serving keeps `LOC` whatever #5 finds for the judge,
because with GPU-direct RDMA on, three GLM-5.3 servers died at their first cross-node sends. Check the NCCL log of
a run: `set by environment to LOC` and channels `via NET/AWS Libfabric/<n>` without `/GDRDMA` (the 4-node launch
747685 on 2026-11-18 shows both).

## 2. A writable container freezes on fuse-overlayfs

**Setup.** One agent node runs up to 60 agents in one CE container step. With `writable = true` the CE stacks
a fuse-overlayfs upper layer over the squashfs image.

**What happens.** The container stops making progress all at once. Processes already streaming keep writing
to their bind mounts, but no new process starts: the node monitor stops sampling, no agent issues another
request, and nothing reports an error, for up to 7 h. With Qwen3.8 at temperature 1.5, whose agents issue
commands such as `rm -rf /root`, the agent node's monitor and its last request stop at the same second. A copy-up
of a large image tree through fuse-overlayfs is consistent with every symptom. The later
`Transport endpoint is not connected` on `/opt/venv` is the teardown unmounting the image under exiting
processes, not the cause.

**What we do.** The agent container runs read-only (`writable = false`, set by `derived_edf` in
`hpcagent_bench/cluster/container_runtime.sh`), with the node's own `/tmp` bound in. Agents write only to bind
mounts and to the per-agent tmpfs the seal gives each one, so a stray write fails at once with `EROFS`. A run
that still sees no change under its agent and judge folders for `AGENT_STALL_ABORT_SECONDS` (default 3600) is
stopped and its unfinished kernels are rerun.

## 3. vLLM with `ROCM_AITER_FA` garbles Qwen3.8 under agent load

vLLM 0.28 with aiter flash attention (`ROCM_AITER_FA`, `VLLM_ROCM_USE_AITER=1`) passed every synthetic gate
(300 to 500 output tokens), but under real agent load 65-75% of the Qwen3.8 agents produced random CJK text in
their reasoning and called tools that do not exist. gpt-oss-120b on the same image stayed clean. Qwen3.8 serves on
SGLang instead. Short coherence checks do not catch it (1 of 40 conversations at 4 turns), so an engine switch
needs a long multi-turn check, not a throughput number alone.

## 4. Self-tuned kernel tables break Qwen3.8

aiter ships tuned bf16 GEMM tables for gfx950 and gfx1250 only, so on MI300A (gfx942) several shapes run hipBLASLt.
A self-generated MI300A table selects split-K kernels that accumulate in bf16, which puts up to 3.75% of the
elements of a GEMM outside tolerance; Qwen3.8 agents served under it end in degenerate reasoning loops. Serving
uses the stock tables and accepts the slower fallbacks
([`knobs.md`](knobs.md#aiter-correctness-first-fallbacks-accepted)). Speed tuning of kernels must be checked
for numerical tolerance against a reference, not only for throughput.

## 5-8. Fabric and memory

- **GPU-direct RDMA.** RCCL enables it by itself, and on this fabric it silently corrupts cross-node sums.
  Its connect can also fail outright: a GLM-5.3 server dies at the first cross-node pipeline send with
  `NET/OFI Couldn't register memory region with regattr. RC: -22` / `Unable to register memory (type = 1)` in the
  NCCL log and `cxil_map: write error` in the server log.
  Turn it off with `NCCL_NET_GDR_LEVEL=LOC` (not `NCCL_GDR_LEVEL`, and not `0`: RCCL reads a number as an
  old-style path level). The value has to be set **inside the container, on the server command**: the CSCS
  netstack hook (`com.hooks.netstack` in the EDF) writes its own defaults into the container after the step
  environment, `NCCL_NET_GDR_LEVEL=PHB` among them; a value set only in the step environment is logged as
  `NCCL_NET_GDR_LEVEL set by environment to PHB` and connects via `GDRDMA`. `run_cluster.sh` passes the layer's
  value on the server command, taken from the batch shell (`SERVER_NCCL_NET_GDR_LEVEL`): the role step cannot
  re-read the layer env inside its container (the EDF does not mount `experiments/`), and `NCCL_NET_GDR_LEVEL`
  read there is the hook's PHB. `serve-only.sbatch` has no batch-shell `run_cluster.sh`:
  its role step re-reads `$RUN_DIR/serve.env` inside the container (the registered EDF mounts `$SCRATCH`),
  which re-exports the layer's `NCCL_NET_GDR_LEVEL=LOC` over the hook's PHB. Check the NCCL log
  (`vllm/nccl.<host>.<pid>.log`): `set by environment to LOC`, `use ring ... GDR 0`, and channels `via NET/AWS
  Libfabric/<n>` without `/GDRDMA`. The hook sets other values the same way (`NCCL_MIN_NCHANNELS=112`,
  `FI_MR_CACHE_MAX_COUNT`, `FI_CXI_RDZV_*`, `FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD`); any of them we want different
  goes on the server command as well.
- **GPU-direct RDMA: what could fix it (open, 11-19).** RCCL under HIP always asks for a flush after a GPU-direct
  receive (`ncclTopoNeedFlush` sets it unconditionally, `src/graph/paths.cc`). It flushes through the plugin's
  `iflush`, which aws-ofi-nccl 1.20 serves with an RDMA read into a host page unless `OFI_NCCL_GDR_FLUSH_DISABLE=1`.
  RCCL's own HDP flush (`RCCL_NET_HDP_FLUSH`) is off by default. No CSCS page, aws-ofi-nccl issue or libfabric issue
  reports this corruption on MI300A with cxi. The nearest reports are a missing GDR flush on RoCE/IB hosts (ROCm
  rocm-systems PR #14237), a cxi fault avoided with `FI_HMEM_ROCR_USE_DMABUF=0` (libfabric #12775) and MR-cache
  monitor faults (libfabric #11451). The judge runs its ML grades with GDR on (`NCCL_NET_GDR_LEVEL=PHB` from the
  hook and the EDF; RCCL logs `GDR 1`, job 754679). `debug/gdr/job.sbatch` under
  `regrades/mlscale20-1118-p16/` checks RCCL `all_reduce`, `reduce_scatter` and `all_gather` across two nodes at
  P=8 against host references. It runs PHB and LOC with and without address churn, then PHB with one change each:
  `RCCL_NET_HDP_FLUSH=1`, `NCCL_NET_GDR_READ=0`, `NCCL_PROTO=Simple`, `OFI_NCCL_MR_CACHE_DISABLE=1`,
  `FI_MR_CACHE_MONITOR` set to `userfaultfd`, `memhooks` or `kdreg2`, `FI_HMEM_ROCR_USE_DMABUF=0` or `1`,
  `NCCL_DMABUF_ENABLE=0`, `OFI_NCCL_PROTOCOL=RDMA`, `FI_CXI_RX_MATCH_MODE=hardware` and `HSA_XNACK=1`. Not yet run.
- **TCP fallback.** Without the plugin RCCL runs over TCP: correct, several times slower, no error.
  `grep -a "Using network" server-0.log` must say `AWS Libfabric`; our EDFs set `NCCL_NET_PLUGIN=ofi` so a
  missing plugin is an error instead.
- **Split groups.** Any SGLang/vLLM option that enables split process groups hangs init on this RCCL build.
  No serving setup passes one. `VLLM_DISTRIBUTED_USE_SPLIT_GROUP` is a vLLM variable that SGLang never reads,
  so it is not set.
- **APU memory.** GPUs and host share one pool, `--mem-fraction-static` is node-wide, and an out-of-memory death
  is the host OOM killer: the log stops with no traceback.

## 9. gpt-oss-120b EngineCore segfault

**Symptom.** Within minutes of serving its first requests, a gpt-oss-120b vLLM server prints
`!!!!!!! Segfault encountered !!!!!!!` with no usable frame (`File "<unknown>", line 0`). The EngineCore then raises
`RuntimeError: cancelled` from `shm_broadcast.acquire_read`, the API server answers every request with HTTP 500
(`EngineDeadError`), and the run shuts down. Its judge reports `judge_tail FAIL rank-0: no grades`. The run is
20 to 30 minutes old by then, most of it the server's boot. In the two runs whose server logs survive, the crash
came 1 and 8 minutes after the first request, with one request running.

**Cause: the CPU binding of issue 10.** Every gpt-oss-120b run started by a fused launcher running as a one-CPU step
before the fix crashed this way, and no run started from a batch shell did (umbrella jobs 752845 and 753775,
2026-11-18/19):

| Runs | Launched from | Segfaults | Run length |
|---|---|---|---|
| 75284521, 75284522 (`mlscale-oss-r8`, `-r9`), 75284552, 75284553 (`gitsc13-oss`) | one-CPU step, before the fix | 4 of 4 | 26-32 min |
| 75284503, 75284504, 75284505, 75284506, 75377503 | the batch shell | 0 of 5 | 39 min to 4 h 19 min |
| 75284554 | one-CPU step, after the fix | 0 of 1 | 21 min, ended with its job |

The two groups share image and engine arguments. `mlscale-oss-r8` and `-r9` crashed on the node that had served
75284503 for 1 h 40 min from the batch shell. 75377503 reran the setup env file and checkout of 75284552 and
75284553 from the batch shell and served for 4 h 19 min. The node monitor shows the cap. Under load the vLLM node
of an unbound run uses 5.5% of its CPUs, against 2.3% idle; the bound runs stay at 2.9-3.4% idle and 3.3-3.4%
loaded. 75284554 shows the unbound profile (5.5%). The bound servers also logged four or five 60-second
`shm_broadcast` stalls while they booted, against one in an unbound run. Which code faults is unknown, since the log
has no frame.

**Fix.** The fix of issue 10. No run since then has seen the segfault. A step-launched run past 30 minutes has not
happened yet: 75284554 was cut at 21 minutes when its job ended. On a run, check `grep -c 'Segfault encountered'`
over the job's logs and the vLLM node's `cpu_pct` in `monitor/vllm-<node>.csv`.

## 10. CPU binding inherited from a one-CPU launcher step

**Symptom.** In a run started by `launch_fused.sbatch` from a step (`srun --overlap -c1`), `taskset -cp` of the
inference server and of the agents shows `0,96`: two hardware threads of one core. A run started from the batch
script shows `0-191`. Judges and grade-under ranks are not affected, since they pass their own `--cpu-bind`.

**Cause.** Slurm exports the launcher step's binding as `SLURM_CPU_BIND=quiet,mask_cpu:...` (with `_LIST`, `_TYPE`
and `_VERBOSE`), and every srun that does not name its own `--cpu-bind` applies it to its tasks.

**Fix.** A launcher started as a step unsets `SLURM_CPU_BIND*` before starting its runs, next to its existing
`SLURM_CPUS_ON_NODE` correction. Test: `tests/test_cluster_launch_scripts.py::test_a_launcher_started_as_a_step_hands_its_runs_the_jobs_cpus_not_the_steps`.

## Model behaviour that is not infrastructure

At sampling temperature 1.5 both Qwen3.8 and gpt-oss-120b degrade: Qwen3.8 emits long runs of random tokens
inside otherwise valid tool calls, and gpt-oss-120b emits malformed harmony tool headers, so its episodes end after
one turn. That is the model, not the server; the temperature3 study measures it.
