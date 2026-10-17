# Known issues serving on Beverin (MI300A)

Problems of LLM endpoints and agent fleets on Beverin, each with its symptoms and the configuration that
avoids it. Most of them produce no error, so a job looks healthy while its results are wrong. The
per-model pages and [`knobs.md`](knobs.md) carry the configuration that avoids them.

| # | Issue | Visible as | Status |
|---|---|---|---|
| 1 | RCCL internal error on cross-node sends and all-reduces (GLM-5.3, 4 nodes) | server dies after 1-4 h of agent load (tp=4 pp=4), or at graph capture (tp=8 pp=2) | tp=4 pp=4: the 1-4 h deaths are host OOM on a stage (pool cut to 1.5 M); tp=8 pp=2: CSCS CXI settings |
| 2 | A writable CE container freezes on fuse-overlayfs | every agent on the node stalls at once, no error | fixed in hb: the agent container is read-only |
| 3 | vLLM `ROCM_AITER_FA` garbles Qwen3.8 under agent load | random tokens in reasoning, invented tool names | avoided: Qwen3.8 serves on SGLang |
| 4 | Self-tuned GEMM kernel tables break Qwen3.8 numerics | agents loop on degenerate reasoning | avoided: stock aiter tables only |
| 5 | RCCL GPU-direct RDMA corrupts cross-node sums; its connect can fail registering GPU memory | wrong outputs, no error; or `NET/OFI Couldn't register memory region with regattr. RC: -22` and the server dies at the first cross-node send | avoided: `NCCL_NET_GDR_LEVEL=LOC` on the server command (`run_cluster.sh`); the CSCS netstack hook overwrites a value set in the step environment |
| 6 | RCCL silently falls back to TCP | several times slower, correct | check the `Using network` log line |
| 7 | SGLang split groups hang on the RCCL/ROCm build | init hang | avoided: no serving setup enables them (SGLang has no environment switch for them) |
| 8 | APU out-of-memory kills without a traceback | log stops, exit by signal | fraction rules in [`knobs.md`](knobs.md#the-mi300a-memory-model) |

## 1. GLM-5.3: RCCL `ncclInternalError` on the pipeline send

**Setup.** `zai-org/GLM-5.3` fp8 on SGLang, `tp=4 pp=4` over four MI300A nodes, RCCL over Slingshot through
the `aws-ofi-nccl` plugin (`NCCL_NET="AWS Libfabric"`, `FI_PROVIDER=cxi`, `FI_CXI_RX_MATCH_MODE=software`,
`FI_MR_CACHE_MONITOR=disabled`; GPU-direct RDMA on, see #5). 12 to 48 concurrent agent conversations.

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

**Host memory, not RCCL, ends the long `tp=4 pp=4` runs.** With the CXI settings, runs end in `NODE_FAIL` after
2 h 49 min to 3 h, with an `oom_kill` in the inference step recorded by Slurm; the surviving stages then fail with
`gloo ... Connection closed by peer`. In every such run the node monitor shows the heaviest stage at 502-512 of
513 GB when serving stops. A run that dies at its first cross-node send at 441 GB is the GPU-direct registration
failure of [#5](#5-8-fabric-and-memory). A peer killed mid-send explains the RCCL internal error. The KV pool is
capped at 1.5 M tokens with `--max-total-tokens`; 1.8 M is OOM-killed ([glm53.md](glm53.md)).

**Open.** Whether `tp=4 pp=4` with the 1.5 M pool and `NCCL_NET_GDR_LEVEL=LOC` serves for the length of an
experiment. Questions for CSCS: the netstack artifact `gpu_rocm7-cxi_13.1.0-ofi_2.6.0-aws_1.20.0` (26.08.1) with
RCCL 2.27.7, and known RCCL/libfabric issues with long-lived cross-node traffic on MI300A.

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
- **TCP fallback.** Without the plugin RCCL runs over TCP: correct, several times slower, no error.
  `grep -a "Using network" server-0.log` must say `AWS Libfabric`; our EDFs set `NCCL_NET_PLUGIN=ofi` so a
  missing plugin is an error instead.
- **Split groups.** Any SGLang/vLLM option that enables split process groups hangs init on this RCCL build.
  No serving setup passes one. `VLLM_DISTRIBUTED_USE_SPLIT_GROUP` is a vLLM variable that SGLang never reads,
  so it is not set.
- **APU memory.** GPUs and host share one pool, `--mem-fraction-static` is node-wide, and an out-of-memory death
  is the host OOM killer: the log stops with no traceback.

## Model behaviour that is not infrastructure

At sampling temperature 1.5 both Qwen3.8 and gpt-oss-120b degrade: Qwen3.8 emits long runs of random tokens
inside otherwise valid tool calls, and gpt-oss-120b emits malformed harmony tool headers, so its episodes end after
one turn. That is the model, not the server; the temperature3 study measures it.
