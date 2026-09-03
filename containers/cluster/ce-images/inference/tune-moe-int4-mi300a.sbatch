#!/usr/bin/env bash
# Tune the fused_moe Triton config kimi K2.7 is missing on MI300A.
#
# 595040/595049 both decode at ~90-100 tok/s aggregate at 64 concurrent and both warn
# "Using default MoE config ... E=384,N=512,device_name=AMD_Instinct_MI300A,dtype=int4_w4a16.json
# not found". aiter does not cover the int4 W4A16 expert path (595049 proved it: warning still
# printed, 89.8 tok/s), so the fix is vLLM's own tuner: benchmark_moe.py --tune derives the expert
# shape from the model config and emits exactly that JSON. Serving picks it up via
# VLLM_TUNED_CONFIG_FOLDER -- no image change.
#
# The tuner parallelizes the search over every GPU ray sees, hence a full node.
#SBATCH --job-name=tune-moe-int4
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/tune-moe-int4-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/tune-moe-int4-%j.err

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
export RUN_DIR="${SCRATCH}/kimi-smoke/${SLURM_JOB_ID}"
FAST_SCRATCH="/iopsstor/scratch/cscs/${USER}"
[[ -d "${FAST_SCRATCH}" ]] || FAST_SCRATCH="${SCRATCH}"
export HF_HOME="${HF_HOME:-${FAST_SCRATCH}/hf}"
export MODEL_REPO="${MODEL_REPO:-moonshotai/Kimi-K2.7-Code}"
export TP_SIZE="${TP_SIZE:-4}"

INFERENCE_EDF="${INFERENCE_EDF:-rocm723-vllm-0.27.1-pytorch211-ofi}"
EXPECTED_JSON="${EXPECTED_JSON:-E=384,N=512,device_name=AMD_Instinct_MI300A,dtype=int4_w4a16.json}"

mkdir -p "${RUN_DIR}"

# 192 logical CPUs per node = 4 sockets x 24 physical cores x 2 SMT threads. --exclusive gives the
# JOB the node; it does NOT give a STEP the node's CPUs, and a step without --cpus-per-task claims
# ONE. That is not a slow-down, it is a hang: 620948's serving step ran on 2 CPUs across 2 nodes,
# its Triton JIT of alloc_extend_kernel never finished, and the peer's PP SEND blocked until the
# 600 s NCCL watchdog aborted every rank. Every vLLM smoke before this served on 1 CPU too, so
# their tok/s were floor numbers. run_cluster.sh:749 carries the full account; an acceptance run
# must allocate the way the campaign does or it is not measuring the campaign.
SERVE_CPUS="${SERVE_CPUS:-${SLURM_CPUS_ON_NODE:-192}}"
# Clients and gates run --overlap ALONGSIDE the server, so they take one socket's PHYSICAL cores
# (--hint=nomultithread) rather than the node -- enough not to be the bottleneck, not so much that
# the measurement contends with what it is measuring.
CLIENT_CPUS="${CLIENT_CPUS:-24}"

srun --nodes=1 --ntasks=1 --exclusive --export=ALL --cpus-per-task="${SERVE_CPUS}" \
    --environment="${INFERENCE_EDF}" bash -c '
    set -Eeuo pipefail
    export TOKENIZERS_PARALLELISM=false
    # Slurm hands out ROCR_VISIBLE_DEVICES; ray hard-errors on it and wants HIP_ (595060).
    if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
        export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES}}"
        unset ROCR_VISIBLE_DEVICES
    fi

    MODEL_PATH="$(python3 - <<PY | tail -n 1
import os
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=os.environ["MODEL_REPO"], local_files_only=True))
PY
)"
    test -d "${MODEL_PATH}"
    export MODEL_PATH
    echo "model: ${MODEL_PATH}"

    # NOT benchmark_moe.py --tune: its ray harness gives one whole batch-size per worker,
    # streams tqdm only into ray worker logs inside the container /tmp, and needs a ray
    # install fighting the visible-devices env (595059/595060/595061/595167/595201 -- two of
    # those ran for hours with zero observable progress). Same search space, same
    # benchmark_config, same save_configs -- driven sequentially on one GPU with one printed
    # line per config, so cost and ETA are visible from the first minute. ray is imported by
    # the module at load but never initialized.
    if ! python3 -c "import ray" 2>/dev/null; then
        python3 -m pip install --no-cache-dir --quiet --target "${RUN_DIR}/pydeps" ray
        export PYTHONPATH="${RUN_DIR}/pydeps:${PYTHONPATH:-}"
    fi

    cd "${RUN_DIR}"
    export TUNE_BATCH_SIZES="${TUNE_BATCH_SIZES:-64}"
    python3 - <<PY
import importlib.util
import json
import os
import random
import time

import torch
import triton

spec = importlib.util.spec_from_file_location("benchmark_moe", "/opt/vllm-src/benchmarks/kernels/benchmark_moe.py")
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)

model = os.environ["MODEL_PATH"]
tp = int(os.environ["TP_SIZE"])
run_dir = os.environ["RUN_DIR"]
sizes = [int(s) for s in os.environ["TUNE_BATCH_SIZES"].split()]

config = bm.get_config(model=model, trust_remote_code=True)
# NOT bm.get_model_params: KimiK25ForConditionalGeneration is a multimodal wrapper it does not
# know, so it falls into the Mixtral default and grabs the DENSE intermediate_size 18432 --
# 595206 tuned N=4608 while serving wants N=512. The MoE fields live on the text sub-config.
# NOT get_text_config().to_dict(): this transformers build prunes model-specific keys from
# to_dict (num_experts vanished, 598780). config.json raw is the one authority both models obey;
# kimi nests the MoE fields under text_config, qwen keeps them top-level.
raw = json.load(open(os.path.join(model, "config.json")))
tc = raw.get("text_config") or raw
# deepseek-style configs (kimi) name it n_routed_experts; qwen3-moe names it num_experts.
E = tc.get("n_routed_experts") or tc["num_experts"]
topk = tc["num_experts_per_tok"]
intermediate_size = tc["moe_intermediate_size"]
hidden_size = tc["hidden_size"]
shard_intermediate_size = 2 * intermediate_size // tp
dtype = bm.resolve_dtype(config)
# Three weight formats: int4 W4A16 (kimi, has group_size), fp8 w8a8 (qwen FP8 checkpoints,
# quant_method=fp8 + block-quant shape), plain bf16/fp16 (no quantization_config).
qc = raw.get("quantization_config") or {}
use_fp8 = qc.get("quant_method") == "fp8"
group_size = bm.get_quantization_group_size(config)
use_int4 = group_size is not None and not use_fp8
if use_int4:
    block_quant_shape = [0, group_size]
elif use_fp8:
    block_quant_shape = bm.get_weight_block_size_safety(config)
else:
    block_quant_shape = None
print(f"E={E} topk={topk} shard_intermediate={shard_intermediate_size} hidden={hidden_size} "
      f"group_size={group_size} dtype={dtype} sizes={sizes}", flush=True)

# benchmark_config allocates device-less; BenchmarkWorker relied on this too (595203).
torch.set_default_device("cuda")
best_by_size = {}
for bs in sizes:
    space = bm.get_configs_compute_bound(False, None)
    space = bm.prune_rocm_search_space(bs, shard_intermediate_size, hidden_size, space, False, topk)
    # SPLIT_K splits the K reduction across workgroups. It was pinned to 1, which is the one
    # dimension the space never explored -- and the decode regime (small M, K=hidden 7168) is
    # exactly where splitting K pays. TUNE_SPLIT_K="1 2 4 8" opens it; the default keeps the
    # pinned behaviour so every earlier run stays reproducible.
    # A SPLIT_K > 1 winner is NOT servable on the tuner verdict alone: split-K variants need a
    # reduction workspace, and a serving path that does not allocate one returns silent zeros
    # rather than an error. Gate any such config on a correctness run before shipping it.
    split_k_values = [int(v) for v in os.environ.get("TUNE_SPLIT_K", "1").split()]
    space = [{**cfg, "SPLIT_K": k} for cfg in space for k in split_k_values]
    # ~18s per config, compile-dominated (595205): the full 4496-config space is a 23h job.
    # A seeded random sample lands near-optimal with high probability and fits the wall limit.
    max_configs = int(os.environ.get("TUNE_MAX_CONFIGS", "800"))
    if len(space) > max_configs:
        random.seed(0)
        space = random.sample(space, max_configs)
    print(f"[bs={bs}] search space: {len(space)} configs", flush=True)
    best, best_time = None, float("inf")
    for idx, cfg in enumerate(space):
        wall = time.time()
        try:
            kt = bm.benchmark_config(cfg, bs, E, shard_intermediate_size, hidden_size, topk, dtype,
                                     use_fp8, False, use_int4, num_iters=20,
                                     block_quant_shape=block_quant_shape, use_deep_gemm=False)
        except triton.runtime.autotuner.OutOfResources:
            continue
        if kt < best_time:
            best_time, best = kt, cfg
        print(f"[bs={bs}] {idx + 1}/{len(space)} kernel={kt:.1f} wall={time.time() - wall:.1f}s "
              f"best={best_time:.1f}", flush=True)
        if idx > 0 and idx % bm.TRITON_CACHE_CLEAR_INTERVAL == 0:
            bm.clear_triton_cache()
        # Checkpoint: a wall-limit death must still leave the best-so-far JSON behind.
        if best is not None and idx > 0 and idx % 100 == 0:
            bm.save_configs({**best_by_size, bs: bm.sort_config(best)}, E, shard_intermediate_size,
                            hidden_size, topk, dtype, use_fp8, False, use_int4, block_quant_shape, run_dir)
    bm.clear_triton_cache()
    assert best is not None, f"no config compiled for bs={bs}"
    best_by_size[bs] = bm.sort_config(best)
    print(f"[bs={bs}] DONE best={best_by_size[bs]}", flush=True)

bm.save_configs(best_by_size, E, shard_intermediate_size, hidden_size, topk, dtype,
                use_fp8, False, use_int4, block_quant_shape, run_dir)
PY
'

echo "===== emitted configs ====="
ls -l "${RUN_DIR}"/*.json
test -s "${RUN_DIR}/${EXPECTED_JSON}"

echo "===== TUNE-MOE PASSED: ${RUN_DIR}/${EXPECTED_JSON} ====="
