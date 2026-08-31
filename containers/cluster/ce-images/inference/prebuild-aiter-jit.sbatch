#!/usr/bin/env bash
# Populate a shared AITER JIT cache so serving nodes never build kernels at startup.
#
# aiter ships no prebuilt .so: every process JIT-builds module_aiter_core (21-58s measured) and
# then each op module lazily on first use. That is why 598021 -- the one kimi run with aiter
# enabled -- never answered /v1/models and died on "API not ready in 5400s".
#
# Building once here and pointing every later run at the same AITER_JIT_DIR turns that into a
# cache hit. Do NOT let several nodes populate a cold shared dir concurrently: aiter writes the
# .so in place, so parallel builds race on the same path and a rank can load a half-written
# module. One node fills it, later jobs only read it (gate any user with --dependency=afterok).
#SBATCH --job-name=prebuild-aiter
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=logs/prebuild-aiter-%j.out
#SBATCH --error=logs/prebuild-aiter-%j.err

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
FAST_SCRATCH="/iopsstor/scratch/cscs/${USER}"
[[ -d "${FAST_SCRATCH}" ]] || FAST_SCRATCH="${SCRATCH}"
export HF_HOME="${HF_HOME:-${FAST_SCRATCH}/hf}"
INFERENCE_EDF="${INFERENCE_EDF:-rocm723-vllm-0.27.1-pytorch211-ofi}"
# Same root and the same image key run_cluster.sh uses: capstor for build artefacts, and one
# subtree per image because these .so files are built against one aiter/ROCm build and a rank
# that loads a mismatched one fails late or silently. HOME goes with it -- aiter falls back to
# expanduser("~")/.aiter for template ops whatever AITER_JIT_DIR says.
CACHE_ROOT="${JIT_CACHE_ROOT:-${SCRATCH}/.jit-cache}/${INFERENCE_EDF}"
export AITER_JIT_DIR="${AITER_JIT_DIR:-${CACHE_ROOT}/aiter}"
export HOME="${CACHE_ROOT}/home"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}"
export MODEL_REPO="${MODEL_REPO:-moonshotai/Kimi-K2.7-Code}"
export SERVED_MODEL="${SERVED_MODEL:-optarena-vllm}"

mkdir -p "${AITER_JIT_DIR}"
printf 'job:   %s\ncache: %s\n\n' "${SLURM_JOB_ID}" "${AITER_JIT_DIR}"

srun --nodes=1 --ntasks=1 --exclusive --export=ALL --environment="${INFERENCE_EDF}" bash -c '
    set -Eeuo pipefail
    export TOKENIZERS_PARALLELISM=false
    if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
        export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES}}"
        unset ROCR_VISIBLE_DEVICES
    fi
    # Every aiter path vLLM can dispatch to, so the cache covers whatever the server picks.
    export VLLM_ROCM_USE_AITER=1

    echo "=== before ==="; ls -1 "${AITER_JIT_DIR}" 2>/dev/null | wc -l

    python3 - <<PY
import os
import torch

import aiter
print("aiter loaded from", aiter.__file__, flush=True)
print("jit dir:", os.environ.get("AITER_JIT_DIR"), flush=True)

# Importing builds module_aiter_core. Touching the op families vLLM dispatches to builds the
# rest -- each import is wrapped because coverage varies by aiter build and a missing symbol
# must not abort the whole prebuild.
d = torch.device("cuda")
for name in ("aiter.ops.rmsnorm", "aiter.ops.norm", "aiter.ops.gemm_op_a8w8", "aiter.ops.moe_op",
             "aiter.ops.mha", "aiter.ops.attention", "aiter.ops.rope", "aiter.ops.quant",
             "aiter.ops.shuffle", "aiter.fused_moe"):
    try:
        __import__(name)
        print("built/imported:", name, flush=True)
    except Exception as exc:
        print(f"skip {name}: {type(exc).__name__}: {exc}", flush=True)

# A real rmsnorm call forces the kernel module rather than only the python wrapper.
try:
    x = torch.randn(4, 7168, dtype=torch.bfloat16, device=d)
    w = torch.ones(7168, dtype=torch.bfloat16, device=d)
    aiter.rms_norm(x, w, 1e-6)
    print("rms_norm kernel built", flush=True)
except Exception as exc:
    print(f"skip rms_norm call: {type(exc).__name__}: {exc}", flush=True)
PY

    echo "=== after ==="; ls -1 "${AITER_JIT_DIR}" | wc -l; ls -1 "${AITER_JIT_DIR}" | head -20
'

echo "===== aiter JIT cache populated at ${AITER_JIT_DIR} ====="
find "${AITER_JIT_DIR}" -name "*.so" | wc -l | xargs printf 'prebuilt .so count: %s\n'
