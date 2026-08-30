#!/usr/bin/env bash
# Does aiter actually DISPATCH to FlyDSL on gfx942, and which ops?
#
# 613511 installed flydsl 0.3.2 into the shared prefix and it imports, but installing it proves
# nothing about dispatch: the flydsl symbols this aiter build exposes are gemm_a8w8_* -- A8W8, not
# the A16W16 bf16 path our GEMM misses live on -- which is also the likeliest reason the GEMM
# tuner returned an Empty DataFrame for all 672 bf16 shapes.
#
# The MoE path is what matters anyway: AMD's own profiling puts fused_moe at 87.8-89.7% of GPU
# time, and their +162% number comes from replacing it, not from tuning GEMMs.
#
# sglang issue 32548 (Kimi-K3 on MI355) runs SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1
# AITER_FLYDSL_FORCE=1 AITER_SITUV2_A8W4=1. That is gfx950 and K3, so this only asks which of
# those switches this gfx942 aiter build even honours, before any of it reaches a serving arm.
#SBATCH --job-name=probe-opus
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/probe-opus-%j.out
#SBATCH --error=logs/probe-opus-%j.err
set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
INFERENCE_EDF="${INFERENCE_EDF:-sglang-rocm-mi30x}"
PREFIX="${FLYDSL_PREFIX:-${SCRATCH}/pyprefix/${INFERENCE_EDF}}"
CACHE_ROOT="${JIT_CACHE_ROOT:-${SCRATCH}/.jit-cache}/${INFERENCE_EDF}"
ENGINE_PYTHON="${SGLANG_PYTHON:-/opt/venv/bin/python3}"
mkdir -p "${CACHE_ROOT}/home" "${CACHE_ROOT}/xdg"

srun --nodes=1 --ntasks=1 --exclusive --export=ALL --environment="${INFERENCE_EDF}" \
     --cpus-per-task="${SLURM_CPUS_ON_NODE:-64}" bash -c "
    set -Eeuo pipefail
    export HOME='${CACHE_ROOT}/home' XDG_CACHE_HOME='${CACHE_ROOT}/xdg'
    export AITER_JIT_DIR='${CACHE_ROOT}/aiter' TOKENIZERS_PARALLELISM=false
    export PYTHONPATH='${PREFIX}':\${PYTHONPATH:-}
    if [[ -n \"\${ROCR_VISIBLE_DEVICES:-}\" ]]; then
        export HIP_VISIBLE_DEVICES=\"\${HIP_VISIBLE_DEVICES:-\${ROCR_VISIBLE_DEVICES}}\"
        unset ROCR_VISIBLE_DEVICES
    fi
    export SGLANG_USE_AITER=1 AITER_FLYDSL_FORCE=1
    ${ENGINE_PYTHON} /capstor/scratch/cscs/ybudanaz/x86_64/optarena/containers/cluster/ce-images/inference/tune-aiter-opus-gemm.py
"

