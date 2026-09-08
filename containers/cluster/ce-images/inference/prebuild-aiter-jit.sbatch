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
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/prebuild-aiter-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/prebuild-aiter-%j.err

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
FAST_SCRATCH="/iopsstor/scratch/cscs/${USER}"
[[ -d "${FAST_SCRATCH}" ]] || FAST_SCRATCH="${SCRATCH}"
export HF_HOME="${HF_HOME:-${FAST_SCRATCH}/hf}"
INFERENCE_EDF="${INFERENCE_EDF:-vllm-latest}"
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

# Importing an op module builds NOTHING: @compile_ops returns a wrapper that only calls
# get_module() -- and therefore only builds -- when the op is INVOKED. The previous version of
# this script imported ten families and made exactly one real call (rms_norm), so exactly one
# family was ever prebuilt. Serving GLM-5.3 then JIT-built 13 modules x 4 ranks at run time,
# ~15 min of a 24 min start, and waited on a build baton for module_custom_all_reduce.
#
# So: one real CALL per module. Arguments need not be semantically right -- the jit/core.py
# wrapper is def wrapper(*args, custom_build_args=None, **kwargs) and reaches get_module()
# before the args ever go to the C++ side, so a shape error still leaves the module built.
# Each call is wrapped because op coverage varies across aiter builds and a missing symbol
# must not abort the prebuild.
d = torch.device("cuda")
bf16, fp8 = torch.bfloat16, torch.float8_e4m3fnuz


def norm_call():
    n = 7168
    x = torch.randn(4, n, dtype=bf16, device=d)
    aiter.layernorm2d_fwd_with_add(torch.empty_like(x), x, x.clone(), torch.empty_like(x),
                                   torch.ones(n, dtype=bf16, device=d),
                                   torch.zeros(n, dtype=bf16, device=d), 1e-6)


def quant_call():
    x = torch.randn(4, 7168, dtype=bf16, device=d)
    aiter.dynamic_per_tensor_quant(torch.empty(x.shape, dtype=fp8, device=d), x,
                                   torch.ones(1, dtype=torch.float32, device=d))


def activation_call():
    x = torch.randn(4, 2 * 7168, dtype=bf16, device=d)
    aiter.silu_and_mul(torch.empty(4, 7168, dtype=bf16, device=d), x)


def cache_call():
    k = torch.randn(4, 8, 128, dtype=bf16, device=d)
    kc = torch.zeros(64, 8, 128, 16, dtype=bf16, device=d)
    aiter.reshape_and_cache(k, k.clone(), kc, kc.clone(),
                            torch.arange(4, dtype=torch.int64, device=d), "auto")


def moe_sum_call():
    # topk MUST be one of 2/4/5: any other value takes a python-side torch.sum branch that
    # never reaches the kernel, which would prebuild nothing while looking like it worked.
    x = torch.randn(8, 4, 7168, dtype=bf16, device=d)
    aiter.moe_sum(x, torch.empty(8, 7168, dtype=bf16, device=d))


def sample_call():
    logits = torch.randn(4, 1024, dtype=torch.float32, device=d)
    aiter.greedy_sample(torch.empty(4, dtype=torch.int64, device=d), logits)


CALLS = (
    ("module_norm", norm_call),
    ("module_quant", quant_call),
    ("module_activation", activation_call),
    ("module_cache", cache_call),
    ("module_moe_asm", moe_sum_call),
    ("module_sample", sample_call),
    ("module_gemm_common", lambda: aiter.get_padded_m(128, 7168, 7168, 1)),
    ("module_moe_sorting_opus", lambda: aiter.moe_sorting_opus_get_workspace_size(128, 256, 8)),
    ("module_rmsnorm", lambda: aiter.rms_norm(torch.randn(4, 7168, dtype=bf16, device=d),
                                              torch.ones(7168, dtype=bf16, device=d), 1e-6)),
)

for md, fn in CALLS:
    try:
        fn()
        print("called:", md, flush=True)
    except Exception as exc:
        # A raise here is expected for shape/dtype mismatches and is harmless: the module is
        # already built by the time the C++ side sees the arguments.
        print(f"called (raised, module still built) {md}: {type(exc).__name__}: {exc}", flush=True)
PY

    echo "=== after ==="; ls -1 "${AITER_JIT_DIR}" | wc -l; ls -1 "${AITER_JIT_DIR}" | head -20
'

echo "===== aiter JIT cache populated at ${AITER_JIT_DIR} ====="
find "${AITER_JIT_DIR}" -name "*.so" | wc -l | xargs printf 'prebuilt .so count: %s\n'
