#!/usr/bin/env bash
# Add AMD aiter to the pt211 vLLM image for its ROCm MoE kernels.
#
# 595040: kimi K2.7 (int4 W4A16, E=384 experts) decodes at 96 tok/s aggregate at 64 concurrent
# because fused_moe falls back to a default Triton config -- "Config file not found at
# .../E=384,N=512,device_name=AMD_Instinct_MI300A,dtype=int4_w4a16.json". aiter is AMD's tuned
# kernel library for exactly this; vLLM switches to it with VLLM_ROCM_USE_AITER=1 at serve time
# (opt-in per submission, --export=ALL forwards it -- no serve-script change needed).
#
# The aiter commit is NOT ours to choose: vLLM pins one in its ROCm dockerfiles, and mixing an
# arbitrary aiter with vllm 0.23 is how silent kernel mismatches happen. The pin is discovered
# from /opt/vllm-src inside the image; only if none exists does main get used.
#
# JIT mode, no PREBUILD_KERNELS: prebuilding every gfx942 kernel is a multi-hour compile. JIT
# costs a first-use compile per kernel shape per job (writes land in the CE overlay), which the
# kimi smoke will price in before any campaign decision.
#
# Patches the image IN PLACE with a .before-aiter backup, exactly like add-flash-attn-pt211.sbatch.
# The name keeps its version so every env file and EDF keep working.
#
# VLLM_VERSION picks WHICH image to patch (default 0.23.0, the measured one). 0.27.1 needs this
# too, and not only for MoE speed: its Kimi ViT patch-embed does a bare `from aiter.ops.triton...`
# import inside the multimodal dummy profile_run, so an aiter-less 0.27.1 image dies at startup
# with ModuleNotFoundError before the API ever binds -- what 600649 hit, on a text-only campaign.
#SBATCH --job-name=pt211-aiter
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/add-aiter-pt211-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/add-aiter-pt211-%j.err

set -Eeuxo pipefail
umask 0022

# Derive from SLURM_SUBMIT_DIR, not BASH_SOURCE: slurm copies the script to
# /var/spool/slurmd/job<id>/slurm_script. Submit from containers/cluster/ce-images/inference.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -n "${VLLM_BUILD_ROOT:-}" ]]; then
    DEFAULT_IMAGE_DIR="${VLLM_BUILD_ROOT}/containers"
else
    # Standalone patch of the production image, which lives BESIDE the repo, not inside it.
    DEFAULT_IMAGE_DIR="$(dirname -- "$(cd -- "${SUBMIT_DIR}/../../../.." && pwd)")/ce-images"
fi

CE_IMAGE_DIR="${CE_IMAGE_DIR:-$DEFAULT_IMAGE_DIR}"

VLLM_VERSION="${VLLM_VERSION:-0.23.0}"
IMAGE_STEM="${CE_IMAGE_DIR}/rocm723-vllm-${VLLM_VERSION}-pytorch211-ofi"

SOURCE_IMAGE="${IMAGE_STEM}.sqsh"
TEMP_IMAGE="${IMAGE_STEM}.aiter.tmp.sqsh"
FINAL_IMAGE="${IMAGE_STEM}.sqsh"
BACKUP_IMAGE="${IMAGE_STEM}.before-aiter.sqsh"

# Enroot names are machine-global, so the version keeps two concurrent patches from colliding.
BUILD_NAME="rocm723-pt211-aiter-patch-${VLLM_VERSION//./}"

test -f "$SOURCE_IMAGE"

echo "Host:        $(hostname)"
echo "Source:      ${SOURCE_IMAGE}"
echo "Final:       ${FINAL_IMAGE}"

enroot remove --force "$BUILD_NAME" 2>/dev/null || true
rm -f "$TEMP_IMAGE" "${TEMP_IMAGE}.sha256"

enroot create \
  --name "$BUILD_NAME" \
  "$SOURCE_IMAGE"

enroot start \
  --root \
  --rw \
  -e AITER_PIN="${AITER_PIN:-}" \
  "$BUILD_NAME" \
  bash -lc '
    set -Eeuxo pipefail

    export VIRTUAL_ENV=/opt/pytorch211
    export PATH=/opt/pytorch211/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/rocm/lib:/opt/rocm/lib64
    export GPU_ARCHS=gfx942

    if ! command -v git; then apt-get update -qq && apt-get install -y -qq git; fi

    # vLLM pins the aiter it was validated against in its ROCm dockerfiles -- EXCEPT when it
    # does not: 0.27.1 spells that pin `main`, a moving branch, so 600659 built whatever main
    # happened to be that day. Pass AITER_PIN explicitly to get a tagged release instead; the
    # tags state which ROCm they target, `main` states nothing.
    AITER_PIN="${AITER_PIN:-}"
    if [[ -z "${AITER_PIN}" ]]; then
        AITER_PIN="$(grep -rhoE "AITER_(BRANCH|COMMIT|BRANCH_OR_COMMIT)=\"?[A-Za-z0-9._/-]+" \
            /opt/vllm-src/docker/ 2>/dev/null | head -1 | sed "s/.*=\"\\?//" || true)"
        echo "aiter pin from vllm docker files: ${AITER_PIN:-<none, using main>}"
    else
        echo "aiter pin from AITER_PIN: ${AITER_PIN}"
    fi

    cd /opt
    rm -rf aiter-src
    git clone https://github.com/ROCm/aiter.git aiter-src
    cd aiter-src
    if [[ -n "${AITER_PIN}" ]]; then git checkout "${AITER_PIN}"; fi
    git submodule sync
    git submodule update --init --recursive
    git log -1 --oneline

    /opt/pytorch211/bin/python -m pip install --no-cache-dir uv
    /opt/pytorch211/bin/python -m uv pip install \
      --python /opt/pytorch211/bin/python \
      --no-cache \
      ninja cmake "pybind11==3.0.*" pandas
    # pybind11 PINNED below 3.1. pybind11 3.1.0 (2026-08-06) bumped its internals id from v11 to
    # v12, and pybind11 refuses to share registered types across modules whose internals differ.
    # The pip headers win over the torch bundled ones on the include path, so an unpinned build
    # compiles aiter against v12 while the torch extensions stay v11 -- and every torch.Tensor
    # handed to an aiter op is then rejected as the wrong type. That is the
    # "invalid argument for fmha_v3_varlen_fwd" in 600662/600673/600688/600689: an ABI split,
    # not a bad kernel. Upstream: ROCm/aiter#4770, vllm-project/vllm#52400.

    # --no-build-isolation: aiter setup imports torch, which must be the image torch.
    /opt/pytorch211/bin/python -m uv pip install \
      --python /opt/pytorch211/bin/python \
      --no-build-isolation \
      --no-cache \
      .

    # The ASM v3 varlen path cannot express our mask. cpp_itfs/mha_fwd.cu maps causal to
    # mask 2 ONLY when seqlen_q == seqlen_k, and returns -1 otherwise -- which surfaces from
    # asm_mha_varlen_fwd.cu:494 as "invalid argument for fmha_v3_varlen_fwd". Chunked prefill
    # against a warm prefix cache ALWAYS has q shorter than k (88-95% hit rate on our arms), so
    # every MLA prefill hits it. can_impl_fmha_v3_fwd claims the shape anyway: its one 192x128
    # guard is skipped when block_table is set, and vLLM always pages. Adding the missing
    # condition makes the predicate fall through to the else branch -- aiter CK-tile
    # mha_varlen_fwd, still aiter -- instead of crashing. Upstream bug: the predicate promises
    # what the dispatcher cannot serve.
    /opt/pytorch211/bin/python - <<PATCH
import pathlib
p = pathlib.Path("/opt/pytorch211/lib/python3.12/site-packages/aiter/ops/mha.py")
s = p.read_text()
tail = "        return ret\n"
old = "        if hdim_q == 192 and hdim_v == 128 and block_table is None:\n            ret = ret and not _fmha_kv_byte_extent_ge_u32(max_seqlen_k, k, v)\n" + tail
assert s.count(old) == 1, "anchor count " + str(s.count(old))
new = old[: -len(tail)] + "        ret = ret and ((not causal) or (max_seqlen_q == max_seqlen_k))\n" + tail
p.write_text(s.replace(old, new, 1))
print("patched can_impl_fmha_v3_fwd varlen: mask/seqlen guard added")
PATCH

    export VLLM_ROCM_USE_AITER=1
    export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

    /opt/pytorch211/bin/python - <<PY
import sys

import torch

import aiter

print("Python:  ", sys.executable)
print("aiter:   ", aiter.__file__)
print("PyTorch: ", torch.__version__)
print("HIP:     ", torch.version.hip)
print("GPUs:    ", torch.cuda.device_count())

assert sys.executable == "/opt/pytorch211/bin/python"
assert aiter.__file__.startswith("/opt/pytorch211/") or aiter.__file__.startswith("/opt/aiter-src/")
assert torch.__version__.startswith("2.11.0+rocm7.2")
assert torch.cuda.device_count() >= 1

# vLLM must both see the env knob and resolve its aiter MoE path -- the package being
# importable is not the same thing as vLLM electing to use it. 0.23 centralizes this in
# vllm/_aiter_ops.py (the old rocm_aiter_fused_moe module is gone, 595044).
import vllm.envs as envs

assert envs.VLLM_ROCM_USE_AITER, "vLLM does not see VLLM_ROCM_USE_AITER=1"

from vllm._aiter_ops import is_aiter_found, rocm_aiter_ops

assert is_aiter_found(), "vLLM does not find the aiter package"
assert rocm_aiter_ops.is_fused_moe_enabled(), "vLLM aiter fused-MoE path reports disabled"

# flash-attn from the previous patch must still import -- both live in the same venv.
from flash_attn import flash_attn_varlen_func  # noqa: F401

print("AITER + VLLM MOE ENABLEMENT VALIDATION PASSED")
PY

    rm -rf /root/.cache/pip /root/.cache/uv
  '

enroot export \
  --output "$TEMP_IMAGE" \
  "$BUILD_NAME"

test -s "$TEMP_IMAGE"

if [[ ! -e "$BACKUP_IMAGE" ]]; then
    mv "$FINAL_IMAGE" "$BACKUP_IMAGE"
else
    rm -f "$FINAL_IMAGE"
fi

mv "$TEMP_IMAGE" "$FINAL_IMAGE"

sha256sum "$FINAL_IMAGE" |
  tee "${FINAL_IMAGE}.sha256"

enroot remove --force "$BUILD_NAME" 2>/dev/null || true

ls -lh "$FINAL_IMAGE" "${FINAL_IMAGE}.sha256" "$BACKUP_IMAGE"

echo "========================================"
echo "AITER IMAGE PATCH PASSED"
echo "========================================"
