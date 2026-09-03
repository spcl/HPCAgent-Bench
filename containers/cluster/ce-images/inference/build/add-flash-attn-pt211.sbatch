#!/usr/bin/env bash
# Add upstream flash-attn to the pt211 vLLM image so K2.7 gets an MLA PREFILL backend.
#
# 589001 died with:
#   ValueError: No valid MLA prefill backend found ... Reasons: {FLASH_ATTN: [required
#   dependencies not available]}
# MLA decode already worked there (TRITON_MLA); only prefill was missing. vLLM's ROCm path
# wants exactly one import -- `from flash_attn import flash_attn_varlen_func`
# (vllm/v1/attention/backends/fa_utils.py:34) -- so plain upstream flash-attn is the whole fix.
# aiter is NOT needed: its only tie-in here is QuarkOCP_MX quantization and K2.7 is
# compressed-tensors.
#
# Triton backend, not composable-kernel: FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE makes setup.py
# set SKIP_CK_BUILD, turning a multi-hour gfx942 CK compile into a minutes-long pure-Triton
# install. The image already ships triton 3.7.1. The sdist DOES carry csrc/composable_kernel, so
# a CK rebuild for faster kernels stays possible later.
#
# WARNING the same env var is read at IMPORT time (flash_attn_interface.py:11): without it
# flash_attn imports flash_attn_2_cuda, which a Triton-only install does not have. vLLM catches
# that ImportError and silently reports the backend unavailable -- 589001 again, but with the
# package installed and looking fine. It MUST also be in the EDF.
#
# Patches the image IN PLACE, keeping a .before-flash-attn backup, exactly like
# add-numpy-pytorch211.sbatch. The name stays rocm723-vllm-0.23.0-pytorch211-ofi so every env
# file and the existing EDF keep working untouched.
#SBATCH --job-name=pt211-flash-attn
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/add-flash-attn-pt211-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/add-flash-attn-pt211-%j.err

set -Eeuxo pipefail
umask 0022

FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"

# Derive from SLURM_SUBMIT_DIR, not BASH_SOURCE: slurm copies the script to
# /var/spool/slurmd/job<id>/slurm_script, so BASH_SOURCE walks up to "/". Submit from
# containers/cluster/ce-images/inference, which is four levels below the repo root.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -n "${VLLM_BUILD_ROOT:-}" ]]; then
    # build-chain.sh rebuild: images land in the self-contained work tree, like every sibling step.
    DEFAULT_IMAGE_DIR="${VLLM_BUILD_ROOT}/containers"
else
    # Standalone patch of the production image, which lives BESIDE the repo, not inside it.
    DEFAULT_IMAGE_DIR="$(dirname -- "$(cd -- "${SUBMIT_DIR}/../../../.." && pwd)")/ce-images"
fi

CE_IMAGE_DIR="${CE_IMAGE_DIR:-$DEFAULT_IMAGE_DIR}"

SOURCE_IMAGE="${CE_IMAGE_DIR}/rocm723-vllm-0.23.0-pytorch211-ofi.sqsh"
TEMP_IMAGE="${CE_IMAGE_DIR}/rocm723-vllm-0.23.0-pytorch211-ofi.fa.tmp.sqsh"
FINAL_IMAGE="${CE_IMAGE_DIR}/rocm723-vllm-0.23.0-pytorch211-ofi.sqsh"
BACKUP_IMAGE="${CE_IMAGE_DIR}/rocm723-vllm-0.23.0-pytorch211-ofi.before-flash-attn.sqsh"

BUILD_NAME="rocm723-pt211-flash-attn-patch"

test -f "$SOURCE_IMAGE"

echo "Host:        $(hostname)"
echo "Source:      ${SOURCE_IMAGE}"
echo "Final:       ${FINAL_IMAGE}"
echo "flash-attn:  ${FLASH_ATTN_VERSION}"

enroot remove --force "$BUILD_NAME" 2>/dev/null || true
rm -f "$TEMP_IMAGE" "${TEMP_IMAGE}.sha256"

enroot create \
  --name "$BUILD_NAME" \
  "$SOURCE_IMAGE"

enroot start \
  --root \
  --rw \
  --env "FLASH_ATTN_VERSION=${FLASH_ATTN_VERSION}" \
  "$BUILD_NAME" \
  bash -lc '
    set -Eeuxo pipefail

    export VIRTUAL_ENV=/opt/pytorch211
    export PATH=/opt/pytorch211/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/rocm/lib:/opt/rocm/lib64

    # Triton path -- skips the composable-kernel compile. Build AND import both read this.
    export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
    export FLASH_ATTENTION_SKIP_CK_BUILD=TRUE

    # uv is not in the image; pip installs it, then uv does the real work.
    /opt/pytorch211/bin/python -m pip install --no-cache-dir uv

    # --no-build-isolation: flash-attn setup.py imports torch, which must be the image torch
    # rather than a fresh wheel pulled into an isolated build env.
    /opt/pytorch211/bin/python -m uv pip install \
      --python /opt/pytorch211/bin/python \
      --no-build-isolation \
      --no-cache \
      "flash-attn==${FLASH_ATTN_VERSION}"

    /opt/pytorch211/bin/python - <<PY
import os
import sys

import torch

assert os.environ.get("FLASH_ATTENTION_TRITON_AMD_ENABLE") == "TRUE"

import flash_attn
from flash_attn import flash_attn_varlen_func

print("Python:      ", sys.executable)
print("flash-attn:  ", flash_attn.__version__)
print("flash-attn f:", flash_attn.__file__)
print("PyTorch:     ", torch.__version__)
print("HIP:         ", torch.version.hip)
print("GPU count:   ", torch.cuda.device_count())

assert sys.executable == "/opt/pytorch211/bin/python"
assert flash_attn.__file__.startswith("/opt/pytorch211/")
assert torch.__version__.startswith("2.11.0+rocm7.2")
assert torch.cuda.device_count() >= 1

# Triton backend, not the CK extension -- prove which one got imported.
from flash_attn import flash_attn_interface

assert flash_attn_interface.USE_TRITON_ROCM, "imported the CK path, not triton_amd"

# vLLMs OWN predicate. This is the gate 589001 failed; asserting the import alone would
# not catch a vLLM-side signature change.
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available

assert is_flash_attn_varlen_func_available(), "vLLM still reports flash-attn unavailable"

# And the selector itself must now hand back FLASH_ATTN for a K2.7-shaped MLA config.
from vllm.v1.attention.backends.mla.prefill.flash_attn import FlashAttnPrefillBackend

assert FlashAttnPrefillBackend.is_available(), "FlashAttnPrefillBackend still unavailable"
assert FlashAttnPrefillBackend.get_name() == "FLASH_ATTN"

# Real varlen call on device: bf16, MLA-ish head dims, so a broken triton build fails HERE
# and not four hours into a K2.7 load.
n_heads, head_dim, total_q = 8, 128, 64
q = torch.randn(total_q, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")
k = torch.randn(total_q, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")
v = torch.randn(total_q, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")
cu = torch.tensor([0, total_q], dtype=torch.int32, device="cuda")

out = flash_attn_varlen_func(q, k, v, cu, cu, total_q, total_q, causal=True)
if isinstance(out, tuple):
    out = out[0]

assert out.shape == (total_q, n_heads, head_dim), out.shape
assert torch.isfinite(out).all(), "flash_attn_varlen_func returned non-finite values"

print("FLASH-ATTN + MLA PREFILL VALIDATION PASSED")
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
echo "FLASH-ATTN IMAGE PATCH PASSED"
echo "========================================"
