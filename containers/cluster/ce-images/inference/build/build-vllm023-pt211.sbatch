#!/usr/bin/env bash
#SBATCH --job-name=vllm023-pt211
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/build-vllm023-pt211-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/build-vllm023-pt211-%j.err

set -Eeuxo pipefail
umask 0022

ROOT="${VLLM_BUILD_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"

# VLLM_VERSION selects the tag AND the artefact name, so a newer build lands BESIDE the old
# image rather than over it -- every measured arm so far ran on 0.23.0 and a silent overwrite
# would make those runs unreproducible.
VLLM_VERSION="${VLLM_VERSION:-0.23.0}"
# The phase-1 candidate was not kept; pass BASE_IMAGE to build on a surviving image
# (`...before-aiter.sqsh` is the closest equivalent: PyTorch 2.11 + OFI + flash-attn,
# no aiter, so a pinned aiter can be installed afterwards without fighting a stale one).
BASE_IMAGE="${BASE_IMAGE:-${ROOT}/containers/rocm723-pytorch211-ofi-phase1-candidate.sqsh}"
BUILD_NAME="rocm723-vllm${VLLM_VERSION//./}-pytorch211"
INNER_SCRIPT="${ROOT}/build/build-vllm023-pt211-inner.sh"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-${ROOT}/containers/rocm723-vllm-${VLLM_VERSION}-pytorch211-ofi.sqsh}"

[[ -d "${ROOT}/build" ]] || {
    echo "VLLM_BUILD_ROOT=${ROOT} has no build/ -- Slurm spools the batch script, so the
BASH_SOURCE fallback resolves to the spool dir. Submit with
--export=ALL,VLLM_BUILD_ROOT=<inference dir>." >&2
    exit 2
}
test -f "$BASE_IMAGE"
test -x "$INNER_SCRIPT"

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}"
echo "Base image: ${BASE_IMAGE}"
echo "Output image: ${OUTPUT_IMAGE}"

enroot remove --force "$BUILD_NAME" 2>/dev/null || true

enroot create \
  --name "$BUILD_NAME" \
  "$BASE_IMAGE"

enroot start \
  --root \
  --rw \
  --mount "${ROOT}:${ROOT}" \
  "$BUILD_NAME" \
  env VLLM_VERSION="${VLLM_VERSION}" bash "$INNER_SCRIPT"

rm -f \
  "$OUTPUT_IMAGE" \
  "${OUTPUT_IMAGE}.sha256"

enroot export \
  --output "$OUTPUT_IMAGE" \
  "$BUILD_NAME"

sha256sum "$OUTPUT_IMAGE" |
  tee "${OUTPUT_IMAGE}.sha256"

ls -lh \
  "$OUTPUT_IMAGE" \
  "${OUTPUT_IMAGE}.sha256"

echo "========================================"
echo "VLLM PYTORCH 2.11 IMAGE READY"
echo "========================================"
