#!/usr/bin/env bash
#SBATCH --job-name=vllm-base-torch
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=logs/build-vllm-%j.out
#SBATCH --error=logs/build-vllm-%j.err

set -Eeuxo pipefail
umask 0022

ROOT="${VLLM_BUILD_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"

BASE_IMAGE="${ROOT}/phase1-passed/rocm723-ofi-host-diag-phase1.sqsh"
BUILD_NAME="rocm723-vllm-base-torch"
OUTPUT_IMAGE="${ROOT}/containers/rocm723-ofi-vllm-0.23.0-base-torch.sqsh"
INNER_SCRIPT="${ROOT}/build/build-vllm-base-torch-inner.sh"

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
    bash "$INNER_SCRIPT"

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
echo "SQUASHFS READY"
echo "========================================"
