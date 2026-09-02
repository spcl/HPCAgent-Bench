#!/usr/bin/env bash
# Build the vLLM inference image on a compute node.
#
# Replaces the six-job chain in ce-images/inference/build/build-chain.sh (phase1 OFI, phase1 gate,
# pytorch 2.11, numpy, vLLM 0.23.0, flash-attn) plus the two in-place patch jobs that followed it,
# with one job. It asks for no GPUs, but the mi300 node exposes /dev/kfd anyway and
# build.sh hands it to podman, so a RUN step sees gfx942 (measured, job 619976). It still proves
# less than a serve: an import and a JIT build are not a kernel launch. See LIMITATION in the
# Dockerfile.
#
# Cost: one mi300 node. The stages it merges asked for 4 h (phase1) + 2 h (pytorch) + 2 h (vLLM) +
# 2 h (aiter) of wall clock, and the aiter JIT prebuild adds its own compile, so 8 h is the safe
# request and roughly 4-6 h the expectation.
#
#   sbatch --chdir=containers/cluster/ce-images/vllm build.sbatch
#   OUTPUT_SQSH=$SCRATCH/ce-images/optarena-vllm-candidate.sqsh sbatch ... build.sbatch
#
#SBATCH --job-name=build-vllm-image
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/build-vllm-image-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/build-vllm-image-%j.err

set -Eeuxo pipefail

# Slurm SPOOLS the batch script, so BASH_SOURCE points at /var/spool/slurmd -- pass the directory
# rather than deriving it (this is exactly what killed build job 600467). --chdir sets it for free.
VLLM_DIR="${VLLM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -x "${VLLM_DIR}/build.sh"

export OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?}/ce-images/optarena-vllm-candidate.sqsh}"

# Refuse to write over an image some EDF points a running job at: overwriting a mounted squashfs is
# how a live arm starts reading a half-written inode table. Ask the EDFs rather than a hard-coded
# name, since they are what a job actually resolves. Build to a candidate, verify, promote by
# rename when nothing is using it.
edf_images() {
    grep -hoE '^[[:space:]]*image[[:space:]]*=[[:space:]]*"[^"]+"' "${HOME}/.edf"/*.toml 2>/dev/null \
      | sed -E 's/.*"(.*)"/\1/' | sed -E "s|\\\$\\{SCRATCH\\}|${SCRATCH:-}|g; s|\\\$SCRATCH|${SCRATCH:-}|g"
}
if edf_images | grep -qxF "${OUTPUT_SQSH}"; then
    echo "refusing to overwrite ${OUTPUT_SQSH}: an EDF in ${HOME}/.edf mounts it, so a running" >&2
    echo "arm would read a half-written squashfs. Build to a candidate name and promote by rename." >&2
    exit 2
fi

"${VLLM_DIR}/build.sh"

ls -lh "${OUTPUT_SQSH}"
sha256sum "${OUTPUT_SQSH}" | tee "${OUTPUT_SQSH}.sha256"
echo "VLLM IMAGE READY: ${OUTPUT_SQSH}"
echo "NOT VERIFIED: the GPU gates the six-job chain ran (device_count, allreduce, one-rank RCCL"
echo "over CXI, MLA prefill selector) need hardware. Smoke-test before promoting."
