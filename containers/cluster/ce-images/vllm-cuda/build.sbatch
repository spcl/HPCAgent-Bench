#!/usr/bin/env bash
# Build AND verify the GH200 vLLM image on one Daint node.
#
# No partition or account is written here; sbatch reads both from the environment (or pass -A/-p):
#
#   export SBATCH_ACCOUNT=<project> SBATCH_PARTITION=normal
#   cd containers/cluster/ce-images && IMAGE_DIR=$PWD/vllm-cuda sbatch vllm-cuda/build.sbatch
#
# IMAGE_DIR, not --chdir: SLURM_SUBMIT_DIR is where sbatch was INVOKED, whatever --chdir says.
#
# COST: one node, well under an hour -- a ~14 GB pull (cached under $SCRATCH/base-images after the
# first), one small gate layer, the squashfs import, then verify_image.py inside the result. The
# serving half a build cannot answer is inference/serve-daint.sbatch MODE=smoke, per model.
#
#SBATCH --job-name=build-vllm-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --no-requeue

set -Eeuo pipefail

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_nid<node>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0

# Slurm SPOOLS the batch script; the directory is passed in or is the submit directory (job 600467).
IMAGE_DIR="${IMAGE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -x "${IMAGE_DIR}/build.sh"
# shellcheck source=../build_common.sh
source "${IMAGE_DIR}/../build_common.sh"
# shellcheck source=../images.env
source "${IMAGE_DIR}/../images.env"

export OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?}/ce-images/${INFERENCE_VLLM_CUDA_SQSH%.sqsh}-candidate.sqsh}"

# Refuse to write over an image an EDF mounts: a running server would read a half-written squashfs.
if grep -hoE '^[[:space:]]*image[[:space:]]*=[[:space:]]*"[^"]+"' "${HOME}/.edf"/*.toml 2>/dev/null \
    | sed -E 's/.*"(.*)"/\1/' | grep -qxF "${OUTPUT_SQSH}"; then
    echo "refusing to overwrite ${OUTPUT_SQSH}: an EDF in ${HOME}/.edf mounts it" >&2
    exit 2
fi

"${IMAGE_DIR}/build.sh"

sha256sum "${OUTPUT_SQSH}" | tee "${OUTPUT_SQSH}.sha256"
ce_verify_candidate "${INFERENCE_VLLM_CUDA_TEMPLATE}" "${OUTPUT_SQSH}" vllm-cuda
echo "VLLM GH200 CANDIDATE BUILT AND VERIFIED: ${OUTPUT_SQSH}"
echo "Smoke one model on it before promoting (README.md, Daint section), then:"
echo "  CE_PLATFORM=gh200 containers/cluster/ce-images/promote_image.sh vllm-cuda"
