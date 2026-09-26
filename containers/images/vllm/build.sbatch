#!/usr/bin/env bash
# Build the vLLM inference image on a compute node.
#
# It asks for no GPUs, but the mi300 node exposes /dev/kfd anyway and build.sh hands it to podman,
# so a RUN step sees gfx942. It still proves less than a serve: an import and a JIT build are not a
# kernel launch. See LIMITATION in the Dockerfile.
#
# Cost: one mi300 node; 8 h is the safe request and roughly 4-6 h the expectation.
#
#   sbatch --chdir=containers/images/vllm build.sbatch
#   OUTPUT_SQSH=$SCRATCH/ce-images/hpcagent-bench-vllm-candidate.sqsh sbatch ... build.sbatch
#
#SBATCH --job-name=build-vllm-image
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 96 PHYSICAL cores, not 192 logical: the node is Sockets=4 CoresPerSocket=24
# ThreadsPerCore=2, and an SMT sibling is not extra compute for a compile.
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --no-requeue

set -Eeuxo pipefail

# Slurm propagates the submitter's core limit and a dump lands in the CWD (an inode-quota checkout).
ulimit -c 0

# Slurm SPOOLS the batch script, so BASH_SOURCE points at /var/spool/slurmd -- pass the directory
# rather than deriving it.
IMAGE_DIR="${IMAGE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -x "${IMAGE_DIR}/build.sh"

# shellcheck source=../build_common.sh
source "${IMAGE_DIR}/../build_common.sh"
# shellcheck source=../images.env
source "${IMAGE_DIR}/../images.env"
ce_require_partition vllm || exit 2
export OUTPUT_SQSH="${OUTPUT_SQSH:-${CE_IMAGES:?}/${INFERENCE_VLLM_CANDIDATE}}"

ce_refuse_mounted "${OUTPUT_SQSH}" || exit 2

"${IMAGE_DIR}/build.sh"

ls -lh "${OUTPUT_SQSH}"
sha256sum "${OUTPUT_SQSH}" | tee "${OUTPUT_SQSH}.sha256"
echo "VLLM IMAGE READY: ${OUTPUT_SQSH}"
echo "NOT VERIFIED: the GPU gates (device_count, allreduce, one-rank RCCL over CXI, MLA prefill"
echo "selector) need a serving job. Smoke-test before promoting."
