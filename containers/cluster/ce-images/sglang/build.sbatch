#!/usr/bin/env bash
# Build the SGLang inference image on a compute node.
#
# It asks for no GPUs, but the mi300 node exposes /dev/kfd anyway and build.sh hands it to podman,
# so a RUN step sees gfx942. It still proves less than a serve: an import and a JIT build are not a
# kernel launch. See LIMITATION in the Dockerfile.
#
# Cost: one mi300 node. The 52 GB base pull dominates; after it, cupy and the aiter JIT prebuild
# are each minutes. 4 h is the safe request and roughly 1-2 h the expectation, most of it network.
#
#   sbatch --chdir=containers/cluster/ce-images/sglang build.sbatch
#   OUTPUT_SQSH=$SCRATCH/ce-images/hpcagent-bench-sglang-candidate.sqsh sbatch ... build.sbatch
#
#SBATCH --job-name=build-sglang-image
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 96 PHYSICAL cores, not 192 logical: the node is Sockets=4 CoresPerSocket=24
# ThreadsPerCore=2, and an SMT sibling is not extra compute for a compile.
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --no-requeue

set -Eeuxo pipefail

# Slurm propagates the submitter's core limit and a dump lands in the CWD (an inode-quota checkout).
ulimit -c 0

# Slurm SPOOLS the batch script, so BASH_SOURCE points at /var/spool/slurmd -- pass the directory
# rather than deriving it. $PWD, not SLURM_SUBMIT_DIR: --chdir sets the job's cwd but not
# SLURM_SUBMIT_DIR, which stays where sbatch was run.
IMAGE_DIR="${IMAGE_DIR:-$PWD}"
test -x "${IMAGE_DIR}/build.sh" || { echo "no build.sh under IMAGE_DIR=${IMAGE_DIR}" >&2; exit 2; }

# shellcheck source=../build_common.sh
source "${IMAGE_DIR}/../build_common.sh"
# shellcheck source=../images.env
source "${IMAGE_DIR}/../images.env"
ce_require_partition sglang || exit 2
export OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?}/ce-images/${INFERENCE_SGLANG_CANDIDATE}}"

ce_refuse_mounted "${OUTPUT_SQSH}" || exit 2

"${IMAGE_DIR}/build.sh"

ls -lh "${OUTPUT_SQSH}"
sha256sum "${OUTPUT_SQSH}" | tee "${OUTPUT_SQSH}.sha256"
echo "SGLANG IMAGE READY: ${OUTPUT_SQSH}"
echo "NOT VERIFIED: the build has no Slingshot NIC. The aiter prebuild does launch kernels -- the"
echo "build gets /dev/kfd and /dev/dri -- but serving is a separate question: smoke-test with"
echo "inference/smoke-kimi-sglang.sbatch before promoting."
