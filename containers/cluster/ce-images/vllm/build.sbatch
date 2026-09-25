#!/usr/bin/env bash
# Build the vLLM inference image on a compute node.
#
# It asks for no GPUs, but the mi300 node exposes /dev/kfd anyway and build.sh hands it to podman,
# so a RUN step sees gfx942. It still proves less than a serve: an import and a JIT build are not a
# kernel launch. See LIMITATION in the Dockerfile.
#
# Cost: one mi300 node; 8 h is the safe request and roughly 4-6 h the expectation.
#
#   sbatch --chdir=containers/cluster/ce-images/vllm build.sbatch
#   OUTPUT_SQSH=$SCRATCH/ce-images/hpcagent-bench-vllm-candidate.sqsh sbatch ... build.sbatch
#
#SBATCH --job-name=build-vllm-image
#SBATCH --partition=mi300
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

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_nid<node>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0

# Slurm SPOOLS the batch script, so BASH_SOURCE points at /var/spool/slurmd -- pass the directory
# rather than deriving it. --chdir sets it for free. IMAGE_DIR, the same name all five builders take.
IMAGE_DIR="${IMAGE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -x "${IMAGE_DIR}/build.sh"

# shellcheck source=../images.env
source "${IMAGE_DIR}/../images.env"
ce_require_partition vllm || exit 2
export OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?}/ce-images/${INFERENCE_VLLM_CANDIDATE}}"

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

"${IMAGE_DIR}/build.sh"

ls -lh "${OUTPUT_SQSH}"
sha256sum "${OUTPUT_SQSH}" | tee "${OUTPUT_SQSH}.sha256"
echo "VLLM IMAGE READY: ${OUTPUT_SQSH}"
echo "NOT VERIFIED: the GPU gates the six-job chain ran (device_count, allreduce, one-rank RCCL"
echo "over CXI, MLA prefill selector) need hardware. Smoke-test before promoting."
