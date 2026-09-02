#!/usr/bin/env bash
# Build the SGLang inference image on a compute node.
#
# Replaces a PULL plus two out-of-image steps: install-flydsl.sbatch, which built a --target prefix
# on scratch that the arms reached through PYTHONPATH, and prebuild-aiter-jit.sbatch, whose whole
# job was to warm a cache the image can simply carry. It asks for no GPUs, because a podman build
# cannot use them -- which is also the reason it proves less. See LIMITATION in the Dockerfile.
#
# Cost: one mi300 node. The 52 GB base pull dominates; after it, the Slingshot stack (libcxi,
# libfabric, aws-ofi-nccl) and the aiter JIT prebuild are each minutes. 4 h is the safe request and
# roughly 1-2 h the expectation, most of it network.
#
#   sbatch --chdir=containers/cluster/ce-images/sglang build.sbatch
#   OUTPUT_SQSH=$SCRATCH/ce-images/optarena-sglang-candidate.sqsh sbatch ... build.sbatch
#
#SBATCH --job-name=build-sglang-image
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/build-sglang-image-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/build-sglang-image-%j.err

set -Eeuxo pipefail

# Slurm SPOOLS the batch script, so BASH_SOURCE points at /var/spool/slurmd -- pass the directory
# rather than deriving it (this is exactly what killed build job 600467). --chdir sets it for free.
SGLANG_DIR="${SGLANG_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -x "${SGLANG_DIR}/build.sh"

export OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?}/ce-images/optarena-sglang-candidate.sqsh}"

# Refuse to write over an image some EDF points a running job at: overwriting a mounted squashfs is
# how a live arm starts reading a half-written inode table. This matters more here than anywhere --
# 12 of the 18 v9 arms mount the SGLang image. Ask the EDFs rather than a hard-coded name, since
# they are what a job actually resolves.
edf_images() {
    grep -hoE '^[[:space:]]*image[[:space:]]*=[[:space:]]*"[^"]+"' "${HOME}/.edf"/*.toml 2>/dev/null \
      | sed -E 's/.*"(.*)"/\1/' | sed -E "s|\\\$\\{SCRATCH\\}|${SCRATCH:-}|g; s|\\\$SCRATCH|${SCRATCH:-}|g"
}
if edf_images | grep -qxF "${OUTPUT_SQSH}"; then
    echo "refusing to overwrite ${OUTPUT_SQSH}: an EDF in ${HOME}/.edf mounts it, so a running" >&2
    echo "arm would read a half-written squashfs. Build to a candidate name and promote by rename." >&2
    exit 2
fi

"${SGLANG_DIR}/build.sh"

ls -lh "${OUTPUT_SQSH}"
sha256sum "${OUTPUT_SQSH}" | tee "${OUTPUT_SQSH}.sha256"
echo "SGLANG IMAGE READY: ${OUTPUT_SQSH}"
echo "NOT VERIFIED: the self-contained CXI stack has never opened a /dev/cxi, and the prebuilt"
echo "aiter modules have never launched. Smoke-test with inference/smoke-kimi-sglang.sbatch"
echo "before promoting, and measure tok/s before enabling the OFI plugin in any EDF."
