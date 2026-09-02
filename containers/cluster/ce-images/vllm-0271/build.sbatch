#!/usr/bin/env bash
# Build the vLLM 0.27.1 inference image on a compute node.
#
# Sibling of ce-images/vllm/build.sbatch. Both may be built; they write different artifacts and
# ce-images/vllm (0.23.0) stays the fallback until this one has SERVED oss120b under an accuracy
# gate. It asks for no GPUs, but the mi300 node exposes /dev/kfd anyway and
# build.sh hands it to podman, so a RUN step sees gfx942 (measured, job 619976). It still proves
# less than a serve: an import and a JIT build are not a kernel launch. See LIMITATION in the
# Dockerfile.
#
# Cost: one mi300 node, same shape as the 0.23.0 build. 8 h requested, 4-6 h expected.
#
# SUBMIT WITH THE DIRECTORY NAMED. The guard below exits in about a second without it, which is
# what three failed jobs looked like:
#
#   sbatch --export=ALL,VLLM_0271_DIR=$PWD/containers/cluster/ce-images/vllm-0271 \
#          containers/cluster/ce-images/vllm-0271/build.sbatch
#
# or equivalently `sbatch --chdir=containers/cluster/ce-images/vllm-0271 build.sbatch`, since the
# fallback reads SLURM_SUBMIT_DIR.
#
#SBATCH --job-name=build-vllm-0271-image
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/build-vllm-0271-image-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/build-vllm-0271-image-%j.err

set -Eeuxo pipefail

# Slurm SPOOLS the batch script, so BASH_SOURCE points at /var/spool/slurmd -- pass the directory
# rather than deriving it (this is exactly what killed build job 600467).
VLLM_0271_DIR="${VLLM_0271_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -x "${VLLM_0271_DIR}/build.sh" ]]; then
    echo "no build.sh under ${VLLM_0271_DIR}. Submit with" >&2
    echo "  sbatch --export=ALL,VLLM_0271_DIR=<abs path to ce-images/vllm-0271> ..." >&2
    echo "or --chdir into that directory." >&2
    exit 2
fi

export OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?}/ce-images/optarena-vllm-0271-candidate.sqsh}"

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

"${VLLM_0271_DIR}/build.sh"

ls -lh "${OUTPUT_SQSH}"
sha256sum "${OUTPUT_SQSH}" | tee "${OUTPUT_SQSH}.sha256"
echo "VLLM 0.27.1 CANDIDATE READY: ${OUTPUT_SQSH}"
echo "UNVALIDATED. The build gate proves imports and an argument parse, nothing more. The aiter"
echo "master switch also imported cleanly and then died under load with fmha_v3_varlen_fwd."
echo "Serve oss120b under the VLLM_EXTRA_ARGS in IMAGE_REQUIREMENTS.md and run an accuracy gate"
echo "on gfx942 before promoting. ce-images/vllm (0.23.0) stays the fallback until it passes."
