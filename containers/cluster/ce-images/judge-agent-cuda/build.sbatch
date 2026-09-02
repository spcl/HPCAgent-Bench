#!/usr/bin/env bash
# Build the CUDA / GH200 judge+agent image on a compute node.
#
# THIS DOES NOT RUN ON BEVERIN. `sinfo` on alps-beverin lists mi300 and mi200 only, both x86_64 and
# both AMD -- there is no GH200 partition to submit to, so `--partition=mi300` (the right answer
# for every other sbatch in this repo) is exactly the wrong one here. Submit this from an Alps
# GH200 cluster (daint / clariden), where `normal` is the GPU partition; `debug` is too short.
# Set PARTITION to override without editing this file.
#
# Cost, at one node for the full wall clock: gcc 16.1 bootstrap and llvm 22.1 are roughly 1.5 h
# each on a 72-core Grace, MAGMA against a single cuda_arch about 1 h, the PETSc layer about 1 h,
# the rest of the spack stack about 1 h, the NGC base pull and the python/torch layers about 1 h.
# Call it 7-9 h typical and 12 h budgeted, so ~12 node-hours worst case for one node. That is
# inside a 28-of-36 node budget by a wide margin because it is ONE node; the constraint here is
# wall clock, not width.
#
#   PARTITION=normal sbatch --chdir=<repo>/containers/cluster/ce-images/judge-agent-cuda build.sbatch
#
# The output name is DELIBERATELY not a live image. Overwriting a squashfs some EDF mounts is how a
# running arm starts reading a half-written inode table. Build to a candidate, verify, promote by
# rename when nothing is using it.
#
#SBATCH --job-name=build-judge-agent-cuda
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/logs/build-judge-agent-cuda-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/logs/build-judge-agent-cuda-%j.err

set -Eeuxo pipefail

# Slurm SPOOLS the batch script, so BASH_SOURCE points at /var/spool/slurmd -- pass the directory
# rather than deriving it (this is what killed build job 600467).
IMAGE_DIR="${IMAGE_DIR:-${SLURM_SUBMIT_DIR:?}}"
test -x "${IMAGE_DIR}/build.sh"

export OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?}/ce-images/optarena-judge-agent-cuda-candidate.sqsh}"

# Refuse to write over an image some EDF points a running job at. Ask the EDFs rather than keeping
# a hard-coded filename in step with the live image by hand: the cost is that it also protects
# images nothing currently mounts, which is the safe direction to be wrong in.
edf_images() {
    grep -hoE '^[[:space:]]*image[[:space:]]*=[[:space:]]*"[^"]+"' "${HOME}/.edf"/*.toml 2>/dev/null \
      | sed -E 's/.*"(.*)"/\1/' | sed -E "s|\\\$\\{SCRATCH\\}|${SCRATCH:-}|g; s|\\\$SCRATCH|${SCRATCH:-}|g"
}
if edf_images | grep -qxF "${OUTPUT_SQSH}"; then
    echo "refusing to overwrite ${OUTPUT_SQSH}: an EDF in ${HOME}/.edf mounts it, so a running" >&2
    echo "arm would read a half-written squashfs. Build to a candidate and promote by rename." >&2
    exit 2
fi

"${IMAGE_DIR}/build.sh"

ls -lh "${OUTPUT_SQSH}"
sha256sum "${OUTPUT_SQSH}" | tee "${OUTPUT_SQSH}.sha256"

# The half of the image the BUILD cannot answer. A build container has no driver, so PAPI's cuda
# and nvml components can only be shown to be COMPILED IN there; whether they come up is a
# driver/device/permission question that needs a real GPU. Same for cupy and jax actually seeing a
# device. Enumerating the component's events first is what brings a lazily-initialized PAPI 7
# component up -- reading its status flag without that reports a working component as disabled.
srun --environment="${OUTPUT_SQSH}" --gpus-per-node=1 bash -c '
  set -eu
  python3 -c "import cupy; print(\"cupy device count\", cupy.cuda.runtime.getDeviceCount())"
  python3 -c "import jax; print(\"jax devices\", jax.devices())"
  papi_component_avail | sed -n "1,60p"
  papi_native_avail --help >/dev/null 2>&1 || true
  papi_native_avail 2>/dev/null | grep -c "cuda:::" || echo "cuda component enumerated no events"
  nvidia-smi -L
  ncu --version | head -1
  nsys --version | head -1
'

echo "CUDA JUDGE+AGENT IMAGE READY: ${OUTPUT_SQSH}"
