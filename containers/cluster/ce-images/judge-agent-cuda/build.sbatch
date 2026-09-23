#!/usr/bin/env bash
# Build AND verify the GH200 judge+agent images (targets agent and judge) on one Daint node.
#
# NOT FOR BEVERIN: its partitions are x86_64 AMD. Submit from a GH200 cluster (Daint). No partition
# or account is written here -- sbatch reads both from the environment (or pass -A/-p):
#
#   export SBATCH_ACCOUNT=<project> SBATCH_PARTITION=normal
#   cd containers/cluster/ce-images && IMAGE_DIR=$PWD/judge-agent-cuda sbatch judge-agent-cuda/build.sbatch
#
# IMAGE_DIR, not --chdir: SLURM_SUBMIT_DIR is where sbatch was INVOKED, whatever --chdir says.
#
# COST: one node for up to 24 h. The spack compiler layer (gcc 16, then llvm 22) and MAGMA/PETSc
# against CUDA are the long part; a rebuild with a warm $SCRATCH/spack-buildcache-aarch64 skips the
# compilers. No measurement exists yet -- read the per-layer timestamps before planning around it.
#
# Each target goes to a CANDIDATE name, is verified inside itself under its production EDF template
# (build_common.sh ce_verify_candidate), and gets the .verified marker promote_image.sh requires only
# on a clean verdict. Then the half a build container cannot answer is probed on the GPUs.
#
#SBATCH --job-name=build-judge-agent-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --no-requeue

set -Eeuo pipefail

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_nid<node>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0

# Slurm SPOOLS the batch script, so BASH_SOURCE points into /var/spool/slurmd -- the directory is
# passed in (or is the submit directory), never derived.
IMAGE_DIR="${IMAGE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -x "${IMAGE_DIR}/build.sh"
# shellcheck source=../build_common.sh
source "${IMAGE_DIR}/../build_common.sh"
# shellcheck source=../images.env
source "${IMAGE_DIR}/../images.env"

export BUILD_TARGETS="${BUILD_TARGETS:-agent judge}"
# Candidates only: an OUTPUT_SQSH left in the environment would send a one-target build.sh
# somewhere this job then does not look.
unset OUTPUT_SQSH
CE="${CE_DIR:-${SCRATCH:?}/ce-images}"
declare -A CANDIDATE=(
    [agent]="${CE}/${JUDGE_AGENT_CUDA_SQSH%.sqsh}-candidate.sqsh"
    [judge]="${CE}/${JUDGE_CUDA_SQSH%.sqsh}-candidate.sqsh"
)
declare -A TEMPLATE=([agent]="${JUDGE_AGENT_CUDA_TEMPLATE}" [judge]="${JUDGE_CUDA_TEMPLATE}")
declare -A PROFILE=([agent]=judge-agent-cuda [judge]=judge-cuda)

# Refuse to write over an image an EDF mounts: a running arm would read a half-written squashfs.
edf_images() {
    grep -hoE '^[[:space:]]*image[[:space:]]*=[[:space:]]*"[^"]+"' "${HOME}/.edf"/*.toml 2>/dev/null \
      | sed -E 's/.*"(.*)"/\1/'
}
for target in ${BUILD_TARGETS}; do
    if edf_images | grep -qxF "${CANDIDATE[${target}]}"; then
        echo "refusing to overwrite ${CANDIDATE[${target}]}: an EDF in ${HOME}/.edf mounts it" >&2
        exit 2
    fi
done

"${IMAGE_DIR}/build.sh"

failed=0
for target in ${BUILD_TARGETS}; do
    sqsh="${CANDIDATE[${target}]}"
    [ -f "${sqsh}" ] || { echo "MISSING: ${sqsh} -- build.sh did not produce it" >&2; failed=$((failed + 1)); continue; }
    sha256sum "${sqsh}" | tee "${sqsh}.sha256"
    ce_verify_candidate "${TEMPLATE[${target}]}" "${sqsh}" "${PROFILE[${target}]}" || failed=$((failed + 1))
done

# The half the BUILD cannot answer: a build container has no driver, so PAPI's cuda/nvml components
# were only shown to be COMPILED IN, and cupy/jax only to import. Enumerating a component's events is
# what brings a lazily-initialized PAPI 7 component up. Reported, and counted.
agent_sqsh="${CANDIDATE[agent]}"
if [ -f "${agent_sqsh}" ]; then
    edf="${SCRATCH}/.tmp/gpu-probe-${SLURM_JOB_ID}.toml"
    sed -e "s|\${SCRATCH}|${SCRATCH}|g" -e "s|\"<hpcagent_bench_edf_mounts>\",||" \
        -e "s|^image = .*|image = \"${agent_sqsh}\"|" -e "s|^workdir = .*|workdir = \"/\"|" \
        "${IMAGE_DIR}/../${JUDGE_AGENT_CUDA_TEMPLATE}" > "${edf}"
    srun --ntasks=1 --environment="${edf}" bash -c '
        set -eu
        ulimit -c 0
        nvidia-smi -L
        python3 -c "import cupy; n = cupy.cuda.runtime.getDeviceCount(); print(\"cupy devices\", n); assert n > 0"
        python3 -c "import jax; d = jax.devices(); print(\"jax devices\", d); assert d[0].platform == \"gpu\""
        python3 -c "import torch; assert torch.cuda.is_available(); print(\"torch\", torch.cuda.get_device_name(0))"
        papi_native_avail 2>/dev/null | grep -c "cuda:::" || { echo "PAPI cuda component enumerated no events" >&2; exit 1; }
        ncu --version | head -1
        nsys --version | head -1
    ' || { failed=$((failed + 1)); rm -f "${CANDIDATE[agent]}.verified" "${CANDIDATE[judge]}.verified"; }
    rm -f "${edf}"
fi

[ "${failed}" -eq 0 ] || { echo "${failed} build/verify stage(s) FAILED; nothing to promote" >&2; exit 1; }
echo "GH200 JUDGE+AGENT CANDIDATES BUILT AND VERIFIED: ${CANDIDATE[*]}"
echo "Promote by rename once nothing mounts the live names:"
echo "  CE_PLATFORM=gh200 containers/cluster/ce-images/promote_image.sh judge-agent-cuda judge-cuda"
