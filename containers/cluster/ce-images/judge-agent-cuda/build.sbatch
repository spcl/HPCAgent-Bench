#!/usr/bin/env bash
# Build and verify the GH200 judge+agent candidates on one Daint node. Partition and account come
# from the environment:
#
#   export SBATCH_ACCOUNT=<project> SBATCH_PARTITION=normal
#   cd containers/cluster/ce-images && IMAGE_DIR=$PWD/judge-agent-cuda sbatch judge-agent-cuda/build.sbatch
#
# Up to 24 h on a cold $SCRATCH/spack-buildcache-aarch64 (gcc 16, llvm 22, MAGMA, PETSc). Each
# target is verified inside itself (ce_verify_candidate), then the agent image is probed on the GPUs.
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

ulimit -c 0

# Slurm runs a spooled copy, so the image directory is passed in (or is the submit directory).
IMAGE_DIR="${IMAGE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -x "${IMAGE_DIR}/build.sh"
# shellcheck source=../build_common.sh
source "${IMAGE_DIR}/../build_common.sh"
# shellcheck source=../images.env
source "${IMAGE_DIR}/../images.env"

export BUILD_TARGETS="${BUILD_TARGETS:-agent judge}"
unset OUTPUT_SQSH
CE="${CE_DIR:-${SCRATCH:?}/ce-images}"
declare -A CANDIDATE=(
    [agent]="${CE}/${JUDGE_AGENT_CUDA_CANDIDATE}"
    [judge]="${CE}/${JUDGE_CUDA_CANDIDATE}"
)
declare -A TEMPLATE=([agent]="${JUDGE_AGENT_CUDA_TEMPLATE}" [judge]="${JUDGE_CUDA_TEMPLATE}")
declare -A PROFILE=([agent]=judge-agent-cuda [judge]=judge-cuda)

for target in ${BUILD_TARGETS}; do
    ce_refuse_mounted "${CANDIDATE[${target}]}" || exit 2
done

"${IMAGE_DIR}/build.sh"

failed=0
for target in ${BUILD_TARGETS}; do
    sqsh="${CANDIDATE[${target}]}"
    [ -f "${sqsh}" ] || { echo "MISSING: ${sqsh} -- build.sh did not produce it" >&2; failed=$((failed + 1)); continue; }
    sha256sum "${sqsh}" | tee "${sqsh}.sha256"
    ce_verify_candidate "${TEMPLATE[${target}]}" "${sqsh}" "${PROFILE[${target}]}" || failed=$((failed + 1))
done

# What a driverless build cannot show: devices visible to cupy/jax/torch and PAPI's cuda component
# enumerating events.
agent_sqsh="${CANDIDATE[agent]}"
if [ -f "${agent_sqsh}" ]; then
    edf="${SCRATCH}/.tmp/gpu-probe-${SLURM_JOB_ID}.toml"
    mkdir -p "$(dirname "${edf}")"
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
