#!/usr/bin/env bash
# Build and verify the CPU-only judge+agent candidates for the architecture of the node it runs on
# (about an hour). Partition and account come from the environment:
#
#   export SBATCH_ACCOUNT=<project> SBATCH_PARTITION=normal     # Daint: aarch64
#   cd containers/cluster/ce-images && IMAGE_DIR=$PWD/judge-agent-cpu sbatch judge-agent-cpu/build.sbatch
#
#SBATCH --job-name=build-judge-agent-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
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
    [agent]="${CE}/${JUDGE_AGENT_CPU_CANDIDATE}"
    [judge]="${CE}/${JUDGE_CPU_CANDIDATE}"
)
declare -A TEMPLATE=([agent]="${JUDGE_AGENT_CPU_TEMPLATE}" [judge]="${JUDGE_CPU_TEMPLATE}")
declare -A PROFILE=([agent]=judge-agent-cpu [judge]=judge-cpu)

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
[ "${failed}" -eq 0 ] || { echo "${failed} build/verify stage(s) FAILED; nothing to promote" >&2; exit 1; }
echo "CPU JUDGE+AGENT CANDIDATES ($(uname -m)) BUILT AND VERIFIED: ${CANDIDATE[*]}"
echo "Promote by rename once nothing mounts the live names:"
echo "  CE_PLATFORM=cpu containers/cluster/ce-images/promote_image.sh judge-agent-cpu judge-cpu"
