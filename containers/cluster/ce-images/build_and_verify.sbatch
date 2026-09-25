#!/usr/bin/env bash
# Build an image AND prove it works, in one job -- the end-to-end entry point.
#
#   IMAGE_DIR=containers/cluster/ce-images/judge-agent-amd \
#     sbatch containers/cluster/ce-images/build_and_verify.sbatch
#
# judge-agent-amd builds for the partition it runs on; the mi200 pair (gfx90a, spack target zen3):
#   REPO=$PWD IMAGE_DIR=containers/cluster/ce-images/judge-agent-amd \
#     sbatch --partition=mi200 --cpus-per-task=64 --gpus-per-node=8 containers/cluster/ce-images/build_and_verify.sbatch
#
# sglang-mi200 builds and verifies on mi200 (64 physical cores); override the directives below:
#   REPO=$PWD IMAGE_DIR=containers/cluster/ce-images/sglang-mi200 \
#     sbatch --partition=mi200 --cpus-per-task=64 --gpus-per-node=8 containers/cluster/ce-images/build_and_verify.sbatch
#
# One job builds and verifies, so a build that cannot pass verification never reports success.
# Every target of IMAGE_DIR built on this partition (images.env rows with that dir and partition)
# is verified under its own profile and marked .verified for promote_image.sh.
#
#SBATCH --job-name=build-verify-image
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 96 physical cores; SMT siblings add no compute.
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --no-requeue

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:?set SCRATCH}"
REPO="${REPO:-${SCRATCH}/hpcagent-bench}"
IMAGE_DIR="${IMAGE_DIR:?set IMAGE_DIR, e.g. containers/cluster/ce-images/judge-agent-amd}"
case "${IMAGE_DIR}" in /*) ;; *) IMAGE_DIR="${REPO}/${IMAGE_DIR}";; esac
PROFILE="${PROFILE:-$(basename "${IMAGE_DIR}")}"
# shellcheck source=build_common.sh
source "${REPO}/containers/cluster/ce-images/build_common.sh"
PARTITION="${SLURM_JOB_PARTITION:?build_and_verify.sbatch runs inside a Slurm allocation}"

# VERIFY_ONLY=1 re-verifies the candidates an earlier build wrote (after a verifier fix).
if [[ "${VERIFY_ONLY:-0}" == 1 ]]; then
    echo "=== stage 1/2: build SKIPPED (VERIFY_ONLY=1)"
else
echo "=== stage 1/2: build  (${IMAGE_DIR})"
build_out="${SCRATCH}/ce-images/logs/build-inner-${SLURM_JOB_ID}.log"
mkdir -p "$(dirname "${build_out}")"
# build.sbatch is plain bash below its #SBATCH block: run it inside this allocation.
IMAGE_DIR="${IMAGE_DIR}" bash "${IMAGE_DIR}/build.sbatch" 2>&1 | tee "${build_out}"
fi

# shellcheck source=images.env
source "${REPO}/containers/cluster/ce-images/images.env"
declare -A TARGET_SQSH=()
targets=""
for role in $(ce_roles); do
    [[ "$(ce_image "${role}" dir 2>/dev/null)" == "${PROFILE}" ]] || continue
    [[ "$(ce_image "${role}" partition 2>/dev/null)" == "${PARTITION}" ]] || continue
    candidate="$(ce_image "${role}" candidate 2>/dev/null)" || continue
    prof="$(ce_image "${role}" profile)"
    TARGET_SQSH[${prof}]="${SCRATCH}/ce-images/${candidate}"
    targets="${targets:+${targets} }${prof}"
done
# An unknown profile is an error: a skip would report success having verified nothing.
[ -n "${targets}" ] || { echo "images.env has no ${PROFILE} target for partition ${PARTITION}" >&2; exit 1; }

# The image this job's own build wrote: the first target unless OUTPUT_SQSH overrode it.
sqsh="${OUTPUT_SQSH:-${TARGET_SQSH[${targets%% *}]}}"
[ -f "${sqsh}" ] || { echo "build reported success but ${sqsh} is not there" >&2; exit 1; }

echo "=== stage 2/2: verify (${sqsh})"
verify_rc=0
for prof in ${targets}; do
    img="${TARGET_SQSH[${prof}]}"
    if [ ! -f "${img}" ]; then
        echo "${prof}: ${img} was not produced by this build" >&2
        verify_rc=1
        continue
    fi
    echo "=== verifying target ${prof} -> ${img}"
    # Removed first, so a failed re-verification cannot leave a stale pass behind.
    rm -f "${img}.verified"
    if IMAGE="${img}" PROFILE="${prof}" REPO="${REPO}" \
        bash "${REPO}/containers/cluster/ce-images/verify_image.sbatch"; then
        printf 'verified profile=%s job=%s digest=%s\n' "${prof}" "${SLURM_JOB_ID}" \
            "$(cat "${img}.digest" 2>/dev/null || echo unknown)" > "${img}.verified"
    else
        verify_rc=$?
    fi
done
[ "${verify_rc}" -eq 0 ] || { echo "a target FAILED verification" >&2; exit "${verify_rc}"; }

echo "=== toolchain provenance recorded in the image"
echo "IMAGE BUILT AND VERIFIED: ${sqsh}"
echo "It keeps the CANDIDATE name. Promote by rename once nothing mounts the target."
