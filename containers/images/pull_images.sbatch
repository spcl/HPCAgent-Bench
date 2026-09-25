#!/usr/bin/env bash
# Fetch every published image of CE_PLATFORM (images.env rows with a tag, optional ones excepted)
# into the squashfs files the EDFs mount. Pulling gets the published bytes, so a cited digest is
# the digest that ran; build only when changing an image (containers/README.md).
#
#   sbatch containers/images/pull_images.sbatch
#   ROLES="judge-agent-amd judge" sbatch containers/images/pull_images.sbatch
#   TAG=sha-<digest> ROLES=judge sbatch containers/images/pull_images.sbatch
#
#SBATCH --job-name=pull-images
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --no-requeue

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:?set SCRATCH}"
# sbatch exports the submitting shell's environment, which may carry a stale SCRATCH.
if [ ! -d "${SCRATCH}" ] || [ ! -w "${SCRATCH}" ]; then
    echo "SCRATCH=${SCRATCH} is not an existing writable directory." >&2
    echo "  Re-login for a fresh shell, or export the current SCRATCH yourself before submitting." >&2
    exit 2
fi
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-${SCRATCH}/.enroot}"
mkdir -p "${ENROOT_CACHE_PATH}"
# sbatch runs a spooled copy of this file, so the checkout is found from the environment.
if [ -z "${SCRIPT_DIR:-}" ]; then
    for _cand in "${HPCAGENT_BENCH_REPO:-}/containers/images" \
                 "${SLURM_SUBMIT_DIR:-}/containers/images" \
                 "${SLURM_SUBMIT_DIR:-}"; do
        if [ -n "${_cand#/containers*}" ] && [ -x "${_cand}/pull_image.sh" ]; then
            SCRIPT_DIR="${_cand}"; break
        fi
    done
    unset _cand
fi
if [ ! -x "${SCRIPT_DIR:-}/pull_image.sh" ]; then
    echo "cannot locate pull_image.sh: submit from the repo root, or set" >&2
    echo "  HPCAGENT_BENCH_REPO=/path/to/hpcagent-bench or SCRIPT_DIR=/path/to/containers/images" >&2
    exit 2
fi
echo "SCRIPT_DIR=${SCRIPT_DIR}"
# shellcheck source=images.env
source "${SCRIPT_DIR}/images.env"
mkdir -p "${SCRATCH}/ce-images/logs"

if [ -z "${ROLES:-}" ]; then
    for role in $(ce_roles "${CE_PLATFORM:-amd}"); do
        ce_image "${role}" tag >/dev/null 2>&1 || continue
        [[ ",$(ce_image "${role}" flags 2>/dev/null)," == *,opt,* ]] && continue
        ROLES="${ROLES:+${ROLES} }${role}"
    done
fi

# Every role is attempted and reported; the job fails at the end if any did.
failed=""
for role in ${ROLES}; do
    echo "=== pulling ${role}"
    if "${SCRIPT_DIR}/pull_image.sh" "${role}" ${TAG:+"${TAG}"}; then
        echo "=== ${role} OK"
    else
        echo "=== ${role} FAILED" >&2
        failed="${failed} ${role}"
    fi
done

echo
if [ -n "${failed}" ]; then
    echo "FAILED:${failed}" >&2
    echo "A role that has never been published has nothing to pull -- build it (containers/README.md)." >&2
    exit 1
fi
echo "all roles fetched. Install the EDFs that name them:"
echo "  ${SCRIPT_DIR}/install_edfs.sh"
