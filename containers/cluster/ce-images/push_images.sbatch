#!/usr/bin/env bash
# Publish every role from its OCI archive, under ONE repository, with per-role tags.
#
#   sbatch containers/cluster/ce-images/push_images.sbatch          # all four roles
#   ROLES="judge" sbatch containers/cluster/ce-images/push_images.sbatch
#   DRY_RUN=1 sbatch ...                                            # every gate, no bytes out
#
# NOTHING IS PUBLISHED WITHOUT DRY_RUN=0 AND AN EXPLICIT ROLES/ALL. Publishing is one-way: a tag
# that reaches a public registry has been fetched and cached before anyone notices it was wrong.
#
# WHY THIS EXISTS. push_image.sh takes a local tag and a PUSH_REPO and knows nothing about roles,
# so publishing was four hand-typed invocations with a per-image repository -- which contradicts
# images.env, where every role shares REGISTRY_REPO and the TAG names the role. Four hand-typed
# commands is also how one gets forgotten: the judge role was added and nothing enumerated it.
#
# The archive, not the squashfs: a squashfs is a flattened filesystem, so reimporting one collapses
# the image into a single layer past the registry's per-layer ceiling and drops the image config.
#
#SBATCH --job-name=push-images
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/push-images-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/push-images-%j.err

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/${USER}/x86_64}"
SCRIPT_DIR="${SCRIPT_DIR:-${SCRATCH}/optarena/containers/cluster/ce-images}"
# shellcheck source=images.env
source "${SCRIPT_DIR}/images.env"
CE="${CE_IMAGES:-${SCRATCH}/ce-images}"
mkdir -p "${SCRATCH}/ce-images/logs"

# DRY_RUN defaults to 1. Turning it off is a deliberate act, not the default path.
DRY_RUN="${DRY_RUN:-1}"
ROLES="${ROLES:-judge-agent-amd judge sglang vllm}"

role_sqsh() {
    case "$1" in
        judge-agent-amd) printf '%s' "${JUDGE_AGENT_AMD_SQSH}" ;;
        judge)           printf '%s' "${JUDGE_AMD_SQSH}" ;;
        sglang)          printf '%s' "${INFERENCE_SGLANG_SQSH}" ;;
        vllm)            printf '%s' "${INFERENCE_VLLM_SQSH}" ;;
        *) return 2 ;;
    esac
}

echo "repository: ${REGISTRY_REPO}"
echo "roles:      ${ROLES}"
echo "dry run:    ${DRY_RUN}"
echo

failed=""
for role in ${ROLES}; do
    sqsh="$(role_sqsh "${role}")" || { echo "unknown role ${role}" >&2; failed="${failed} ${role}"; continue; }
    tag="$(sqsh_tag "${sqsh}")"
    archive="${CE}/${sqsh%.sqsh}.oci.tar"
    echo "=== ${role}  ->  ${REGISTRY_REPO}:${tag}"
    echo "    archive ${archive}"
    if [ ! -f "${archive}" ]; then
        echo "    NO ARCHIVE -- build the role first; a squashfs cannot stand in for one" >&2
        failed="${failed} ${role}"
        continue
    fi
    # --check-only runs every gate that can reject an upload -- layer sizes against the registry
    # ceiling, config sanity, archive load -- and sends no bytes and does no login.
    if [ "${DRY_RUN}" != "0" ]; then
        "${SCRIPT_DIR}/push_image.sh" --check-only --from-archive "${archive}" \
            || { echo "    PREFLIGHT FAILED" >&2; failed="${failed} ${role}"; }
        continue
    fi
    PUSH_REPO="${REGISTRY_REPO}" "${SCRIPT_DIR}/push_image.sh" --from-archive "${archive}" "${tag}" \
        || { echo "    PUSH FAILED" >&2; failed="${failed} ${role}"; }
done

echo
if [ -n "${failed}" ]; then
    echo "FAILED:${failed}" >&2
    exit 1
fi
[ "${DRY_RUN}" != "0" ] && echo "preflight PASSED for: ${ROLES} (nothing was published; set DRY_RUN=0 to publish)"
echo done
