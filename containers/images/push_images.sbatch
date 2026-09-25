#!/usr/bin/env bash
# Publish images from their OCI archives to REGISTRY_REPO, each under its images.env tag.
# DRY_RUN=1 (the default) runs every gate push_image.sh has and sends nothing.
#
#   DRY_RUN=1 sbatch containers/images/push_images.sbatch            # default roles
#   REGISTRY_USER=<user> REGISTRY_TOKEN=<token> DRY_RUN=0 ROLES="sglang vllm" sbatch ...
#   ROLES=judge-release JUDGE_AMD_RELEASE_TAG=judge-mi300-<sha12> DRY_RUN=0 sbatch ...
#
#SBATCH --job-name=push-images
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --no-requeue

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:?set SCRATCH}"
SCRIPT_DIR="${SCRIPT_DIR:-${HPCAGENT_BENCH_REPO:-${SCRATCH}/hpcagent-bench}/containers/images}"
# shellcheck source=images.env
source "${SCRIPT_DIR}/images.env"
CE="${CE_IMAGES:-${SCRATCH}/ce-images}"
mkdir -p "${SCRATCH}/ce-images/logs"

DRY_RUN="${DRY_RUN:-1}"
if [ -z "${ROLES:-}" ]; then
    for role in $(ce_roles "${CE_PLATFORM:-amd}"); do
        ce_image "${role}" tag >/dev/null 2>&1 || continue
        [[ ",$(ce_image "${role}" flags 2>/dev/null)," == *,opt,* ]] && continue
        ROLES="${ROLES:+${ROLES} }${role}"
    done
fi

echo "repository: ${REGISTRY_REPO}"
echo "roles:      ${ROLES}"
echo "dry run:    ${DRY_RUN}"
echo

failed=""
for role in ${ROLES}; do
    sqsh="$(ce_image "${role}" sqsh)" || { failed="${failed} ${role}"; continue; }
    tag="$(ce_image "${role}" tag 2>/dev/null)" || {
        echo "=== ${role}: no tag; set <PREFIX>_TAG (build-judge-release.sh prints the release one)" >&2
        failed="${failed} ${role}"
        continue
    }
    archive="${CE}/${sqsh%.sqsh}.oci.tar"
    echo "=== ${role}  ->  ${REGISTRY_REPO}:${tag}"
    echo "    archive ${archive}"
    if [ ! -f "${archive}" ]; then
        echo "    NO ARCHIVE -- build the role first; a squashfs cannot stand in for one" >&2
        failed="${failed} ${role}"
        continue
    fi
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
echo "done"
