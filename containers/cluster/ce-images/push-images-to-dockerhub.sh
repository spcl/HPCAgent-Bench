#!/usr/bin/env bash
# Publish the promoted judge-agent-amd + judge mi300 images to Docker Hub, from their saved OCI
# archives. Scratch-local, not committed: needs a Docker Hub PAT nobody on this cluster account
# has today (checked ~/.docker/config.json, podman auth.json, enroot .credentials -- none exist).
# Run this yourself once you have one; promotion (promote_image.sh + install_edfs.sh) does not
# wait for it and neither does anything else in the repo.
#
#   REGISTRY_USER=<dockerhub user> REGISTRY_TOKEN=<PAT with write to spcleth/hpcagent-bench> \
#     ./push-images-to-dockerhub.sh
#
# Publishes docker.io/spcleth/hpcagent-bench, tagged three ways per role (agent, judge):
#   <role>-<git-shortsha>   the exact commit that built the image, for citing a run
#   <role>-<YYYYMMDD>       the date this push ran
#   <role>-latest           the rolling tag images.env documents as TARGET_TAG
# push_image.sh also always adds its own sha-<image-digest> tag (immutable content hash) --
# that one is not duplicated here, it comes for free every push.
#
# ROLES="agent" ./push-images-to-dockerhub.sh   to push one role only.
set -Eeuo pipefail

SCRATCH="${SCRATCH:?set SCRATCH}"
REPO_CHECKOUT="${HPCAGENT_BENCH_REPO:-${SCRATCH}/hpcagent-bench}"
CE_DIR="${SCRATCH}/ce-images"
PUSH_REPO="docker.io/spcleth/hpcagent-bench"
ROLES="${ROLES:-agent judge}"

: "${REGISTRY_USER:?export REGISTRY_USER (Docker Hub username) before running this}"
: "${REGISTRY_TOKEN:?export REGISTRY_TOKEN (Docker Hub PAT, write access to spcleth/hpcagent-bench) before running this}"

GIT_SHA="$(git -C "${REPO_CHECKOUT}" rev-parse --short=12 HEAD)"
DATE_TAG="$(date +%Y%m%d)"

archive_of() {  # role -> the LIVE (post-promotion) OCI archive promote_image.sh moved it to
    case "$1" in
        agent) printf '%s/hpcagent-bench-agent-mi300.oci.tar' "${CE_DIR}" ;;
        judge) printf '%s/hpcagent-bench-judge-mi300.oci.tar' "${CE_DIR}" ;;
        *) return 2 ;;
    esac
}

echo "repo:    ${PUSH_REPO}"
echo "roles:   ${ROLES}"
echo "commit:  ${GIT_SHA}  ($(git -C "${REPO_CHECKOUT}" log -1 --format=%s HEAD 2>/dev/null | cut -c1-72))"
echo "tags:    <role>-${GIT_SHA}  <role>-${DATE_TAG}  <role>-latest  (+ push_image.sh's own sha-<digest>)"
echo

for role in ${ROLES}; do
    archive="$(archive_of "${role}")" || { echo "unknown role ${role}" >&2; exit 2; }
    [ -f "${archive}" ] || { echo "${role}: NO ARCHIVE at ${archive} -- promote before pushing" >&2; exit 2; }
done

# The upload itself needs a compute node: podman's rootless unpack graphroot must be tmpfs
# (/dev/shm), and the judge+agent archive needs 60+ GB of it free -- see push_image.sh's own
# header for why. One sbatch job, one role per iteration inside it, so this script can be
# launched from a login node and does the whole thing in one submission.
mkdir -p "${CE_DIR}/logs" "${SCRATCH}/.tmp"
JOB_SCRIPT="$(mktemp "${SCRATCH}/.tmp/push-dockerhub-XXXXXX.sbatch")"

{
    printf '#!/usr/bin/env bash\n'
    printf '#SBATCH --job-name=push-dockerhub\n'
    printf '#SBATCH -A a-g34\n'
    printf '#SBATCH --partition=mi300\n'
    printf '#SBATCH --nodes=1\n'
    printf '#SBATCH --ntasks=1\n'
    printf '#SBATCH --cpus-per-task=96\n'
    printf '#SBATCH --hint=nomultithread\n'
    printf '#SBATCH --mem=0\n'
    printf '#SBATCH --no-requeue\n'
    printf '#SBATCH --time=08:00:00\n'
    printf '#SBATCH --output=%s/logs/%%x-%%j.out\n' "${CE_DIR}"
    printf '#SBATCH --error=%s/logs/%%x-%%j.err\n' "${CE_DIR}"
    printf 'set -Eeuo pipefail\n'
    printf 'ulimit -c 0\n'
    printf 'cd %q\n' "${REPO_CHECKOUT}/containers/cluster/ce-images"
    printf 'REGISTRY_USER=%q\n' "${REGISTRY_USER}"
    printf 'REGISTRY_TOKEN=%q\n' "${REGISTRY_TOKEN}"
    printf 'export REGISTRY_USER REGISTRY_TOKEN\n'
    printf 'rc=0\n'
    for role in ${ROLES}; do
        archive="$(archive_of "${role}")"
        printf 'echo "===== pushing %s  (%s)"\n' "${role}" "${archive}"
        printf 'if PUSH_REPO=%q ./push_image.sh --from-archive %q %q %q %q; then\n' \
            "${PUSH_REPO}" "${archive}" "${role}-${GIT_SHA}" "${role}-${DATE_TAG}" "${role}-latest"
        printf '    echo "  PUSHED %s"\n' "${role}"
        printf 'else\n'
        printf '    echo "  FAILED %s" >&2\n' "${role}"
        printf '    rc=1\n'
        printf 'fi\n'
    done
    printf 'exit "${rc}"\n'
} > "${JOB_SCRIPT}"
chmod +x "${JOB_SCRIPT}"

echo "submitting: sbatch ${JOB_SCRIPT}"
sbatch "${JOB_SCRIPT}"
