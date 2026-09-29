#!/usr/bin/env bash
# Build an image AND prove it works, in one job -- the end-to-end entry point.
#
#   sbatch -p mi300 containers/images/build_and_verify.sbatch judge-agent-amd
#   sbatch -p <cpu partition> containers/images/build_and_verify.sbatch judge-agent-cpu
#   VERIFY_ONLY=1 sbatch ... <image>        re-verify the candidates an earlier build wrote
#
# The partition is SBATCH_PARTITION (site layer) unless the command line names one.
# An AMD image is portable (every gpu_arch.env target, cpu_target.env's baseline CPU), so it builds
# once, on the partition its images.env row names, and verify_image.sbatch then checks it on the
# other partitions.
#
# One job builds and verifies, so a build that cannot pass verification never reports success.
# The build pulls first and keeps its caches by default (CE_PULL, CE_BUILD_CACHE in build_common.sh);
# a pulled image is verified exactly like a built one.
# Every role the build wrote is verified (verify_image.sbatch) and marked .verified for
# `registry.sh promote`.
#
#SBATCH --job-name=build-verify-image
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 96 physical cores; SMT siblings add no compute.
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --no-requeue

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:?set SCRATCH}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:?submit from the checkout root or set REPO}}"
IMAGE="${1:?usage: build_and_verify.sbatch <image directory under containers/images>}"
# shellcheck source=build_common.sh
source "${REPO}/containers/images/build_common.sh"

if [[ "${VERIFY_ONLY:-0}" == 1 ]]; then
    echo "=== stage 1/2: build SKIPPED (VERIFY_ONLY=1)"
else
    echo "=== stage 1/2: build ${IMAGE}"
    mkdir -p "${CE_IMAGES:?}/logs"
    bash "${REPO}/containers/images/build.sh" "${IMAGE}" 2>&1 | tee "${CE_IMAGES}/logs/build-${IMAGE}-${SLURM_JOB_ID}.log"
fi

# The roles the build wrote, as build.sh reads them from <image>/image.sh.
roles="$(bash "${REPO}/containers/images/build.sh" "${IMAGE}" --roles)"
[[ -n "${roles}" ]] || { echo "${IMAGE}/image.sh names no roles" >&2; exit 1; }

echo "=== stage 2/2: verify"
failed=0
for role in ${roles}; do
    sqsh="${CE_IMAGES}/$(ce_image "${role}" candidate)"
    # Removed first, so a failed re-verification cannot leave a stale pass behind.
    rm -f "${sqsh}.verified"
    if ROLE="${role}" IMAGE="${sqsh}" REPO="${REPO}" bash "${REPO}/containers/images/verify_image.sbatch"; then
        printf 'verified role=%s job=%s digest=%s\n' "${role}" "${SLURM_JOB_ID}" \
            "$(cat "${sqsh}.digest" 2>/dev/null || echo unknown)" > "${sqsh}.verified"
        echo "VERIFIED: ${role} ${sqsh}"
    else
        echo "NOT VERIFIED: ${role} ${sqsh}" >&2
        failed=$((failed + 1))
    fi
done
[[ "${failed}" -eq 0 ]] || { echo "${failed} role(s) FAILED verification; nothing to promote" >&2; exit 1; }
echo "BUILT AND VERIFIED: ${roles//$'\n'/ }. They keep their candidate names; promote with"
echo "  containers/images/registry.sh promote ${roles//$'\n'/ }"
