#!/usr/bin/env bash
# Build an image AND prove it works, in one job -- the end-to-end entry point.
#
#   IMAGE_DIR=containers/cluster/ce-images/judge-agent-amd \
#     sbatch containers/cluster/ce-images/build_and_verify.sbatch
#
# WHY ONE JOB. build.sbatch produced an artifact and told the operator, in prose, to go and verify
# it separately. Nothing enforced that, so an image could be built, named and promoted without the
# verifier ever running -- which is how a v6 with no agent tools reached a campaign and emptied it.
# Chaining the two here means a build that cannot pass verification does not report success, and
# the artifact keeps its CANDIDATE name until a human promotes it.
#
# The build itself is unchanged: same build.sbatch body, same Dockerfile, same spack binary
# buildcache on scratch, so a rebuild reuses gcc 16 and llvm 22 rather than recompiling them. This
# script adds no patching step of its own -- an image that needs one is a Dockerfile that needs
# fixing.
#
#SBATCH --job-name=build-verify-image
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 96, not 192, and --hint=nomultithread with it. The node is Sockets=4 CoresPerSocket=24
# ThreadsPerCore=2 -- 96 PHYSICAL cores behind 192 logical CPUs. Asking for 192 hands out SMT
# siblings, which on this system are not extra compute: two threads share one core's execution
# units, so a compile or a rank pinned across a sibling pair runs at core speed while the count
# says it has two. Every campaign shape here is physical-core based (4 ranks x 24), and the judge
# already spells it --cpus-per-task=24 --hint=nomultithread.
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/build-verify-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/build-verify-%j.err

set -Eeuo pipefail
ulimit -c 0

# /x86_64: the real SCRATCH has it and this file's own log paths assume it, so a fallback
# without it silently points at a tree that does not exist.
SCRATCH="${SCRATCH:-/capstor/scratch/cscs/${USER}/x86_64}"
REPO="${REPO:-${SCRATCH}/optarena}"
IMAGE_DIR="${IMAGE_DIR:?set IMAGE_DIR, e.g. containers/cluster/ce-images/judge-agent-amd}"
case "${IMAGE_DIR}" in /*) ;; *) IMAGE_DIR="${REPO}/${IMAGE_DIR}";; esac
PROFILE="${PROFILE:-$(basename "${IMAGE_DIR}")}"

echo "=== stage 1/2: build  (${IMAGE_DIR})"
build_out="${SCRATCH}/ce-images/logs/build-inner-${SLURM_JOB_ID}.log"
mkdir -p "$(dirname "${build_out}")"
# The build sbatch is a plain bash script below its #SBATCH block, so it runs directly inside the
# allocation this job already holds -- no nested sbatch, no second queue wait.
IMAGE_DIR="${IMAGE_DIR}" bash "${IMAGE_DIR}/build.sbatch" 2>&1 | tee "${build_out}"

# EVERY target this build produced, not just the first. judge-agent-amd builds `agent judge` by
# default, and verifying only the agent meant the JUDGE -- the image that grades every submission
# -- shipped unverified. Each is verified under its own profile so the judge's extra layer (the
# installed hpcagent_bench and the firewall assertion) is actually exercised.
# Keyed by IMAGE_DIR, because the map used to name only the judge-agent pair: for an sglang or
# vllm build every entry was missing, the loop below skipped both, verify_rc stayed 0 and the job
# reported SUCCESS having verified nothing. An unknown profile is now an error, not a skip.
declare -A BUILD_TARGETS_OF=(
    [judge-agent-amd]="judge-agent-amd judge"
    [sglang]="sglang"
    [vllm]="vllm"
)
declare -A TARGET_SQSH=(
    [judge-agent-amd]="${SCRATCH}/ce-images/optarena-ce-amd-mi300-candidate.sqsh"
    [judge]="${SCRATCH}/ce-images/optarena-ce-judge-amd-mi300-candidate.sqsh"
    [sglang]="${SCRATCH}/ce-images/optarena-sglang-candidate.sqsh"
    [vllm]="${SCRATCH}/ce-images/optarena-vllm-candidate.sqsh"
)
targets="${BUILD_TARGETS_OF[${PROFILE}]:-}"
[ -n "${targets}" ] || { echo "no verify targets known for profile ${PROFILE}" >&2; exit 1; }

# The image this job's own build wrote, which is the first target unless OUTPUT_SQSH overrode it.
sqsh="${OUTPUT_SQSH:-${TARGET_SQSH[${targets%% *}]}}"
[ -f "${sqsh}" ] || { echo "build reported success but ${sqsh} is not there" >&2; exit 1; }

echo "=== stage 2/2: verify (${sqsh})"
# verify_image.sbatch generates its own EDF for a candidate; run its body here for the same reason
# the build runs here -- one allocation, one verdict.
verify_rc=0
for prof in ${targets}; do
    img="${TARGET_SQSH[${prof}]}"
    if [ ! -f "${img}" ]; then
        echo "${prof}: ${img} was not produced by this build" >&2
        verify_rc=1
        continue
    fi
    echo "=== verifying target ${prof} -> ${img}"
    # The marker is what promote_image.sh requires before it will move a candidate over a live
    # name. Removed FIRST, so a re-verification that fails cannot leave a stale pass behind and
    # let a broken image be promoted on the strength of an earlier run.
    rm -f "${img}.verified"
    if IMAGE="${img}" PROFILE="${prof}" \
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
