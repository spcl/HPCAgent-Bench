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
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/build-verify-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/build-verify-%j.err

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/${USER}}"
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

sqsh="${OUTPUT_SQSH:-${SCRATCH}/ce-images/optarena-ce-amd-mi300-candidate.sqsh}"
[ -f "${sqsh}" ] || { echo "build reported success but ${sqsh} is not there" >&2; exit 1; }

echo "=== stage 2/2: verify (${sqsh})"
# verify_image.sbatch generates its own EDF for a candidate; run its body here for the same reason
# the build runs here -- one allocation, one verdict.
IMAGE="${sqsh}" PROFILE="${PROFILE}" bash "${REPO}/containers/cluster/ce-images/verify_image.sbatch"

echo "=== toolchain provenance recorded in the image"
echo "IMAGE BUILT AND VERIFIED: ${sqsh}"
echo "It keeps the CANDIDATE name. Promote by rename once nothing mounts the target."
