#!/usr/bin/env bash
# Build hpcagent-bench-judge-agent-amd on a compute node.
#
# COST, before you submit it: one mi300 node for 2-4 hours with the spack buildcache warm (build
# and verify together, 2.2-3.5 h measured), hence the 12 h limit. The compiler layer (gcc 16
# bootstrap, then llvm 22 with clang/flang/MLIR/Polly) is the multi-hour part of a cold build,
# PETSc and MAGMA the next largest: with no buildcache, submit with --time=24:00:00.
#
# On a COMPUTE node rather than the login node: the build pulls a ~30 GB ROCm base and podman
# storage on login is not worth the space.
#
#   IMAGE_DIR=containers/images/judge-agent-amd sbatch "${IMAGE_DIR}/build.sbatch"
#
# One portable image for every AMD partition: device code for gpu_arch.env's AMD_GPU_TARGETS and
# cpu_target.env's baseline CPU target, whichever partition (SBATCH_PARTITION) builds it.
#
#SBATCH --job-name=build-judge-agent-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 96 PHYSICAL cores, not 192 logical: the node is Sockets=4 CoresPerSocket=24
# ThreadsPerCore=2, and an SMT sibling is not extra compute for a compile.
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --no-requeue

set -Eeuxo pipefail

# The cluster's core_pattern dumps into the crashing process's CWD; Slurm propagates the
# submitter's core limit, so the floor has to be set here to avoid littering the checkout.
ulimit -c 0

# Slurm SPOOLS the batch script, so BASH_SOURCE points into /var/spool/slurmd -- the directory is
# passed in rather than derived. SLURM_SUBMIT_DIR is the submit directory, so
# `cd <this dir> && sbatch build.sbatch` needs no variable.
IMAGE_DIR="${IMAGE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -x "${IMAGE_DIR}/build.sh"

# shellcheck source=../build_common.sh
source "${IMAGE_DIR}/../build_common.sh"
mkdir -p "${CE_IMAGES:?set SCRATCH or CE_IMAGES}/logs"
export OUTPUT_SQSH="${OUTPUT_SQSH:-${CE_IMAGES}/$(ce_amd_candidate agent)}"
export BUILD_TARGETS="${BUILD_TARGETS:-agent judge}"

# WHICH FILES build.sh WILL ACTUALLY WRITE.
#
# build.sh honours OUTPUT_SQSH only for a SINGLE-target build -- with two targets one name cannot
# mean both -- and otherwise writes each target's own candidate name. This mirrors that rule.
target_sqsh() {
    local name
    name="$(ce_amd_candidate "$1")" || return 2
    printf '%s/%s' "${CE_IMAGES}" "${name}"
}
built_paths() {
    local t
    if [ "$(printf '%s\n' ${BUILD_TARGETS} | wc -w)" -eq 1 ]; then
        printf '%s\n' "${OUTPUT_SQSH}"
        return
    fi
    for t in ${BUILD_TARGETS}; do target_sqsh "${t}"; printf '\n'; done
}

# Check EVERY path this build will write, not just OUTPUT_SQSH: on a two-target build the agent
# and judge candidates are what land on disk, and neither of them is OUTPUT_SQSH.
mapfile -t built < <(built_paths)
ce_refuse_mounted "${built[@]}" || exit 2

"${IMAGE_DIR}/build.sh"

# Report every image the build wrote. A missing one is a failure, checked at the path build.sh
# USED, so a successful build never fails on a name nobody wrote.
missing=0
while read -r sqsh; do
    [ -n "${sqsh}" ] || continue
    if [ ! -f "${sqsh}" ]; then
        echo "MISSING: ${sqsh} -- build.sh did not produce it" >&2
        missing=$((missing + 1))
        continue
    fi
    ls -lh "${sqsh}"
    sha256sum "${sqsh}" | tee "${sqsh}.sha256"
    [ -f "${sqsh}.digest" ] && cat "${sqsh}.digest"
    echo "IMAGE READY: ${sqsh}"
done <<EOF
$(built_paths)
EOF
[ "${missing}" -eq 0 ] || { echo "${missing} image(s) missing" >&2; exit 2; }
echo "Verify it before it goes live: scripts/smoke_gpu_profilers.sh via"
echo "scripts/submit_gpu_profiler_smoke.sbatch -- it reads ARTIFACTS, not exit codes, and fails"
echo "rocprof-compute specifically when its passes run and its rows are dropped."
