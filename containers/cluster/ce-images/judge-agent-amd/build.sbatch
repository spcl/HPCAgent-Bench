#!/usr/bin/env bash
# Build optarena-judge-agent-amd on a compute node.
#
# COST, before you submit it: one mi300 node for up to 24 hours, which is the partition maximum.
# The compiler layer alone (gcc 16 bootstrap, then llvm 22 with clang/flang/MLIR/Polly) is the
# multi-hour part; PETSc with its solver chain and MAGMA are the next largest. A cold build has
# no reference measurement yet -- treat 24h as the budget, not as an estimate, and read the
# per-layer timestamps out of the log before assuming a second build is cheaper.
#
# On a COMPUTE node rather than the login node: the build pulls a ~30 GB ROCm base and podman
# storage on login is not worth the space.
#
#   IMAGE_DIR=containers/cluster/ce-images/judge-agent-amd sbatch "${IMAGE_DIR}/build.sbatch"
#
#SBATCH --job-name=build-judge-agent-amd
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 96 PHYSICAL cores, not 192 logical: the node is Sockets=4 CoresPerSocket=24
# ThreadsPerCore=2, and an SMT sibling is not extra compute for a compile.
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/build-judge-agent-amd-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/build-judge-agent-amd-%j.err

set -Eeuxo pipefail

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_nid<node>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0

# Slurm SPOOLS the batch script, so BASH_SOURCE points into /var/spool/slurmd -- the directory is
# passed in rather than derived. Deriving it is what killed build job 600467. SLURM_SUBMIT_DIR is
# the submit directory, so `cd <this dir> && sbatch build.sbatch` needs no variable; this is the
# same fallback the other four builders use, and having a fifth spelling of it cost job 619973.
IMAGE_DIR="${IMAGE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -x "${IMAGE_DIR}/build.sh"

mkdir -p "${SCRATCH:?}/ce-images/logs"

export OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?}/ce-images/optarena-ce-amd-mi300-candidate.sqsh}"

# Refuse to write over an image some EDF points a running job at: overwriting a mounted squashfs
# is how a live arm starts reading a half-written inode table. Ask the EDFs rather than hard-code
# one filename, since an EDF is what a job actually resolves. The cost is that images nothing
# currently mounts are protected too, which is the safe direction to be wrong in.
edf_images() {
    grep -hoE '^[[:space:]]*image[[:space:]]*=[[:space:]]*"[^"]+"' "${HOME}/.edf"/*.toml 2>/dev/null \
      | sed -E 's/.*"(.*)"/\1/' | sed -E "s|\$\{SCRATCH\}|${SCRATCH}|g; s|\$SCRATCH|${SCRATCH}|g"
}
if edf_images | grep -qxF "${OUTPUT_SQSH}"; then
    echo "refusing to overwrite ${OUTPUT_SQSH}: an EDF in ${HOME}/.edf mounts it, so a running" >&2
    echo "arm would read a half-written squashfs. Build to a candidate name and promote by" >&2
    echo "rename once nothing is using it." >&2
    exit 2
fi

"${IMAGE_DIR}/build.sh"

ls -lh "${OUTPUT_SQSH}"
sha256sum "${OUTPUT_SQSH}" | tee "${OUTPUT_SQSH}.sha256"
cat "${OUTPUT_SQSH}.digest"
echo "AGENT IMAGE READY: ${OUTPUT_SQSH}"

# The SECOND target. build.sh builds `agent judge` by default, and this block reported only the
# first -- so the judge candidate shipped with no .sha256 and nothing saying it existed. The judge
# is the image that grades every submission; it should not be the one nobody mentions.
judge_sqsh="${SCRATCH:?}/ce-images/optarena-ce-judge-amd-mi300-candidate.sqsh"
if [ -f "${judge_sqsh}" ]; then
    ls -lh "${judge_sqsh}"
    sha256sum "${judge_sqsh}" | tee "${judge_sqsh}.sha256"
    [ -f "${judge_sqsh}.digest" ] && cat "${judge_sqsh}.digest"
    echo "JUDGE IMAGE READY: ${judge_sqsh}"
else
    echo "note: no judge candidate at ${judge_sqsh} (single-target build?)"
fi
echo "Verify it before it goes live: scripts/smoke_gpu_profilers.sh via"
echo "scripts/submit_gpu_profiler_smoke.sbatch -- it reads ARTIFACTS, not exit codes, and fails"
echo "rocprof-compute specifically when its passes run and its rows are dropped."
