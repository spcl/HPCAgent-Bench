#!/usr/bin/env bash
# Run mpi_gpu_check.sh against an image, on a GPU node, in a batch context.
#
#   IMAGE=$CE_IMAGES/<candidate>.sqsh \
#     sbatch containers/images/mpi_check.sbatch
#
# Needs a batch context: a login shell reports SKIP for the multi-rank and GPU legs.
#SBATCH --job-name=mpi-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 96 physical cores (not 192 SMT threads); matches the judge's 4 ranks x 24 shape.
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --no-requeue

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:?set SCRATCH}"
REPO="${REPO:-${SCRATCH}/hpcagent-bench}"
# shellcheck source=images.env
source "${REPO}/containers/images/images.env"
IMAGE="${IMAGE:-${CE_IMAGES}/${JUDGE_AGENT_AMD_SQSH}}"
[ -f "${IMAGE}" ] || { echo "no image at ${IMAGE}" >&2; exit 2; }

# Generated, not taken from ~/.edf, so a candidate image can be checked before any EDF names it.
# All FIVE hook annotations, matching judge-agent-amd/agent.edf.toml.in: any subset leaves libfabric
# or the RCCL plugin unresolved. FI_PROVIDER is deliberately unset: setting it aborts MPI_Init.
# logs/ is mounted so the provenance probe's raw output survives the check script's own cleanup.
EDF="${SCRATCH}/.tmp/mpicheck-${SLURM_JOB_ID}.toml"
mkdir -p "$(dirname "${EDF}")" "${CE_IMAGES}/logs"
cat > "${EDF}" <<TOML
image = "${IMAGE}"
mounts = ["${REPO}:${REPO}", "${CE_IMAGES}/logs:${CE_IMAGES}/logs"]
workdir = "${REPO}"

[annotations]
com.hooks.netstack.source = "host"
# "host" grafts host libraries into the image; requires the base's glibc >= what they need
# (currently true). A missing artifact would silently drop RCCL to TCP instead of failing loud.
# version/name are omitted: they only index the artifact tree.
com.hooks.cxi.enabled = "true"
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "rocm6"
# Required in host mode (unset is a hard error in the hook); ignored in artifact mode. Verify on
# mi200 before reuse: gfx90a may need a different variant.

[env]
PATH = "/opt/gcc/bin:/opt/view/bin:/opt/papi/bin:/opt/rocm/bin:/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
LD_LIBRARY_PATH = "/opt/gcc/lib64:/opt/gcc/lib:/opt/view/lib:/opt/view/lib64:/opt/papi/lib:/opt/rocm/lib:/opt/rocm/lib64"
HWLOC_COMPONENTS = "-opencl,-levelzero,-gl"
FI_CXI_RX_MATCH_MODE = "software"
FI_MR_CACHE_MONITOR = "disabled"
TOML

echo "=== image:  ${IMAGE}"
echo "=== node:   $(hostname)"
rc=0
MPI_CHECK_EVIDENCE="${CE_IMAGES}/logs/mpi-provenance-${SLURM_JOB_ID}.log" \
    srun --environment="${EDF}" "${REPO}/containers/images/mpi_gpu_check.sh" || rc=$?
rm -f "${EDF}"
echo "=== mpi_gpu_check rc=${rc}"
exit "${rc}"
