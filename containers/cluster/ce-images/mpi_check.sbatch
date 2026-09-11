#!/usr/bin/env bash
# Run mpi_gpu_check.sh against an image, on a GPU node, in a batch context.
#
#   IMAGE=$SCRATCH/ce-images/optarena-ce-amd-mi300.sqsh \
#     sbatch containers/cluster/ce-images/mpi_check.sbatch
#
# A BATCH context, not a login shell: the multi-rank leg runs `mpiexec -n 4`, and the GPU legs
# need visible devices. On a login node it would report SKIP for exactly the checks that matter.
#
#SBATCH --job-name=mpi-check
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
#SBATCH --time=00:20:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/mpi-check-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/mpi-check-%j.err

set -Eeuo pipefail
ulimit -c 0

# /x86_64: the real SCRATCH has it and the #SBATCH log paths above hardcode it, so a
# fallback without it silently points at a tree that does not exist.
SCRATCH="${SCRATCH:-/capstor/scratch/cscs/${USER}/x86_64}"
REPO="${REPO:-${SCRATCH}/optarena}"
IMAGE="${IMAGE:-${SCRATCH}/ce-images/optarena-ce-amd-mi300.sqsh}"
[ -f "${IMAGE}" ] || { echo "no image at ${IMAGE}" >&2; exit 2; }

# Generated, not taken from ~/.edf: this may point at a CANDIDATE no EDF names yet, and a
# checked-in EDF pointing at a candidate is how a candidate becomes live by accident.
#
# All FIVE hook annotations, matching judge-agent-amd/edf.toml.example exactly. The cxi hook alone
# was never enough: it supplies the provider, while 10-netstack supplies the libfabric itself and
# 90-aws-ofi-nccl the RCCL plugin, and that last one exit(0)s unless enabled == "true" -- which is
# why RCCL logged "Could not find: libnccl-net.so" and fell back to TCP (629800/629802). The image
# now ships no libfabric at all, so with a partial hook set libmpi.so does not even resolve and
# this check would fail a correct image for its own missing configuration.
#
# FI_PROVIDER is deliberately ABSENT: setting it on an image that also runs MPI aborted MPI_Init
# with "OFI call getinfo failed" (629966). Production judge-agent EDFs omit it and so must this.
# The log directory is mounted below so the provenance probe can leave its raw output behind: the
# check script's scratch dir is deleted when it exits, and that output is exactly what is wanted by
# hand when a provider or libfabric verdict comes back INCONCL.
EDF="${SCRATCH}/.tmp/mpicheck-${SLURM_JOB_ID}.toml"
mkdir -p "$(dirname "${EDF}")" "${SCRATCH}/ce-images/logs"
cat > "${EDF}" <<TOML
image = "${IMAGE}"
mounts = ["${REPO}:${REPO}", "${SCRATCH}/ce-images/logs:${SCRATCH}/ce-images/logs"]
workdir = "${REPO}"

[annotations]
com.hooks.netstack.source = "artifact"
com.hooks.netstack.version = "26.08.1"
com.hooks.netstack.name = "gpu_rocm7-cxi_13.1.0-ofi_2.6.0-aws_1.20.0"
com.hooks.cxi.enabled = "true"
com.hooks.aws_ofi_nccl.enabled = "true"

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
MPI_CHECK_EVIDENCE="${SCRATCH}/ce-images/logs/mpi-provenance-${SLURM_JOB_ID}.log" \
    srun --environment="${EDF}" "${REPO}/containers/cluster/ce-images/mpi_gpu_check.sh" || rc=$?
rm -f "${EDF}"
echo "=== mpi_gpu_check rc=${rc}"
exit "${rc}"
