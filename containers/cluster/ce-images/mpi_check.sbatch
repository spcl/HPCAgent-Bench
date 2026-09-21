#!/usr/bin/env bash
# Run mpi_gpu_check.sh against an image, on a GPU node, in a batch context.
#
#   IMAGE=$SCRATCH/ce-images/hpcagent-bench-ce-amd-mi300.sqsh \
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
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --no-requeue

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:?set SCRATCH}"
REPO="${REPO:-${SCRATCH}/hpcagent-bench}"
IMAGE="${IMAGE:-${SCRATCH}/ce-images/hpcagent-bench-ce-amd-mi300.sqsh}"
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
com.hooks.netstack.source = "host"
# The hooks hardcode the artifact path at /capstor/store; that tree does not exist,
# so no version or name resolves there.
# The hooks do NOT fail on a missing artifact -- they point libfabric and the RCCL plugin
# at files that are not there, the bind-mounts quietly do nothing, and RCCL drops from
# CXI to TCP sockets. The job still finishes; it is just far slower, and it reads as a
# slow model rather than a dead fabric.
#
# "host" was previously the documented OPT-OUT because it grafts 29 host libraries into
# the image and job 629822 died that way: a host libcurl needing glibc 2.38 met an image
# with 2.35. THAT NO LONGER APPLIES -- the base moved to Ubuntu 24.04 and the shipped
# libc.so.6 is "Ubuntu GLIBC 2.39-0ubuntu8.8", verified by extracting it from the pulled
# squashfs. glibc is backward compatible, so 2.39 satisfies every GLIBC_2.38 the grafted
# libraries ask for. Re-check this if the base image is ever moved backwards.
#
# Fabric speed is unaffected. fi_pingpong over cxi, 1 MB, beverin 2026-09-16:
#   host -> 2.3.1  19329 MB/s | 1.22.0  20011 MB/s | 2.3.1 explicit  19878 MB/s
# -- within noise of each other, at 4 CXI NICs per node.
#
# version/name are NOT set: they only index into the dead artifact tree. The host path is
# /opt/cray/libfabric/host, a root-owned symlink (-> 2.3.1). It cannot be pinned by
# annotation, so scripts/cscs/netstack_preflight.sh ASSERTS what it resolves to and aborts
# on a repoint, rather than letting one campaign straddle two fabric stacks.
com.hooks.cxi.enabled = "true"
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "rocm6"
# REQUIRED in host mode and ignored in artifact mode: the hook selects the plugin by a
# case on this value and calls common::err on an unrecognised one, so leaving it unset
# makes the hook a hard error rather than a silent fallback. rocm6 is the only variant
# installed on beverin (/opt/cscs/aws-ofi-ccl-plugin/). VERIFY on mi200 before using an
# mi200 image -- gfx90a may carry a different one.

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
