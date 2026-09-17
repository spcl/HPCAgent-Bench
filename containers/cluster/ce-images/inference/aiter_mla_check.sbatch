#!/usr/bin/env bash
# Do aiter's MLA kernels work on the sglang image, and do they agree with a reference?
#
#   IMAGE=$SCRATCH/ce-images/hpcagent-bench-sglang-candidate.sqsh \
#     sbatch containers/cluster/ce-images/inference/aiter_mla_check.sbatch
#
# ONE NODE, MINUTES, NO SERVER. Every previous aiter attempt here was inconclusive for a reason
# that was never about the kernels: SGLANG_USE_AITER=1 drives the JIT into a per-module baton lock
# that wedges serving for hours (0-for-6), and the master switch separately broke MLA prefill. So
# the question "are the MLA kernels correct on this stack" has never actually been asked. This asks
# it directly and cannot wedge, because it never starts a server.
#
# It is a PREREQUISITE for enabling AITER MLA in a serving config, not a substitute for measuring
# one: a kernel that is correct here can still lose to triton end to end.
#SBATCH --job-name=aiter-mla-check
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:?set SCRATCH}"
REPO="${REPO:-${SCRATCH}/hpcagent-bench}"
IMAGE="${IMAGE:-${SCRATCH}/ce-images/hpcagent-bench-sglang.sqsh}"
[ -f "${IMAGE}" ] || { echo "no image at ${IMAGE}" >&2; exit 2; }
WORK="${SCRATCH}/ce-images/logs"
mkdir -p "${WORK}"

# Generated, so this can point at a CANDIDATE no EDF names yet. The fabric annotations are the
# production set even though this test is single-node and touches no network: an EDF that differs
# from production is not testing the image anyone will run.
EDF="${WORK}/aiter-mla-${SLURM_JOB_ID}.toml"
cat > "${EDF}" <<TOML
image = "${IMAGE}"
mounts = ["${REPO}:${REPO}", "${WORK}:${WORK}"]
workdir = "${REPO}"
writable = true

[annotations]
com.hooks.netstack.source = "host"
# The artifact bundle lived under /capstor/store, which CSCS decommissioned in Sep 2026;
# the path is hardcoded in the hooks, so no version or name can resolve there any more.
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
PATH = "/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
HWLOC_COMPONENTS = "-gl"
TOML

echo "=== image: ${IMAGE}"
echo "=== node:  $(hostname)"
rc=0
srun --environment="${EDF}" /opt/venv/bin/python3 \
     "${REPO}/containers/cluster/ce-images/inference/aiter_mla_check.py" "$@" || rc=$?
rm -f "${EDF}"
echo "=== aiter_mla_check rc=${rc}"
exit "${rc}"
