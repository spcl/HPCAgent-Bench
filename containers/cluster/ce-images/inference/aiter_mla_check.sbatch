#!/usr/bin/env bash
# Do aiter's MLA kernels work on the sglang image, and do they agree with a reference?
#
#   IMAGE=$SCRATCH/ce-images/optarena-sglang-candidate.sqsh \
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
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/aiter-mla-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/aiter-mla-%j.err

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/${USER}/x86_64}"
REPO="${REPO:-${SCRATCH}/optarena}"
IMAGE="${IMAGE:-${SCRATCH}/ce-images/optarena-sglang.sqsh}"
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
com.hooks.netstack.source = "artifact"
com.hooks.netstack.version = "26.08.1"
com.hooks.netstack.name = "gpu_rocm7-cxi_13.1.0-ofi_2.6.0-aws_1.20.0"
com.hooks.cxi.enabled = "true"
com.hooks.aws_ofi_nccl.enabled = "true"

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
