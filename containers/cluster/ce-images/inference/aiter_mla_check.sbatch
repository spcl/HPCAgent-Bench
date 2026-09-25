#!/usr/bin/env bash
# Do aiter's MLA kernels work on the sglang image, and do they agree with a reference?
#
#   IMAGE=$SCRATCH/ce-images/hpcagent-bench-sglang-candidate.sqsh \
#     sbatch containers/cluster/ce-images/inference/aiter_mla_check.sbatch
#
# One node, minutes, no server: SGLANG_USE_AITER=1 previously wedged serving via a JIT baton
# lock before the kernels themselves could even be judged, so this asks the kernel-correctness
# question directly instead. A prerequisite for enabling AITER MLA in serving, not a substitute
# for measuring it: correct here can still lose to triton end to end.
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
#SBATCH --no-requeue

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:?set SCRATCH}"
REPO="${REPO:-${SCRATCH}/hpcagent-bench}"
IMAGE="${IMAGE:-${SCRATCH}/ce-images/hpcagent-bench-sglang.sqsh}"
[ -f "${IMAGE}" ] || { echo "no image at ${IMAGE}" >&2; exit 2; }
WORK="${SCRATCH}/ce-images/logs"
mkdir -p "${WORK}"

# Generated, so this can point at a CANDIDATE no EDF names yet. Fabric annotations match
# production even though this test touches no network: a differing EDF tests the wrong image.
EDF="${WORK}/aiter-mla-${SLURM_JOB_ID}.toml"
cat > "${EDF}" <<TOML
image = "${IMAGE}"
mounts = ["${REPO}:${REPO}", "${WORK}:${WORK}"]
workdir = "${REPO}"
writable = true

[annotations]
com.hooks.netstack.source = "host"
# "host" grafts host libraries; requires the base's glibc >= what they need (currently true).
# version/name are omitted: they only index the artifact tree. netstack_preflight.sh asserts the
# resolved host libfabric path so one campaign cannot straddle two fabric stacks unnoticed.
com.hooks.cxi.enabled = "true"
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "rocm6"
# Required in host mode (unset is a hard error in the hook); ignored in artifact mode. Verify on
# mi200 before reuse: gfx90a may need a different variant.

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
