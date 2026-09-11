#!/usr/bin/env bash
# Does the CSCS aws_ofi_nccl hook give RCCL the fabric? NO MPI ANYWHERE IN THIS TEST.
#
# 629966 tried to answer this and could not: it bootstraps RCCL over MPI, and MPI_Init aborted
# first -- FI_PROVIDER=cxi against MPICH's spack libfabric, which has no cxi provider, so
# "OFI call getinfo failed". The RCCL question was never reached. torch.distributed needs only
# MASTER_ADDR/PORT, so this isolates RCCL from MPI entirely.
#
#SBATCH --job-name=rccl-hook
#SBATCH --partition=mi300
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/rccl-hook-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/rccl-hook-%j.err

set -Eeuo pipefail
SCRATCH="${SCRATCH:-/capstor/scratch/cscs/${USER}/x86_64}"
# ANY image, not just the judge-agent one. This replaced sglang_cxi_check.sbatch and
# experiments/test-rccl-ofi-2node.sbatch, both of which ran a variant of this test but were written
# for the abandoned design -- one expecting the image's own /opt/ofi/lib to win over the hook, the
# other gating a plugin at /opt/aws-ofi-nccl. Nothing ships a fabric library now, so both tested a
# configuration that can no longer occur:
#
#   IMAGE=${SCRATCH}/ce-images/optarena-sglang.sqsh sbatch rccl_hook_check.sbatch
#
# The PATH below ends in /opt/venv/bin, which is where the inference images keep their python, so
# one EDF serves every role.
IMAGE="${IMAGE:-${SCRATCH}/ce-images/optarena-ce-amd-mi300.sqsh}"
WORK="${SCRATCH}/.tmp/rcclhook-${SLURM_JOB_ID}"
mkdir -p "${WORK}" "${SCRATCH}/ce-images/logs"

EDF="${WORK}/e.toml"
cat > "${EDF}" <<TOML
image = "${IMAGE}"
mounts = ["${WORK}:${WORK}"]
workdir = "${WORK}"

[annotations]
com.hooks.netstack.source = "artifact"
com.hooks.netstack.version = "26.08.1"
com.hooks.netstack.name = "gpu_rocm7-cxi_13.1.0-ofi_2.6.0-aws_1.20.0"
com.hooks.cxi.enabled = "true"
com.hooks.aws_ofi_nccl.enabled = "true"

[env]
PATH = "/opt/gcc/bin:/opt/view/bin:/opt/papi/bin:/opt/rocm/bin:/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# FI_PROVIDER is deliberately NOT set here. Setting it globally is what killed 629966: MPI picked
# it up and aborted on a libfabric with no cxi. RCCL reaches libfabric through the plugin, which
# the hook installs already pointed at the artifact's own libfabric.
HWLOC_COMPONENTS = "-gl"
FI_CXI_RX_MATCH_MODE = "software"
FI_MR_CACHE_MONITOR = "disabled"
NCCL_DEBUG = "INFO"
NCCL_DEBUG_SUBSYS = "INIT,NET"
TOML

cat > "${WORK}/t.py" <<'PY'
import os, sys, torch, torch.distributed as dist
rank = int(os.environ["SLURM_PROCID"]); world = int(os.environ["SLURM_NTASKS"])
local = int(os.environ["SLURM_LOCALID"])
torch.cuda.set_device(local)
dist.init_process_group("nccl", rank=rank, world_size=world)
names = [None] * world
dist.all_gather_object(names, os.uname().nodename)
nodes = len(set(names))
t = torch.full((1 << 20,), float(rank + 1), device=f"cuda:{local}")
dist.all_reduce(t, op=dist.ReduceOp.SUM)
expect = world * (world + 1) / 2
ok = bool(torch.all(t == expect).item())
if rank == 0:
    # After the collective, so the plugin and its libfabric are already dlopened. The listing
    # further up only says which files EXIST; these lines say which ones RCCL actually loaded.
    with open("/proc/self/maps") as fh:
        mapped = sorted({l.split()[-1] for l in fh if "libfabric" in l or "ccl-net" in l})
    for path in mapped:
        print(f"MAPPED {path}", flush=True)
    if not mapped:
        print("MAPPED none -- no libfabric and no net plugin in this address space", flush=True)
    print(f"RESULT world={world} nodes={nodes} allreduce={t[0].item():.0f} expected={expect:.0f} correct={ok}", flush=True)
dist.destroy_process_group()
sys.exit(0 if (ok and nodes >= 2) else 1)
PY

echo "=== image ${IMAGE}"
echo "--- what the hooks put in the container"
srun -N1 -n1 -c24 --overlap --environment="${EDF}" bash -c '
  ls -la /opt/cscs/netstack/ 2>/dev/null | head -8 || echo "  no /opt/cscs/netstack"
  echo "  plugin:"; find /opt /usr/lib /usr/lib64 \( -name "libnccl-net.so*" -o -name "librccl-net.so*" \) \
       2>/dev/null | head -4
  echo "  libfabric:"; find /opt /usr/lib /usr/lib64 -name "libfabric.so.1" 2>/dev/null | head -4' || true

echo
echo "--- cross-node RCCL, 2 nodes x 4 ranks, no MPI"
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"
export MASTER_PORT=29601
rc=0
srun -N2 --ntasks-per-node=4 -c24 --environment="${EDF}" \
     --export=ALL,MASTER_ADDR="${MASTER_ADDR}",MASTER_PORT="${MASTER_PORT}" \
     python3 "${WORK}/t.py" 2>&1 | tee "${WORK}/r.log" || rc=$?

echo
echo "--- transport"
# THE VERDICT HAS TO REACH THE EXIT CODE. A TCP allreduce is numerically correct, so rc alone is
# 0 on exactly the failure this job exists to catch: right answer, wrong fabric. Inconclusive
# counts as a failure too -- no transport line means the run proved nothing either way.
tv=0
if grep -qiE 'NET/OFI|AWS Libfabric' "${WORK}/r.log"; then
    sel="$(grep -oiE 'Using network [A-Za-z ]+|NET/OFI[^ ]*' "${WORK}/r.log" | sort -u | tr '\n' ' ' || true)"
    echo "  OFI SELECTED: ${sel}"
elif grep -qiE 'NET/Socket' "${WORK}/r.log"; then
    echo "  STILL SOCKET -- correct result, wrong fabric" >&2
    grep -m2 -iE 'NET/Plugin|Using network' "${WORK}/r.log" || true
    tv=1
else
    echo "  INCONCLUSIVE: no transport line in the log" >&2
    tail -6 "${WORK}/r.log" || true
    tv=1
fi
grep -E "^MAPPED|^RESULT" "${WORK}/r.log" || true

s=$(( rc + tv ))
# The logs are wanted precisely when this fails; the old unconditional rm threw them away.
for f in "${WORK}"/*.log; do
    if [ -e "${f}" ]; then cp -f "${f}" "${SCRATCH}/ce-images/logs/rccl-hook-${SLURM_JOB_ID}-$(basename "${f}")"; fi
done
if [ "${s}" -eq 0 ]; then rm -rf "${WORK}"; fi
[ "${s}" -eq 0 ] || exit 1
exit 0
