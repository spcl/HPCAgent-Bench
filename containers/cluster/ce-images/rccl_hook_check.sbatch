#!/usr/bin/env bash
# Does the CSCS aws_ofi_nccl hook give RCCL the fabric? No MPI here: an MPI_Init failure would
# hide the RCCL answer, and torch.distributed only needs MASTER_ADDR/PORT.
#SBATCH --job-name=rccl-hook
#SBATCH --partition=mi300
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --no-requeue

set -Eeuo pipefail

# No core dumps: they land in the CWD on an inode-quota filesystem.
ulimit -c 0
SCRATCH="${SCRATCH:?set SCRATCH}"
# Any image: none ships a fabric library, the hooks supply it.
#   IMAGE=${SCRATCH}/ce-images/hpcagent-bench-sglang-mi300.sqsh sbatch rccl_hook_check.sbatch
IMAGE="${IMAGE:-${SCRATCH}/ce-images/hpcagent-bench-agent-mi300.sqsh}"
WORK="${SCRATCH}/.tmp/rcclhook-${SLURM_JOB_ID}"
mkdir -p "${WORK}" "${SCRATCH}/ce-images/logs"

EDF="${WORK}/e.toml"
cat > "${EDF}" <<TOML
image = "${IMAGE}"
mounts = ["${WORK}:${WORK}"]
workdir = "${WORK}"

[annotations]
# "host" variant=rocm6 is ABI-dead here: its librccl-net.so needs libamdhip64.so.6, but the image
# ships ROCm 7.2 (.so.7 only). "artifact" carries its own matching libamdhip64.so.7.
com.hooks.netstack.source = "artifact"
com.hooks.netstack.version = "26.08.1"
com.hooks.netstack.name = "gpu_rocm7-cxi_13.1.0-ofi_2.6.0-aws_1.20.0"
com.hooks.cxi.enabled = "true"
com.hooks.aws_ofi_nccl.enabled = "true"
# Must stay set even in artifact mode: the hook's case statement has no `set -u` guard on its
# *) branch message, so an unset variant is a hard crash regardless of whether it is read.
com.hooks.aws_ofi_nccl.variant = "rocm6"

[env]
PATH = "/opt/gcc/bin:/opt/view/bin:/opt/papi/bin:/opt/rocm/bin:/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# Required: the RCCL net plugin is dlopen()ed with no RPATH and needs libamdhip64.so.6 from here.
# torch carries its own RPATH to ROCm, so a single-node check can pass while this is missing and
# the cross-node collective still fails.
LD_LIBRARY_PATH = "/opt/rocm/lib:/opt/rocm/lib64:/opt/gcc/lib64:/opt/gcc/lib:/opt/view/lib:/opt/view/lib64:/opt/papi/lib:/usr/lib/x86_64-linux-gnu"
# FI_PROVIDER deliberately unset: MPI would pick it up and abort on a libfabric with no cxi.
# RCCL reaches libfabric through the plugin instead.
HWLOC_COMPONENTS = "-gl"
FI_CXI_RX_MATCH_MODE = "software"
FI_MR_CACHE_MONITOR = "disabled"
NCCL_DEBUG = "INFO"
NCCL_DEBUG_SUBSYS = "INIT,NET"
TOML

# LAUNCH=pyxis (default) | enroot; both read the same EDF. Comm hooks are forced on for enroot
# since this job exists to test them and enroot_srun.sh would otherwise default them off.
LAUNCH="${LAUNCH:-pyxis}"
REPO="${HPCAGENT_BENCH_REPO:-${SCRATCH}/hpcagent-bench}"
in_image() {
    if [ "${LAUNCH}" = "enroot" ]; then
        EDF_DIR="${WORK}" HPCAGENT_BENCH_COMM_HOOKS=on "${REPO}/scripts/cscs/enroot_srun.sh" e "$@"
    else
        local srun_args=() ; while [ $# -gt 0 ] && [ "$1" != "--" ]; do srun_args+=("$1"); shift; done
        shift
        srun "${srun_args[@]}" --environment="${EDF}" "$@"
    fi
}
echo "=== launch: ${LAUNCH}"

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
    # After the collective so the plugin is already dlopened: this says what RCCL actually loaded,
    # not just what files exist.
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
in_image -N1 -n1 -c24 --overlap -- bash -c '
  ls -la /opt/cscs/netstack/ 2>/dev/null | head -8 || echo "  no /opt/cscs/netstack"
  echo "  plugin:"; find /opt /usr/lib /usr/lib64 \( -name "libnccl-net.so*" -o -name "librccl-net.so*" \) \
       2>/dev/null | head -4
  echo "  libfabric:"; find /opt /usr/lib /usr/lib64 -name "libfabric.so.1" 2>/dev/null | head -4' || true

echo
echo "--- cross-node RCCL, 2 nodes x 4 ranks, no MPI"
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"
export MASTER_PORT=29601
rc=0
in_image -N2 --ntasks-per-node=4 -c24 \
     --export=ALL,MASTER_ADDR="${MASTER_ADDR}",MASTER_PORT="${MASTER_PORT}" \
     -- python3 "${WORK}/t.py" 2>&1 | tee "${WORK}/r.log" || rc=$?

echo
echo "--- transport"
# The verdict must reach the exit code: a TCP allreduce is numerically correct, so rc alone misses
# the failure this job exists to catch (right answer, wrong fabric). Inconclusive also fails.
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
# Preserve logs only on failure, when they are wanted for diagnosis.
for f in "${WORK}"/*.log; do
    if [ -e "${f}" ]; then cp -f "${f}" "${SCRATCH}/ce-images/logs/rccl-hook-${SLURM_JOB_ID}-$(basename "${f}")"; fi
done
if [ "${s}" -eq 0 ]; then rm -rf "${WORK}"; fi
[ "${s}" -eq 0 ] || exit 1
exit 0
