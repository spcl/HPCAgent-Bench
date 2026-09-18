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
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -Eeuo pipefail

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_nid<node>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
SCRATCH="${SCRATCH:?set SCRATCH}"
# ANY image, not just the judge-agent one. This replaced sglang_cxi_check.sbatch and
# experiments/test-rccl-ofi-2node.sbatch, both of which ran a variant of this test but were written
# for the abandoned design -- one expecting the image's own /opt/ofi/lib to win over the hook, the
# other gating a plugin at /opt/aws-ofi-nccl. Nothing ships a fabric library now, so both tested a
# configuration that can no longer occur:
#
#   IMAGE=${SCRATCH}/ce-images/hpcagent-bench-sglang-mi300.sqsh sbatch rccl_hook_check.sbatch
#
# The PATH below ends in /opt/venv/bin, which is where the inference images keep their python, so
# one EDF serves every role.
IMAGE="${IMAGE:-${SCRATCH}/ce-images/hpcagent-bench-agent-mi300.sqsh}"
WORK="${SCRATCH}/.tmp/rcclhook-${SLURM_JOB_ID}"
mkdir -p "${WORK}" "${SCRATCH}/ce-images/logs"

EDF="${WORK}/e.toml"
cat > "${EDF}" <<TOML
image = "${IMAGE}"
mounts = ["${WORK}:${WORK}"]
workdir = "${WORK}"

[annotations]
# "host" (variant=rocm6) is ABI-dead on this image: /opt/cscs/aws-ofi-ccl-plugin/rocm6/
# librccl-net.so needs libamdhip64.so.6 (`ldd`, confirmed 2026-09-18), and the image ships ROCm
# 7.2 (libamdhip64.so.7 only). Every job that reached RCCL init under host+rocm6 logged the same
# two lines regardless of launcher (CE 640910, enroot 640950/641057/641172):
#   NET/Plugin: libamdhip64.so.6: cannot open shared object file: No such file or directory
#   NET/Plugin: Could not find: ofi.
# /capstor/store is reachable again (confirmed 2026-09-18), so back to "artifact": its bundle
# carries its own libamdhip64.so.7 (RUNPATH points at /opt/cscs/netstack, where 10-netstack.sh
# mounts the whole bundle), matching this image. This is also the config job 629967 actually
# proved ("/opt/cscs/netstack/librccl-net.so", NET/OFI, correct result) -- "host" was a
# migration-era substitute for when /capstor/store was unreachable, and 629967's citation was
# kept without re-running it against the substitute.
com.hooks.netstack.source = "artifact"
com.hooks.netstack.version = "26.08.1"
com.hooks.netstack.name = "gpu_rocm7-cxi_13.1.0-ofi_2.6.0-aws_1.20.0"
com.hooks.cxi.enabled = "true"
com.hooks.aws_ofi_nccl.enabled = "true"

[env]
PATH = "/opt/gcc/bin:/opt/view/bin:/opt/papi/bin:/opt/rocm/bin:/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# LD_LIBRARY_PATH is REQUIRED, not optional. The aws_ofi_nccl hook mounts the RCCL net plugin and
# sets NCCL_NET_PLUGIN=ofi, but the plugin is dlopen()ed and has no RPATH: it needs
# libamdhip64.so.6 from /opt/rocm/lib. Without this line RCCL logged
#   NET/Plugin: libamdhip64.so.6: cannot open shared object file
#   NET/Plugin: Could not find: ofi.
# and, because the hook also forces NCCL_NET='AWS Libfabric', ranks aborted instead of falling back
# (job 640046). torch itself still worked, since it carries its own RPATH to ROCm -- which is why a
# single-node GPU check passes while the cross-node collective fails. Same value as the agent EDF.
LD_LIBRARY_PATH = "/opt/rocm/lib:/opt/rocm/lib64:/opt/gcc/lib64:/opt/gcc/lib:/opt/view/lib:/opt/view/lib64:/opt/papi/lib:/usr/lib/x86_64-linux-gnu"
# FI_PROVIDER is deliberately NOT set here. Setting it globally is what killed 629966: MPI picked
# it up and aborted on a libfabric with no cxi. RCCL reaches libfabric through the plugin, which
# the hook installs already pointed at the artifact's own libfabric.
HWLOC_COMPONENTS = "-gl"
FI_CXI_RX_MATCH_MODE = "software"
FI_MR_CACHE_MONITOR = "disabled"
NCCL_DEBUG = "INFO"
NCCL_DEBUG_SUBSYS = "INIT,NET"
TOML

# LAUNCH=pyxis (default) | enroot. pyxis is the supported path; enroot routes around it while the
# the site enroot.conf default does not work here, and every `srun --environment=`
# dies at task_init(). Both read the SAME ${EDF} written above, so they test the same container.
# Comm hooks are forced ON for enroot: this job exists to test them, and enroot_srun.sh would
# otherwise turn them off (it derives them from INFERENCE_NODES, which is unset here).
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
