#!/usr/bin/env bash
# Cross-node MPI, proven by running across two nodes. MPI only: rccl_hook_check.sbatch already
# covers the identical RCCL leg, so this is not duplicated here.
#
#   IMAGE=$SCRATCH/ce-images/hpcagent-bench-ce-amd-mi300.sqsh \
#     sbatch containers/images/mpi_multinode_check.sbatch
#
# Separate from mpi_gpu_check.sh (single node): intra-node collectives use XGMI/IPC and never
# touch the network, so Slingshot/libfabric/CXI/RCCL-net-plugin claims need two nodes to test.
#
# The failure mode is silent by construction: a broken net plugin falls back to TCP and still
# gets the right (slow) answer, and a launcher mismatch can give each rank its own COMM_WORLD of
# size 1, each solving the whole problem alone. Both look like success in a log.
#SBATCH --job-name=mpi-mn-check
# The allocation's task shape must match the steps: 1 task/node then `--ntasks-per-node=4` fails
# with "More processors requested than permitted".
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
# 4 ranks x 24 physical cores = 96, matching the judge's shape (192 would hand out SMT siblings).
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --no-requeue

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:?set SCRATCH}"
REPO="${REPO:-${SCRATCH}/hpcagent-bench}"
IMAGE="${IMAGE:-${SCRATCH}/ce-images/hpcagent-bench-ce-amd-mi300.sqsh}"
[ -f "${IMAGE}" ] || { echo "no image at ${IMAGE}" >&2; exit 2; }
WORK="${SCRATCH}/.tmp/mpimn-${SLURM_JOB_ID}"
mkdir -p "${WORK}" "${SCRATCH}/ce-images/logs"

# The cxi hook puts the host libfabric (with the CXI provider) in front of the container's
# compile-only one; the judge-agent image ships headers and a link-only .so, no provider.
EDF="${WORK}/edf.toml"
cat > "${EDF}" <<TOML
image = "${IMAGE}"
mounts = ["${REPO}:${REPO}", "${WORK}:${WORK}"]
workdir = "${WORK}"

[annotations]
# All three CSCS hooks: 10-netstack (source), 89-libfabric-cxi (provider, /dev/cxi*), and
# 90-aws-ofi-nccl (exits unless enabled == "true").
# "host" grafts host libraries; the base's glibc satisfies what they need. It is a coherent
# bundle, not just cxi: also carries libc.so.6, libcurl.so.4, libcxi.so.1, libfabric.so.1.
com.hooks.netstack.source = "host"
# version/name are omitted: they only index the artifact tree, unused in host mode. The host path
# is a root-owned symlink.
com.hooks.cxi.enabled = "true"
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "rocm6"
# Required in host mode (unset is a hard error in the hook); ignored in artifact mode. Verify on
# mi200 before reuse: gfx90a may need a different variant.

[env]
PATH = "/opt/gcc/bin:/opt/view/bin:/opt/papi/bin:/opt/rocm/bin:/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# Avoids a hang at NCCL init (opencl/levelzero probes); "-gl" alone is not enough.
HWLOC_COMPONENTS = "-opencl,-levelzero,-gl"
# MPICH is spack-built with its own hwloc RPATH, so it ignores the CXI hook's host hwloc unless
# preloaded; without this, hwloc_get_pcidev_by_busid fails and MPICH aborts in MPI_Init. Preload
# the one file only: LD_LIBRARY_PATH=/opt/cscs/netstack would shadow ~49 host libraries and break
# TaskProlog.
LD_PRELOAD = "/opt/cscs/netstack/libhwloc.so.15"
# FI_PROVIDER deliberately unset: MPICH inherits it and aborts in MPI_Init against a libfabric
# with no cxi provider.
FI_CXI_RX_MATCH_MODE = "software"
FI_MR_CACHE_MONITOR = "disabled"
# Set here, not on `srun --export`: that list splits on commas and would break "INIT,NET" apart.
NCCL_DEBUG = "INFO"
NCCL_DEBUG_SUBSYS = "INIT,NET"
LD_LIBRARY_PATH = "/opt/gcc/lib64:/opt/gcc/lib:/opt/view/lib:/opt/view/lib64:/opt/papi/lib:/opt/rocm/lib:/opt/rocm/lib64"
TOML

cat > "${WORK}/mn_mpi.c" <<'C'
/* Cross-node MPI: one COMM_WORLD spanning both nodes, host AND device buffers. */
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <hip/hip_runtime.h>

#define HIP_OK(c) do { hipError_t e=(c); if(e!=hipSuccess){ \
    fprintf(stderr,"hip %s at %d\n",hipGetErrorString(e),__LINE__); MPI_Abort(MPI_COMM_WORLD,2);} } while(0)

int main(int argc, char **argv) {
    int rank=0, size=0, one=1, sum=0, nnodes=0, ndev=0, host=0;
    char name[MPI_MAX_PROCESSOR_NAME]; int nlen=0;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Get_processor_name(name,&nlen);

    /* WHICH libfabric this process actually mapped. Every assertion below is satisfied just as
       well by cross-node MPI over the tcp provider, so without this the job cannot tell the
       netstack artifact from the image's compile-only stub, which has no cxi provider at all. */
    if (rank==0) {
        FILE *m = fopen("/proc/self/maps","r");
        char line[512]; int shown=0;
        while (m && fgets(line,sizeof(line),m)) {
            char *slash;
            if (!strstr(line,"libfabric.so")) continue;
            line[strcspn(line,"\n")] = 0;
            slash = strchr(line,'/');
            printf("libfabric=%s\n", slash ? slash : line);
            shown=1; break;
        }
        if (m) fclose(m);
        if (!shown) printf("libfabric=none\n");
        fflush(stdout);
    }

    /* How many DISTINCT nodes this communicator actually spans. A launcher that quietly kept the
       ranks on one node would still allreduce correctly and prove nothing about the network. */
    MPI_Comm shared;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    int srank=0; MPI_Comm_rank(shared,&srank);
    int leader = (srank==0) ? 1 : 0;
    MPI_Allreduce(&leader,&nnodes,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

    MPI_Allreduce(&one,&sum,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

    HIP_OK(hipGetDeviceCount(&ndev));
    HIP_OK(hipSetDevice(srank % (ndev>0?ndev:1)));
    if (MPIX_GPU_query_support(MPIX_GPU_SUPPORT_HIP,&host)!=MPI_SUCCESS) host=0;

    int *ds=0,*dr=0,got=0;
    HIP_OK(hipMalloc((void**)&ds,sizeof(int)));
    HIP_OK(hipMalloc((void**)&dr,sizeof(int)));
    HIP_OK(hipMemcpy(ds,&one,sizeof(int),hipMemcpyHostToDevice));
    MPI_Allreduce(ds,dr,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    HIP_OK(hipMemcpy(&got,dr,sizeof(int),hipMemcpyDeviceToHost));

    if (rank==0)
        printf("size=%d nodes=%d host_allreduce=%d device_allreduce=%d gpu_query=%s\n",
               size,nnodes,sum,got,host?"yes":"no");
    MPI_Finalize();
    return (sum==size && got==size && nnodes>=2 && host) ? 0 : 1;
}
C
echo "=== image: ${IMAGE}"
echo "=== nodes: ${SLURM_JOB_NODELIST}"

echo "=== compile (one node, shared filesystem) ==="
timeout --signal=TERM 600 \
  srun -N1 -n1 -c24 --overlap --environment="${EDF}" bash -c "
  set -eux
  mpicc -O0 -o ${WORK}/mn_mpi ${WORK}/mn_mpi.c -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -L/opt/rocm/lib -lamdhip64
"

rc_mpi=0; verdict=0
echo
echo "=== 1. CROSS-NODE MPI (2 nodes x 4 ranks) ==="
# --mpi=pmi2: without it ranks come up isolated instead of finding each other across nodes.
# The capability-sets/core log env vars are the only way to see which provider MPICH selected.
# Passed as an env prefix, not --export, since that list splits on commas.
MPIR_CVAR_CH4_OFI_CAPABILITY_SETS_DEBUG=1 FI_LOG_LEVEL=warn FI_LOG_PROV=core \
timeout --signal=TERM 420 \
  srun -N2 --ntasks-per-node=4 -c24 --mpi=pmi2 --environment="${EDF}" \
     "${WORK}/mn_mpi" 2>&1 | tee "${WORK}/mpi.log" || rc_mpi=$?

echo
echo "=== 2. WHICH libfabric did that process map, and WHICH provider did it choose? ==="
fab="$(grep -m1 -oE 'libfabric=[^ ]+' "${WORK}/mpi.log" | cut -d= -f2- || true)"
case "${fab}" in
    ""|none)
        echo "  INCONCL: the MPI process mapped no libfabric.so at all" ;;
    /opt/ofi-sdk/*)
        echo "  FAIL: mapped ${fab} -- the compile-only link stub; it was not deleted from the image" >&2
        verdict=1 ;;
    /opt/spack-install/*)
        echo "  FAIL: mapped ${fab} -- SPACK built its own; the packages.yaml external did not take" >&2
        verdict=1 ;;
    /opt/cscs/*)
        echo "  OK: mapped ${fab} (the pinned netstack artifact)" ;;
    *)
        echo "  INCONCL: mapped ${fab}" ;;
esac
prov="$(grep -oiE 'provider: *[A-Za-z0-9_]+|using provider[:= ]+[A-Za-z0-9_]+' "${WORK}/mpi.log" \
        | sort -u | tr '\n' ' ' || true)"
if [ -z "${prov}" ]; then
    echo "  INCONCL: MPICH printed no provider line"
elif printf '%s' "${prov}" | grep -qi cxi; then
    echo "  OK: provider ${prov}"
else
    echo "  FAIL: provider ${prov} -- not cxi, so this never touched Slingshot" >&2
    verdict=1
fi

echo
echo "=== cross-node MPI rc=${rc_mpi}  fabric verdict=${verdict}"
s=$(( rc_mpi + verdict ))

# Preserve logs only on failure, when they are wanted for diagnosis.
for f in "${WORK}"/*.log; do
    if [ -e "${f}" ]; then cp -f "${f}" "${SCRATCH}/ce-images/logs/mpi-mn-${SLURM_JOB_ID}-$(basename "${f}")"; fi
done
if [ "${s}" -eq 0 ]; then rm -rf "${WORK}"; fi

# Explicit: `exit` truncates mod 256, so a raw sum could wrap to 0.
[ "${s}" -eq 0 ] || exit 1
exit 0
