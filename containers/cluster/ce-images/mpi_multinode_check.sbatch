#!/usr/bin/env bash
# CROSS-NODE MPI and CROSS-NODE RCCL, proven by running them across two nodes.
#
#   IMAGE=$SCRATCH/ce-images/optarena-ce-amd-mi300.sqsh \
#     sbatch containers/cluster/ce-images/mpi_multinode_check.sbatch
#
# WHY THIS IS SEPARATE FROM mpi_gpu_check.sh. That script runs on ONE node, and one node cannot
# fail the way this project cares about: intra-node collectives use XGMI/IPC and never touch the
# network at all. Every claim about Slingshot, libfabric, the CXI provider and the RCCL net plugin
# is a claim about traffic BETWEEN nodes, and the only way to test it is to have two.
#
# The failure this exists to catch is silent by construction. RCCL with no usable net plugin does
# not error -- it falls back to TCP sockets over hsn*, gets the right answer, and is slow. MPI with
# a wrapper/launcher mismatch does not error either: every rank comes up as its own COMM_WORLD of
# size 1 and each solves the whole problem. Both look like success in a log.
#
#SBATCH --job-name=mpi-mn-check
#SBATCH --partition=mi300
# The ALLOCATION must already have the task shape the steps ask for. Allocating 1 task/node and
# then running `srun --ntasks-per-node=4` gets "More processors requested than permitted" (629800):
# a step cannot exceed the job's task count, and --cpus-per-task=192 leaves room for exactly one.
# 4 ranks x 24 physical cores = 96 fills the node and matches the shape the campaigns use.
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
# 96, not 192, and --hint=nomultithread with it. The node is Sockets=4 CoresPerSocket=24
# ThreadsPerCore=2 -- 96 PHYSICAL cores behind 192 logical CPUs. Asking for 192 hands out SMT
# siblings, which on this system are not extra compute: two threads share one core's execution
# units, so a compile or a rank pinned across a sibling pair runs at core speed while the count
# says it has two. Every campaign shape here is physical-core based (4 ranks x 24), and the judge
# already spells it --cpus-per-task=24 --hint=nomultithread.
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/mpi-mn-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/mpi-mn-%j.err

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/${USER}/x86_64}"
REPO="${REPO:-${SCRATCH}/optarena}"
IMAGE="${IMAGE:-${SCRATCH}/ce-images/optarena-ce-amd-mi300.sqsh}"
[ -f "${IMAGE}" ] || { echo "no image at ${IMAGE}" >&2; exit 2; }
WORK="${SCRATCH}/.tmp/mpimn-${SLURM_JOB_ID}"
mkdir -p "${WORK}" "${SCRATCH}/ce-images/logs"

# The cxi hook is what puts the HOST libfabric (with the CXI provider) in front of the container's
# compile-only one. For the judge-agent image that is the whole runtime story: it ships headers and
# a .so to link against and deliberately no provider.
EDF="${WORK}/edf.toml"
cat > "${EDF}" <<TOML
image = "${IMAGE}"
mounts = ["${REPO}:${REPO}", "${WORK}:${WORK}"]
workdir = "${WORK}"

[annotations]
# THE CSCS-DOCUMENTED SET, not just the cxi hook. Three separate hooks matter and we had only one:
#   10-netstack.sh    defaults to source=artifact; setting source="host" makes it exit(0), which
#                     is how an EDF opts OUT of the good path into a raw host-filesystem graft.
#   89-libfabric-cxi  the cxi provider and /dev/cxi*.
#   90-aws-ofi-nccl   exits(0) unless aws_ofi_nccl.enabled == "true". We never set it, which is
#                     exactly why 629800/629802 logged "NET/Plugin: Could not find:
#                     libnccl-net.so" -- CSCS ships the plugin and we never asked for it.
#
# The artifact is a coherent BUNDLE, not a graft: it carries its own libc.so.6, libcurl.so.4,
# libcxi.so.1, libfabric.so.1. That is why the glibc/libcurl chain that killed sglang (629822)
# does not arise here -- the set is internally consistent by construction.
com.hooks.netstack.source = "artifact"
com.hooks.netstack.version = "26.08.1"
com.hooks.netstack.name = "gpu_rocm7-cxi_13.1.0-ofi_2.6.0-aws_1.20.0"
com.hooks.cxi.enabled = "true"
com.hooks.aws_ofi_nccl.enabled = "true"

[env]
PATH = "/opt/gcc/bin:/opt/view/bin:/opt/papi/bin:/opt/rocm/bin:/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# CSCS-documented, and absent from every EDF we had. HWLOC_COMPONENTS carries the note
# "this solves an hang happening at NCCL initialization time" -- which is the shape of the kimi
# failure: a 600 s stall on a TWO-element broadcast at communicator setup. Value verbatim from the
# production judge-agent EDF -- "-gl" alone left the opencl and levelzero probes in.
HWLOC_COMPONENTS = "-opencl,-levelzero,-gl"
# FI_PROVIDER is deliberately NOT set. MPICH inherits it, and 629966 aborted in MPI_Init with
# "OFI call getinfo failed" against a libfabric that has no cxi provider. The production
# judge-agent EDFs omit it for exactly this reason; only the sglang/vllm EDFs set it.
FI_CXI_RX_MATCH_MODE = "software"
FI_MR_CACHE_MONITOR = "disabled"
# Here and not on `srun --export`: that list splits on COMMAS, so INIT,NET parsed as
# NCCL_DEBUG_SUBSYS=INIT plus a bare variable named NET, and the NET lines the transport step
# greps for were never emitted at all.
NCCL_DEBUG = "INFO"
NCCL_DEBUG_SUBSYS = "INIT,NET"
# No /opt/aws-ofi-nccl/lib: that directory is gone. libfabric, libcxi and the RCCL net plugin all
# arrive from the pinned netstack artifact under /opt/cscs/netstack, put there by the hooks above.
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

cat > "${WORK}/mn_rccl.c" <<'C'
/* Cross-node RCCL: ncclCommInitRank bootstrapped over MPI, allreduce across BOTH nodes. */
#include <mpi.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 65536
#define HIP_OK(c) do { hipError_t e=(c); if(e!=hipSuccess){ \
    fprintf(stderr,"hip %s at %d\n",hipGetErrorString(e),__LINE__); MPI_Abort(MPI_COMM_WORLD,2);} } while(0)
#define NCCL_OK(c) do { ncclResult_t r=(c); if(r!=ncclSuccess){ \
    fprintf(stderr,"rccl %s at %d\n",ncclGetErrorString(r),__LINE__); MPI_Abort(MPI_COMM_WORLD,3);} } while(0)

int main(int argc,char**argv){
    int rank=0,size=0,ndev=0,nnodes=0;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm shared; MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&shared);
    int srank=0; MPI_Comm_rank(shared,&srank);
    int leader=(srank==0); MPI_Allreduce(&leader,&nnodes,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    if(nnodes<2){ if(!rank) fprintf(stderr,"ranks span %d node(s); need 2\n",nnodes);
                  MPI_Finalize(); return 9; }

    HIP_OK(hipGetDeviceCount(&ndev));
    HIP_OK(hipSetDevice(srank % (ndev>0?ndev:1)));

    ncclUniqueId id;
    if(rank==0) NCCL_OK(ncclGetUniqueId(&id));
    MPI_Bcast(&id,sizeof(id),MPI_BYTE,0,MPI_COMM_WORLD);

    ncclComm_t comm; hipStream_t st;
    HIP_OK(hipStreamCreate(&st));
    NCCL_OK(ncclCommInitRank(&comm,size,id,rank));

    /* Rank r contributes r+1, so the sum is size(size+1)/2 -- a value no rank holds, and one a
       no-op collective handing back its input cannot produce. */
    float *ds,*dr,*h=malloc(N*sizeof(float));
    HIP_OK(hipMalloc((void**)&ds,N*sizeof(float)));
    HIP_OK(hipMalloc((void**)&dr,N*sizeof(float)));
    for(int k=0;k<N;k++) h[k]=(float)(rank+1);
    HIP_OK(hipMemcpy(ds,h,N*sizeof(float),hipMemcpyHostToDevice));
    HIP_OK(hipMemset(dr,0,N*sizeof(float)));

    NCCL_OK(ncclAllReduce(ds,dr,N,ncclFloat,ncclSum,comm,st));
    HIP_OK(hipStreamSynchronize(st));
    HIP_OK(hipMemcpy(h,dr,N*sizeof(float),hipMemcpyDeviceToHost));

    float expect=(float)(size*(size+1)/2);
    int bad=0; for(int k=0;k<N;k++) if(h[k]!=expect){bad=1;break;}
    int anybad=0; MPI_Allreduce(&bad,&anybad,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    if(rank==0) printf("ranks=%d nodes=%d allreduce=%.0f expected=%.0f bad_ranks=%d\n",
                       size,nnodes,(double)h[0],(double)expect,anybad);
    ncclCommDestroy(comm);
    MPI_Finalize();
    return anybad?4:0;
}
C
echo "=== image: ${IMAGE}"
echo "=== nodes: ${SLURM_JOB_NODELIST}"

echo "=== compile (one node, shared filesystem) ==="
srun -N1 -n1 -c24 --overlap --environment="${EDF}" bash -c "
  set -eux
  mpicc -O0 -o ${WORK}/mn_mpi ${WORK}/mn_mpi.c -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -L/opt/rocm/lib -lamdhip64
  mpicc -O0 -o ${WORK}/mn_rccl ${WORK}/mn_rccl.c -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -L/opt/rocm/lib -lrccl -lamdhip64
"

rc_mpi=0; rc_rccl=0; verdict=0
echo
echo "=== 1. CROSS-NODE MPI (2 nodes x 4 ranks) ==="
# --mpi=pmi2: MPICH 4.2+ builds both PMI-1 and PMI-2 client interfaces, and pmi2 is what makes
# srun launch ranks that find each other ACROSS nodes. Without it they come up isolated.
# The capability-sets debug and libfabric's own core log are the only place the SELECTED provider
# is named -- the program can report the .so it mapped but not what MPICH chose inside it. Passed
# as an env prefix rather than --export because that list splits on commas (see the EDF above).
MPIR_CVAR_CH4_OFI_CAPABILITY_SETS_DEBUG=1 FI_LOG_LEVEL=warn FI_LOG_PROV=core \
srun -N2 --ntasks-per-node=4 -c24 --mpi=pmi2 --environment="${EDF}" \
     "${WORK}/mn_mpi" 2>&1 | tee "${WORK}/mpi.log" || rc_mpi=$?

echo
echo "=== 2. WHICH libfabric did that process map, and WHICH provider did it choose? ==="
fab="$(grep -m1 -oE 'libfabric=[^ ]+' "${WORK}/mpi.log" | cut -d= -f2- || true)"
case "${fab}" in
    ""|none)
        echo "  INCONCL: the MPI process mapped no libfabric.so at all" ;;
    /opt/ofi-sdk/*|/opt/spack-install/*)
        echo "  FAIL: mapped ${fab} -- the compile-only stub, which ships no cxi provider" >&2
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
echo "=== 3. CROSS-NODE RCCL (2 nodes x 4 ranks, net transport forced into the log) ==="
srun -N2 --ntasks-per-node=4 -c24 --mpi=pmi2 --environment="${EDF}" \
     "${WORK}/mn_rccl" 2>&1 | tee "${WORK}/rccl.log" || rc_rccl=$?

echo
echo "=== 4. which NET transport did RCCL choose? ==="
net="$(grep -oiE '(Using network [A-Za-z ]+|NET/[A-Za-z]+)' "${WORK}/rccl.log" | sort -u | tr '\n' ' ' || true)"
if grep -qiE 'NET/OFI|AWS Libfabric' "${WORK}/rccl.log"; then
    echo "  OFI/libfabric plugin SELECTED: ${net}"
elif grep -qiE 'NET/Socket|Using network Socket' "${WORK}/rccl.log"; then
    echo "  TCP SOCKETS: ${net}"
    echo "  cross-node RCCL is NOT on Slingshot -- the net plugin is missing or not selected" >&2
    verdict=1
else
    echo "  INCONCLUSIVE: no transport line in the RCCL log (${net:-none})"
    tail -5 "${WORK}/rccl.log" || true
fi

echo
echo "=== cross-node MPI rc=${rc_mpi}  cross-node RCCL rc=${rc_rccl}  fabric/transport verdict=${verdict}"
s=$(( rc_mpi + rc_rccl + verdict ))

# A FAILING run is the one whose logs are wanted, and the old unconditional rm deleted them.
for f in "${WORK}"/*.log; do
    if [ -e "${f}" ]; then cp -f "${f}" "${SCRATCH}/ce-images/logs/mpi-mn-${SLURM_JOB_ID}-$(basename "${f}")"; fi
done
if [ "${s}" -eq 0 ]; then rm -rf "${WORK}"; fi

# Explicit, because `exit` truncates mod 256: `exit $(( 255 + 1 ))` reported SUCCESS.
[ "${s}" -eq 0 ] || exit 1
exit 0
