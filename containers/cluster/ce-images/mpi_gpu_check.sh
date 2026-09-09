#!/usr/bin/env bash
# Prove the image's MPI actually WORKS, at three levels, inside the container.
#
# The declarative table in verify_image.py can only ask "is mpicc on PATH" and "does libmpi.so
# exist". Both were true of the distro MPICH that started every rank as its own COMM_WORLD of
# size 1: each rank solved the whole problem, the answer verified, and nothing failed. Presence
# is not capability -- the same lesson the aiter prebuild taught, where importing a module built
# nothing while logging success. So this runs the three things that can actually be false:
#
#   1. multi-rank      mpiexec -n 4 forms ONE communicator of size 4, and an allreduce is right.
#                      A wrapper/launcher mismatch shows up here as size 1, not as an error.
#   2. gpu-aware       MPIX_GPU_query_support(MPIX_GPU_SUPPORT_HIP) says yes AND a device pointer
#                      survives a real allreduce. The query alone is a claim; the transfer is the
#                      evidence. Note the MPICH spelling: it lives in mpi.h, there is no
#                      mpi-ext.h, and the OpenMPI spelling reports a false NO on this stack.
#   3. transport       WHICH libfabric the live process mapped and WHICH provider it chose, and
#                      whether RCCL selects the OFI plugin rather than its TCP fallback. This is
#                      the leg the script used to omit, and its absence is why "MPI works" was
#                      once reported as "MPI is on Slingshot": a correct allreduce proves
#                      CORRECTNESS, never TRANSPORT. The same sum comes back over tcp, slower.
#
# All of libfabric, libcxi and librccl-net come from the pinned CSCS netstack artifact, installed
# by the enroot hooks the EDF enables. The image ships NONE of them -- a build gate fails if any
# survives -- so a missing annotation shows up here as an unresolvable libmpi.so, not as silence.
#
# Exits non-zero on the first hard failure. GPU checks degrade to SKIP with no visible device,
# so this is runnable on a build node without GPUs; it reports what it could not test. Inside a
# batch job on mi300 a missing GPU is a FAILURE, not a skip -- that is a broken EDF.
set -Eeuo pipefail

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
fail=0
skipped=()

say() { printf '%-14s %-8s %s\n' "$1" "$2" "${3:-}"; }

# ---------------------------------------------------------------- 0. which MPI is this, really
if ! command -v mpicc >/dev/null 2>&1; then
    say mpi FAIL "mpicc not on PATH"
    exit 1
fi
mpicc_path="$(command -v mpicc)"
case "${mpicc_path}" in
    /opt/view/bin/*) say wrapper OK "${mpicc_path}" ;;
    *) say wrapper FAIL "mpicc is ${mpicc_path}, not the spack MPICH in /opt/view"; fail=1 ;;
esac
if mpichversion 2>/dev/null | grep -qiE 'rocm|hip'; then
    say mpichversion OK "ROCm in configure line"
else
    say mpichversion FAIL "no ROCm/HIP in mpichversion -- this MPI is not GPU-aware"
    fail=1
fi

# ---------------------------------------------------------------- 1. multi-rank, one COMM_WORLD
cat >"${work}/world.c" <<'C'
#include <mpi.h>
#include <stdio.h>
int main(int argc, char **argv) {
    int size = 0, rank = 0, sum = 0, one = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Allreduce(&one, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) printf("size=%d allreduce=%d\n", size, sum);
    MPI_Finalize();
    return 0;
}
C
if ! mpicc -O0 -o "${work}/world" "${work}/world.c" 2>"${work}/world.log"; then
    say multirank FAIL "compile failed: $(tail -1 "${work}/world.log")"
    exit 1
fi
# -launcher fork keeps the ranks INSIDE this container: hydra's default ssh launcher would leave
# it, and on a CE node that is a hang rather than an error.
if out="$(mpiexec -launcher fork -n 4 "${work}/world" 2>"${work}/run.log")"; then
    if [[ "${out}" == *"size=4 allreduce=4"* ]]; then
        say multirank OK "${out}"
    else
        say multirank FAIL "expected 'size=4 allreduce=4', got '${out}' -- wrapper/launcher mismatch"
        fail=1
    fi
else
    say multirank FAIL "mpiexec failed: $(tail -1 "${work}/run.log")"
    fail=1
fi

# ------------------------------------------------- 1b. WHICH libfabric, and WHICH provider
# The leg this script used to be missing, and the reason "MPI works" was once reported as "MPI is
# on Slingshot". A correct allreduce proves CORRECTNESS, not TRANSPORT -- the same sum comes back
# over libfabric's tcp provider, several times slower, with every assertion above still green.
#
# Read what the LIVE process mapped, not what ldd predicts. The two disagree exactly where it
# matters: MPICH binds its libfabric by RPATH, RPATH is searched before LD_LIBRARY_PATH, and in
# 629966 that pinned it to a providerless /opt/spack-install libfabric while ldd against a
# different search order looked fine. The compile-only stub is deleted from the image so the
# loader falls through to the artifact; this is the check that proves the fall-through happened.
cat >"${work}/prov.c" <<'C'
#include <mpi.h>
#include <stdio.h>
#include <string.h>
int main(int argc, char **argv) {
    int rank = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        FILE *m = fopen("/proc/self/maps", "r");
        char line[1024];
        while (m && fgets(line, sizeof line, m)) {
            if (!strstr(line, "libfabric.so")) continue;
            char *path = strchr(line, '/');
            if (!path) continue;
            path[strcspn(path, "\n")] = 0;
            printf("libfabric=%s\n", path);
            break;
        }
        if (m) fclose(m);
    }
    MPI_Finalize();
    return 0;
}
C
if mpicc -O0 -o "${work}/prov" "${work}/prov.c" 2>"${work}/prov.log"; then
    # MPIR_CVAR_CH4_OFI_CAPABILITY_SETS_DEBUG makes MPICH print the provider it settled on.
    # FI_LOG_LEVEL is the fallback for a build with that CVAR compiled out.
    MPIR_CVAR_CH4_OFI_CAPABILITY_SETS_DEBUG=1 FI_LOG_LEVEL=warn FI_LOG_PROV=core \
        mpiexec -launcher fork -n 1 "${work}/prov" >"${work}/prov.out" 2>&1 || true
    lf="$(grep -m1 '^libfabric=' "${work}/prov.out" | cut -d= -f2- || true)"
    case "${lf}" in
        "")                    say libfabric INCONCL "MPI mapped no libfabric at all -- see ${work}/prov.out"
                               skipped+=("libfabric provenance") ;;
        /opt/ofi-sdk/*|/opt/spack-install/*)
                               say libfabric FAIL "${lf} -- the compile-only stub, NOT the artifact; it has no cxi provider"
                               fail=1 ;;
        /opt/cscs/*)           say libfabric OK "${lf} -> $(readlink -f "${lf}" 2>/dev/null || echo "${lf}")" ;;
        *)                     say libfabric INCONCL "${lf} -- not the netstack artifact and not a known stub" ;;
    esac
    # The provider is what decides Slingshot versus tcp. cxi is the Cassini provider; OFI is only
    # the API around it, so "we have OFI" is not the same claim and must not be read as one.
    prov="$(grep -oiE 'provider: *[a-z0-9_;()]+' "${work}/prov.out" | head -1 || true)"
    [[ -z "${prov}" ]] && prov="$(grep -oiE '\b(cxi|verbs|tcp|sockets|shm|psm3)\b' "${work}/prov.out" | sort -u | tr '\n' ' ' || true)"
    # A non-cxi provider is a FAILURE only where cxi was actually available. On a node with no
    # /dev/cxi* there is no Slingshot to select and failing would reject a good image for the
    # node's shape -- the exact error the three-outcome rule elsewhere in this file exists to
    # avoid. Note this is a ONE-NODE probe: it proves which provider MPI initialised, not that a
    # cross-node transfer rode it. mpi_multinode_check.sbatch is what proves the latter.
    have_cxi=0
    compgen -G '/dev/cxi*' >/dev/null 2>&1 && have_cxi=1
    case "${prov}" in
        "")            say provider INCONCL "MPICH printed no provider line; see ${work}/prov.out"
                       skipped+=("MPI provider selection") ;;
        *cxi*|*CXI*)   say provider OK "${prov}" ;;
        *)             if (( have_cxi )); then
                           say provider FAIL "${prov} -- /dev/cxi* exists but MPI did not select cxi"
                           fail=1
                       else
                           say provider INCONCL "${prov} -- no /dev/cxi* on this node, nothing to select"
                           skipped+=("MPI provider selection (no cxi device)")
                       fi ;;
    esac
    # Keep the evidence when the caller asks for it: this is the one output worth reading by hand
    # when a verdict surprises, and ${work} is deleted on exit.
    [[ -n "${MPI_CHECK_EVIDENCE:-}" ]] && cp -f "${work}/prov.out" "${MPI_CHECK_EVIDENCE}" 2>/dev/null
    true
else
    say libfabric FAIL "provenance probe did not compile: $(tail -1 "${work}/prov.log")"
    fail=1
fi

# ---------------------------------------------------------------- 2. GPU-aware, claim + evidence
have_gpu=0
if command -v rocm-smi >/dev/null 2>&1 && rocm-smi --showid >/dev/null 2>&1; then
    have_gpu=1
fi
# A SKIP here used to be reachable from a batch job, where it turned every piece of real GPU
# evidence into "not tested" and still printed PASSED. On a build node without devices that is
# correct; inside an mi300 allocation it means the EDF or the --gres is broken, so say so.
if (( ! have_gpu )) && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    say gpu FAIL "no visible GPU inside job ${SLURM_JOB_ID} on mi300 -- broken EDF or missing gres"
    fail=1
fi
cat >"${work}/gpu.c" <<'C'
#include <mpi.h>
#include <stdio.h>
#include <hip/hip_runtime.h>

#define HIP_OK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "hip error %s at %d\n", hipGetErrorString(e), __LINE__); return 2; } } while (0)

int main(int argc, char **argv) {
    int rank = 0, size = 0, host = 0;
    int *dsend = 0, *drecv = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* The MPICH spelling, declared in mpi.h. There is no mpi-ext.h on this stack, and the
       OpenMPI spelling (MPIX_Query_rocm_support) reports a false NO here. */
    if (MPIX_GPU_query_support(MPIX_GPU_SUPPORT_HIP, &host) != MPI_SUCCESS || !host) {
        if (rank == 0) fprintf(stderr, "MPIX_GPU_query_support(HIP) says NO\n");
        MPI_Finalize();
        return 3;
    }

    /* The query is a claim. Passing a DEVICE pointer through a collective is the evidence: a
       non-GPU-aware MPI either faults here or silently stages through host memory. */
    /* One device per rank where there are enough, so the allreduce crosses GPUs rather than
       looping back through one. Both ranks on device 0 would still verify and prove less. */
    int ndev = 0;
    HIP_OK(hipGetDeviceCount(&ndev));
    if (ndev < 1) { if (rank == 0) fprintf(stderr, "no HIP device visible\n"); MPI_Finalize(); return 5; }
    HIP_OK(hipSetDevice(rank % ndev));
    HIP_OK(hipMalloc((void **)&dsend, sizeof(int)));
    HIP_OK(hipMalloc((void **)&drecv, sizeof(int)));
    int one = 1, got = 0;
    HIP_OK(hipMemcpy(dsend, &one, sizeof(int), hipMemcpyHostToDevice));
    MPI_Allreduce(dsend, drecv, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    HIP_OK(hipMemcpy(&got, drecv, sizeof(int), hipMemcpyDeviceToHost));
    if (rank == 0) printf("gpu_query=yes device_allreduce=%d expected=%d\n", got, size);
    HIP_OK(hipFree(dsend));
    HIP_OK(hipFree(drecv));
    MPI_Finalize();
    return (got == size) ? 0 : 4;
}
C
if mpicc -O0 -o "${work}/gpu" "${work}/gpu.c" \
        -I/opt/rocm/include -D__HIP_PLATFORM_AMD__ -L/opt/rocm/lib -lamdhip64 \
        2>"${work}/gpu.log"; then
    say gpu-aware-cc OK "compiled against hip + mpi"
    if (( have_gpu )); then
        if out="$(mpiexec -launcher fork -n 2 "${work}/gpu" 2>"${work}/gpurun.log")"; then
            say gpu-aware OK "${out}"
        else
            say gpu-aware FAIL "device allreduce failed: $(tail -2 "${work}/gpurun.log" | tr '\n' ' ')"
            fail=1
        fi
    else
        say gpu-aware SKIP "no visible GPU on this node -- compile-only"
        skipped+=("gpu-aware runtime")
    fi
else
    say gpu-aware-cc FAIL "$(tail -1 "${work}/gpu.log")"
    fail=1
fi

# ---------------------------------------------------------------- 3. RCCL network plugin
# The plugin is the ARTIFACT's, not one this image built. The self-built copy under
# /opt/aws-ofi-nccl is gone and a build gate refuses to ship any replacement, because a shipped
# copy is found first and shadows the artifact -- whose libfabric, libcxi and libc are matched to
# each other and to the host driver in a way a graft is not. Either install name is accepted:
# RCCL looks for libnccl-net.so and the artifact may ship only the rccl spelling.
plugin=""
for cand in /opt/cscs/netstack/librccl-net.so /opt/cscs/netstack/lib/librccl-net.so \
            /opt/cscs/netstack/libnccl-net.so /opt/cscs/netstack/lib/libnccl-net.so; do
    [[ -e "${cand}" ]] && { plugin="${cand}"; break; }
done
if [[ -n "${plugin}" ]]; then
    if readelf -d "${plugin}" | grep -q 'NEEDED.*libfabric\.so\.1'; then
        say rccl-plugin OK "${plugin}"
    else
        say rccl-plugin FAIL "plugin no longer links libfabric -- it would load and do nothing"
        fail=1
    fi
    # ZERO unresolved, not "only libfabric unresolved". The artifact is self-contained: it carries
    # its own libfabric, libcxi, libcurl and libc, so anything still missing is a real defect
    # rather than the deliberate gap the old self-built plugin left for the host hook to fill.
    missing="$(ldd "${plugin}" 2>/dev/null | grep 'not found' | tr '\n' ' ' || true)"
    if [[ -n "${missing}" ]]; then
        say rccl-deps FAIL "unresolved: ${missing}"
        fail=1
    else
        say rccl-deps OK "all deps resolve against the netstack artifact"
    fi
else
    say rccl-plugin FAIL "no librccl-net.so under /opt/cscs/netstack -- the aws_ofi_nccl hook did not run; check com.hooks.aws_ofi_nccl.enabled in the EDF"
    fail=1
fi

# ------------------------------------------------- 3b. RCCL actually RUNS a collective
# Loadable is not working. This runs ncclAllReduce over every visible device and checks the
# NUMBERS, because a collective that silently returns the input is the failure that reads as
# success -- the same shape as the agent-written cupy whose timer returned 0.0 and voided a
# campaign's GPU numbers.
cat >"${work}/rccl.c" <<'C'
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define HIP_OK(c) do { hipError_t e=(c); if(e!=hipSuccess){ \
    fprintf(stderr,"hip %s at %d\n",hipGetErrorString(e),__LINE__); return 2;} } while(0)
#define NCCL_OK(c) do { ncclResult_t r=(c); if(r!=ncclSuccess){ \
    fprintf(stderr,"rccl %s at %d\n",ncclGetErrorString(r),__LINE__); return 3;} } while(0)

int main(void) {
    int ndev = 0;
    HIP_OK(hipGetDeviceCount(&ndev));
    if (ndev < 2) { printf("only %d device(s); need 2 for a collective\n", ndev); return 9; }
    if (ndev > 8) ndev = 8;

    int *devs = malloc(ndev * sizeof(int));
    float **sbuf = malloc(ndev * sizeof(float *)), **rbuf = malloc(ndev * sizeof(float *));
    ncclComm_t *comms = malloc(ndev * sizeof(ncclComm_t));
    hipStream_t *st = malloc(ndev * sizeof(hipStream_t));

    /* Rank r contributes r+1 everywhere, so the sum is ndev(ndev+1)/2 -- a value no single
       rank holds. A no-op allreduce that hands back the input cannot produce it. */
    for (int i = 0; i < ndev; i++) {
        devs[i] = i;
        HIP_OK(hipSetDevice(i));
        /* void** casts: hipMalloc takes void**, and C does not implicitly convert float**. */
        HIP_OK(hipMalloc((void **)&sbuf[i], N * sizeof(float)));
        HIP_OK(hipMalloc((void **)&rbuf[i], N * sizeof(float)));
        HIP_OK(hipStreamCreate(&st[i]));
        float *h = malloc(N * sizeof(float));
        for (int k = 0; k < N; k++) h[k] = (float)(i + 1);
        HIP_OK(hipMemcpy(sbuf[i], h, N * sizeof(float), hipMemcpyHostToDevice));
        HIP_OK(hipMemset(rbuf[i], 0, N * sizeof(float)));
        free(h);
    }

    NCCL_OK(ncclCommInitAll(comms, ndev, devs));
    NCCL_OK(ncclGroupStart());
    for (int i = 0; i < ndev; i++)
        NCCL_OK(ncclAllReduce(sbuf[i], rbuf[i], N, ncclFloat, ncclSum, comms[i], st[i]));
    NCCL_OK(ncclGroupEnd());
    for (int i = 0; i < ndev; i++) { HIP_OK(hipSetDevice(i)); HIP_OK(hipStreamSynchronize(st[i])); }

    float expect = (float)(ndev * (ndev + 1) / 2);
    int bad = 0;
    for (int i = 0; i < ndev; i++) {
        float *h = malloc(N * sizeof(float));
        HIP_OK(hipSetDevice(i));
        HIP_OK(hipMemcpy(h, rbuf[i], N * sizeof(float), hipMemcpyDeviceToHost));
        for (int k = 0; k < N; k++) if (h[k] != expect) { bad++; break; }
        free(h);
    }
    for (int i = 0; i < ndev; i++) ncclCommDestroy(comms[i]);
    printf("devices=%d allreduce=%.0f expected=%.0f wrong_ranks=%d\n",
           ndev, expect, expect, bad);
    return bad ? 4 : 0;
}
C
# NOT hipcc. There is no device code here -- only host-side runtime and collective calls -- and
# hipcc's driver puts a .c file through `--driver-mode=g++ --hip-link -x c` and fails. The plain
# compiler with __HIP_PLATFORM_AMD__ and -lamdhip64 is the recipe the GPU-aware MPI leg above
# already uses successfully, so use the same one rather than fighting the wrapper.
if ${CC:-gcc} -O0 -o "${work}/rccl" "${work}/rccl.c" \
        -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include \
        -L/opt/rocm/lib -lrccl -lamdhip64 2>"${work}/rcclcc.log"; then
    say rccl-cc OK "compiled against rccl.h"
    if (( have_gpu )); then
        # rc captured explicitly: `$?` inside an elif reads the status of whatever ran last, which
        # is a good way to test the wrong thing. 9 is the program's own "not enough devices".
        rccl_rc=0
        out="$("${work}/rccl" 2>"${work}/rcclrun.log")" || rccl_rc=$?
        if (( rccl_rc == 0 )); then
            say rccl-run OK "${out}"
        elif (( rccl_rc == 9 )); then
            say rccl-run SKIP "fewer than 2 visible devices"
            skipped+=("rccl collective")
        else
            say rccl-run FAIL "rc=${rccl_rc} $(tail -2 "${work}/rcclrun.log" | tr '\n' ' ')"
            fail=1
        fi

        # Force the NET transport. On one node RCCL would use XGMI/IPC and never touch the net
        # plugin, so this is the only way to prove the plugin is SELECTED rather than merely
        # present -- selection is what decides whether a cross-node collective rides Slingshot
        # or the TCP fallback.
        #
        # THREE outcomes, not two. A plugin that is present and chosen prints NET/OFI; the
        # built-in fallback prints NET/Socket. But if RCCL prints NO transport line at all --
        # a debug-format change, a build with INIT tracing compiled out -- then treating that
        # as failure would REJECT A GOOD IMAGE on the strength of a missing log line. That is a
        # worse error than the one this leg exists to catch, so absence of evidence is reported
        # as INCONCL with the evidence attached, and the verdict falls back to rccl-plugin,
        # which inspects the file itself and cannot be fooled by logging.
        NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
            "${work}/rccl" >"${work}/net.log" 2>&1 || true
        net_line="$(grep -oiE '(Using network [A-Za-z ]+|NET/[A-Za-z]+)' "${work}/net.log" \
                    | sort -u | tr '\n' ' ' || true)"
        if grep -qiE 'NET/OFI|AWS Libfabric|Using network.*(OFI|Libfabric)' "${work}/net.log"; then
            say rccl-net OK "${net_line}"
        elif grep -qiE 'NET/Socket|Using network Socket' "${work}/net.log"; then
            say rccl-net FAIL "${net_line} -- built-in TCP fallback, plugin NOT selected"
            fail=1
        else
            say rccl-net INCONCL "no transport line in RCCL debug output; see rccl-plugin"
            skipped+=("rccl net transport selection")
            printf '    evidence: %s\n' "$(tail -3 "${work}/net.log" | tr '\n' ' ' | cut -c1-160)"
        fi
    else
        say rccl-run SKIP "no visible GPU on this node -- compile-only"
        skipped+=("rccl collective" "rccl net transport")
    fi
else
    say rccl-cc FAIL "$(grep -m2 -iE 'error|fatal' "${work}/rcclcc.log" | tr '\n' ' ' || tail -2 "${work}/rcclcc.log" | tr '\n' ' ')"
    fail=1
fi

# ---------------------------------------------------------------- 4. PETSc, GPU-enabled
petscconf=""
for c in /opt/view/include/petscconf.h /opt/view/lib/petsc/conf/petscvariables; do
    [[ -e "${c}" ]] && petscconf="${c}" && break
done
if [[ -n "${petscconf}" ]] && grep -q 'PETSC_HAVE_HIP' "${petscconf}" 2>/dev/null; then
    say petsc-gpu OK "PETSC_HAVE_HIP in ${petscconf}"
elif [[ -e /opt/view/lib/libpetsc.so ]]; then
    say petsc-gpu FAIL "libpetsc.so present but no PETSC_HAVE_HIP -- +rocm did not take"
    fail=1
else
    say petsc-gpu FAIL "libpetsc.so missing"
    fail=1
fi

echo
if (( ${#skipped[@]} )); then
    printf 'NOT TESTED HERE: %s\n' "${skipped[*]}"
fi
if (( fail )); then
    echo "MPI/GPU CHECK: FAILED"
    exit 1
fi
echo "MPI/GPU CHECK: PASSED"
