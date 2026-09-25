#!/bin/bash
#SBATCH --job-name=verify-image
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
#SBATCH --no-requeue
# Run verify_image.py INSIDE a built image and fail the job when a required library is missing.
#
#   IMAGE=$SCRATCH/ce-images/hpcagent-bench-judge-agent-amd.sqsh PROFILE=judge-agent-amd sbatch \
#          containers/images/verify_image.sbatch
#
# A standalone submission runs on SBATCH_PARTITION (site layer); build_and_verify.sbatch runs this
# body inside its own allocation, on the build job's partition. mi200 example:
#
#   REPO=$PWD IMAGE=$SCRATCH/ce-images/hpcagent-bench-sglang-mi200-candidate.sqsh PROFILE=sglang-mi200 \
#     sbatch --partition=mi200 --gpus-per-node=8 containers/images/verify_image.sbatch
#
# EDF generated here, not kept in ~/.edf: this runs against a CANDIDATE image no campaign mounts
# yet, and a checked-in EDF pointing at one is how a candidate becomes live by accident.
set -euo pipefail

ulimit -c 0
S="${SCRATCH:?set SCRATCH}"
IMAGE="${IMAGE:?set IMAGE to the .sqsh to verify}"
PROFILE="${PROFILE:-judge-agent-amd}"
# The checkout whose checks and tools are verified; build_and_verify.sbatch passes its own.
REPO="${REPO:-${S}/hpcagent-bench}"
test -f "${IMAGE}"

# hpcagent_bench_edf_mounts below: the bind mounts, derived from SCRATCH/FAST_SCRATCH.
. "${REPO}/scripts/cache_env.sh"

EDF="${S}/.tmp/verify-${SLURM_JOB_ID}.toml"
mkdir -p "$(dirname "${EDF}")"
# PATH declared, not inherited: the CE does not preserve the image's own ENV reliably, and
# verify_image.py probes with sys.executable, so a wrong python3 reports real modules as missing.
# Per profile, not one combined PATH: each image has exactly one interpreter that matters, and
# naming both prefixes would probe the wrong venv first.
case "${PROFILE}" in
    sglang|sglang-mi200) ce_path="/opt/venv/bin" ;;
    vllm)                ce_path="/opt/pytorch211/bin" ;;
    *)                   ce_path="/opt/gcc/bin:/opt/view/bin:/opt/papi/bin:/opt/venv/bin" ;;
esac
# FI_PROVIDER on inference profiles only: setting it globally aborts MPI_Init on images that also
# run MPI, so judge-agent EDFs leave it unset. This generated EDF must match production.
case "${PROFILE}" in
    sglang|sglang-mi200|vllm) ce_fi_provider='FI_PROVIDER = "cxi"' ;;
    *)                        ce_fi_provider='' ;;
esac
# CC/CXX/FC and LD_LIBRARY_PATH carried over from judge-agent-amd/edf.toml.example, the production
# template, so this generated EDF describes the same running image. LD_PRELOAD and PMIX_MCA_psec
# are left out: neither affects the compile/link legs mpi_gpu_check.sh exercises.
case "${PROFILE}" in
    judge-agent-amd|judge)
        ce_toolchain_env='CC = "/opt/gcc/bin/gcc"
CXX = "/opt/gcc/bin/g++"
FC = "/opt/gcc/bin/gfortran"
LD_LIBRARY_PATH = "/opt/rocm/lib:/opt/rocm/lib64:/opt/gcc/lib64:/opt/gcc/lib:/opt/view/lib:/opt/view/lib64:/opt/papi/lib:/usr/lib/x86_64-linux-gnu"'
        ;;
    *) ce_toolchain_env='' ;;
esac
# sglang-mi200 serves with aiter off, as its EDF template does.
case "${PROFILE}" in
    sglang-mi200) ce_aiter='SGLANG_USE_AITER = "0"' ;;
    *)            ce_aiter='' ;;
esac
# The image carries no tool scripts: judge-agent roles get the checkout's agent tree bound where
# run_cluster.sh binds it.
case "${PROFILE}" in
    judge|judge-agent-*) printf -v ce_agent_mount ', "%s"' "${REPO}/containers/agent:/opt/hpcagent-bench-agent" ;;
    *)                   ce_agent_mount='' ;;
esac

# All five hook annotations are required: the images ship no libfabric/libcxi/RCCL net plugin of
# their own, so libmpi.so resolves only against the netstack artifact these hooks install.
cat > "${EDF}" <<TOML
image = "${IMAGE}"
mounts = [$(hpcagent_bench_edf_mounts)${ce_agent_mount}]
workdir = "/"

[annotations]
# Matches judge-agent-amd/edf.toml.example. "host" mode does not work here: libcxi wants versioned
# libnl_3 symbols the base image's libnl-3.so.200 lacks, and host+rocm6 needs libamdhip64.so.6
# which this image (ROCm 7.2) does not ship. "artifact" carries a matching, self-consistent bundle.
# version/name are pinned so a campaign never silently straddles two fabric stacks.
com.hooks.netstack.source = "artifact"
com.hooks.netstack.version = "26.08.1"
com.hooks.netstack.name = "gpu_rocm7-cxi_13.1.0-ofi_2.6.0-aws_1.20.0"
com.hooks.cxi.enabled = "true"
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "rocm6"
# Required in host mode (unset is a hard error in the hook); ignored in artifact mode. Verify on
# mi200 before reuse: gfx90a may need a different variant.

[env]
PATH = "${ce_path}:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
HWLOC_COMPONENTS = "-opencl,-levelzero,-gl"
FI_CXI_RX_MATCH_MODE = "software"
FI_MR_CACHE_MONITOR = "disabled"
${ce_fi_provider}
${ce_aiter}
${ce_toolchain_env}
TOML

# Partition table, image stamp and rocminfo must agree before anything runs, or device code built
# for another arch imports fine and fails only at first launch.
bash "${REPO}/containers/images/gpu_arch_check.sh" "${EDF}" || { rm -f "${EDF}"; exit 2; }

echo "=== verifying ${IMAGE} as profile=${PROFILE} ==="
srun --environment="${EDF}" python3 "${REPO}/containers/images/verify_image.py" \
     --profile "${PROFILE}" --verbose
rc=$?

# An image reaching a host directory for a library it "has" is not reproducible. Same EDF as
# above, so this sees what a real job sees.
echo "=== self-containment: nothing may resolve outside the image ==="
case "${PROFILE}" in
    sglang)        sc_modules="numpy,torch,sglang,flydsl,triton,cupy" ;;
    sglang-mi200)  sc_modules="numpy,torch,sglang,sgl_kernel,triton,cupy" ;;
    vllm)          sc_modules="numpy,torch,vllm,triton" ;;
    *)             sc_modules="" ;;   # judge-agent-amd: the checker's own default list
esac
srun --environment="${EDF}" python3 \
     "${REPO}/containers/images/selfcontained_check.py" \
     ${sc_modules:+--modules "${sc_modules}"}
sc=$?

# MPI presence (mpicc on PATH, libmpi.so exists) is not capability; only running it catches a
# wrapper/launcher mismatch. Also proves the device-pointer path, RCCL net plugin and PETSc's HIP
# support. judge-agent profiles only: inference images ship no compilers.
mpi_rc=0
case "${PROFILE}" in
    judge|judge-agent-*|hpcagent-bench-ce-*)
        echo "=== MPI: multi-rank, GPU-aware, RCCL plugin, PETSc GPU ==="
        srun --environment="${EDF}" \
             "${REPO}/containers/images/mpi_gpu_check.sh" || mpi_rc=$?
        ;;
    *) echo "=== MPI check skipped for profile=${PROFILE} (no compilers in image) ==="; ;;
esac

# Tools load at launch, so the image must import the bound checkout's tools with its own deps.
tools_rc=0
case "${PROFILE}" in
    judge|judge-agent-*)
        echo "=== tools: the checkout's agent and judge tools load in this image ==="
        srun --environment="${EDF}" python3 "${REPO}/containers/images/tools_launch_check.py" \
             --agent-dir /opt/hpcagent-bench-agent --judge-web-search "${REPO}/hpcagent_bench/harness/judge_web_search.py" || tools_rc=$?
        ;;
    *) echo "=== tools check skipped for profile=${PROFILE} (no agent tools in image role) ==="; ;;
esac

# An import loads device code without launching it; this launches sgl_kernel/triton ops against
# torch on the visible GPU. Counted for sglang-mi200; reported only for sglang.
launch_rc=0
case "${PROFILE}" in
    sglang|sglang-mi200)
        echo "=== kernel launch: sgl_kernel and triton ops against torch on the visible GPU ==="
        srun --environment="${EDF}" python3 \
             "${REPO}/containers/inference/sglang_kernel_launch_check.py" || launch_rc=$?
        if [ "${PROFILE}" = sglang ]; then
            echo "=== sglang: launch check rc=${launch_rc} reported, not counted ==="
            launch_rc=0
        fi
        ;;
    *) echo "=== kernel launch check skipped for profile=${PROFILE} ==="; ;;
esac

rm -f "${EDF}"
echo "=== required-failure count: ${rc}; resolving outside the image: ${sc}; mpi: ${mpi_rc}; tools: ${tools_rc}; launch: ${launch_rc} ==="
if [ "${mpi_rc}" -ne 0 ]; then
    echo "MPI/GPU capability check FAILED -- image is not acceptable" >&2
    exit 1
fi
if [ "${launch_rc}" -ne 0 ]; then
    echo "kernel launch check FAILED -- device code does not run on this GPU" >&2
    exit 1
fi
# Any is disqualifying; the status says which dominated.
exit $(( rc + sc + tools_rc ))
