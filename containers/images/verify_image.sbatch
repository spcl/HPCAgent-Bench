#!/bin/bash
#SBATCH --job-name=verify-image
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
#SBATCH --no-requeue
# Verify one image INSIDE itself, mounted exactly as production mounts it, and fail the job on any
# required failure. The one verification entry: build_and_verify.sbatch runs this body in its own
# allocation for every role it built.
#
#   ROLE=judge sbatch containers/images/verify_image.sbatch                 # the role's candidate
#   ROLE=judge IMAGE=$SCRATCH/ce-images/<file>.sqsh sbatch ...              # any squashfs as that role
#   ROLE=judge-agent-amd sbatch --partition=mi200 --gpus-per-node=8 ...     # a portable AMD image on
#                                                                             # the other partition
#
# ROLE is an images.env role: it names the verify_image.py profile, the production EDF template, the
# platform and the partition. The EDF is rendered from that template (build_common.sh
# ce_render_edf, as install_edfs.sh renders ~/.edf) into a temporary file, never into ~/.edf: a
# checked-in EDF pointing at a candidate is how a candidate goes live by accident. Judge and agent
# roles also get the checkout's agent/ tree bound where run_cluster.sh binds it (images carry no
# tool scripts). Checks: verify_image.py (required libraries), selfcontained_check.py (nothing
# resolves outside the image), the tools launch check (judge/agent roles), on AMD the GPU-arch
# agreement and a multi-rank GPU-aware MPI/RCCL/PETSc run, on GH200 a device probe (judge/agent
# roles); sglang's kernel launch check is reported, not counted.
set -euo pipefail
ulimit -c 0
: "${SCRATCH:?set SCRATCH}"
ROLE="${ROLE:?set ROLE to an images.env role}"
# The checkout whose checks and tools are verified; build_and_verify.sbatch passes its own.
REPO="${REPO:-${SLURM_SUBMIT_DIR:?submit from the checkout root or set REPO}}"
# shellcheck source=build_common.sh
source "${REPO}/containers/images/build_common.sh"
# HPCAGENT_BENCH_DATA_ROOTS for selfcontained_check.py: ce_render_edf sources this too, but in a pipeline
# subshell whose exports never reach the srun below (judge-cpu 655841 failed verification on it).
# shellcheck source=../../scripts/cache_env.sh
source "${REPO}/scripts/cache_env.sh"
IMAGE="${IMAGE:-${CE_IMAGES:?}/$(ce_image "${ROLE}" candidate)}"
test -f "${IMAGE}"
PROFILE="$(ce_image "${ROLE}" profile)"
PLATFORM="$(ce_image "${ROLE}" platform)"
PARTITION="${SLURM_JOB_PARTITION:-$(ce_image "${ROLE}" partition || echo -)}"
preload=""
[[ ",$(ce_image "${ROLE}" flags || true)," =~ ,preload=([^,]+), ]] && preload="${BASH_REMATCH[1]}"
case "${PROFILE}" in
    judge*|*agent*) agent_mount=("${REPO}/agent:/opt/hpcagent-bench-agent") ;;
    *)              agent_mount=() ;;
esac

EDF="${SCRATCH}/.tmp/verify-${SLURM_JOB_ID:-$$}-${ROLE}.toml"
mkdir -p "$(dirname "${EDF}")"
# Probes run from `/`, so a `dace` directory in a CWD cannot shadow the image's own.
ce_render_edf "$(ce_image "${ROLE}" template)" "${IMAGE}" "${PARTITION}" "${preload}" "${agent_mount[@]}" \
    | sed -e 's|^workdir = .*|workdir = "/"|' > "${EDF}"
trap 'rm -f "${EDF}"' EXIT

# Partition table, image stamp and rocminfo must agree before anything runs, or device code built for
# another arch imports fine and fails only at first launch.
if [[ "${PLATFORM}" == amd ]]; then
    bash "${REPO}/containers/images/gpu_arch_check.sh" "${EDF}"
fi

echo "=== verifying ${IMAGE} as role=${ROLE} profile=${PROFILE} ==="
rc=0
srun --environment="${EDF}" python3 "${REPO}/containers/images/verify_image.py" --profile "${PROFILE}" --verbose \
    || rc=$?

# An image reaching a host directory for a library it "has" is not reproducible.
echo "=== self-containment: nothing may resolve outside the image ==="
case "${PROFILE}" in
    sglang)  sc_modules="numpy,torch,sglang,flydsl,triton,cupy" ;;
    vllm*)   sc_modules="numpy,torch,vllm,triton" ;;
    *)       sc_modules="" ;;   # the checker's own default list
esac
sc=0
srun --environment="${EDF}" python3 -P "${REPO}/containers/images/selfcontained_check.py" \
    ${sc_modules:+--modules "${sc_modules}"} || sc=$?

# MPI presence is not capability: running it catches a wrapper/launcher mismatch, and proves the
# device-pointer path, the RCCL net plugin and PETSc's HIP support.
mpi_rc=0
if [[ "${PLATFORM}" == amd && ${#agent_mount[@]} -gt 0 ]]; then
    echo "=== MPI: multi-rank, GPU-aware, RCCL plugin, PETSc GPU ==="
    srun --environment="${EDF}" "${REPO}/containers/images/mpi_gpu_check.sh" || mpi_rc=$?
fi

# What a driverless build cannot show on GH200: devices visible to cupy/jax/torch and PAPI's cuda
# component enumerating events.
if [[ "${PLATFORM}" == gh200 && ${#agent_mount[@]} -gt 0 ]]; then
    echo "=== GPU probe: cupy/jax/torch devices, PAPI cuda events, Nsight ==="
    srun --ntasks=1 --environment="${EDF}" bash -c '
        set -eu
        ulimit -c 0
        nvidia-smi -L
        python3 -c "import cupy; n = cupy.cuda.runtime.getDeviceCount(); print(\"cupy devices\", n); assert n > 0"
        python3 -c "import jax; d = jax.devices(); print(\"jax devices\", d); assert d[0].platform == \"gpu\""
        python3 -c "import torch; assert torch.cuda.is_available(); print(\"torch\", torch.cuda.get_device_name(0))"
        papi_native_avail 2>/dev/null | grep -c "cuda:::" || { echo "PAPI cuda component enumerated no events" >&2; exit 1; }
        ncu --version | head -1
        nsys --version | head -1
    ' || mpi_rc=$((mpi_rc + 1))
fi

# Tools load at launch, so the image must import the bound checkout's tools with its own deps.
tools_rc=0
if [[ ${#agent_mount[@]} -gt 0 ]]; then
    echo "=== tools: the checkout's agent and judge tools load in this image ==="
    srun --environment="${EDF}" python3 "${REPO}/containers/images/tools_launch_check.py" \
        --agent-dir /opt/hpcagent-bench-agent \
        --judge-web-search "${REPO}/hpcagent_bench/harness/judge_web_search.py" || tools_rc=$?
fi

# An import loads device code without launching it: this launches sgl_kernel/triton ops on the GPU.
if [[ "${PROFILE}" == sglang ]]; then
    launch_rc=0
    srun --environment="${EDF}" python3 "${REPO}/containers/inference/sglang_kernel_launch_check.py" \
        || launch_rc=$?
    echo "=== sglang: launch check rc=${launch_rc} reported, not counted ==="
fi

echo "=== required failures: ${rc}; resolving outside the image: ${sc}; mpi/gpu: ${mpi_rc}; tools: ${tools_rc} ==="
exit $(( rc + sc + mpi_rc + tools_rc ))
