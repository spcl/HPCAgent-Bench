#!/usr/bin/env bash
#SBATCH --job-name=rccl-ofi-2node
#SBATCH --partition=mi300
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:40:00
#SBATCH --output=/capstor/scratch/cscs/ybudanaz/x86_64/rccl-ofi/logs/rccl-ofi-%j.out
#SBATCH --error=/capstor/scratch/cscs/ybudanaz/x86_64/rccl-ofi/logs/rccl-ofi-%j.err
#
# Gate for the aws-ofi-nccl plugin the pt211 image ships at /opt/aws-ofi-nccl. Until it was, RCCL dlopened
# librccl-net.so, missed, found no verbs device either (CXI is not verbs) and fell back to TCP
# sockets over hsn0-3 -- silently, no error, just the slow path. That fallback is what made the
# kimi pp=4 probe cross PP2->PP3 in 7m08s and blow the 300 s executor deadline.
#
# There is NO log evidence that a 2-node CXI collective has EVER passed on this system: the only
# recorded gate was single-node single-rank. So this job proves the plugin loads AND that CXI
# actually carries an inter-node collective, before any campaign depends on it.
#
#   sbatch test-rccl-ofi-2node.sbatch
#
# Stages:
#   1 preflight, one node   -- ldd -r the plugin, confirm libfabric resolves to the HOST 2.3.1
#                              the CXI hook injects, enumerate cxi0-3, dlopen it RTLD_NOW
#   2 allreduce, 2n x 1 rank -- forces every byte inter-node, no intra-node P2P to hide behind
#   3 allreduce, 2n x 4 rank -- the shape a campaign actually runs
#
# PASS requires, in both allreduce stages: `Selected provider is cxi`, `Using network AWS
# Libfabric`, NO `Using network Socket`, and correct=True. Anything less is a FAIL -- a run that
# reports correct=True over Socket is exactly the silent regression this gate exists to catch.

set -euo pipefail

# Beverin's core_pattern is the machine-global `core_%h_%p` and dumps land in the CWD.
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
RUN_DIR="${SCRATCH}/rccl-ofi/logs/${SLURM_JOB_ID}"
PLUGIN_DIR="${PLUGIN_DIR:-${SCRATCH}/ce-plugins/aws-ofi-nccl-rocm723/opt/aws-ofi-nccl}"
# The allreduce driver ships in the repo. It used to default into a build root
# ($SCRATCH/vllm-mi300-rebuild/phase1-context), which is a REBUILD work tree that gets wiped --
# 600646 died on exactly that. Slurm spools this script, so BASH_SOURCE cannot find the repo:
# pass OPTARENA_REPO (or ALLREDUCE_PY outright), the same way build-agent-image.sbatch takes
# AMD_DIR.
ALLREDUCE_PY="${ALLREDUCE_PY:-${OPTARENA_REPO:?set OPTARENA_REPO (repo root) or ALLREDUCE_PY}/containers/cluster/ce-images/inference/beverin-rocm723-host-ofi-phase1/torch-dist-allreduce-mi300.py}"
INFERENCE_EDF="${INFERENCE_EDF:-vllm-latest}"

MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
SIZE_MB="${SIZE_MB:-256}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
STAGE_TIMEOUT="${STAGE_TIMEOUT:-420}"

mkdir -p "${RUN_DIR}"

for path in "${PLUGIN_DIR}/lib/librccl-net.so" "${ALLREDUCE_PY}"; do
    [[ -f "${path}" ]] || { echo "FAIL: missing ${path}" >&2; exit 2; }
done

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if (( ${#NODES[@]} != 2 )); then
    echo "FAIL: need exactly 2 nodes, allocation has ${#NODES[@]}" >&2
    exit 2
fi

MASTER_ADDR=$(srun --nodes=1 --ntasks=1 --nodelist="${NODES[0]}" \
    bash -c "ip -4 -o addr show hsn0 | awk '{print \$4}' | cut -d/ -f1")
[[ -n "${MASTER_ADDR}" ]] || { echo "FAIL: no hsn0 address on ${NODES[0]}" >&2; exit 2; }

cat <<EOF
job:        ${SLURM_JOB_ID}
nodes:      ${NODES[*]}
edf:        ${INFERENCE_EDF}
plugin:     ${PLUGIN_DIR}
master:     ${MASTER_ADDR}:${MASTER_PORT}
run_dir:    ${RUN_DIR}
EOF

stage() { printf '\n===== stage %s =====\n' "$*"; }

# ---------------------------------------------------------------- stage 1: preflight

cat >"${RUN_DIR}/preflight.sh" <<'PREFLIGHT'
set -uo pipefail
PLUGIN="${PLUGIN_DIR}/lib/librccl-net.so"

echo "-- host: $(hostname)"
echo "-- LD_LIBRARY_PATH=${LD_LIBRARY_PATH-unset}"
echo "-- NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN-unset} NCCL_NET=${NCCL_NET-unset} FI_PROVIDER=${FI_PROVIDER-unset}"

echo "-- ldd -r ${PLUGIN}"
ldd -r "${PLUGIN}" || echo "LDD_RC=$?"

echo "-- libfabric actually bound"
# The image ships its own libfabric 1.20.0 as the anchor the CXI hook overrides with the host
# 2.3.1. The realpath is the tell: .1.29.1 is the host build and has the cxi provider, .1.20.0
# is the image's and does not. Binding the wrong one is a silent loss of the whole fast path.
FABRIC=$(ldd "${PLUGIN}" | awk '/libfabric\.so\.1/ {print $3}')
echo "resolved: ${FABRIC:-NONE} -> $(readlink -f "${FABRIC:-/nonexistent}" 2>/dev/null || echo NONE)"

echo "-- fi_info -p cxi"
"${PLUGIN_DIR}/bin/fi_info-host" -p cxi 2>&1 | grep -E 'provider|domain|fabric' | head -20 \
    || echo "FI_INFO_RC=$?"

echo "-- dlopen RTLD_NOW"
python3 -c "
import ctypes, os
ctypes.CDLL('${PLUGIN}', mode=os.RTLD_NOW)
print('dlopen OK')
" || echo "DLOPEN_FAILED"
PREFLIGHT

stage "1 preflight (${NODES[0]})"
srun --nodes=1 --ntasks=1 --nodelist="${NODES[0]}" \
    --environment="${INFERENCE_EDF}" \
    bash -c "PLUGIN_DIR='${PLUGIN_DIR}' bash '${RUN_DIR}/preflight.sh'" \
    2>&1 | tee "${RUN_DIR}/stage1-preflight.log" || true

# ---------------------------------------------------------------- stages 2 and 3: allreduce

run_allreduce() {
    local procs_per_node="$1" label="$2" log="${RUN_DIR}/${2}.log"

    stage "${label} (2 nodes x ${procs_per_node} rank(s))"
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
        --gpus-per-node=4 --cpus-per-task=32 \
        --environment="${INFERENCE_EDF}" \
        bash -c "
            timeout --signal=TERM --kill-after=15s ${STAGE_TIMEOUT}s \
            torchrun --nnodes=2 --nproc-per-node=${procs_per_node} \
                --node-rank=\${SLURM_NODEID} \
                --master-addr='${MASTER_ADDR}' --master-port='${MASTER_PORT}' \
                '${ALLREDUCE_PY}' \
                    --size-mb ${SIZE_MB} --warmup ${WARMUP} --iters ${ITERS} \
                    --timeout-seconds 180
        " 2>&1 | tee "${log}" || true

    verdict "${label}" "${log}"
}

# A stage passes only if CXI was selected AND Socket never was AND the numeric result is right.
# Each condition is reported separately: "correct but over Socket" is the failure mode that a
# single pass/fail line would hide.
verdict() {
    local label="$1" log="$2" ok=1

    printf '\n-- verdict %s\n' "${label}"
    grep -qE 'Selected provider is cxi'  "${log}" && echo "  cxi provider    PASS" || { echo "  cxi provider    FAIL"; ok=0; }
    grep -qE 'Using network AWS Libfabric' "${log}" && echo "  ofi network     PASS" || { echo "  ofi network     FAIL"; ok=0; }
    grep -qE 'Using network Socket'      "${log}" && { echo "  no tcp fallback FAIL"; ok=0; } || echo "  no tcp fallback PASS"
    grep -qE '^correct=True'             "${log}" && echo "  allreduce       PASS" || { echo "  allreduce       FAIL"; ok=0; }

    grep -E 'Selected provider is|found [0-9]+ nic|Using network|nNodes' "${log}" | sort -u | sed 's/^/  | /' | head -10
    grep -E '^(median_seconds|algorithmic_GBps|estimated_bus_GBps)=' "${log}" | sed 's/^/  | /'

    if (( ok )); then
        echo "  ${label}: PASS"
    else
        echo "  ${label}: FAIL"
        VERDICT_FAILURES+=("${label}")
    fi
}

VERDICT_FAILURES=()
run_allreduce 1 "stage2-2n1r"
run_allreduce 4 "stage3-2n4r"

stage "summary"
if (( ${#VERDICT_FAILURES[@]} == 0 )); then
    echo "RCCL_OFI_GATE: PASS -- pt211 is on CXI, no TCP fallback. Logs: ${RUN_DIR}"
    exit 0
fi
echo "RCCL_OFI_GATE: FAIL -- ${VERDICT_FAILURES[*]}. Logs: ${RUN_DIR}"
exit 1
