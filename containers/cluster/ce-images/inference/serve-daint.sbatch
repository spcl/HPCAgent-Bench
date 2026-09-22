#!/usr/bin/env bash
# Serve one campaign model with vLLM on Daint GH200 nodes (4x H100 96 GB each), from the vllm-cuda
# image. The beverin serving path (experiments/run_cluster.sh, serve-private.sbatch) is AMD-shaped;
# this is the GH200 one, and it keeps the three things the agent side keys on unchanged:
#   * the served model name, hpcagent-bench-vllm (VLLM_SERVED_MODEL in experiments/layers/model-*.env)
#   * the served window, as --max-model-len (agent_driver.served_context reads --max-model-len or
#     --context-length, so claude-code compacts against the same window as on beverin)
#   * the tool-call and reasoning parsers, the same names as the beverin configs
#
#   MODEL     nodes  TP x PP  window  parsers (tool / reasoning)   why that width
#   qwen38    1      4 x 1    262144  qwen3_coder / qwen3          27B FP8, ~28 GB: one node, KV to spare
#   oss120b   1      4 x 1    131072  openai / openai_gptoss       MXFP4, ~65 GB
#   kimi      2+     4 x N    262144  kimi_k2 / kimi_k2            ~595 GB INT4 weights > one node's 384 GB
#
# kimi: PP is the node count. Two nodes hold the weights with ~13 GB per GPU left for KV and graphs;
# four (the beverin width) leave ~50 GB per GPU, which is what concurrent agent streams at a 262144
# window need. MIN_NODES refuses fewer than two.
#
# MODE=serve (default): wait for /health, print the endpoint, write endpoint.json for
#   alps-endpoint.sh, and hold the server until the job ends. Binds this node's hsn0 address and
#   REQUIRES a key: KEY_FILE (default ~/.config/hpcagent-bench/daint-endpoint.key), mode 600, yours.
#   The key reaches vLLM as VLLM_API_KEY in the server's environment, never on argv.
# MODE=smoke: bind 127.0.0.1 (no key), wait for /health, run verify-tools-reasoning.py and
#   accuracy-gate.py, check a multi-node serve's NCCL transport, stop. Exit 0 only if all pass.
#
#   umask 077; mkdir -p ~/.config/hpcagent-bench; openssl rand -hex 32 > ~/.config/hpcagent-bench/daint-endpoint.key
#   export SBATCH_ACCOUNT=<project> SBATCH_PARTITION=normal
#   MODEL=qwen38 MODE=smoke sbatch -N 1 --time=01:00:00 containers/cluster/ce-images/inference/serve-daint.sbatch
#   MODEL=oss120b           sbatch -N 1 --time=12:00:00 containers/cluster/ce-images/inference/serve-daint.sbatch
#   MODEL=kimi              sbatch -N 2 --time=12:00:00 containers/cluster/ce-images/inference/serve-daint.sbatch
#   MODEL=kimi DRY_RUN=1 SLURM_JOB_NUM_NODES=2 bash containers/cluster/ce-images/inference/serve-daint.sbatch
#
# Submit from inside a checkout: the chat template and the gates are read from it.
#SBATCH --job-name=serve-daint
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --no-requeue

set -Eeuo pipefail
ulimit -c 0

refuse() {
    printf 'serve-daint: refusing: %s\n' "$*" >&2
    exit 2
}

MODE="${MODE:-serve}"
DRY_RUN="${DRY_RUN:-0}"
[[ "${MODE}" == serve || "${MODE}" == smoke ]] || refuse "MODE must be serve or smoke, got '${MODE}'"
[[ -n "${SLURM_JOB_ID:-}" || "${DRY_RUN}" == 1 ]] || refuse "submit with sbatch, or DRY_RUN=1 to print the command"

# Under sbatch BASH_SOURCE is the spooled copy; the checkout is where the job was submitted from.
REPO="${REPO:-$(git -C "${SLURM_SUBMIT_DIR:-${PWD}}" rev-parse --show-toplevel 2>/dev/null || true)}"
INFERENCE="${REPO}/containers/cluster/ce-images/inference"
[[ -n "${REPO}" && -f "${INFERENCE}/verify-tools-reasoning.py" ]] || refuse "no checkout: set REPO or submit from inside one"
. "${REPO}/scripts/cache_env.sh"

use_model() {  # sets MODEL_REPO, CONTEXT, MIN_NODES, EFFORT and MODEL_FLAGS
    case "${MODEL:-}" in
        qwen38)
            MODEL_REPO=Qwen/Qwen3.8-27B-FP8 CONTEXT=262144 MIN_NODES=1 EFFORT=xhigh
            MODEL_FLAGS=(--chat-template "${REPO}/experiments/chat-template-qwen38.jinja" --trust-remote-code
                --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3)
            ;;
        oss120b)
            MODEL_REPO=openai/gpt-oss-120b CONTEXT=131072 MIN_NODES=1 EFFORT=high
            MODEL_FLAGS=(--dtype bfloat16 --load-format safetensors --safetensors-load-strategy prefetch
                --enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss)
            ;;
        kimi)
            # No reasoning_effort: Moonshot documents it as a K3-only field (.env.base-kimi27sglang).
            MODEL_REPO=moonshotai/Kimi-K2.7-Code CONTEXT=262144 MIN_NODES=2 EFFORT=""
            MODEL_FLAGS=(--trust-remote-code
                --enable-auto-tool-choice --tool-call-parser kimi_k2 --reasoning-parser kimi_k2)
            ;;
        *) refuse "MODEL must be qwen38, oss120b or kimi, got '${MODEL:-}'" ;;
    esac
}
use_model

EDF="${EDF:-hpcagent-bench-vllm-gh200-latest}"
MODEL_REPO="${MODEL_REPO_OVERRIDE:-${MODEL_REPO}}"
SERVED_MODEL="${SERVED_MODEL:-hpcagent-bench-vllm}"
API_PORT="${API_PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29500}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
KEY_FILE="${KEY_FILE:-${HOME}/.config/hpcagent-bench/daint-endpoint.key}"
# Operator words appended to the command (e.g. "--kv-cache-dtype fp8"); never host, port or key.
EXTRA_ARGS="${EXTRA_ARGS:-}"
NNODES="${SLURM_JOB_NUM_NODES:-1}"
NODE="${SLURMD_NODENAME:-${HOSTNAME%%.*}}"
RUN_DIR="${SCRATCH:?set SCRATCH}/inference-server/daint-${MODEL}/${SLURM_JOB_ID:-dry-$$}"
SERVER_PID=""

((NNODES >= MIN_NODES)) || refuse "${MODEL} needs at least ${MIN_NODES} node(s), the job has ${NNODES}"
for word in ${EXTRA_ARGS}; do
    case "${word}" in --host* | --port* | --api-key* | --headless* | --node-rank*) refuse "EXTRA_ARGS may not set ${word%%=*}" ;; esac
done
[[ "${DRY_RUN}" == 1 || -d "${HF_HOME}/hub/models--${MODEL_REPO//\//--}" ]] \
    || refuse "no ${MODEL_REPO} under ${HF_HOME}/hub: run inference/fetch_weights.sbatch first"

# THIS node is rank 0: the API server binds here and the other ranks rendezvous here, over hsn0.
HSN_ADDR="$(ip -4 -o addr show dev hsn0 2>/dev/null | awk '{ split($4, a, "/"); print a[1]; exit }')"
if [[ "${MODE}" == smoke ]]; then
    BIND_HOST=127.0.0.1
else
    BIND_HOST="${HSN_ADDR}"
    [[ "${DRY_RUN}" == 1 || "${BIND_HOST}" =~ ^[0-9]+(\.[0-9]+){3}$ ]] || refuse "${NODE} has no hsn0 IPv4 address"
fi
if ((NNODES > 1)); then
    [[ "${DRY_RUN}" == 1 || -n "${HSN_ADDR}" ]] || refuse "a multi-node serve rendezvouses over hsn0, and ${NODE} has none"
fi

COMMON=(vllm serve "${MODEL_REPO}"
    --served-model-name "${SERVED_MODEL}"
    --tensor-parallel-size "${GPUS_PER_NODE}"
    --max-model-len "${CONTEXT}"
    --max-num-seqs 128
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --generation-config auto
    "${MODEL_FLAGS[@]}")
# shellcheck disable=SC2206 # EXTRA_ARGS is an operator word list
COMMON+=(${EXTRA_ARGS})
if ((NNODES > 1)); then
    # Same shape as run_cluster.sh's pipeline branch: one engine across the nodes, mp executor,
    # rank 0 serves HTTP and every other rank is --headless. Async scheduling stays off unless asked
    # (run_cluster.sh VLLM_ASYNC_SCHEDULING): the pp.device_group broadcast it adds is what collided
    # at bootstrap on beverin.
    COMMON+=(--pipeline-parallel-size "${NNODES}" --distributed-executor-backend mp
        --nnodes "${NNODES}" --master-addr "${HSN_ADDR}" --master-port "${MASTER_PORT}"
        --distributed-timeout-seconds 3600)
    [[ "${VLLM_ASYNC_SCHEDULING:-0}" == 1 ]] || COMMON+=(--no-async-scheduling)
fi

cat <<EOF
job:      ${SLURM_JOB_ID:-dry run} on ${NODE} (${NNODES} node(s)), mode ${MODE}
model:    ${MODEL_REPO} as "${SERVED_MODEL}", window ${CONTEXT}, TP ${GPUS_PER_NODE} x PP ${NNODES}
image:    ${EDF}
bind:     ${BIND_HOST:-<hsn0>}:${API_PORT}
run dir:  ${RUN_DIR}
EOF
printf 'argv: %s --host %s --port %s\n' "${COMMON[*]}" "${BIND_HOST:-<hsn0>}" "${API_PORT}"
[[ "${DRY_RUN}" != 1 ]] || exit 0

mkdir -p "${RUN_DIR}"
chmod 700 "${RUN_DIR}"
# One launcher for every rank; each picks its role from SLURM_NODEID inside the step.
{
    printf '#!/usr/bin/env bash\nset -euo pipefail\nulimit -c 0\n'
    printf 'argv=(%s)\n' "$(printf '%q ' "${COMMON[@]}")"
    if ((NNODES > 1)); then
        printf 'argv+=(--node-rank "${SLURM_NODEID}")\n'
        printf 'if [[ "${SLURM_NODEID}" != 0 ]]; then exec "${argv[@]}" --headless; fi\n'
    fi
    printf 'exec "${argv[@]}" --host %q --port %q\n' "${BIND_HOST}" "${API_PORT}"
} > "${RUN_DIR}/launch.sh"
chmod 700 "${RUN_DIR}/launch.sh"

SERVER_ENV=(HF_HUB_OFFLINE=1 "HF_HOME=${HF_HOME}" "TRITON_CACHE_DIR=${FAST_SCRATCH}/triton-cache-vllm-gh200"
    "VLLM_CACHE_ROOT=${FAST_SCRATCH}/vllm-cache-gh200")
# A multi-node serve names the libfabric net plugin, so a plugin that does not load fails NCCL init
# loudly instead of falling back to TCP sockets and serving slowly.
((NNODES == 1)) || SERVER_ENV+=("NCCL_NET=AWS Libfabric")
if [[ "${MODE}" == serve ]]; then
    [[ -f "${KEY_FILE}" && "$(stat -c %a -- "${KEY_FILE}")" == 600 && "$(stat -c %u -- "${KEY_FILE}")" == "$(id -u)" ]] \
        || refuse "MODE=serve needs KEY_FILE ${KEY_FILE}: yours, mode 600 (see the header)"
    SERVER_ENV+=("VLLM_API_KEY=$(tr -d '\n' < "${KEY_FILE}")")
fi

stop_server() {
    [[ -n "${SERVER_PID}" ]] || return 0
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
}
cleanup() {
    rm -f -- "${RUN_DIR}/endpoint.json"
    stop_server
}
trap cleanup EXIT
trap 'exit 143' TERM INT

env "${SERVER_ENV[@]}" srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 --export=ALL \
    --environment="${EDF}" --output="${RUN_DIR}/server.log" bash "${RUN_DIR}/launch.sh" &
SERVER_PID=$!

deadline=$(($(date +%s) + READY_TIMEOUT))
until [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://${BIND_HOST}:${API_PORT}/health" || true)" == 200 ]]; do
    kill -0 "${SERVER_PID}" 2>/dev/null || { tail -40 "${RUN_DIR}/server.log" >&2; refuse "the server exited before /health answered"; }
    (($(date +%s) < deadline)) || refuse "not ready within ${READY_TIMEOUT}s; see ${RUN_DIR}/server.log"
    sleep 15
done
echo "ready: http://${BIND_HOST}:${API_PORT}/v1"

# Which NCCL transport a multi-node engine chose: NET/OFI over CXI, or NET/Socket, the silent fallback.
transport_ok=1
if ((NNODES > 1)); then
    grep -aE 'NET/(OFI|Socket|Plugin)|Using network' "${RUN_DIR}/server.log" | sort | uniq -c | head -20 || true
    grep -aq 'NET/Socket' "${RUN_DIR}/server.log" && transport_ok=0
    grep -aqE 'NET/OFI|Using network AWS Libfabric' "${RUN_DIR}/server.log" || transport_ok=0
    ((transport_ok)) || echo "NCCL is NOT on libfabric/CXI across nodes: check com.hooks.aws_ofi_nccl.variant" >&2
fi

if [[ "${MODE}" == smoke ]]; then
    failed=0
    gate() { srun --overlap --nodes=1 --ntasks=1 --environment="${EDF}" python3 "$@" || failed=$((failed + 1)); }
    gate "${INFERENCE}/verify-tools-reasoning.py" --base "http://${BIND_HOST}:${API_PORT}" --model "${SERVED_MODEL}" \
        --reasoning-effort "${EFFORT}"
    gate "${INFERENCE}/accuracy-gate.py" --base "http://${BIND_HOST}:${API_PORT}" --model "${SERVED_MODEL}"
    ((transport_ok)) || failed=$((failed + 1))
    ((failed == 0)) || refuse "SMOKE FAILED (${failed} gate(s)); logs in ${RUN_DIR}"
    echo "SMOKE PASSED: ${MODEL} on ${NNODES} node(s); logs in ${RUN_DIR}"
    exit 0
fi

(umask 077 && printf '{"url": "%s", "served_model": "%s", "key_file": "%s", "node": "%s", "job_id": "%s"}\n' \
    "http://${BIND_HOST}:${API_PORT}/v1" "${SERVED_MODEL}" "${KEY_FILE}" "${NODE}" "${SLURM_JOB_ID}" \
    > "${RUN_DIR}/endpoint.json")
cat <<EOF

===== ${MODEL} is served: http://${BIND_HOST}:${API_PORT}/v1 as "${SERVED_MODEL}", job ${SLURM_JOB_ID} =====
From another job on Daint, before the agent starts:
  source ${INFERENCE}/alps-endpoint.sh ${RUN_DIR}/endpoint.json
Stop: scancel ${SLURM_JOB_ID}. endpoint.json is deleted when this job ends.
EOF
wait "${SERVER_PID}"
