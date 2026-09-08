#!/usr/bin/env bash
#SBATCH --job-name=numpy-vectorize
#SBATCH --partition=mi300
#SBATCH --nodes=9
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/ybudanaz/x86_64/numpy-vectorize/logs/vectorize-%j.out
#SBATCH --error=/capstor/scratch/cscs/ybudanaz/x86_64/numpy-vectorize/logs/vectorize-%j.err

# Vectorize the scientific_computing NumPy references with 10 Kimi K2.7-Code agents.
#
# NOT a benchmark run: no judge, no results DB, no campaign arm. The agents edit the repo and
# verify with scripts/numpy_vectorize/check.py, which reuses the benchmark's own scorer to
# compare each <kernel>_better_numpy.py against the shipped <kernel>_numpy.py.
#
#   mkdir -p /capstor/scratch/cscs/ybudanaz/x86_64/numpy-vectorize/logs
#   sbatch --account=a-g200 scripts/numpy_vectorize/vectorize.sbatch          # REPLICAS=2, 9 nodes
#   REPLICAS=1 sbatch --nodes=5 --account=a-g200 scripts/numpy_vectorize/vectorize.sbatch
#
# Topology: REPLICAS independent Kimi engines at tp=4 pp=4 (the shape proven by
# test-kimi-messages-probe.sbatch), 4 nodes each, plus one agent node -- so 4*REPLICAS+1.
# 136 kernels / 10 agents is ~14 each.
#
# REPLICAS defaults to 2 because ONE engine cannot feed 10 agents. Measured on this exact
# tp=4 pp=4 shape (599301): per-request decode is 9.97 tok/s at concurrency 1, 6.06 at 8,
# 4.44 at 16, and concurrency 64 never reached steady state at all. Ten agents on one engine
# sit at ~5 tok/s each, under the >=8 tok/s an agent needs to finish its shard; striped over
# two engines they see concurrency 5 and ~7 tok/s. run_agents.py deals agent i to
# endpoint i % len(endpoints), so striping needs nothing but a second --endpoint.

set -Eeuo pipefail

# Beverin's core_pattern is machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the repo with 0-byte stubs. Slurm propagates CORE to steps.
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
export RUN_DIR="${SCRATCH}/numpy-vectorize/logs/${SLURM_JOB_ID}"
# iopsstor reads ~11x faster than capstor (job 593523); capstor keeps the durable copy.
export FAST_SCRATCH="/iopsstor/scratch/cscs/${USER}"
[[ -d "${FAST_SCRATCH}" ]] || FAST_SCRATCH="${SCRATCH}"
export HF_HOME="${HF_HOME:-${FAST_SCRATCH}/hf}"
export REPO="${HPCAGENT_BENCH_REPO:-${SCRATCH}/optarena}"

INFERENCE_EDF="${INFERENCE_EDF:-vllm-latest}"
AGENT_EDF="${AGENT_EDF:-optarena-amd-mi300-latest}"

export MODEL_REPO="${MODEL_REPO:-moonshotai/Kimi-K2.7-Code}"
export SERVED_MODEL="${SERVED_MODEL:-optarena-vllm}"
export API_PORT="${API_PORT:-8000}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
READY_TIMEOUT="${READY_TIMEOUT:-7200}"
AGENTS="${AGENTS:-10}"
REPLICAS="${REPLICAS:-2}"
AGENT_TIMEOUT_SECONDS="${AGENT_TIMEOUT_SECONDS:-28800}"

mkdir -p "${RUN_DIR}"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
NEED=$(( 4 * REPLICAS + 1 ))
if (( ${#NODES[@]} != NEED )); then
    echo "FAIL: REPLICAS=${REPLICAS} needs exactly ${NEED} nodes (4 per engine + 1 agent), " \
         "allocation has ${#NODES[@]}" >&2
    echo "submit with: sbatch --nodes=${NEED} ..." >&2
    exit 2
fi
# The agent node is the LAST one, so replica r owns NODES[4r .. 4r+3] whatever REPLICAS is.
AGENT_NODE="${NODES[${#NODES[@]} - 1]}"
export NODE0="${NODES[0]}"

SERVE_SCRIPT="${RUN_DIR}/serve.sh"
AGENT_SCRIPT="${RUN_DIR}/agents.sh"
AGENTS_DONE="${RUN_DIR}/agents.done"

stage() { printf '\n===== stage: %s =====\n' "$*"; }

cat <<EOF
job:       ${SLURM_JOB_ID}
repo:      ${REPO}
inference: ${REPLICAS} x tp=4 pp=4 (${INFERENCE_EDF}) port ${API_PORT}, heads ${NODES[0]}...
agents:    ${AGENT_NODE} (${AGENT_EDF}) x ${AGENTS}
logs:      ${RUN_DIR}
EOF

cat >"${SERVE_SCRIPT}" <<'SERVE'
#!/usr/bin/env bash
set -Eeuo pipefail

rank="${SLURM_PROCID:-0}"
export TOKENIZERS_PARALLELISM=false
export VLLM_ENGINE_READY_TIMEOUT_S=7200
# Per-step deadline for one execute_model RPC. vLLM's own default is 300 s and
# --distributed-timeout-seconds does NOT cover it, so a slow first decode kills the engine.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
# RCCL probes and ENABLES GPU-direct RDMA on its own; on this fabric that path corrupts
# cross-node collectives intermittently and produces silent wrong results rather than an error.
# Omitting the variable is NOT the same as having GDR off.
# The three things that make THIS shape fast and correct, all of which the campaign path
# (containers/cluster/example-script/run_cluster.sh) sets and this standalone job did not:
#
# 1. Tuned fused_moe Triton configs, keyed by (experts, N, device, dtype). Unset, kimi serves on
#    vLLM's default MoE config and warns it is sub-optimal -- measured ~11x off the reference for
#    the E=384,N=512,int4_w4a16 shape, and the tuning committed here is worth 1.71x.
# 2. The eager process-group patch. At pp=4 the lazy path mints a per-pair NCCL communicator over
#    CXI and the engine decodes ZERO tokens (594541-543). This is the same tp=4 pp=4 shape.
# 3. A cache root that survives the job: the default lives under ~/.cache, which is not mounted
#    here, and 592283 paid 3 h 13 m re-running Inductor codegen because of it.
export VLLM_TUNED_CONFIG_FOLDER="${VLLM_TUNED_CONFIG_FOLDER:-${REPO}/containers/cluster/ce-images/inference/moe-configs}"
export PYTHONPATH="${REPO}/containers/cluster/ce-images/inference/external-eager-pg-patch:${PYTHONPATH:-}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${FAST_SCRATCH}/vllm-cache}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_FILE="${RUN_DIR}/nccl.%h.%p.log"
# VLLM_DISABLE_PYNCCL is deliberately NOT set. test-kimi-messages-probe.sbatch sets it to 1 and
# that is the superseded workaround: without PyNCCL every collective goes through
# ProcessGroupNCCL, which is not graph-capturable, so capture stalls, --enforce-eager becomes
# mandatory, and kimi decodes at 1.4 tok/s against 16.8 on this same tp=4 pp=4 4-node shape.
# vLLM reads env VLLM_PORT as the BASE for internal ZMQ ports, not the HTTP port (589170).
unset VLLM_PORT

MODEL_PATH="$(python3 - <<'PY' | tail -n 1
import os
from huggingface_hub import snapshot_download
repo = os.environ["MODEL_REPO"]
try:
    print(snapshot_download(repo_id=repo, local_files_only=True))
except Exception:
    print(snapshot_download(repo_id=repo, max_workers=8))
PY
)"
test -d "${MODEL_PATH}"
echo "rank ${rank} host $(hostname) model ${MODEL_PATH}"

ARGS=(
    "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL}"
    --tensor-parallel-size 4
    --pipeline-parallel-size 4
    --distributed-executor-backend mp
    --nnodes 4
    --node-rank "${rank}"
    --master-addr "${NODE0}"
    --master-port "${MASTER_PORT}"
    --distributed-timeout-seconds 7200
    --load-format safetensors
    --safetensors-load-strategy prefetch
    --generation-config vllm
    --enable-auto-tool-choice
    --tool-call-parser kimi_k2
    --reasoning-parser kimi_k2
    --trust-remote-code
    --max-model-len 65536
    # 0.70, not the 0.85 the probe used: the fraction multiplies TOTAL board memory, but only
    # 96 of the MI300A's 128 GiB is HIP-allocatable, so 0.85 asks for more than exists.
    --gpu-memory-utilization 0.70
    --max-num-seqs 128
)
if (( rank == 0 )); then
    ARGS+=(--host 0.0.0.0 --port "${API_PORT}")
else
    ARGS+=(--headless)
fi

SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM

vllm serve "${ARGS[@]}" >"${RUN_DIR}/vllm-rank${rank}.log" 2>&1 &
SERVER_PID=$!
echo "rank ${rank} vllm pid ${SERVER_PID}"

while [[ ! -e "${RUN_DIR}/agents.done" ]]; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "rank ${rank}: vLLM exited before the agents finished" >&2
        wait "${SERVER_PID}" || true
        exit 1
    fi
    sleep 5
done
echo "rank ${rank}: agents done, shutting down"
SERVE

cat >"${AGENT_SCRIPT}" <<'AGENTS_SH'
#!/usr/bin/env bash
set -Eeuo pipefail
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export CLAUDE_MODEL="${SERVED_MODEL}"

python3 scripts/numpy_vectorize/run_agents.py \
    --agents "${AGENTS}" \
    ${ENDPOINT_FLAGS} \
    --run-dir "${RUN_DIR}/agents" \
    --timeout "${AGENT_TIMEOUT_SECONDS}"

# The agents' own check.py calls are their inner loop; this is the independent sweep over
# everything they wrote, at a bigger size rung than they iterate at. Never fails the job -- a
# partial run is still a result, and the table is what gets reviewed.
echo "===== final verification sweep (preset M) ====="
python3 scripts/numpy_vectorize/check.py --all --preset M --json \
    >"${RUN_DIR}/verify.jsonl" 2>"${RUN_DIR}/verify.log" || true
tail -n 5 "${RUN_DIR}/verify.log" || true
AGENTS_SH

chmod +x "${SERVE_SCRIPT}" "${AGENT_SCRIPT}"
rm -f "${AGENTS_DONE}"

# One srun per replica. Each engine gets its OWN master addr and port -- the ranks inside a
# replica rendezvous with each other and must not find another replica's store.
SERVER_STEP_PIDS=()
ENDPOINT_FLAGS=""
for (( r = 0; r < REPLICAS; r++ )); do
    head="${NODES[4 * r]}"
    slice="${head},${NODES[4 * r + 1]},${NODES[4 * r + 2]},${NODES[4 * r + 3]}"
    stage "starting kimi replica ${r} tp=4 pp=4 on ${slice}"
    NODE0="${head}" MASTER_PORT=$(( MASTER_PORT + r )) REPLICA="${r}" \
    srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --exclusive --export=ALL \
        --nodelist="${slice}" --environment="${INFERENCE_EDF}" \
        bash "${SERVE_SCRIPT}" >"${RUN_DIR}/vllm-step-${r}.log" 2>&1 &
    SERVER_STEP_PIDS+=("$!")
    ENDPOINT_FLAGS+=" --endpoint http://${head}:${API_PORT}"
done
export ENDPOINT_FLAGS

cleanup_steps() {
    touch "${AGENTS_DONE}"
    local pid
    for pid in "${SERVER_STEP_PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            sleep 10
            kill "${pid}" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
}
trap cleanup_steps EXIT INT TERM

# Every replica must answer before the agents start: run_agents.py deals shards round-robin, so
# an endpoint that never came up silently strands its share of the worklist.
deadline=$((SECONDS + READY_TIMEOUT))
for (( r = 0; r < REPLICAS; r++ )); do
    head="${NODES[4 * r]}"
    stage "waiting for http://${head}:${API_PORT}/v1/models (up to ${READY_TIMEOUT}s total)"
    until curl -fsS --max-time 10 "http://${head}:${API_PORT}/v1/models" \
              -o "${RUN_DIR}/models-${r}.json"; do
        kill -0 "${SERVER_STEP_PIDS[r]}" 2>/dev/null || {
            echo "FAIL: vLLM step for replica ${r} exited before its API came up" >&2
            tail -n 80 "${RUN_DIR}/vllm-rank0.log" >&2 || true
            exit 1
        }
        (( SECONDS < deadline )) || { echo "FAIL: replica ${r} not ready in ${READY_TIMEOUT}s" >&2; exit 1; }
        sleep 15
    done
    cat "${RUN_DIR}/models-${r}.json"
done

stage "running ${AGENTS} agents on ${AGENT_NODE}"
set +e
srun --nodes=1 --ntasks=1 --exclusive --export=ALL,REPO="${REPO}",AGENTS="${AGENTS}",AGENT_TIMEOUT_SECONDS="${AGENT_TIMEOUT_SECONDS}",ENDPOINT_FLAGS="${ENDPOINT_FLAGS}" \
    --nodelist="${AGENT_NODE}" --environment="${AGENT_EDF}" bash "${AGENT_SCRIPT}"
agent_status=$?
set -e
touch "${AGENTS_DONE}"

printf '\n===== NUMPY VECTORIZE DONE (agents exited %s) =====\nlogs: %s\n' "${agent_status}" "${RUN_DIR}"
exit "${agent_status}"
