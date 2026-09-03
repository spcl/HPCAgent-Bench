#!/usr/bin/env bash
# Kimi K2.7 REPLICATED serving smoke: N independent tp=4 pp=REPLICA_SIZE vLLM groups in ONE
# allocation, one API endpoint per group, saturation probe fanned over all endpoints.
#
# This is the topology the kimi campaign needs (~900 tok/s gate / ~154 tok/s per tuned 4-node
# replica = ~6 replicas); the per-group serve line is byte-compatible with the proven
# smoke-kimi-eager-pg.sbatch, only the (nnodes, node-rank, master) triple is group-local.
# Submit: mkdir -p logs && sbatch smoke-kimi-replicas.sbatch     # 8 nodes = 2 replicas
#         sbatch --nodes=12 ... for 3 replicas, etc. NNODES must divide by REPLICA_SIZE.
#SBATCH --job-name=smoke-kimi-replicas
#SBATCH --partition=mi300
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/smoke-kimi-replicas-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/smoke-kimi-replicas-%j.err

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
export RUN_DIR="${SCRATCH}/kimi-smoke/${SLURM_JOB_ID}"
FAST_SCRATCH="/iopsstor/scratch/cscs/${USER}"
[[ -d "${FAST_SCRATCH}" ]] || FAST_SCRATCH="${SCRATCH}"
export HF_HOME="${HF_HOME:-${FAST_SCRATCH}/hf}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PG_PATCH_REL="containers/cluster/ce-images/inference/external-eager-pg-patch"
for cand in "${PG_PATCH_DIR:-}" "${SUBMIT_DIR}/external-eager-pg-patch" "${SUBMIT_DIR}/${PG_PATCH_REL}"; do
    if [[ -n "${cand}" && -f "${cand}/sitecustomize.py" ]]; then
        export PG_PATCH_DIR="${cand}"
        break
    fi
done
test -f "${PG_PATCH_DIR:-}/sitecustomize.py" 2>/dev/null || {
    echo "FAIL: no sitecustomize.py found from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
    exit 2
}
# The tuned fused_moe config is part of what this smoke measures: without it every replica
# pays the 1.71x untuned-kernel tax (597893 vs 597894).
MOE_DIR="${PG_PATCH_DIR%/external-eager-pg-patch}/moe-configs"
[[ -d "${MOE_DIR}" ]] && export VLLM_TUNED_CONFIG_FOLDER="${VLLM_TUNED_CONFIG_FOLDER:-${MOE_DIR}}"

INFERENCE_EDF="${INFERENCE_EDF:-rocm723-vllm-0.27.1-pytorch211-ofi}"
export MODEL_REPO="${MODEL_REPO:-moonshotai/Kimi-K2.7-Code}"
export SERVED_MODEL="${SERVED_MODEL:-optarena-vllm}"
export API_PORT="${API_PORT:-8000}"
export MASTER_PORT_BASE="${MASTER_PORT_BASE:-$((20000 + SLURM_JOB_ID % 20000))}"
READY_TIMEOUT="${READY_TIMEOUT:-5400}"
# Per REPLICA, matching the campaign's per-server load; total streams = this x NREPLICAS.
PROBE_STREAMS="${PROBE_STREAMS:-64}"
PROBE_MAX_TOKENS="${PROBE_MAX_TOKENS:-512}"

mkdir -p "${RUN_DIR}"
mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
NNODES="${#NODES[@]}"
export REPLICA_SIZE="${REPLICA_SIZE:-4}"
if (( NNODES % REPLICA_SIZE != 0 )); then
    echo "FAIL: ${NNODES} nodes do not divide into replicas of ${REPLICA_SIZE}" >&2
    exit 2
fi
NREPLICAS=$((NNODES / REPLICA_SIZE))
export TP_SIZE="${TP_SIZE:-4}"
export TOOL_PARSER="${TOOL_PARSER:-kimi_k2}"
export REASONING_PARSER="${REASONING_PARSER:-kimi_k2}"
export EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"
NODELIST="$(IFS=,; echo "${NODES[*]}")"
# Rank r serves group r/REPLICA_SIZE; the group's master is its first node.
MASTERS=()
for ((g = 0; g < NREPLICAS; g++)); do MASTERS+=("${NODES[g * REPLICA_SIZE]}"); done
export MASTERS_CSV="$(IFS=,; echo "${MASTERS[*]}")"

SERVE_SCRIPT="${RUN_DIR}/serve.sh"
SMOKE_DONE="${RUN_DIR}/smoke.done"
stage() { printf '\n===== stage: %s =====\n' "$*"; }

cat <<EOF
job:      ${SLURM_JOB_ID}
model:    ${MODEL_REPO} tp=${TP_SIZE} pp=${REPLICA_SIZE} x ${NREPLICAS} replicas on ${NNODES} nodes
masters:  ${MASTERS_CSV}
patch:    ${PG_PATCH_DIR}
moe cfg:  ${VLLM_TUNED_CONFIG_FOLDER:-<none>}
logs:     ${RUN_DIR}
EOF

cat >"${SERVE_SCRIPT}" <<'SERVE'
#!/usr/bin/env bash
set -Eeuo pipefail
rank="${SLURM_PROCID:-0}"
group=$((rank / REPLICA_SIZE))
lrank=$((rank % REPLICA_SIZE))
IFS=, read -ra MASTERS <<<"${MASTERS_CSV}"
GROUP_MASTER="${MASTERS[group]}"
# Distinct rendezvous port per group; the HTTP port needs no offset (one rank0 per host).
GROUP_MASTER_PORT=$((MASTER_PORT_BASE + group * 16))

export TOKENIZERS_PARALLELISM=false
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_FILE="${RUN_DIR}/nccl.%h.%p.log"
export PYTHONPATH="${PG_PATCH_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
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
echo "rank ${rank} group ${group}/${lrank} host $(hostname) master ${GROUP_MASTER}:${GROUP_MASTER_PORT}"

ARGS=(
    "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL}"
    --tensor-parallel-size "${TP_SIZE}"
    --pipeline-parallel-size "${REPLICA_SIZE}"
    --distributed-executor-backend mp
    --nnodes "${REPLICA_SIZE}"
    --node-rank "${lrank}"
    --master-addr "${GROUP_MASTER}"
    --master-port "${GROUP_MASTER_PORT}"
    --distributed-timeout-seconds 7200
    --load-format safetensors
    --safetensors-load-strategy prefetch
    --generation-config vllm
    --enable-auto-tool-choice
    --tool-call-parser "${TOOL_PARSER}"
    --trust-remote-code
    --max-model-len 65536
    --gpu-memory-utilization 0.70
    --max-num-seqs 128
)
if [[ -n "${REASONING_PARSER}" ]]; then
    ARGS+=(--reasoning-parser "${REASONING_PARSER}")
fi
if [[ -n "${EXTRA_SERVE_ARGS}" ]]; then
    # shellcheck disable=SC2206 -- word-splitting is the contract, same as VLLM_EXTRA_ARGS
    ARGS+=(${EXTRA_SERVE_ARGS})
fi
if (( lrank == 0 )); then
    ARGS+=(--host 0.0.0.0 --port "${API_PORT}")
else
    ARGS+=(--headless)
fi

SERVER_PID=""
cleanup() { [[ -n "${SERVER_PID}" ]] && kill "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 143' TERM

vllm serve "${ARGS[@]}" >"${RUN_DIR}/vllm-rank${rank}.log" 2>&1 &
SERVER_PID=$!
echo "rank ${rank} vllm pid ${SERVER_PID}"

while [[ ! -e "${RUN_DIR}/smoke.done" ]]; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "rank ${rank}: vLLM exited before the smoke finished" >&2
        wait "${SERVER_PID}" || true
        exit 1
    fi
    sleep 5
done
SERVE
chmod +x "${SERVE_SCRIPT}"
rm -f "${SMOKE_DONE}"

stage "serving ${NREPLICAS} x (tp=${TP_SIZE} pp=${REPLICA_SIZE}) with the eager-PG patch"
srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 --exclusive --export=ALL \
    --nodelist="${NODELIST}" --environment="${INFERENCE_EDF}" \
    bash "${SERVE_SCRIPT}" >"${RUN_DIR}/vllm-step.log" 2>&1 &
SERVER_STEP_PID=$!
cleanup_steps() {
    touch "${SMOKE_DONE}"
    kill -0 "${SERVER_STEP_PID}" 2>/dev/null && { sleep 10; kill "${SERVER_STEP_PID}" 2>/dev/null || true; }
    wait 2>/dev/null || true
}
trap cleanup_steps EXIT INT TERM

metric_on() { { curl -fsS --max-time 10 "http://$1:${API_PORT}/metrics" 2>/dev/null || true; } |
    awk -v k="vllm:$2" '$0 ~ "^"k {print $2; exit}'; }
metric_sum() {
    local total=0 v m
    for m in "${MASTERS[@]}"; do
        v="$(metric_on "${m}" "$1")"
        total=$(awk -v a="${total}" -v b="${v:-0}" 'BEGIN{print a + b}')
    done
    echo "${total}"
}

stage "waiting for ALL ${NREPLICAS} APIs (up to ${READY_TIMEOUT}s)"
deadline=$((SECONDS + READY_TIMEOUT))
for m in "${MASTERS[@]}"; do
    until curl -fsS --max-time 10 "http://${m}:${API_PORT}/v1/models" -o "${RUN_DIR}/models-${m}.json"; do
        kill -0 "${SERVER_STEP_PID}" 2>/dev/null || { echo "FAIL: vLLM step died before ${m} came up" >&2; exit 1; }
        (( SECONDS < deadline )) || { echo "FAIL: ${m} API not ready in ${READY_TIMEOUT}s" >&2; exit 1; }
        sleep 15
    done
    echo "ready: ${m}"
done

stage "gate 1: did the eager-PG patch load?"
if grep -qh "\[external-eager-pg\]" "${RUN_DIR}"/vllm-rank*.log 2>/dev/null; then
    grep -ho "\[external-eager-pg\] [a-z_]*" "${RUN_DIR}"/vllm-rank*.log | sort | uniq -c
else
    echo "FAIL: no [external-eager-pg] lines -- sitecustomize never loaded" >&2
    exit 1
fi

stage "gate 2: per-replica decode -- one completion against EACH endpoint"
for m in "${MASTERS[@]}"; do
    before=$(metric_on "${m}" generation_tokens_total)
    curl -fsS --max-time "${FIRST_TOKEN_TIMEOUT:-300}" "http://${m}:${API_PORT}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${SERVED_MODEL}\",\"prompt\":\"Write one line of C.\",\"max_tokens\":64}" \
        -o "${RUN_DIR}/single-${m}.json" || true
    after=$(metric_on "${m}" generation_tokens_total)
    echo "${m} generation_tokens_total: ${before:-0} -> ${after:-0}"
    awk -v a="${before:-0}" -v b="${after:-0}" 'BEGIN{ if (b-a <= 0) exit 1 }' || {
        echo "FAIL: replica ${m} decoded zero tokens" >&2
        exit 1
    }
done

stage "gate 3: saturation probe, ${PROBE_STREAMS} concurrent x ${PROBE_MAX_TOKENS} tokens PER REPLICA"
g0=$(metric_sum generation_tokens_total); t0=$SECONDS
PROBE_PIDS=()
for ((i = 0; i < PROBE_STREAMS * NREPLICAS; i++)); do
    m="${MASTERS[i % NREPLICAS]}"
    curl -fsS --max-time 900 "http://${m}:${API_PORT}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${SERVED_MODEL}\",\"prompt\":\"Explain loop tiling, variation ${i}.\",\"max_tokens\":${PROBE_MAX_TOKENS},\"ignore_eos\":true}" \
        -o "${RUN_DIR}/probe-${i}.json" &
    PROBE_PIDS+=($!)
done
: >"${RUN_DIR}/probe-conc.txt"
( while :; do metric_sum num_requests_running >>"${RUN_DIR}/probe-conc.txt"; sleep 5; done ) &
SAMPLER_PID=$!
wait "${PROBE_PIDS[@]}" || true
g1=$(metric_sum generation_tokens_total); t1=$SECONDS
kill "${SAMPLER_PID}" 2>/dev/null || true
awk -v a="${g0:-0}" -v b="${g1:-0}" -v dt="$((t1-t0))" -v nr="${NREPLICAS}" '
    $1 > 0 { s += $1; n += 1; if ($1 > peak) peak = $1 }
    END {
        conc = n ? s / n : 0
        rate = dt > 0 ? (b - a) / dt : 0
        printf "SATURATED DECODE: %.1f tok/s aggregate over %ds (avg %.0f / peak %.0f concurrent, %d replicas)\n", rate, dt, conc, peak, nr
        printf "                  %.1f tok/s per replica\n", rate / nr
        if (conc > 0) printf "                  %.1f tok/s per request\n", rate / conc
        if (rate <= 0) { print "FAIL: no decode progress under load"; exit 1 }
    }' "${RUN_DIR}/probe-conc.txt" || exit 1

touch "${SMOKE_DONE}"
printf '\n===== KIMI REPLICAS SMOKE PASSED =====\nlogs: %s\n' "${RUN_DIR}"
