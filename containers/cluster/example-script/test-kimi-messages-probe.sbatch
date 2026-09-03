#!/usr/bin/env bash
#SBATCH --job-name=kimi-messages-probe
#SBATCH --partition=mi300
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=03:00:00
#SBATCH --output=/capstor/scratch/cscs/ybudanaz/x86_64/claude-roundtrip/logs/kimi-probe-%j.out
#SBATCH --error=/capstor/scratch/cscs/ybudanaz/x86_64/claude-roundtrip/logs/kimi-probe-%j.err
#
# Diagnostic for the kimi /v1/messages failure seen in smoke 589327: every request returned
# an instant 200 with zero engine activity, and the claude client burned all 10 retries on
# error=unknown/error_status=null. The anthropic adapter is byte-identical to the working
# rocm723 image, so the suspects are the kimi_k2 tool parser and the Kimi chat template --
# both only reachable with a live server. This job boots the campaign exact pp=4
# serve once (eager mode: CUDA graphs are irrelevant to protocol bugs and cost up to 2 h),
# then captures RAW response bodies for the request shapes claude-code sends.
#
#   sbatch --account=<account> test-kimi-messages-probe.sbatch
#
# Stages (client node, after readiness):
#   1 GET  /v1/models
#   2 POST /v1/messages          non-stream, trivial            -> raw body
#   3 POST /v1/messages          stream, trivial                -> raw SSE
#   4 POST /v1/messages          stream, system+tools+metadata  -> raw SSE  (claude-code shape)
#   5 POST /v1/chat/completions  tools[], non-stream            -> kimi_k2 parser in isolation
#   6 claude -p direct against /v1/messages                     -> the hard gate
#
# The server runs with VLLM_LOGGING_LEVEL=DEBUG on rank 0: the anthropic router logs the
# received request, the converted OpenAI request, and the full response at debug level.

set -euo pipefail

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the repo with 0-byte core_nid<node>_<pid> stubs (the size limit
# truncates the dump after the kernel has created the file). Slurm propagates CORE to steps.
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
export RUN_DIR="${SCRATCH}/claude-roundtrip/logs/${SLURM_JOB_ID}"
# iopsstor reads ~11x faster than capstor (job 593523); capstor keeps the durable copy.
FAST_SCRATCH="/iopsstor/scratch/cscs/${USER}"
[[ -d "${FAST_SCRATCH}" ]] || FAST_SCRATCH="${SCRATCH}"
export HF_HOME="${HF_HOME:-${FAST_SCRATCH}/hf}"

INFERENCE_EDF="${INFERENCE_EDF:-rocm723-vllm-0.27.1-pytorch211-ofi}"
AGENT_EDF="${AGENT_EDF:-optarena-amd-mi300}"

# Flag-for-flag the llr4 kimi campaign serve (see .env.llr4-kimi27code-*), plus --enforce-eager.
export MODEL_REPO="${MODEL_REPO:-moonshotai/Kimi-K2.7-Code}"
export SERVED_MODEL="${SERVED_MODEL:-optarena-vllm}"
export API_PORT="${API_PORT:-8000}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
READY_TIMEOUT="${READY_TIMEOUT:-7200}"

mkdir -p "${RUN_DIR}"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if (( ${#NODES[@]} != 5 )); then
    echo "FAIL: need exactly 5 nodes, allocation has ${#NODES[@]}" >&2
    exit 2
fi
export NODE0="${NODES[0]}"
NODE1="${NODES[1]}"
NODE2="${NODES[2]}"
NODE3="${NODES[3]}"
NODE4="${NODES[4]}"

SERVE_SCRIPT="${RUN_DIR}/serve.sh"
CLIENT_SCRIPT="${RUN_DIR}/client.sh"
CLIENT_DONE="${RUN_DIR}/client.done"

stage() { printf '\n===== stage: %s =====\n' "$*"; }

cat <<EOF
job:       ${SLURM_JOB_ID}
inference: ${NODE0}..${NODE3} (${INFERENCE_EDF}, pp=4 tp=4) port ${API_PORT}
client:    ${NODE4} (${AGENT_EDF})
model:     ${MODEL_REPO} served as ${SERVED_MODEL}
run dir:   ${RUN_DIR}
EOF

# --------------------------------------------------------------------------------------------
# serve: one step, two tasks; SLURM_PROCID is the pipeline node rank, as in run_cluster.sh.
# --------------------------------------------------------------------------------------------
cat >"${SERVE_SCRIPT}" <<'SERVE'
#!/usr/bin/env bash
set -Eeuo pipefail

rank="${SLURM_PROCID:-0}"
export TOKENIZERS_PARALLELISM=false
# VLLM_DISABLE_PYNCCL is deliberately NOT set. It was the workaround this probe was written
# under, and it is superseded: without PyNCCL every collective goes through ProcessGroupNCCL,
# which is not graph-capturable, so capture stalls, --enforce-eager becomes mandatory, and
# kimi decodes at 1.4 tok/s against 16.8 on this same shape. run_cluster.sh stopped
# defaulting it on 2026-08-13 and vectorize.sbatch names this file as the last holdout.
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export NCCL_DEBUG_FILE="${RUN_DIR}/nccl.%h.%p.log"
# Debug logging only on the API rank: that is where the anthropic router logs live.
if (( rank == 0 )); then
    export VLLM_LOGGING_LEVEL=DEBUG
fi
# vLLM reads env VLLM_PORT as the BASE for internal ZMQ ports (589170 lesson): keep it unset.
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
    --generation-config vllm
    --enable-auto-tool-choice
    --tool-call-parser kimi_k2
    --trust-remote-code
    --max-model-len 65536
    --gpu-memory-utilization 0.85
    --enforce-eager
    --reasoning-parser kimi_k2
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

while [[ ! -e "${RUN_DIR}/client.done" ]]; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "rank ${rank}: vLLM exited before the client stage finished" >&2
        wait "${SERVER_PID}" || true
        exit 1
    fi
    sleep 5
done
echo "rank ${rank}: client done, shutting down"
SERVE

# --------------------------------------------------------------------------------------------
# client: raw-body probes, then claude -p direct.
# --------------------------------------------------------------------------------------------
cat >"${CLIENT_SCRIPT}" <<'CLIENT'
#!/usr/bin/env bash
set -uo pipefail
# no -e: every probe must run even if an earlier one fails; verdicts are printed per probe.

BASE="http://${NODE0}:${API_PORT}"
export HOME="${RUN_DIR}/agent-home"
export CLAUDE_CONFIG_DIR="${HOME}/.claude"
export DISABLE_AUTOUPDATER=1 DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1
export DISABLE_NON_ESSENTIAL_MODEL_CALLS=1
mkdir -p "${HOME}" "${CLAUDE_CONFIG_DIR}" "${RUN_DIR}/agent-cwd"

# The CXI hook breaks the container's curl (libldap, seen job 583987); python is immune.
probe() {
    # probe <name> <path> <payload-file|-> ; dumps status, headers and raw body to files.
    local name="$1" path="$2" payload="$3"
    python3 - "$name" "${BASE}${path}" "$payload" <<'PY'
import sys, urllib.request, urllib.error, os

name, url, payload = sys.argv[1], sys.argv[2], sys.argv[3]
run_dir = os.environ["RUN_DIR"]
out = os.path.join(run_dir, f"probe-{name}")

if payload == "-":
    req = urllib.request.Request(url)
else:
    req = urllib.request.Request(url, data=open(payload, "rb").read(),
                                 headers={"Content-Type": "application/json",
                                          "anthropic-version": "2023-06-01"})
status, headers, body = None, "", b""
try:
    with urllib.request.urlopen(req, timeout=900) as resp:
        status = resp.status
        headers = str(resp.headers)
        body = resp.read()
except urllib.error.HTTPError as err:
    status = err.code
    headers = str(err.headers)
    body = err.read()
except Exception as err:
    body = repr(err).encode()

with open(out + ".status", "w") as fh:
    fh.write(f"{status}\n")
with open(out + ".headers", "w") as fh:
    fh.write(headers)
with open(out + ".body", "wb") as fh:
    fh.write(body)
print(f"----- {name}: status={status} bytes={len(body)}")
print(body[:2000].decode("utf-8", "replace"))
PY
}

echo "===== probe 1: GET /v1/models ====="
probe models /v1/models -

echo "===== probe 2: /v1/messages non-stream trivial ====="
cat >"${RUN_DIR}/p2.json" <<EOF
{"model": "${SERVED_MODEL}", "max_tokens": 64,
 "messages": [{"role": "user", "content": "Reply with the single word hello."}]}
EOF
probe messages-plain /v1/messages "${RUN_DIR}/p2.json"

echo "===== probe 3: /v1/messages STREAM trivial ====="
cat >"${RUN_DIR}/p3.json" <<EOF
{"model": "${SERVED_MODEL}", "max_tokens": 64, "stream": true,
 "messages": [{"role": "user", "content": "Reply with the single word hello."}]}
EOF
probe messages-stream /v1/messages "${RUN_DIR}/p3.json"

echo "===== probe 4: /v1/messages STREAM claude-code shape (system list + tools + metadata) ====="
cat >"${RUN_DIR}/p4.json" <<EOF
{"model": "${SERVED_MODEL}", "max_tokens": 256, "stream": true,
 "system": [{"type": "text", "text": "You are a coding agent. Use tools when asked."}],
 "metadata": {"user_id": "probe"},
 "tools": [{"name": "Bash", "description": "Run a shell command.",
            "input_schema": {"type": "object",
                             "properties": {"command": {"type": "string"}},
                             "required": ["command"]}}],
 "messages": [{"role": "user",
               "content": [{"type": "text", "text": "Use the Bash tool to run ls."}]}]}
EOF
probe messages-tools /v1/messages "${RUN_DIR}/p4.json"

echo "===== probe 5: /v1/chat/completions tools[] (kimi_k2 parser isolation) ====="
cat >"${RUN_DIR}/p5.json" <<EOF
{"model": "${SERVED_MODEL}", "max_tokens": 256, "temperature": 0,
 "messages": [{"role": "user", "content": "What is the weather in Zurich? Use the get_weather tool."}],
 "tools": [{"type": "function", "function": {"name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}}],
 "tool_choice": "auto"}
EOF
probe chat-tools /v1/chat/completions "${RUN_DIR}/p5.json"

echo "===== probe 6: claude -p DIRECT against /v1/messages ====="
export ANTHROPIC_AUTH_TOKEN="EMPTY"
export ANTHROPIC_API_KEY="EMPTY"
(
    cd "${RUN_DIR}/agent-cwd" || exit 1
    ANTHROPIC_BASE_URL="${BASE}" timeout 900 claude -p \
        "Use the Bash tool to run 'ls'. Then reply with exactly what it printed." \
        --model "${SERVED_MODEL}" --max-turns 4 \
        --output-format stream-json --verbose \
        --allowedTools "Bash(ls:*)" "Bash(ls*)" "Bash"
) >"${RUN_DIR}/claude-direct.jsonl" 2>"${RUN_DIR}/claude-direct.stderr.log"
echo "claude exit: $?"
echo "----- claude-direct.jsonl tail:"
tail -n 5 "${RUN_DIR}/claude-direct.jsonl" || true
echo "----- claude-direct.stderr tail:"
tail -n 20 "${RUN_DIR}/claude-direct.stderr.log" || true

echo "probes complete; raw bodies in ${RUN_DIR}/probe-*.body"
CLIENT

chmod +x "${SERVE_SCRIPT}" "${CLIENT_SCRIPT}"
rm -f "${CLIENT_DONE}"

stage "starting kimi pp=4 on ${NODE0}-${NODE3}"
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --exclusive --export=ALL \
    --nodelist="${NODE0},${NODE1},${NODE2},${NODE3}" --environment="${INFERENCE_EDF}" \
    bash "${SERVE_SCRIPT}" >"${RUN_DIR}/vllm-step.log" 2>&1 &
SERVER_STEP_PID=$!

cleanup_steps() {
    touch "${CLIENT_DONE}"
    if kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
        sleep 10
        kill "${SERVER_STEP_PID}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
}
trap cleanup_steps EXIT INT TERM

stage "waiting for http://${NODE0}:${API_PORT}/v1/models (up to ${READY_TIMEOUT}s)"
deadline=$((SECONDS + READY_TIMEOUT))
until curl -fsS --max-time 10 "http://${NODE0}:${API_PORT}/v1/models" -o "${RUN_DIR}/models.json"; do
    kill -0 "${SERVER_STEP_PID}" 2>/dev/null || {
        echo "FAIL: the vLLM step exited before the API came up" >&2
        tail -n 80 "${RUN_DIR}/vllm-rank0.log" >&2 || true
        exit 1
    }
    (( SECONDS < deadline )) || { echo "FAIL: API not ready in ${READY_TIMEOUT}s" >&2; exit 1; }
    sleep 15
done
cat "${RUN_DIR}/models.json"

stage "client probes on ${NODE2}"
set +e
srun --nodes=1 --ntasks=1 --exclusive --export=ALL --nodelist="${NODE4}" \
    --environment="${AGENT_EDF}" bash "${CLIENT_SCRIPT}"
client_status=$?
set -e
touch "${CLIENT_DONE}"

stage "rank0 anthropic debug log excerpts"
grep -a -m 40 -i 'anthropic\|Received messages\|Convert to OpenAI\|Error in' \
    "${RUN_DIR}/vllm-rank0.log" | head -60 || true

(( client_status == 0 )) || { echo "FAIL: client stage exited ${client_status}" >&2; exit 1; }
printf '\n===== KIMI MESSAGES PROBE DONE =====\nlogs: %s\n' "${RUN_DIR}"
