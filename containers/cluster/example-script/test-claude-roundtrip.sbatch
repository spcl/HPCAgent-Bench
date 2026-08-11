#!/usr/bin/env bash
#SBATCH --job-name=claude-roundtrip
#SBATCH --partition=mi300
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:45:00
#SBATCH --output=/capstor/scratch/cscs/ybudanaz/x86_64/claude-roundtrip/logs/roundtrip-%j.out
#SBATCH --error=/capstor/scratch/cscs/ybudanaz/x86_64/claude-roundtrip/logs/roundtrip-%j.err
#
# End-to-end sanity job for the agent path: claude CLI -> LiteLLM -> vLLM gpt-oss-120b, tool calls
# included. Nothing here is a benchmark; every stage either PASSes or the job exits nonzero.
#
#   mkdir -p /capstor/scratch/cscs/ybudanaz/x86_64/claude-roundtrip/logs
#   sbatch --account=<account> \
#       /capstor/scratch/cscs/ybudanaz/x86_64/optarena/containers/cluster/example-script/test-claude-roundtrip.sbatch
#
# No account is hardcoded; pass it on the command line. Slurm spools this file to its own
# directory, so nothing may be resolved relative to the script path: SCRATCH anchors everything.
#
# Stages:
#   1 parser probe   - ask the running vLLM build which tool/reasoning parsers it registers
#   2 server         - vLLM gpt-oss-120b on node 0, TP=4, held open for the client stage
#   3 api ready      - /v1/models answers on node 0
#   4 direct tools   - raw chat/completions with tools[]; isolates the vLLM parser from claude
#   5 claude roundtrip - claude -p through LiteLLM, forced through the Bash tool
#
# The inference harness containers/cluster/ce-images/inference/test-vllm-gpt-oss-120b-1n4g.sh is
# NOT reused as the server: its EXIT trap kills vLLM as soon as its own smoke test finishes, so it
# cannot host a second client stage. Its exact serve flags are copied instead.

set -euo pipefail

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
REPO="${HPCAGENT_BENCH_REPO:-${SCRATCH}/optarena}"

export RUN_DIR="${SCRATCH}/claude-roundtrip/logs/${SLURM_JOB_ID}"
# The 120b weights are already in this cache; a cold cache turns stage 2 into a 65 GiB download.
export HF_HOME="${SCRATCH}/hf"

INFERENCE_EDF="${INFERENCE_EDF:-rocm723-vllm-0.23.0-pytorch211-ofi}"
AGENT_EDF="${AGENT_EDF:-optarena-amd-mi300}"

export MODEL_REPO="${MODEL_REPO:-openai/gpt-oss-120b}"
export SERVED_MODEL="${SERVED_MODEL:-stress-model}"
export TP_SIZE="${TP_SIZE:-4}"
export PP_SIZE="${PP_SIZE:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export API_PORT="${API_PORT:-8000}"
# Per-job ports so a second copy of this job on the same node does not collide.
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export LITELLM_PORT="${LITELLM_PORT:-$((4000 + SLURM_JOB_ID % 1000))}"
export CLAUDE_MODEL="${CLAUDE_MODEL:-optarena-llm}"
# Unguessable enough that the model cannot name it without actually running ls.
export MARKER="roundtrip-marker-${SLURM_JOB_ID}.txt"

READY_TIMEOUT="${READY_TIMEOUT:-1800}"

mkdir -p "${RUN_DIR}"
command -v curl >/dev/null || { echo "FAIL: no curl on the batch host" >&2; exit 1; }

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if (( ${#NODES[@]} != 2 )); then
    echo "FAIL: need exactly 2 nodes, allocation has ${#NODES[@]}" >&2
    exit 2
fi
export NODE0="${NODES[0]}"
NODE1="${NODES[1]}"

SERVE_SCRIPT="${RUN_DIR}/serve.sh"
CLIENT_SCRIPT="${RUN_DIR}/client.sh"
SERVER_LOG="${RUN_DIR}/vllm-server.log"
CLIENT_DONE="${RUN_DIR}/client.done"

stage() { printf '\n===== stage: %s =====\n' "$*"; }
pass() { printf 'PASS: %s\n' "$*"; }

dump_logs() {
    local log
    for log in "${RUN_DIR}/vllm-step.log" "${SERVER_LOG}" "${RUN_DIR}/litellm.log" \
               "${RUN_DIR}/claude-direct.stderr.log" "${RUN_DIR}/claude-litellm.stderr.log"; do
        [[ -s "${log}" ]] || continue
        printf '\n----- tail %s -----\n' "${log}"
        tail -n 120 "${log}"
    done
}

fail() {
    printf '\nFAIL: %s\n' "$*" >&2
    dump_logs >&2
    exit 1
}

cat <<EOF
job:        ${SLURM_JOB_ID}
inference:  ${NODE0} (${INFERENCE_EDF}) port ${API_PORT}
agent:      ${NODE1} (${AGENT_EDF}) litellm port ${LITELLM_PORT}
model:      ${MODEL_REPO} served as ${SERVED_MODEL}
run dir:    ${RUN_DIR}
repo:       ${REPO}
EOF

# --------------------------------------------------------------------------------------------
# node 0: parser probe, then the server, held open until the client stage is done.
# --------------------------------------------------------------------------------------------
cat >"${SERVE_SCRIPT}" <<'SERVE'
#!/usr/bin/env bash
set -Eeuo pipefail

# Runs inside the inference EDF on node 0. Every value arrives through the job environment.
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_LOGGING_LEVEL=INFO
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG_FILE="${RUN_DIR}/nccl.%h.%p.log"

echo "host: $(hostname)  python: $(command -v python)  vllm: $(command -v vllm)"
test "$(command -v python)" = "/opt/pytorch211/bin/python"

echo "===== tool / reasoning parser probe ====="

# Probes the registry of the build that is about to serve, rather than trusting a name. In
# vLLM 0.23.0 the parsers moved to vllm.tool_parsers and are registered LAZILY, so the old
# import path and the plain .tool_parsers dict both come up short; hence the module list, the
# list_registered() preference, and the source-table fallback.
python - >"${RUN_DIR}/parsers.env" <<'PY_PROBE'
import importlib
import json
import re
import sys
from pathlib import Path

TOOL_MODULES = ("vllm.tool_parsers", "vllm.entrypoints.openai.tool_parsers")
REASON_MODULES = ("vllm.reasoning", "vllm.entrypoints.openai.reasoning_parsers")
# Ordered and explicit. Substring matching is forbidden here: "seed_oss" is a different model's
# parser and wins any "*oss*" heuristic. "openai" is the harmony parser gpt-oss needs.
TOOL_CANDIDATES = ("openai", "gpt_oss", "gptoss", "harmony")
REASON_CANDIDATES = ("openai_gptoss", "gpt_oss", "gptoss")


def manager_names(module_paths, manager_name):
    for path in module_paths:
        try:
            module = importlib.import_module(path)
        except ImportError:
            continue
        members = vars(module)
        if manager_name not in members:
            continue
        manager = members[manager_name]
        entries = vars(manager)
        if "list_registered" in entries:
            return sorted(manager.list_registered()), path
        names = set()
        for key in ("tool_parsers", "reasoning_parsers", "lazy_parsers"):
            table = entries.get(key)
            if isinstance(table, dict):
                names.update(table)
        if names:
            return sorted(names), path
    return [], ""


def source_names(relative):
    # No importable registry: read the registration table out of the shipped source instead.
    path = Path("/opt/vllm-src/vllm") / relative
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    return sorted(set(re.findall(r'^\s*"([A-Za-z0-9_.-]+)":\s*\(', text, re.M)))


def pick(candidates, names):
    for name in candidates:
        if name in names:
            return name
    return ""


tool_names, tool_origin = manager_names(TOOL_MODULES, "ToolParserManager")
if not tool_names:
    tool_names, tool_origin = source_names("tool_parsers/__init__.py"), "source"

reason_names, reason_origin = manager_names(REASON_MODULES, "ReasoningParserManager")
if not reason_names:
    reason_names, reason_origin = source_names("reasoning/__init__.py"), "source"

print("tool parsers from", tool_origin or "nowhere", json.dumps(tool_names), file=sys.stderr)
print("reasoning parsers from", reason_origin or "nowhere", json.dumps(reason_names), file=sys.stderr)

tool = pick(TOOL_CANDIDATES, tool_names)
if not tool:
    print("no gpt-oss tool parser among", json.dumps(tool_names), file=sys.stderr)
    raise SystemExit(1)

reason = pick(REASON_CANDIDATES, reason_names)

print(f"TOOL_CALL_PARSER={tool}")
if reason:
    print(f"REASONING_PARSER={reason}")
PY_PROBE

# shellcheck disable=SC1090
. "${RUN_DIR}/parsers.env"
echo "chosen tool parser:      ${TOOL_CALL_PARSER}"
echo "chosen reasoning parser: ${REASONING_PARSER:-<none>}"

echo "===== resolving model snapshot ====="
MODEL_PATH="$(python - <<'PY_SNAPSHOT' | tail -n 1
import os

from huggingface_hub import snapshot_download

print(snapshot_download(repo_id=os.environ["MODEL_REPO"], max_workers=8))
PY_SNAPSHOT
)"
test -d "${MODEL_PATH}"
echo "model path: ${MODEL_PATH}"

# Flag-for-flag the set that passed as job 582991, plus the two tool-calling flags.
ARGS=(
    "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL}"
    --dtype bfloat16
    --load-format safetensors
    --tensor-parallel-size "${TP_SIZE}"
    --pipeline-parallel-size "${PP_SIZE}"
    --distributed-executor-backend mp
    --nnodes 1
    --node-rank 0
    --master-addr "$(hostname)"
    --master-port "${MASTER_PORT}"
    --distributed-timeout-seconds 3600
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --generation-config vllm
    --host 0.0.0.0
    --port "${API_PORT}"
    --enable-auto-tool-choice
    --tool-call-parser "${TOOL_CALL_PARSER}"
)
if [[ -n "${REASONING_PARSER:-}" ]]; then
    ARGS+=(--reasoning-parser "${REASONING_PARSER}")
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

echo "===== starting vLLM ====="
vllm serve "${ARGS[@]}" >"${RUN_DIR}/vllm-server.log" 2>&1 &
SERVER_PID=$!
echo "vllm pid ${SERVER_PID}"

# The server outlives its own readiness on purpose: the client stage on the other node is what
# ends it, through this sentinel on the shared filesystem.
while [[ ! -e "${RUN_DIR}/client.done" ]]; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "vLLM exited before the client stage finished" >&2
        wait "${SERVER_PID}" || true
        exit 1
    fi
    sleep 5
done

echo "client stage finished; shutting the server down"
SERVE

# --------------------------------------------------------------------------------------------
# node 1: LiteLLM, one direct tools[] call, one claude roundtrip.
# --------------------------------------------------------------------------------------------
cat >"${CLIENT_SCRIPT}" <<'CLIENT'
#!/usr/bin/env bash
set -Eeuo pipefail

# Runs inside the agent EDF on node 1.
# The CXI hook injects host (SLES) libcurl through the container's ld.so cache, so even a
# clean-env /usr/bin/curl dies on libldap_r-2.4 (seen live, job 583987). python3 is immune.
http_get() { python3 -c '
import sys, urllib.request
try:
    urllib.request.urlopen(sys.argv[1], timeout=float(sys.argv[2])).read()
except Exception:
    raise SystemExit(1)
' "$1" "$2"; }
http_post_json() { python3 -c '
import sys, urllib.request
req = urllib.request.Request(sys.argv[1], data=open(sys.argv[2], "rb").read(),
                             headers={"Content-Type": "application/json"})
try:
    body = urllib.request.urlopen(req, timeout=float(sys.argv[4])).read()
except Exception as err:
    sys.stderr.write("POST failed: %s\n" % err)
    raise SystemExit(1)
open(sys.argv[3], "wb").write(body)
' "$1" "$2" "$3" "$4"; }

VLLM_URL="http://${NODE0}:${API_PORT}/v1"
AGENT_CWD="${RUN_DIR}/agent-cwd"
CONFIG="${RUN_DIR}/litellm.yaml"

# A throwaway HOME keeps the user's ~/.claude (settings, hooks, memory, MCP servers) out of a
# sanity check, and gives the CLI somewhere writable inside the container.
export HOME="${RUN_DIR}/agent-home"
export CLAUDE_CONFIG_DIR="${HOME}/.claude"
# Compute nodes have no egress; the updater and telemetry would only stall the run.
export DISABLE_AUTOUPDATER=1
export DISABLE_TELEMETRY=1
export DISABLE_ERROR_REPORTING=1
export DISABLE_NON_ESSENTIAL_MODEL_CALLS=1

mkdir -p "${AGENT_CWD}" "${CLAUDE_CONFIG_DIR}"
: >"${AGENT_CWD}/${MARKER}"

command -v litellm >/dev/null 2>&1 || export PATH="/opt/venv/bin:${PATH}"
echo "host: $(hostname)  claude: $(command -v claude)  litellm: $(command -v litellm)"
claude --version || true

cat >"${CONFIG}" <<EOF
model_list:
  - model_name: ${CLAUDE_MODEL}
    litellm_params:
      model: hosted_vllm/${SERVED_MODEL}
      api_base: ${VLLM_URL}
      api_key: EMPTY
  - model_name: "*"
    litellm_params:
      model: hosted_vllm/${SERVED_MODEL}
      api_base: ${VLLM_URL}
      api_key: EMPTY
litellm_settings:
  drop_params: true
  set_verbose: false
EOF

PROXY_PID=""
cleanup() {
    if [[ -n "${PROXY_PID}" ]] && kill -0 "${PROXY_PID}" 2>/dev/null; then
        kill "${PROXY_PID}" 2>/dev/null || true
        wait "${PROXY_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Soft stage: upstream litellm proxy wheels are broken across releases (masked ImportError
# storm); the hard gate below is claude DIRECT against vLLM's native /v1/messages.
LITELLM_OK=0
litellm --config "${CONFIG}" --host 127.0.0.1 --port "${LITELLM_PORT}" \
    >"${RUN_DIR}/litellm.log" 2>&1 &
PROXY_PID=$!

deadline=$((SECONDS + 240))
while :; do
    if http_get "http://127.0.0.1:${LITELLM_PORT}/v1/models" 5; then
        LITELLM_OK=1
        echo "PASS: litellm is serving on 127.0.0.1:${LITELLM_PORT}"
        break
    fi
    kill -0 "${PROXY_PID}" 2>/dev/null || { echo "WARN: litellm died during startup"; break; }
    (( SECONDS < deadline )) || { echo "WARN: litellm not ready after 240s"; break; }
    sleep 3
done

echo
echo "===== stage: direct tools[] roundtrip against vLLM ====="

# Straight at vLLM, no proxy and no CLI: if this fails the tool-call parser is the problem, and
# if it passes but claude does not, the problem is in the LiteLLM/Anthropic translation.
direct_call() {
    local choice="$1" out="$2"
    cat >"${RUN_DIR}/direct-payload.json" <<EOF
{
  "model": "${SERVED_MODEL}",
  "messages": [
    {"role": "user", "content": "What is the weather in Zurich right now? Use the get_weather tool."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string", "description": "City name"}},
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "${choice}",
  "temperature": 0,
  "max_tokens": 512
}
EOF
    http_post_json "http://${NODE0}:${API_PORT}/v1/chat/completions" \
        "${RUN_DIR}/direct-payload.json" "${out}" 300 || return 1

    python3 - "${out}" <<'PY_ASSERT'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)

choices = payload.get("choices") or []
if not choices:
    print("no choices:", json.dumps(payload)[:2000])
    raise SystemExit(1)

calls = choices[0].get("message", {}).get("tool_calls") or []
print(json.dumps(calls, indent=2)[:2000] if calls else json.dumps(choices[0])[:2000])
raise SystemExit(0 if calls else 1)
PY_ASSERT
}

if direct_call auto "${RUN_DIR}/direct-auto.json"; then
    echo "PASS: vLLM emitted a tool_call with tool_choice=auto"
elif direct_call required "${RUN_DIR}/direct-required.json"; then
    echo "WARN: tool_choice=auto produced no tool_call; the parser works under tool_choice=required"
    echo "PASS: vLLM emitted a tool_call with tool_choice=required"
else
    echo "FAIL: vLLM returned no tool_call for either tool_choice" >&2
    exit 1
fi

export ANTHROPIC_AUTH_TOKEN="EMPTY"
export ANTHROPIC_API_KEY="${ANTHROPIC_AUTH_TOKEN}"

PROMPT="Use the Bash tool to run 'ls' in the current directory. Then reply with the exact file"
PROMPT="${PROMPT} names it printed, nothing else."

# Both allowlist spellings: accepted syntax changed across CLI versions, and a denied tool
# would read as a model failure.
claude_roundtrip() {
    local base="$1" model="$2" stream="$3" status
    set +e
    (
        cd "${AGENT_CWD}" || exit 1
        ANTHROPIC_BASE_URL="${base}" timeout 900 claude -p "${PROMPT}" \
            --model "${model}" \
            --max-turns 10 \
            --output-format stream-json \
            --verbose \
            --allowedTools "Bash(ls:*)" "Bash(ls*)" "Bash" \
            --disallowedTools "WebFetch" "WebSearch" "Task"
    ) >"${stream}" 2>"${stream%.jsonl}.stderr.log"
    status=$?
    set -e
    if (( status != 0 )); then
        echo "claude exited ${status} against ${base}" >&2
        tail -n 40 "${stream%.jsonl}.stderr.log" >&2 || true
        return 1
    fi
    check_stream "${stream}"
}

check_stream() {
    python3 - "$1" "${MARKER}" <<'PY_STREAM'
import json
import sys

stream_path, marker = sys.argv[1], sys.argv[2]
used_bash = False
result_text = ""
saw_marker = False

with open(stream_path, encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        if marker in line:
            saw_marker = True
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            result_text = str(event.get("result") or "")
        content = (event.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == "Bash":
                used_bash = True

print("bash tool_use:", used_bash)
print("marker in transcript:", saw_marker)
print("result text:", result_text[:1000] or "<empty>")

failures = []
if not used_bash:
    failures.append("claude never issued a Bash tool_use")
if not saw_marker:
    failures.append(f"the marker file {marker} never appeared in the transcript")
if not result_text.strip():
    failures.append("claude produced an empty final result")
if marker not in result_text:
    print(f"WARN: the final answer does not repeat {marker}")

for failure in failures:
    print("FAIL:", failure, file=sys.stderr)
raise SystemExit(1 if failures else 0)
PY_STREAM
}

echo
echo "===== stage: claude -p DIRECT against vLLM /v1/messages (HARD GATE) ====="
claude_roundtrip "http://${NODE0}:${API_PORT}" "${SERVED_MODEL}" "${RUN_DIR}/claude-direct.jsonl" \
    || { echo "FAIL: claude direct against vLLM native /v1/messages" >&2; exit 1; }
echo "PASS: claude drove a Bash tool call straight against vLLM /v1/messages"

echo
echo "===== stage: claude -p through LiteLLM (soft) ====="
litellm_result=SKIPPED
if (( LITELLM_OK )); then
    if claude_roundtrip "http://127.0.0.1:${LITELLM_PORT}" "${CLAUDE_MODEL}" "${RUN_DIR}/claude-litellm.jsonl"; then
        litellm_result=PASS
    else
        litellm_result=FAIL
        echo "WARN: claude through LiteLLM failed"
    fi
fi

echo "summary: direct=/v1/messages PASS, litellm ${litellm_result}"
CLIENT

chmod +x "${SERVE_SCRIPT}" "${CLIENT_SCRIPT}"

rm -f "${CLIENT_DONE}"

srun_step=(--nodes=1 --ntasks=1 --ntasks-per-node=1 --exclusive --export=ALL)

stage "starting the vLLM server on ${NODE0}"
srun "${srun_step[@]}" --nodelist="${NODE0}" --environment="${INFERENCE_EDF}" \
    bash "${SERVE_SCRIPT}" >"${RUN_DIR}/vllm-step.log" 2>&1 &
SERVER_STEP_PID=$!

cleanup_steps() {
    touch "${CLIENT_DONE}"
    if kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
        # The sentinel is the polite exit; give it a moment before Slurm cancels the step.
        sleep 10
        kill "${SERVER_STEP_PID}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
}
trap cleanup_steps EXIT INT TERM

stage "waiting for http://${NODE0}:${API_PORT}/v1/models (up to ${READY_TIMEOUT}s)"
deadline=$((SECONDS + READY_TIMEOUT))
until curl -fsS --max-time 10 "http://${NODE0}:${API_PORT}/v1/models" -o "${RUN_DIR}/models.json"; do
    kill -0 "${SERVER_STEP_PID}" 2>/dev/null || fail "the vLLM step exited before the API came up"
    (( SECONDS < deadline )) || fail "the API did not answer within ${READY_TIMEOUT}s"
    sleep 10
done
cat "${RUN_DIR}/models.json"
pass "vLLM serves ${SERVED_MODEL} on ${NODE0}:${API_PORT}"
if [[ -s "${RUN_DIR}/parsers.env" ]]; then
    cat "${RUN_DIR}/parsers.env"
fi

stage "client roundtrip on ${NODE1}"
set +e
srun "${srun_step[@]}" --nodelist="${NODE1}" --environment="${AGENT_EDF}" bash "${CLIENT_SCRIPT}"
client_status=$?
set -e

touch "${CLIENT_DONE}"
(( client_status == 0 )) || fail "the client stage exited ${client_status}"

printf '\n===== CLAUDE ROUNDTRIP PASSED =====\n'
printf 'logs: %s\n' "${RUN_DIR}"
