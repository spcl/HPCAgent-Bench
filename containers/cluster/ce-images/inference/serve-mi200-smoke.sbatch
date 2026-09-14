#!/usr/bin/env bash
# Private SGLang serving smoke on one mi200 node (8x MI250X, gfx90a): Qwen3.8-27B BF16, one fresh
# server per leg in LEGS, each "tp<N>:<mem-fraction-static>".
#
# PRIVATE BY CONSTRUCTION:
#   * --host 127.0.0.1 on every leg, last on the command line; nothing off this node can connect.
#   * The API key comes from KEY_FILE (default ~/.config/optarena/mi200-endpoint.key), refused unless
#     mode 600 and owned by you. It reaches sglang only through --config <yaml>, never argv.
#   * The run dir is mode 700 with scratch's default ACLs stripped. It holds the yaml, the curl header
#     file, the in-container key file and sglang's server log, which prints server_args incl. api_key.
#   * The key is never printed; the three secret files are deleted when the script exits.
#
#   umask 077; mkdir -p ~/.config/optarena; openssl rand -hex 32 > ~/.config/optarena/mi200-endpoint.key
#   cd <optarena checkout or worktree>
#   sbatch containers/cluster/ce-images/inference/serve-mi200-smoke.sbatch
#   LEGS="tp4:0.80" EDF=<candidate.toml> sbatch containers/cluster/ce-images/inference/serve-mi200-smoke.sbatch
#   DRY_RUN=1 bash containers/cluster/ce-images/inference/serve-mi200-smoke.sbatch    # checks, argv, no launch
#
# Per leg: wait on /health (auth-exempt), record the KV/Mamba allocator lines, assert 401 without the
# key and 200 with it, run verify-tools-reasoning.py and a concurrent load probe with the key, stop.
# Lives beside smoke-kimi-*.sbatch: a standalone smoke that owns its launch line, not a campaign role.
#SBATCH --job-name=serve-mi200-smoke
#SBATCH --partition=mi200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/serve-mi200-smoke-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/serve-mi200-smoke-%j.out

set -Eeuo pipefail
ulimit -c 0

refuse() {
    printf 'serve-mi200-smoke: refusing: %s\n' "$*" >&2
    exit 2
}

DRY_RUN="${DRY_RUN:-0}"
[[ -n "${SLURM_JOB_ID:-}" || "${DRY_RUN}" == 1 ]] || refuse "submit with sbatch, or DRY_RUN=1 to check without launching"
[[ -z "${SLURM_JOB_ID:-}" || "${SLURM_JOB_PARTITION:-}" == mi200 ]] || refuse "partition '${SLURM_JOB_PARTITION:-}' is not mi200"

# Under sbatch BASH_SOURCE is the spooled copy; the checkout is where the job was submitted from.
REPO="${REPO:-$(git -C "${SLURM_SUBMIT_DIR:-${PWD}}" rev-parse --show-toplevel 2>/dev/null || true)}"
GATE="${REPO}/containers/cluster/ce-images/inference/verify-tools-reasoning.py"
[[ -n "${REPO}" && -f "${GATE}" ]] || refuse "no checkout: set REPO or submit from inside one"

KEY_FILE="${KEY_FILE:-${HOME}/.config/optarena/mi200-endpoint.key}"
EDF="${EDF:-sglang-mi200-latest}"
LEGS="${LEGS:-tp4:0.80 tp4:0.88 tp8:0.80}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.8-27B}"
SERVED_MODEL="${SERVED_MODEL:-optarena-vllm}"
API_PORT="${API_PORT:-30000}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
LOAD_CONCURRENCY="${LOAD_CONCURRENCY:-16}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
# Operator words placed before --host/--port; they may not name the host, port, key or config.
EXTRA_ARGS="${EXTRA_ARGS:-}"
FAST_SCRATCH="/iopsstor/scratch/cscs/${USER:-$(id -un)}"
[[ -d "${FAST_SCRATCH}" ]] || FAST_SCRATCH="${SCRATCH:?set SCRATCH}"
HF_HOME="${HF_HOME:-${FAST_SCRATCH}/hf}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH:?set SCRATCH}/inference-server/mi200-smoke}"
RUN_DIR="${RUN_ROOT}/${SLURM_JOB_ID:-dry-$$}"
SERVE_CPUS="${SLURM_CPUS_ON_NODE:-128}"
SERVER_PID=""
MONITOR_PID=""

# Sets API_KEY, unexported, so srun --export=ALL never carries it.
read_key_file() {
    local mode
    [[ -f "${KEY_FILE}" ]] || refuse "no key file ${KEY_FILE}; create it as the header shows"
    mode="$(stat -c %a -- "${KEY_FILE}")"
    [[ "${mode}" == 600 ]] || refuse "key file ${KEY_FILE} is mode ${mode}, not 600"
    [[ "$(stat -c %u -- "${KEY_FILE}")" == "$(id -u)" ]] || refuse "key file ${KEY_FILE} is not owned by $(id -un)"
    API_KEY="$(tr -d '\n' < "${KEY_FILE}")"
    [[ ${#API_KEY} -ge 32 ]] || refuse "key in ${KEY_FILE} is shorter than 32 characters"
    [[ "${API_KEY}" =~ ^[A-Za-z0-9._~+/=-]+$ ]] || refuse "key in ${KEY_FILE} has characters outside [A-Za-z0-9._~+/=-]"
}

validate_extra_args() {
    local word
    for word in ${EXTRA_ARGS}; do
        case "${word}" in
            --host*|--port*|--api*|--admin*|--config*) refuse "EXTRA_ARGS may not set ${word%%=*}" ;;
        esac
    done
}

parse_leg() {  # sets LEG_TP and LEG_FRAC
    [[ "$1" =~ ^tp(1|2|4|8):(0\.[0-9]+)$ ]] || refuse "leg '$1' is not tp<1|2|4|8>:<0.fraction>"
    LEG_TP="${BASH_REMATCH[1]}"
    LEG_FRAC="${BASH_REMATCH[2]}"
}

make_private_dir() {
    local acl
    (umask 077 && mkdir -p -- "$1")
    chmod 700 -- "$1"
    # Scratch default ACLs (group:csstaff r-x) are inherited and beat umask.
    if command -v setfacl >/dev/null 2>&1; then setfacl -b -- "$1" 2>/dev/null || true; fi
    [[ "$(stat -c %a -- "$1")" == 700 ]] || refuse "$1 is not mode 700"
    if command -v getfacl >/dev/null 2>&1; then
        acl="$(getfacl -cp -- "$1" 2>/dev/null || true)"
        if grep -qvE '^(user|group|other)::|^$' <<< "${acl}"; then refuse "$1 still carries extended ACL entries"; fi
    fi
}

write_secret() {  # write_secret <file> <line>: printf is a builtin, so the content never reaches argv
    (umask 077 && printf '%s\n' "$2" > "$1")
    chmod 600 -- "$1"
    [[ "$(stat -c %a -- "$1")" == 600 ]] || refuse "$1 is not mode 600"
}

server_argv() {  # server_argv <tp> <fraction> <port>: sets SERVER_ARGV
    # shellcheck disable=SC2206 # EXTRA_ARGS is an operator word list
    SERVER_ARGV=(/opt/venv/bin/python3 -m sglang.launch_server
        --config "${RUN_DIR}/sglang-auth.yaml"
        --model-path "${MODEL_REPO}" --served-model-name "${SERVED_MODEL}"
        --tp-size "$1" --mem-fraction-static "$2"
        --attention-backend triton --disable-custom-all-reduce
        --chat-template "${REPO}/experiments/chat-template-qwen38.jinja" --trust-remote-code --language-only
        --watchdog-timeout 1800 --context-length 262144 --mamba-full-memory-ratio 0.5 --max-running-requests 128
        --enable-metrics --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-cache-report
        ${EXTRA_ARGS}
        --host 127.0.0.1 --port "$3")
}

without_key() {  # filter stdin: drop any line carrying the key (sglang logs server_args)
    grep -avFf "${RUN_DIR}/api.key" || true
}

wait_ready() {  # wait_ready <port> <dir>
    local deadline code
    deadline=$(($(date +%s) + READY_TIMEOUT))
    while :; do
        code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$1/health" || true)"
        [[ "${code}" == 200 ]] && return 0
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "server step exited before /health answered" >&2
            tail -40 "$2/server.log" 2>/dev/null | without_key >&2
            return 1
        fi
        if (($(date +%s) > deadline)); then
            echo "not ready within ${READY_TIMEOUT}s" >&2
            return 1
        fi
        sleep 15
    done
}

record_memory() {  # record_memory <dir>
    grep -aE 'KV Cache is allocated|max_total_num_tokens|Mamba Cache' "$1/server.log" 2>/dev/null \
        | without_key > "$1/memory.txt" || true
    cat "$1/memory.txt"
}

http_status() {  # http_status <port> <body> [header-file]
    local auth=()
    [[ $# -lt 3 ]] || auth=(-H "@$3")
    curl -s -o /dev/null -w '%{http_code}' --max-time 600 -H 'Content-Type: application/json' "${auth[@]}" \
        --data-binary "@$2" "http://127.0.0.1:$1/v1/chat/completions" || true
}

auth_checks() {  # auth_checks <port> <dir>
    local closed open
    closed="$(http_status "$1" "${RUN_DIR}/hello.json")"
    open="$(http_status "$1" "${RUN_DIR}/hello.json" "${RUN_DIR}/auth.header")"
    printf 'unauthenticated POST: %s (want 401)\nauthenticated POST:   %s (want 200)\n' "${closed}" "${open}" \
        | tee "$2/auth.txt"
    [[ "${closed}" == 401 && "${open}" == 200 ]]
}

tools_gate() {  # tools_gate <port> <dir>
    local rc=0
    srun --overlap --ntasks=1 --cpus-per-task=4 --export=ALL --environment="${EDF}" \
        /opt/venv/bin/python3 "${GATE}" --base "http://127.0.0.1:$1" --model "${SERVED_MODEL}" \
        --api-key-file "${RUN_DIR}/api.key" --reasoning-effort "${REASONING_EFFORT}" \
        > "$2/tools-reasoning.out" 2>&1 || rc=$?
    cat "$2/tools-reasoning.out"
    return "${rc}"
}

load_probe() {  # load_probe <port> <dir>: LOAD_CONCURRENCY authenticated chats, 256 tokens each
    local i start wall ok tokens pids=()
    start="$(date +%s.%N)"
    for ((i = 0; i < LOAD_CONCURRENCY; i++)); do
        curl -s -o "$2/load-${i}.json" -w '%{http_code}\n' --max-time 1800 -H 'Content-Type: application/json' \
            -H "@${RUN_DIR}/auth.header" --data-binary "@${RUN_DIR}/load.json" \
            "http://127.0.0.1:$1/v1/chat/completions" > "$2/load-${i}.status" 2>/dev/null &
        pids+=("$!")
    done
    wait "${pids[@]}" || true
    wall="$(awk -v a="${start}" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')"
    ok="$(cat "$2"/load-*.status | grep -cx 200 || true)"
    tokens="$(cat "$2"/load-*.json 2>/dev/null | grep -oE '"completion_tokens": *[0-9]+' | grep -oE '[0-9]+$' \
        | awk '{ s += $1 } END { print s + 0 }' || true)"
    printf 'load: %s/%s requests 200, %s completion tokens in %ss (%s tok/s)\n' "${ok}" "${LOAD_CONCURRENCY}" \
        "${tokens}" "${wall}" "$(awk -v t="${tokens}" -v w="${wall}" 'BEGIN { printf "%.1f", w > 0 ? t / w : 0 }')" \
        | tee "$2/load.txt"
    [[ "${ok}" == "${LOAD_CONCURRENCY}" ]]
}

stop_server() {
    local i
    [[ -n "${SERVER_PID}" ]] || return 0
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    for ((i = 0; i < 60; i++)); do
        kill -0 "${SERVER_PID}" 2>/dev/null || break
        sleep 2
    done
    kill -KILL "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
}

run_leg() {  # run_leg <index> <spec>
    local port dir failed=0
    parse_leg "$2"
    port=$((API_PORT + $1))
    dir="${RUN_DIR}/leg$1-${2/:/-}"
    mkdir -p -- "${dir}"
    server_argv "${LEG_TP}" "${LEG_FRAC}" "${port}"
    printf '\n===== leg %s: %s tp=%s mem-fraction-static=%s port=%s =====\n' "$1" "$2" "${LEG_TP}" "${LEG_FRAC}" "${port}"
    printf 'argv: %s\n' "${SERVER_ARGV[*]}"
    [[ "${DRY_RUN}" != 1 ]] || return 0
    srun --ntasks=1 --cpus-per-task="${SERVE_CPUS}" --export=ALL --environment="${EDF}" \
        --output="${dir}/server.log" --error="${dir}/server.log" \
        env SGLANG_USE_AITER=0 SGLANG_SET_CPU_AFFINITY=0 HF_HUB_OFFLINE=1 "HF_HOME=${HF_HOME}" \
        "TRITON_CACHE_DIR=${FAST_SCRATCH}/triton-cache-sglang-mi200" "${SERVER_ARGV[@]}" &
    SERVER_PID=$!
    wait_ready "${port}" "${dir}" || failed=1
    record_memory "${dir}"
    if ((failed == 0)); then
        auth_checks "${port}" "${dir}" || failed=1
        tools_gate "${port}" "${dir}" || failed=1
        load_probe "${port}" "${dir}" || failed=1
    fi
    stop_server
    # Released GPU memory before the next leg allocates.
    sleep 30
    return "${failed}"
}

cleanup() {
    stop_server
    [[ -z "${MONITOR_PID}" ]] || kill -TERM "${MONITOR_PID}" 2>/dev/null || true
    rm -f -- "${RUN_DIR}/sglang-auth.yaml" "${RUN_DIR}/api.key" "${RUN_DIR}/auth.header"
}

read_key_file
validate_extra_args
for spec in ${LEGS}; do parse_leg "${spec}"; done
[[ -d "${HF_HOME}/hub/models--${MODEL_REPO//\//--}" ]] || refuse "no ${MODEL_REPO} under ${HF_HOME}/hub"

make_private_dir "${RUN_DIR}"
trap cleanup EXIT
write_secret "${RUN_DIR}/sglang-auth.yaml" "api-key: \"${API_KEY}\""
write_secret "${RUN_DIR}/api.key" "${API_KEY}"
write_secret "${RUN_DIR}/auth.header" "Authorization: Bearer ${API_KEY}"
printf '{"model": "%s", "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 16}\n' \
    "${SERVED_MODEL}" > "${RUN_DIR}/hello.json"
filler="$(printf 'The kernel iterates over the array and accumulates into the result buffer. %.0s' {1..200})"
printf '{"model": "%s", "messages": [{"role": "user", "content": "%s Explain loop tiling."}], "max_tokens": 256, "ignore_eos": true}\n' \
    "${SERVED_MODEL}" "${filler}" > "${RUN_DIR}/load.json"

cat <<EOF
job:      ${SLURM_JOB_ID:-dry run}
model:    ${MODEL_REPO} as "${SERVED_MODEL}" (HF_HOME=${HF_HOME})
image:    ${EDF}
legs:     ${LEGS}
run dir:  ${RUN_DIR} (mode $(stat -c %a -- "${RUN_DIR}"))
config:   ${RUN_DIR}/sglang-auth.yaml (mode $(stat -c %a -- "${RUN_DIR}/sglang-auth.yaml"))
EOF

if [[ "${DRY_RUN}" != 1 ]]; then
    ROLE=vllm OUT_DIR="${RUN_DIR}/monitor" INTERVAL=10 bash "${REPO}/experiments/node_monitor.sh" &
    MONITOR_PID=$!
fi

FAILURES=()
index=0
for spec in ${LEGS}; do
    run_leg "${index}" "${spec}" || FAILURES+=("${spec}")
    index=$((index + 1))
done

if ((${#FAILURES[@]} > 0)); then
    printf '\nSMOKE FAILED legs: %s\nlogs: %s\n' "${FAILURES[*]}" "${RUN_DIR}" >&2
    exit 1
fi
printf '\nSMOKE PASSED legs: %s\nlogs: %s\n' "${LEGS}" "${RUN_DIR}"
