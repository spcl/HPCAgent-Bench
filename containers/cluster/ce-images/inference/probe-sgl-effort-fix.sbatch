#!/usr/bin/env bash
# Attempt 2: serve qwen38 at xhigh from the SERVER, so no request can collapse it.
#
# anthropic/serving.py:624-632 sets chat_request.reasoning_effort ONLY when output_config.effort is
# present, and rewrites xhigh to max there because the OpenAI Literal has no xhigh -- which is why
# every /v1/messages call in 610139 returned 500 against Qwen3.8's template. A request that omits
# output_config never touches that field, and server_args.py:1438 says
# --default-chat-template-kwargs is "applied to every request when not overridden per-request".
# Two channels, so the level can be supplied on the one the adapter does not write to.
#
# Legs: with the default kwarg and without it, each probed across request shapes. The pairing is
# what distinguishes "the flag worked" from "this shape never sent an effort anyway".
#SBATCH --job-name=probe-sgl-effortfix
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/probe-sgl-effortfix-%j.out
#SBATCH --error=logs/probe-sgl-effortfix-%j.err

set -Eeuo pipefail
ulimit -c 0
SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
FAST_SCRATCH="/iopsstor/scratch/cscs/${USER}"
[[ -d "${FAST_SCRATCH}" ]] || FAST_SCRATCH="${SCRATCH}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export RUN_DIR="${SCRATCH}/serve-configs/${SLURM_JOB_ID}"
export HF_HOME="${HF_HOME:-${FAST_SCRATCH}/hf}"
export MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.8-27B-FP8}"
export SERVED_MODEL="${SERVED_MODEL:-optarena-vllm}"
export API_PORT="${API_PORT:-8000}"
export SCRATCH FAST_SCRATCH
EDF="${INFERENCE_EDF:-sglang-rocm-mi30x}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
mkdir -p "${RUN_DIR}" logs
HOST="$(hostname)"; BASE="http://${HOST}:${API_PORT}"
printf 'job:   %s\nmodel: %s\nlogs:  %s\n' "${SLURM_JOB_ID}" "${MODEL_REPO}" "${RUN_DIR}"

leg() {  # leg <name> <extra sglang args>
    local name="$1" extra="$2" log="${RUN_DIR}/sglang-${1}.log"
    printf '\n===== leg: %s =====\n  extra: %s\n' "${name}" "${extra:-<none>}"
    LEG_EXTRA="${extra}" srun --nodes=1 --ntasks=1 --exclusive --export=ALL \
        --environment="${EDF}" bash -c '
        set -Eeuo pipefail
        export SGLANG_USE_AITER=1 SGLANG_SET_CPU_AFFINITY=0 HF_HUB_OFFLINE=1
        CACHE_ROOT="${JIT_CACHE_ROOT:-${SCRATCH}/.jit-cache}/sglang-rocm-mi30x"
        export HOME="${CACHE_ROOT}/home" XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
        export AITER_JIT_DIR="${CACHE_ROOT}/aiter" TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
        mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${AITER_JIT_DIR}" "${TRITON_CACHE_DIR}"
        read -r -a extra <<<"${LEG_EXTRA}"
        exec /opt/venv/bin/python3 -m sglang.launch_server \
            --model-path "${MODEL_REPO}" --served-model-name "${SERVED_MODEL}" \
            --trust-remote-code --language-only --tp-size 4 --mem-fraction-static 0.18 \
            --context-length 262144 --max-running-requests 128 --watchdog-timeout 1800 \
            --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
            --host 0.0.0.0 --port "${API_PORT}" "${extra[@]}"' >"${log}" 2>&1 &
    local pid=$! deadline=$(( $(date +%s) + READY_TIMEOUT ))
    until python3 -c "import urllib.request; urllib.request.urlopen('${BASE}/v1/models', timeout=10)" 2>/dev/null; do
        kill -0 "${pid}" 2>/dev/null || { echo "  server exited before the API came up"; tail -25 "${log}"; return 0; }
        (( $(date +%s) < deadline )) || { echo "  API not ready in ${READY_TIMEOUT}s"; tail -25 "${log}"; kill "${pid}"; return 0; }
        sleep 15
    done
    python3 "${SUBMIT_DIR}/messages_effort_probe.py" --base "${BASE}" --model "${SERVED_MODEL}" \
        2>&1 | tee "${RUN_DIR}/messages-${name}.txt" || true
    echo "  --- what the template actually rendered as the effort ---"
    grep -aoiE "reasoning_effort[^,)]{0,40}|xhigh|'max'" "${log}" | sort | uniq -c | head -6 || true
    kill "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true; sleep 20
}

leg "no-default" ""
leg "default-xhigh" "--default-chat-template-kwargs {\"reasoning_effort\":\"xhigh\"}"

printf '\n===== attempt 2 verdict =====\n'
for f in "${RUN_DIR}"/messages-*.txt; do printf '\n-- %s\n' "$(basename "${f}")"; cat "${f}"; done
