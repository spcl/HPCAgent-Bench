#!/usr/bin/env bash
# What does qwen38 actually decode at on SGLang?
#
# It has never been measured. 610139 served the model fine and then returned 500 on every
# /v1/messages: SGLang's Anthropic adapter rewrites the effort level to "max" and Qwen3.8's chat
# template raises on anything outside (low, medium, xhigh). That is an adapter bug, not a
# throughput verdict -- and the probe clients here speak /v1/completions and /v1/chat/completions,
# so they never touch the adapter and the number can be had regardless.
#
# Worth the node because vLLM gives this model 18.8 tok/s on the same 12-stream probe that gets
# 301.8 out of gpt-oss (610165 vs 610183), and SGLang supports the hybrid backbone natively
# (linear_attn_backend, mamba_backend, max_mamba_cache_size are first-class server args).
#
# mem-fraction-static is 0.18, not the usual 0.8-0.9: MI300A reports is_integrated, so SGLang reads
# node-wide host RAM as EACH rank's free memory and four ranks at 0.42 asked for 1.7x the node
# (610137 host-OOM). Qwen3.8 is a VLM checkpoint, so the value is derated a further 0.85 on the way
# in. The text-only switch is --language-only; --language-model-only is a different,
# standalone-encoder flag this architecture rejects outright (610189).
#SBATCH --job-name=probe-q38-sglang
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/probe-q38-sglang-%j.out
#SBATCH --error=logs/probe-q38-sglang-%j.err

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
export MEM_FRACTION="${MEM_FRACTION:-0.18}"
export CONTEXT_LEN="${CONTEXT_LEN:-262144}"
export SCRATCH FAST_SCRATCH
EDF="${INFERENCE_EDF:-sglang-rocm-mi30x}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
PROBE_STREAMS="${PROBE_STREAMS:-12}"
PROBE_DURATION="${PROBE_DURATION:-300}"
# aiter is OFF by default here, and that is the whole point of this probe. SGLang auto-selects the
# aiter attention backend, whose kernels JIT-compile on the FIRST request -- behind a build that
# outlives SGLang's own 600 s warmup read timeout, so the server kills itself before serving one
# token. 610217, 610220 and 610225 all died exactly there, with the cache measurably warmer each
# time (791 s -> 332 s to respond) and no effect on the outcome. Pinning triton removes aiter from
# the path and measures the ENGINE, which is the question. Set SGLANG_USE_AITER=1 with an empty
# ATTENTION_BACKEND to go back to the aiter path once a prebuild exists.
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-0}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
export ATTENTION_BACKEND

mkdir -p "${RUN_DIR}" logs
HOST="$(hostname)"; BASE="http://${HOST}:${API_PORT}"
printf 'job:   %s\nmodel: %s on sglang tp=4, mem-fraction %s\nlogs:  %s\n\n' \
    "${SLURM_JOB_ID}" "${MODEL_REPO}" "${MEM_FRACTION}" "${RUN_DIR}"

srun --nodes=1 --ntasks=1 --exclusive --export=ALL --environment="${EDF}" bash -c '
  set -Eeuo pipefail
  export SGLANG_SET_CPU_AFFINITY=0 HF_HUB_OFFLINE=1
  CACHE_ROOT="${JIT_CACHE_ROOT:-${SCRATCH}/.jit-cache}/sglang-rocm-mi30x"
  export HOME="${CACHE_ROOT}/home" XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
  export AITER_JIT_DIR="${CACHE_ROOT}/aiter" TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
  mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${AITER_JIT_DIR}" "${TRITON_CACHE_DIR}"
  # if, not `[[ ]] &&` -- a false test is a nonzero status and `set -e` would abort right here.
  backend=()
  if [[ -n "${ATTENTION_BACKEND}" ]]; then backend=(--attention-backend "${ATTENTION_BACKEND}"); fi
  exec /opt/venv/bin/python3 -m sglang.launch_server \
    --model-path "${MODEL_REPO}" --served-model-name "${SERVED_MODEL}" "${backend[@]}" \
    --trust-remote-code --language-only --tp-size 4 \
    --mem-fraction-static "${MEM_FRACTION}" --context-length "${CONTEXT_LEN}" \
    --max-running-requests 128 --watchdog-timeout 1800 --enable-metrics \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --host 0.0.0.0 --port "${API_PORT}"' >"${RUN_DIR}/sglang.log" 2>&1 &
PID=$!

started=$(date +%s); deadline=$(( started + READY_TIMEOUT ))
until python3 -c "import urllib.request; urllib.request.urlopen('${BASE}/v1/models', timeout=10)" 2>/dev/null; do
    kill -0 "${PID}" 2>/dev/null || { echo "FAIL: server exited before the API came up"; tail -40 "${RUN_DIR}/sglang.log"; exit 1; }
    (( $(date +%s) < deadline )) || { echo "FAIL: API not ready in ${READY_TIMEOUT}s"; tail -40 "${RUN_DIR}/sglang.log"; exit 1; }
    sleep 15
done
echo "ready in $(( $(date +%s) - started ))s"

echo "=== backends sglang picked ==="
# `|| true`, and it is load-bearing: head closes the pipe, the upstream greps take SIGPIPE, and
# under `set -o pipefail` that aborts the whole probe one line after the server came up (610207).
grep -aiE "attention backend|aiter|linear_attn|mamba|kv cache dtype|page" "${RUN_DIR}/sglang.log" \
    | grep -av "server_args=" | head -12 || true

echo "=== accuracy gate ==="
python3 "${SUBMIT_DIR}/accuracy-gate.py" --base "${BASE}" --model "${SERVED_MODEL}" | tee "${RUN_DIR}/accuracy.txt" || true
echo "=== saturation probe ==="
python3 "${SUBMIT_DIR}/agentlike-probe.py" --base "${BASE}" --model "${SERVED_MODEL}" \
    --streams "${PROBE_STREAMS}" --duration "${PROBE_DURATION}" | tee "${RUN_DIR}/throughput.txt" || true

kill "${PID}" 2>/dev/null || true; wait "${PID}" 2>/dev/null || true
