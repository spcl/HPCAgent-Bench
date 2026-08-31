#!/usr/bin/env bash
# Does AITER's master switch pay for itself on this model, and is the output still correct?
#
# Two legs against one node, everything but VLLM_ROCM_USE_AITER held fixed: serve, grade the
# long-context retrieval gate, run the agent-shaped saturation probe, tear down, repeat. The
# accuracy leg is not optional -- 604789 read 78.5 tok/s with the master on and was silently
# corrupt at long context (aiter #1455 signature), so tok/s alone cannot clear a serving change.
#
#   sbatch probe-aiter-ab.sbatch                                  # qwen38 on 0.27.1
#   MODEL_REPO=openai/gpt-oss-120b INFERENCE_EDF=rocm723-vllm-0.23.0-pytorch211-ofi \
#     EXTRA_ARGS="--dtype bfloat16 ..." sbatch probe-aiter-ab.sbatch
#SBATCH --job-name=probe-aiter-ab
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=logs/probe-aiter-ab-%j.out
#SBATCH --error=logs/probe-aiter-ab-%j.err

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
FAST_SCRATCH="/iopsstor/scratch/cscs/${USER}"
[[ -d "${FAST_SCRATCH}" ]] || FAST_SCRATCH="${SCRATCH}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export RUN_DIR="${SCRATCH}/aiter-ab/${SLURM_JOB_ID}"
export HF_HOME="${HF_HOME:-${FAST_SCRATCH}/hf}"
export FAST_SCRATCH SUBMIT_DIR

INFERENCE_EDF="${INFERENCE_EDF:-rocm723-vllm-0.27.1-pytorch211-ofi}"
export MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.8-27B-FP8}"
export SERVED_MODEL="${SERVED_MODEL:-optarena-vllm}"
export API_PORT="${API_PORT:-8000}"
# The campaign's own serve line for this model, minus host/port/TP which are set below.
export EXTRA_ARGS="${EXTRA_ARGS:---dtype auto --load-format safetensors --safetensors-load-strategy prefetch --generation-config auto --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --max-model-len 262144 --gpu-memory-utilization 0.70 --max-num-seqs 128 --language-model-only}"
READY_TIMEOUT="${READY_TIMEOUT:-2400}"
PROBE_STREAMS="${PROBE_STREAMS:-12}"
PROBE_DURATION="${PROBE_DURATION:-300}"
export INFERENCE_EDF

mkdir -p "${RUN_DIR}" logs
HOST="$(hostname)"
# No /v1 suffix: the probe clients append their own path, as the judge and agents do.
BASE="http://${HOST}:${API_PORT}"
printf 'job:   %s\nmodel: %s\nimage: %s\nlogs:  %s\n\n' \
    "${SLURM_JOB_ID}" "${MODEL_REPO}" "${INFERENCE_EDF}" "${RUN_DIR}"

wait_ready() {  # wait_ready <log> <pid>
    local deadline=$(( $(date +%s) + READY_TIMEOUT ))
    until python3 -c "import sys,urllib.request; urllib.request.urlopen('${BASE}/v1/models', timeout=10)" 2>/dev/null; do
        kill -0 "$2" 2>/dev/null || { echo "FAIL: server exited before the API came up"; tail -40 "$1"; return 1; }
        (( $(date +%s) < deadline )) || { echo "FAIL: API not ready in ${READY_TIMEOUT}s"; tail -40 "$1"; return 1; }
        sleep 15
    done
}

leg() {  # leg <0|1>
    local aiter="$1" log="${RUN_DIR}/vllm-aiter${1}.log"
    printf '\n===== leg: VLLM_ROCM_USE_AITER=%s =====\n' "${aiter}"
    AITER="${aiter}" srun --nodes=1 --ntasks=1 --exclusive --export=ALL \
        --environment="${INFERENCE_EDF}" bash -c '
        set -Eeuo pipefail
        export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
        export VLLM_CACHE_ROOT="${FAST_SCRATCH}/vllm-cache"
        export TRITON_CACHE_DIR="${FAST_SCRATCH}/triton-cache"
        export VLLM_ROCM_USE_AITER="${AITER}"
        export AITER_JIT_DIR="${FAST_SCRATCH}/aiter-jit/${INFERENCE_EDF}"
        mkdir -p "${AITER_JIT_DIR}" "${TRITON_CACHE_DIR}"
        # Serve the resolved snapshot PATH, as run_cluster.sh does. Handed a bare repo id, vLLM
        # resolves it itself and refuses gpt-oss: the cached snapshot omits the original/ and
        # metal/ format copies nothing here serves, and that reads as IncompleteSnapshotError.
        model_path="$(python3 - <<PYEOF | tail -n 1
import os

from huggingface_hub import snapshot_download

print(snapshot_download(repo_id=os.environ["MODEL_REPO"], local_files_only=True,
                        ignore_patterns=["original/*", "metal/*"]))
PYEOF
)"
        test -d "${model_path}"
        read -r -a extra <<<"${EXTRA_ARGS}"
        exec vllm serve "${model_path}" --served-model-name "${SERVED_MODEL}" \
            --tensor-parallel-size 4 --host 0.0.0.0 --port "${API_PORT}" "${extra[@]}"
    ' >"${log}" 2>&1 &
    local pid=$!
    wait_ready "${log}" "${pid}" || { kill "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true; return 1; }

    echo "--- which kernels vllm picked ---"
    grep -aiE "aiter|Using default|sub-optimal|attention backend|Config not found" "${log}" | head -12 || true

    echo "--- accuracy gate ---"
    python3 "${SUBMIT_DIR}/accuracy-gate.py" --base "${BASE}" --model "${SERVED_MODEL}" \
        | tee "${RUN_DIR}/accuracy-aiter${aiter}.txt" || true

    echo "--- saturation probe (streams=${PROBE_STREAMS}, ${PROBE_DURATION}s) ---"
    python3 "${SUBMIT_DIR}/agentlike-probe.py" --base "${BASE}" --model "${SERVED_MODEL}" \
        --streams "${PROBE_STREAMS}" --duration "${PROBE_DURATION}" \
        | tee "${RUN_DIR}/throughput-aiter${aiter}.txt" || true

    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    sleep 30
}

leg 0 || echo "leg 0 did not complete"
leg 1 || echo "leg 1 did not complete"

printf '\n===== summary =====\n'
for a in 0 1; do
    printf 'aiter=%s  %s | %s\n' "${a}" \
        "$(grep -h '^ACCURACY' "${RUN_DIR}/accuracy-aiter${a}.txt" 2>/dev/null || echo 'ACCURACY n/a')" \
        "$(grep -h '^FINAL' "${RUN_DIR}/throughput-aiter${a}.txt" 2>/dev/null || echo 'FINAL n/a')"
done
