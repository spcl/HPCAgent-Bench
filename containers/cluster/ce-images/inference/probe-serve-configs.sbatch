#!/usr/bin/env bash
# Serve one model under several configurations back to back, grading each the same way.
#
# qwen38 decodes at 18.8 tok/s where oss120b reaches 301.8 on the identical 12-stream probe
# (610165 vs 610183), so the gap is configuration, not size. The vLLM recipe for this model names
# three things our serve line omits -- fp8 KV, the built-in MTP draft head, prefix caching -- and
# the 610145 log adds a fourth: the ROCm custom paged-attention kernel declines and the full
# attention layers fall back to Triton, which is the path aiter's MHA would take instead.
#
# Every leg holds the runner fixed and changes one thing, and every leg is graded on the
# long-context retrieval gate before its throughput number is allowed to count.
#
#   PROFILE=qwen38 sbatch probe-serve-configs.sbatch
#   PROFILE=qwen38 INFERENCE_EDF=rocm723-vllm-0.23.0-pytorch211-ofi sbatch probe-serve-configs.sbatch
#SBATCH --job-name=probe-serve-configs
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=logs/probe-serve-configs-%j.out
#SBATCH --error=logs/probe-serve-configs-%j.err

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
FAST_SCRATCH="/iopsstor/scratch/cscs/${USER}"
[[ -d "${FAST_SCRATCH}" ]] || FAST_SCRATCH="${SCRATCH}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export RUN_DIR="${SCRATCH}/serve-configs/${SLURM_JOB_ID}"
export HF_HOME="${HF_HOME:-${FAST_SCRATCH}/hf}"
export FAST_SCRATCH SUBMIT_DIR SCRATCH

PROFILE="${PROFILE:-qwen38}"
INFERENCE_EDF="${INFERENCE_EDF:-rocm723-vllm-0.27.1-pytorch211-ofi}"
export API_PORT="${API_PORT:-8000}"
export SERVED_MODEL="${SERVED_MODEL:-optarena-vllm}"
# Generous on purpose: 610165's aiter leg was killed at exactly 2400 s while still starting, which
# measured the timeout rather than the model. A cold start has to be allowed to finish and be timed.
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
PROBE_STREAMS="${PROBE_STREAMS:-12}"
PROBE_DURATION="${PROBE_DURATION:-300}"

# The campaign's own serve line for the profile, minus host/port/TP, plus the legs to compare.
# Fields: name | extra env (space separated K=V) | extra serve args appended to BASE_ARGS.
case "${PROFILE}" in
qwen38)
    export MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.8-27B-FP8}"
    BASE_ARGS="--dtype auto --load-format safetensors --safetensors-load-strategy prefetch"
    BASE_ARGS+=" --generation-config auto --enable-auto-tool-choice --tool-call-parser qwen3_xml"
    BASE_ARGS+=" --reasoning-parser qwen3 --max-model-len 262144 --gpu-memory-utilization 0.70"
    BASE_ARGS+=" --max-num-seqs 128 --language-model-only"
    MTP='--speculative-config {"method":"mtp","num_speculative_tokens":3}'
    # Every aiter component defaults ON but is gated by the master switch, so a component is
    # selected by turning the master on and the others explicitly off (probe 602360). The master
    # alone is the upper bound AND the thing that took >40 min to become ready in 610165, so the
    # granular legs are what tell us whether the win can be had without that cost.
    OFF="VLLM_ROCM_USE_AITER_LINEAR=0 VLLM_ROCM_USE_AITER_MOE=0 VLLM_ROCM_USE_AITER_RMSNORM=0"
    OFF+=" VLLM_ROCM_USE_AITER_FP8BMM=0 VLLM_ROCM_USE_AITER_FP4BMM=0"
    OFF+=" VLLM_ROCM_USE_AITER_TRITON_GEMM=0 VLLM_ROCM_USE_AITER_CUSTOM_AR=0"
    MHA_LINEAR="VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=1 VLLM_ROCM_USE_AITER_LINEAR=1"
    MHA_LINEAR+=" VLLM_ROCM_USE_AITER_FP8BMM=1 VLLM_ROCM_USE_AITER_MOE=0"
    MHA_LINEAR+=" VLLM_ROCM_USE_AITER_FP4BMM=0 VLLM_ROCM_USE_AITER_CUSTOM_AR=0"
    LEGS=(
        "baseline||"
        "fp8kv||--kv-cache-dtype fp8"
        "mtp||${MTP}"
        "fp8kv+mtp||--kv-cache-dtype fp8 ${MTP}"
        "aiter-mha|VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=1 ${OFF}|--kv-cache-dtype fp8 ${MTP}"
        "aiter-mha-linear|${MHA_LINEAR}|--kv-cache-dtype fp8 ${MTP}"
        "aiter-all|VLLM_ROCM_USE_AITER=1|--kv-cache-dtype fp8 ${MTP}"
    )
    ;;
oss120b)
    export MODEL_REPO="${MODEL_REPO:-openai/gpt-oss-120b}"
    BASE_ARGS="--dtype bfloat16 --load-format safetensors --safetensors-load-strategy prefetch"
    BASE_ARGS+=" --generation-config auto --enable-auto-tool-choice --tool-call-parser openai"
    BASE_ARGS+=" --reasoning-parser openai_gptoss --max-model-len 131072"
    BASE_ARGS+=" --gpu-memory-utilization 0.70 --max-num-seqs 128"
    OFF="VLLM_ROCM_USE_AITER_LINEAR=0 VLLM_ROCM_USE_AITER_MOE=0 VLLM_ROCM_USE_AITER_RMSNORM=0"
    OFF+=" VLLM_ROCM_USE_AITER_FP8BMM=0 VLLM_ROCM_USE_AITER_FP4BMM=0"
    OFF+=" VLLM_ROCM_USE_AITER_TRITON_GEMM=0 VLLM_ROCM_USE_AITER_CUSTOM_AR=0"
    LEGS=(
        "baseline||"
        "aiter-mha|VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=1 ${OFF}|"
        "aiter-all|VLLM_ROCM_USE_AITER=1|"
    )
    ;;
*)
    echo "FATAL: unknown PROFILE=${PROFILE}" >&2; exit 2 ;;
esac
export BASE_ARGS INFERENCE_EDF

mkdir -p "${RUN_DIR}" logs
HOST="$(hostname)"
BASE="http://${HOST}:${API_PORT}"
printf 'job:     %s\nprofile: %s\nmodel:   %s\nimage:   %s\nlogs:    %s\n\n' \
    "${SLURM_JOB_ID}" "${PROFILE}" "${MODEL_REPO}" "${INFERENCE_EDF}" "${RUN_DIR}"

wait_ready() {  # wait_ready <log> <pid> -> seconds to ready on stdout
    local started deadline now
    started="$(date +%s)"; deadline=$(( started + READY_TIMEOUT ))
    until python3 -c "import urllib.request; urllib.request.urlopen('${BASE}/v1/models', timeout=10)" 2>/dev/null; do
        kill -0 "$2" 2>/dev/null || { echo "FAIL: server exited before the API came up" >&2; tail -30 "$1" >&2; return 1; }
        now="$(date +%s)"
        (( now < deadline )) || { echo "FAIL: API not ready in ${READY_TIMEOUT}s" >&2; tail -30 "$1" >&2; return 1; }
        sleep 15
    done
    echo $(( $(date +%s) - started ))
}

leg() {  # leg <name> <env assignments> <extra serve args>
    local name="$1" leg_env="$2" leg_args="$3"
    local tag="${name//[^a-zA-Z0-9]/_}"
    local log="${RUN_DIR}/serve-${tag}.log"
    printf '\n===== leg: %s =====\n  env:  %s\n  args: %s\n' "${name}" "${leg_env:-<none>}" "${leg_args:-<none>}"
    LEG_ENV="${leg_env}" LEG_ARGS="${leg_args}" srun --nodes=1 --ntasks=1 --exclusive --export=ALL \
        --environment="${INFERENCE_EDF}" bash -c '
        set -Eeuo pipefail
        export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
        # One capstor root, keyed by image, exactly as run_cluster.sh does it -- HOME included,
        # because aiter falls back to expanduser("~")/.aiter for template ops.
        CACHE_ROOT="${JIT_CACHE_ROOT:-${SCRATCH}/.jit-cache}/${INFERENCE_EDF}"
        export HOME="${CACHE_ROOT}/home" XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
        export AITER_JIT_DIR="${CACHE_ROOT}/aiter" VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
        export TRITON_CACHE_DIR="${CACHE_ROOT}/triton" TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor"
        mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${AITER_JIT_DIR}" "${VLLM_CACHE_ROOT}" \
            "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
        for kv in ${LEG_ENV}; do export "${kv?}"; done
        model_path="$(python3 - <<PYEOF | tail -n 1
import os

from huggingface_hub import snapshot_download

print(snapshot_download(repo_id=os.environ["MODEL_REPO"], local_files_only=True,
                        ignore_patterns=["original/*", "metal/*"]))
PYEOF
)"
        test -d "${model_path}"
        read -r -a args <<<"${BASE_ARGS} ${LEG_ARGS}"
        exec vllm serve "${model_path}" --served-model-name "${SERVED_MODEL}" \
            --tensor-parallel-size 4 --host 0.0.0.0 --port "${API_PORT}" "${args[@]}"
    ' >"${log}" 2>&1 &
    local pid=$! ready
    if ! ready="$(wait_ready "${log}" "${pid}")"; then
        echo "${name}: DID NOT SERVE" | tee -a "${RUN_DIR}/summary.txt"
        kill "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true; sleep 20; return 0
    fi
    echo "  ready in ${ready}s"

    python3 "${SUBMIT_DIR}/accuracy-gate.py" --base "${BASE}" --model "${SERVED_MODEL}" \
        >"${RUN_DIR}/accuracy-${tag}.txt" 2>&1 || true
    python3 "${SUBMIT_DIR}/agentlike-probe.py" --base "${BASE}" --model "${SERVED_MODEL}" \
        --streams "${PROBE_STREAMS}" --duration "${PROBE_DURATION}" \
        >"${RUN_DIR}/throughput-${tag}.txt" 2>&1 || true
    printf '%-16s ready=%-5ss %s | %s\n' "${name}" "${ready}" \
        "$(grep -h '^ACCURACY' "${RUN_DIR}/accuracy-${tag}.txt" 2>/dev/null || echo 'ACCURACY n/a')" \
        "$(grep -h '^FINAL' "${RUN_DIR}/throughput-${tag}.txt" 2>/dev/null || echo 'FINAL n/a')" \
        | tee -a "${RUN_DIR}/summary.txt"

    kill "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true; sleep 20
}

for spec in "${LEGS[@]}"; do
    IFS='|' read -r leg_name leg_env leg_args <<<"${spec}"
    leg "${leg_name}" "${leg_env}" "${leg_args}"
done

printf '\n===== summary (%s on %s) =====\n' "${PROFILE}" "${INFERENCE_EDF}"
cat "${RUN_DIR}/summary.txt" 2>/dev/null || echo "no legs completed"
