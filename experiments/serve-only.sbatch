#!/usr/bin/env bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# Inference only: one OpenAI-compatible endpoint and nothing else. The payload is run_cluster.sh's
# own --vllm-node role, so this serves what a campaign serves, without the judge and agent roles.
#   ./serve-only.sbatch                      qwen38 on the nodes its env file asks for
#   MODEL=kimi27sglang ./serve-only.sbatch   another model (any .env.base-<MODEL>)
#   SUBMIT=0 ./serve-only.sbatch             print what would be submitted, submit nothing
# Run it from experiments/. It derives --nodes from the env file, so never pass --nodes yourself.
#SBATCH --job-name=serve-only
#SBATCH --partition=mi300
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/inference-server/slurm/serve-only-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/inference-server/slurm/serve-only-%j.out

set -euo pipefail

# Beverin's core_pattern is machine-global and drops a stub in the crashing process's CWD.
ulimit -c 0

# Under sbatch BASH_SOURCE is the spooled copy in /var/spool/slurmd, not this file.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SCRIPT_DIR="${CLUSTER_SCRIPT_DIR:-${SLURM_SUBMIT_DIR}}"
fi
export SCRIPT_DIR

MODEL="${MODEL:-qwen38}"
BASE_ENV="${SERVE_BASE_ENV:-${SCRIPT_DIR}/.env.base-${MODEL}}"
SERVE_ENV="${SERVE_ENV_FILE:-${SCRIPT_DIR}/serve-only.env}"
for env_file in "${BASE_ENV}" "${SERVE_ENV}"; do
    [[ -f "${env_file}" ]] || { echo "missing env file ${env_file}" >&2; exit 2; }
done

# Campaign values first, role split second: last assignment wins, as the submit scripts do it.
set -a
# shellcheck disable=SC1090
. "${BASE_ENV}"
# shellcheck disable=SC1090
. "${SERVE_ENV}"
set +a

INFERENCE_NODES="${INFERENCE_NODES:-1}"
INFERENCE_MODE="${INFERENCE_MODE:-replicas}"
VLLM_PORT="${VLLM_PORT:-8000}"
SERVED_MODEL="${VLLM_SERVED_MODEL:-optarena-vllm}"
# same precedence the campaign path uses, so one key covers both; 2400 could not hold a
# GLM-5.3 start, whose slowest stage alone reaches 4200 s before KV alloc and graph capture
READY_TIMEOUT="${VLLM_READY_TIMEOUT_SECONDS:-${AGENT_READY_TIMEOUT_SECONDS:-7200}}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT must name a writable directory}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    submit=(sbatch --nodes="${INFERENCE_NODES}"
        --export=ALL,MODEL="${MODEL}",CLUSTER_SCRIPT_DIR="${SCRIPT_DIR}"
        "${SCRIPT_DIR}/serve-only.sbatch")
    if [[ "${SUBMIT:-1}" != 1 ]]; then
        printf 'would serve %s on %s node(s)\n' "${MODEL}" "${INFERENCE_NODES}"
        printf '  env      %s + %s\n' "${BASE_ENV##*/}" "${SERVE_ENV##*/}"
        printf '  image    %s\n' "${INFERENCE_CE_ENV:-unset}"
        printf '  engine   %s (%s), port %s, model name %s\n' \
            "${INFERENCE_ENGINE:-vllm}" "${INFERENCE_MODE}" "${VLLM_PORT}" "${SERVED_MODEL}"
        printf '  weights  %s\n' "${VLLM_MODEL:-unset}"
        printf '  submit   %s\n' "${submit[*]}"
        exit 0
    fi
    mkdir -p "${RUN_ROOT}/slurm" "${SCRATCH:?}/inference-server/slurm"
    exec "${submit[@]}"
fi

mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if (( ${#nodes[@]} != INFERENCE_NODES )); then
    echo "allocation has ${#nodes[@]} nodes; ${BASE_ENV##*/} asks for ${INFERENCE_NODES}" >&2
    echo "submit with ./serve-only.sbatch, which derives --nodes from that file" >&2
    exit 2
fi

RUN_DIR="${RUN_ROOT}/${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
# The one file run_cluster.sh re-reads in the container, and the record of what this job served.
cat -- "${BASE_ENV}" "${SERVE_ENV}" >"${RUN_DIR}/serve.env"

export CLUSTER_ENV_FILE="${RUN_DIR}/serve.env"
export HPCAGENT_BENCH_REPO="${HPCAGENT_BENCH_REPO:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
export VLLM_MASTER_HOST="${nodes[0]}"
BASE_URL="http://${VLLM_MASTER_HOST}:${VLLM_PORT}"

cat <<EOF
job:     ${SLURM_JOB_ID}
model:   ${VLLM_MODEL:-unset} as "${SERVED_MODEL}"
engine:  ${INFERENCE_ENGINE:-vllm} (${INFERENCE_MODE}) on ${INFERENCE_NODES} node(s): ${nodes[*]}
image:   ${INFERENCE_CE_ENV:-unset}
logs:    ${RUN_DIR}
EOF

# Without --cpus-per-task the step claims ONE core of 192 and the engine's host-side work degrades
# with load rather than failing. --environment takes the registered EDF by name.
srun --nodes="${INFERENCE_NODES}" --ntasks="${INFERENCE_NODES}" --ntasks-per-node=1 \
    --cpus-per-task="${SLURM_CPUS_ON_NODE:-$(nproc)}" --exclusive --kill-on-bad-exit=1 \
    --export=ALL --environment="${INFERENCE_CE_ENV:?INFERENCE_CE_ENV must name a registered EDF}" \
    --output="${RUN_DIR}/server-%t.log" --error="${RUN_DIR}/server-%t.log" \
    "${SCRIPT_DIR}/run_cluster.sh" --vllm-node &
SERVER_PID=$!

echo "waiting for the API (up to ${READY_TIMEOUT}s); weight load dominates this"
deadline=$(( $(date +%s) + READY_TIMEOUT ))
until curl -sf "${BASE_URL}/v1/models" -o "${RUN_DIR}/models.json"; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "FAIL: the serving step exited before the API came up" >&2
        tail -40 "${RUN_DIR}"/server-*.log >&2 || true
        exit 1
    fi
    if (( $(date +%s) > deadline )); then
        echo "FAIL: API not ready within ${READY_TIMEOUT}s" >&2
        tail -40 "${RUN_DIR}"/server-*.log >&2 || true
        exit 1
    fi
    sleep 15
done

endpoints=("${BASE_URL}/v1")
if [[ "${INFERENCE_MODE}" == "replicas" ]]; then
    endpoints=()
    for node in "${nodes[@]}"; do endpoints+=("http://${node}:${VLLM_PORT}/v1"); done
fi

cat <<EOF

===== endpoint is live =====
base URL:   ${endpoints[0]}
model name: ${SERVED_MODEL}
replicas:   ${endpoints[*]}
health:     curl -s ${BASE_URL}/v1/models
metrics:    curl -s ${BASE_URL}/metrics
server log: ${RUN_DIR}/server-0.log

curl -s ${BASE_URL}/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"${SERVED_MODEL}","max_tokens":128,"messages":[{"role":"user","content":"Say hi."}]}'

The endpoint takes no API key. It stays up until this job ends; scancel ${SLURM_JOB_ID} to stop it.
EOF

wait "${SERVER_PID}"
