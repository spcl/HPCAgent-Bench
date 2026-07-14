#!/bin/bash
# CSCS Alps campaign: agent pool + vLLM node(s) + a MULTI-NODE judge pool -- the
# 3-tier agent / reference / inference topology, all in one SLURM allocation.
#
# Topology (all in one SLURM allocation):
#   node[0]        AGENT POOL -- runs AGENT_WORKERS concurrent optarena agent workers
#                               that "think" (query vLLM over HTTP). No GPU work here,
#                               so the agents never idle a timing GPU.
#   node[1..V]     vLLM       -- serves the model (single node TP by default; ray-based
#                               tensor+pipeline parallel across V nodes only if V>1).
#   node[V+1..N-1] JUDGE      -- the device pool the judge "measures" on (reference +
#                               candidate timing); each node's GPUs become judge slots.
#                               One kernel per slot; the judge work-steals across ALL
#                               judge nodes' GPUs.
#
# Both pools are driven from judge_scheduler: OPTARENA_JUDGE_NODES_EXPANDED feeds
# JudgeConfig (a slot per (judge node, GPU); remote grades via the OPTARENA_JUDGE_LAUNCHER
# srun template -- which MUST carry --environment so the dispatched grade runs INSIDE the CE
# image), and OPTARENA_AGENT_NODES_EXPANDED / OPTARENA_AGENT_WORKERS_PER_NODE feed
# AgentPoolConfig, so TwoStageScheduler pipelines think (agent pool) -> grade (judge pool)
# with the two pools running concurrently.
#
# Submit:  sbatch -A <account> --nodes=$N scripts/cscs/run_campaign.sbatch
#   `-A <account>` is MANDATORY on Alps (or export SBATCH_ACCOUNT before submit).
# Tune via env (export before sbatch, or --export): VLLM_NODES, MODEL_ID, TP_SIZE,
#   JUDGE_GPUS_PER_NODE, AGENT_WORKERS, KERNELS, BASELINE, VLLM_PORT.
# The EDF (env.toml) selects the aarch64 (Grace-Hopper) image + CXI/NCCL hooks; see
# scripts/cscs/env.toml.example and docs/RUNTIME.md.
#SBATCH --job-name=optarena-campaign
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail

REPO="${OPTARENA_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
ENV_TOML="${OPTARENA_ENV_TOML:-${REPO}/env.toml}"   # enroot/CXI env (see docs/RUNTIME.md)
# Per-role EDFs: each defaults to ENV_TOML, so an unset role var = today's single-EDF behavior.
# Split them (per-role EDF copies) to point each tier at a different Container-Engine image.
VLLM_ENV_TOML="${OPTARENA_VLLM_ENV_TOML:-${ENV_TOML}}"    # inference tier image (e.g. Lorenzo's vLLM EDF)
AGENT_ENV_TOML="${OPTARENA_AGENT_ENV_TOML:-${ENV_TOML}}"  # agent (think) tier image
JUDGE_ENV_TOML="${OPTARENA_JUDGE_ENV_TOML:-${ENV_TOML}}"  # judge (measure) tier image

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Coder-30B}"
TP_SIZE="${TP_SIZE:-4}"                 # tensor-parallel width = GPUs on the vLLM head node
VLLM_NODES="${VLLM_NODES:-1}"           # nodes vLLM spans (1 = single-node TP; >1 = ray, pipeline across them)
VLLM_PORT="${VLLM_PORT:-8000}"
RAY_PORT="${RAY_PORT:-6379}"
JUDGE_GPUS_PER_NODE="${JUDGE_GPUS_PER_NODE:-4}"
AGENT_WORKERS="${AGENT_WORKERS:-8}"     # concurrent agent (think) workers on the agent node
KERNELS="${KERNELS:-all}"
BASELINE="${BASELINE:-track}"           # per-track default (foundation->c-autopar, ml/hpc->numpy)
VLLM_WAIT_S="${VLLM_WAIT_S:-600}"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
N="${#NODES[@]}"
(( N >= VLLM_NODES + 2 )) || { echo "need >= VLLM_NODES+2 nodes (agent + vLLM + >=1 judge); got $N" >&2; exit 2; }

AGENT_NODE="${NODES[0]}"
VLLM_HEAD="${NODES[1]}"
VLLM_HOSTS=("${NODES[@]:1:VLLM_NODES}")
JUDGE_HOSTS=("${NODES[@]:1+VLLM_NODES}")            # everything after the vLLM block
JUDGE_EXPANDED="$(IFS=,; echo "${JUDGE_HOSTS[*]}")"
VLLM_URL="http://${VLLM_HEAD}:${VLLM_PORT}/v1"

# Remote judge grades srun back onto a judge node. Carry --environment so the grade runs
# INSIDE the CE image (its frameworks/toolchain), and --overlap so it shares the step; {node}
# is substituted per slot by judge_scheduler.resolve_launcher. Override with OPTARENA_JUDGE_LAUNCHER.
# Build the default in a PLAIN var first: a `{node}` literal inside a ${VAR:-...} default is
# mangled by bash brace-matching (the `}` closes the expansion early), which would corrupt the
# per-slot substitution. Assigning the default separately keeps `{node}` intact.
DEFAULT_JUDGE_LAUNCHER="srun --nodelist {node} --gpus 1 -n 1 --overlap --environment ${JUDGE_ENV_TOML}"
JUDGE_LAUNCHER="${OPTARENA_JUDGE_LAUNCHER:-$DEFAULT_JUDGE_LAUNCHER}"

echo "agent=${AGENT_NODE}  vllm=${VLLM_HOSTS[*]} (head ${VLLM_HEAD})  judge=${JUDGE_HOSTS[*]}"
echo "vLLM URL=${VLLM_URL}  judge nodes=${#JUDGE_HOSTS[@]} x ${JUDGE_GPUS_PER_NODE} GPU"

cleanup() { jobs -p | xargs -r kill || true; wait || true; }
trap cleanup EXIT

# --- vLLM server ---
# Default: a SINGLE node, tensor-parallel across its GPUs -- Qwen3-Coder-30B fits one GH200
# node (4x96GB), so no ray, no cross-node NCCL. The simplest, most robust option.
# Multi-node (VLLM_NODES>1, only for a model too big for one node): a real ray cluster (head
# + workers) THEN ONE `vllm serve` on the head with --distributed-executor-backend ray and
# pipeline-parallel across the nodes. (Per-rank --node-rank/--master-addr is NOT vLLM's
# multi-node API and does not form a server.) The EDF must enable com.hooks.aws_ofi_nccl so
# NCCL rides Slingshot (see env.toml.example).
SERVE="vllm serve '${MODEL_ID}' --host 0.0.0.0 --port '${VLLM_PORT}' --tensor-parallel-size '${TP_SIZE}'"
if (( VLLM_NODES == 1 )); then
  srun --nodes=1 --ntasks=1 --nodelist="${VLLM_HEAD}" --gpus-per-node="${TP_SIZE}" \
    --environment="${VLLM_ENV_TOML}" bash -lc "cd '${REPO}' && ${SERVE}" &
  VLLM_PID=$!
else
  srun --nodes=1 --ntasks=1 --nodelist="${VLLM_HEAD}" --gpus-per-node="${TP_SIZE}" \
    --environment="${VLLM_ENV_TOML}" bash -lc "ray start --head --port='${RAY_PORT}' --block" &
  for host in "${VLLM_HOSTS[@]:1}"; do
    srun --nodes=1 --ntasks=1 --nodelist="${host}" --gpus-per-node="${TP_SIZE}" \
      --environment="${VLLM_ENV_TOML}" bash -lc "ray start --address='${VLLM_HEAD}:${RAY_PORT}' --block" &
  done
  sleep 20   # let the ray head + workers register before serve attaches
  srun --nodes=1 --ntasks=1 --nodelist="${VLLM_HEAD}" --overlap --environment="${VLLM_ENV_TOML}" \
    bash -lc "cd '${REPO}' && ${SERVE} --pipeline-parallel-size '${VLLM_NODES}' \
      --distributed-executor-backend ray" &
  VLLM_PID=$!
fi

# --- wait for the endpoint, fail fast if the server step dies ---
echo "waiting for vLLM at ${VLLM_URL} (up to ${VLLM_WAIT_S}s) ..."
ready=0
for (( i = 0; i < VLLM_WAIT_S / 5; i++ )); do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM server step exited before becoming ready -- aborting" >&2; exit 3
  fi
  if curl -fsS "${VLLM_URL}/models" >/dev/null 2>&1; then ready=1; echo "vLLM ready"; break; fi
  sleep 5
done
(( ready == 1 )) || { echo "vLLM not ready after ${VLLM_WAIT_S}s -- aborting" >&2; exit 3; }

# --- run the agent + multi-node judge on the agent node ---
# Agent pool -> vLLM over HTTP (OpenAIAgent); judge -> the judge-node GPU pool.
# TwoStageScheduler pipelines think (agent pool) -> grade (judge pool).
srun --nodes=1 --ntasks=1 --nodelist="${AGENT_NODE}" --environment="${AGENT_ENV_TOML}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    export VLLM_BASE_URL='${VLLM_URL}'
    export OPTARENA_OPENAI_MODEL='${MODEL_ID}'
    export OPTARENA_AGENT_NODES_EXPANDED='${AGENT_NODE}'
    export OPTARENA_AGENT_WORKERS_PER_NODE='${AGENT_WORKERS}'
    export OPTARENA_JUDGE_NODES_EXPANDED='${JUDGE_EXPANDED}'
    export OPTARENA_JUDGE_GPUS_PER_NODE='${JUDGE_GPUS_PER_NODE}'
    export OPTARENA_JUDGE_LAUNCHER='${JUDGE_LAUNCHER}'
    python -m optarena.cli agent openai --kernels '${KERNELS}' --baseline '${BASELINE}'
  "
