#!/bin/bash
# SHORT 5-node smoke of the full 3-tier pipeline on Alps -- proves the whole
# agent -> multi-node vLLM -> multi-node judge wiring end-to-end on real hardware,
# on a tiny amount of work so it finishes fast.
#
# Node split (5 nodes, from run_campaign.sbatch's partition: node0 agent, then
# VLLM_NODES vLLM, then the rest judge):
#   node0        AGENT     (1) -- the think pool, queries vLLM over HTTP
#   node1-2      vLLM      (2) -- inference tier, pipeline-parallel across 2 nodes
#   node3-4      JUDGE     (2) -- the "test"/measure pool (reference + candidate timing)
#
# Submit:  sbatch -A <account> scripts/cscs/run_smoke_5node.sbatch
#   `-A <account>` is MANDATORY on Alps (or export SBATCH_ACCOUNT first).
#   Needs env.toml at the repo root (the CE/EDF); see scripts/cscs/env.toml.example.
# Everything below is overridable from the environment before sbatch.
#SBATCH --job-name=optarena-smoke5
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

# --- 5-node shape: 1 agent + 2 inference + 2 judge ---
export VLLM_NODES="${VLLM_NODES:-2}"                 # 2 inference nodes (pipeline-parallel serve)
export TP_SIZE="${TP_SIZE:-4}"                       # tensor-parallel width = GPUs per vLLM node
export JUDGE_GPUS_PER_NODE="${JUDGE_GPUS_PER_NODE:-4}"
export AGENT_WORKERS="${AGENT_WORKERS:-4}"           # concurrent think workers on the agent node
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Coder-30B}"
export VLLM_WAIT_S="${VLLM_WAIT_S:-900}"             # generous: multi-node ray + model load is slow

# --- keep the WORK short: a couple of tiny kernels, smallest preset, single-shot ---
export KERNELS="${KERNELS:-gemm,gesummv}"            # 2 tiny, robustly-C-translatable BLAS kernels
export LANGUAGES="${LANGUAGES:-c}"
export PRESET="${PRESET:-S}"                         # smallest data size (not fuzzed)
export REPEAT="${REPEAT:-2}"                         # 2 timed reps (default is 5)
export REPAIR_ROUNDS="${REPAIR_ROUNDS:-1}"          # single shot, no repair loop
export BASELINE="${BASELINE:-numpy}"                # hpc/ml speedup denominator

echo "[smoke5] 5-node smoke: 1 agent + ${VLLM_NODES} vLLM + judge; kernels=${KERNELS} preset=${PRESET}"
# Reuse the full campaign verbatim -- the #SBATCH lines below in run_campaign.sbatch are inert
# when it is run as a plain script; the allocation + knobs above drive it.
exec bash "${DIR}/run_campaign.sbatch"
