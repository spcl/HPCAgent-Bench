#!/usr/bin/env bash
# Gate 5 for vLLM 0.27.1: does the CAMPAIGN's serving surface still exist there?
#
# Gates 1-4 covered the things that announce themselves -- the image builds, RCCL crosses nodes
# with correct sums, the tuned-MoE lookup key is byte-identical, and K2.7 decodes. What none of
# them covers is the surface the campaign actually drives, where 0.27.1 fails SILENTLY:
#
#   - vllm warns on an unrecognised VLLM_* variable and carries on ("Unknown vLLM environment
#     variable detected", envs.py). So a renamed VLLM_TUNED_CONFIG_FOLDER does not fail the run,
#     it just serves untuned -- and the 1.71x measured in 08-17 quietly disappears with nothing
#     in the log but a warning we already print two of.
#   - a dropped or renamed serve flag kills the head at startup. The campaign path was already
#     wedged once by a serve-arg delta, so parity is a fact to check, not to assume.
#   - our sitecustomize patch calls GroupCoordinator.make_sibling_device_group(group_desc=...).
#     That is an internal API. If 0.27.1 moved it, the pp collective split -- the fix that just
#     un-wedged 600518 -- fails at import inside every rank.
#   - the qwen3_coder tool parser ate turn-1 tool calls and voided both qwen arms; qwen3_xml is
#     the replacement. Whether 0.27.1 registers it decides whether the qwen arms can move here.
#
# Every check is static: no weights, no decode, one node, a couple of minutes. It answers "can
# the campaign move to 0.27.1" without spending a 4-node hour to find out.
#SBATCH --job-name=gate5-0271-surface
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/gate5-0271-surface-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/gate5-0271-surface-%j.err

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
# Slurm spools the batch script, so BASH_SOURCE lands in the spool dir: take the directory
# explicitly, the same contract as build-agent-image.sbatch's AMD_DIR.
INFERENCE_DIR="${INFERENCE_DIR:-${SCRATCH}/optarena/containers/cluster/ce-images/inference}"
test -d "${INFERENCE_DIR}/external-eager-pg-patch"
test -d "${INFERENCE_DIR}/moe-configs"

INFERENCE_EDF="${INFERENCE_EDF:-rocm723-vllm-0.27.1-pytorch211-ofi}"
# The exact word list the llr4 kimi arm serves with, plus the flags cluster_launch.py always
# passes. Kept as one string so it can be diffed against the .env file it was copied from.
CAMPAIGN_ARGS="${CAMPAIGN_ARGS:---host --port --tensor-parallel-size --pipeline-parallel-size --distributed-executor-backend --load-format --safetensors-load-strategy --generation-config --enable-auto-tool-choice --tool-call-parser --trust-remote-code --max-model-len --gpu-memory-utilization --reasoning-parser --max-num-seqs}"
export CAMPAIGN_ARGS
export VLLM_TUNED_CONFIG_FOLDER="${INFERENCE_DIR}/moe-configs"
export PG_PATCH_DIR="${INFERENCE_DIR}/external-eager-pg-patch"

echo "job:   ${SLURM_JOB_ID}"
echo "edf:   ${INFERENCE_EDF}"
echo "args:  ${CAMPAIGN_ARGS}"

# No --mount: the EDF already binds /capstor, which is where INFERENCE_DIR lives.
# 192 logical CPUs per node = 4 sockets x 24 physical cores x 2 SMT threads. --exclusive gives the
# JOB the node; it does NOT give a STEP the node's CPUs, and a step without --cpus-per-task claims
# ONE. That is not a slow-down, it is a hang: 620948's serving step ran on 2 CPUs across 2 nodes,
# its Triton JIT of alloc_extend_kernel never finished, and the peer's PP SEND blocked until the
# 600 s NCCL watchdog aborted every rank. Every vLLM smoke before this served on 1 CPU too, so
# their tok/s were floor numbers. run_cluster.sh:749 carries the full account; an acceptance run
# must allocate the way the campaign does or it is not measuring the campaign.
SERVE_CPUS="${SERVE_CPUS:-${SLURM_CPUS_ON_NODE:-192}}"
# Clients and gates run --overlap ALONGSIDE the server, so they take one socket's PHYSICAL cores
# (--hint=nomultithread) rather than the node -- enough not to be the bottleneck, not so much that
# the measurement contends with what it is measuring.
CLIENT_CPUS="${CLIENT_CPUS:-24}"

srun --ntasks=1 --export=ALL --cpus-per-task="${CLIENT_CPUS}" --hint=nomultithread \
    --environment="${INFERENCE_EDF}" \
    python3 "${INFERENCE_DIR}/gate_0271_surface.py"
