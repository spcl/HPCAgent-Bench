#!/usr/bin/env bash
# Enumerate every cache root a serving rank writes into -- see cache_roots.py for why.
#
#   sbatch probe-cache-roots.sbatch                                   # vLLM 0.27.1
#   INFERENCE_EDF=sglang-rocm-mi30x sbatch probe-cache-roots.sbatch   # SGLang
#SBATCH --job-name=probe-cache-roots
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=logs/probe-cache-roots-%j.out
#SBATCH --error=logs/probe-cache-roots-%j.err

set -Eeuo pipefail
ulimit -c 0
INFERENCE_EDF="${INFERENCE_EDF:-rocm723-vllm-0.27.1-pytorch211-ofi}"
export PROBE="${SLURM_SUBMIT_DIR:-$PWD}/cache_roots.py"
echo "image:  ${INFERENCE_EDF}"
echo "script: ${PROBE}"

# The probe is a FILE, not an inline heredoc: a single quote inside the bash -c body closed it
# early and the shell read the next word as a redirect (610184/610187, "unset: No such file").
srun --environment="${INFERENCE_EDF}" bash -c '
PY=python3
[[ -x /opt/venv/bin/python3 ]] && PY=/opt/venv/bin/python3
"${PY}" "${PROBE}"'
