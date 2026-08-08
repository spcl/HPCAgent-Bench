#!/usr/bin/env bash
set -euo pipefail

ENVIRONMENT=${ENVIRONMENT:-rocm723-ofi-host-diag}
ROOT="${VLLM_BUILD_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR=${LOG_DIR:-${ROOT}/logs/phase1-host-ofi}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-120}
mkdir -p "$LOG_DIR"

STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$LOG_DIR/one-rank.${STAMP}.log"

echo "environment=$ENVIRONMENT log=$OUT"
srun \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-task=1 \
  --cpus-per-task=16 \
  --environment="$ENVIRONMENT" \
  bash -lc "
    set -euo pipefail
    host-loader-gate
    echo '===== one-rank RCCL plugin initialization ====='
    timeout --signal=TERM --kill-after=10s ${TIMEOUT_SECONDS}s \
      torchrun --standalone --nnodes=1 --nproc-per-node=1 \
      /usr/local/bin/torch-dist-allreduce-mi300.py \
      --size-mb 1 --warmup 1 --iters 3 --timeout-seconds 60
  " 2>&1 | tee "$OUT"
