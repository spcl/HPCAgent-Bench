#!/usr/bin/env bash
set -euo pipefail

ENVIRONMENT=${ENVIRONMENT:-rocm723-ofi-host-diag}
ROOT="${VLLM_BUILD_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
NPROC=${NPROC:-2}
SIZE_MB=${SIZE_MB:-64}
WARMUP=${WARMUP:-3}
ITERS=${ITERS:-10}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-180}

case "$NPROC" in
  2|4) ;;
  *)
    echo "NPROC must be 2 or 4" >&2
    exit 2
    ;;
esac

LOG_DIR="${ROOT}/logs/phase1-host-ofi"
mkdir -p "$LOG_DIR"

STAMP=$(date +%Y%m%d-%H%M%S)
OUT="${LOG_DIR}/${NPROC}-rank.${STAMP}.log"

echo "environment=${ENVIRONMENT}"
echo "nproc=${NPROC}"
echo "log=${OUT}"

srun \
  --partition=mi300 --nodes=1 \
  --ntasks=1 \
  --gpus-per-task="${NPROC}" \
  --cpus-per-task=64 \
  --environment="${ENVIRONMENT}" \
  bash -lc "
    set -euo pipefail

    echo '===== visible devices ====='
    echo \"ROCR_VISIBLE_DEVICES=\${ROCR_VISIBLE_DEVICES-}\"

    python3 - <<'PY'
import torch
print('torch', torch.__version__)
print('device_count', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

    echo '===== ${NPROC}-rank RCCL test ====='

    timeout --signal=TERM --kill-after=15s ${TIMEOUT_SECONDS}s \
      torchrun \
        --standalone \
        --nnodes=1 \
        --nproc-per-node=${NPROC} \
        /usr/local/bin/torch-dist-allreduce-mi300.py \
        --size-mb ${SIZE_MB} \
        --warmup ${WARMUP} \
        --iters ${ITERS} \
        --timeout-seconds 120
  " 2>&1 | tee "$OUT"
