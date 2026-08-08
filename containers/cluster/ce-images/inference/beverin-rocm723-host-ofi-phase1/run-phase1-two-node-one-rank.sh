#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?Run inside a two-node Slurm allocation}"
: "${SLURM_JOB_NODELIST:?Missing SLURM_JOB_NODELIST}"

ENVIRONMENT=${ENVIRONMENT:-rocm723-ofi-host-diag}
ROOT="${VLLM_BUILD_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
MASTER_PORT=${MASTER_PORT:-29500}
SIZE_MB=${SIZE_MB:-256}
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-20}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-300}

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")

if (( ${#NODES[@]} < 2 )); then
    echo "Need at least two allocated nodes" >&2
    exit 2
fi

MASTER_NODE=${NODES[0]}

MASTER_ADDR=$(
    srun \
      --partition=mi300 \
      --nodes=1 \
      --ntasks=1 \
      --nodelist="$MASTER_NODE" \
      bash -lc \
      "ip -4 -o addr show hsn0 | awk '{print \$4}' | cut -d/ -f1"
)

if [[ -z "$MASTER_ADDR" ]]; then
    echo "Could not resolve hsn0 address on $MASTER_NODE" >&2
    exit 2
fi

LOG_DIR="${ROOT}/logs/phase1-host-ofi"
mkdir -p "$LOG_DIR"

STAMP=$(date +%Y%m%d-%H%M%S)
OUT="${LOG_DIR}/two-node-1ppn.${STAMP}.log"

export MASTER_ADDR MASTER_PORT SIZE_MB WARMUP ITERS TIMEOUT_SECONDS

echo "nodes=${NODES[*]}"
echo "master_node=${MASTER_NODE}"
echo "master_addr=${MASTER_ADDR}"
echo "environment=${ENVIRONMENT}"
echo "log=${OUT}"

srun \
  --partition=mi300 \
  --nodes=2 \
  --ntasks=2 \
  --ntasks-per-node=1 \
  --gpus-per-task=1 \
  --cpus-per-task=32 \
  --environment="$ENVIRONMENT" \
  bash -lc '
    set -euo pipefail

    echo \
      "host=$(hostname)" \
      "node_rank=${SLURM_NODEID}" \
      "visible=${ROCR_VISIBLE_DEVICES-}" \
      "master=${MASTER_ADDR}:${MASTER_PORT}"

    export NCCL_DEBUG_FILE="'"${LOG_DIR}"'/rccl.%h.%p.log"

    timeout \
      --signal=TERM \
      --kill-after=15s \
      "${TIMEOUT_SECONDS}s" \
      torchrun \
        --nnodes=2 \
        --nproc-per-node=1 \
        --node-rank="${SLURM_NODEID}" \
        --master-addr="${MASTER_ADDR}" \
        --master-port="${MASTER_PORT}" \
        /usr/local/bin/torch-dist-allreduce-mi300.py \
          --size-mb "${SIZE_MB}" \
          --warmup "${WARMUP}" \
          --iters "${ITERS}" \
          --timeout-seconds 180
  ' 2>&1 | tee "$OUT"
