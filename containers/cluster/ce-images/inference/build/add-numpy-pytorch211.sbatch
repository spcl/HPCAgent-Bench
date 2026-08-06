#!/usr/bin/env bash
#SBATCH --job-name=pt211-numpy
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/add-numpy-pt211-%j.out
#SBATCH --error=logs/add-numpy-pt211-%j.err

set -Eeuxo pipefail
umask 0022

ROOT="${VLLM_BUILD_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"

SOURCE_IMAGE="${ROOT}/containers/rocm723-pytorch211-ofi-phase1-candidate.sqsh"
TEMP_IMAGE="${ROOT}/containers/rocm723-pytorch211-ofi-phase1-candidate.numpy.tmp.sqsh"
FINAL_IMAGE="${ROOT}/containers/rocm723-pytorch211-ofi-phase1-candidate.sqsh"
BACKUP_IMAGE="${ROOT}/containers/rocm723-pytorch211-ofi-phase1-candidate.before-numpy.sqsh"

BUILD_NAME="rocm723-pytorch211-numpy-patch"

test -f "$SOURCE_IMAGE"

echo "Host: $(hostname)"
echo "Source: ${SOURCE_IMAGE}"
echo "Temporary output: ${TEMP_IMAGE}"

enroot remove --force "$BUILD_NAME" 2>/dev/null || true
rm -f "$TEMP_IMAGE" "${TEMP_IMAGE}.sha256"

enroot create \
  --name "$BUILD_NAME" \
  "$SOURCE_IMAGE"

enroot start \
  --root \
  --rw \
  "$BUILD_NAME" \
  bash -lc '
    set -Eeuxo pipefail

    export VIRTUAL_ENV=/opt/pytorch211
    export PATH=/opt/pytorch211/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/rocm/lib:/opt/rocm/lib64

    /opt/pytorch211/bin/python -m pip install \
      --no-cache-dir \
      "numpy<3"

    /opt/pytorch211/bin/python - <<PY
import sys
import numpy
import torch

print("Python:", sys.executable)
print("NumPy:", numpy.__version__)
print("NumPy file:", numpy.__file__)
print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("RCCL:", torch.cuda.nccl.version())
print("GPU count:", torch.cuda.device_count())

assert sys.executable == "/opt/pytorch211/bin/python"
assert numpy.__file__.startswith("/opt/pytorch211/")
assert torch.__version__.startswith("2.11.0+rocm7.2")
assert torch.cuda.device_count() >= 1

cpu = torch.arange(16, dtype=torch.float32)
array = cpu.numpy()

assert array.shape == (16,)
assert float(array.sum()) == 120.0

gpu = torch.ones(1024, device="cuda")
assert gpu.sum().item() == 1024

print("NUMPY + PYTORCH VALIDATION PASSED")
PY

    rm -rf /root/.cache/pip
  '

enroot export \
  --output "$TEMP_IMAGE" \
  "$BUILD_NAME"

test -s "$TEMP_IMAGE"

sha256sum "$TEMP_IMAGE" |
  tee "${TEMP_IMAGE}.sha256"

echo "Temporary image exported successfully"

if [[ ! -e "$BACKUP_IMAGE" ]]; then
    mv "$FINAL_IMAGE" "$BACKUP_IMAGE"
else
    rm -f "$FINAL_IMAGE"
fi

mv "$TEMP_IMAGE" "$FINAL_IMAGE"

sha256sum "$FINAL_IMAGE" |
  tee "${FINAL_IMAGE}.sha256"

rm -f "${TEMP_IMAGE}.sha256"

ls -lh \
  "$FINAL_IMAGE" \
  "${FINAL_IMAGE}.sha256" \
  "$BACKUP_IMAGE"

echo "========================================"
echo "NUMPY IMAGE PATCH PASSED"
echo "========================================"
