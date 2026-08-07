#!/usr/bin/env bash
set -Eeuo pipefail

: "${SLURM_JOB_ID:?This script must run inside Slurm}"

ROOT="${VLLM_BUILD_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
BASE_SQSH="${ROOT}/phase1-passed/rocm723-ofi-host-diag-phase1.sqsh"
OUTPUT_SQSH="${ROOT}/containers/rocm723-ofi-vllm-0.23.0.sqsh"
NAME="rocm723-ofi-vllm-023-${SLURM_JOB_ID}"

test -s "$BASE_SQSH" || {
    echo "Missing base image: $BASE_SQSH" >&2
    exit 1
}

mkdir -p "${ROOT}/containers"

cleanup() {
    enroot remove "$NAME" 2>/dev/null || true
}
trap cleanup EXIT

echo "===== create writable rootfs ====="
enroot remove "$NAME" 2>/dev/null || true

enroot create \
    --name "$NAME" \
    "$BASE_SQSH"

echo "===== install prebuilt vLLM stack ====="
enroot start \
    --root \
    --rw \
    "$NAME" \
    bash -s <<'IN_CONTAINER'
set -Eeuo pipefail

python3 -m pip install \
    --no-cache-dir \
    --upgrade \
    uv

rm -rf /opt/vllm-venv

uv venv \
    --python "$(command -v python3)" \
    --seed \
    /opt/vllm-venv

VLLM_INDEX="https://wheels.vllm.ai/rocm/0.23.0/rocm723"

echo "===== install vLLM wheel ====="
uv pip install \
    --python /opt/vllm-venv/bin/python \
    --extra-index-url "${VLLM_INDEX}" \
    --index-strategy unsafe-best-match \
    --no-cache \
    "vllm==0.23.0+rocm723"

echo "===== version gate ====="

/opt/vllm-venv/bin/python - <<'PY'
import torch
import vllm

print("vLLM:", vllm.__version__)
print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("Python:", __import__("sys").version)

assert vllm.__version__.startswith("0.23.0"), vllm.__version__
assert torch.version.hip is not None
assert torch.version.hip.startswith("7.2"), torch.version.hip
PY

echo "===== CLI gate ====="
/opt/vllm-venv/bin/vllm --version

echo "===== AWS-OFI plugin gate ====="
test -e /opt/aws-ofi-nccl/lib/librccl-net.so

ldd -r /opt/aws-ofi-nccl/lib/librccl-net.so

echo "===== private fabric-library gate ====="
PRIVATE_LIBS=$(
    find \
        /opt/aws-ofi-nccl \
        /opt/vllm-venv \
        \( -name 'libfabric.so*' -o -name 'libcxi.so*' \) \
        -print 2>/dev/null || true
)

if [[ -n "$PRIVATE_LIBS" ]]; then
    echo "ERROR: private libfabric/libcxi found:" >&2
    printf '%s\n' "$PRIVATE_LIBS" >&2
    exit 1
fi

echo "vLLM installation passed"
IN_CONTAINER

echo "===== export squash image ====="
rm -f "$OUTPUT_SQSH"

enroot export \
    --output "$OUTPUT_SQSH" \
    "$NAME"

test -s "$OUTPUT_SQSH"

ls -lh "$OUTPUT_SQSH"

sha256sum "$OUTPUT_SQSH" |
    tee "${OUTPUT_SQSH}.sha256"

echo "===== complete ====="
echo "image=$OUTPUT_SQSH"
