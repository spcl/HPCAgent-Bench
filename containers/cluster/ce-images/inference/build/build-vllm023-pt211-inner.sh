#!/usr/bin/env bash
set -Eeuxo pipefail

ROOT="${VLLM_BUILD_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"

export VIRTUAL_ENV=/opt/pytorch211
export PATH=/opt/pytorch211/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH="/opt/aws-ofi-nccl/lib:/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}"

export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH=gfx942
export ROCM_PATH=/opt/rocm
export MAX_JOBS="${MAX_JOBS:-32}"
export CMAKE_BUILD_TYPE=Release
export VLLM_VERSION_OVERRIDE=0.23.0
export SETUPTOOLS_SCM_PRETEND_VERSION=0.23.0

PYTHON=/opt/pytorch211/bin/python
PIP=("$PYTHON" -m pip)

echo "========================================"
echo "VALIDATE QUALIFIED PYTORCH BASE"
echo "========================================"

"$PYTHON" - <<'PY'
import sys
import numpy
import torch

print("Python:", sys.executable)
print("NumPy:", numpy.__version__)
print("PyTorch:", torch.__version__)
print("PyTorch file:", torch.__file__)
print("HIP:", torch.version.hip)
print("RCCL:", torch.cuda.nccl.version())
print("GPU count:", torch.cuda.device_count())

assert sys.executable == "/opt/pytorch211/bin/python"
assert torch.__version__ == "2.11.0+rocm7.2"
assert torch.__file__.startswith("/opt/pytorch211/")
assert torch.version.hip is not None
assert torch.cuda.device_count() == 4
PY

echo "========================================"
echo "INSTALL MATCHING TORCHVISION/TORCHAUDIO"
echo "========================================"

"${PIP[@]}" install \
  --no-deps \
  --index-url https://download.pytorch.org/whl/rocm7.2 \
  'torchvision==0.26.0+rocm7.2' \
  'torchaudio==2.11.0+rocm7.2'

# torchvision was installed with --no-deps, so install its image dependency.
"${PIP[@]}" install 'pillow>=10.0'

"$PYTHON" - <<'PY'
import torch
import torchvision
import torchaudio

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)

assert torch.__version__ == "2.11.0+rocm7.2"
assert torchvision.__version__ == "0.26.0+rocm7.2"
assert torchaudio.__version__ == "2.11.0+rocm7.2"
PY

echo "========================================"
echo "INSTALL BUILD TOOLS"
echo "========================================"

apt-get update

DEBIAN_FRONTEND=noninteractive apt-get install -y \
  --no-install-recommends \
  build-essential \
  ca-certificates \
  git \
  ninja-build \
  pkg-config \
  python3-dev \
  libdrm-dev \
  libnuma-dev \
  libfmt-dev \
  libmsgpack-dev \
  libsuitesparse-dev

rm -rf /var/lib/apt/lists/*

"${PIP[@]}" install --upgrade \
  'cmake>=3.26.1,<4' \
  ninja \
  pybind11 \
  'packaging>=24.2' \
  'setuptools>=77.0.3,<81' \
  'setuptools-scm>=8' \
  'setuptools-rust>=1.9.0' \
  wheel \
  'jinja2>=3.1.6' \
  more-itertools

echo "========================================"
echo "CLONE VLLM 0.23.0"
echo "========================================"

"${PIP[@]}" uninstall -y vllm 2>/dev/null || true
rm -rf /opt/vllm-src

git clone \
  --branch v0.23.0 \
  --depth 1 \
  https://github.com/vllm-project/vllm.git \
  /opt/vllm-src

cd /opt/vllm-src

git rev-parse HEAD
git describe --tags --always

echo "========================================"
echo "PRESERVE EXISTING ROCM PYTORCH"
echo "========================================"

"$PYTHON" use_existing_torch.py --prefix

if grep -RniE \
  '^[[:space:]]*(torch|torchvision|torchaudio)[[:space:]]*[=<>]' \
  requirements pyproject.toml
then
    echo "ERROR: direct PyTorch pins remain after helper"
    exit 1
fi

echo "========================================"
echo "CREATE STRICT GPU STACK CONSTRAINTS"
echo "========================================"

"$PYTHON" - <<'PY' >/tmp/pt211-rocm.constraints
import importlib.metadata

required = {
    "torch": "2.11.0+rocm7.2",
    "torchvision": "0.26.0+rocm7.2",
    "torchaudio": "2.11.0+rocm7.2",
}

for package, expected in required.items():
    actual = importlib.metadata.version(package)

    if actual != expected:
        raise SystemExit(
            f"{package}: expected {expected}, found {actual}"
        )

    print(f"{package}=={actual}")

for package in (
    "pytorch-triton-rocm",
    "triton",
):
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        continue

    print(f"{package}=={version}")
PY

cat /tmp/pt211-rocm.constraints

echo "========================================"
echo "INSTALL VLLM ROCM DEPENDENCIES"
echo "========================================"

"${PIP[@]}" install \
  --constraint /tmp/pt211-rocm.constraints \
  --requirement requirements/rocm.txt

echo "========================================"
echo "VERIFY PIP DID NOT REPLACE PYTORCH"
echo "========================================"

"$PYTHON" - <<'PY'
import importlib.metadata
import torch
import torchvision
import torchaudio

expected = {
    "torch": "2.11.0+rocm7.2",
    "torchvision": "0.26.0+rocm7.2",
    "torchaudio": "2.11.0+rocm7.2",
}

for package, wanted in expected.items():
    actual = importlib.metadata.version(package)
    print(f"{package}: {actual}")

    if actual != wanted:
        raise RuntimeError(
            f"{package} changed: expected {wanted}, got {actual}"
        )

assert torch.version.hip is not None
assert torch.__file__.startswith("/opt/pytorch211/")

cuda_packages = sorted(
    dist.metadata["Name"]
    for dist in importlib.metadata.distributions()
    if (dist.metadata.get("Name") or "").lower().startswith("nvidia-")
)

print("NVIDIA packages:", cuda_packages)

if cuda_packages:
    raise RuntimeError(
        "CUDA packages were unexpectedly installed: "
        + ", ".join(cuda_packages)
    )

print("QUALIFIED PYTORCH STACK PRESERVED")
PY

echo "========================================"
echo "INSTALL MATCHING AMD SMI"
echo "========================================"

"${PIP[@]}" uninstall -y amdsmi 2>/dev/null || true

"${PIP[@]}" install \
  --no-deps \
  --force-reinstall \
  /opt/rocm/share/amd_smi

echo "========================================"
echo "REMOVE CUDA-ONLY OPTIONAL EXTENSION"
echo "========================================"

"${PIP[@]}" uninstall -y torch-c-dlpack-ext || true

rm -rf \
  /root/.cache/tvm-ffi \
  /root/.cache/torch_extensions

echo "========================================"
echo "BUILD VLLM 0.23.0"
echo "========================================"

cd /opt/vllm-src

rm -rf \
  build \
  dist \
  .eggs \
  vllm.egg-info

BUILD_LOG="${ROOT}/logs/vllm023-pt211-build.${SLURM_JOB_ID:-manual}.log"

"${PIP[@]}" install \
  --editable . \
  --no-build-isolation \
  --no-deps \
  --verbose \
  2>&1 | tee "$BUILD_LOG"

echo "========================================"
echo "VALIDATE EXTENSIONS AND OPERATORS"
echo "========================================"

"$PYTHON" - <<'PY'
import importlib
import sys

import torch
import vllm
from vllm.platforms import current_platform

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("PyTorch file:", torch.__file__)
print("HIP:", torch.version.hip)
print("RCCL:", torch.cuda.nccl.version())
print("vLLM:", vllm.__version__)
print("vLLM file:", vllm.__file__)
print("Platform:", current_platform.__class__)
print("GPU count:", torch.cuda.device_count())

assert sys.executable == "/opt/pytorch211/bin/python"
assert torch.__version__ == "2.11.0+rocm7.2"
assert torch.__file__.startswith("/opt/pytorch211/")
assert vllm.__file__.startswith("/opt/vllm-src/")
assert "RocmPlatform" in current_platform.__class__.__name__

for module in (
    "vllm._C",
    "vllm._rocm_C",
    "vllm._C_stable_libtorch",
):
    importlib.import_module(module)
    print(module, "OK")

required_ops = (
    "silu_and_mul",
    "gelu_and_mul",
    "rms_norm",
    "fused_add_rms_norm",
)

missing = []

for name in required_ops:
    present = hasattr(torch.ops._C, name)
    print(f"torch.ops._C.{name}: {present}")

    if not present:
        missing.append(name)

if missing:
    raise RuntimeError(
        "Missing vLLM operators: " + ", ".join(missing)
    )

print("VLLM EXTENSION AND OPERATOR VALIDATION PASSED")
PY

command -v python
command -v vllm

vllm --version
vllm serve --help >/tmp/vllm-serve-help.txt

rm -rf /root/.cache/pip

echo "========================================"
echo "VLLM 0.23.0 + PYTORCH 2.11 BUILD PASSED"
echo "========================================"
