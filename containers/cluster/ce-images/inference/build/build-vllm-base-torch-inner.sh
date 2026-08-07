#!/usr/bin/env bash
set -Eeuxo pipefail

ROOT="${VLLM_BUILD_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"

export VIRTUAL_ENV=/opt/venv
export PATH=/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}"

export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH=gfx942
export ROCM_PATH=/opt/rocm
export MAX_JOBS="${MAX_JOBS:-32}"
export CMAKE_BUILD_TYPE=Release
export VLLM_REQUIRE_RUST_FRONTEND=0
export VLLM_VERSION_OVERRIDE=0.23.0
export SETUPTOOLS_SCM_PRETEND_VERSION=0.23.0

PYTHON=/opt/venv/bin/python
PIP=("$PYTHON" -m pip)

echo "========================================"
echo "VALIDATING BASE PYTORCH"
echo "========================================"

EXPECTED_TORCH=$(
    "$PYTHON" -c 'import torch; print(torch.__version__)'
)

"$PYTHON" - <<'PY'
import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("PyTorch file:", torch.__file__)
print("HIP:", torch.version.hip)
print("GPU count:", torch.cuda.device_count())

assert sys.executable == "/opt/venv/bin/python"
assert torch.__version__.startswith("2.9.1")
assert torch.__file__.startswith("/opt/venv/")
assert torch.version.hip is not None
assert torch.cuda.device_count() >= 1
PY

echo "Preserving PyTorch: ${EXPECTED_TORCH}"

echo "========================================"
echo "INSTALLING BUILD TOOLS"
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

"${PIP[@]}" install \
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
echo "CLONING VLLM 0.23.0"
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
echo "PRESERVING EXISTING PYTORCH"
echo "========================================"

"$PYTHON" use_existing_torch.py --prefix

if grep -RniE \
    '^[[:space:]]*(torch|torchvision|torchaudio)[[:space:]]*[=<>]' \
    requirements pyproject.toml
then
    echo "ERROR: pinned PyTorch requirements remain"
    exit 1
fi

echo "========================================"
echo "PATCHING COMPRESSED-TENSORS REQUIREMENT"
echo "========================================"

cp -a \
  requirements/common.txt \
  requirements/common.txt.before-base-torch-patch

sed -i \
  '/^[[:space:]]*compressed-tensors[[:space:]]*==[[:space:]]*0\.17\.0/d' \
  requirements/common.txt

if grep -n \
    '^[[:space:]]*compressed-tensors' \
    requirements/common.txt
then
    echo "ERROR: compressed-tensors requirement was not removed"
    exit 1
fi

echo "========================================"
echo "DISABLING STABLE LIBTORCH EXTENSION"
echo "========================================"

cp -a setup.py setup.py.before-base-torch-patch

"$PYTHON" - <<'STABLEPATCH'
from pathlib import Path
import re

path = Path("setup.py")
text = path.read_text()

pattern = re.compile(
    r"(?m)^(    def build_extensions\(self\) -> None:\n)"
)

insertion = (
    r"\1"
    '        # ROCm compatibility build against PyTorch 2.9.1.\n'
    '        # Skip stable-libtorch, which requires newer PyTorch APIs.\n'
    '        self.extensions = [\n'
    '            ext\n'
    '            for ext in self.extensions\n'
    '            if ext.name != "vllm._C_stable_libtorch"\n'
    '        ]\n'
)

patched, count = pattern.subn(insertion, text, count=1)

if count != 1:
    raise SystemExit(
        f"Expected one build_extensions method; patched {count}"
    )

path.write_text(patched)
print("Stable-libtorch extension filter inserted")
STABLEPATCH

grep -n -A16 -B2 \
  'def build_extensions' \
  setup.py

echo "========================================"
echo "CREATING GPU STACK CONSTRAINTS"
echo "========================================"

"$PYTHON" - <<'PY' >/tmp/base-gpu-stack.constraints
import importlib.metadata

for package in (
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "pytorch-triton-rocm",
):
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        continue

    print(f"{package}=={version}")
PY

cat /tmp/base-gpu-stack.constraints

echo "========================================"
echo "INSTALLING VLLM ROCM DEPENDENCIES"
echo "========================================"

"${PIP[@]}" install \
    --constraint /tmp/base-gpu-stack.constraints \
    --requirement requirements/rocm.txt

CURRENT_TORCH=$(
    "$PYTHON" -c 'import torch; print(torch.__version__)'
)

if [[ "$CURRENT_TORCH" != "$EXPECTED_TORCH" ]]; then
    echo "ERROR: dependency installation replaced PyTorch"
    echo "Before: $EXPECTED_TORCH"
    echo "After:  $CURRENT_TORCH"
    exit 1
fi

echo "========================================"
echo "INSTALLING COMPRESSED-TENSORS COMPATIBLY"
echo "========================================"

"${PIP[@]}" install loguru

"${PIP[@]}" install \
    --no-deps \
    compressed-tensors==0.17.0

"$PYTHON" - <<'PY'
import importlib.metadata
import compressed_tensors
import torch

print("PyTorch:", torch.__version__)
print(
    "compressed-tensors:",
    importlib.metadata.version("compressed-tensors"),
)
print("compressed_tensors import: OK")
PY

echo "========================================"
echo "REMOVING CUDA-ONLY TVM EXTENSION"
echo "========================================"

"${PIP[@]}" uninstall -y torch-c-dlpack-ext || true

rm -rf \
  /root/.cache/tvm-ffi \
  /root/.cache/torch_extensions

"$PYTHON" - <<'PY'
import tvm_ffi
import xgrammar

print("tvm_ffi import: OK")
print("xgrammar import: OK")
PY

echo "========================================"
echo "INSTALLING MATCHING AMD SMI PYTHON PACKAGE"
echo "========================================"

AMD_SMI_SRC=""

for candidate in \
    /opt/rocm/share/amd_smi \
    /opt/rocm-7.2.3/share/amd_smi
do
    if [[ -d "$candidate" ]]; then
        AMD_SMI_SRC="$candidate"
        break
    fi
done

if [[ -z "$AMD_SMI_SRC" ]]; then
    echo "ERROR: matching AMD SMI source directory not found"
    exit 1
fi

echo "AMD SMI source: ${AMD_SMI_SRC}"

"${PIP[@]}" uninstall -y amdsmi 2>/dev/null || true

"${PIP[@]}" install \
    --force-reinstall \
    --no-deps \
    "$AMD_SMI_SRC"

"$PYTHON" - <<'PY'
import amdsmi

print("amdsmi:", amdsmi.__file__)

amdsmi.amdsmi_init()

try:
    handles = amdsmi.amdsmi_get_processor_handles()
    print("AMD SMI handles:", len(handles))
    assert len(handles) >= 1
finally:
    amdsmi.amdsmi_shut_down()

print("AMD SMI PASSED")
PY

echo "========================================"
echo "BUILDING VLLM"
echo "========================================"

cd /opt/vllm-src

rm -rf \
    build \
    dist \
    .eggs \
    vllm.egg-info

BUILD_LOG="${ROOT}/logs/vllm-source-build.${SLURM_JOB_ID:-manual}.log"

"${PIP[@]}" install \
    --editable . \
    --no-build-isolation \
    --no-deps \
    --verbose \
    2>&1 | tee "$BUILD_LOG"

if grep -F -- '--target=_C_stable_libtorch' "$BUILD_LOG"; then
    echo "ERROR: stable-libtorch target was unexpectedly built"
    exit 1
fi

CURRENT_TORCH=$(
    "$PYTHON" -c 'import torch; print(torch.__version__)'
)

if [[ "$CURRENT_TORCH" != "$EXPECTED_TORCH" ]]; then
    echo "ERROR: vLLM build replaced PyTorch"
    echo "Before: $EXPECTED_TORCH"
    echo "After:  $CURRENT_TORCH"
    exit 1
fi

echo "========================================"
echo "VALIDATING CUSTOM BUILD"
echo "========================================"

"$PYTHON" - <<'PY'
import importlib
import sys

import amdsmi
import compressed_tensors
import torch
import tvm_ffi
import vllm
import xgrammar
from vllm.platforms import current_platform

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("PyTorch file:", torch.__file__)
print("HIP:", torch.version.hip)
print("vLLM:", vllm.__version__)
print("vLLM file:", vllm.__file__)
print("Platform class:", current_platform.__class__)
print("Platform name:", current_platform.device_name)
print("Device type:", current_platform.device_type)
print("GPU count:", torch.cuda.device_count())

assert sys.executable == "/opt/venv/bin/python"
assert torch.__version__.startswith("2.9.1")
assert torch.__file__.startswith("/opt/venv/")
assert vllm.__file__.startswith("/opt/vllm-src/")
assert "RocmPlatform" in current_platform.__class__.__name__

for module in (
    "vllm._C",
    "vllm._rocm_C",
):
    importlib.import_module(module)
    print(module, "OK")

try:
    importlib.import_module("vllm._C_stable_libtorch")
except ModuleNotFoundError:
    print("vllm._C_stable_libtorch absent: expected")
else:
    raise RuntimeError(
        "vllm._C_stable_libtorch should not exist in this build"
    )

print("compressed_tensors import: OK")
print("tvm_ffi import: OK")
print("xgrammar import: OK")
print("CUSTOM PYTHON VALIDATION PASSED")
PY

command -v python
command -v vllm

vllm --version
vllm serve --help >/tmp/vllm-serve-help.txt

echo "========================================"
echo "CUSTOM VLLM BUILD PASSED"
echo "========================================"

rm -rf /root/.cache/pip
