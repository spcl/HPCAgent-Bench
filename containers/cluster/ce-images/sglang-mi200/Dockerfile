# syntax=docker/dockerfile:1
#
# SGLang inference on AMD MI250X (gfx90a), Slurm partition mi200: 8 GPUs x 64 GiB per node.
# Sibling of ../sglang (MI300A, gfx942) on the SAME pinned base; README.md says what differs and why.
#
# Build context is the REPOSITORY ROOT:
#   podman build -f containers/cluster/ce-images/sglang-mi200/Dockerfile -t optarena-sglang-mi200 .
# build.sh does that with the right podman flags; build.sbatch runs build.sh on an mi200 node.

# Same digest as ../sglang/Dockerfile and both build.sh files; tests/test_sglang_mi200_image.py pins it.
ARG BASE_IMAGE=docker.io/lmsysorg/sglang-rocm:v0.5.19-rocm724-mi30x-20260908\
@sha256:0405baaf36945fa8164c57d4f1b6b178bae5804fae606ff2db3816a6cab6dafc

FROM ${BASE_IMAGE} AS final
USER root

# BASE_IMAGE is re-declared because an ARG before FROM is out of scope in a stage. BASE_IMAGE_REF is
# the registry reference, not the dir: path a base-cache hit rewrites BASE_IMAGE to.
ARG BASE_IMAGE
ARG BASE_IMAGE_REF="${BASE_IMAGE}"
LABEL org.opencontainers.image.title="optarena-sglang-mi200" \
      org.opencontainers.image.description="SGLang inference for MI250X (gfx90a): ROCm 7.2.4, sgl_kernel rebuilt for gfx90a, aiter off" \
      org.opencontainers.image.version="0.5.19" \
      org.opencontainers.image.source="https://github.com/spcl/HPCAgent-Bench" \
      org.opencontainers.image.licenses="GPL-3.0-or-later" \
      org.opencontainers.image.base.name="${BASE_IMAGE_REF}"

# The one GPU arch every AOT device build below targets. Named, never detected: setup_rocm.py and
# cupy otherwise take the GPU visible at build time, and on an mi300 node that is gfx942.
ARG ROCM_ARCH=gfx90a

# cupy from source, the ../sglang recipe for gfx90a. numpy is constrained to the installed version
# and the build fails if it moved.
ARG CUPY_VERSION=14.2.0
RUN set -eux; \
    numpy_before="$(/opt/venv/bin/python3 -c 'import numpy; print(numpy.__version__)')"; \
    printf 'numpy==%s\n' "${numpy_before}" > /tmp/constraint.txt; \
    CUPY_INSTALL_USE_HIP=1 ROCM_HOME=/opt/rocm HCC_AMDGPU_TARGET="${ROCM_ARCH}" \
    CUPY_NUM_BUILD_JOBS="$(nproc)" PIP_CONSTRAINT=/tmp/constraint.txt \
      /opt/venv/bin/python3 -m pip install --no-cache-dir "cupy==${CUPY_VERSION}"; \
    rm -f /tmp/constraint.txt; \
    test "$(/opt/venv/bin/python3 -c 'import numpy; print(numpy.__version__)')" = "${numpy_before}" \
      || { echo "cupy moved numpy off ${numpy_before}" >&2; exit 2; }; \
    /opt/venv/bin/python3 -c "import cupy; print('cupy', cupy.__version__)"

# Gate: no cupy extension carries gfx942 device code (a build that autodetected an MI300A GPU).
RUN set -eux; \
    command -v strings; \
    d="$(/opt/venv/bin/python3 -P -c 'import importlib.util as u; print(u.find_spec("cupy").submodule_search_locations[0])')"; \
    n90a="$(find "${d}" -name '*.so' -exec strings {} + | grep -c gfx90a || true)"; \
    n942="$(find "${d}" -name '*.so' -exec strings {} + | grep -c gfx942 || true)"; \
    echo "cupy device-code strings: gfx90a=${n90a} gfx942=${n942}"; \
    test "${n942}" -eq 0 || { echo "cupy carries gfx942 device code" >&2; exit 1; }

# No fabric library in a prefix this image controls: RPATH would bind to it and shadow the CSCS
# netstack artifact the EDF hooks install (../sglang/Dockerfile carries the job history).
RUN set -eux; \
    rm -f /opt/ofi-sdk/lib/libfabric.so*; \
    found="$(find /opt/ofi-sdk /opt/ofi /opt/aws-ofi-nccl \
                  \( -name 'libfabric.so*' -o -name 'libcxi.so*' -o -name 'librccl-net.so*' \) \
                  2>/dev/null || true)"; \
    [ -z "${found}" ] \
      || { echo "this image ships a fabric library in a prefix it controls:" >&2; \
           echo "${found}" >&2; exit 1; }

# sgl_kernel common_ops, REBUILT for gfx90a. The base's copy is gfx942-only (its build.ninja says
# --amdgpu-target=gfx942; 28 strings name gfx942, none gfx90a) and it is on the serve path:
# SiluAndMul on HIP imports sgl_kernel.silu_and_mul.
#
# Three asserted edits to setup_rocm.py; a pattern that is not there exactly once fails the build.
#   * AMDGPU_TARGET wins over the visible GPU (stock :70-72 overwrite it with gcnArchName).
#   * gfx90a joins the allow-list (stock :78-82 sys.exit(1) for anything but gfx942/gfx950/gfx1250).
#   * -DENABLE_FP8 goes: no file under kernels/aot reads it, and MI250X has no FP8 types.
# The FP8 TYPE macro stays: setup picks -DHIP_FP8_TYPE_E4M3 for non-gfx942 (:84-86), and
# include/utils.h:381-388 #errors without one in every TU that includes it. It only names a host-side
# c10 alias there. The one FP8-writing source in the list, elementwise/dsv4_norm_rope.cu (DeepSeek-V4
# ops), takes its software uint8 packer on gfx90a: HIP defines FNUZ=OCP=1 outside its FP8 arch list
# (amd_hip_fp8.h:33-51), so the native-OCP branch (:132) is skipped. Dropping that source instead
# would leave its three m.impl registrations in common_extension_rocm.cc:56-67 unresolved.
RUN /opt/venv/bin/python3 <<'PY'
import pathlib
import sys

SETUP = pathlib.Path("/sgl-workspace/sglang/python/sglang/kernels/aot/setup_rocm.py")
EDITS = (
    ("if torch.cuda.is_available():\n", 'if "AMDGPU_TARGET" not in os.environ and torch.cuda.is_available():\n'),
    (
        'if amdgpu_target not in ["gfx942", "gfx950", "gfx1250"]:\n',
        'if amdgpu_target not in ["gfx942", "gfx950", "gfx1250", "gfx90a"]:\n',
    ),
    ('    "-DENABLE_FP8",\n', ""),
)
text = SETUP.read_text()
for old, new in EDITS:
    count = text.count(old)
    if count != 1:
        sys.exit(f"{SETUP}: expected one {old!r}, found {count}; upstream changed, re-derive the gfx90a edit")
    text = text.replace(old, new)
SETUP.write_text(text)
print("setup_rocm.py edited for gfx90a", flush=True)
PY

RUN set -eux; \
    cd /sgl-workspace/sglang/python/sglang/kernels/aot; \
    rm -rf build; \
    AMDGPU_TARGET="${ROCM_ARCH}" MAX_JOBS="$(nproc)" /opt/venv/bin/python3 setup_rocm.py build_ext; \
    ninja_file="$(find build -name build.ninja)"; \
    grep -q -- "--amdgpu-target=${ROCM_ARCH}" "${ninja_file}"; \
    if grep -q -- '-DENABLE_FP8' "${ninja_file}"; then echo "ENABLE_FP8 survived the edit" >&2; exit 1; fi; \
    built="$(find build -name 'common_ops.*.so')"; \
    test "$(printf '%s\n' "${built}" | wc -l)" -eq 1; \
    pkg="$(/opt/venv/bin/python3 -P -c 'import importlib.util as u; print(u.find_spec("sgl_kernel").submodule_search_locations[0])')"; \
    installed="${pkg}/$(basename "${built}")"; \
    test -f "${installed}"; \
    cp -f "${built}" "${installed}"; \
    cmp "${built}" "${installed}"

# Gate: the INSTALLED common_ops carries gfx90a device code and nothing built for gfx942.
RUN set -eux; \
    pkg="$(/opt/venv/bin/python3 -P -c 'import importlib.util as u; print(u.find_spec("sgl_kernel").submodule_search_locations[0])')"; \
    so="$(find "${pkg}" -maxdepth 1 -name 'common_ops.*.so')"; \
    test -f "${so}"; \
    n90a="$(strings "${so}" | grep -c gfx90a || true)"; \
    n942="$(strings "${so}" | grep -c gfx942 || true)"; \
    echo "common_ops device-code strings: gfx90a=${n90a} gfx942=${n942}"; \
    test "${n90a}" -gt 0 || { echo "common_ops carries no gfx90a device code" >&2; exit 1; }; \
    test "${n942}" -eq 0 || { echo "common_ops still carries gfx942 device code" >&2; exit 1; }

# The EDF restates these (the CE does not reliably keep image ENV); here they are under the digest.
# SGLANG_USE_AITER=0: aiter ships no gfx90a kernels, and this image carries no aiter JIT prebuild.
ENV SGLANG_USE_AITER=0 \
    HSA_ENABLE_IPC_MODE_LEGACY=1 \
    SAFETENSORS_FAST_GPU=1 \
    HIP_FORCE_DEV_KERNARG=1 \
    TOKENIZERS_PARALLELISM=false \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64 \
    PATH=/usr/local/bin:/opt/rocm/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN /opt/venv/bin/python3 <<'PY'
import pathlib
import subprocess
import sys

import cupy
import sgl_kernel
import sglang
import torch
import triton

print("torch", torch.__version__, "triton", triton.__version__, "cupy", cupy.__version__, file=sys.stderr)
assert torch.version.hip is not None, "not a ROCm torch"
sglang_paths = [str(p) for p in sglang.__path__]
SGLANG_TREE = "/sgl-workspace/sglang"
assert any(p == SGLANG_TREE or p.startswith(SGLANG_TREE + "/") for p in sglang_paths), sglang_paths
subprocess.run([sys.executable, "-m", "sglang.launch_server", "--help"], check=True, stdout=subprocess.DEVNULL)

# The common_ops sgl_kernel loads is the file the gate above inspected.
ops = pathlib.Path(sgl_kernel.common_ops.__file__)
assert ops.parent == pathlib.Path(sgl_kernel.__file__).parent, ops
assert cupy.__file__.startswith("/opt/"), cupy.__file__
assert not pathlib.Path("/opt/aiter-jit").exists(), "an aiter JIT prebuild shipped"
for stray in ("libfabric.so.1", "libcxi.so.1", "librccl-net.so"):
    assert not list(pathlib.Path("/opt").rglob(stray)), f"{stray} ships and would shadow the artifact"
PY

WORKDIR /workspace

# LIMITATION: the build proves compile targets and imports, not a launch. verify_image.sbatch runs
# inference/sglang_kernel_launch_check.py on an mi200 GPU; serving is inference/serve-private.sbatch.
