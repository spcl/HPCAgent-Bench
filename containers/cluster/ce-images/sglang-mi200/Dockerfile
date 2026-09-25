# syntax=docker/dockerfile:1
#
# SGLang inference on AMD MI250X (gfx90a), Slurm partition mi200: 8 GPUs x 64 GiB per node.
# Sibling of ../sglang (MI300A, gfx942) on the SAME pinned base; README.md says what differs and why.
#
# Build context is the REPOSITORY ROOT:
#   podman build -f containers/cluster/ce-images/sglang-mi200/Dockerfile -t hpcagent-bench-sglang-mi200 .
# build.sh does that with the right podman flags; build.sbatch runs build.sh on an mi200 node.

# Same digest as ../sglang/Dockerfile; tests/test_sglang_mi200_image.py pins it.
ARG BASE_IMAGE=docker.io/lmsysorg/sglang-rocm:v0.5.19-rocm724-mi30x-20260908\
@sha256:0405baaf36945fa8164c57d4f1b6b178bae5804fae606ff2db3816a6cab6dafc

FROM ${BASE_IMAGE} AS final
USER root

# BASE_IMAGE is re-declared because an ARG before FROM is out of scope in a stage. BASE_IMAGE_REF is
# the registry reference, not the base-cache path BASE_IMAGE gets rewritten to.
ARG BASE_IMAGE
ARG BASE_IMAGE_REF="${BASE_IMAGE}"
# The one GPU arch every AOT device build below targets. Named, never detected: setup_rocm.py and
# cupy otherwise take the GPU visible at build time. build.sh passes it from ../gpu_arch.env for
# the build job's partition; there is no default.
ARG ROCM_ARCH
LABEL org.opencontainers.image.title="hpcagent-bench-sglang-mi200" \
      org.opencontainers.image.description="SGLang inference for MI250X (${ROCM_ARCH}): ROCm 7.2.4, sgl_kernel rebuilt for ${ROCM_ARCH}, aiter off" \
      org.opencontainers.image.version="0.5.19" \
      org.opencontainers.image.source="https://github.com/spcl/HPCAgent-Bench" \
      org.opencontainers.image.licenses="GPL-3.0-or-later" \
      org.opencontainers.image.base.name="${BASE_IMAGE_REF}"

# /opt/gpu-arch is the arch as a FILE: the CE does not reliably keep image ENV. The ENV overrides the
# base's GPU_ARCH_LIST and PYTORCH_ROCM_ARCH, which name gfx942.
RUN set -eu; \
    test -n "${ROCM_ARCH:-}" || { echo "ROCM_ARCH is unset: build with build.sh, which reads gpu_arch.env" >&2; exit 2; }; \
    printf '%s\n' "${ROCM_ARCH}" > /opt/gpu-arch
ENV HCC_AMDGPU_TARGET=${ROCM_ARCH} \
    PYTORCH_ROCM_ARCH=${ROCM_ARCH} \
    GPU_ARCHS=${ROCM_ARCH} \
    GPU_ARCH_LIST=${ROCM_ARCH}
COPY containers/cluster/ce-images/device_arch_gate.sh /usr/local/bin/device_arch_gate.sh

# cupy from source, the ../sglang recipe for ROCM_ARCH. numpy is constrained to the installed version
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

# Gate: cupy's device code targets ROCM_ARCH and nothing else (a build that autodetected another GPU).
RUN set -eux; \
    d="$(/opt/venv/bin/python3 -P -c 'import importlib.util as u; print(u.find_spec("cupy").submodule_search_locations[0])')"; \
    /usr/local/bin/device_arch_gate.sh --exact "${ROCM_ARCH}" "${d}"

# No fabric library in a prefix this image controls: RPATH would bind to it and shadow the CSCS
# netstack artifact the EDF hooks install.
RUN set -eux; \
    rm -f /opt/ofi-sdk/lib/libfabric.so*; \
    found="$(find /opt/ofi-sdk /opt/ofi /opt/aws-ofi-nccl \
                  \( -name 'libfabric.so*' -o -name 'libcxi.so*' -o -name 'librccl-net.so*' \) \
                  2>/dev/null || true)"; \
    [ -z "${found}" ] \
      || { echo "this image ships a fabric library in a prefix it controls:" >&2; \
           echo "${found}" >&2; exit 1; }

# sgl_kernel common_ops, REBUILT for ROCM_ARCH. The base's copy is built for another arch only, and
# it is on the serve path: SiluAndMul on HIP imports sgl_kernel.silu_and_mul.
#
# Three asserted edits to setup_rocm.py; a pattern that is not there exactly once fails the build.
#   * AMDGPU_TARGET wins over the visible GPU (stock code overwrites it with gcnArchName).
#   * ROCM_ARCH joins the allow-list (stock sys.exit(1) for anything but its hardcoded archs).
#   * -DENABLE_FP8 goes: no file under kernels/aot reads it, and MI250X has no FP8 types.
# The FP8 TYPE macro stays: other TUs #error without one. It only names a host-side c10 alias there.
# The one FP8-writing source, elementwise/dsv4_norm_rope.cu, takes its software uint8 packer on
# gfx90a instead, so the native-OCP branch is simply skipped rather than dropping the source (which
# would leave its m.impl registrations in common_extension_rocm.cc unresolved).
RUN /opt/venv/bin/python3 <<'PY'
import os
import pathlib
import sys

SETUP = pathlib.Path("/sgl-workspace/sglang/python/sglang/kernels/aot/setup_rocm.py")
ARCH = os.environ["ROCM_ARCH"]
# Upstream's allow-list line, verbatim from the pinned base.
ALLOW = 'if amdgpu_target not in ["gfx942", "gfx950", "gfx1250"]:\n'
EDITS = (
    ("if torch.cuda.is_available():\n", 'if "AMDGPU_TARGET" not in os.environ and torch.cuda.is_available():\n'),
    (ALLOW, ALLOW.replace("]:", f', "{ARCH}"]:')),
    ('    "-DENABLE_FP8",\n', ""),
)
text = SETUP.read_text()
for old, new in EDITS:
    count = text.count(old)
    if count != 1:
        sys.exit(f"{SETUP}: expected one {old!r}, found {count}; upstream changed, re-derive the {ARCH} edit")
    text = text.replace(old, new)
SETUP.write_text(text)
print(f"setup_rocm.py edited for {ARCH}", flush=True)
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

# Gate: the INSTALLED common_ops carries device code for ROCM_ARCH and nothing else.
RUN set -eux; \
    pkg="$(/opt/venv/bin/python3 -P -c 'import importlib.util as u; print(u.find_spec("sgl_kernel").submodule_search_locations[0])')"; \
    so="$(find "${pkg}" -maxdepth 1 -name 'common_ops.*.so')"; \
    test -f "${so}"; \
    /usr/local/bin/device_arch_gate.sh --exact "${ROCM_ARCH}" "${so}"

# The EDF restates these (the CE does not reliably keep image ENV); here they are under the digest.
# SGLANG_USE_AITER=0: aiter ships no kernels for this arch, and this image carries no aiter JIT prebuild.
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
# the launch check on an mi200 GPU; serving is inference/serve-private.sbatch.
