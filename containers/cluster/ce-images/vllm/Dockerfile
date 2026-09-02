# syntax=docker/dockerfile:1
#
# vLLM inference on AMD MI300A (gfx942) -- the engine the 6 oss120b arms of llr40v9 serve on.
# Build context is containers/cluster/ce-images/inference/beverin-rocm723-host-ofi-phase1,
# which must first contain a freshly packed host SDK:
#   ./pack-beverin-host-sdk.sh beverin-host-sdk.tar.gz
#   podman build -f ../../vllm/Dockerfile -t optarena-vllm .
#
# NOT YET BUILT. This collapses into one file the six-job Slurm chain that actually produced the
# deployed image (inference/build/build-chain.sh: phase1 OFI build, phase1 gate, pytorch 2.11,
# numpy, vLLM 0.23.0, flash-attn). Read the LIMITATION note at the bottom before trusting it:
# the chain validates on real GPUs at build time and a podman build cannot.
#
# vLLM is PINNED to 0.23.0 and must not be "upgraded" as part of consolidating. 0.27.1 routes
# gpt-oss through its mxfp4 path in process_weights_after_loading regardless of --dtype bfloat16,
# and that path imports triton_kernels.matmul_ogs (fused_moe/oracle/mxfp4.py:1137). The ROCm
# triton_kernels 1.0.0+amd.rocm7.2.0 in this base has no matmul_ogs submodule -- only upstream
# PyPI 0.1.0 does, and swapping AMD's build out is unvalidated on gfx942. Jobs 601854-601857 all
# died there in ~7 minutes; 600516 is the last oss120b that served, and it ran on 0.23.0.
ARG BASE_IMAGE=docker.io/rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.9.1
ARG AWS_OFI_NCCL_REF=v1.20.0

# ---------------------------------------------------------------------------------------------
# Stage 1: the AWS OFI plugin, compiled against a packed copy of the Beverin host Slingshot SDK.
# Captured contract (beverin-rocm723-host-ofi-phase1/README-FIRST.md): host glibc 2.38, libfabric
# 2.3.1 -> libfabric.so.1.29.1, symbol ceiling FABRIC_1.8, libcxi.so.1.5.0. The plugin is compiled
# against those headers but the runtime libfabric/libcxi come from the CE host hook, so no
# libfabric or libcxi may be copied into the final image -- the last RUN here asserts that.
# ---------------------------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS ofi-builder
ARG BASE_IMAGE
ARG AWS_OFI_NCCL_REF
USER root
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      autoconf automake binutils build-essential ca-certificates git \
      libhwloc-dev libnuma-dev libtool make patchelf pkg-config \
 && rm -rf /var/lib/apt/lists/*

COPY beverin-host-sdk.tar.gz /tmp/beverin-host-sdk.tar.gz
RUN tar -C / -xzf /tmp/beverin-host-sdk.tar.gz \
 && rm /tmp/beverin-host-sdk.tar.gz \
 && test -f /opt/beverin-sdk/include/rdma/fabric.h \
 && test -e /opt/beverin-sdk/lib64/libfabric.so.1 \
 && test -x /opt/beverin-sdk/bin/fi_info

WORKDIR /tmp/build
RUN git clone --branch "${AWS_OFI_NCCL_REF}" --depth 1 \
      https://github.com/aws/aws-ofi-nccl.git aws-ofi-nccl \
 && cd aws-ofi-nccl \
 && git rev-parse HEAD | tee /tmp/aws-ofi-nccl.commit \
 && ./autogen.sh \
 && env \
      PKG_CONFIG_PATH=/opt/beverin-sdk/lib64/pkgconfig \
      LD_LIBRARY_PATH=/opt/beverin-sdk/lib64:/opt/rocm/lib \
      CPPFLAGS='-I/opt/beverin-sdk/include' \
      LDFLAGS='-L/opt/beverin-sdk/lib64 -Wl,-rpath-link,/opt/beverin-sdk/lib64' \
      ./configure \
        --prefix=/opt/aws-ofi-nccl \
        --with-libfabric=/opt/beverin-sdk \
        --with-rocm=/opt/rocm \
        --disable-tests \
 && make -j"$(nproc)" && make install

# RCCL loads libnccl-net.so; the plugin installs as librccl-net.so. Strip the builder rpath so
# nothing resolves back to /opt/beverin-sdk at run time, and refuse any symbol above the host's
# FABRIC_1.8 ceiling -- that is what makes the plugin loadable against the host hook's libfabric.
RUN set -eux; \
    test -e /opt/aws-ofi-nccl/lib/librccl-net.so; \
    ln -sfn librccl-net.so /opt/aws-ofi-nccl/lib/libnccl-net.so; \
    find /opt/aws-ofi-nccl/lib -maxdepth 1 -type f -name '*.so*' -exec patchelf --remove-rpath {} \; ; \
    ! readelf -d /opt/aws-ofi-nccl/lib/librccl-net.so | grep -E 'RPATH|RUNPATH.*beverin-sdk'; \
    ! readelf --version-info /opt/aws-ofi-nccl/lib/librccl-net.so \
        | grep -E 'FABRIC_1\.(9|[1-9][0-9])|FABRIC_[2-9]'

# ---------------------------------------------------------------------------------------------
# Stage 2: PyTorch 2.11 + vLLM 0.23.0 + flash-attn, in their own venv at /opt/pytorch211.
# A SEPARATE venv from the base image's /opt/venv, deliberately: the base ships torch 2.9.1 and
# the chain kept it intact as a control.
# ---------------------------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS final
USER root

ARG VLLM_VERSION=0.23.0
ARG TORCH_VERSION=2.11.0+rocm7.2
ARG TORCHVISION_VERSION=0.26.0+rocm7.2
ARG TORCHAUDIO_VERSION=2.11.0+rocm7.2
ARG FLASH_ATTN_VERSION=2.8.3
ARG TORCH_INDEX=https://download.pytorch.org/whl/rocm7.2

ENV VIRTUAL_ENV=/opt/pytorch211 \
    ROCM_PATH=/opt/rocm \
    PYTORCH_ROCM_ARCH=gfx942 \
    VLLM_TARGET_DEVICE=rocm \
    HSA_ENABLE_IPC_MODE_LEGACY=1 \
    SAFETENSORS_FAST_GPU=1 \
    HIP_FORCE_DEV_KERNARG=1 \
    TOKENIZERS_PARALLELISM=false \
    PATH=/opt/pytorch211/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/rocm/lib:/opt/rocm/lib64
# Read at IMPORT, not only at build: flash_attn_interface.py:11 picks .flash_attn_triton_amd on
# TRUE and imports flash_attn_2_cuda -- the compiled CK extension a triton-only install does NOT
# have -- on FALSE. vLLM wraps that import in try/except and merely reports the backend
# unavailable, which is the 589001 "No valid MLA prefill backend" failure with the package
# installed and looking fine. It must ALSO be in the EDF, because ENV alone does not survive
# every launcher path.
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

COPY --from=ofi-builder /opt/aws-ofi-nccl /opt/aws-ofi-nccl

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential ca-certificates git ninja-build pkg-config python3-dev \
      libdrm-dev libnuma-dev libfmt-dev libmsgpack-dev libsuitesparse-dev \
      libhwloc15 libnuma1 numactl pciutils procps \
 && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    python3.12 -m venv /opt/pytorch211; \
    /opt/pytorch211/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel; \
    /opt/pytorch211/bin/python -m pip install --no-cache-dir \
      --index-url "${TORCH_INDEX}" "torch==${TORCH_VERSION}"; \
    /opt/pytorch211/bin/python -m pip install --no-cache-dir --no-deps \
      --index-url "${TORCH_INDEX}" \
      "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}"; \
    /opt/pytorch211/bin/python -m pip install --no-cache-dir 'pillow>=10.0' 'numpy<3'

# pybind11 pinned below 3.1: 3.1.0 bumped its internals id v11 -> v12 and refuses to share
# registered types across that boundary, so a v12 build cannot accept tensors from the v11 torch
# extensions (ROCm/aiter#4770, vllm-project/vllm#52400).
RUN /opt/pytorch211/bin/python -m pip install --no-cache-dir --upgrade \
      'cmake>=3.26.1,<4' ninja 'pybind11==3.0.*' 'packaging>=24.2' \
      'setuptools>=77.0.3,<81' 'setuptools-scm>=8' 'setuptools-rust>=1.9.0' wheel \
      'jinja2>=3.1.6' more-itertools

# use_existing_torch.py strips every torch/torchvision/torchaudio pin out of vLLM's requirements
# so pip cannot replace the ROCm build with a CUDA wheel; the grep afterwards is what proves it
# worked. The constraint file pins the three back for the dependency resolve.
RUN set -eux; \
    git clone --branch "v${VLLM_VERSION}" --depth 1 https://github.com/vllm-project/vllm.git /opt/vllm-src; \
    cd /opt/vllm-src; \
    /opt/pytorch211/bin/python use_existing_torch.py --prefix; \
    ! grep -RniE '^[[:space:]]*(torch|torchvision|torchaudio)[[:space:]]*[=<>]' requirements pyproject.toml; \
    printf 'torch==%s\ntorchvision==%s\ntorchaudio==%s\n' \
      "${TORCH_VERSION}" "${TORCHVISION_VERSION}" "${TORCHAUDIO_VERSION}" > /tmp/pt211-rocm.constraints; \
    /opt/pytorch211/bin/python -m pip install --no-cache-dir \
      --constraint /tmp/pt211-rocm.constraints --requirement requirements/rocm.txt; \
    /opt/pytorch211/bin/python -m pip install --no-cache-dir --no-deps --force-reinstall \
      /opt/rocm/share/amd_smi; \
    /opt/pytorch211/bin/python -m pip uninstall -y torch-c-dlpack-ext || true

ENV MAX_JOBS=32 \
    CMAKE_BUILD_TYPE=Release
RUN set -eux; \
    cd /opt/vllm-src; \
    SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}" VLLM_VERSION_OVERRIDE="${VLLM_VERSION}" \
      /opt/pytorch211/bin/python -m pip install --editable . --no-build-isolation --no-deps

# Triton path, not composable-kernel: SKIP_CK_BUILD turns a multi-hour gfx942 CK compile into a
# minutes-long pure-Triton install. --no-build-isolation because flash-attn's setup.py imports
# torch and must see the image's ROCm build, not a fresh wheel.
RUN set -eux; \
    FLASH_ATTENTION_SKIP_CK_BUILD=TRUE \
      /opt/pytorch211/bin/python -m pip install --no-cache-dir --no-build-isolation \
        "flash-attn==${FLASH_ATTN_VERSION}"

# What can be checked WITHOUT a GPU. Everything below is import- and metadata-level; the real
# gates (device_count()==4, an allreduce, which MLA backend the selector returns) need hardware
# and live in the smoke test -- see LIMITATION.
RUN <<'PY' /opt/pytorch211/bin/python
import importlib, importlib.metadata as md, sys
for pkg, want in (("torch", "2.11.0+rocm7.2"), ("torchvision", "0.26.0+rocm7.2"),
                  ("torchaudio", "2.11.0+rocm7.2")):
    got = md.version(pkg)
    assert got == want, f"{pkg}: expected {want}, got {got}"
cuda = [d.metadata["Name"] for d in md.distributions()
        if (d.metadata.get("Name") or "").lower().startswith("nvidia-")]
assert not cuda, f"CUDA packages leaked in: {cuda}"
import torch, vllm
assert torch.version.hip is not None, "not a ROCm torch"
assert torch.__file__.startswith("/opt/pytorch211/")
assert vllm.__file__.startswith("/opt/vllm-src/")
for mod in ("vllm._C", "vllm._rocm_C", "vllm._C_stable_libtorch"):
    importlib.import_module(mod)
from flash_attn import flash_attn_interface
assert flash_attn_interface.USE_TRITON_ROCM, "imported the CK path, not triton_amd"
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
assert is_flash_attn_varlen_func_available(), "vLLM still reports flash-attn unavailable"
print("vllm", vllm.__version__, "torch", torch.__version__, file=sys.stderr)
PY

# aiter is deliberately NOT installed. It is a 0.27.1 requirement (that generation's Kimi ViT
# patch-embed does a bare `from aiter.ops.triton...` inside the multimodal dummy profile_run, so
# an aiter-less 0.27.1 dies at startup -- job 600649). 0.23.0 has no such import; here aiter would
# only be a MoE speed option, and its master switch breaks MLA prefill on gfx942
# (fmha_v3_varlen_fwd invalid arg). Add it with inference/build/add-aiter-pt211.sbatch if a future
# arm needs the tuned MoE kernels, and prebuild its JIT cache offline first.

WORKDIR /workspace

# LIMITATION -- read before trusting this file.
# The Slurm chain this replaces ran its install steps INSIDE a container on a 4-GPU node
# (enroot start --root --rw, then enroot export) and asserted torch.cuda.device_count() == 4,
# ran a real allreduce, and confirmed the MLA prefill selector returns FlashAttnPrefillBackend.
# A podman build has no GPUs, so those gates CANNOT run here and were moved to the smoke test.
# This Dockerfile therefore proves strictly less at build time than the chain did. Until it has
# been built AND smoke-tested end to end, inference/build/build-chain.sh remains the authority
# and this file is a candidate, not a replacement.
