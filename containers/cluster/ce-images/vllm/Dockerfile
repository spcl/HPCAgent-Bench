# syntax=docker/dockerfile:1
#
# vLLM inference on AMD MI300A (gfx942). Serves oss120b, and nothing else: every campaign that
# names a vLLM engine names vLLM 0.23.0, and the 12 of 18 v9 arms that serve qwen38 or kimi go to
# the SGLang image instead.
#
# Build context is the REPOSITORY ROOT, because this file bakes in two things that live in the
# repo today and are reached from outside the image at run time:
#   podman build -f containers/cluster/ce-images/vllm/Dockerfile -t optarena-vllm .
# build.sh does that with the right podman flags; build.sbatch runs build.sh on a compute node.
#
# NOT YET BUILT. It collapses into one file the six-job Slurm chain that produced the deployed
# image (inference/build/build-chain.sh: phase1 OFI, phase1 gate, pytorch 2.11, numpy, vLLM
# 0.23.0, flash-attn) plus the two in-place patch jobs that ran after it (add-aiter-pt211,
# add-flash-attn-pt211). Read LIMITATION at the bottom before treating it as equivalent: the
# chain validated on real GPUs at build time and a podman build cannot.
#
# vLLM is PINNED to 0.23.0. The recorded reason is that 0.27.1 routes gpt-oss through its mxfp4
# path in process_weights_after_loading regardless of --dtype bfloat16, and that path imports
# triton_kernels.matmul_ogs (fused_moe/oracle/mxfp4.py:1137), which AMD's triton_kernels does not
# have -- jobs 601854-601857 died there in ~7 minutes. That reason has since been MEASURED to be
# repairable in one line, but the repair has only been proven to the import, not to a served and
# accuracy-gated model, so the pin stands here. See the note at the end of this file.
ARG BASE_IMAGE=docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1\
@sha256:a3b65813621095e3389269417e963725b59310184588c9d2490d44e6e83fa01c

# ------------------------------------------------------------------------------------------------
# Stage 1: the AWS OFI plugin (RCCL-OFI), compiled against libfabric built from source at the
# version the Beverin host runs.
#
# The captured host contract (inference/beverin-rocm723-host-ofi-phase1/README-FIRST.md) is:
# host glibc 2.38, libfabric 2.3.1 -> libfabric.so.1.29.1, symbol ceiling FABRIC_1.8,
# libcxi.so.1.5.0, and the runtime libfabric/libcxi arrive from the CE host hook
# (com.hooks.netstack.source = "host"). So the plugin must COMPILE against 2.3.1 headers and must
# LINK no higher than FABRIC_1.8, and no libfabric or libcxi may ship in the final image.
#
# The chain met that by packing an SDK off a live login node with pack-beverin-host-sdk.sh, which
# is a "run this script first" step -- exactly what one-Dockerfile-end-to-end forbids, and it also
# makes the image depend on whatever the login node happened to have that day. Building libfabric
# v2.3.1 from ofiwg source instead makes the compile-time ABI a pinned, reproducible input. The
# assertions below are what keep that honest: the plugin must need only libfabric.so.1, and must
# reference no symbol above FABRIC_1.8.
# ------------------------------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS ofi-builder
ARG BASE_IMAGE
ARG LIBFABRIC_REF=v2.3.1
ARG AWS_OFI_NCCL_REF=v1.20.0
USER root

# Every network git call goes through this, and where a mirror is mounted none of them
# reaches GitHub at all. The `insteadOf` rewrite is what covers the callers the wrapper
# cannot: spack clones its package repo in-process and vLLM's CMake FetchContent clones
# triton itself, and both died on the same rate limiting with no retry of their own.
# Conditional, so a build with no mirror mounted behaves exactly as it did before.
# `protocol.file.allow` goes with it: git has refused local-path submodule clones since the
# CVE-2022-39253 fix, and the rewrite turns every submodule URL into exactly that, so
# `submodule update` aborts with `transport 'file' not allowed` before it looks for the repo.
# Every network git call goes through this. An unauthenticated clone or fetch from a busy egress
# IP gets a 401, which git reports as `could not read Username for https://github.com`: it looks
# like a credentials failure and is a transient one. GIT_TERMINAL_PROMPT=0 makes it fail instead
# of blocking on a prompt no build has a terminal for; the backoff is what actually fixes it.
RUN set -eux; \
    printf '%s\n' \
      '#!/bin/sh' \
      'set -eu' \
      'export GIT_TERMINAL_PROMPT=0' \
      'n=1' \
      'while :; do' \
      '  git "$@" && exit 0' \
      '  [ "${n}" -ge 10 ] && exit 1' \
      '  d=$((n * n * 10)); if [ "${d}" -gt 300 ]; then d=300; fi' \
      '  sleep "${d}"' \
      '  n=$((n + 1))' \
      'done' \
      > /usr/local/bin/gitretry; \
    chmod 0755 /usr/local/bin/gitretry; \
    if [ -d /git-mirrors ]; then \
      git config --global "url./git-mirrors/.insteadOf" "https://github.com/"; \
      git config --global --add safe.directory '*'; \
      git config --global protocol.file.allow always; \
    fi

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      autoconf automake binutils build-essential ca-certificates git \
      libhwloc-dev libnuma-dev libtool make patchelf pkg-config \
 && rm -rf /var/lib/apt/lists/*

# Compile-only libfabric. --disable-verbs and friends keep it to the sockets/tcp core: the plugin
# needs the headers and the .so to link against, never a provider, because at run time the host
# hook's libfabric (with cxi) is what gets loaded.
WORKDIR /tmp/build
RUN set -eux; \
    gitretry clone --branch "${LIBFABRIC_REF}" --depth 1 https://github.com/ofiwg/libfabric.git libfabric; \
    cd libfabric; \
    git rev-parse HEAD > /tmp/libfabric.commit; \
    ./autogen.sh; \
    ./configure --prefix=/opt/ofi-sdk --disable-verbs --disable-psm3 --disable-efa --disable-usnic \
        --disable-opx --disable-shm --enable-sockets --enable-tcp; \
    make -j"$(nproc)"; \
    make install; \
    test -e /opt/ofi-sdk/lib/libfabric.so.1; \
    readlink -f /opt/ofi-sdk/lib/libfabric.so.1 > /tmp/libfabric.soname

RUN set -eux; \
    gitretry clone --branch "${AWS_OFI_NCCL_REF}" --depth 1 https://github.com/aws/aws-ofi-nccl.git aws-ofi-nccl; \
    cd aws-ofi-nccl; \
    git rev-parse HEAD > /tmp/aws-ofi-nccl.commit; \
    ./autogen.sh; \
    env \
      PKG_CONFIG_PATH=/opt/ofi-sdk/lib/pkgconfig \
      LD_LIBRARY_PATH=/opt/ofi-sdk/lib:/opt/rocm/lib \
      CPPFLAGS='-I/opt/ofi-sdk/include' \
      LDFLAGS='-L/opt/ofi-sdk/lib -Wl,-rpath-link,/opt/ofi-sdk/lib' \
      ./configure --prefix=/opt/aws-ofi-nccl --with-libfabric=/opt/ofi-sdk --with-rocm=/opt/rocm \
        --disable-tests; \
    make -j"$(nproc)"; \
    make install

# RCCL loads libnccl-net.so; the plugin installs as librccl-net.so. Strip the builder rpath so
# nothing resolves back to /opt/ofi-sdk at run time, and refuse any symbol above the host's
# FABRIC_1.8 ceiling -- that is what makes the plugin loadable against the host hook's libfabric.
RUN set -eux; \
    test -e /opt/aws-ofi-nccl/lib/librccl-net.so; \
    ln -sfn librccl-net.so /opt/aws-ofi-nccl/lib/libnccl-net.so; \
    find /opt/aws-ofi-nccl/lib -maxdepth 1 -type f -name '*.so*' -exec patchelf --remove-rpath {} \; ; \
    ! readelf -d /opt/aws-ofi-nccl/lib/librccl-net.so | grep -E 'RPATH|RUNPATH'; \
    readelf -d /opt/aws-ofi-nccl/lib/librccl-net.so | grep -q 'NEEDED.*libfabric\.so\.1'; \
    ! readelf --version-info /opt/aws-ofi-nccl/lib/librccl-net.so \
        | grep -E 'FABRIC_1\.(9|[1-9][0-9])|FABRIC_[2-9]'; \
    ! find /opt/aws-ofi-nccl \( -name 'libfabric.so*' -o -name 'libcxi.so*' \) | grep .

RUN { \
      echo "base_image=${BASE_IMAGE}"; \
      echo "libfabric_ref=${LIBFABRIC_REF}"; \
      printf 'libfabric_commit='; cat /tmp/libfabric.commit; \
      printf 'libfabric_soname='; basename "$(cat /tmp/libfabric.soname)"; \
      echo "aws_ofi_nccl_ref=${AWS_OFI_NCCL_REF}"; \
      printf 'aws_ofi_nccl_commit='; cat /tmp/aws-ofi-nccl.commit; \
      echo 'compile_fabric_symbol_ceiling=FABRIC_1.8'; \
      echo 'runtime_libfabric=provided_by_CSCS_host_hook'; \
    } > /opt/aws-ofi-nccl/BUILD-MANIFEST.txt

# ------------------------------------------------------------------------------------------------
# Stage 2: PyTorch 2.11 + vLLM 0.23.0 + flash-attn + aiter, in their own venv at /opt/pytorch211.
# A SEPARATE venv from the base image's /opt/venv, deliberately: the base ships torch 2.9.1 and
# the chain kept it intact as a control.
# ------------------------------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS final
USER root

ARG VLLM_VERSION=0.23.0
ARG TORCH_VERSION=2.11.0+rocm7.2
ARG TORCHVISION_VERSION=0.26.0+rocm7.2
ARG TORCHAUDIO_VERSION=2.11.0+rocm7.2
ARG FLASH_ATTN_VERSION=2.8.3
# The aiter that is IN the deployed 0.23.0 image, read back from it: amd_aiter 0.1.13.post1 at
# git 3a38da3991275e02e98a7906b7503b7087d67be3. Pinned to the tag rather than discovered from
# vLLM's docker/ files, because 0.27.1 spells that pin `main` -- a moving branch, which is how
# 600659 built whatever main was that day.
ARG AITER_REF=v0.1.13.post1
ARG TORCH_INDEX=https://download.pytorch.org/whl/rocm7.2

ENV VIRTUAL_ENV=/opt/pytorch211 \
    ROCM_PATH=/opt/rocm \
    PYTORCH_ROCM_ARCH=gfx942 \
    GPU_ARCHS=gfx942 \
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

# aiter's master switch, OFF, stated in the image rather than left to the launcher. Turning it on
# is what broke 610251/610252 (RPC timeout at step_counter=0), 600662 (MLA prefill on gfx942,
# fmha_v3_varlen_fwd invalid arg) and both qwen38 legs of 610203/610204. Every oss120b arm that
# has ever finished ran with it off. run_cluster.sh reads it as ${VLLM_ROCM_USE_AITER:-0}, so this
# value stands unless an env file overrides it -- which is the intended way to retry aiter.
ENV VLLM_ROCM_USE_AITER=0

COPY --from=ofi-builder /opt/aws-ofi-nccl /opt/aws-ofi-nccl

# Every network git call goes through this, and where a mirror is mounted none of them
# reaches GitHub at all. The `insteadOf` rewrite is what covers the callers the wrapper
# cannot: spack clones its package repo in-process and vLLM's CMake FetchContent clones
# triton itself, and both died on the same rate limiting with no retry of their own.
# Conditional, so a build with no mirror mounted behaves exactly as it did before.
# `protocol.file.allow` goes with it: git has refused local-path submodule clones since the
# CVE-2022-39253 fix, and the rewrite turns every submodule URL into exactly that, so
# `submodule update` aborts with `transport 'file' not allowed` before it looks for the repo.
# Every network git call goes through this. An unauthenticated clone or fetch from a busy egress
# IP gets a 401, which git reports as `could not read Username for https://github.com`: it looks
# like a credentials failure and is a transient one. GIT_TERMINAL_PROMPT=0 makes it fail instead
# of blocking on a prompt no build has a terminal for; the backoff is what actually fixes it.
RUN set -eux; \
    printf '%s\n' \
      '#!/bin/sh' \
      'set -eu' \
      'export GIT_TERMINAL_PROMPT=0' \
      'n=1' \
      'while :; do' \
      '  git "$@" && exit 0' \
      '  [ "${n}" -ge 10 ] && exit 1' \
      '  d=$((n * n * 10)); if [ "${d}" -gt 300 ]; then d=300; fi' \
      '  sleep "${d}"' \
      '  n=$((n + 1))' \
      'done' \
      > /usr/local/bin/gitretry; \
    chmod 0755 /usr/local/bin/gitretry; \
    if [ -d /git-mirrors ]; then \
      git config --global "url./git-mirrors/.insteadOf" "https://github.com/"; \
      git config --global --add safe.directory '*'; \
      git config --global protocol.file.allow always; \
    fi

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
# extensions (ROCm/aiter#4770, vllm-project/vllm#52400). This is the ABI split behind the
# "invalid argument for fmha_v3_varlen_fwd" of 600662/600673/600688/600689 -- not a bad kernel.
RUN /opt/pytorch211/bin/python -m pip install --no-cache-dir --upgrade \
      'cmake>=3.26.1,<4' ninja 'pybind11==3.0.*' 'packaging>=24.2' \
      'setuptools>=77.0.3,<81' 'setuptools-scm>=8' 'setuptools-rust>=1.9.0' wheel \
      'jinja2>=3.1.6' more-itertools uv

# use_existing_torch.py strips every torch/torchvision/torchaudio pin out of vLLM's requirements
# so pip cannot replace the ROCm build with a CUDA wheel; the grep afterwards is what proves it
# worked. The constraint file pins the three back for the dependency resolve.
RUN set -eux; \
    gitretry clone --branch "v${VLLM_VERSION}" --depth 1 https://github.com/vllm-project/vllm.git /opt/vllm-src; \
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

# aiter. It is in the deployed image and dropping it here would be a regression, so it ships --
# with its master switch off above. --no-build-isolation because aiter's setup imports torch and
# must see the image torch, not a fresh wheel.
RUN set -eux; \
    gitretry clone https://github.com/ROCm/aiter.git /opt/aiter-src; \
    cd /opt/aiter-src; \
    git checkout "${AITER_REF}"; \
    git submodule sync; \
    gitretry submodule update --init --recursive; \
    git describe --tags --exact-match; \
    git rev-parse HEAD; \
    /opt/pytorch211/bin/python -m uv pip install --python /opt/pytorch211/bin/python --no-cache \
      ninja cmake 'pybind11==3.0.*' pandas; \
    /opt/pytorch211/bin/python -m uv pip install --python /opt/pytorch211/bin/python \
      --no-build-isolation --no-cache .

# NO aiter mask patch here, deliberately. The sibling 0.27.1 image carries one for the ASM v3
# varlen path, and it does NOT belong in this image: its anchor is aiter v0.1.20's
# can_impl_fmha_v3_fwd (mha.py:2858), and this image pins v0.1.13.post1, which has neither
# _fmha_kv_byte_extent_ge_u32 nor that guard. Carried over anyway, the patch's own assert
# stopped the build at `anchor count 0` -- which is the assert doing its job rather than
# silently no-op'ing. This image also ships VLLM_ROCM_USE_AITER=0, so the path is off by
# default. If a future pin here needs the guard, write it against THIS image's aiter.

# aiter's JIT, built HERE instead of on the first request.
#
# aiter ships no prebuilt .so. Every process JIT-builds module_aiter_core (21-58 s measured) and
# then each op module lazily on FIRST USE -- behind a baton lock in AITER_JIT_DIR that outlives the
# engine's RPC deadline. That is 598021 (no /v1/models for 5400 s), 610164 (300 s watchdog) and
# 610251/610252 (TimeoutError on sample_tokens at step_counter=0). prebuild-aiter-jit.sbatch worked
# around it as a separate GPU job; the wedge is a property of the image, so it is fixed in the image.
#
# No GPU is needed for this: the modules compile with hipcc against GPU_ARCHS=gfx942, and the build
# was measured completing on a device-less login node. Only a kernel LAUNCH needs hardware, and
# none is issued here -- which is also why this cannot prove the modules run, only that they built.
#
# AITER_JIT_DIR points INSIDE the image so a hit is a read off the squashfs. A miss still writes,
# into the per-job CE overlay, so an op this prebuild did not cover degrades to today's behaviour
# rather than failing. run_cluster.sh reads it as ${AITER_JIT_DIR:-...}, so this value wins over
# its scratch default without a launcher change.
ENV AITER_JIT_DIR=/opt/aiter-jit
RUN mkdir -p /opt/aiter-jit /tmp/aiter-home
RUN HOME=/tmp/aiter-home /opt/pytorch211/bin/python <<'PY'
import aiter

print("aiter core built at", aiter.__file__, flush=True)
# The op families vLLM can dispatch to. Each import is guarded because coverage varies by aiter
# build and a family this release does not carry must not abort the image build.
for name in ("aiter.ops.rmsnorm", "aiter.ops.norm", "aiter.ops.gemm_op_a8w8", "aiter.ops.moe_op",
             "aiter.ops.mha", "aiter.ops.attention", "aiter.ops.rope", "aiter.ops.quant",
             "aiter.ops.shuffle", "aiter.fused_moe"):
    try:
        __import__(name)
        print("built:", name, flush=True)
    except Exception as exc:
        print(f"skip {name}: {type(exc).__name__}: {exc}", flush=True)
PY
# A prebuild that produced nothing is the wedge with extra steps, so it fails the build.
RUN set -eux; \
    find /opt/aiter-jit -name '*.so' | wc -l | xargs printf 'aiter prebuilt modules: %s\n'; \
    test -n "$(find /opt/aiter-jit -name '*.so' -print -quit)"; \
    rm -rf /tmp/aiter-home

# Tuned fused_moe Triton configs, keyed by (experts, N, device, dtype). Losing these once cost a
# whole set of throughput numbers: without them kimi served on vLLM's default MoE config at ~90
# tok/s aggregate, ~11x off the reference for its shape (595040/595049). E=128,N=192 is oss120b's,
# which is this image's model; E=384,N=512 int4 is kimi's, kept because vLLM is still the engine
# that can serve it and a config that exists nowhere is a config that is gone.
#
# run_cluster.sh:181 sets VLLM_TUNED_CONFIG_FOLDER only when the repo folder exists and only when
# the variable is unset, so this ENV is what a launcher with that folder deleted will use.
COPY containers/cluster/ce-images/inference/moe-configs/ /opt/moe-configs/
ENV VLLM_TUNED_CONFIG_FOLDER=/opt/moe-configs

# The eager-PG / PP-collective-split / MLA-lse-layout patch, moved inside the image.
#
# NOT in site-packages, deliberately. A bare sitecustomize.py there runs at interpreter startup for
# every process in this venv -- pip included -- and imports torch unconditionally, and its patches
# default ON (VLLM_PP_COLLECTIVE_SPLIT, VLLM_FIX_MLA_LSE_LAYOUT), so installing it that way would
# silently apply them to arms that never asked. Its own directory keeps the opt-in that
# VLLM_EAGER_PG_PATCH=1 expresses, while putting the file itself under the image digest.
COPY containers/cluster/ce-images/inference/external-eager-pg-patch/sitecustomize.py \
     /opt/vllm-eager-pg/sitecustomize.py

# What can be checked WITHOUT a GPU. Everything here is import- and metadata-level; the real gates
# (device_count()==4, an allreduce, which MLA backend the selector returns) need hardware and live
# in the smoke test -- see LIMITATION.
RUN /opt/pytorch211/bin/python <<'PY'
import importlib
import importlib.metadata as md
import sys

for pkg, want in (("torch", "2.11.0+rocm7.2"), ("torchvision", "0.26.0+rocm7.2"),
                  ("torchaudio", "2.11.0+rocm7.2")):
    got = md.version(pkg)
    assert got == want, f"{pkg}: expected {want}, got {got}"
print("amd_aiter", md.version("amd_aiter"), file=sys.stderr)
cuda = [d.metadata["Name"] for d in md.distributions()
        if (d.metadata.get("Name") or "").lower().startswith("nvidia-")]
assert not cuda, f"CUDA packages leaked in: {cuda}"

import torch
import vllm

assert torch.version.hip is not None, "not a ROCm torch"
assert torch.__file__.startswith("/opt/pytorch211/")
assert vllm.__file__.startswith("/opt/vllm-src/")
for mod in ("vllm._C", "vllm._rocm_C", "vllm._C_stable_libtorch"):
    importlib.import_module(mod)

from flash_attn import flash_attn_interface
assert flash_attn_interface.USE_TRITON_ROCM, "imported the CK path, not triton_amd"
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
assert is_flash_attn_varlen_func_available(), "vLLM still reports flash-attn unavailable"

import vllm.envs as envs
assert not envs.VLLM_ROCM_USE_AITER, "the aiter master switch is on in the image"
from vllm._aiter_ops import is_aiter_found
assert is_aiter_found(), "vLLM does not find the aiter package"

# The baked-in inputs, asserted rather than assumed: a COPY that silently landed nothing is how a
# tuned-MoE folder goes missing again.
import json
import pathlib

configs = sorted(pathlib.Path("/opt/moe-configs").glob("E=*.json"))
assert len(configs) >= 2, f"tuned MoE configs missing: {configs}"
for c in configs:
    json.loads(c.read_text())
patch = pathlib.Path("/opt/vllm-eager-pg/sitecustomize.py")
assert "patch_pp_collectives" in patch.read_text(), "eager-pg patch did not land"
assert not pathlib.Path("/opt/pytorch211/lib/python3.12/site-packages/sitecustomize.py").exists(), \
    "the eager-pg patch is unconditionally active"

print("vllm", vllm.__version__, "torch", torch.__version__, file=sys.stderr)
PY

# Drop the mirror rewrite from the SHIPPED image. It is a build-time convenience pointing at a
# path that exists only while the mirror is mounted; left in /root/.gitconfig, every git clone a
# user runs inside this container would silently rewrite to a directory that is not there.
RUN git config --global --unset-all "url./git-mirrors/.insteadOf" || true; \
    git config --global --unset-all protocol.file.allow || true

WORKDIR /workspace

# LIMITATION -- read before trusting this file.
#
# The Slurm chain this replaces ran its install steps INSIDE a container on a 4-GPU node
# (enroot start --root --rw, then enroot export) and could therefore assert things a podman build
# cannot. Specifically, these gates from the chain have NO equivalent above:
#
#   * torch.cuda.device_count() == 4                       (build-pytorch211-phase1-inner.sh)
#   * a real torch.distributed allreduce over the 4 GPUs   (torch-dist-allreduce-mi300.py)
#   * the host-loader gate and one-rank RCCL init over CXI (run-phase1-one-rank.sh, job J1)
#   * rocm_aiter_ops.is_fused_moe_enabled()                (add-aiter-pt211.sbatch; it needs a
#     live device to resolve, so only is_aiter_found() is asserted here)
#   * that the MLA prefill selector returns FlashAttnPrefillBackend
#   * that the aiter modules prebuilt above actually LAUNCH
#
# So this Dockerfile proves strictly less at build time than the chain did. Until it has been built
# AND smoke-tested end to end on a 4-GPU mi300 node, inference/build/build-chain.sh remains the
# authority and this file is a candidate, not a replacement.
#
# THE 0.23.0 PIN, MEASURED. The recorded premise -- "AMD's triton_kernels has no matmul_ogs, only
# upstream PyPI's does" -- is the wrong diagnosis. Measured in the deployed 0.27.1 image:
#
#   * site-packages carries triton_kernels 1.0.0+amd.rocm7.2.0.git89002410, which has matmul.py and
#     matmul_details/ where upstream had matmul_ogs.py. It is a RENAME in a newer snapshot, not an
#     AMD-versus-upstream feature gap.
#   * vLLM 0.27.1 ALREADY SHIPS a working copy at vllm/third_party/triton_kernels, matmul_ogs and
#     all, plus a fallback in vllm/utils/import_utils.py:import_triton_kernels that aliases it into
#     sys.modules["triton_kernels"].
#   * That fallback never fires, because it prefers site-packages -- and AMD's package being
#     present is precisely what defeats it.
#   * With AMD's package hidden and the vendored one aliased, the mxfp4.py:1137 import line
#     succeeds: FlexCtx, PrecisionConfig and InFlexData all resolve.
#
# So the pin is retirable by removing AMD's triton_kernels from the 0.27.1 venv, and the target is
# four images rather than five. Nothing in that venv imports the top-level package (aiter vendors
# its own under aiter/ops/triton/_triton_kernels), so the removal has no other consumer. It is NOT
# done here because an import is not a serve: it needs a served oss120b and an accuracy gate on
# gfx942 before any campaign result rests on it.
