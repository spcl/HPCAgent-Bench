# syntax=docker/dockerfile:1
#
# vLLM 0.27.1 inference on AMD MI300A (gfx942), serving oss120b. The SIBLING of ce-images/vllm/,
# which stays on 0.23.0 as the known-good fallback -- 33 env files select it today and its value is
# that it is unchanged. This one is the candidate.
#
# Build context is the REPOSITORY ROOT:
#   podman build -f containers/cluster/ce-images/vllm-0271/Dockerfile -t optarena-vllm-0271 .
# build.sh does that with the right podman flags; build.sbatch runs build.sh on a compute node.
#
# WHY THIS IMAGE CAN EXIST AT ALL
#
# 0.27.1 was written off because it routes gpt-oss through its mxfp4 path in
# process_weights_after_loading regardless of --dtype bfloat16, and that path imports
# triton_kernels.matmul_ogs (fused_moe/oracle/mxfp4.py:1137). Jobs 601854-601857 died there in
# ~7 minutes. The recorded diagnosis -- "AMD's triton_kernels has no matmul_ogs, only upstream
# PyPI's does" -- is wrong, and measuring it in the deployed image gives the real one:
#
#   * site-packages carries triton_kernels 1.0.0+amd.rocm7.2.0.git89002410, which has matmul.py and
#     matmul_details/ where upstream had matmul_ogs.py. A RENAME in a newer snapshot, not a missing
#     feature. AMD tracks post-rename upstream, so no AMD build will ever have matmul_ogs.
#   * vLLM 0.27.1 ALREADY SHIPS a working pre-rename copy at vllm/third_party/triton_kernels, and
#     vllm/utils/import_utils.py:import_triton_kernels aliases it into sys.modules["triton_kernels"]
#     when site-packages has none.
#   * That fallback never fires, because it PREFERS site-packages. AMD's package being present is
#     precisely what defeats it.
#
# So the bug was never a missing feature. It was TWO copies with a preference order, and the
# preference picked the wrong one. The fix is to delete the copy that loses, and the rule this
# image applies throughout is ONE VERSION OF ONE THING, NO FALLBACK -- asserted, not assumed.
#
# Verified in the deployed 0.27.1 image before this file was written: after
# `pip uninstall -y triton_kernels`, has_triton_kernels() is True, triton_kernels resolves to
# /opt/vllm-src/vllm/third_party/triton_kernels/__init__.py, the literal mxfp4.py:1137 imports
# succeed, the two startup "Failed to import Triton kernels" ERROR lines disappear, and AMD's ROCm
# triton 3.7.0+amd.rocm7.2.0.git89002410 survives untouched.
#
# THAT IS AN IMPORT, NOT A SERVE. See LIMITATION at the bottom before promoting this.
ARG BASE_IMAGE=docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1\
@sha256:a3b65813621095e3389269417e963725b59310184588c9d2490d44e6e83fa01c

# ------------------------------------------------------------------------------------------------
# Stage 1: the AWS OFI plugin (RCCL-OFI), compiled against libfabric built from source at the
# version the Beverin host runs. Identical to ce-images/vllm: same host contract (glibc 2.38,
# libfabric 2.3.1 -> libfabric.so.1.29.1, symbol ceiling FABRIC_1.8, libcxi.so.1.5.0), same reason
# no libfabric or libcxi may ship in the final image -- the runtime pair comes from the CE host
# hook, and this base is glibc 2.39, so that graft works here.
# ------------------------------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS ofi-builder
ARG BASE_IMAGE
ARG LIBFABRIC_REF=v2.3.1
ARG AWS_OFI_NCCL_REF=v1.20.0
USER root

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
      '  [ "${n}" -ge 6 ] && exit 1' \
      '  sleep $((n * 30))' \
      '  n=$((n + 1))' \
      'done' \
      > /usr/local/bin/gitretry; \
    chmod 0755 /usr/local/bin/gitretry

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      autoconf automake binutils build-essential ca-certificates git \
      libhwloc-dev libnuma-dev libtool make patchelf pkg-config \
 && rm -rf /var/lib/apt/lists/*

# Compile-only libfabric: the plugin needs the headers and a .so to link against, never a provider.
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
# Stage 2: PyTorch 2.11 + vLLM 0.27.1 + flash-attn + the full aiter stack, in /opt/pytorch211.
# ------------------------------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS final
USER root

ARG VLLM_VERSION=0.27.1
ARG TORCH_VERSION=2.11.0+rocm7.2
ARG TORCHVISION_VERSION=0.26.0+rocm7.2
ARG TORCHAUDIO_VERSION=2.11.0+rocm7.2
ARG FLASH_ATTN_VERSION=2.8.3
# The aiter that is IN the deployed 0.27.1 image, read back from it: amd_aiter 0.1.20 at git
# fc2e5d57fb5b8ad8e7e23f7103071dde798ea618. Pinned to the TAG rather than discovered from vLLM's
# docker/ files, because 0.27.1 spells that pin `main` -- a moving branch, which is how 600659
# built whatever main happened to be that day. A tag states which ROCm it targets; `main` states
# nothing, and "one version of one thing" is not satisfied by a version that changes daily.
ARG AITER_REF=v0.1.20
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

# aiter's MASTER SWITCH stays OFF, and this does not contradict shipping the full stack below.
# 0.27.1 NEEDS aiter INSTALLED or it dies before the API binds: that generation's Kimi ViT
# patch-embed does a bare `from aiter.ops.triton...` inside the multimodal dummy profile_run
# (job 600649, on a text-only campaign). Installed and dispatched-to are different questions.
# Turning the switch on is what broke 610251/610252 (RPC timeout on sample_tokens at
# step_counter=0, not one token decoded) and 600662 (MLA prefill on gfx942, fmha_v3_varlen_fwd
# invalid arg); every oss120b arm that has ever finished ran with it off. Note this is a
# vLLM/gfx942 fact and does NOT transfer to SGLang, where SGLANG_USE_AITER=1 is on every live arm.
ENV VLLM_ROCM_USE_AITER=0

COPY --from=ofi-builder /opt/aws-ofi-nccl /opt/aws-ofi-nccl

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
      '  [ "${n}" -ge 6 ] && exit 1' \
      '  sleep $((n * 30))' \
      '  n=$((n + 1))' \
      'done' \
      > /usr/local/bin/gitretry; \
    chmod 0755 /usr/local/bin/gitretry

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
# Another two-copies-with-a-preference bug: the pip headers win over torch's bundled ones on the
# include path, so an unpinned build compiles aiter against v12 while torch stays v11.
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

# ================================================================================================
# THE FIX. One version of one thing, no fallback.
#
# requirements/rocm.txt pulls AMD's triton_kernels, which is post-rename and has no matmul_ogs.
# vLLM's own vendored copy at vllm/third_party/triton_kernels is pre-rename and does. Two copies,
# and import_utils.import_triton_kernels prefers the wrong one. Removing the loser is the whole
# repair -- measured on the deployed image, where it turns the two startup
# "Failed to import Triton kernels" ERROR lines into a clean resolve.
#
# Nothing else in this venv imports the top-level package: aiter vendors its own under
# aiter/ops/triton/_triton_kernels, and conch_triton_kernels is a different distribution with a
# different top-level name. Checked before removing, and checked again in the gate below.
# ================================================================================================
RUN set -eux; \
    /opt/pytorch211/bin/python -m pip uninstall -y triton_kernels; \
    ! /opt/pytorch211/bin/python -m pip show triton_kernels 2>/dev/null | grep -q '^Name:'

# Triton path, not composable-kernel: SKIP_CK_BUILD turns a multi-hour gfx942 CK compile into a
# minutes-long pure-Triton install. --no-build-isolation because flash-attn's setup.py imports
# torch and must see the image's ROCm build, not a fresh wheel.
RUN set -eux; \
    FLASH_ATTENTION_SKIP_CK_BUILD=TRUE \
      /opt/pytorch211/bin/python -m pip install --no-cache-dir --no-build-isolation \
        "flash-attn==${FLASH_ATTN_VERSION}"

# The full aiter stack for gfx942, with its dependencies resolved rather than a --no-deps minimum:
# 0.27.1 imports aiter during profile_run, and a partial install fails there rather than at build.
# GPU_ARCHS=gfx942 is in the ENV above so the hipcc target is fixed for every layer, not just this
# one. --no-build-isolation because aiter's setup imports torch and must see the image torch.
#
# NOT PREBUILD_KERNELS. That is aiter's ahead-of-time path and compiles every gfx942 kernel, a
# multi-hour build; the JIT prebuild below removes the wedge it exists to remove, at minutes.
RUN set -eux; \
    gitretry clone https://github.com/ROCm/aiter.git /opt/aiter-src; \
    cd /opt/aiter-src; \
    git checkout "${AITER_REF}"; \
    git submodule sync; \
    gitretry submodule update --init --recursive; \
    git describe --tags --exact-match; \
    git rev-parse HEAD > /tmp/aiter.commit; \
    /opt/pytorch211/bin/python -m uv pip install --python /opt/pytorch211/bin/python --no-cache \
      ninja cmake 'pybind11==3.0.*' pandas; \
    /opt/pytorch211/bin/python -m uv pip install --python /opt/pytorch211/bin/python \
      --no-build-isolation --no-cache .

# The ASM v3 varlen path cannot express our mask. cpp_itfs/mha_fwd.cu maps causal to mask 2 ONLY
# when seqlen_q == seqlen_k and returns -1 otherwise, which surfaces from asm_mha_varlen_fwd.cu:494
# as "invalid argument for fmha_v3_varlen_fwd". Chunked prefill against a warm prefix cache ALWAYS
# has q shorter than k (88-95% hit rate on our arms), so every MLA prefill hits it, and
# can_impl_fmha_v3_fwd claims the shape anyway: its one 192x128 guard is skipped when block_table
# is set, and vLLM always pages. Adding the missing condition makes the predicate fall through to
# aiter's CK-tile mha_varlen_fwd -- still aiter -- instead of crashing.
#
# The anchor is asserted, not searched for leniently: aiter moved from v0.1.13.post1 to v0.1.20
# here, and a patch that silently finds nothing is worse than a build that stops.
RUN /opt/pytorch211/bin/python <<'PATCH'
import pathlib
p = pathlib.Path("/opt/pytorch211/lib/python3.12/site-packages/aiter/ops/mha.py")
s = p.read_text()
tail = "        return ret\n"
old = ("        if hdim_q == 192 and hdim_v == 128 and block_table is None:\n"
       "            ret = ret and not _fmha_kv_byte_extent_ge_u32(max_seqlen_k, k, v)\n" + tail)
assert s.count(old) == 1, "anchor count " + str(s.count(old)) + " -- aiter moved, re-derive the patch"
new = old[: -len(tail)] + "        ret = ret and ((not causal) or (max_seqlen_q == max_seqlen_k))\n" + tail
p.write_text(s.replace(old, new, 1))
print("patched can_impl_fmha_v3_fwd varlen: mask/seqlen guard added")
PATCH

# aiter's JIT, built HERE instead of on the first request.
#
# aiter ships no prebuilt .so. Every process JIT-builds module_aiter_core (21-58 s measured) and
# then each op module lazily on FIRST USE -- behind a baton lock in AITER_JIT_DIR that outlives the
# engine's RPC deadline. That is 598021 (no /v1/models for 5400 s), 610164 (300 s watchdog) and
# 610251/610252 (TimeoutError on sample_tokens at step_counter=0). prebuild-aiter-jit.sbatch worked
# around it as a separate GPU job; the wedge is a property of the image, so it is fixed in the image.
#
# No GPU is needed: measured building on a device-less node in 26 s. That is also the limit of what
# it proves -- the modules built, not that they launch.
#
# AITER_JIT_DIR points INSIDE the image so a hit is a read off the squashfs. A miss still writes,
# into the per-job CE overlay, so an op this prebuild did not cover degrades to today's cold-cache
# behaviour rather than failing. run_cluster.sh reads it as ${AITER_JIT_DIR:-...}, so this value
# wins over its scratch default without a launcher change.
ENV AITER_JIT_DIR=/opt/aiter-jit
RUN mkdir -p /opt/aiter-jit /tmp/aiter-home
RUN HOME=/tmp/aiter-home /opt/pytorch211/bin/python <<'PY'
import aiter

print("aiter core built at", aiter.__file__, flush=True)
# Every op family vLLM can dispatch to, so the cache covers whatever the server picks. Each import
# is guarded because coverage varies by aiter build and a family this release does not carry must
# not abort the image build -- the count gate below is what makes an empty prebuild fatal.
for name in ("aiter.ops.rmsnorm", "aiter.ops.norm", "aiter.ops.gemm_op_a8w8", "aiter.ops.gemm_op_a4w4",
             "aiter.ops.moe_op", "aiter.ops.moe_sorting", "aiter.ops.mha", "aiter.ops.attention",
             "aiter.ops.rope", "aiter.ops.quant", "aiter.ops.shuffle", "aiter.ops.activation",
             "aiter.ops.pos_encoding", "aiter.fused_moe"):
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
# which is this image's model.
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

# Record the resolved pins. The tag carries no version by design, so this file plus the image
# digest is what a later result gets attributed to.
RUN set -eux; \
    { \
      echo "vllm=${VLLM_VERSION}"; \
      echo "aiter_ref=${AITER_REF}"; \
      printf 'aiter_commit='; cat /tmp/aiter.commit; \
      /opt/pytorch211/bin/python -m pip freeze \
        | grep -iE '^(torch|torchvision|torchaudio|triton|numpy|flash-attn|amd-aiter|pybind11)=='; \
    } > /opt/BUILD-PINS.txt; \
    cat /opt/BUILD-PINS.txt

# ================================================================================================
# The GPU-free gate. Everything here is import-, metadata- and argument-level.
# ================================================================================================
RUN /opt/pytorch211/bin/python <<'PY'
import importlib
import importlib.metadata as md
import pathlib
import sys

for pkg, want in (("torch", "2.11.0+rocm7.2"), ("torchvision", "0.26.0+rocm7.2"),
                  ("torchaudio", "2.11.0+rocm7.2")):
    got = md.version(pkg)
    assert got == want, f"{pkg}: expected {want}, got {got}"
cuda = [d.metadata["Name"] for d in md.distributions()
        if (d.metadata.get("Name") or "").lower().startswith("nvidia-")]
assert not cuda, f"CUDA packages leaked in: {cuda}"

def dists(*names):
    want = {n.replace('-', '_').lower() for n in names}
    return [d for d in md.distributions()
            if (d.metadata.get('Name') or '').replace('-', '_').lower() in want]


import torch
import vllm

assert torch.version.hip is not None, "not a ROCm torch"
assert torch.__file__.startswith("/opt/pytorch211/")
assert vllm.__file__.startswith("/opt/vllm-src/")
assert vllm.__version__ == "0.27.1", vllm.__version__
for mod in ("vllm._C", "vllm._rocm_C"):
    importlib.import_module(mod)

# --- ONE VERSION OF ONE THING: triton ---------------------------------------------------------
# Exactly one distribution named `triton`, and it must be the ROCm build. PyPI's `triton` is the
# NVIDIA distribution under the same name, so a resolver that reaches for it OVERWRITES this one
# and the failure is silent until a kernel compiles for the wrong target. The local version tag is
# the only honest discriminator -- both wheels expose an `amd` backend.
tritons = dists("triton", "pytorch-triton", "pytorch-triton-rocm")
assert len(tritons) == 1, f"expected exactly one triton, got {[d.metadata['Name'] for d in tritons]}"
triton_version = md.version("triton")
assert "+amd.rocm" in triton_version, f"not the ROCm triton: {triton_version}"
import triton
import triton.backends
assert "amd" in triton.backends.backends, sorted(triton.backends.backends)
assert triton.__file__.startswith("/opt/pytorch211/"), triton.__file__

# --- ONE VERSION OF ONE THING: triton_kernels -------------------------------------------------
# The whole reason this image exists. There must be no site-packages copy, and the name must
# resolve through vLLM's own shim to the vendored pre-rename copy.
assert not dists("triton_kernels"), \
    "AMD's triton_kernels is still installed and will shadow the vendored copy"
from vllm.utils.import_utils import has_triton_kernels, import_triton_kernels
assert has_triton_kernels(), "vLLM cannot find any triton_kernels at all"
import_triton_kernels()
import triton_kernels
assert triton_kernels.__file__.startswith("/opt/vllm-src/vllm/third_party/triton_kernels/"), \
    triton_kernels.__file__
# The literal fused_moe/oracle/mxfp4.py:1137 imports, the ones 601854-601857 died on.
from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig  # noqa: F401
from triton_kernels.numerics import InFlexData  # noqa: F401

# --- ONE VERSION OF ONE THING: aiter -----------------------------------------------------------
# One distribution, and one IMPORTABLE copy. /opt/aiter-src stays on disk because aiter's JIT
# compiles from sources the install carries and deleting the tree is an unverified risk, so the
# check is that it is not on the path rather than that it is gone.
aiters = dists("aiter", "amd-aiter")
assert len(aiters) == 1, f"expected exactly one aiter, got {[d.metadata['Name'] for d in aiters]}"
import aiter
assert aiter.__file__.startswith("/opt/pytorch211/lib/python3.12/site-packages/aiter/"), aiter.__file__
assert not any(p.rstrip("/") == "/opt/aiter-src" for p in sys.path), "the aiter source tree is importable"
import vllm.envs as envs
assert not envs.VLLM_ROCM_USE_AITER, "the aiter master switch is on in the image"
from vllm._aiter_ops import is_aiter_found
assert is_aiter_found(), "vLLM does not find the aiter package"

# --- flash-attn --------------------------------------------------------------------------------
from flash_attn import flash_attn_interface
assert flash_attn_interface.USE_TRITON_ROCM, "imported the CK path, not triton_amd"
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
assert is_flash_attn_varlen_func_available(), "vLLM still reports flash-attn unavailable"

# --- the baked-in inputs -----------------------------------------------------------------------
import json

configs = sorted(pathlib.Path("/opt/moe-configs").glob("E=*.json"))
assert len(configs) >= 2, f"tuned MoE configs missing: {configs}"
for c in configs:
    json.loads(c.read_text())
patch = pathlib.Path("/opt/vllm-eager-pg/sitecustomize.py")
assert "patch_pp_collectives" in patch.read_text(), "eager-pg patch did not land"
assert not pathlib.Path("/opt/pytorch211/lib/python3.12/site-packages/sitecustomize.py").exists(), \
    "the eager-pg patch is unconditionally active"

# --- the live oss120b serving arguments --------------------------------------------------------
# From IMAGE_REQUIREMENTS.md, "The serving configs the new images must stay compatible with", read
# out of the v9 env files. Parsing them is not serving, but it is the one part of that contract a
# GPU-free build CAN check, and it catches the real upgrade hazard: a flag 0.23 had and 0.27.1
# renamed or dropped. `vllm serve --help` is a SUMMARY and would miss it -- --enable-auto-tool-choice
# is absent from even --help=all on 0.27.1 yet parses fine -- so ask the parser, not the help text.
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser

live = ("--dtype bfloat16 --load-format safetensors --safetensors-load-strategy prefetch "
        "--generation-config auto --enable-auto-tool-choice --tool-call-parser openai "
        "--reasoning-parser openai_gptoss --max-model-len 131072 "
        "--gpu-memory-utilization 0.70 --max-num-seqs 128").split()
ns = make_arg_parser(FlexibleArgumentParser()).parse_args(["--model", "dummy"] + live)
assert ns.enable_auto_tool_choice is True
# `vllm` DISCARDS the model's own generation_config.json and silently changed sampling in twenty
# env files once already. If a future vLLM changes this default, the launcher must be told, not the
# other way round.
assert ns.generation_config == "auto", ns.generation_config

print("vllm", vllm.__version__, "torch", torch.__version__, "triton", triton_version, file=sys.stderr)
PY

WORKDIR /workspace

# LIMITATION -- read before promoting this image.
#
# THIS IS UNVALIDATED UNTIL IT SERVES. Everything the gate above checks is an import or an argument
# parse. The precedent is exactly this shape and it is why the rule exists: the aiter master switch
# imported cleanly, passed every import gate, and died under load with "invalid argument for
# fmha_v3_varlen_fwd". Build to a -candidate name, serve oss120b under the VLLM_EXTRA_ARGS in
# IMAGE_REQUIREMENTS.md, and run an accuracy gate on gfx942 before any campaign result rests on it.
# ce-images/vllm/ (0.23.0) remains the fallback and must not be retired until that passes.
#
# A podman build also has no GPUs, so these gates from the six-job chain have no equivalent here:
# torch.cuda.device_count() == 4; a real 4-GPU allreduce; the host-loader gate and one-rank RCCL
# init over CXI; rocm_aiter_ops.is_fused_moe_enabled(), which needs a live device to resolve; that
# the MLA prefill selector returns FlashAttnPrefillBackend; and that the aiter modules prebuilt
# above actually LAUNCH.
