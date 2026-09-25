# syntax=docker/dockerfile:1
#
# vLLM inference on AMD MI300A (gfx942). Serves oss120b only; other models go to the SGLang image.
#
# Build context is the REPOSITORY ROOT, because this file bakes in two things that live in the
# repo today and are reached from outside the image at run time:
#   podman build -f containers/cluster/ce-images/vllm/Dockerfile -t hpcagent-bench-vllm .
# build.sh does that with the right podman flags; build.sbatch runs build.sh on a compute node.
#
# Read LIMITATION at the bottom before treating a podman build as equivalent to the Slurm chain
# it replaces: that chain validated on real GPUs at build time and a podman build cannot.
#
# vLLM is PINNED to 0.23.0: 0.27.1 routes gpt-oss through a mxfp4 path that imports
# triton_kernels.matmul_ogs, which AMD's triton_kernels package does not have. A one-line repair
# is known (see LIMITATION) but unproven against a served, accuracy-gated model, so the pin stands.
ARG BASE_IMAGE=docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1\
@sha256:a3b65813621095e3389269417e963725b59310184588c9d2490d44e6e83fa01c

# ------------------------------------------------------------------------------------------------
# ONE stage. This image builds NO fabric software -- no libfabric, no libcxi, no RCCL net plugin.
# CSCS ships the whole stack as one internally consistent artifact (its own libc, libcurl, libcxi,
# libfabric and plugin), installed by the enroot hooks when the EDF sets com.hooks.netstack /
# .cxi / .aws_ofi_nccl. A shipped copy would SHADOW that artifact, which is the failure the gate
# below refuses. vLLM reaches the fabric through RCCL, not MPI, so there is not even a
# compile-time link target left to produce.
# ------------------------------------------------------------------------------------------------
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
# Pinned to the exact tag, not `main`: 0.27.1's docker/ files pin aiter to a moving branch.
ARG AITER_REF=v0.1.13.post1
ARG TORCH_INDEX=https://download.pytorch.org/whl/rocm7.2

# base.name records what FROM resolved to, the whole reproducibility claim given the digest pin
# above. BASE_IMAGE is re-declared because an ARG from before the first FROM is out of scope
# inside a stage.
ARG BASE_IMAGE
# The registry reference, not whatever FROM resolved to: a cache hit rewrites BASE_IMAGE to a
# local dir: path, and that must not become the published base.name.
ARG BASE_IMAGE_REF="${BASE_IMAGE}"
LABEL org.opencontainers.image.title="hpcagent-bench-vllm" \
      org.opencontainers.image.description="vLLM ${VLLM_VERSION} inference for MI300A: ROCm 7.2, CSCS netstack fabric, aiter ${AITER_REF}" \
      org.opencontainers.image.version="${VLLM_VERSION}" \
      org.opencontainers.image.source="https://github.com/spcl/HPCAgent-Bench" \
      org.opencontainers.image.licenses="GPL-3.0-or-later" \
      org.opencontainers.image.base.name="${BASE_IMAGE_REF}"

# build.sh passes ROCM_ARCH from ../gpu_arch.env for the build job's partition; there is no default.
# /opt/gpu-arch records it as a FILE, because the CE does not reliably keep image ENV.
ARG ROCM_ARCH
RUN set -eu; \
    test -n "${ROCM_ARCH:-}" || { echo "ROCM_ARCH is unset: build with build.sh, which reads gpu_arch.env" >&2; exit 2; }; \
    printf '%s\n' "${ROCM_ARCH}" > /opt/gpu-arch
COPY containers/cluster/ce-images/device_arch_gate.sh /usr/local/bin/device_arch_gate.sh
ENV VIRTUAL_ENV=/opt/pytorch211 \
    ROCM_PATH=/opt/rocm \
    HCC_AMDGPU_TARGET=${ROCM_ARCH} \
    PYTORCH_ROCM_ARCH=${ROCM_ARCH} \
    GPU_ARCHS=${ROCM_ARCH} \
    GPU_ARCH_LIST=${ROCM_ARCH} \
    VLLM_TARGET_DEVICE=rocm \
    HSA_ENABLE_IPC_MODE_LEGACY=1 \
    SAFETENSORS_FAST_GPU=1 \
    HIP_FORCE_DEV_KERNARG=1 \
    TOKENIZERS_PARALLELISM=false \
    PATH=/opt/pytorch211/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64

# Read at IMPORT, not only at build: flash_attn_interface.py picks the triton_amd path on TRUE
# and the compiled CK extension (which a triton-only install lacks) on FALSE; on FALSE vLLM
# merely reports the backend unavailable rather than failing loudly. Must ALSO be in the EDF,
# because ENV alone does not survive every launcher path.
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

# aiter's master switch, OFF, stated in the image rather than left to the launcher: every
# finished oss120b arm ran with it off, while turning it on has broken RPC timeouts and MLA
# prefill on gfx942. run_cluster.sh reads it as ${VLLM_ROCM_USE_AITER:-0}, so this value stands
# unless an env file overrides it -- the intended way to retry aiter.
#
# It is an ARG so a candidate image can DECLARE the opposite without forking this file: some
# MoE models need VLLM_ROCM_USE_AITER=1 with --linear-backend aiter --moe-backend aiter. Default
# stays 0, so hpcagent-bench-vllm.sqsh rebuilds byte-identical.
ARG VLLM_ROCM_AITER_SWITCH=0
ENV VLLM_ROCM_USE_AITER=${VLLM_ROCM_AITER_SWITCH}


# Every network git call goes through this retry wrapper, and where a mirror is mounted none of
# them reaches GitHub at all. The `insteadOf` rewrite covers callers the wrapper cannot: some
# tools clone in-process with no retry of their own. `protocol.file.allow` goes with it: git
# refuses local-path submodule clones since CVE-2022-39253, and the rewrite turns every submodule
# URL into exactly that. GIT_TERMINAL_PROMPT=0 makes an unauthenticated 401 fail instead of
# blocking on a prompt no build has a terminal for; the backoff is what actually fixes it.
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
      for m in /git-mirrors/*/*.git; do \
        r="${m#/git-mirrors/}"; r="${r%.git}"; \
        git config --global "url./git-mirrors/${r}.git.insteadOf" "https://github.com/${r}.git"; \
        git config --global --add "url./git-mirrors/${r}.git.insteadOf" "https://github.com/${r}"; \
      done; \
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
    /opt/pytorch211/bin/python -m pip install --no-cache-dir 'pillow>=10.0' 'numpy<3'; \
    /opt/pytorch211/bin/python -m pip show triton-rocm \
      | awk '/^Version:/{print $2}' > /opt/triton-rocm.version; \
    test -s /opt/triton-rocm.version

# pybind11 pinned below 3.1: 3.1.0 bumped its internals id v11 -> v12 and refuses to share
# registered types across that boundary, so a v12 build cannot accept tensors from the v11 torch
# extensions (ROCm/aiter#4770, vllm-project/vllm#52400). An ABI split, not a bad kernel.
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
# varlen path, and it does NOT belong here: its anchor (can_impl_fmha_v3_fwd, aiter v0.1.20) does
# not exist in this image's v0.1.13.post1. This image also ships VLLM_ROCM_USE_AITER=0, so the
# path is off by default. If a future pin here needs the guard, write it against THIS aiter.

# aiter's JIT, built HERE instead of on the first request.
#
# aiter ships no prebuilt .so. Every process JIT-builds module_aiter_core on import and then each
# op module lazily on FIRST USE, behind a baton lock in AITER_JIT_DIR that can outlive the
# engine's RPC deadline. The wedge is a property of the image, so it is fixed in the image.
#
# A GPU IS needed: `import aiter` shells out to rocminfo at module load and ignores GPU_ARCHS by
# design, so build.sh must hand podman /dev/kfd. The op calls below DO issue launches, which is
# what builds the op modules at all -- /dev/dri and the seccomp/keep-groups flags in
# build_common.sh are what make that work, and the gate below fails the build if they ever stop.
#
# AITER_JIT_DIR points INSIDE the image so a hit is a read off the squashfs. A miss still writes
# into the per-job CE overlay, so an op this prebuild did not cover degrades to a cold-cache
# build rather than failing.
# ONE VERSION OF ONE THING: triton, restored after the resolver replaces it.
#
# xgrammar declares an unpinned `triton`. On PyPI that name is the NVIDIA distribution, so pip
# installs it OVER the triton-rocm that torch brought -- same top-level package, two owners --
# and leaves pip's `~riton` rename behind. torch 2.11.0+rocm7.2 is built against the ROCm triton,
# so the intruder is removed and the ROCm build reinstalled to restore every file it owns.
#
# The version comes from /opt/triton-rocm.version, recorded right after the torch install: reading
# it back from pip HERE does not work, because installing PyPI triton leaves triton-rocm's
# metadata gone rather than merely shadowed.
RUN set -eux; \
    site="$(/opt/pytorch211/bin/python -c 'import site; print(site.getsitepackages()[0])')"; \
    ver="$(cat /opt/triton-rocm.version)"; \
    test -n "${ver}"; \
    /opt/pytorch211/bin/python -m pip uninstall -y triton || true; \
    rm -rf "${site}"/~riton*; \
    /opt/pytorch211/bin/python -m pip install --no-cache-dir --no-deps --force-reinstall \
      --index-url "${TORCH_INDEX}" "triton-rocm==${ver}"; \
    ! /opt/pytorch211/bin/python -m pip show triton 2>/dev/null | grep -qx 'Name: triton'; \
    /opt/pytorch211/bin/python -c "import triton, triton.backends; \
      assert 'amd' in triton.backends.backends, sorted(triton.backends.backends)"

ENV AITER_JIT_DIR=/opt/aiter-jit \
    AITER_ROOT_DIR=/opt/aiter-root
RUN mkdir -p /opt/aiter-jit /opt/aiter-root /tmp/aiter-home
RUN HOME=/tmp/aiter-home /opt/pytorch211/bin/python <<'PY'
import aiter

print("aiter core built at", aiter.__file__, flush=True)
# The op families vLLM can dispatch to. Guarded: coverage varies by aiter build, and a family
# this release lacks must not abort the build.
for name in ("aiter.ops.rmsnorm", "aiter.ops.norm", "aiter.ops.gemm_op_a8w8", "aiter.ops.moe_op",
             "aiter.ops.mha", "aiter.ops.attention", "aiter.ops.rope", "aiter.ops.quant",
             "aiter.ops.shuffle", "aiter.fused_moe"):
    try:
        __import__(name)
        print("imported (a build still needs a call):", name, flush=True)
    except Exception as exc:
        print(f"skip {name}: {type(exc).__name__}: {exc}", flush=True)

# Importing an op family does NOT build its kernel module: @compile_ops is lazy and builds on the
# first CALL, so the loop above only ever produces module_aiter_core.so. The calls below build the
# rest here instead of at serve time, and the artefact hashes must match what the server asks for
# so a serve is a cache hit rather than a near-miss.
import os
import pathlib

import torch

_dev = torch.device("cuda")
_jit = pathlib.Path(os.environ["AITER_JIT_DIR"])


def _prebuild(label, fn):
    # Report the module each call produced: aiter builds differ across images and the .so
    # names do not match across them, so a gate must not assume one by name.
    before = {p.name for p in _jit.rglob("*.so")}
    try:
        fn()
    except Exception as exc:
        print(f"skip {label}: {type(exc).__name__}: {exc}", flush=True)
        return
    print("prebuilt:", label, "->", sorted({p.name for p in _jit.rglob("*.so")} - before), flush=True)


def _rms_norm():
    # 2-D, bf16, hidden <= 8192 is the case that dispatches to the hand-tuned HIP
    # module_rmsnorm_quant; anything else takes the generic opus path and builds nothing.
    x = torch.randn(4, 7168, dtype=torch.bfloat16, device=_dev)
    w = torch.ones(7168, dtype=torch.bfloat16, device=_dev)
    aiter.rms_norm(x, w, 1e-6)


def _probs():
    return torch.softmax(torch.randn(4, 1024, dtype=torch.float32, device=_dev), dim=-1).contiguous()


def _top_k_top_p():
    import aiter.ops.sampling  # noqa: F401
    k = torch.full((4,), 50, dtype=torch.int32, device=_dev)
    p = torch.full((4,), 0.9, dtype=torch.float32, device=_dev)
    torch.ops.aiter.top_k_top_p_sampling_from_probs(_probs(), None, k, 0, p, 0.0, deterministic=True)


def _top_p():
    p = torch.full((4,), 0.9, dtype=torch.float32, device=_dev)
    torch.ops.aiter.top_p_sampling_from_probs(_probs(), None, p, 0.0, deterministic=True)


def _top_k_renorm():
    k = torch.full((4,), 50, dtype=torch.int32, device=_dev)
    torch.ops.aiter.top_k_renorm_probs(_probs(), k, 0)


# Signatures and deterministic=True come from the engine call site, so what is prebuilt is what
# gets asked for. Each is guarded: an op a release does not carry must degrade to a runtime build.
_prebuild("rms_norm", _rms_norm)
_prebuild("top_k_top_p_sampling_from_probs", _top_k_top_p)
_prebuild("top_p_sampling_from_probs", _top_p)
_prebuild("top_k_renorm_probs", _top_k_renorm)
# The serving path calls far more than sampling. These eight were read off what live models
# actually JIT-build at serve time, so the image ships them instead of paying for them behind a
# baton lock inside the RPC deadline.
_bf16, _fp8 = torch.bfloat16, torch.float8_e4m3fnuz


def _layernorm2d():
    n = 7168
    x = torch.randn(4, n, dtype=_bf16, device=_dev)
    aiter.layernorm2d_fwd_with_add(torch.empty_like(x), x, x.clone(), torch.empty_like(x),
                                   torch.ones(n, dtype=_bf16, device=_dev),
                                   torch.zeros(n, dtype=_bf16, device=_dev), 1e-6)


def _dynamic_quant():
    x = torch.randn(4, 7168, dtype=_bf16, device=_dev)
    aiter.dynamic_per_tensor_quant(torch.empty(x.shape, dtype=_fp8, device=_dev), x,
                                   torch.ones(1, dtype=torch.float32, device=_dev))


def _silu_and_mul():
    x = torch.randn(4, 2 * 7168, dtype=_bf16, device=_dev)
    aiter.silu_and_mul(torch.empty(4, 7168, dtype=_bf16, device=_dev), x)


def _reshape_and_cache():
    k = torch.randn(4, 8, 128, dtype=_bf16, device=_dev)
    kc = torch.zeros(64, 8, 128, 16, dtype=_bf16, device=_dev)
    aiter.reshape_and_cache(k, k.clone(), kc, kc.clone(),
                            torch.arange(4, dtype=torch.int64, device=_dev), "auto")


def _moe_sum():
    # topk MUST be one of 2/4/5. Any other value takes a python-side torch.sum branch that never
    # reaches the kernel: it would build nothing while _prebuild reported success.
    x = torch.randn(8, 4, 7168, dtype=_bf16, device=_dev)
    aiter.moe_sum(x, torch.empty(8, 7168, dtype=_bf16, device=_dev))


def _greedy_sample():
    logits = torch.randn(4, 1024, dtype=torch.float32, device=_dev)
    aiter.greedy_sample(torch.empty(4, dtype=torch.int64, device=_dev), logits)


_prebuild("layernorm2d_fwd_with_add", _layernorm2d)
_prebuild("dynamic_per_tensor_quant", _dynamic_quant)
_prebuild("silu_and_mul", _silu_and_mul)
_prebuild("reshape_and_cache", _reshape_and_cache)
_prebuild("moe_sum", _moe_sum)
_prebuild("greedy_sample", _greedy_sample)
_prebuild("get_padded_m", lambda: aiter.get_padded_m(128, 7168, 7168, 1))
_prebuild("moe_sorting_opus_get_workspace_size",
          lambda: aiter.moe_sorting_opus_get_workspace_size(128, 256, 8))

# The defect this step exists to catch, stated as the build's own assertion: an import-only
# prebuild left module_aiter_core.so and nothing else, and the shell gate below cannot tell that
# apart from a healthy image without naming a module.
_ops = {p.name for p in _jit.rglob("*.so")} - {"module_aiter_core.so"}
if not _ops:
    raise SystemExit("aiter prebuild produced no op module beyond module_aiter_core")
print("aiter op modules prebuilt:", sorted(_ops), flush=True)

PY
# The prebuild step above already fails the build if it produced no op module. What is left for
# the shell is the AITER_ROOT_DIR pin: aiter's template ops default to $HOME/.aiter, so if the pin
# does not hold, HOME grows a .aiter and the artefacts are outside the image. aiter takes its arch
# from rocminfo, not GPU_ARCHS, so the prebuilt modules are gated on ROCM_ARCH as well.
RUN set -eux; \
    find /opt/aiter-jit -name '*.so' | wc -l | xargs printf 'aiter prebuilt modules: %s\n'; \
    find /opt/aiter-root -mindepth 1 -maxdepth 2 | head -20; \
    test ! -e /tmp/aiter-home/.aiter; \
    rm -rf /tmp/aiter-home; \
    /usr/local/bin/device_arch_gate.sh --exact "${ROCM_ARCH}" /opt/aiter-jit

# Tuned fused_moe Triton configs, keyed by (experts, N, device, dtype). Baked rather than dropped:
# without them a MoE model serves on vLLM's default config at a large throughput penalty, and a
# config that exists nowhere is a config that is gone.
#
# run_cluster.sh sets VLLM_TUNED_CONFIG_FOLDER only when the repo folder exists and only when the
# variable is unset, so this ENV is what a launcher with that folder deleted will use.
COPY containers/cluster/ce-images/inference/moe-configs/ /opt/moe-configs/
ENV VLLM_TUNED_CONFIG_FOLDER=/opt/moe-configs

# The eager-PG / PP-collective-split / MLA-lse-layout patch, moved inside the image.
#
# NOT in site-packages, deliberately. A bare sitecustomize.py there runs at interpreter startup
# for every process in this venv, pip included, and its patches default ON, so installing it that
# way would silently apply them to arms that never asked. Its own directory keeps the opt-in that
# VLLM_EAGER_PG_PATCH=1 expresses, while putting the file itself under the image digest.
COPY containers/cluster/ce-images/inference/external-eager-pg-patch/sitecustomize.py \
     /opt/vllm-eager-pg/sitecustomize.py

# What can be checked WITHOUT a GPU. Everything here is import- and metadata-level; the real gates
# (device_count()==4, an allreduce, which MLA backend the selector returns) need hardware and live
# in the smoke test -- see LIMITATION.
RUN /opt/pytorch211/bin/python <<'PY'
import importlib
import importlib.metadata as md
import os
import sys

for pkg, want in (("torch", "2.11.0+rocm7.2"), ("torchvision", "0.26.0+rocm7.2"),
                  ("torchaudio", "2.11.0+rocm7.2")):
    got = md.version(pkg)
    assert got == want, f"{pkg}: expected {want}, got {got}"
print("amd_aiter", md.version("amd_aiter"), file=sys.stderr)
cuda = [d.metadata["Name"] for d in md.distributions()
        if (d.metadata.get("Name") or "").lower().startswith("nvidia-")]
assert not cuda, f"CUDA packages leaked in: {cuda}"


# --- ONE VERSION OF ONE THING: triton ----------------------------------------------------------
# xgrammar declares an unpinned `triton`, and pip can install PyPI's NVIDIA distribution over the
# triton-rocm that torch was built against without anything noticing. The repair RUN above
# removes the intruder; this is what proves the repair held.
#
# The discriminator is the DISTRIBUTION NAME, not a version tag -- triton-rocm's version is a plain
# 3.6.0 with no `+amd.rocm` local part -- and only the ROCm index publishes that name.
def dists(*names):
    want = {n.replace("-", "_").lower() for n in names}
    return [d for d in md.distributions()
            if (d.metadata.get("Name") or "").replace("-", "_").lower() in want]


tritons = dists("triton", "pytorch-triton", "pytorch-triton-rocm", "triton-rocm")
assert len(tritons) == 1, f"expected exactly one triton, got {[d.metadata['Name'] for d in tritons]}"
triton_dist = (tritons[0].metadata["Name"] or "").replace("-", "_").lower()
assert triton_dist == "triton_rocm", f"not the ROCm triton distribution: {triton_dist}"

import triton
import triton.backends

assert "amd" in triton.backends.backends, sorted(triton.backends.backends)
assert triton.__file__.startswith("/opt/pytorch211/"), triton.__file__

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
_want_aiter = os.environ.get("VLLM_ROCM_USE_AITER", "0") == "1"
assert bool(envs.VLLM_ROCM_USE_AITER) == _want_aiter, (
    f"aiter master switch is {envs.VLLM_ROCM_USE_AITER}, image declared {_want_aiter}")
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

# Drop the mirror rewrite from the SHIPPED image: left in /root/.gitconfig, every git clone a
# user runs inside this container would silently rewrite to a directory that is not there.
RUN set -eux; \
    git config --global --name-only --get-regexp '^url\..*\.insteadof$' 2>/dev/null \
      | sort -u | while read -r k; do git config --global --unset-all "${k}" || true; done; \
    git config --global --unset-all protocol.file.allow || true

# No fabric library ships. All three -- libfabric, libcxi and the RCCL net plugin -- arrive from
# the pinned CSCS netstack artifact via the enroot hooks, and that artifact carries its own libc
# and libcurl, so it is internally consistent in a way a host graft is not. A copy inside the
# image would be found first and shadow it.
#
# Ship no fabric library OF OUR OWN, but the gate below checks only paths this image controls: the
# hook bind-mounts (or, in host mode, binds from the node) its own libfabric/RCCL plugin at run
# time, and refusing any libfabric anywhere on the filesystem would forbid the hook its own mount
# point.
#
# What must NOT ship is a fabric library in a prefix WE control, because those are what RPATH can
# bind to, and RPATH is searched before everything else: a providerless libfabric there would
# shadow the hook's and neither it nor LD_LIBRARY_PATH could displace it.
RUN set -eux; \
    rm -f /opt/ofi-sdk/lib/libfabric.so*; \
    found="$(find /opt/ofi-sdk /opt/ofi /opt/aws-ofi-nccl \
                  \( -name 'libfabric.so*' -o -name 'libcxi.so*' -o -name 'librccl-net.so*' \) \
                  2>/dev/null || true)"; \
    [ -z "${found}" ] \
      || { echo "this image ships a fabric library in a prefix it controls:" >&2; \
           echo "${found}" >&2; \
           echo "RPATH would bind to it and shadow the CSCS netstack artifact" >&2; exit 1; }; \
    echo "no fabric library in /opt/ofi-sdk, /opt/ofi or /opt/aws-ofi-nccl -- the artifact supplies them"

WORKDIR /workspace

# LIMITATION -- read before trusting this file.
#
# The Slurm chain this replaces ran its install steps INSIDE a container on a 4-GPU node and could
# therefore assert things a podman build cannot: torch.cuda.device_count() == 4, a real
# torch.distributed allreduce, a one-rank RCCL init over CXI, rocm_aiter_ops.is_fused_moe_enabled()
# (only is_aiter_found() is asserted here), that the MLA prefill selector returns
# FlashAttnPrefillBackend, and that the aiter modules prebuilt above actually LAUNCH.
#
# So this Dockerfile proves strictly less at build time than the chain did. Until it has been
# built AND smoke-tested end to end on a 4-GPU mi300 node, the Slurm chain remains the authority
# and this file is a candidate, not a replacement.
#
# THE 0.23.0 PIN, MEASURED. The recorded premise -- "AMD's triton_kernels has no matmul_ogs, only
# upstream PyPI's does" -- is the wrong diagnosis: AMD's package is a newer snapshot that renamed
# matmul_ogs.py to matmul.py, and vLLM 0.27.1 already ships a working vendored copy with a
# fallback that aliases it into sys.modules -- but that fallback never fires while AMD's own
# package is present in site-packages and takes priority.
#
# So the pin is retirable by removing AMD's triton_kernels from the 0.27.1 venv; nothing else in
# that venv imports the top-level package. It is NOT done here because an import is not a serve:
# it needs a served oss120b and an accuracy gate on gfx942 before any campaign result rests on it.
