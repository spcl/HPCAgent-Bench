# syntax=docker/dockerfile:1
#
# vLLM inference on AMD Instinct, one image for every gpu_arch.env AMD target (MI250X, MI300, MI355X).
# Serves oss120b on mi300 and Qwen3.8 on mi200; qwen38, kimi and GLM-5.3 on mi300 are on SGLang. Build context is the repository root:
#
#   containers/images/vllm/build.sh
#
# THIN ON PURPOSE, like vllm-cuda/Dockerfile: the official vLLM ROCm release image already carries
# the engine, a ROCm torch, triton, aiter and every tool-call / reasoning parser the campaign names.
# This file pins that base by digest, bakes in the repo's tuned MoE configs, and GATES what the
# serving configs need. One vLLM release on every platform: the same v0.28.0 as vllm-cuda.
#
# No fabric software: the EDF's hooks inject libfabric/CXI and the RCCL net plugin (edf.toml.in);
# a copy in the image would shadow them, which the last gate refuses.
#
# PINNED BY DIGEST: vllm/vllm-openai-rocm:v0.28.0 (amd64), the tag's manifest digest, resolved from
# the registry without a pull.
ARG BASE_IMAGE=docker.io/vllm/vllm-openai-rocm:v0.28.0@sha256:e0a3b2bd3fe7ec563916c3a5d949898d133458c18d6b2f460c906885cfb32032
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
# The registry reference, not a base-image cache dir: path (build_common.sh ce_cache_base_image).
ARG BASE_IMAGE_REF="${BASE_IMAGE}"
ARG IMAGE_VERSION=dev
# The release the base digest is. Asserted below, so a BASE_IMAGE override to another release has
# to say so rather than silently serving on a different engine.
ARG VLLM_VERSION=0.28.0
LABEL org.opencontainers.image.title="hpcagent-bench-vllm" \
      org.opencontainers.image.description="HPCAgent-Bench vLLM ${VLLM_VERSION} inference for AMD Instinct (oss120b)" \
      org.opencontainers.image.source="https://github.com/spcl/HPCAgent-Bench" \
      org.opencontainers.image.licenses="GPL-3.0-or-later" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.base.name="${BASE_IMAGE_REF}"

# build.sh passes ROCM_ARCH from ../gpu_arch.env for the build job's partition; there is no default.
# /opt/gpu-arch records it as a FILE (gpu_arch_check.sh reads it at launch), because the CE does not
# reliably keep image ENV.
ARG ROCM_ARCH
# The same list comma-separated, for hipcc's HCC_AMDGPU_TARGET (it splices it into a shell line).
ARG ROCM_ARCH_CSV
RUN set -eu; \
    test -n "${ROCM_ARCH:-}" || { echo "ROCM_ARCH is unset: build with build.sh, which reads gpu_arch.env" >&2; exit 2; }; \
    test "${ROCM_ARCH_CSV}" = "$(echo "${ROCM_ARCH}" | tr ";" ",")" \
      || { echo "ROCM_ARCH_CSV '${ROCM_ARCH_CSV}' is not ROCM_ARCH '${ROCM_ARCH}' comma-separated" >&2; exit 2; }; \
    printf '%s\n' "${ROCM_ARCH}" > /opt/gpu-arch
# The image's arch list; an EDF narrows these to its partition's arch for run-time JIT builds.
ENV HCC_AMDGPU_TARGET=${ROCM_ARCH_CSV} \
    PYTORCH_ROCM_ARCH=${ROCM_ARCH} \
    GPU_ARCHS=${ROCM_ARCH} \
    GPU_ARCH_LIST=${ROCM_ARCH}
# vLLM's compiled ops are a fat binary: every listed arch has to be one of its device targets.
COPY containers/lib/device_arch_gate.sh /usr/local/bin/device_arch_gate.sh
RUN /usr/local/bin/device_arch_gate.sh --contains "${ROCM_ARCH}" "$(python3 -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')" >/dev/null

# Tuned fused_moe Triton configs, keyed by (experts, N, device, dtype). E=128,N=192 is oss120b's.
# run_cluster.sh prefers this baked folder over the repo copy.
COPY containers/inference/moe-configs/ /opt/moe-configs/
ENV VLLM_TUNED_CONFIG_FOLDER=/opt/moe-configs

# What can be checked WITHOUT a GPU. The device half (RCCL over CXI, a real serve) is a serving
# smoke; build.sbatch runs verify_image.py --profile vllm inside the image.
RUN python3 - <<'PY'
import importlib
import importlib.metadata as md
import json
import os
import pathlib
import sys

import torch
import vllm

want = os.environ["VLLM_VERSION"]
assert vllm.__version__.split("+")[0] == want, f"vllm {vllm.__version__}, image declares {want}"
assert torch.version.hip is not None, "not a ROCm torch"
assert sys.executable.startswith("/usr/bin/python3"), f"the EDF runs /usr/bin/python3, the base installs into {sys.executable}"
for mod in ("triton", "aiter", "huggingface_hub", "vllm._C", "vllm._rocm_C"):
    importlib.import_module(mod)
cuda = [d.metadata["Name"] for d in md.distributions() if (d.metadata.get("Name") or "").lower().startswith("nvidia-")]
assert not cuda, f"CUDA packages in a ROCm image: {cuda}"

# Every parser a serving config in this repo names, resolved the way `vllm serve` resolves it.
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager

ToolParserManager.get_tool_parser("openai")
ReasoningParserManager.get_reasoning_parser("openai_gptoss")

configs = sorted(pathlib.Path("/opt/moe-configs").glob("E=*.json"))
assert configs, "tuned MoE configs missing"
for c in configs:
    json.loads(c.read_text())

# The images ship no fabric library of their own (IMAGE_REQUIREMENTS.md): one in a prefix the image
# controls would shadow what the CE hooks inject.
names = ("libfabric.so*", "libcxi.so*", "librccl-net*.so*", "libnccl-net*.so*")
shipped = [str(p) for root in ("/opt", "/usr/local/lib") for n in names for p in pathlib.Path(root).rglob(n)]
assert not shipped, f"the image ships a fabric library: {shipped}"

pathlib.Path("/usr/local/share").mkdir(parents=True, exist_ok=True)
pathlib.Path("/usr/local/share/image-provenance").write_text(
    f"vllm {vllm.__version__}\ntorch {torch.__version__} hip {torch.version.hip}\n"
    f"triton {md.version('triton')}\n"
)
print(pathlib.Path("/usr/local/share/image-provenance").read_text())
PY

# The base's ENTRYPOINT is `vllm serve`; the CE and run_cluster.sh launch their own commands.
ENTRYPOINT []
WORKDIR /workspace
