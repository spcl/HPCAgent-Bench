# syntax=docker/dockerfile:1
#
# vLLM inference on NVIDIA GH200 (aarch64 Grace + H100, CSCS Daint). Serves every campaign model
# there -- qwen38, kimi and oss120b -- from ONE engine, where beverin needs SGLang for two of them.
# Build context is the repository root:
#
#   containers/cluster/ce-images/vllm-cuda/build.sh
#
# THIN ON PURPOSE. The official vLLM OpenAI image already carries the engine, torch, NCCL, triton,
# flashinfer and every tool-call / reasoning parser the campaign names. What beverin's vllm/Dockerfile
# adds on top of its base is ROCm-shaped and does not transfer:
#   * aiter, flash-attn's triton_amd path, the aiter JIT prebuild     -- ROCm kernels
#   * inference/moe-configs/                                          -- tuned for AMD_Instinct_MI300A;
#                                                                        vLLM ships its own H100 configs
#   * inference/external-eager-pg-patch/                              -- an RCCL pipeline-bootstrap
#                                                                        workaround, opt-in, never measured on NCCL
# So this file pins the base by digest, labels it, and GATES what the serving configs need: the
# engine version, a CUDA torch, and each parser name by the registry the server resolves it from --
# a misnamed parser starts the server and fails the first request (verify-tools-reasoning.py).
#
# No fabric software: the CE hooks inject libfabric/CXI and the NCCL net plugin (edf.toml.example).
#
# PINNED BY DIGEST: vllm/vllm-openai:v0.30.0-aarch64-cu129, the arm64-only CUDA 12.9 build of the
# v0.30.0 release (2026-09-22): CUDA 12.9.1, python 3.12, NCCL 2.30.7, TORCH_CUDA_ARCH_LIST includes
# 9.0, Ubuntu 24.04, ~13.7 GB compressed. The digest is the tag's manifest digest, resolved from the
# registry without a pull. CUDA 12.9 rather than the release's default CUDA 13 build for the reasons
# judge-agent-cuda gives: any driver >= 525 runs it, and it matches the CE's cuda12 NCCL plugin.
#   the CUDA 13.0 build, v0.30.0-aarch64, if Daint ever moves to a cuda13 plugin:
#     sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56
ARG BASE_IMAGE=docker.io/vllm/vllm-openai:v0.30.0-aarch64-cu129@sha256:d2f87fcd67d8c80c7c68d2faf3e97f6e6facb688d04734d906f7ee2f70e6a311
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
# The registry reference, not a base-image cache dir: path (build_common.sh ce_cache_base_image).
ARG BASE_IMAGE_REF="${BASE_IMAGE}"
ARG IMAGE_VERSION=dev
# The release and CUDA series the base digest is. Asserted below, so a BASE_IMAGE override to another
# release or CUDA major has to say so rather than silently serving on a different engine or runtime.
ARG VLLM_VERSION=0.30.0
ARG TORCH_CUDA=12.9
LABEL org.opencontainers.image.title="hpcagent-bench-vllm-cuda" \
      org.opencontainers.image.description="HPCAgent-Bench vLLM ${VLLM_VERSION} inference for GH200 (qwen38, kimi, oss120b)" \
      org.opencontainers.image.source="https://github.com/spcl/HPCAgent-Bench" \
      org.opencontainers.image.licenses="GPL-3.0-or-later" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.base.name="${BASE_IMAGE_REF}"

# What can be checked WITHOUT a GPU. The device half (NCCL over CXI, a real serve) is the smoke in
# README.md's Daint section; build.sbatch runs verify_image.py --profile vllm-cuda inside the image.
RUN python3 - <<'PY'
import importlib
import importlib.metadata as md
import os
import pathlib

import torch
import vllm

want = os.environ["VLLM_VERSION"]
assert vllm.__version__.split("+")[0] == want, f"vllm {vllm.__version__}, image declares {want}"
assert (torch.version.cuda or "").startswith(os.environ["TORCH_CUDA"]), f"torch CUDA {torch.version.cuda}"
assert "9.0" in os.environ.get("TORCH_CUDA_ARCH_LIST", ""), "the base was not built for sm_90 (GH200)"
for mod in ("vllm._C", "triton", "huggingface_hub", "vllm.entrypoints.anthropic.api_router"):
    importlib.import_module(mod)

# Every parser a serving config in this repo names, resolved the way `vllm serve` resolves it.
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager

for name in ("qwen3_coder", "kimi_k2", "openai"):
    ToolParserManager.get_tool_parser(name)
for name in ("qwen3", "kimi_k2", "openai_gptoss"):
    ReasoningParserManager.get_reasoning_parser(name)

# The images ship no fabric library of their own (IMAGE_REQUIREMENTS.md): one in the image would
# shadow what the CE hooks inject.
shipped = [str(p) for root in ("/usr/lib", "/usr/local/lib", "/opt") for p in pathlib.Path(root).rglob("libnccl-net*.so*")]
assert not shipped, f"the image ships an NCCL net plugin: {shipped}"

pathlib.Path("/usr/local/share").mkdir(parents=True, exist_ok=True)
pathlib.Path("/usr/local/share/image-provenance").write_text(
    f"vllm {vllm.__version__}\ntorch {torch.__version__} cuda {torch.version.cuda}\n"
    f"nccl {'.'.join(map(str, torch.cuda.nccl.version()))}\ntriton {md.version('triton')}\n"
)
print(pathlib.Path("/usr/local/share/image-provenance").read_text())
PY

WORKDIR /workspace
