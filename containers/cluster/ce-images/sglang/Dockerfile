# syntax=docker/dockerfile:1
#
# SGLang inference on AMD MI300A (gfx942) -- the engine 12 of the 18 llr40v9 arms serve on
# (qwen38 and kimi27; only oss120b stays on vLLM). Build context is the repository root:
#   podman build -f containers/cluster/ce-images/sglang/Dockerfile -t optarena-sglang .
#
# NOT YET BUILT. Authored from the deployed artifact, which was a PULL and never had a
# Dockerfile: ~/.edf/sglang-rocm-mi30x.toml names the stock image below, pulled by the
# pull-vllm-kimi-k3.sbatch recipe as job 604787 (52.3 GB). Everything here is derived from
# that toml, from run_cluster.sh's sglang branch, and from the SGLANG_EXTRA_ARGS of the live
# .env.llr40v9-{kimi27sglang,qwen38}-* files. Verify by building and smoke-testing with
# inference/smoke-kimi-sglang.sbatch before this replaces the pull.
#
# The base is ROCm 7.2.0, NOT the 7.2.3 the agent/judge image uses. That is the stock tag AMD
# publishes for this SGLang release; it is what every measured kimi and qwen38 number was
# produced on, so it is pinned rather than aligned to 7.2.3 for tidiness.
ARG BASE_IMAGE=docker.io/lmsysorg/sglang-rocm:v0.5.18-rocm720-mi30x-20260822
FROM ${BASE_IMAGE}

# FlyDSL is the MoE kernel DSL kimi's int4 gates dispatch through. It is currently NOT in the
# image: the arms reach it through PYTHONPATH=${SCRATCH}/pyprefix/sglang-rocm-mi30x, an
# out-of-image directory built by inference/install-flydsl.sbatch. Baking it in is the one
# substantive change this file makes over the pulled image -- an external PYTHONPATH is
# invisible to the image digest, so two runs of "the same image" can differ by it.
ARG FLYDSL_VERSION=0.3.2
RUN /opt/venv/bin/python3 -m pip install --no-cache-dir "flydsl==${FLYDSL_VERSION}"

# Deliberately ABSENT: aws-ofi-nccl / the CXI plugin. It is not bind-mountable into this image
# (image glibc 2.39 vs host 2.35) and grafting the host network stack kills the image's loader
# (595217), so the EDF sets neither com.hooks.cxi.enabled nor com.hooks.netstack.source and
# cross-node RCCL rides TCP over hsn*. The image already ships libfabric and libcxi.
#
# Deliberately ABSENT: an aiter master switch. SGLANG_USE_AITER=1 is set per RUN by the launcher,
# paired with SGLANG_ROCM_FUSED_DECODE_MLA=0; aiter JITs on the first request behind a baton lock
# that outlives any deadline, so AITER_JIT_DIR must point at persistent scratch at run time
# (smoke-kimi-sglang.sbatch does this after 610164 blew the 300 s watchdog).

ENV HSA_ENABLE_IPC_MODE_LEGACY=1 \
    SAFETENSORS_FAST_GPU=1 \
    HIP_FORCE_DEV_KERNARG=1 \
    TOKENIZERS_PARALLELISM=false \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64 \
    PATH=/usr/local/bin:/opt/rocm/bin:/usr/sbin:/usr/bin:/sbin:/bin

# The image's PATH omits its own venv, so a bare python3 has no sglang and no huggingface_hub.
# run_cluster.sh names the interpreter explicitly for that reason (SGLANG_PYTHON); the check
# below fails the build rather than letting a moved venv surface as a launcher error.
RUN set -eux; \
    /opt/venv/bin/python3 -c 'import sglang; print("sglang", sglang.__version__)'; \
    /opt/venv/bin/python3 -c 'import flydsl; print("flydsl", flydsl.__version__)'; \
    /opt/venv/bin/python3 -c 'import huggingface_hub; print("hf_hub", huggingface_hub.__version__)'; \
    /opt/venv/bin/python3 -m sglang.launch_server --help > /dev/null

WORKDIR /workspace
