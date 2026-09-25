#!/usr/bin/env bash
# GLM-5.3 on MI300A via SGLang: the proven config, as a file rather than a shell line.
#
# Deviations from the kimi defaults in smoke-kimi-sglang.sbatch worth flagging:
#   LANGUAGE_ONLY=0  --language-only selects the VLM encoder-disaggregation RECEIVER role here,
#                    not "text only"; GlmMoeDsaForCausalLM is off its allowlist and raises at launch.
#   MEM_FRACTION     a ceiling on weights+KV together on MI300A, not a KV reservation. 0.55 keeps
#                    the heaviest (uneven) PP stage under the host OOM killer.
#
# GLM-5.3-Flash cannot run here: its index_kpool forces IndexerKPool, CUDA-only. Plain 5.3 uses
# the ROCm-capable DSA Indexer.
set -Eeuo pipefail

ulimit -c 0
cd "$(dirname "${BASH_SOURCE[0]}")"

# DSA backend, overridable via SGLANG_EXTRA_ARGS: GlmMoeDsaForCausalLM ignores --attention-backend.
# On gfx942 the option set is {tilelang, aiter}; tilelang is the proven default.
#
# Nothing may be inserted between the assignments below and the sbatch they prefix: this is one
# backslash-continued command, and a comment line in the middle would end it at the '#', silently
# dropping every variable after it.
# EDF pinned: an image whose sglang still reads weight_scale.format_ue8m0 as a plain attribute
# dies ~31 min in loading GLM, after the weights are already on the nodes.
EDF="${EDF:-${HOME}/.edf/hpcagent-bench-sglang-mi300-latest.toml}" \
MODEL_REPO=zai-org/GLM-5.3 \
SERVED_MODEL=glm-5.3 \
TOOL_PARSER=glm47 \
REASONING_PARSER=glm45 \
LANGUAGE_ONLY=0 \
MEM_FRACTION="${MEM_FRACTION:-0.55}" \
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}" \
CONTEXT_LEN="${CONTEXT_LEN:-131072}" \
TP_SIZE=4 \
READY_TIMEOUT=7200 \
GATE_MAX_TOKENS="${GATE_MAX_TOKENS:-2048}" \
REASONING_EFFORT="${REASONING_EFFORT:-max}" \
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:---dsa-prefill-backend tilelang --dsa-decode-backend tilelang}" \
    sbatch --job-name=smoke-glm53-sglang "$@" smoke-kimi-sglang.sbatch
