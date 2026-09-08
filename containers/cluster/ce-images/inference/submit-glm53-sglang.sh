#!/usr/bin/env bash
# GLM-5.3 on MI300A via SGLang: the config that 628603 proved, as a file rather than a shell line.
#
# Everything here is a deviation from the kimi defaults in smoke-kimi-sglang.sbatch. The two that
# are not obvious:
#   LANGUAGE_ONLY=0  --language-only selects the VLM encoder-disaggregation RECEIVER role in this
#                    SGLang, not "text only". GlmMoeDsaForCausalLM is off its allowlist and the
#                    server raises at launch (628586, 628589).
#   MEM_FRACTION     a CEILING on weights+KV together on MI300A, not a KV reservation. The pool is
#                    KV(f) = 443.4 * (f - 0.486) GB/rank at tp=4 pp=4; below 0.486 sglang refuses
#                    outright, and 0.62 let the heaviest PP stage (node3, 206.1 GB -- the stages
#                    are UNEVEN, size against the max) reach the host OOM killer. 0.55 measured
#                    within 1.5% of the law.
#
# GLM-5.3-Flash cannot run here at all: its config.json carries index_kpool, which forces
# IndexerKPool, which raises "kpool indexer is only supported on CUDA". Plain 5.3 uses the DSA
# Indexer, which is ROCm-capable. Do not re-derive this.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

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
SGLANG_EXTRA_ARGS="--dsa-prefill-backend tilelang --dsa-decode-backend tilelang" \
    sbatch --job-name=smoke-glm53-sglang "$@" smoke-kimi-sglang.sbatch
