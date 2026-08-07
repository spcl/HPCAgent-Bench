#!/usr/bin/env bash
set -Eeuo pipefail

# Preset wrapper for GPT-OSS-120B on 1 node / 4 MI300A GPUs. Thin shim over
# test-vllm-gpt-oss-20b-1n4g.sh: sets the model-specific knobs, then execs
# the real harness so its runtime checks and 16-stream test run unchanged.
#
# TOOL_CALL_PARSER / REASONING_PARSER are left UNSET below. Confirm the
# exact parser names for gpt-oss under vLLM 0.23.0 at the first smoke run:
# run `vllm serve --help` inside the inference container and read the
# accepted --tool-call-parser / --reasoning-parser values for this build.
# Function calling for the agent loop needs --enable-auto-tool-choice plus
# the matching --tool-call-parser; the harness wires both in automatically
# once TOOL_CALL_PARSER is exported (see test-vllm-gpt-oss-20b-1n4g.sh).

export MODEL_REPO="${MODEL_REPO:-openai/gpt-oss-120b}"
export TP_SIZE="${TP_SIZE:-4}"
export PP_SIZE="${PP_SIZE:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
# GPU_MEMORY_UTILIZATION intentionally left at the harness default.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/test-vllm-gpt-oss-20b-1n4g.sh" "$@"
