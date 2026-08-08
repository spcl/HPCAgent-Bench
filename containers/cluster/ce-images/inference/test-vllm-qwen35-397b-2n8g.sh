#!/usr/bin/env bash
set -Eeuo pipefail

: "${SLURM_JOB_ID:?Must run inside Slurm}"
: "${SLURM_NODEID:?Missing SLURM_NODEID}"
: "${SLURM_JOB_NODELIST:?Missing SLURM_JOB_NODELIST}"

ROOT="${VLLM_BUILD_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"

MODEL_REPO="${MODEL_REPO:?Set MODEL_REPO}"
SERVED_MODEL="${SERVED_MODEL:-stress-model}"

NODE_RANK="${SLURM_NODEID}"
NNODES="${SLURM_NNODES:-2}"

# Avoid collisions with other jobs.
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
API_PORT="${API_PORT:-8000}"

mapfile -t NODES < <(
    scontrol show hostnames "${SLURM_JOB_NODELIST}"
)

if [[ "${#NODES[@]}" -ne "$NNODES" ]]; then
    echo "Expected ${NNODES} nodes, found ${#NODES[@]}" >&2
    exit 1
fi

MASTER_ADDR="${NODES[0]}"

RUN_DIR="${ROOT}/logs/vllm/2n8g.${SLURM_JOB_ID}"
SERVER_LOG="${RUN_DIR}/node-${NODE_RANK}.$(hostname).log"

MODEL_PATH_FILE="${RUN_DIR}/model.path"
PREFETCH_DONE="${RUN_DIR}/model-prefetch.done"
PREFETCH_FAILED="${RUN_DIR}/model-prefetch.failed"
JOB_DONE="${RUN_DIR}/job.done"

mkdir -p "$RUN_DIR"

export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=INFO

# The direct PyNCCL communicator stalled in the first 2-node run.
# Use the already-qualified torch.distributed/RCCL path instead.

# Model loading plus ROCm compilation and graph capture can exceed 600 seconds.
export VLLM_ENGINE_READY_TIMEOUT_S=3600

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_DEBUG_FILE="${RUN_DIR}/nccl.%h.%p.log"

echo "=================================================="
echo "host:             $(hostname)"
echo "node rank:        ${NODE_RANK}"
echo "nodes:            ${NODES[*]}"
echo "master:           ${MASTER_ADDR}:${MASTER_PORT}"
echo "visible devices:  ${ROCR_VISIBLE_DEVICES:-unset}"
echo "server log:       ${SERVER_LOG}"
echo "=================================================="

echo "===== runtime and GPU validation ====="

echo "python: $(command -v python)"
echo "vllm: $(command -v vllm)"
echo "VIRTUAL_ENV: ${VIRTUAL_ENV:-unset}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-unset}"

test "$(command -v python)" = "/opt/pytorch211/bin/python"
test "$(command -v vllm)" = "/opt/pytorch211/bin/vllm"
test "${VIRTUAL_ENV:-}" = "/opt/pytorch211"

python - <<'PY_RUNTIME'
import importlib
import sys

import torch
import vllm
from vllm.platforms import current_platform

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("PyTorch file:", torch.__file__)
print("HIP:", torch.version.hip)
print("RCCL:", torch.cuda.nccl.version())
print("vLLM:", vllm.__version__)
print("vLLM file:", vllm.__file__)
print("Platform:", current_platform.__class__)
print("GPU count:", torch.cuda.device_count())

assert sys.executable == "/opt/pytorch211/bin/python"
assert torch.__version__ == "2.11.0+rocm7.2"
assert torch.__file__.startswith("/opt/pytorch211/")
assert torch.version.hip is not None
assert torch.version.hip.startswith("7.2")
assert vllm.__version__ == "0.23.0"
assert vllm.__file__.startswith("/opt/vllm-src/")
assert "RocmPlatform" in current_platform.__class__.__name__
assert torch.cuda.device_count() == 4

for module in (
    "vllm._C",
    "vllm._rocm_C",
    "vllm._C_stable_libtorch",
):
    importlib.import_module(module)
    print(module, "OK")

for name in (
    "silu_and_mul",
    "gelu_and_mul",
    "rms_norm",
    "fused_add_rms_norm",
):
    assert hasattr(torch.ops._C, name), name
    print(f"torch.ops._C.{name}: OK")

for index in range(torch.cuda.device_count()):
    print(
        f"GPU {index}:",
        torch.cuda.get_device_name(index),
    )

print("RUNTIME VALIDATION PASSED")
PY_RUNTIME

echo "===== shared model prefetch ====="

if [[ "$NODE_RANK" -eq 0 ]]; then
    rm -f \
        "$MODEL_PATH_FILE" \
        "$PREFETCH_DONE" \
        "$PREFETCH_FAILED" \
        "$JOB_DONE"

    if ! python - \
        "$MODEL_REPO" \
        "$MODEL_PATH_FILE" <<'PY_PREFETCH'
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
path_file = Path(sys.argv[2])

print("Repository:", repo_id)
print("Using global Hugging Face configuration")

snapshot = Path(
    snapshot_download(
        repo_id=repo_id,
        max_workers=8,
    )
).resolve()

if not snapshot.is_dir():
    raise SystemExit(f"Snapshot directory does not exist: {snapshot}")

required_files = [
    snapshot / "config.json",
    snapshot / "tokenizer_config.json",
]

missing_required = [
    str(path)
    for path in required_files
    if not path.is_file()
]

if missing_required:
    raise SystemExit(
        "Missing required model files: "
        + ", ".join(missing_required)
    )

weight_files = sorted(snapshot.glob("*.safetensors"))

if not weight_files:
    raise SystemExit(
        f"No safetensors weight files found in {snapshot}"
    )

broken_weights = [
    str(path)
    for path in weight_files
    if not path.is_file()
]

if broken_weights:
    raise SystemExit(
        "Broken or missing weight files: "
        + ", ".join(broken_weights)
    )

total_bytes = sum(
    path.stat().st_size
    for path in weight_files
)

total_gib = total_bytes / 1024**3

print("Resolved snapshot:", snapshot)
print("Weight shards:", len(weight_files))
print("Weight size GiB:", round(total_gib, 3))

for path in weight_files:
    print(
        f"{path.name}: "
        f"{path.stat().st_size / 1024**3:.3f} GiB"
    )

minimum_gib = float(os.environ.get("MIN_MODEL_GIB", "1"))
if total_gib < minimum_gib:
    raise SystemExit(
        f"Model snapshot appears incomplete: {total_gib:.3f} GiB; "
        f"minimum is {minimum_gib:.3f} GiB"
    )

temporary = path_file.with_suffix(".tmp")
temporary.write_text(str(snapshot) + "\n")
os.replace(temporary, path_file)

print("Model path file:", path_file)
PY_PREFETCH
    then
        touch "$PREFETCH_FAILED"
        exit 1
    fi

    touch "$PREFETCH_DONE"
else
    deadline=$((SECONDS + 3600))

    while [[ ! -e "$PREFETCH_DONE" ]]; do
        if [[ -e "$PREFETCH_FAILED" ]]; then
            echo "Model prefetch failed on node rank 0" >&2
            exit 1
        fi

        if (( SECONDS >= deadline )); then
            echo "Timed out waiting for model prefetch" >&2
            exit 1
        fi

        sleep 5
    done
fi

if [[ ! -s "$MODEL_PATH_FILE" ]]; then
    echo "Missing model path file: $MODEL_PATH_FILE" >&2
    exit 1
fi

IFS= read -r MODEL_PATH < "$MODEL_PATH_FILE"

echo "===== model validation on $(hostname) ====="

python - "$MODEL_PATH" <<'PY_VALIDATE'
import sys
import os
from pathlib import Path

snapshot = Path(sys.argv[1])

weights = sorted(snapshot.glob("*.safetensors"))
total_bytes = sum(
    path.stat().st_size
    for path in weights
    if path.is_file()
)

print("Snapshot:", snapshot)
print("Weight shards:", len(weights))
print("Weight size GiB:", round(total_bytes / 1024**3, 3))

assert snapshot.is_dir()
assert snapshot.joinpath("config.json").is_file()
assert snapshot.joinpath("tokenizer_config.json").is_file()
assert weights
minimum_gib = float(os.environ.get("MIN_MODEL_GIB", "1"))
assert total_bytes >= minimum_gib * 1024**3
PY_VALIDATE

echo "===== launch configuration ====="
echo "model path:       $MODEL_PATH"
echo "tensor parallel:  4"
echo "pipeline parallel: ${PP_SIZE:-2}"
echo "world size:       8"

COMMON_ARGS=(
    "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL"
    --dtype bfloat16
    --load-format safetensors
    --tensor-parallel-size "${TP_SIZE:-4}"
    --pipeline-parallel-size "${PP_SIZE:-2}"
    --distributed-executor-backend mp
    --nnodes "$NNODES"
    --node-rank "$NODE_RANK"
    --master-addr "$MASTER_ADDR"
    --master-port "$MASTER_PORT"
    --distributed-timeout-seconds 3600
    --max-model-len "${MAX_MODEL_LEN:-8192}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
    --generation-config vllm
)

if [[ "${DISABLE_PREFIX_CACHING:-0}" == "1" ]]; then
    COMMON_ARGS+=(--no-enable-prefix-caching)
fi

if [[ "${LANGUAGE_MODEL_ONLY:-0}" == "1" ]]; then
    COMMON_ARGS+=(--language-model-only)
fi

if [[ -n "${TOOL_CALL_PARSER:-}" ]]; then
    COMMON_ARGS+=(
        --enable-auto-tool-choice
        --tool-call-parser "$TOOL_CALL_PARSER"
    )
fi

if [[ -n "${REASONING_PARSER:-}" ]]; then
    COMMON_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi

if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
    COMMON_ARGS+=(
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    )
fi

if [[ -n "${MAX_NUM_SEQS:-}" ]]; then
    COMMON_ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
fi

if [[ -n "${MAX_CUDAGRAPH_CAPTURE_SIZE:-}" ]]; then
    COMMON_ARGS+=(
        --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"
    )
fi

if [[ "${ENABLE_EXPERT_PARALLEL:-0}" == "1" ]]; then
    COMMON_ARGS+=(--enable-expert-parallel)
fi

if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
    COMMON_ARGS+=(--trust-remote-code)
fi

if [[ -n "${CHAT_TEMPLATE_CONTENT_FORMAT:-}" ]]; then
    COMMON_ARGS+=(
        --chat-template-content-format
        "$CHAT_TEMPLATE_CONTENT_FORMAT"
    )
fi

if [[ -n "${MM_ENCODER_TP_MODE:-}" ]]; then
    COMMON_ARGS+=(
        --mm-encoder-tp-mode
        "$MM_ENCODER_TP_MODE"
    )
fi

if [[ -n "${HF_OVERRIDES_JSON:-}" ]]; then
    COMMON_ARGS+=(
        --hf-overrides
        "$HF_OVERRIDES_JSON"
    )
fi

if [[ -n "${SPECULATIVE_METHOD:-}" ]]; then
    COMMON_ARGS+=(
        --speculative-config.method
        "$SPECULATIVE_METHOD"
    )
fi

if [[ -n "${NUM_SPECULATIVE_TOKENS:-}" ]]; then
    COMMON_ARGS+=(
        --speculative-config.num_speculative_tokens
        "$NUM_SPECULATIVE_TOKENS"
    )
fi

if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
    read -r -a EXTRA_ARGS_ARRAY <<<"$EXTRA_VLLM_ARGS"
    COMMON_ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi

if [[ "$NODE_RANK" -eq 0 ]]; then
    SERVER_ARGS=(
        "${COMMON_ARGS[@]}"
        --host 0.0.0.0
        --port "$API_PORT"
    )
else
    SERVER_ARGS=(
        "${COMMON_ARGS[@]}"
        --headless
    )
fi

SERVER_PID=""

cleanup() {
    status=$?

    if [[ "$NODE_RANK" -eq 0 ]]; then
        touch "$JOB_DONE"
    fi

    if [[ -n "$SERVER_PID" ]] &&
       kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true

        for _ in {1..20}; do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done

        if kill -0 "$SERVER_PID" 2>/dev/null; then
            kill -9 "$SERVER_PID" 2>/dev/null || true
        fi

        wait "$SERVER_PID" 2>/dev/null || true
    fi

    if (( status != 0 )); then
        echo
        echo "===== failure log: $SERVER_LOG ====="
        tail -n 250 "$SERVER_LOG" 2>/dev/null || true
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "===== starting vLLM on node rank ${NODE_RANK} ====="

vllm serve "${SERVER_ARGS[@]}" \
    >"$SERVER_LOG" 2>&1 &

SERVER_PID=$!

echo "vLLM PID: $SERVER_PID"

if [[ "$NODE_RANK" -ne 0 ]]; then
    echo "Headless worker waiting for the head node."

    while [[ ! -e "$JOB_DONE" ]]; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Headless vLLM process exited unexpectedly" >&2
            wait "$SERVER_PID" || true
            exit 1
        fi

        if [[ -r "/proc/${SERVER_PID}/stat" ]]; then
            process_state=$(
                awk '{print $3}' "/proc/${SERVER_PID}/stat"
            )

            if [[ "$process_state" == "Z" ]]; then
                echo "Headless vLLM process became a zombie" >&2
                exit 1
            fi
        fi

        sleep 3
    done

    echo "Head node signalled completion."
    exit 0
fi

echo "===== waiting for distributed API ====="

python - "$SERVER_PID" "$API_PORT" <<'PY_READY'
import json
import os
import sys
import time
import urllib.error
import urllib.request

pid = int(sys.argv[1])
port = int(sys.argv[2])

url = f"http://127.0.0.1:{port}/v1/models"
deadline = time.monotonic() + 3600

while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit(
            "Head vLLM process exited before API readiness"
        )

    stat_path = f"/proc/{pid}/stat"

    try:
        process_state = open(
            stat_path,
            encoding="utf-8",
        ).read().split()[2]
    except FileNotFoundError:
        raise SystemExit(
            "Head vLLM process disappeared before API readiness"
        )

    if process_state == "Z":
        raise SystemExit(
            "Head vLLM process became a zombie before API readiness"
        )

    try:
        with urllib.request.urlopen(
            url,
            timeout=5,
        ) as response:
            if response.status == 200:
                payload = json.load(response)
                print(json.dumps(payload, indent=2))
                print("Distributed API is ready")
                break
    except (
        urllib.error.URLError,
        TimeoutError,
        ConnectionError,
    ):
        pass

    time.sleep(3)
else:
    raise SystemExit(
        "Distributed API did not become ready within 3600 seconds"
    )
PY_READY

echo "===== concurrent inference test ====="

python - "$API_PORT" "$SERVED_MODEL" <<'PY_REQUESTS'
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import statistics
import sys
import time
import urllib.request

port = int(sys.argv[1])
served_model = sys.argv[2]
url = f"http://127.0.0.1:{port}/v1/chat/completions"

system_prompt = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a concise and accurate technical assistant.",
)

temperature = float(
    os.environ.get("REQUEST_TEMPERATURE", "0")
)

max_output_tokens = int(
    os.environ.get("REQUEST_MAX_TOKENS", "128")
)

enable_thinking = os.environ.get(
    "ENABLE_THINKING",
    "",
).strip().lower()

request_extra_json = os.environ.get(
    "REQUEST_EXTRA_JSON",
    "",
).strip()

request_extra = {}

if request_extra_json:
    request_extra = json.loads(request_extra_json)

    if not isinstance(request_extra, dict):
        raise TypeError(
            "REQUEST_EXTRA_JSON must decode to a JSON object"
        )

prompts = [
    "Explain tensor parallelism and pipeline parallelism.",
    "Explain why collective communication affects LLM inference.",
    "Describe KV-cache memory use in an inference server.",
    "Compare eager execution with graph execution.",
    "Explain why GPU memory bandwidth matters for inference.",
    "Describe the purpose of an HPC interconnect.",
    "Explain the difference between latency and throughput.",
    "List five checks for a distributed GPU deployment.",
] * 2


def send_request(index: int, prompt: str) -> dict:
    payload = {
        "model": served_model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }

    if enable_thinking in {
        "1",
        "true",
        "yes",
        "on",
    }:
        chat_template_kwargs = payload.setdefault(
            "chat_template_kwargs",
            {},
        )

        chat_template_kwargs["enable_thinking"] = True

    elif enable_thinking in {
        "0",
        "false",
        "no",
        "off",
    }:
        chat_template_kwargs = payload.setdefault(
            "chat_template_kwargs",
            {},
        )

        chat_template_kwargs["enable_thinking"] = False

    payload.update(request_extra)

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.monotonic()

    with urllib.request.urlopen(
        request,
        timeout=600,
    ) as response:
        result = json.load(response)

    elapsed = time.monotonic() - started

    choices = result.get("choices", [])
    assert choices, f"Request {index} returned no choices"

    message = choices[0].get("message") or {}

    if not isinstance(message, dict):
        raise TypeError(
            f"Request {index} returned a non-object message: "
            f"{message!r}"
        )

    def normalize(value) -> str:
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(value, list):
            parts = [
                normalize(item)
                for item in value
            ]

            return "\n".join(
                part
                for part in parts
                if part.strip()
            )

        if isinstance(value, dict):
            for key in (
                "text",
                "content",
                "output_text",
                "reasoning",
                "reasoning_content",
            ):
                if key in value:
                    normalized = normalize(value[key])

                    if normalized.strip():
                        return normalized

            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
            )

        return str(value)

    content = normalize(
        message.get("content")
    )

    reasoning = normalize(
        message.get("reasoning_content")
        or message.get("reasoning")
        or message.get("analysis")
    )

    refusal = normalize(
        message.get("refusal")
    )

    tool_calls = (
        message.get("tool_calls")
        or []
    )

    if not isinstance(tool_calls, list):
        tool_calls = [tool_calls]

    visible_text = content or reasoning or refusal

    has_output = bool(
        visible_text.strip()
        or tool_calls
    )

    if not has_output:
        raise AssertionError(
            f"Request {index} returned no content, reasoning, "
            f"refusal, or tool calls. Full result: "
            f"{json.dumps(result, ensure_ascii=False)}"
        )

    if content.strip():
        response_kind = "content"
    elif reasoning.strip():
        response_kind = "reasoning"
    elif tool_calls:
        response_kind = "tool_calls"
    else:
        response_kind = "refusal"

    usage = result.get("usage") or {}

    return {
        "index": index,
        "elapsed": elapsed,
        "prompt_tokens": usage.get(
            "prompt_tokens",
            0,
        ),
        "completion_tokens": usage.get(
            "completion_tokens",
            0,
        ),
        "response_kind": response_kind,
        "text": visible_text,
        "tool_calls": tool_calls,
        "has_output": has_output,
    }


overall_started = time.monotonic()
results = []

with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [
        executor.submit(send_request, index, prompt)
        for index, prompt in enumerate(prompts)
    ]

    for future in as_completed(futures):
        result = future.result()
        results.append(result)

        print(
            f"request={result['index']:02d} "
            f"seconds={result['elapsed']:.3f} "
            f"prompt_tokens={result['prompt_tokens']} "
            f"completion_tokens={result['completion_tokens']} "
            f"kind={result['response_kind']}"
        )

overall_elapsed = time.monotonic() - overall_started

latencies = [
    result["elapsed"]
    for result in results
]

prompt_tokens = sum(
    result["prompt_tokens"]
    for result in results
)

completion_tokens = sum(
    result["completion_tokens"]
    for result in results
)

print()
print("===== DISTRIBUTED RESULTS =====")
print("requests:", len(results))
print("wall seconds:", round(overall_elapsed, 3))
print(
    "median request seconds:",
    round(statistics.median(latencies), 3),
)
print(
    "maximum request seconds:",
    round(max(latencies), 3),
)
print("prompt tokens:", prompt_tokens)
print("completion tokens:", completion_tokens)
print(
    "aggregate completion tokens/s:",
    round(completion_tokens / overall_elapsed, 3),
)
print()
sample = results[0]

print("sample response kind:")
print(sample["response_kind"])

print("sample response:")
print(sample["text"] or "<no visible text>")

if sample["tool_calls"]:
    print("sample tool calls:")
    print(
        json.dumps(
            sample["tool_calls"],
            indent=2,
            ensure_ascii=False,
        )
    )
print("==============================")

assert len(results) == 16
assert any(
    result["has_output"]
    for result in results
)
PY_REQUESTS

echo "===== vLLM topology evidence ====="

grep -E \
    'world_size=8|assigned as DP rank|PP rank|TP rank|Starting to load model|Model loading took|Graph capturing|CUDAGraph|API server' \
    "$RUN_DIR"/node-*.log \
    | tail -n 200 || true

echo "===== RCCL / CXI evidence ====="

shopt -s nullglob

NCCL_LOGS=("$RUN_DIR"/nccl.*.log)

if (( ${#NCCL_LOGS[@]} == 0 )); then
    echo "No separate NCCL logs were produced."
else
    grep -E \
        'NET/Plugin|AWS Libfabric|Using network|libfabric|cxi|Connected all rings|Init COMPLETE' \
        "${NCCL_LOGS[@]}" \
        | tail -n 300 || true
fi

echo
echo "TWO-NODE EIGHT-GPU VLLM TEST PASSED"

touch "$JOB_DONE"
