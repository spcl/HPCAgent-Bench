#!/usr/bin/env bash
set -Eeuo pipefail

: "${SLURM_JOB_ID:?Must run inside Slurm}"
: "${SLURM_NODEID:?Missing SLURM_NODEID}"
: "${SLURM_JOB_NODELIST:?Missing SLURM_JOB_NODELIST}"

ROOT="${VLLM_BUILD_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"

MODEL_REPO="Qwen/Qwen2.5-14B-Instruct"
SERVED_MODEL="qwen14b-distributed"

NODE_RANK="${SLURM_NODEID}"
NNODES="${SLURM_NNODES:-2}"

# Avoid collisions with other jobs.
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
API_PORT=8000

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
export VLLM_DISABLE_PYNCCL=1

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

# Qwen2.5-14B BF16 should be substantially larger than 20 GiB.
if total_gib < 20:
    raise SystemExit(
        f"Model snapshot appears incomplete: {total_gib:.3f} GiB"
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
assert total_bytes >= 20 * 1024**3
PY_VALIDATE

echo "===== launch configuration ====="
echo "model path:       $MODEL_PATH"
echo "tensor parallel:  4"
echo "pipeline parallel: 2"
echo "world size:       8"
echo "PyNCCL disabled:  ${VLLM_DISABLE_PYNCCL}"

COMMON_ARGS=(
    "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL"
    --dtype bfloat16
    --load-format safetensors
    --tensor-parallel-size 4
    --pipeline-parallel-size 2
    --distributed-executor-backend mp
    --nnodes "$NNODES"
    --node-rank "$NODE_RANK"
    --master-addr "$MASTER_ADDR"
    --master-port "$MASTER_PORT"
    --distributed-timeout-seconds 3600
    --max-model-len 8192
    --gpu-memory-utilization 0.85
    --generation-config vllm
    --enforce-eager
)

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

python - "$API_PORT" <<'PY_REQUESTS'
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import statistics
import sys
import time
import urllib.request

port = int(sys.argv[1])
url = f"http://127.0.0.1:{port}/v1/chat/completions"

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
        "model": "qwen14b-distributed",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a concise and accurate technical assistant."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0,
        "max_tokens": 128,
    }

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

    text = choices[0]["message"]["content"]
    assert text.strip(), f"Request {index} returned empty text"

    usage = result.get("usage", {})

    return {
        "index": index,
        "elapsed": elapsed,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "text": text,
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
            f"completion_tokens={result['completion_tokens']}"
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
print("sample response:")
print(results[0]["text"])
print("==============================")

assert len(results) == 16
assert completion_tokens > 0
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
