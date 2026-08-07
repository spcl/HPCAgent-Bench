#!/usr/bin/env bash
set -Eeuo pipefail

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
SERVED_MODEL="qwen-test"

ROOT="${VLLM_BUILD_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
LOGDIR="${ROOT}/logs/vllm"
SERVER_LOG="${LOGDIR}/qwen-one-gpu.${SLURM_JOB_ID}.log"

mkdir -p "$LOGDIR"

export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=INFO

echo "Node:       $(hostname)"
echo "Model:      $MODEL"
echo "Server log: $SERVER_LOG"

echo "===== GPU visibility ====="
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("GPU count:", torch.cuda.device_count())

assert torch.cuda.device_count() == 1

print("GPU:", torch.cuda.get_device_name(0))
PY

echo "===== starting vLLM server ====="

vllm serve "$MODEL" \
    --served-model-name "$SERVED_MODEL" \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.80 \
    --generation-config vllm \
    >"$SERVER_LOG" 2>&1 &

SERVER_PID=$!

cleanup() {
    status=$?

    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi

    if (( status != 0 )); then
        echo
        echo "===== server log after failure ====="
        tail -n 200 "$SERVER_LOG" || true
    fi
}
trap cleanup EXIT

echo "Server PID: $SERVER_PID"

echo "===== waiting for API readiness ====="

python - "$SERVER_PID" <<'PY'
import json
import os
import sys
import time
import urllib.error
import urllib.request

server_pid = int(sys.argv[1])
models_url = "http://127.0.0.1:8000/v1/models"
deadline = time.monotonic() + 600

while time.monotonic() < deadline:
    try:
        os.kill(server_pid, 0)
    except ProcessLookupError:
        raise SystemExit("vLLM server exited before becoming ready")

    try:
        with urllib.request.urlopen(models_url, timeout=5) as response:
            if response.status == 200:
                payload = json.load(response)
                print(json.dumps(payload, indent=2))
                print("vLLM API is ready")
                break
    except (
        urllib.error.URLError,
        TimeoutError,
        ConnectionError,
    ):
        pass

    time.sleep(2)
else:
    raise SystemExit("vLLM API did not become ready")
PY

echo "===== sending chat-completions request ====="

python - <<'PY'
import json
import urllib.request

payload = {
    "model": "qwen-test",
    "messages": [
        {
            "role": "system",
            "content": "You are a concise technical assistant.",
        },
        {
            "role": "user",
            "content": (
                "In two sentences, explain what GPU tensor parallelism is."
            ),
        },
    ],
    "temperature": 0,
    "max_tokens": 128,
}

request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(request, timeout=300) as response:
    result = json.load(response)

print(json.dumps(result, indent=2))

choices = result.get("choices", [])
assert choices, "Response contains no choices"

text = choices[0]["message"]["content"]
assert text and text.strip(), "Generated response is empty"

print()
print("===== GENERATED ANSWER =====")
print(text.strip())
print("============================")
PY

echo
echo "REAL VLLM INFERENCE TEST PASSED"
