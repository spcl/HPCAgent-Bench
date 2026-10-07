# Qwen3.8-27B on one GH200 node (Daint): local vLLM

`Qwen/Qwen3.8-27B-FP8` on vLLM 0.28 (`hpcagent-bench-vllm-gh200-latest`), one node, TP4, no MTP.
The server listens on `127.0.0.1:8000` of the node and answers to the model name `q38`. Measured
with 40 concurrent agents, 40k-token shared prefix and 2k suffix: 595-677 output tok/s, median TTFT
about 1.1 s, prefix-cache hit rate about 75 %. Background and the
alternatives that lost (DP4, MTP): [`qwen38.md`](qwen38.md#gh200-daint).

## 1. One-time setup

The weights are already in `$FAST_SCRATCH/.hpcagentbench-cache/hf` (fetch others with
`containers/inference/fetch_weights.sbatch`). Write the EDF once:

```bash
mkdir -p ~/.edf
cat > ~/.edf/qwen38-gh200.toml <<EOF
image = "$SCRATCH/ce-images/hpcagent-bench-vllm-gh200-latest.sqsh"
mounts = [
  "$FAST_SCRATCH/.hpcagentbench-cache/hf:/hf",
  "$SCRATCH/.hpcagentbench-cache:$SCRATCH/.hpcagentbench-cache",
  "$SCRATCH/hpcagent-bench-daint/containers/inference:/tmpl",
]
workdir = "$SCRATCH"
[env]
HF_HOME = "/hf"
HF_HUB_OFFLINE = "1"
EOF
```

## 2. Start the server

`serve-qwen38.sh` (anywhere under `$SCRATCH`):

```bash
#!/bin/bash
# JIT caches under the shared cache folder: first start 504 s, later starts 132 s.
c=$SCRATCH/.hpcagentbench-cache k=hpcagent-bench-vllm-gh200-latest
export HOME=$c/.home/$k XDG_CACHE_HOME=$c/.xdg/$k VLLM_CACHE_ROOT=$c/.vllm/$k TRITON_CACHE_DIR=$c/.triton/$k \
  TORCHINDUCTOR_CACHE_DIR=$c/.inductor/$k TORCH_EXTENSIONS_DIR=$c/.torch-ext/$k
mkdir -p $HOME $XDG_CACHE_HOME $VLLM_CACHE_ROOT $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $TORCH_EXTENSIONS_DIR
exec vllm serve Qwen/Qwen3.8-27B-FP8 --host 127.0.0.1 --port 8000 --served-model-name q38 \
  --tensor-parallel-size 4 --language-model-only --max-model-len 262144 \
  --gpu-memory-utilization 0.90 --max-num-seqs 64 --max-num-batched-tokens 16384 --enable-prefix-caching \
  --chat-template /tmpl/chat-template-qwen38.jinja --trust-remote-code \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
```

```bash
sbatch -A "$SBATCH_ACCOUNT" -p normal --nodes=1 --gpus-per-node=4 --time=04:00:00 -o q38-%j.out \
  --wrap "srun --ntasks=1 --environment=qwen38-gh200 bash $SCRATCH/serve-qwen38.sh"
```

Use `-p debug --time=00:30:00` for a quick try. The server is ready when the log prints
`Application startup complete`.

## 3. Send requests

The endpoint is bound to the node's loopback, so clients run on that node. Open a shell there:

```bash
srun --jobid=<jobid> --overlap --pty --environment=qwen38-gh200 bash   # has curl, python3, openai
curl -s localhost:8000/health && echo ok
```

Plain chat. The answer is in `choices[0].message.content` and the thinking trace in
`choices[0].message.reasoning`. Disable thinking per request with
`"chat_template_kwargs": {"enable_thinking": false}`.

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "q38",
  "messages": [{"role": "user", "content": "What is a stencil kernel? Two sentences."}],
  "max_tokens": 1024, "temperature": 1.0, "top_p": 0.95, "top_k": 20
}' | python3 -m json.tool
```

## 4. Tool calls

Pass OpenAI-style `tools`. The model answers with `message.tool_calls`. Send the result back as a
`role: tool` message and the model continues.

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}]
messages = [{"role": "user", "content": "What is the weather in Lugano?"}]

reply = client.chat.completions.create(model="q38", messages=messages, tools=tools, max_tokens=2048)
msg = reply.choices[0].message
call = msg.tool_calls[0]
print(call.function.name, json.loads(call.function.arguments))  # get_weather {'city': 'Lugano'}

messages += [msg, {"role": "tool", "tool_call_id": call.id, "content": json.dumps({"temp_c": 21, "sky": "clear"})}]
final = client.chat.completions.create(model="q38", messages=messages, tools=tools, max_tokens=2048)
print(final.choices[0].message.content)
```

Check tool calls and reasoning in one go:

```bash
python3 $SCRATCH/hpcagent-bench-daint/containers/inference/verify-tools-reasoning.py \
  --base http://localhost:8000 --model q38
```

## Notes

- About 88 of 96 GB per GPU is in use. KV is bf16 (only 16 of the 64 layers carry KV): 4.91 M
  tokens, 18.7 requests at the full 262144 context.
- Every request shares one prefix cache, so agents that resend a long common prompt get it almost free.
- To reach the server from another node or a laptop, you have to add `--api-key` and change `--host`. Do not
  bind `0.0.0.0` without a key.
