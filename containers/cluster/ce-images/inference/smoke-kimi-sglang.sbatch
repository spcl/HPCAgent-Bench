#!/usr/bin/env bash
# SGLang leg of the Kimi-on-gfx942 decision tree. vLLM 0.27.1 is exhausted: TRITON_MLA is
# correct but prefills at ~300 tok/s (starves the campaign), aiter master-on prefills 3x and
# runs the campaign shape at 78 tok/s but silently corrupts output past ~9k context (probes
# 604783/604789/604790). This runs the official SGLang K2.7 gfx942 recipe -- SGLANG_USE_AITER=1,
# fused decode MLA off, TP=4 in-node (AITER MLA needs heads%16==0) x PP=2 across 2 nodes,
# fp8_e4m3 KV, page 64 -- and gates on the SAME long-context accuracy questions before any
# throughput number counts. Short-context accuracy passes on broken kernels; only the long leg
# discriminates.
#SBATCH --job-name=smoke-kimi-sglang
#SBATCH --partition=mi300
# PP spans the allocation: -N4 gives the campaign's own topology, -N2 the 2-node recipe.
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=logs/smoke-kimi-sglang-%j.out
#SBATCH --error=logs/smoke-kimi-sglang-%j.err

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
export RUN_DIR="${SCRATCH}/kimi-smoke/${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
FAST_SCRATCH="/iopsstor/scratch/cscs/${USER}"
[[ -d "${FAST_SCRATCH}" ]] || FAST_SCRATCH="${SCRATCH}"
export HF_HOME="${HF_HOME:-${FAST_SCRATCH}/hf}"
export MODEL_REPO="${MODEL_REPO:-moonshotai/Kimi-K2.7-Code}"
export SERVED_MODEL="${SERVED_MODEL:-kimi-k27}"
export API_PORT="${API_PORT:-8000}"
export READY_TIMEOUT="${READY_TIMEOUT:-5400}"
export EDF="${EDF:-${HOME}/.edf/sglang-rocm-mi30x.toml}"
export MEM_FRACTION="${MEM_FRACTION:-0.93}"  # x0.85 multimodal derate -> ~0.79 effective; weights floor 0.727
export CONTEXT_LEN="${CONTEXT_LEN:-65536}"
export SWEEP_CONCURRENCY="${SWEEP_CONCURRENCY:-1 8 16}"

export MASTER="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
echo "job:    ${SLURM_JOB_ID}"
echo "model:  ${MODEL_REPO} tp=4 pp=${SLURM_NNODES} on ${SLURM_NNODES} nodes (sglang)"
echo "master: ${MASTER}"
echo "logs:   ${RUN_DIR}"

stage() { printf '\n===== stage: %s =====\n' "$*"; }

stage "serving tp=4 pp=2 across 2 nodes via sglang"
srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --exclusive --export=ALL \
    --environment="${EDF}" bash -c '
  export SGLANG_USE_AITER=1
  export SGLANG_ROCM_FUSED_DECODE_MLA=0
  # slurm cgroup rejects sglang default CPU pinning (psutil EINVAL, 604937 node1)
  export SGLANG_SET_CPU_AFFINITY=0
  export HF_HUB_OFFLINE=1
  export TRITON_CACHE_DIR='"${FAST_SCRATCH}"'/triton-cache-sglang
  # aiter builds into the read-only image by default; the mha_varlen build on the first request
  # outran the 300 s watchdog in 610164. Persist it, and the second run pays nothing.
  export AITER_JIT_DIR='"${FAST_SCRATCH}"'/aiter-jit/sglang-rocm-mi30x
  mkdir -p "${TRITON_CACHE_DIR}" "${AITER_JIT_DIR}"
  exec /opt/venv/bin/python3 -m sglang.launch_server \
    --model-path "${MODEL_REPO}" \
    --served-model-name "${SERVED_MODEL}" \
    --trust-remote-code \
    --language-only \
    --watchdog-timeout 1800 \
    --tp-size 4 --pp-size '"${SLURM_NNODES}"' \
    --nnodes '"${SLURM_NNODES}"' --node-rank "${SLURM_PROCID}" \
    --dist-init-addr "'"${MASTER}"':29765" \
    --kv-cache-dtype fp8_e4m3 \
    --page-size 64 \
    --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 \
    --mem-fraction-static "${MEM_FRACTION}" \
    --context-length "${CONTEXT_LEN}" \
    --max-running-requests 128 \
    --enable-metrics \
    --pre-warm-nccl \
    --host 0.0.0.0 --port "${API_PORT}" \
    > "${RUN_DIR}/sglang-node${SLURM_PROCID}.log" 2>&1
' &
SERVER_PID=$!

stage "waiting for the API (up to ${READY_TIMEOUT}s)"
deadline=$(( $(date +%s) + READY_TIMEOUT ))
until curl -sf "http://${MASTER}:${API_PORT}/v1/models" -o "${RUN_DIR}/models.json"; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "FAIL: server step exited before the API came up"
        tail -40 "${RUN_DIR}"/sglang-node*.log || true
        exit 1
    fi
    if (( $(date +%s) > deadline )); then
        echo "FAIL: API not ready within ${READY_TIMEOUT}s"
        tail -40 "${RUN_DIR}"/sglang-node*.log || true
        exit 1
    fi
    sleep 15
done
echo "ready: ${MASTER}"

stage "which kernels sglang actually picked"
grep -aiE "aiter|attention backend|mla|kv cache dtype|page" "${RUN_DIR}/sglang-node0.log" | head -20 || true

cat > "${RUN_DIR}/client.py" <<'PYEOF'
import json, os, sys, threading, time, urllib.request

HOST = os.environ['MASTER']
PORT = os.environ.get('API_PORT', '8000')
MODEL = os.environ.get('SERVED_MODEL', 'kimi-k27')
BASE = f'http://{HOST}:{PORT}'
FILLER = 'The kernel iterates over the array and accumulates into the result buffer. '

def post(path, body, timeout=3600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=timeout)

def completion_text(prompt, max_tokens=32):
    with post('/v1/completions', {'model': MODEL, 'prompt': prompt,
                                  'max_tokens': max_tokens, 'temperature': 0}) as r:
        return json.load(r)['choices'][0]['text']

# --- accuracy gate: long context is the only leg that discriminates (probe 604789 passed short)
# varied long context: literal repetition creates an echo attractor at temp 0 that even a
# correct backend falls into (604790 long-reason); varied text isolates real corruption
LONG = ''.join(f'Step {i}: the kernel processes block {i} and accumulates partial sums '
               f'into buffer {i % 7}. ' for i in range(650))
gate = [
    ('short-math', 'What is 17 * 23? Answer with only the number.'),
    ('long-math', LONG + '\n\nIgnore the text above. What is 17 * 23? Answer with only the number.'),
    ('long-reason', LONG + '\n\nIgnore the text above. A farmer has 12 sheep and buys 3 more, then '
                    'sells 5. How many sheep remain? Answer with only the number.'),
    ('long-code', LONG + '\n\nIgnore the text above. Write a C function that sums a float array. '
                  'Code only.'),
]
print('===== accuracy gate =====', flush=True)
for tag, prompt in gate:
    try:
        txt = completion_text(prompt, 48)
        print(f'ACC {tag}: {txt.strip()[:160]!r}', flush=True)
    except Exception as e:
        print(f'ACC {tag}: ERR {repr(e)[:120]}', flush=True)

# --- throughput: streamed completions at 10k-token prompts, per-request-distinct prefixes so
# every request pays real prefill (the campaign's cache-cold shape is what killed the vLLM arm)
def one_request(idx, arm, out, tools=False):
    prompt = f'request {idx} of arm {arm}. ' + FILLER * 700 + '\nExplain loop tiling in detail.'
    if tools:
        body = {'model': MODEL, 'max_tokens': 256, 'temperature': 0, 'stream': True,
                'ignore_eos': True,
                'messages': [{'role': 'user', 'content': prompt}],
                'tools': [{'type': 'function', 'function': {
                    'name': 'run_shell', 'description': 'Run a shell command',
                    'parameters': {'type': 'object', 'properties': {
                        'cmd': {'type': 'string'}}, 'required': ['cmd']}}}]}
        path = '/v1/chat/completions'
    else:
        body = {'model': MODEL, 'prompt': prompt, 'max_tokens': 256, 'temperature': 0,
                'stream': True, 'ignore_eos': True}
        path = '/v1/completions'
    t0 = time.time(); ttft = None; n = 0
    try:
        with post(path, body) as r:
            for line in r:
                if not line.startswith(b'data:') or line.strip() == b'data: [DONE]':
                    continue
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
        out[idx] = (ttft, time.time() - t0, n)
    except Exception as e:
        out[idx] = ('ERR', repr(e)[:80], 0)

def run_arm(name, conc, tools=False):
    out = {}
    threads = [threading.Thread(target=one_request, args=(i, f'{name}{conc}', out, tools))
               for i in range(conc)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0
    good = [v for v in out.values() if v[0] != 'ERR']
    errs = conc - len(good)
    if not good:
        print(f'{name} conc={conc} ALL {errs} FAILED: {list(out.values())[:2]}', flush=True)
        return
    ttft_max = max(v[0] for v in good)
    toks = sum(v[2] for v in good)
    decode = [(v[2] - 1) / (v[1] - v[0]) for v in good if v[1] > v[0] and v[2] > 1]
    per_req = sum(decode) / len(decode) if decode else 0.0
    print(f'{name} conc={conc} errs={errs} ttft_max={ttft_max:6.1f}s '
          f'agg={toks / wall:7.1f} tok/s per_req={per_req:5.2f} tok/s', flush=True)

print('===== sweep: streamed completions, 10k-token cache-cold prompts =====', flush=True)
for c in [int(x) for x in os.environ.get('SWEEP_CONCURRENCY', '1 8 16').split()]:
    run_arm('sglang-sweep', c)
print('===== campaign shape: chat+stream+tools =====', flush=True)
run_arm('sglang-tools', 16, tools=True)
PYEOF

stage "accuracy gate + concurrency sweep (client on ${MASTER})"
srun --overlap --nodes=1 --ntasks=1 --nodelist="${MASTER}" --export=ALL \
    --environment="${EDF}" /opt/venv/bin/python3 "${RUN_DIR}/client.py" \
    2>&1 | tee "${RUN_DIR}/client.out"

printf '\n===== SMOKE DONE (sglang tp=4 pp=%s) =====\nlogs: %s\n' "${SLURM_NNODES}" "${RUN_DIR}"
