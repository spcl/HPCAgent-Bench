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
# PP spans the allocation, so the node count decides how much of the model each stage holds and
# therefore what --mem-fraction-static can mean. 4 is the campaign's own topology and the one
# MEM_FRACTION below is sized for; -N2 is the 2-node recipe and needs MEM_FRACTION >= 0.76.
#SBATCH --nodes=4
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
export EDF="${EDF:-${HOME}/.edf/sglang-latest.toml}"
# 0.42, not the 0.8-0.9 this knob takes anywhere else: MI300A reports is_integrated, so SGLang
# sizes the static budget against NODE-WIDE memory and every rank claims all of it. 0.93 read
# as ~0.79 after the VLM derate and the serving step was OOM-killed at 58.5 GB RSS (620806,
# exit 0:125) after 68 minutes of load and capture. 0.42 is the fraction the campaign serves.
#
# It is only meaningful WITH the node count above. The fraction applies to pre-weight-load
# node-wide free memory, so the floor is 1 - available/pre and PP decides the numerator: at pp=4
# the campaign loads 171 GB of compressed-tensors weights per stage against 412 GB pre, a floor
# of 0.415 that 0.42 just clears. At pp=2 each stage holds twice that and sglang refuses on the
# spot -- "minimum viable = 0.7525" (621070). Raise the NODES, not this number.
# 0.42 needs --cuda-graph-max-bs-decode in the launch args below to mean what it means in the
# campaign. Without it sglang captures graphs for its default decode batch and the residual comes
# out of KV: 628052 and its live-image control 628063 both died on an effective 0.357 against a
# 0.3988 floor, at the campaign's own 4 nodes and pp=4. The cap is in every kimi SGLANG_EXTRA_ARGS;
# the smoke had dropped it, so it was measuring a topology the campaign never serves.
export MEM_FRACTION="${MEM_FRACTION:-0.42}"
export CONTEXT_LEN="${CONTEXT_LEN:-65536}"
export SWEEP_CONCURRENCY="${SWEEP_CONCURRENCY:-1 8 16}"
# Trusted operator-controlled word list, the same contract as run_cluster.sh's SGLANG_EXTRA_ARGS.
# Acceptance runs pass the campaign's own knobs through here so a pass means the image serves what
# the campaign serves, not what this recipe defaults to.
export SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"
# Model-specific knobs. Defaults ARE the kimi recipe, so this file still serves kimi unchanged.
# Both parsers must be named for any model: SGLang starts fine without them and then fails at the
# FIRST request with a parser error, which reads as a serving bug rather than a launch one.
export TOOL_PARSER="${TOOL_PARSER:-kimi_k2}"
export REASONING_PARSER="${REASONING_PARSER:-kimi_k2}"
# TP is per node; PP spans the allocation. GLM-5.3 (GlmMoeDsaForCausalLM, 755.7 GB of fp8 weights)
# needs tp=4 x pp=4 over 4 nodes: at pp=2 the weights alone are 378 GB of the ~412 GB a node has
# free pre-load, a 0.917 floor that leaves no KV pool at all.
export TP_SIZE="${TP_SIZE:-4}"
# --language-only is NOT a generic "text only" switch in this SGLang: it selects the VLM
# ENCODER-DISAGGREGATION receiver role, and server_args._handle_encoder_disaggregation raises
# outright for any architecture off its allowlist. GLM-5.3 (GlmMoeDsaForCausalLM) is text-only
# to begin with, so it must pass EMPTY here -- 628586 died at launch on exactly this.
# A 0/1 knob, NOT a string default: ${VAR:-default} substitutes on EMPTY as well as unset, so
# passing LANGUAGE_ONLY_ARG="" silently reverted to --language-only and killed 628589 with the
# same encoder-disaggregation error it had already killed 628586 with. 1 for VLM-capable models
# (kimi), 0 for text-only ones (GLM-5.3).
export LANGUAGE_ONLY="${LANGUAGE_ONLY:-1}"
LANGUAGE_ONLY_ARG=""
[ "${LANGUAGE_ONLY}" = "1" ] && LANGUAGE_ONLY_ARG="--language-only"
export LANGUAGE_ONLY_ARG
# fp8_e4m3 KV is the default because it is what every arm serves, but it is NOT free: an fp8
# checkpoint ships no calibrated KV scales, so the engine quantizes at runtime against scale 1.0.
# That is exactly the kind of damage the accuracy gate below exists to catch, so the dtype is a
# knob and the fp8-vs-bf16 A/B is a rerun, not an edit.
export KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"
# The gate asks over /v1/chat/completions, so a reasoning model spends its budget thinking before
# it answers at all. 2048 covers GLM-5.3 at effort=max on the long legs; below ~512 it answers
# nothing and the gate reports a corruption that is really a truncation.
export GATE_MAX_TOKENS="${GATE_MAX_TOKENS:-2048}"
# Empty omits the field entirely, which is what kimi wants. GLM-5.3 maps every unrecognised value
# to max in its own chat template, so naming max here is a statement of intent, not a switch.
export REASONING_EFFORT="${REASONING_EFFORT:-}"
# Structured tool_calls and reasoning_content are contract, not throughput: the sweep SENDS tools
# but only counts tokens, so a server answering with prose ABOUT the call scores as healthy.
export VERIFY_SCRIPT="${VERIFY_SCRIPT:-${SLURM_SUBMIT_DIR:-.}/verify-tools-reasoning.py}"

export MASTER="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
echo "job:    ${SLURM_JOB_ID}"
echo "model:  ${MODEL_REPO} tp=4 pp=${SLURM_NNODES} on ${SLURM_NNODES} nodes (sglang)"
echo "master: ${MASTER}"
echo "image:  ${EDF}"
echo "logs:   ${RUN_DIR}"

stage() { printf '\n===== stage: %s =====\n' "$*"; }

# 192 logical CPUs per node = 4 sockets x 24 physical cores x 2 SMT threads. --exclusive gives the
# JOB the node; it does NOT give a STEP the node's CPUs, and a step without --cpus-per-task claims
# ONE. That is not a slow-down, it is a hang: 620948's serving step ran on 2 CPUs across 2 nodes,
# its Triton JIT of alloc_extend_kernel never finished, and the peer's PP SEND blocked until the
# 600 s NCCL watchdog aborted every rank. Every vLLM smoke before this served on 1 CPU too, so
# their tok/s were floor numbers. run_cluster.sh:749 carries the full account; an acceptance run
# must allocate the way the campaign does or it is not measuring the campaign.
SERVE_CPUS="${SERVE_CPUS:-${SLURM_CPUS_ON_NODE:-192}}"
# Clients and gates run --overlap ALONGSIDE the server, so they take one socket's PHYSICAL cores
# (--hint=nomultithread) rather than the node -- enough not to be the bottleneck, not so much that
# the measurement contends with what it is measuring.
CLIENT_CPUS="${CLIENT_CPUS:-24}"

stage "serving tp=${TP_SIZE} pp=${SLURM_NNODES} across ${SLURM_NNODES} nodes via sglang"
srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --exclusive --export=ALL \
    --cpus-per-task="${SERVE_CPUS}" --environment="${EDF}" bash -c '
  export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
  # FUSED DECODE MLA, off by default because the recipe it came from says so. Overridable from
  # the submitting environment so an A/B can flip exactly this one flag and nothing else: job
  # 630338 verified aiter.mla.mla_decode_fwd -- the kernel this routes decode to -- against an
  # fp32 reference (max|d| 6e-4 at kv1024, 7e-4 at kv4096), which is the prerequisite for
  # trying it here, NOT evidence that serving with it on is correct. The long-context gates
  # below are what decide that; short-context accuracy passes on broken kernels.
  export SGLANG_ROCM_FUSED_DECODE_MLA="${SGLANG_ROCM_FUSED_DECODE_MLA:-0}"
  # slurm cgroup rejects sglang default CPU pinning (psutil EINVAL, 604937 node1)
  export SGLANG_SET_CPU_AFFINITY=0
  export HF_HUB_OFFLINE=1
  export TRITON_CACHE_DIR='"${FAST_SCRATCH}"'/triton-cache-sglang
  # aiter builds into the read-only image by default; the mha_varlen build on the first request
  # outran the 300 s watchdog in 610164. Persist it, and the second run pays nothing.
  # An image that ships a prebuilt /opt/aiter-jit wins; older ones still fall back to
  # persistent scratch, because aiter JIT-building into a read-only image is what blew the
  # 300 s watchdog in 610164.
  export AITER_JIT_DIR="${AITER_JIT_DIR:-'"${FAST_SCRATCH}"'/aiter-jit/sglang-rocm-mi30x}"
  mkdir -p "${TRITON_CACHE_DIR}" "${AITER_JIT_DIR}"
  exec /opt/venv/bin/python3 -m sglang.launch_server \
    --model-path "${MODEL_REPO}" \
    --served-model-name "${SERVED_MODEL}" \
    --trust-remote-code \
    ${LANGUAGE_ONLY_ARG} \
    --watchdog-timeout 1800 \
    --tp-size '"${TP_SIZE}"' --pp-size '"${SLURM_NNODES}"' \
    --nnodes '"${SLURM_NNODES}"' --node-rank "${SLURM_PROCID}" \
    --dist-init-addr "'"${MASTER}"':29765" \
    --kv-cache-dtype '"${KV_DTYPE}"' \
    --page-size 64 \
    --tool-call-parser '"${TOOL_PARSER}"' --reasoning-parser '"${REASONING_PARSER}"' \
    --mem-fraction-static "${MEM_FRACTION}" \
    --context-length "${CONTEXT_LEN}" \
    --max-running-requests 128 \
    --cuda-graph-max-bs-decode 64 \
    --enable-metrics \
    --pre-warm-nccl \
    ${SGLANG_EXTRA_ARGS} \
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

stage "tool-call + reasoning contract"
if [[ -r "${VERIFY_SCRIPT}" ]]; then
    cp "${VERIFY_SCRIPT}" "${RUN_DIR}/verify-tools-reasoning.py"
    srun --overlap --nodes=1 --ntasks=1 --nodelist="${MASTER}" --export=ALL \
        --cpus-per-task=4 --hint=nomultithread --environment="${EDF}" \
        /opt/venv/bin/python3 "${RUN_DIR}/verify-tools-reasoning.py" \
        --base "http://${MASTER}:${API_PORT}" --model "${SERVED_MODEL}" \
        --reasoning-effort "${REASONING_EFFORT}" \
        2>&1 | tee "${RUN_DIR}/verify-tools-reasoning.out"
else
    echo "SKIP: no verifier at ${VERIFY_SCRIPT}"
fi

stage "which kernels sglang actually picked"
grep -aiE "aiter|attention backend|mla|kv cache dtype|page" "${RUN_DIR}/sglang-node0.log" | head -20 || true

cat > "${RUN_DIR}/client.py" <<'PYEOF'
import json, os, re, sys, threading, time, urllib.request

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

GATE_MAX_TOKENS = int(os.environ.get('GATE_MAX_TOKENS', '2048'))
REASONING_EFFORT = os.environ.get('REASONING_EFFORT', '')

# The gate has to ask the way the campaign asks. /v1/completions bypasses the chat template, so a
# reasoning model never gets its thinking tags, and a 48-token budget then truncates mid-thought:
# every ACC line GLM-5.3 produced (628603/628619) was a fragment of deliberation being scored as
# an answer. Chat plus a real budget is the only shape that tells a corrupt kernel apart from a
# truncated one.
def chat_answer(prompt, max_tokens):
    body = {'model': MODEL, 'temperature': 0, 'max_tokens': max_tokens,
            'messages': [{'role': 'user', 'content': prompt}]}
    if REASONING_EFFORT:
        body['reasoning_effort'] = REASONING_EFFORT
    with post('/v1/chat/completions', body) as r:
        choice = json.load(r)['choices'][0]
    msg = choice['message']
    return (msg.get('content') or '', len(msg.get('reasoning_content') or ''),
            choice.get('finish_reason') or '?')

def last_int(text):
    nums = re.findall(r'-?\d+', text)
    return nums[-1] if nums else None

# --- accuracy gate: long context is the only leg that discriminates (probe 604789 passed short)
# varied long context: literal repetition creates an echo attractor at temp 0 that even a
# correct backend falls into (604790 long-reason); varied text isolates real corruption
LONG = ''.join(f'Step {i}: the kernel processes block {i} and accumulates partial sums '
               f'into buffer {i % 7}. ' for i in range(650))
# Third field is the expected answer. A gate that only PRINTS is not a gate: 628603 served four
# truncated fragments and still exited 0, so the wrong answer has to fail the job.
gate = [
    ('short-math', 'What is 17 * 23? Answer with only the number.', '391'),
    ('long-math', LONG + '\n\nIgnore the text above. What is 17 * 23? Answer with only the number.',
     '391'),
    ('long-reason', LONG + '\n\nIgnore the text above. A farmer has 12 sheep and buys 3 more, then '
                    'sells 5. How many sheep remain? Answer with only the number.', '10'),
    ('long-code', LONG + '\n\nIgnore the text above. Write a C function that sums a float array. '
                  'Code only.', None),
]
# A request that raised is the engine failing, not a datum. 620806 OOM-killed its server mid-sweep
# and this script still exited 0, so slurm recorded the job COMPLETED while nothing was served.
FAILURES = []

print('===== accuracy gate =====', flush=True)
for tag, prompt, expect in gate:
    try:
        txt, think, finish = chat_answer(prompt, GATE_MAX_TOKENS)
    except Exception as e:
        print(f'ACC {tag}: ERR {repr(e)[:120]}', flush=True)
        FAILURES.append(f'accuracy gate {tag} raised')
        continue
    if expect is None:
        ok = bool(txt.strip())
        why = '' if ok else 'empty answer'
    else:
        got = last_int(txt)
        ok = got == expect
        why = '' if ok else f'expected {expect}, got {got}'
    # finish_reason=length means the budget ran out, which reads as a wrong answer but is not a
    # kernel fault -- name it so a raised GATE_MAX_TOKENS is the obvious next move.
    if not ok and finish == 'length':
        why += ' (truncated: raise GATE_MAX_TOKENS)'
    print(f'ACC {tag}: {"ok" if ok else "WRONG"} think={think}c finish={finish} '
          f'answer={txt.strip()[:160]!r}', flush=True)
    if not ok:
        FAILURES.append(f'accuracy gate {tag}: {why}')

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
        FAILURES.append(f'{name} conc={conc}: all {errs} requests failed')
        return
    if errs:
        FAILURES.append(f'{name} conc={conc}: {errs} of {conc} requests failed')
    ttft_max = max(v[0] for v in good)
    toks = sum(v[2] for v in good)
    decode = [(v[2] - 1) / (v[1] - v[0]) for v in good if v[1] > v[0] and v[2] > 1]
    per_req = sum(decode) / len(decode) if decode else 0.0
    print(f'{name} conc={conc} errs={errs} ttft_max={ttft_max:6.1f}s '
          f'agg={toks / wall:7.1f} tok/s per_req={per_req:5.2f} tok/s', flush=True)

# TOOLS ON EVERY LEG. The campaign never sends a bare completion: every agent request carries a
# tool schema and goes through /v1/chat/completions, which is a different server path (chat
# template, tool-call parser, structured-output bookkeeping) from /v1/completions. Measuring
# three of four legs without tools measured a shape nothing runs, so the sweep now uses the
# campaign shape at every concurrency. SWEEP_TOOLS=0 restores the bare-completion legs for a
# deliberate A/B of the tool path's own cost -- it is not the default.
sweep_tools = os.environ.get('SWEEP_TOOLS', '1') != '0'
print(f'===== sweep: streamed chat, 10k-token cache-cold prompts, tools={sweep_tools} =====', flush=True)
for c in [int(x) for x in os.environ.get('SWEEP_CONCURRENCY', '1 8 16').split()]:
    run_arm('sglang-sweep', c, tools=sweep_tools)
# Same shape and concurrency as the last sweep leg, deliberately: a SECOND measurement at conc 16
# inside one run, which is the only within-run estimate of variance this harness produces.
print('===== campaign shape: chat+stream+tools (repeat at conc 16) =====', flush=True)
run_arm('sglang-tools', 16, tools=True)

if FAILURES:
    print('===== SMOKE FAILED =====', flush=True)
    for line in FAILURES:
        print('  ' + line, flush=True)
    sys.exit(1)
PYEOF

stage "accuracy gate + concurrency sweep (client on ${MASTER})"
srun --overlap --nodes=1 --ntasks=1 --nodelist="${MASTER}" --export=ALL \
    --cpus-per-task="${CLIENT_CPUS}" --hint=nomultithread \
    --environment="${EDF}" /opt/venv/bin/python3 "${RUN_DIR}/client.py" \
    2>&1 | tee "${RUN_DIR}/client.out"

printf '\n===== SMOKE DONE (sglang tp=4 pp=%s) =====\nlogs: %s\n' "${SLURM_NNODES}" "${RUN_DIR}"
