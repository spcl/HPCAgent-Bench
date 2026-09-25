#!/usr/bin/env bash
# Does Kimi K2.7 at tp=4 pp=4 actually DECODE, with eager process-group init?
#
# The failure mode: a live API that produces nothing -- engine up, /v1/models answering, requests
# accepted, GPUs pegged, generation_tokens_total frozen. On PP ranks only, torch logs an unbatched
# P2P op minting a fresh lazy-init NCCL communicator; gpt-oss (pp=1) never does this, so the split
# is pipeline parallelism exhausting a CXI resource one 2-rank communicator at a time.
# external-eager-pg-patch/sitecustomize.py injects device_id= into init_process_group/new_group so
# the PGs initialise EAGERLY instead ("eager" here is eager PG INIT, not --enforce-eager).
#
# A wedged engine still answers the API, so this gates on DECODE PROGRESS and nothing else, then
# measures the saturated ceiling.
#
#   cd containers/cluster/ce-images/inference
#   mkdir -p logs && sbatch smoke-kimi-eager-pg.sbatch
#
#SBATCH --job-name=smoke-kimi-eager-pg
#SBATCH --partition=mi300
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --no-requeue

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:?set SCRATCH}"
export RUN_DIR="${SCRATCH}/kimi-smoke/${SLURM_JOB_ID}"
# Under sbatch BASH_SOURCE is the spooled copy; the checkout is where the job was submitted from.
REPO="${REPO:-$(git -C "${SLURM_SUBMIT_DIR:-${PWD}}" rev-parse --show-toplevel 2>/dev/null || true)}"
[[ -n "${REPO}" && -f "${REPO}/scripts/cache_env.sh" ]] || { echo "FAIL: no checkout: set REPO or submit from inside one" >&2; exit 1; }
. "${REPO}/scripts/cache_env.sh"
# Unset, this defaults under ~/.cache, which is unmounted here, so every smoke would re-run
# Inductor codegen from scratch before the API could bind.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${FAST_SCRATCH}/vllm-cache}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
# Tuned fused_moe Triton configs, the folder run_cluster.sh points the campaign at. Unset, vLLM
# falls back to its default MoE config for this shape, which is much slower and not a topology
# result. Set TUNED_MOE_DIR= to run untuned deliberately.
export TUNED_MOE_DIR="${TUNED_MOE_DIR-${SUBMIT_DIR}/moe-configs}"
# Slurm spools this script to /var/spool/slurmd, so BASH_SOURCE cannot locate the patch and
# SLURM_SUBMIT_DIR is all there is. The job may be submitted from the repo root or from here, so
# try both the sibling-directory and repo-root spellings before giving up.
PG_PATCH_REL="containers/cluster/ce-images/inference/external-eager-pg-patch"
for cand in "${PG_PATCH_DIR:-}" "${SUBMIT_DIR}/external-eager-pg-patch" "${SUBMIT_DIR}/${PG_PATCH_REL}"; do
    if [[ -n "${cand}" && -f "${cand}/sitecustomize.py" ]]; then
        export PG_PATCH_DIR="${cand}"
        break
    fi
done

# Pinned below 0.27.1, which decodes noticeably slower on this shape.
INFERENCE_EDF="${INFERENCE_EDF:-hpcagent-bench-vllm-mi300-latest}"
export MODEL_REPO="${MODEL_REPO:-moonshotai/Kimi-K2.7-Code}"
export SERVED_MODEL="${SERVED_MODEL:-hpcagent-bench-vllm}"
export API_PORT="${API_PORT:-8000}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
READY_TIMEOUT="${READY_TIMEOUT:-5400}"
# How long the workers may go completely silent before we call it wedged. A healthy start is
# SLOW but never INDEFINITELY quiet; a wedged one stops all ranks at the same NCCL bootstrap line
# and writes nothing. Silence separates the two cases; elapsed time does not.
STALL_LIMIT="${STALL_LIMIT:-2700}"
PROBE_STREAMS="${PROBE_STREAMS:-64}"
PROBE_MAX_TOKENS="${PROBE_MAX_TOKENS:-512}"
# Gate 6, agent-shaped load, is not part of accepting an image, so it is OFF unless a run asks
# for it.
AGENTLIKE_DURATION_S="${AGENTLIKE_DURATION_S:-0}"
AGENTLIKE_STREAMS="${AGENTLIKE_STREAMS:-64}"
AGENTLIKE_SHARED_WORDS="${AGENTLIKE_SHARED_WORDS:-2000}"

test -f "${PG_PATCH_DIR:-}/sitecustomize.py" 2>/dev/null || {
    echo "FAIL: no sitecustomize.py found from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
    echo "      set PG_PATCH_DIR=<dir holding sitecustomize.py> and resubmit" >&2
    exit 2
}
mkdir -p "${RUN_DIR}"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export NODE0="${NODES[0]}"
# Topology follows the ALLOCATION, so the same script also serves a pp=1 control on one node
# (sbatch --nodes=1 MODEL_REPO=openai/gpt-oss-120b), which tells us whether a low tok/s number is
# the stack or just the offered load.
export NNODES="${#NODES[@]}"
export TP_SIZE="${TP_SIZE:-4}"
export PP_SIZE="${PP_SIZE:-${NNODES}}"
# Parsers are MODEL-specific (gpt-oss has no kimi think tokens; the API server dies on them).
# Defaults fit kimi; the control passes its own (e.g. TOOL_PARSER=openai).
export TOOL_PARSER="${TOOL_PARSER:-kimi_k2}"
export REASONING_PARSER="${REASONING_PARSER:-kimi_k2}"
export EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"
NODELIST="$(IFS=,; echo "${NODES[*]}")"

SERVE_SCRIPT="${RUN_DIR}/serve.sh"
SMOKE_DONE="${RUN_DIR}/smoke.done"
stage() { printf '\n===== stage: %s =====\n' "$*"; }

cat <<EOF
job:    ${SLURM_JOB_ID}
model:  ${MODEL_REPO} tp=${TP_SIZE} pp=${PP_SIZE} on ${NNODES} node(s): ${NODELIST}
image:  ${INFERENCE_EDF}
patch:  ${PG_PATCH_DIR}
logs:   ${RUN_DIR}
moe:    ${TUNED_MOE_DIR:-<none: vLLM default MoE config>}
extra:  ${EXTRA_SERVE_ARGS:-<none>}
EOF

cat >"${SERVE_SCRIPT}" <<'SERVE'
#!/usr/bin/env bash
set -Eeuo pipefail
rank="${SLURM_PROCID:-0}"

export TOKENIZERS_PARALLELISM=false
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
# Overridable so a run can tell a DEADLOCK from mere slowness. 602296 hung in a gloo
# recv_object on the pp pair; gloo's pair timeout and this RPC timeout are both 1800 s, so
# one 30-minute stall trips both at once and we cannot see whether the engine would have
# recovered. Raise both and the answer is direct: recovery means backpressure, continued
# silence means a true deadlock.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
if [[ -n "${TUNED_MOE_DIR:-}" ]]; then export VLLM_TUNED_CONFIG_FOLDER="${TUNED_MOE_DIR}"; fi
export NCCL_DEBUG_FILE="${RUN_DIR}/nccl.%h.%p.log"
# THE POINT OF THIS JOB. sitecustomize is imported automatically by the interpreter at startup
# for every entry on sys.path, so putting the patch dir on PYTHONPATH reaches every vLLM worker
# process without touching the image or the launcher.
export PYTHONPATH="${PG_PATCH_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# NOT set: VLLM_DISABLE_PYNCCL. It is the superseded workaround that costs ~20x and blocks
# graph capture; run_cluster.sh deliberately stopped defaulting it on 2026-08-13.
# vLLM reads env VLLM_PORT as the BASE for internal ZMQ ports, not the HTTP port (589170).
unset VLLM_PORT

MODEL_PATH="$(python3 - <<'PY' | tail -n 1
import os
from huggingface_hub import snapshot_download
repo = os.environ["MODEL_REPO"]
try:
    print(snapshot_download(repo_id=repo, local_files_only=True))
except Exception:
    print(snapshot_download(repo_id=repo, max_workers=8))
PY
)"
test -d "${MODEL_PATH}"
echo "rank ${rank} host $(hostname) model ${MODEL_PATH} PYTHONPATH=${PYTHONPATH}"

ARGS=(
    "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL}"
    --tensor-parallel-size "${TP_SIZE}"
    --pipeline-parallel-size "${PP_SIZE}"
    --distributed-executor-backend mp
    --nnodes "${NNODES}"
    --node-rank "${rank}"
    --master-addr "${NODE0}"
    --master-port "${MASTER_PORT}"
    --distributed-timeout-seconds 7200
    --load-format safetensors
    --safetensors-load-strategy prefetch
    --generation-config auto
    --enable-auto-tool-choice
    --tool-call-parser "${TOOL_PARSER}"
    --trust-remote-code
    --max-model-len 65536
    # 0.70 not 0.85: the fraction multiplies TOTAL board memory but only 96 of the MI300A's
    # 128 GiB is HIP-allocatable.
    --gpu-memory-utilization 0.70
    --max-num-seqs 128
)
# Optional: Qwen3-Coder-Instruct has no reasoning parser and the campaign env files pass none.
if [[ -n "${REASONING_PARSER}" ]]; then
    ARGS+=(--reasoning-parser "${REASONING_PARSER}")
fi
if [[ -n "${EXTRA_SERVE_ARGS}" ]]; then
    # shellcheck disable=SC2206 -- word-splitting is the contract, same as VLLM_EXTRA_ARGS
    ARGS+=(${EXTRA_SERVE_ARGS})
fi
if (( rank == 0 )); then
    ARGS+=(--host 0.0.0.0 --port "${API_PORT}")
else
    ARGS+=(--headless)
fi

SERVER_PID=""
cleanup() { [[ -n "${SERVER_PID}" ]] && kill "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 143' TERM

vllm serve "${ARGS[@]}" >"${RUN_DIR}/vllm-rank${rank}.log" 2>&1 &
SERVER_PID=$!
echo "rank ${rank} vllm pid ${SERVER_PID}"

while [[ ! -e "${RUN_DIR}/smoke.done" ]]; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "rank ${rank}: vLLM exited before the smoke finished" >&2
        wait "${SERVER_PID}" || true
        exit 1
    fi
    sleep 5
done
SERVE
chmod +x "${SERVE_SCRIPT}"
rm -f "${SMOKE_DONE}"

# --exclusive gives the JOB the node, not a STEP its CPUs: a step without --cpus-per-task claims
# ONE, and a serving step on too few CPUs hangs (Triton JIT never finishes).
SERVE_CPUS="${SERVE_CPUS:-${SLURM_CPUS_ON_NODE:-192}}"
# Clients and gates run --overlap ALONGSIDE the server, so they take one socket's PHYSICAL cores
# (--hint=nomultithread) rather than the node, so the measurement does not contend with itself.
CLIENT_CPUS="${CLIENT_CPUS:-24}"

stage "serving ${MODEL_REPO} tp=${TP_SIZE} pp=${PP_SIZE} on ${NNODES} node(s) with the eager-PG patch"
srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 --exclusive --export=ALL \
    --nodelist="${NODELIST}" --environment="${INFERENCE_EDF}" --cpus-per-task="${SERVE_CPUS}" \
    bash "${SERVE_SCRIPT}" >"${RUN_DIR}/vllm-step.log" 2>&1 &
SERVER_STEP_PID=$!
cleanup_steps() {
    touch "${SMOKE_DONE}"
    kill -0 "${SERVER_STEP_PID}" 2>/dev/null && { sleep 10; kill "${SERVER_STEP_PID}" 2>/dev/null || true; }
    wait 2>/dev/null || true
}
trap cleanup_steps EXIT INT TERM

# The || true matters twice under set -eo pipefail: a dead server must read as an empty
# metric the gates can act on, not a silent script abort.
metric() { { curl -fsS --max-time 10 "http://${NODE0}:${API_PORT}/metrics" 2>/dev/null || true; } |
    awk -v k="vllm:$1" '$0 ~ "^"k {print $2; exit}'; }

newest_log_write() {
    local newest=0 mtime
    for log in "${RUN_DIR}"/vllm-rank*.log "${RUN_DIR}"/nccl.*.log; do
        [[ -f "${log}" ]] || continue
        mtime=$(stat -c %Y "${log}")
        (( mtime > newest )) && newest=${mtime}
    done
    echo "${newest}"
}

report_stall() {
    echo "--- rank progress at the stall ---" >&2
    local done_count
    done_count=$({ grep -lh "Init COMPLETE" "${RUN_DIR}"/nccl.*.log 2>/dev/null || true; } | wc -l)
    echo "nccl logs reaching Init COMPLETE: ${done_count}" >&2
    for log in "${RUN_DIR}"/vllm-rank*.log; do
        [[ -f "${log}" ]] || continue
        echo "$(basename "${log}"): $(tail -1 "${log}" | cut -c1-140)" >&2
    done
}

stage "waiting for the API (up to ${READY_TIMEOUT}s, wedged if silent for ${STALL_LIMIT}s)"
deadline=$((SECONDS + READY_TIMEOUT))
start_epoch=$(date +%s)
until curl -fsS --max-time 10 "http://${NODE0}:${API_PORT}/v1/models" -o "${RUN_DIR}/models.json"; do
    kill -0 "${SERVER_STEP_PID}" 2>/dev/null || { echo "FAIL: vLLM died before the API came up" >&2; exit 1; }
    (( SECONDS < deadline )) || { echo "FAIL: API not ready in ${READY_TIMEOUT}s" >&2; exit 1; }
    newest=$(newest_log_write)
    # Before any log exists there is nothing to have gone quiet, so measure from the loop instead.
    (( newest > 0 )) || newest=$((start_epoch))
    quiet=$(( $(date +%s) - newest ))
    if (( quiet >= STALL_LIMIT )); then
        echo "FAIL: startup wedged -- no worker wrote a log line in ${quiet}s" >&2
        report_stall
        exit 1
    fi
    sleep 15
done

stage "gate 1: did the eager-PG patch load?"
if grep -qh "\[external-eager-pg\]" "${RUN_DIR}"/vllm-rank*.log 2>/dev/null; then
    grep -ho "\[external-eager-pg\] [a-z_]*" "${RUN_DIR}"/vllm-rank*.log | sort | uniq -c
else
    echo "FAIL: no [external-eager-pg] lines -- sitecustomize never loaded" >&2
    exit 1
fi

stage "gate 2: unbatched-P2P warnings (594541 had 24; expect 0)"
# Zero matches is the PASS here, and grep exits nonzero on it -- which pipefail propagates
# through wc -l, so the bare pipeline kills the job at the moment it succeeds.
# The || true must sit on grep itself, inside the pipe.
p2p=$({ grep -h "unbatched P2P op" "${RUN_DIR}"/vllm-rank*.log 2>/dev/null || true; } | wc -l)
echo "unbatched P2P warnings: ${p2p}"

stage "gate 3: DOES IT DECODE -- the gate every previous smoke skipped"
# aiter JIT-compiles kernels on FIRST use, so a cold cache can blow a 300 s budget and read as
# "wedged" for a server that is merely compiling. Raise it whenever AITER_JIT_DIR starts empty.
before=$(metric generation_tokens_total)
curl -fsS --max-time "${FIRST_TOKEN_TIMEOUT:-300}" "http://${NODE0}:${API_PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED_MODEL}\",\"prompt\":\"Write one line of C.\",\"max_tokens\":64}" \
    -o "${RUN_DIR}/single.json" || true
after=$(metric generation_tokens_total)
echo "generation_tokens_total: ${before:-0} -> ${after:-0}"
awk -v a="${before:-0}" -v b="${after:-0}" 'BEGIN{ if (b-a <= 0) exit 1 }' || {
    echo "FAIL: zero tokens decoded -- still wedged" >&2
    exit 1
}

stage "gate 4: saturation probe, ${PROBE_STREAMS} concurrent x ${PROBE_MAX_TOKENS} tokens"
# max_tokens is a floor on how long each stream lives, and ignore_eos keeps it from finishing
# early -- otherwise short answers drain the batch and the "saturated" number is measured against
# a half-empty engine. Bracket the burst instead of timing a fixed window after a fill sleep: a
# burst can finish INSIDE the fill sleep, and a fixed window would then measure an idle engine.
g0=$(metric generation_tokens_total); t0=$SECONDS
PROBE_PIDS=()
for i in $(seq 1 "${PROBE_STREAMS}"); do
    curl -fsS --max-time 900 "http://${NODE0}:${API_PORT}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${SERVED_MODEL}\",\"prompt\":\"Explain loop tiling, variation ${i}.\",\"max_tokens\":${PROBE_MAX_TOKENS},\"ignore_eos\":true}" \
        -o "${RUN_DIR}/probe-${i}.json" &
    PROBE_PIDS+=($!)
done
: >"${RUN_DIR}/probe-conc.txt"
# Three columns per sample -- elapsed, cumulative generated tokens, resident requests. The token
# column is what makes a steady-state read possible: the aggregate below divides by the WHOLE
# burst, ramp included, so the headline number can read well under the engine's saturated rate.
( while :; do printf '%s %s %s\n' "${SECONDS}" "$(metric generation_tokens_total)" \
    "$(metric num_requests_running)" >>"${RUN_DIR}/probe-conc.txt"; sleep 5; done ) &
SAMPLER_PID=$!
wait "${PROBE_PIDS[@]}" || true
g1=$(metric generation_tokens_total); t1=$SECONDS
kill "${SAMPLER_PID}" 2>/dev/null || true
awk -v a="${g0:-0}" -v b="${g1:-0}" -v dt="$((t1-t0))" -v span="${STEADY_SPAN_S:-60}" '
    # A failed /metrics scrape contributes an EMPTY field, and an empty field is not a zero
    # measurement: carried as one it would make the following window look like a burst and hand
    # the steady line a rate the engine never reached. Keep only fully numeric samples.
    NF == 3 && $2 ~ /^[0-9]+([.][0-9]+)?$/ && $3 ~ /^[0-9]+([.][0-9]+)?$/ {
        k += 1; et[k] = $1; tok[k] = $2; run[k] = $3
        if ($3 > 0) { s += $3; n += 1; if ($3 > peak) peak = $3 }
    }
    END {
        conc = n ? s / n : 0
        rate = dt > 0 ? (b - a) / dt : 0
        printf "SATURATED DECODE: %.1f tok/s aggregate over %ds (avg %.0f / peak %.0f concurrent)\n", rate, dt, conc, peak
        if (conc > 0) printf "                  %.1f tok/s per request\n", rate / conc
        # Best sustained rate over any window of at least `span` seconds -- the same quantity the
        # serving probe reports after excluding its ramp, and the only one comparable to it.
        for (i = 1; i <= k; i++)
            for (j = i + 1; j <= k; j++)
                if (et[j] - et[i] >= span) {
                    r = (tok[j] - tok[i]) / (et[j] - et[i])
                    if (r > best) { best = r; bs = et[j] - et[i] }
                    break
                }
        if (best > 0) {
            printf "STEADY DECODE:    %.1f tok/s best sustained over %ds", best, bs
            if (conc > 0) printf " (%.1f per request)", best / conc
            printf "\n"
        }
        if (rate <= 0) { print "FAIL: no decode progress under load"; exit 1 }
    }' "${RUN_DIR}/probe-conc.txt" || exit 1

stage "gate 5: long-context accuracy -- throughput alone never said whether a change was SAFE"
# A backend can serve a high tok/s while silently corrupting output (echoing filler, or worse);
# grading decode progress alone would promote that backend on its rate.
srun --nodes=1 --ntasks=1 --overlap --nodelist="${NODE0}" --environment="${INFERENCE_EDF}" \
    --cpus-per-task="${CLIENT_CPUS}" --hint=nomultithread \
    /opt/pytorch211/bin/python "$(dirname "${PG_PATCH_DIR}")/accuracy-gate.py" \
        --base "http://${NODE0}:${API_PORT}" --model "${SERVED_MODEL}" \
    || { echo "FAIL: long-context accuracy gate" >&2; touch "${SMOKE_DONE}"; exit 1; }

if (( AGENTLIKE_DURATION_S > 0 )); then
    stage "gate 6: agent-shaped load, ${AGENTLIKE_STREAMS} streams x long shared prefix for ${AGENTLIKE_DURATION_S}s"
    # Probe resolved beside the patch dir, not the SUBMIT dir, so a driver submitting from
    # elsewhere still finds it. Keep this comment here: one between the line continuations below
    # would end the command early and run the probe on the batch node instead.
    srun --nodes=1 --ntasks=1 --overlap --nodelist="${NODE0}" --environment="${INFERENCE_EDF}" \
        --cpus-per-task="${CLIENT_CPUS}" --hint=nomultithread \
        /opt/pytorch211/bin/python "$(dirname "${PG_PATCH_DIR}")/agentlike-probe.py" \
            --base "http://${NODE0}:${API_PORT}" --model "${SERVED_MODEL}" \
            --streams "${AGENTLIKE_STREAMS}" --shared-words "${AGENTLIKE_SHARED_WORDS}" \
            --max-tokens "${PROBE_MAX_TOKENS}" --duration "${AGENTLIKE_DURATION_S}" \
        || { echo "FAIL: agent-shaped load did not survive" >&2; touch "${SMOKE_DONE}"; exit 1; }
fi

touch "${SMOKE_DONE}"
printf '\n===== KIMI EAGER-PG SMOKE PASSED =====\nlogs: %s\n' "${RUN_DIR}"
