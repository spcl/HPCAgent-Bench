#!/usr/bin/env bash
# One line per llr campaign arm: where it is, and whether its agents are actually working.
#   ./arm_status.sh              # every arm of ours in the queue
#   ./arm_status.sh 604719 ...   # named jobs, running or finished
# mcp < 1.0: the MCP-storm case that logs success but submits nothing. tok/s: per-active-request
# rate decides agent sizing, below ~2 is starved. Zeros right after engine-up are startup, not fail.
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
RUN_ROOT="${RUN_ROOT:-${SCRATCH:-/iopsstor/scratch/cscs/$USER}/hpcagent-bench-runs}"

jobs=("$@")
if (( ${#jobs[@]} == 0 )); then
    mapfile -t jobs < <(squeue -u "$USER" -h -o "%i" 2>/dev/null)
fi
(( ${#jobs[@]} )) || { echo "no jobs"; exit 0; }

printf '%-9s %-24s %-9s %5s %7s %7s %s\n' JOBID NAME STATE NODES MCP TURNS THROUGHPUT
for job in "${jobs[@]}"; do
    read -r name state nodes < <(squeue -j "$job" -h -o "%j %T %D" 2>/dev/null)
    name="${name:-?}"; state="${state:-GONE}"; nodes="${nodes:-0}"
    node_dir="${RUN_ROOT}/${job}/agents/node-0"
    started=$(ls "${node_dir}" 2>/dev/null | wc -l)
    connected=$(grep -ho '"status":"connected"' "${node_dir}"/*/claude.log 2>/dev/null | wc -l)
    turns=$(grep -ho '"type":"assistant"' "${node_dir}"/*/claude.log 2>/dev/null | wc -l)
    log="${RUN_ROOT}/slurm/beverin-services-${job}.out"
    line=$([[ -f "${log}" ]] && tr '\r' '\n' <"${log}" |
        grep -oE "aggregate throughput: t=[0-9]+s [0-9.]+ tok/s running=[0-9]+ waiting=[0-9]+" | tail -1)
    if [[ -n "${line}" ]]; then
        tps=$(grep -oE '[0-9.]+ tok/s' <<<"${line}" | cut -d' ' -f1)
        run=$(grep -oE 'running=[0-9]+' <<<"${line}" | cut -d= -f2)
        wait=$(grep -oE 'waiting=[0-9]+' <<<"${line}" | cut -d= -f2)
        through=$(awk -v t="${tps}" -v r="${run}" -v w="${wait}" \
            'BEGIN{ printf "%s tok/s over %s running (%s waiting) = %.2f/req", t, r, w, (r>0 ? t/r : 0) }')
    else
        through="no sample yet"
    fi
    printf '%-9s %-24s %-9s %5s %7s %7s %s\n' \
        "${job}" "${name:0:24}" "${state}" "${nodes}" "${connected}/${started}" "${turns}" "${through}"
done
