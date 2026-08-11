#!/usr/bin/env bash
# Samples cpu/mem/gpu utilization on this node every INTERVAL seconds and appends
# to a per-node CSV, so a benchmark run can be checked for workload balance
# (idle judges, saturated inference, starved agents) after the fact. gpu_pct is the
# cross-GPU average; gpu0_pct..gpu(N-1)_pct (appended after the fixed 8 columns)
# give per-GPU utilization, for spotting judge-to-GPU imbalance.
#
# env: ROLE (vllm|judge|agent, required), OUT_DIR (default "${RUN_DIR:-.}/monitor"),
#      INTERVAL (default 5, seconds)
set -uo pipefail
# no set -e: a bad read must produce a blank field, never kill the loop

ROLE="${ROLE:-${1:-}}"
OUT_DIR="${OUT_DIR:-${RUN_DIR:-.}/monitor}"
INTERVAL="${INTERVAL:-5}"

if [ -z "${ROLE}" ]; then
    echo "node_monitor.sh: ROLE env var (vllm|judge|agent) required" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
CSV="${OUT_DIR}/${ROLE}-$(hostname).csv"

STOP=0
trap 'STOP=1' TERM INT

# Decide once which GPU tool works and how many GPUs it reports; per-sample code
# below just calls that one tool, so a missing/broken rocm-smi never costs a second
# failing fork every 5s, and the header (below) stays fixed for the life of the CSV.
GPU_TOOL=none
NGPU=0
if command -v rocm-smi >/dev/null 2>&1; then
    probe=$(rocm-smi --showuse --showmeminfo vram --json 2>/dev/null)
    if [ $? -eq 0 ]; then
        GPU_TOOL=rocm
        NGPU=$(printf '%s' "${probe}" | grep -oE '"GPU use \(%\)"' | wc -l | tr -d ' ')
    fi
fi
if [ "${GPU_TOOL}" = none ] && command -v amd-smi >/dev/null 2>&1; then
    probe=$(amd-smi metric -u --json 2>/dev/null | tr -s '[:space:]' ' ')
    if [ $? -eq 0 ]; then
        GPU_TOOL=amdsmi
        NGPU=$(printf '%s' "${probe}" | grep -oE '"gfx_activity"' | wc -l | tr -d ' ')
    fi
fi

# Per-GPU columns are appended after the fixed 8, so old consumers keep working.
HEADER="ts,cpu_pct,load1,mem_used_mib,mem_total_mib,gpu_pct,vram_used_mib,vram_total_mib"
i=0
while [ "${i}" -lt "${NGPU}" ]; do
    HEADER="${HEADER},gpu${i}_pct"
    i=$((i + 1))
done
[ -s "${CSV}" ] || echo "${HEADER}" > "${CSV}"

cpu_fields() {
    # prints the numeric fields of the "cpu " line from /proc/stat, space separated
    awk '/^cpu / {for (i = 2; i <= NF; i++) printf "%s ", $i; print ""; exit}' \
        /proc/stat 2>/dev/null
}

per_gpu_list() {
    # reads one per-GPU value per stdin line, echoes exactly NGPU comma-separated
    # fields (blank-padded if the smi output had fewer entries than expected) -- the
    # order is the smi tool's own card/gpu index order, which is the assumption the
    # gpu0_pct.. header columns rely on
    awk -v n="${NGPU}" '{
        v[NR] = $1
    } END {
        for (i = 1; i <= n; i++) {
            if (i > 1) printf ","
            if (i in v) printf "%s", v[i]
        }
    }'
}

gpu_row() {
    # args: util vu vt per_gpu_csv; appends the per-GPU columns only when NGPU > 0
    if [ "${NGPU}" -gt 0 ]; then
        echo "${1},${2},${3},${4}"
    else
        echo "${1},${2},${3}"
    fi
}

sample_gpu() {
    # echoes "gpu_pct,vram_used_mib,vram_total_mib[,gpu0_pct,...]"; blank fields on
    # any failure
    local json util vu vt per blank
    blank=$(per_gpu_list < /dev/null)
    case "${GPU_TOOL}" in
        rocm)
            json=$(rocm-smi --showuse --showmeminfo vram --json 2>/dev/null) || { gpu_row "" "" "" "${blank}"; return; }
            util=$(printf '%s' "${json}" | grep -oE '"GPU use \(%\)": *"?[0-9.]+' \
                | grep -oE '[0-9.]+' | awk '{s += $1; n++} END {if (n > 0) printf "%.1f", s / n}')
            vu=$(printf '%s' "${json}" | grep -oE '"VRAM Total Used Memory \(B\)": *"?[0-9]+' \
                | grep -oE '[0-9]+' | awk '{s += $1} END {if (s > 0) printf "%d", s / 1048576}')
            vt=$(printf '%s' "${json}" | grep -oE '"VRAM Total Memory \(B\)": *"?[0-9]+' \
                | grep -oE '[0-9]+' | awk '{s += $1} END {if (s > 0) printf "%d", s / 1048576}')
            per=$(printf '%s' "${json}" | grep -oE '"GPU use \(%\)": *"?[0-9.]+' \
                | grep -oE '[0-9.]+' | per_gpu_list)
            gpu_row "${util}" "${vu}" "${vt}" "${per}"
            ;;
        amdsmi)
            # metric -u is usage-only: no vram fields come back from this call
            json=$(amd-smi metric -u --json 2>/dev/null | tr -s '[:space:]' ' ') || { gpu_row "" "" "" "${blank}"; return; }
            util=$(printf '%s' "${json}" | grep -oE '"gfx_activity": *\{ *"value": *[0-9.]+' \
                | grep -oE '[0-9.]+' | awk '{s += $1; n++} END {if (n > 0) printf "%.1f", s / n}')
            per=$(printf '%s' "${json}" | grep -oE '"gfx_activity": *\{ *"value": *[0-9.]+' \
                | grep -oE '[0-9.]+' | per_gpu_list)
            gpu_row "${util}" "" "" "${per}"
            ;;
        *)
            gpu_row "" "" "" "${blank}"
            ;;
    esac
}

GPU_BLANK_LINE=$(gpu_row "" "" "" "$(per_gpu_list < /dev/null)")

prev=$(cpu_fields)
while [ "${STOP}" -eq 0 ]; do
    sleep "${INTERVAL}" & wait $!
    [ "${STOP}" -eq 1 ] && break

    curr=$(cpu_fields)
    cpu_pct=$(awk -v p="${prev}" -v c="${curr}" 'BEGIN {
        n = split(p, pf, " "); split(c, cf, " ")
        if (n < 4) { print ""; exit }
        pidle = pf[4] + pf[5]; cidle = cf[4] + cf[5]
        ptot = 0; ctot = 0
        for (i = 1; i <= n; i++) { ptot += pf[i]; ctot += cf[i] }
        dtot = ctot - ptot; didle = cidle - pidle
        if (dtot <= 0) { print ""; exit }
        printf "%.1f", (dtot - didle) / dtot * 100
    }' 2>/dev/null)
    prev="${curr}"

    load1=$(awk '{print $1}' /proc/loadavg 2>/dev/null)

    mem_line=$(awk '
        /^MemTotal:/ {t = $2}
        /^MemAvailable:/ {a = $2}
        END {if (t > 0) printf "%d,%d", (t - a) / 1024, t / 1024; else printf ","}
    ' /proc/meminfo 2>/dev/null)
    mem_line="${mem_line:-,}"

    gpu_line=$(sample_gpu 2>/dev/null)
    gpu_line="${gpu_line:-${GPU_BLANK_LINE}}"

    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "${ts},${cpu_pct},${load1},${mem_line},${gpu_line}" >> "${CSV}"
done
exit 0
