#!/usr/bin/env bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# The deterministic compiler columns (numba, cc, cc_autopar, dace_cpu[_canonicalize],
# dace_gpu[_canonicalize], pluto, ...) over one tag's roster, no agents and no judge; one job per
# column, timed at run_cluster.sh's grading width (a baseline on a different core count is not one).
#   TAG=llr-focus40 SUBMIT=0 ./submit-canon.sh            # dry run
#   TAG=llr-focus40 COLUMNS=pluto SUBMIT=1 ./submit-canon.sh
#   KERNELS_FILE=owed/arm.txt ./submit-canon.sh           # one kernel name per line, replaces the roster
set -euo pipefail
ulimit -c 0
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
OPT=${OPT:-$(dirname "${PWD}")}
. "$(dirname -- "${BASH_SOURCE[0]}")/roster.sh"
# HPCAGENT_BENCH_RUNS_ROOT, so a canon campaign's work dir (CSVs, opt reports, and -- inside
# canon_column.sh -- the DaCe build tree + per-rank shard DB it clears on a verified merge) lives
# under the cache, not loose in $SCRATCH where nothing sweeps it. See .cache/README.md's "Job work dirs".
. "${OPT}/experiments/env.sh"
STAMP=${STAMP:-$(date +%Y%m%d)}
OUT_ROOT=${OUT_ROOT:-${HPCAGENT_BENCH_RUNS_ROOT}/canon/${TAG:-llr-focus40}-${STAMP}}
PRESET=${PRESET:-fuzzed}
TIME_LIMIT=${TIME_LIMIT:-12:00:00}
# one kernel name per line; narrows the roster as submit.sh's KERNELS_FILE does. Empty = the whole tag.
KERNELS_FILE=${KERNELS_FILE:-}
if [[ -n "${KERNELS_FILE}" ]]; then
    [[ -s "${KERNELS_FILE}" ]] || { echo "KERNELS_FILE ${KERNELS_FILE} is missing or empty" >&2; exit 2; }
    KERNELS=$(sed -e 's/#.*//' -e 's/[[:space:]]*$//' "${KERNELS_FILE}" | grep . | sort -u | paste -sd, -)
    [[ -n "${KERNELS}" ]] || { echo "KERNELS_FILE ${KERNELS_FILE} names no kernels" >&2; exit 2; }
    # canon_column.sh runs a kernel by name with no registry check of its own (a typo only fails deep
    # inside the job, after a node was already held for it); resolved the same way make_problems.py's
    # --kernels-file resolves a selector, so the message and the accepted spellings match everywhere.
    "${HPCAGENT_BENCH_HOST_PYTHON}" -c '
import sys
from hpcagent_bench.spec import KERNELS
unknown = []
for name in sys.argv[1].split(","):
    try:
        KERNELS.select_keys(name)
    except KeyError as exc:
        unknown.append(f"{name} ({exc.args[0]})")
if unknown:
    sys.exit("KERNELS_FILE names unknown kernel(s): " + "; ".join(unknown))
' "${KERNELS}" || exit 2
else
    KERNELS=${KERNELS:-$(roster_for "${TAG:-llr-focus40}")}
fi

COLUMNS=${COLUMNS:-"numba cc cc_autopar dace_cpu dace_cpu_canonicalize dace_gpu dace_gpu_canonicalize"}

#: Full opt/vectorization reports + the assembly of the exact measured build, for every C/C++/
#: Fortran column in COLUMNS (hpcagent_bench/opt_reports.py; canon_column.sh's inner loop reads
#: this as CANON_OPT_REPORTS). ON by default per the compiler-column reporting requirement: the
#: cost is one extra compile-only pass per kernel, and only for a column cpp_runtime.FRAMEWORK_LANG
#: marks as native -- a dace/numba column in COLUMNS gets a one-line "not a compiled column"
#: manifest instead of a wasted compile. Set OPT_REPORTS=0 to skip it entirely (e.g. a pure timing
#: run where even that manifest write is unwanted).
OPT_REPORTS=${OPT_REPORTS:-1}
export CANON_OPT_REPORTS="${OPT_REPORTS}"

# Every column must be a framework the registry knows, checked HERE: inside the job an unknown name
# crashes on every kernel of every rank, after the node was already held for it.
"${HPCAGENT_BENCH_HOST_PYTHON}" -c '
import sys
from hpcagent_bench.frameworks.framework import FRAMEWORK_META
unknown = sorted({c for c in sys.argv[1:] if c not in FRAMEWORK_META})
if unknown:
    sys.exit(f"unknown canon column(s): {unknown}; known: {sorted(FRAMEWORK_META)}")
' $(tr ', ' '\n\n' <<<"${COLUMNS}") || exit 2

mkdir -p "${OUT_ROOT}"
printf 'roster: %s kernels\n' "$(tr ',' '\n' <<<"${KERNELS}" | wc -l)"

if [[ "${ONE_JOB:-1}" == 1 ]]; then
    COLUMNS="$(tr ' ' ',' <<<"${COLUMNS}" | sed 's/,\+/,/g; s/^,//; s/,$//')"
    TIME_LIMIT=${TIME_LIMIT_ONE_JOB:-24:00:00}
fi

# The job name is the experiment PREFIX that extraction matches on (README: --experiment is a
# prefix). It was hardcoded canon40 whatever the roster, so a full-track run of 248 kernels was
# filed under the same prefix as the 40-kernel focus set and extracted as one campaign. Kept as
# canon40 for the default focus roster so existing runs keep their name.
if [[ "${TAG:-llr-focus40}" == "llr-focus40" ]]; then
    JOB_PREFIX=${JOB_PREFIX:-canon40}
else
    JOB_PREFIX=${JOB_PREFIX:-canon-${TAG}}
fi

for col in ${COLUMNS}; do
    gres=()
    # A DEVICE column gets GPUs: *gpu* (dace_gpu*) and ppcg* (ppcg_hip). tests/test_canon_device_columns.py
    # keeps this pattern equal to the set cpp_runtime.FRAMEWORK_LANG marks as hip/cuda.
    # Tested PER COLUMN: with TIME_LIMIT_ONE_JOB the columns are packed into one comma-joined
    # value, and "numba,ppcg_hip" neither contains "gpu" nor starts with "ppcg".
    for one in ${col//,/ }; do
        [[ "${one}" == *gpu* || "${one}" == ppcg* ]] && gres=(--gres=gpu:4)
    done
    if [[ "${SUBMIT:-0}" != 1 ]]; then
        echo "would submit ${col}${BEGIN:+ (begin ${BEGIN})}"
        continue
    fi
    dep=(); [[ -n "${DEPEND_ON:-}" ]] && dep=(--dependency="afterany:${DEPEND_ON}")
    #: NICE=300, e.g., puts a gap-filling canon run behind the priority queue's LLR/cpfsrc waves
    #: but ahead of a background scicomp sweep without touching either queue's own submitter.
    #: Unset, the site's default nice (HPCAGENT_BENCH_NICE, scripts/site_env.sh).
    nice=(--nice="${NICE:-${HPCAGENT_BENCH_NICE}}")
    jid=$(sbatch --parsable --no-requeue --nodes=1 --exclusive --mem=0 \
        "${gres[@]}" --time="${TIME_LIMIT}" --job-name="${JOB_PREFIX}-${JOB_TAG:-${col%%,*}}" \
        "${dep[@]}" "${nice[@]}" ${BEGIN:+--begin="${BEGIN}"} \
        --output="${OUT_ROOT}/%x-%j.out" --error="${OUT_ROOT}/%x-%j.err" \
        --wrap "bash ${PWD}/canon_column.sh outer ${col} ${OUT_ROOT} ${KERNELS} ${PRESET} ${OPT}")
    echo "submitted ${col} -> ${jid}${BEGIN:+ (begin ${BEGIN})}"
done
