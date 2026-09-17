#!/usr/bin/env bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# The COMPILER-BASELINE half of llr40: seven columns (numba, cc, cc_autopar, dace_cpu[_canonicalize],
# dace_gpu[_canonicalize]) over llr-focus40, no agents; one job per column, timed at run_cluster.sh's
# grading width (a baseline on a different core count is not a baseline).
#   ./submit-canon-llr40.sh   BEGIN=saturday|DEPEND_ON=<jid:jid>|SUBMIT=0 ./submit-canon-llr40.sh
set -euo pipefail
ulimit -c 0
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# The Slurm account for every sbatch below (scripts/cscs/account_env.sh; this script does not source
# submit_common.sh, which is where the family submitters get it).
. "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/cscs/account_env.sh" || { echo "no Slurm account resolved; see scripts/cscs/account_env.sh" >&2; exit 2; }

OPT=${OPT:-$(dirname "${PWD}")}
. "$(dirname -- "${BASH_SOURCE[0]}")/roster.sh"
STAMP=${STAMP:-$(date +%Y%m%d)}
OUT_ROOT=${OUT_ROOT:-${SCRATCH:?}/canon-llr40-${STAMP}}
PRESET=${PRESET:-fuzzed}
TIME_LIMIT=${TIME_LIMIT:-12:00:00}
KERNELS=${KERNELS:-$(roster_for "${TAG:-llr-focus40}")}

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
PY=${PY:-${SCRATCH:?}/venv-hpcagent-bench-314/bin/python}
PYTHONPATH="${OPT}:${OPT}/hpcagent_bench/numpy_translators/src" "${PY}" -c '
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
    # A DEVICE column gets GPUs. The name test used to be *gpu* alone, which catches dace_gpu* and
    # misses the PPCG columns -- so ppcg_hip, the AMD CUDA->HIP column, was submitted with no GPU
    # and could only fail or run nowhere near a device. tests/test_canon_device_columns.py keeps
    # this pattern equal to the set cpp_runtime.FRAMEWORK_LANG marks as hip/cuda, so a new device
    # column cannot be added there and silently left CPU-only here.
    # Tested PER COLUMN: with TIME_LIMIT_ONE_JOB the columns are packed into one comma-joined
    # value, and "numba,ppcg_hip" neither contains "gpu" nor starts with "ppcg".
    for one in ${col//,/ }; do
        [[ "${one}" == *gpu* || "${one}" == ppcg* ]] && gres=(--gres=gpu:4)
    done
    if [[ "${SUBMIT:-1}" != 1 ]]; then
        echo "would submit ${col}${BEGIN:+ (begin ${BEGIN})}"
        continue
    fi
    dep=(); [[ -n "${DEPEND_ON:-}" ]] && dep=(--dependency="afterany:${DEPEND_ON}")
    jid=$(sbatch --parsable --no-requeue --partition=mi300 --nodes=1 --exclusive --mem=0 \
        "${gres[@]}" --time="${TIME_LIMIT}" --job-name="${JOB_PREFIX}-${JOB_TAG:-${col%%,*}}" \
        "${dep[@]}" ${BEGIN:+--begin="${BEGIN}"} \
        --output="${OUT_ROOT}/%x-%j.out" --error="${OUT_ROOT}/%x-%j.err" \
        --wrap "bash ${PWD}/canon_column.sh outer ${col} ${OUT_ROOT} ${KERNELS} ${PRESET} ${OPT}")
    echo "submitted ${col} -> ${jid}${BEGIN:+ (begin ${BEGIN})}"
done
