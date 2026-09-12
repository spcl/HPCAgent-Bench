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

OPT=${SCRATCH:?}/optarena
PY=${SCRATCH:?}/venv-optarena-314/bin/python
. "$(dirname -- "${BASH_SOURCE[0]}")/roster.sh"
STAMP=${STAMP:-$(date +%Y%m%d)}
OUT_ROOT=${OUT_ROOT:-${SCRATCH:?}/canon-llr40-${STAMP}}
PRESET=${PRESET:-fuzzed}
TIME_LIMIT=${TIME_LIMIT:-12:00:00}
KERNELS=${KERNELS:-$(roster_for "${TAG:-llr-focus40}")}

COLUMNS=${COLUMNS:-"numba cc cc_autopar dace_cpu dace_cpu_canonicalize dace_gpu dace_gpu_canonicalize"}

mkdir -p "${OUT_ROOT}"
printf 'roster: %s kernels\n' "$(tr ',' '\n' <<<"${KERNELS}" | wc -l)"

if [[ "${ONE_JOB:-1}" == 1 ]]; then
    COLUMNS="$(tr ' ' ',' <<<"${COLUMNS}" | sed 's/,\+/,/g; s/^,//; s/,$//')"
    TIME_LIMIT=${TIME_LIMIT_ONE_JOB:-24:00:00}
fi

for col in ${COLUMNS}; do
    gres=()
    [[ "${col}" == *gpu* ]] && gres=(--gres=gpu:4)
    if [[ "${SUBMIT:-1}" != 1 ]]; then
        echo "would submit ${col}${BEGIN:+ (begin ${BEGIN})}"
        continue
    fi
    dep=(); [[ -n "${DEPEND_ON:-}" ]] && dep=(--dependency="afterany:${DEPEND_ON}")
    jid=$(sbatch --parsable --partition=mi300 --nodes=1 --exclusive --mem=0 \
        "${gres[@]}" --time="${TIME_LIMIT}" --job-name="canon40-${JOB_TAG:-${col%%,*}}" \
        "${dep[@]}" ${BEGIN:+--begin="${BEGIN}"} \
        --output="${OUT_ROOT}/%x-%j.out" --error="${OUT_ROOT}/%x-%j.err" \
        --wrap "bash ${PWD}/canon_column.sh outer ${col} ${OUT_ROOT} ${KERNELS} ${PRESET} ${OPT}")
    echo "submitted ${col} -> ${jid}${BEGIN:+ (begin ${BEGIN})}"
done
