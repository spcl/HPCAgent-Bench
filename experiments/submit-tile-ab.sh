#!/usr/bin/env bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail
ulimit -c 0
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

OPT=${SCRATCH:?}/optarena
ARM_A=${ARM_A:-${SCRATCH:?}/dace}
ARM_B=${ARM_B:-${SCRATCH:?}/dace-tile128}
OUT_ROOT=${OUT_ROOT:-${SCRATCH:?}/tile-ab-$(date +%Y%m%d%H%M)}
COL=${COL:-dace_gpu_canonicalize}
PRESET=${PRESET:-fuzzed}
KERNELS=${KERNELS:-wf_triangular,wf_diff_skew,tsvc_2_s119,tsvc_2_s115}

mkdir -p "${OUT_ROOT}"
jid=$(sbatch --parsable --partition=mi300 --nodes=1 --exclusive --mem=0 --gres=gpu:4 \
    --time="${TIME_LIMIT:-04:00:00}" --job-name=tile-ab \
    --output="${OUT_ROOT}/%x-%j.out" --error="${OUT_ROOT}/%x-%j.err" \
    --wrap "set -x
        DACE_TREE=${ARM_A} bash ${PWD}/canon_column.sh outer ${COL} ${OUT_ROOT}/tile64 ${KERNELS} ${PRESET} ${OPT}
        DACE_TREE=${ARM_B} bash ${PWD}/canon_column.sh outer ${COL} ${OUT_ROOT}/tile128 ${KERNELS} ${PRESET} ${OPT}")
echo "submitted tile A/B -> ${jid}  (${OUT_ROOT})"
