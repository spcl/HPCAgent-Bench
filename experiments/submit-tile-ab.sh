#!/usr/bin/env bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# A/B the GPU wavefront tile extent (wavefront_skew.DEFAULT_GPU_TILE_SIZE) on the kernels that
# take the skewed lowering. The two arms are two dace WORKTREES, named by DACE_TREE, so nothing
# but that constant differs -- and because canon_column.sh keys its PCH root on the dace commit,
# the arms cannot share a header cache either.
#
# Both arms run SEQUENTIALLY IN ONE JOB on ONE node: a tile size is a bandwidth/occupancy trade,
# and two arms on two nodes measure the nodes as much as the constant.
#
#   ARM_A=$SCRATCH/dace ARM_B=$SCRATCH/dace-tile128 ./submit-tile-ab.sh
set -euo pipefail
ulimit -c 0
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

OPT=${SCRATCH:?}/optarena
ARM_A=${ARM_A:-${SCRATCH:?}/dace}
ARM_B=${ARM_B:-${SCRATCH:?}/dace-tile128}
OUT_ROOT=${OUT_ROOT:-${SCRATCH:?}/tile-ab-$(date +%Y%m%d%H%M)}
COL=${COL:-dace_gpu_canonicalize}
PRESET=${PRESET:-fuzzed}
#: The kernels whose canon lowering is the skewed wavefront, plus two 2-D scans as controls: the
#: tile extent must not move a kernel it does not apply to, and a control that DID move says the
#: arms differ by something other than the constant.
KERNELS=${KERNELS:-wf_triangular,wf_diff_skew,tsvc_2_s119,tsvc_2_s115}

mkdir -p "${OUT_ROOT}"
jid=$(sbatch --parsable --partition=mi300 --nodes=1 --exclusive --mem=0 --gres=gpu:4 \
    --time="${TIME_LIMIT:-04:00:00}" --job-name=tile-ab \
    --output="${OUT_ROOT}/%x-%j.out" --error="${OUT_ROOT}/%x-%j.err" \
    --wrap "set -x
        DACE_TREE=${ARM_A} bash ${PWD}/canon_column.sh outer ${COL} ${OUT_ROOT}/tile64 ${KERNELS} ${PRESET} ${OPT}
        DACE_TREE=${ARM_B} bash ${PWD}/canon_column.sh outer ${COL} ${OUT_ROOT}/tile128 ${KERNELS} ${PRESET} ${OPT}")
echo "submitted tile A/B -> ${jid}  (${OUT_ROOT})"
