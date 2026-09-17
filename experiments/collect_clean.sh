#!/usr/bin/env bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# What the LLR-Clean wave leaves behind, in one pass on one node: the owed lists, the llr-focus40-cpu
# extraction, and a commit of the artifact repo. Run by collect_clean.sbatch after the arms finish;
# runnable by hand on a login node with the same environment.
#   ARMS_JIDS=639060:639061 ./collect_clean.sh
# It does NOT push: a compute node has no route to GitHub. The READY file names the commit to push.
set -euo pipefail
ulimit -c 0
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

SCRATCH_ROOT=${SCRATCH:?}
OPT=${OPT:-$(dirname "${PWD}")}
PY=${PY:-${SCRATCH_ROOT}/venv-hpcagent-bench-314/bin/python}
RUNS=${RUNS:-${SCRATCH_ROOT}/hpcagent-bench-runs}
ARTIFACT=${ARTIFACT:-${SCRATCH_ROOT}/ICLR26Reproducibility-wt/restructure}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-experiments/llr-focus40-cpu}
# the jobs this collection is the tail of; recorded in the commit message so a table names its wave
JIDS=${ARMS_JIDS:?set ARMS_JIDS to the colon-joined job ids this collection follows}
TAG=${TAG:-llr-focus40}
RUN_ROOTS=${RUN_ROOTS:-${RUNS}/cpf-llr-focus40-2026*}
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
OWED=${RUNS}/owed/${STAMP}

# (i) what every arm of the wave still owes, one <arm>.txt per arm that owes anything
mkdir -p "${OWED}"
roots=()
for root in ${RUN_ROOTS}; do if [[ -d "${root}" ]]; then roots+=(--run-root "${root}"); fi; done
[[ ${#roots[@]} -gt 0 ]] || { echo "no run root matches ${RUN_ROOTS}" >&2; exit 2; }
"${PY}" ./remaining_kernels.py "${roots[@]}" --tag "${TAG}" --opt "${OPT}" --out-dir "${OWED}" \
    | tee "${OWED}/remaining.txt"

# (ii) the artifact's own extraction and figures: --extract rebuilds data/llr-focus40-cpu.db from the
# judge databases, the second run redraws every figure and table off it and checks the checksums.
# common.sh reads HPCAGENT_BENCH (the checkout) and PYTHON (the interpreter that has pandas).
export HPCAGENT_BENCH="${OPT}" PYTHON="${PY}" RUNS
pushd "${ARTIFACT}/${EXPERIMENT_DIR}" >/dev/null
./reproduce.sh --extract
./reproduce.sh
popd >/dev/null

# (iii) the commit. NO push: this runs on a compute node, which cannot reach GitHub.
git -C "${ARTIFACT}" add -A "${EXPERIMENT_DIR}"
git -C "${ARTIFACT}" commit -m "llr-focus40-cpu: LLR-Clean arms (${JIDS})" -m \
"The clean re-run supersedes the arms it re-ran: every figure and table here is drawn from the
clean tasks alone (spec X9), and the owed lists beside it are the clean arms' own."

# (iv) what a login node needs to push it
commit=$(git -C "${ARTIFACT}" rev-parse HEAD)
printf 'commit %s\nartifact %s\njobs %s\n' "${commit}" "${ARTIFACT}" "${JIDS}" >"${OWED}/READY"
echo "collected: ${OWED}/READY -> ${commit} (push it from a login node)"
