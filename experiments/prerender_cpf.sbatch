#!/usr/bin/env bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#SBATCH --job-name=cpf-prerender
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=11:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/cpf-prerender-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/cpf-prerender-%j.out
#
# THE preparation job for canonical parallel forms. Renders a roster's read forms and drop-ins into
# the content-addressed cache (hpcagent_bench/cpf_cache.py) and pins a VIEW, the one directory an arm
# names in HPCAGENT_BENCH_SERVICE_CANONICAL_PARALLEL_FORM_DIR and CPF_DROPIN_DIR. Nothing renders
# after this job: the judge, the drop-in staging and the launch gates only read the cache. A hit
# renders nothing, so a rerun renders only what changed.
#
#   VIEW=<dir> KERNELS_FILE=kernels-scicomp40.txt sbatch prerender_cpf.sbatch
#   VIEW=<dir> TAG=llr-focus40 TARGET=gpu sbatch prerender_cpf.sbatch
#
# DACE_TREE and OPT may name frozen snapshots. The key hashes the dace source, and a dace edit during
# the run withdraws what the run rendered.
set -euo pipefail
ulimit -c 0
SCRATCH_ROOT=/capstor/scratch/cscs/${USER}/x86_64
OPT=${OPT:-${SCRATCH_ROOT}/optarena}
export DACE_TREE=${DACE_TREE:-${SCRATCH_ROOT}/dace}
CPF_CACHE=${CPF_CACHE:-${SCRATCH_ROOT}/cpf-cache}
TARGET=${TARGET:-cpu}
VIEW=${VIEW:?set VIEW to the directory the arms will point at}
trap 'rm -rf "${CPF_CACHE}/.scratch/${SLURM_JOB_ID:-local}-"*' EXIT

if [[ -n "${KERNELS:-}" ]]; then
    ROSTER=${KERNELS}
elif [[ -n "${KERNELS_FILE:-}" ]]; then
    [[ -f "${KERNELS_FILE}" ]] || KERNELS_FILE="${OPT}/experiments/${KERNELS_FILE}"
    # a roster line may carry a trailing `# note`, so the name is what precedes the first `#`
    ROSTER=$(sed -e 's/#.*//' -e 's/[[:space:]]*$//' "${KERNELS_FILE}" | grep . | paste -sd, -)
else
    . "${OPT}/experiments/roster.sh"
    ROSTER=$(roster_for "${TAG:?set KERNELS, KERNELS_FILE or TAG}")
fi
# Sharding is by position, so the renders that outrun the rest go first, each on a rank of its own.
FIRST=${FIRST:-cloudsc sw4_rhs4sg}
ordered=()
for k in ${FIRST}; do [[ ",${ROSTER}," == *",${k},"* ]] && ordered+=("${k}"); done
for k in ${ROSTER//,/ }; do [[ " ${FIRST} " == *" ${k} "* ]] || ordered+=("${k}"); done
KERNEL_LIST=$(IFS=,; echo "${ordered[*]}")

# The env script signals failure by its status; a shell that continues renders with the system gcc.
cd /
set +u
source "${SCRATCH_ROOT}/dace-env.sh" || { echo "prerender: dace-env.sh failed" >&2; exit 1; }
set -u
[[ "${CXX:-}" == */spack/* && -n "${OPENBLAS_DIR:-}" ]] || { echo "prerender: not the spack toolchain (CXX=${CXX:-})" >&2; exit 1; }
export PYTHONPATH="${DACE_TREE}:${OPT}:${OPT}/hpcagent_bench/numpy_translators/src"
export PYTHONHASHSEED=0
cd "${OPT}"
python3 -c 'import sys, dace; sys.exit(dace.__file__ != sys.argv[1])' "${DACE_TREE}/dace/__init__.py" \
    || { echo "prerender: dace does not resolve to ${DACE_TREE}" >&2; exit 1; }

ranks=$(lscpu -p=SOCKET | grep -v '^#' | sort -u | wc -l)
cores=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | awk -F, '$2 == 0' | wc -l)
echo "prerender ${TARGET}: ${#ordered[@]} kernels, ${ranks} ranks x ${cores} cores"
echo "  cache ${CPF_CACHE}"
echo "  view  ${VIEW}"
echo "  dace  ${DACE_TREE} ($(git -C "${DACE_TREE}" rev-parse HEAD 2>/dev/null || echo 'no checkout'))"
# --kill-on-bad-exit=0: a rank's own fail verdicts never end the shard early. A rank exits nonzero
# only for an internal error (broken toolchain, unwritable cache, dace edited mid-run), and only
# then should this status stay nonzero.
status=0
srun --ntasks="${ranks}" --cpus-per-task="${cores}" --hint=nomultithread --mem=0 --kill-on-bad-exit=0 \
    python3 -m hpcagent_bench.cpf_prerender --cache "${CPF_CACHE}" --view "${VIEW}" \
    --kernels "${KERNEL_LIST}" --target "${TARGET}" || status=$?

# The roster-wide verdict table: every shard has finished (or failed hard) by the time this runs, so
# a miss here is a real gap, never a sibling rank still rendering.
langs="c c++"
[[ "${TARGET}" == gpu ]] && langs=hip
checks_status=0
for lang in ${langs}; do
    for mode in form dropin; do
        miss=$(python3 -m hpcagent_bench.cpf_cache check --view "${VIEW}" --kernels "${KERNEL_LIST}" \
               --language "${lang}" --mode "${mode}") || checks_status=1
        echo "${lang} ${mode}: $(( ${#ordered[@]} - $(grep -c . <<<"${miss}" || true) )) of ${#ordered[@]} served"
        [[ -z "${miss}" ]] || sed 's/^/  missing /' <<<"${miss}"
    done
done
exit $(( status == 0 ? checks_status : 1 ))
