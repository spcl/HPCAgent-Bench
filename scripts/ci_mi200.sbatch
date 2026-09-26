#!/usr/bin/env bash
# The CI verdict on one mi200 node, inside the judge image (its gcc 16 / ROCm toolchain is the one
# graded runs use; the host gcc has no -std=c23). Submit from the checkout root.
#
#   P=--partition=${HPCAGENT_BENCH_CI_PARTITION}   # the site's MI250X partition (site layer)
#   sbatch $P scripts/ci_mi200.sbatch                       every job of tests.yml
#   CI_EDF=hpcagent-bench-judge-mi300-latest sbatch --partition=mi300 scripts/ci_mi200.sbatch   another image
#   sbatch $P scripts/ci_mi200.sbatch --ci --jobs unit,mpi  chosen jobs (scripts/ci_replay.py)
#   sbatch $P scripts/ci_mi200.sbatch -q tests/test_x.py    a pytest selection
#   scripts/run_tests.sh --container [same args]                      submit and wait
#
# The default run replays every job with the gpu job's hardware groups, skipping what needs a
# GitHub-only service (the docker image build and its launch, the coverage merge). Per-step logs
# and summary.txt land in ${CI_OUT:-$SCRATCH/ci-replay/<jobid>}.
#
#SBATCH --job-name=ci-mi200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --no-requeue
set -Eeuo pipefail
ulimit -c 0

# sbatch runs a spool copy of this script, so the checkout is the submit directory.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
[[ -x "${REPO}/scripts/run_tests.sh" ]] || { echo "no scripts/run_tests.sh under REPO=${REPO}" >&2; exit 2; }
: "${SCRATCH:?set SCRATCH}"
job="${SLURM_JOB_ID:-local}"
# On disk: /tmp is tmpfs here and the compile-heavy jobs fill it.
tmp="${SCRATCH}/.tmp/ci-mi200-${job}"
mkdir -p "${tmp}"
trap 'rm -rf "${tmp}"' EXIT

if [[ $# -eq 0 ]]; then
    set -- --ci --out "${CI_OUT:-${SCRATCH}/ci-replay/${job}}" --parallel "${CI_PARALLEL:-12}" \
        --skip "coverage,integration/Phase 6 -- container launch"
fi

# dace: the image's /opt/dace at the job's commit (containers/images/dace_refresh.sh).
HPCAGENT_BENCH_DACE_REF="$("${REPO}/containers/images/dace_refresh.sh" --resolve)"
export HPCAGENT_BENCH_DACE_REF

# The image's venv is the interpreter; the host venv under $SCRATCH is built for the login node's OS.
srun --environment="${CI_EDF:-hpcagent-bench-judge-mi200-latest}" --overlap --nodes=1 --ntasks=1 \
    bash -c '"${0%/scripts/run_tests.sh}/containers/images/dace_refresh.sh" || exit 1
             export TMPDIR="$1" HPCAGENT_BENCH_HOST_PYTHON=/opt/venv/bin/python; shift; exec "$0" "$@"' \
    "${REPO}/scripts/run_tests.sh" "${tmp}" "$@"
