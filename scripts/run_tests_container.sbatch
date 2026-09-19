#!/usr/bin/env bash
# Run scripts/run_tests.sh INSIDE the judge EDF's own container image, on one mi300 compute node.
#
# Never call this directly -- `scripts/run_tests.sh --container [pytest args...]` submits it (--wait)
# and passes the args through; that script's header explains why this exists (login/compute-node
# gcc has no -std=c23, the judge image does).
#
#SBATCH --job-name=run-tests-container
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --no-requeue
set -Eeuo pipefail
ulimit -c 0

# SLURM_SUBMIT_DIR, not BASH_SOURCE: sbatch COPIES the script to /var/spool/slurmd, so inside the
# job BASH_SOURCE names the spool copy and dirname/.. is /var/spool/slurmd (same trap as
# tools/suite.sbatch, which this mirrors).
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
[[ -x "${REPO}/scripts/cscs/enroot_srun.sh" ]] || { echo "no scripts/cscs/enroot_srun.sh under REPO=${REPO}" >&2; exit 2; }
[[ -x "${REPO}/scripts/run_tests.sh" ]] || { echo "no scripts/run_tests.sh under REPO=${REPO}" >&2; exit 2; }

# `enroot start` does not inherit the host environment (scripts/cscs/enroot_srun.sh's own header):
# PATH and the compiler toolchain come entirely from the judge image, which is the point -- the
# host/compute-node gcc this whole script exists to route around never enters the picture.
#
# The inner command resolves the CONTAINER's own python3 BEFORE scripts/run_tests.sh (sourced next)
# gets a chance to prepend a host-mounted $SCRATCH venv onto PATH: SCRATCH is one of the few
# variables enroot_srun.sh's rank-mode forwarding carries in (scripts/cscs/enroot_forward.sh), and
# that venv is readable but built on the LOGIN NODE'S OS, not this image's -- the wrong interpreter
# to run pytest under here. Capturing the absolute path first and exporting it as PY means
# experiments/env.sh's now-overridable PY export (see its own comment) keeps this one.
exec "${REPO}/scripts/cscs/enroot_srun.sh" hpcagent-bench-judge-mi300-latest \
    --overlap --nodes=1 --ntasks=1 -- \
    bash -c 'PY="$(command -v python3)"; export PY; exec "$0" "$@"' \
    "${REPO}/scripts/run_tests.sh" "$@"
