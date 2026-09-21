#!/usr/bin/env bash
# The full pytest suite, on a compute node.
#
# It compiles and runs native kernels for ~3 hours. That does not belong on a shared login node,
# and a login run is also slower and less honest: conftest picks a thread count from the machine it
# is on, so a contended login node quietly changes what the timing tests measure.
#
#SBATCH --job-name=suite
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out
#SBATCH --no-requeue
set -Eeuo pipefail
ulimit -c 0

# SLURM_SUBMIT_DIR, not BASH_SOURCE: sbatch COPIES the script to /var/spool/slurmd, so inside the
# job BASH_SOURCE names the spool copy and dirname/.. is /var/spool/slurmd.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
[[ -x "${REPO}/scripts/run_tests.sh" ]] || { echo "no scripts/run_tests.sh under REPO=${REPO}" >&2; exit 2; }
# On disk, not /tmp: /tmp is tmpfs here and a compile-heavy suite fills it.
export TMPDIR="${SCRATCH:?set SCRATCH}/.tmp/suite-${SLURM_JOB_ID:-0}"
mkdir -p "${TMPDIR}"

# run_tests.sh is the only thing that knows the environment the suite needs -- venv bin on PATH for
# pythran's console script, PKG_CONFIG_PATH and CPATH for spack openblas. Without it 84 of 90
# failures in a full run are the shell rather than the tree.
cd -- "${REPO}"
# xdist workers: serially the suite reached 19% in 3 h on a compute node; 64 workers ran out of memory
# (469 GB RSS) and stalled, 16 finish inside the limit.
"${REPO}/scripts/run_tests.sh" -q --maxfail=25 -n "${SUITE_WORKERS:-16}" --dist loadgroup tests/
