#!/bin/bash
#SBATCH --job-name=fft-hoister-test
#SBATCH -A a-g34
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=00:30:00
#SBATCH --nice=0
#SBATCH --output=%x-%j.out
# Final gate before merge: the real pytest suite, in the real judge container, for the FFT
# lowering fix (hoister dtype bug, already on main @ 3c14b73e) plus the thread-cap fix for the
# XL hang (this branch, not yet merged).
set -uo pipefail
REPO="${REPO:?set REPO}"
srun --environment=hpcagent-bench-judge-mi300-latest bash -c '
  set -x

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_nid<node>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
  export SCRATCH="'"${SCRATCH}"'"
  # experiments/env.sh derives VENV/PY from $SCRATCH by default, a login-node-only venv
  # (symlinked into $HOME, not mounted in this container -- 643138 hit exactly that: "No
  # such file or directory"). VENV already-set wins (env.sh: VENV="${VENV:-...}"), so point
  # it at the containers own venv, the one the judge EDF says has hpcagent_bench installed.
  export VENV=/opt/venv
  cd "'"${REPO}"'"
  bash scripts/run_tests.sh hpcagent_bench/numpy_translators/tests/test_fft_library_lowering.py -q
'
