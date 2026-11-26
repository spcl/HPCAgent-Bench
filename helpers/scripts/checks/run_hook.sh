#!/usr/bin/env bash
# Runs a pre-commit hook script under hpcagent_bench/cluster/env.sh, so hooks importing hpcagent_bench work from any shell.
set -euo pipefail
# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
source "$(dirname "${BASH_SOURCE[0]}")/../../../hpcagent_bench/cluster/env.sh" >/dev/null
# The interpreter's bin first: a tool the hook starts (pyright looks up `python`) finds the checkout's environment,
# not an older install on PATH.
PATH="$(dirname "${HPCAGENT_BENCH_HOST_PYTHON}"):${PATH}"
export PATH
exec "${HPCAGENT_BENCH_HOST_PYTHON}" "$@"
