#!/usr/bin/env bash
# Runs a pre-commit hook script under experiments/env.sh, so hooks importing hpcagent_bench work from any shell.
set -euo pipefail
# Hooks never write a cache; without SCRATCH (CI) env.sh's own HPCAGENT_BENCH_REPO fallback
# (scripts/cache_env.sh) already lands JIT_CACHE_ROOT under this checkout's .cache/jit -- no need
# to compute that here too.

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
source "$(dirname "${BASH_SOURCE[0]}")/../experiments/env.sh" >/dev/null
# env.sh defaults VENV to the cluster scratch; a checkout without that venv runs the hook on the PATH python.
[ -x "${PY}" ] || PY="$(command -v python3)"
exec "${PY}" "$@"
