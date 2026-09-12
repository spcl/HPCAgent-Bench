#!/usr/bin/env bash
# Runs a pre-commit hook script under experiments/env.sh, so hooks importing hpcagent_bench work from any shell.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../experiments/env.sh" >/dev/null
exec "${PY}" "$@"
