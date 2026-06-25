#!/usr/bin/env bash
# Thin launcher for tests/extended_smoke.py.
#   OPTARENA_TEST_PYTHON=/path/to/python tests/extended_smoke.sh [args]
# defaults to the python3 on PATH.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYBIN="${OPTARENA_TEST_PYTHON:-python3}"
exec "${PYBIN}" "${SCRIPT_DIR}/extended_smoke.py" "$@"
