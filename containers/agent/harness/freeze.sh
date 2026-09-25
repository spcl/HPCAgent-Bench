#!/usr/bin/env bash
# Regenerate the harness npm lock with the node the images install:
#   node/package-lock.json    npm install --package-lock-only from node/package.json
# The CLI pins are node/package.json; the Python harness pins are the harness-<name> dependency
# groups of pyproject.toml, which the images install with uv directly.
#
#   containers/agent/harness/freeze.sh
set -Eeuo pipefail

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
export npm_config_cache="${WORK}/npm-cache" npm_config_update_notifier=false

sh "${HERE}/install_tools.sh" "${WORK}/tools"
export PATH="${WORK}/tools/bin:${PATH}"

cd "${HERE}/node"
npm install --package-lock-only --ignore-scripts --no-audit --no-fund
