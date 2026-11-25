#!/usr/bin/env bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# The released DaCe extended (dace-extended) and relayout (dace-relayout) this checkout runs: pip and uv install the
# same PyPI releases, pyproject.toml holds each lower bound and uv.lock the exact version.
#   pins.sh [dace|relayout]  print the version uv.lock holds (dace: dace-extended, the default)
#   pins.sh --bump           raise both lower bounds to the newest PyPI versions and relock them, and move the user
#                            plugins to their newest PyPI releases (user_plugins.py bump)
set -euo pipefail
ulimit -c 0
root="$(dirname -- "${BASH_SOURCE[0]}")/../.."
declare -A distribution=([dace]="dace-extended" [relayout]="dace-relayout")
# Every package releases from its own repository; --bump only moves this checkout's pins to what PyPI has.

locked() {
    python3 - "${root}/uv.lock" "$1" <<'EOF'
import sys
import tomllib

with open(sys.argv[1], "rb") as lock:
    versions = [p["version"] for p in tomllib.load(lock)["package"] if p["name"] == sys.argv[2]]
if len(versions) != 1:
    sys.exit(f"uv.lock: expected one {sys.argv[2]}, found {len(versions)}")
print(versions[0])
EOF
}

bump() {
    local name version before plugins_before
    # A failed step puts both files back, so pyproject.toml never disagrees with uv.lock beside it.
    before="$(cat "${root}/pyproject.toml"; echo x)"
    plugins_before="$(cat "${root}/.github/user-plugins.toml"; echo x)"
    restore() {
        printf '%s' "${before%x}" >"${root}/pyproject.toml"
        printf '%s' "${plugins_before%x}" >"${root}/.github/user-plugins.toml"
        echo "pins.sh: the bump failed; pyproject.toml and user-plugins.toml keep the previous pins" >&2
        exit 1
    }
    # uv finds the newest releases (PyPI's JSON pages lag an upload; the index uv reads is what installs), and
    # each lower bound then rises to the version it locked, so pip never resolves an older release than uv.
    (cd "${root}" && uv run --frozen --no-sync python helpers/scripts/user_plugins.py bump &&
        uv lock -q --refresh-package dace-extended --refresh-package dace-relayout \
            --upgrade-package dace-extended --upgrade-package dace-relayout) || restore
    for name in "${distribution[@]}"; do
        version="$(locked "${name}")"
        sed -i "s/\"${name}>=[0-9.]*\"/\"${name}>=${version}\"/" "${root}/pyproject.toml"
        grep -q "\"${name}>=${version}\"" "${root}/pyproject.toml" || restore
        echo "${name}>=${version}"
    done
    (cd "${root}" && uv lock -q) || restore
}

if [[ "${1:-}" == --bump ]]; then
    bump
    exit 0
fi
package="${1:-dace}"
[[ -n "${distribution[${package}]:-}" ]] || { echo "pins.sh: unknown package '${package}' (dace or relayout)" >&2; exit 2; }
locked "${distribution[${package}]}"
