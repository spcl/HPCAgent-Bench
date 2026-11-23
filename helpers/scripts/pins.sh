#!/usr/bin/env bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# The released DaCe extended (dace-extended) and relayout (dace-relayout) this checkout runs: pip and uv install the
# same PyPI releases, pyproject.toml holds each lower bound and uv.lock the exact version.
#   pins.sh [dace|relayout]  print the version uv.lock holds (dace: dace-extended, the default)
#   pins.sh --bump           raise both lower bounds to the newest PyPI versions and relock them, and move the user
#                            plugins to their repositories' newest vX.Y.Z tags (user_plugins.py bump)
#   pins.sh --release        first patch-release DaCe extended (dace_extended_release.sh) and every library below (its
#                            helpers/release_patch.sh, run in a fresh clone of main: tag, GitHub release, PyPI upload;
#                            needs TWINE_USER and TWINE_TOKEN), then --bump. A failed release does not stop the others.
set -euo pipefail
ulimit -c 0
root="$(dirname -- "${BASH_SOURCE[0]}")/../.."
declare -A distribution=([dace]="dace-extended" [relayout]="dace-relayout")
LIBRARIES=(
    "https://github.com/ThrudPrimrose/dace-relayout.git"
    "https://github.com/ThrudPrimrose/dace-alloy.git"
    "https://github.com/spcl/tactics.git"
)

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

newest() {
    curl -fsS "https://pypi.org/pypi/$1/json" | python3 -c 'import json, sys; print(json.load(sys.stdin)["info"]["version"])'
}

release() {
    local failed=() url work
    "${root}/helpers/scripts/dace_extended_release.sh" || failed+=("DaCe extended")
    for url in "${LIBRARIES[@]}"; do
        echo "=== ${url} ==="
        work="$(mktemp -d "${TMPDIR:-/tmp}/pins-release.XXXXXX")"
        if ! git clone -q --branch main "${url}" "${work}/repo"; then
            failed+=("${url} (clone)")
        elif [[ ! -x "${work}/repo/helpers/release_patch.sh" ]]; then
            failed+=("${url} (no helpers/release_patch.sh)")
        else
            "${work}/repo/helpers/release_patch.sh" || failed+=("${url}")
        fi
        rm -rf "${work}"
    done
    ((${#failed[@]} == 0)) || printf 'pins.sh: not released: %s\n' "${failed[@]}" >&2
    return "${#failed[@]}"
}

bump() {
    local name version before plugins_before
    # A failed relock puts both files back, so pyproject.toml never disagrees with uv.lock beside it.
    before="$(cat "${root}/pyproject.toml"; echo x)"
    plugins_before="$(cat "${root}/.github/user-plugins.toml"; echo x)"
    for name in "${distribution[@]}"; do
        version="$(newest "${name}")"
        [[ "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "pins.sh: no PyPI release of ${name}" >&2; exit 2; }
        sed -i "s/\"${name}>=[0-9.]*\"/\"${name}>=${version}\"/" "${root}/pyproject.toml"
        grep -q "\"${name}>=${version}\"" "${root}/pyproject.toml" || { echo "pins.sh: no ${name}>= bound" >&2; exit 2; }
        echo "${name}>=${version}"
    done
    if ! (cd "${root}" && uv run --frozen --no-sync python helpers/scripts/user_plugins.py bump &&
        uv lock -q --upgrade-package dace-extended --upgrade-package dace-relayout); then
        printf '%s' "${before%x}" >"${root}/pyproject.toml"
        printf '%s' "${plugins_before%x}" >"${root}/.github/user-plugins.toml"
        echo "pins.sh: the bump failed; pyproject.toml and user-plugins.toml keep the previous pins" >&2
        exit 1
    fi
}

released=0
case "${1:-}" in
    --release)
        release || released=$?
        bump
        exit "${released}"
        ;;
    --bump)
        bump
        exit 0
        ;;
esac
package="${1:-dace}"
[[ -n "${distribution[${package}]:-}" ]] || { echo "pins.sh: unknown package '${package}' (dace or relayout)" >&2; exit 2; }
locked "${distribution[${package}]}"
