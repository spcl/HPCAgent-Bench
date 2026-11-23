#!/usr/bin/env bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# The released DaCe extended (dace-extended) and relayout (dace-relayout) this checkout runs: pip and uv install the
# same PyPI releases, pyproject.toml holds each lower bound and uv.lock the exact version.
#   pins.sh [dace|relayout]  print the version uv.lock holds (dace: dace-extended, the default)
#   pins.sh --bump           raise both lower bounds to the newest PyPI versions and relock them, and move the user
#                            plugins to their repositories' newest vX.Y.Z tags (user_plugins.py bump)
#   pins.sh --release        first patch-release DaCe extended (dace_extended_release.sh) and the library below (its
#                            helpers/release_patch.sh, run in a fresh clone of main: tag and GitHub release, whose
#                            Publish workflow uploads to PyPI; DaCe extended needs TWINE_USER and TWINE_TOKEN), then
#                            --bump. A failed release does not stop the others.
set -euo pipefail
ulimit -c 0
root="$(dirname -- "${BASH_SOURCE[0]}")/../.."
declare -A distribution=([dace]="dace-extended" [relayout]="dace-relayout")
# Only what the bench imports: dace-relayout, released after DaCe extended. The user plugins (dace-alloy, tactics)
# import hpcagent-bench, not the reverse: they release from their own helpers/release_patch.sh, and --bump moves the
# commits the user-plugins CI job checks out to their newest release tags.
LIBRARIES=(
    "https://github.com/ThrudPrimrose/dace-relayout.git"
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
