#!/usr/bin/env bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# Print the commit this release pins (pyproject.toml [tool.uv.sources] rev) of dace (the default) or relayout: the one
# commit every image bakes and every job runs.
#   pins.sh [dace|relayout]  print the pinned commit
#   pins.sh --bump           move every pin to its newest release and relock dace and dace-relayout:
#                            dace to the commit of the newest extended-vX.Y.Z on the ThrudPrimrose/dace fork;
#                            dace-relayout to its newest vX.Y.Z tag, the commit its PyPI release was built from (the
#                            [tool.uv.sources] rev and the dace-relayout>= floor); the user plugins to their
#                            repositories' newest vX.Y.Z tags (user_plugins.py bump)
#   pins.sh --release        first patch-release DaCe extended (dace_extended_release.sh) and every library below (its
#                            helpers/release_patch.sh, run in a fresh clone of main: tag, GitHub release, PyPI upload;
#                            needs TWINE_USER and TWINE_TOKEN), then --bump. A failed release does not stop the others.
set -euo pipefail
ulimit -c 0
root="$(dirname -- "${BASH_SOURCE[0]}")/../.."
# relayout is published as dace-relayout: its [tool.uv.sources] key and uv package name.
declare -A key=([dace]="dace" [relayout]="dace-relayout")
declare -A repo=([dace]="https://github.com/spcl/dace.git" [relayout]="https://github.com/ThrudPrimrose/dace-relayout.git")
DACE_FORK="https://github.com/ThrudPrimrose/dace.git"
LIBRARIES=(
    "https://github.com/ThrudPrimrose/dace-relayout.git"
    "https://github.com/ThrudPrimrose/dace-alloy.git"
    "https://github.com/spcl/tactics.git"
)

pinned() {
    local url="${repo[$1]//\//\\/}"
    sed -n "s/^${key[$1]} = { git = \"${url}\", rev = \"\([0-9a-f]\{40\}\)\" }$/\1/p" "${root}/pyproject.toml"
}

# "X.Y.Z commit" of the newest <prefix>X.Y.Z tag of a repository (an annotated tag's commit, not the tag object).
newest_release() {
    git ls-remote --tags "$1" "refs/tags/$2*" | awk -v prefix="refs/tags/$2" '
        index($2, prefix) == 1 {
            name = $2
            peeled = sub(/\^\{\}$/, "", name)
            version = substr(name, length(prefix) + 1)
            if (version ~ /^[0-9]+\.[0-9]+\.[0-9]+$/ && (peeled || !(version in commit))) commit[version] = $1
        }
        END { for (version in commit) print version, commit[version] }' | sort -V | tail -1
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
    local dace_version dace_commit relayout_version relayout_commit before plugins_before
    read -r dace_version dace_commit <<<"$(newest_release "${DACE_FORK}" extended-v)"
    read -r relayout_version relayout_commit <<<"$(newest_release "${repo[relayout]}" v)"
    [[ -n "${dace_commit:-}" ]] || { echo "pins.sh: ${DACE_FORK} has no extended-vX.Y.Z release" >&2; exit 2; }
    [[ -n "${relayout_commit:-}" ]] || { echo "pins.sh: ${repo[relayout]} has no vX.Y.Z release" >&2; exit 2; }
    # A failed relock puts both files back, so pyproject.toml never disagrees with uv.lock beside it.
    before="$(cat "${root}/pyproject.toml"; echo x)"
    plugins_before="$(cat "${root}/.github/user-plugins.toml"; echo x)"
    # The rev in [tool.uv.sources] and dace's override-dependencies URL (which wins over a plugin's pin) move together.
    sed -i "s/rev = \"$(pinned dace)\"/rev = \"${dace_commit}\"/; s/@$(pinned dace)\"/@${dace_commit}\"/" \
        "${root}/pyproject.toml"
    sed -i "s/rev = \"$(pinned relayout)\"/rev = \"${relayout_commit}\"/; s/\"dace-relayout>=[0-9.]*\"/\"dace-relayout>=${relayout_version}\"/" \
        "${root}/pyproject.toml"
    echo "dace: extended-v${dace_version} ${dace_commit}"
    echo "dace-relayout: v${relayout_version} ${relayout_commit}"
    if ! (cd "${root}" && uv run --frozen --no-sync python helpers/scripts/user_plugins.py bump &&
        uv lock -q --upgrade-package dace --upgrade-package dace-relayout); then
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
[[ -n "${repo[${package}]:-}" ]] || { echo "pins.sh: unknown package '${package}' (dace or relayout)" >&2; exit 2; }
pin="$(pinned "${package}")"
[[ -n "${pin}" ]] || { echo "pins.sh: pyproject.toml holds no 40-character ${package} rev" >&2; exit 2; }
echo "${pin}"
