#!/usr/bin/env bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# Print the commit this release pins (pyproject.toml [tool.uv.sources] rev) of dace (the default) or relayout: the one
# commit every image bakes and every job runs. `--bump` first moves BOTH pins to their branch heads (spcl/dace@extended,
# relayout@main) and relocks those two packages only.
#   pins.sh [--bump] [dace|relayout]
set -euo pipefail
ulimit -c 0
root="$(dirname -- "${BASH_SOURCE[0]}")/../.."
declare -A repo=([dace]="https://github.com/spcl/dace.git" [relayout]="https://github.com/ThrudPrimrose/relayout.git")
declare -A branch=([dace]="extended" [relayout]="main")

pinned() {
    local url="${repo[$1]//\//\\/}"
    sed -n "s/^$1 = { git = \"${url}\", rev = \"\([0-9a-f]\{40\}\)\" }$/\1/p" "${root}/pyproject.toml"
}

bump=0
[[ "${1:-}" == "--bump" ]] && { bump=1; shift; }
package="${1:-dace}"
[[ -n "${repo[${package}]:-}" ]] || { echo "pins.sh: unknown package '${package}' (dace or relayout)" >&2; exit 2; }
for name in dace relayout; do
    pin="$(pinned "${name}")"
    [[ -n "${pin}" ]] || { echo "pins.sh: pyproject.toml holds no 40-character ${name} rev" >&2; exit 2; }
    if ((bump)); then
        new="$(git ls-remote "${repo[${name}]}" "refs/heads/${branch[${name}]}" | cut -f1)"
        [[ "${new}" =~ ^[0-9a-f]{40}$ ]] || { echo "pins.sh: no ${branch[${name}]} head from ${repo[${name}]}" >&2; exit 2; }
        # The rev in [tool.uv.sources] and dace's override-dependencies URL (which wins over a plugin's pin) move together.
        sed -i "s/rev = \"${pin}\"/rev = \"${new}\"/; s/@${pin}\"/@${new}\"/" "${root}/pyproject.toml"
    fi
done
((bump)) && (cd "${root}" && uv lock -q --upgrade-package dace --upgrade-package relayout)
pinned "${package}"
