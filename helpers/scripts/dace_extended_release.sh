#!/usr/bin/env bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# Patch release of DaCe extended on the ThrudPrimrose/dace fork: sync the fork's main and extended to spcl/dace's,
# then tag the extended head as the next extended-vX.Y.Z (extended-vX.Y.(Z+1); the first is extended-v1.0.0) and
# create its GitHub release with a wheel and an sdist. The fork mirrors spcl/dace exactly and holds no commits of its
# own and runs no CI (its Actions are off): the gate reads spcl/dace's own CI runs on the commit. Tags live only on the
# fork, so spcl/dace's tag and release lists stay clean. Only a head every spcl/dace workflow passed on is released
# (--allow-running also takes workflows still running, never a failed one); a head with a tag is left alone.
#   helpers/scripts/dace_extended_release.sh            # release
#   helpers/scripts/dace_extended_release.sh --dry-run  # CI gate, next tag, build + twine check, notes; change nothing
#   helpers/scripts/dace_extended_release.sh --allow-running [--dry-run]
# pins.sh --bump then moves the bench's dace pin to the newest release's commit; pins.sh --release runs this first.
set -euo pipefail
ulimit -c 0
FORK="ThrudPrimrose/dace"
UPSTREAM="spcl/dace"
dry=0
running=0
for arg in "$@"; do
    case "${arg}" in
        --dry-run) dry=1 ;;
        --allow-running) running=1 ;;
        *) echo "usage: $0 [--allow-running] [--dry-run]" >&2; exit 2 ;;
    esac
done

sha="$(git ls-remote "https://github.com/${UPSTREAM}.git" refs/heads/extended | cut -f1)"
[[ "${sha}" =~ ^[0-9a-f]{40}$ ]] || { echo "error: no extended head on ${UPSTREAM}" >&2; exit 1; }
echo "=== ${UPSTREAM}@extended head ${sha} ==="

tags="$(git ls-remote --tags "https://github.com/${FORK}.git" 'refs/tags/extended-v*' | grep -v '\^{}$' || true)"
done_tag="$(awk -v s="${sha}" '$1 == s { sub("refs/tags/", "", $2); print $2 }' <<<"${tags}")"
[[ -z "${done_tag}" ]] || { echo "${sha:0:9} is already ${FORK} ${done_tag}, nothing to release"; exit 0; }

runs="$(gh run list -R "${UPSTREAM}" -c "${sha}" -L 100 --json workflowName,status,conclusion)"
[[ "$(jq length <<<"${runs}")" -gt 0 ]] || { echo "error: no ${UPSTREAM} CI run on ${sha:0:9}" >&2; exit 1; }
open="$(jq -r --argjson running "${running}" '.[]
    | select((.status != "completed" and $running == 0)
        or (.status == "completed" and .conclusion != "success" and .conclusion != "skipped"))
    | "  \(.workflowName): \(.status) \(.conclusion)"' <<<"${runs}")"
[[ -z "${open}" ]] || { printf 'error: %s CI on %s is not green:\n%s\n' "${UPSTREAM}" "${sha:0:9}" "${open}" >&2; exit 1; }
unfinished="$(jq '[.[] | select(.status != "completed")] | length' <<<"${runs}")"
echo "CI: $(jq length <<<"${runs}") ${UPSTREAM} workflows, none failed, ${unfinished} still running"

last="$(sed -n 's#.*refs/tags/\(extended-v[0-9]*\.[0-9]*\.[0-9]*\)$#\1#p' <<<"${tags}" | sort -V | tail -1)"
if [[ -z "${last}" ]]; then
    tag="extended-v1.0.0"
    since=""
else
    IFS=. read -r major minor patch <<<"${last#extended-v}"
    tag="extended-v${major}.${minor}.$((patch + 1))"
    since="$(awk -v t="refs/tags/${last}" '$2 == t { print $1 }' <<<"${tags}")"
fi
echo "tag: ${tag} (previous: ${last:-none})"

# Wheel and sdist of exactly that commit, submodules (cub, moodycamel, webclient) included.
work="$(mktemp -d "${TMPDIR:-/tmp}/dace-extended-release.XXXXXX")"
trap 'rm -rf "${work}"' EXIT
git -C "${work}" init -q src
git -C "${work}/src" fetch -q --depth 1 "https://github.com/${UPSTREAM}.git" "${sha}"
git -C "${work}/src" checkout -q FETCH_HEAD
git -C "${work}/src" submodule update -q --init --depth 1
(cd "${work}/src" && uv build -q -o "${work}/dist")
uvx twine check --strict "${work}"/dist/*

notes="${work}/notes.md"
{
    echo "DaCe extended snapshot: [${UPSTREAM}@extended \`${sha:0:9}\`](https://github.com/${UPSTREAM}/commit/${sha})."
    if ((unfinished == 0)); then
        echo "This fork mirrors ${UPSTREAM}; the tag names that commit, and every ${UPSTREAM} CI workflow passed on it."
    else
        echo "This fork mirrors ${UPSTREAM}; the tag names that commit. No ${UPSTREAM} CI workflow failed on it; workflows still running at release time: ${unfinished}."
    fi
    echo
    echo "Install by commit, or from the wheel attached below:"
    echo
    echo '```'
    echo "pip install \"dace @ git+https://github.com/${UPSTREAM}.git@${sha}\""
    echo '```'
    if [[ -n "${since}" ]]; then
        echo
        echo "## Changes since ${last}"
        echo
        gh api "repos/${UPSTREAM}/compare/${since}...${sha}" \
            -q '.commits[] | "- " + (.commit.message | split("\n")[0]) + " (" + .sha[0:9] + ")"'
    fi
} >"${notes}"
cat "${notes}"
if ((dry)); then
    echo "dry run: nothing changed; rerun without --dry-run to sync ${FORK} and release ${tag}"
    exit 0
fi

for branch in main extended; do
    gh repo sync "${FORK}" --source "${UPSTREAM}" --branch "${branch}" --force
done
status="$(gh api "repos/${FORK}/compare/${sha}...extended" -q .status)"
[[ "${status}" == identical || "${status}" == ahead ]] || {
    echo "error: ${sha:0:9} is not on ${FORK} extended (${status})" >&2
    exit 1
}
gh release create "${tag}" -R "${FORK}" --target "${sha}" --title "DaCe extended ${tag#extended-}" \
    --notes-file "${notes}" "${work}"/dist/*
echo "released https://github.com/${FORK}/releases/tag/${tag}"
