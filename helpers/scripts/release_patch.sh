#!/usr/bin/env bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# Patch release vX.Y.Z -> vX.Y.(Z+1) of hpcagent-bench and hpcagent-agent, in two runs from a clean main equal to
# origin/main:
#   helpers/scripts/release_patch.sh                   # 1. patch-release what the bench imports (DaCe extended,
#                                                      #    dace-relayout) and move every pin to its newest release
#                                                      #    (pins.sh --release); when a pin moved, commit and push that
#                                                      #    and stop, so CI tests the new pins
#   helpers/scripts/release_patch.sh --skip-libraries  # 2. once CI on main is green: bump the three versions, add the
#                                                      #    CHANGELOG section, commit, tag, push, build + smoke-test
#                                                      #    (do_release.sh), GitHub release; publishing the release
#                                                      #    fires .github/workflows/publish.yml, which uploads to PyPI
#   add --dry-run to either to print the plan and change nothing.
# Needs gh (authenticated) and uv; run 1 also TWINE_USER and TWINE_TOKEN (the DaCe fork has no Actions, so
# dace_extended_release.sh uploads with twine).
# The user plugins (dace-alloy, tactics) import hpcagent-bench: they release after run 2 from their own
# helpers/release_patch.sh.
set -euo pipefail
ulimit -c 0
die() { echo "release_patch.sh: $*" >&2; exit 1; }
root="$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${root}"
SLUG="spcl/HPCAgent-Bench"
VERSIONED=(pyproject.toml agent/pyproject.toml adapters/hpcagent_bench/pyproject.toml)
libraries=1
dry=0
for arg in "$@"; do
    case "${arg}" in
        --skip-libraries) libraries=0 ;;
        --dry-run) dry=1 ;;
        *) echo "usage: $0 [--skip-libraries] [--dry-run]" >&2; exit 2 ;;
    esac
done

[[ "$(git branch --show-current)" == main ]] || die "not on main"
[[ -z "$(git status --porcelain)" ]] || die "the working tree has uncommitted changes"
git fetch -q origin main --tags
[[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]] || die "main is not origin/main; pull or push first"
((dry || !libraries)) || [[ -n "${TWINE_USER:-}" && -n "${TWINE_TOKEN:-}" ]] || die "TWINE_USER and TWINE_TOKEN must be set"

if ((libraries)); then
    if ((dry)); then
        echo "dry run: would run pins.sh --release (patch releases of DaCe extended and every library, then the bump)"
        exit 0
    fi
    released=0
    helpers/scripts/pins.sh --release || released=$?
    if [[ -n "$(git status --porcelain)" ]]; then
        git add -A
        git commit -q -m "Pin the newest releases of DaCe extended, dace-relayout and the user plugins"
        git push -q origin main
        echo "pushed the new pins as $(git rev-parse --short HEAD); once CI on it is green, run $0 --skip-libraries"
    else
        echo "every pin is on its newest release; run $0 --skip-libraries"
    fi
    exit "${released}"
fi

head="$(git rev-parse HEAD)"
runs="$(gh run list -R "${SLUG}" -c "${head}" -w CI -L 20 --json status,conclusion)"
[[ "$(jq '[.[] | select(.status == "completed" and .conclusion == "success")] | length' <<<"${runs}")" -gt 0 ]] ||
    die "no green CI run on ${head:0:9}"

old="$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
[[ "${old}" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]] || die "version '${old}' is not X.Y.Z"
new="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.$((BASH_REMATCH[3] + 1))"
tag="v${new}"
previous="v${old}"
git rev-parse -q --verify "refs/tags/${tag}" >/dev/null && die "tag ${tag} exists already"
git rev-parse -q --verify "refs/tags/${previous}" >/dev/null || die "no tag ${previous} for the current version"
commits="$(git log --no-merges --format='- %s (%h)' "${previous}..HEAD")"
[[ -n "${commits}" ]] || { echo "no commits since ${previous}, nothing to release"; exit 0; }
echo "=== hpcagent-bench ${old} -> ${new} (main ${head:0:9}) ==="
echo "${commits}"
if ((dry)); then
    echo "dry run: nothing changed; rerun without --dry-run to release ${tag}"
    exit 0
fi

for file in "${VERSIONED[@]}"; do
    sed -i "0,/^version = \"${old}\"$/s//version = \"${new}\"/" "${file}"
    grep -q "^version = \"${new}\"$" "${file}" || die "could not set version ${new} in ${file}"
done
uv lock -q
section="## ${new} ($(date +%F))"$'\n\n'"${commits}"$'\n'
python3 - "${section}" <<'EOF'
import pathlib
import sys

path = pathlib.Path("CHANGELOG.md")
text = path.read_text()
at = text.find("\n## ")
at = len(text) if at < 0 else at + 1
path.write_text(text[:at] + sys.argv[1] + "\n" + text[at:])
EOF
git add -A
git commit -q -m "Release ${tag}"
git tag -a "${tag}" -m "hpcagent-bench ${new}"
git push -q origin main "${tag}"

outdir="$(mktemp -d "${TMPDIR:-/tmp}/bench-release.XXXXXX")"
helpers/scripts/do_release.sh --outdir "${outdir}"
gh release create "${tag}" -R "${SLUG}" --verify-tag --title "HPCAgent-Bench ${new}" --generate-notes \
    --notes "Patch release. Changes since ${previous}:"$'\n\n'"${commits}"
echo "released https://github.com/${SLUG}/releases/tag/${tag}; the Publish workflow uploads it to https://pypi.org/project/hpcagent-bench/${new}/"
