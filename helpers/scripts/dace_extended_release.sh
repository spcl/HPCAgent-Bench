#!/usr/bin/env bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# Patch release of DaCe extended as the PyPI package dace-extended (import dace), from the ThrudPrimrose/dace fork.
# The fork's main mirrors spcl/dace main exactly. Its extended is spcl/dace@extended plus ONE release commit that
# changes only the package metadata: [project] name "dace-extended", a static version X.Y.Z, and the alloy extra
# `dace-alloy>=0.1.0` (PyPI refuses direct URLs). The tag extended-vX.Y.Z names that commit, so a git pin of the tag
# and the PyPI release are the same distribution. Each run releases the spcl/dace@extended head as the next patch
# (extended-vX.Y.(Z+1); the first is extended-v1.0.0): fork sync, release commit, tag, wheel + sdist, GitHub
# release, PyPI upload. The fork runs no CI (its Actions are off): the gate reads spcl/dace's own CI runs on the head.
#   helpers/scripts/dace_extended_release.sh [--allow-running] [--no-upload] [--outdir DIR] [--dry-run]
#     --allow-running  also take spcl/dace workflows still running on the head (never a failed one)
#     --no-upload      skip the PyPI upload (TWINE_USER and TWINE_TOKEN are needed otherwise)
#     --outdir DIR     keep the wheel and sdist in DIR (default: a temporary folder)
#     --dry-run        gate, next tag, release commit, build + twine check, notes; push and upload nothing
# A head that already has a release is left alone. pins.sh --bump then moves the pins to the newest release.
set -euo pipefail
ulimit -c 0
FORK="ThrudPrimrose/dace"
UPSTREAM="spcl/dace"
die() { echo "dace_extended_release.sh: $*" >&2; exit 1; }
dry=0
running=0
upload=1
outdir=""
while (($#)); do
    case "$1" in
        --dry-run) dry=1 ;;
        --allow-running) running=1 ;;
        --no-upload) upload=0 ;;
        --outdir) outdir="$2"; shift ;;
        *) echo "usage: $0 [--allow-running] [--no-upload] [--outdir DIR] [--dry-run]" >&2; exit 2 ;;
    esac
    shift
done
((dry || !upload)) || [[ -n "${TWINE_USER:-}" && -n "${TWINE_TOKEN:-}" ]] ||
    die "TWINE_USER and TWINE_TOKEN must be set (or pass --no-upload)"

sha="$(git ls-remote "https://github.com/${UPSTREAM}.git" refs/heads/extended | cut -f1)"
[[ "${sha}" =~ ^[0-9a-f]{40}$ ]] || die "no extended head on ${UPSTREAM}"
echo "=== ${UPSTREAM}@extended head ${sha} ==="

work="$(mktemp -d "${TMPDIR:-/tmp}/dace-extended-release.XXXXXX")"
trap 'rm -rf "${work}"' EXIT
src="${work}/src"
git init -q "${src}"
git -C "${src}" remote add fork "https://github.com/${FORK}.git"
if git -C "${src}" ls-remote --exit-code --tags fork 'refs/tags/extended-v*' >/dev/null; then
    git -C "${src}" fetch -q --depth 2 fork '+refs/tags/extended-v*:refs/tags/extended-v*'
fi
# The released spcl/dace commit of each tag is its release commit's parent.
last=""
since=""
while read -r name; do
    [[ -n "${name}" ]] || continue
    parent="$(git -C "${src}" rev-parse "${name}^{commit}^")"
    [[ "${parent}" != "${sha}" ]] || { echo "${sha:0:9} is already ${FORK} ${name}, nothing to release"; exit 0; }
    last="${name}"
    since="${parent}"
done < <(git -C "${src}" tag -l 'extended-v*' | sort -V)

runs="$(gh run list -R "${UPSTREAM}" -c "${sha}" -L 100 --json workflowName,status,conclusion)"
[[ "$(jq length <<<"${runs}")" -gt 0 ]] || die "no ${UPSTREAM} CI run on ${sha:0:9}"
open="$(jq -r --argjson running "${running}" '.[]
    | select((.status != "completed" and $running == 0)
        or (.status == "completed" and .conclusion != "success" and .conclusion != "skipped"))
    | "  \(.workflowName): \(.status) \(.conclusion)"' <<<"${runs}")"
[[ -z "${open}" ]] || die "${UPSTREAM} CI on ${sha:0:9} is not green:"$'\n'"${open}"
unfinished="$(jq '[.[] | select(.status != "completed")] | length' <<<"${runs}")"
echo "CI: $(jq length <<<"${runs}") ${UPSTREAM} workflows, none failed, ${unfinished} still running"

if [[ -z "${last}" ]]; then
    version="1.0.0"
else
    IFS=. read -r major minor patch <<<"${last#extended-v}"
    version="${major}.${minor}.$((patch + 1))"
fi
tag="extended-v${version}"
echo "tag: ${tag} (previous: ${last:-none})"

# The release commit: the head plus the PyPI metadata, nothing else.
git -C "${src}" fetch -q --depth 1 "https://github.com/${UPSTREAM}.git" "${sha}"
git -C "${src}" checkout -q --detach "${sha}"
python3 - "${src}/pyproject.toml" "${version}" <<'EOF'
import pathlib
import re
import sys

path, version = pathlib.Path(sys.argv[1]), sys.argv[2]
text = path.read_text()
edits = [
    ('\n[project]\nname = "dace"\ndynamic = ["version"]\n', f'\n[project]\nname = "dace-extended"\nversion = "{version}"\n'),
    ('[tool.setuptools.dynamic]\nversion = {attr = "dace.version.__version__"}\n\n', ""),
]
for old, new in edits:
    if text.count(old) != 1:
        sys.exit(f"pyproject.toml: expected one {old.strip()!r}")
    text = text.replace(old, new)
text, count = re.subn(r'^alloy = \["alloy @ git\+[^"]*"\]$', "alloy = [\"dace-alloy>=0.1.0 ; python_version >= '3.12'\"]",
                      text, flags=re.MULTILINE)
if count != 1:
    sys.exit("pyproject.toml: expected one alloy extra with a git URL")
if re.search(r'"[^"]* @ (git\+|https?://)', text):
    sys.exit("pyproject.toml: a direct URL is left, which PyPI refuses")
path.write_text(text)
EOF
git -C "${src}" -c user.name="$(git config user.name)" -c user.email="$(git config user.email)" commit -q -am \
    "dace-extended ${version}: PyPI metadata for ${UPSTREAM}@extended ${sha:0:9}" \
    -m "Name dace-extended (import dace), version ${version}, the alloy extra as dace-alloy>=0.1.0. Nothing else changes."
release="$(git -C "${src}" rev-parse HEAD)"
git -C "${src}" submodule update -q --init --depth 1
dist="${outdir:-${work}/dist}"
mkdir -p "${dist}"
(cd "${src}" && uv build -q -o "${dist}")
uvx twine check --strict "${dist}"/dace_extended-"${version}"*

notes="${work}/notes.md"
{
    echo "dace-extended ${version}: DaCe's extended branch at [${UPSTREAM}@extended \`${sha:0:9}\`](https://github.com/${UPSTREAM}/commit/${sha}),"
    echo "on PyPI as [\`dace-extended\`](https://pypi.org/project/dace-extended/${version}/). The import name stays \`dace\`."
    echo "The release commit \`${release:0:9}\` changes only the package metadata (name, version, the alloy extra)."
    echo
    echo '```'
    echo "pip install dace-extended==${version}"
    echo '```'
    echo
    if ((unfinished == 0)); then
        echo "Every ${UPSTREAM} CI workflow passed on \`${sha:0:9}\`."
    else
        echo "No ${UPSTREAM} CI workflow failed on \`${sha:0:9}\`; workflows still running at release time: ${unfinished}."
    fi
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
    echo "dry run: nothing pushed; rerun without --dry-run to release ${tag} (${release:0:9})"
    exit 0
fi

# Both branches first take spcl/dace's heads, so the fork holds the release commit's parent before the push.
for branch in main extended; do
    gh repo sync "${FORK}" --source "${UPSTREAM}" --branch "${branch}" --force
done
git -C "${src}" push -q --force fork "${release}:refs/heads/extended"
git -C "${src}" tag -a "${tag}" -m "dace-extended ${version}" "${release}"
git -C "${src}" push -q fork "refs/tags/${tag}"
gh release create "${tag}" -R "${FORK}" --verify-tag --title "dace-extended ${version}" --notes-file "${notes}" \
    "${dist}"/dace_extended-"${version}"*
echo "released https://github.com/${FORK}/releases/tag/${tag} (${release})"
if ((upload)); then
    TWINE_USERNAME="${TWINE_USER}" TWINE_PASSWORD="${TWINE_TOKEN}" \
        uvx twine upload --non-interactive "${dist}"/dace_extended-"${version}"*
    echo "uploaded https://pypi.org/project/dace-extended/${version}/"
fi
