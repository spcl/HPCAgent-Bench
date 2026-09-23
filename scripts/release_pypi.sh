#!/usr/bin/env bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Build the sdist + wheel and run `twine check` on them. Dry run by default -- nothing ever
# leaves this machine unless --upload is passed explicitly:
#
#   scripts/release_pypi.sh <outdir>
#   scripts/release_pypi.sh <outdir> --upload [--repository testpypi|pypi]
#
# `build` and `twine` are NOT installed into the shared repo venv (it is never pip-installed
# into, see docs/launch.md); this script makes its own throwaway venv for them, reused across
# runs.
#
# Refuses --upload when:
#   * pyproject.toml would publish a direct-URL (`name @ git+...`) requirement -- PyPI rejects the
#     upload anyway, and failing here is a clearer error than PyPI's
#     (scripts/check_direct_url_requirements.py).
#   * the git worktree is dirty -- an upload must come from a committed tree, not a scratch edit.
#
# Credentials come from the environment (TWINE_USERNAME/TWINE_PASSWORD or TWINE_API_KEY /
# a ~/.pypirc twine already reads) -- this script never asks for or stores one.
#
# Env:
#   HPCAGENT_BENCH_PYTHON   interpreter used to create the throwaway build venv (default: python3)
#   TWINE_*                 read by `twine upload` itself; unset unless --upload is used
set -euo pipefail

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_<host>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${HPCAGENT_BENCH_PYTHON:-python3}"

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <outdir> [--upload [--repository testpypi|pypi]]" >&2
  exit 2
fi
OUTDIR="$1"
shift

UPLOAD=0
REPOSITORY="pypi"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --upload) UPLOAD=1; shift ;;
    --repository)
      [ "$#" -ge 2 ] || { echo "error: --repository needs testpypi|pypi" >&2; exit 2; }
      REPOSITORY="$2"
      shift 2
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
case "${REPOSITORY}" in
  testpypi|pypi) ;;
  *) echo "error: --repository must be testpypi or pypi, got '${REPOSITORY}'" >&2; exit 2 ;;
esac

VERSION=$("${PY}" -c "import tomllib; print(tomllib.load(open('${REPO_ROOT}/pyproject.toml','rb'))['project']['version'])")
echo "=== hpcagent_bench ${VERSION} ==="

if [ "${UPLOAD}" -eq 1 ]; then
  # PyPI rejects a published Requires-Dist that names a URL (PEP 508 direct reference); catch
  # it here rather than at PyPI's upload validator. Parsed, not grepped: pyproject's comments spell
  # the separate `pip install "dace @ git+..."` and are not requirements.
  "${PY}" "${REPO_ROOT}/scripts/check_direct_url_requirements.py" "${REPO_ROOT}/pyproject.toml" || {
    echo "error: PyPI will reject the upload" >&2
    exit 1
  }
  if [ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]; then
    echo "error: git worktree is dirty; commit before --upload" >&2
    git -C "${REPO_ROOT}" status --short >&2
    exit 1
  fi
fi

VENV="${OUTDIR}/.build-venv"
if [ ! -x "${VENV}/bin/python" ]; then
  echo "=== creating throwaway build venv at ${VENV} ==="
  "${PY}" -m venv "${VENV}"
  "${VENV}/bin/python" -m pip install --upgrade pip build twine
fi

rm -rf "${OUTDIR}/dist" "${OUTDIR}/build"
echo "=== building sdist + wheel -> ${OUTDIR}/dist ==="
"${VENV}/bin/python" -m build --outdir "${OUTDIR}/dist" "${REPO_ROOT}"

echo "=== twine check ==="
"${VENV}/bin/twine" check "${OUTDIR}"/dist/*

if [ "${UPLOAD}" -eq 1 ]; then
  echo "=== uploading to ${REPOSITORY} ==="
  "${VENV}/bin/twine" upload --repository "${REPOSITORY}" "${OUTDIR}"/dist/*
else
  echo "dry run only (pass --upload [--repository testpypi|pypi] to publish)"
fi
