#!/usr/bin/env bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Build, check and smoke-test the PyPI release of hpcagent-bench; upload only when asked.
#
#   scripts/do_release.sh                          # dry run: build + twine check + wheel smoke
#   scripts/do_release.sh --upload testpypi        # same, then upload to TestPyPI
#   scripts/do_release.sh --upload pypi            # same, then upload to PyPI
#
# Options:
#   --outdir DIR    where the sdist + wheel land (default: <repo>/dist)
#   --no-smoke      skip the fresh-venv install and smoke test
#   --keep          keep the temp work dir (source export, venvs) for inspection
#
# The build input is `git archive HEAD` (tracked, committed files only), exported to a local temp
# dir, so untracked build products never reach the wheel. build/twine and the smoke venv are fresh
# venvs under that temp dir; the shared repo venv is never touched. The smoke installs the wheel
# with its dependencies from PyPI, so it needs network access.
#
# Credentials for --upload come from the environment (TWINE_USERNAME=__token__ plus
# TWINE_PASSWORD=<api token>) or ~/.pypirc; this script never asks for or stores one.
#
# On a Beverin mi200 node (the login gcc is 7.5; point PATH at a gcc >= 14 first):
#   sbatch --partition=mi200 -N1 --time=00:30:00 --no-requeue \
#       --output=release-smoke-%j.out --wrap "PATH=<gcc14-bin>:\$PATH scripts/do_release.sh"
#
# Env:
#   HPCAGENT_BENCH_PYTHON   interpreter the temp venvs are created from (default: python3, >= 3.12)
set -euo pipefail
ulimit -c 0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${HPCAGENT_BENCH_PYTHON:-python3}"
OUTDIR="${REPO_ROOT}/dist"
UPLOAD=""
SMOKE=1
KEEP=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --outdir) OUTDIR="$2"; shift 2 ;;
    --upload) UPLOAD="$2"; shift 2 ;;
    --no-smoke) SMOKE=0; shift ;;
    --keep) KEEP=1; shift ;;
    -h|--help) sed -n '5,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
case "${UPLOAD}" in
  ""|testpypi|pypi) ;;
  *) echo "error: --upload takes testpypi or pypi, got '${UPLOAD}'" >&2; exit 2 ;;
esac

"${PY}" -c 'import sys; sys.exit(sys.version_info < (3, 12))' || {
  echo "error: ${PY} is older than 3.12" >&2
  exit 1
}

VERSION=$("${PY}" -c "import tomllib; print(tomllib.load(open('${REPO_ROOT}/pyproject.toml', 'rb'))['project']['version'])")
echo "=== hpcagent-bench ${VERSION} (HEAD $(git -C "${REPO_ROOT}" rev-parse --short HEAD)) ==="

# PyPI rejects a Requires-Dist with a direct URL; fail here with a clearer message.
"${PY}" "${REPO_ROOT}/scripts/check_direct_url_requirements.py" "${REPO_ROOT}/pyproject.toml"

if [ -n "${UPLOAD}" ] && [ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=no)" ]; then
  echo "error: uncommitted changes; the release is built from HEAD, commit first" >&2
  git -C "${REPO_ROOT}" status --short --untracked-files=no >&2
  exit 1
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/hpcagent-bench-release.XXXXXX")"
cleanup() { if [ "${KEEP}" -eq 1 ]; then echo "kept ${WORK}"; else rm -rf "${WORK}"; fi; }
trap cleanup EXIT

echo "=== export HEAD -> ${WORK}/src ==="
mkdir -p "${WORK}/src"
git -C "${REPO_ROOT}" archive --format=tar HEAD | tar -x -C "${WORK}/src"

echo "=== build tools venv ==="
"${PY}" -m venv "${WORK}/tools"
"${WORK}/tools/bin/python" -m pip install --quiet --upgrade pip build twine

echo "=== build sdist + wheel -> ${OUTDIR} ==="
mkdir -p "${OUTDIR}"
rm -f "${OUTDIR}"/hpcagent_bench-*
"${WORK}/tools/bin/python" -m build --outdir "${OUTDIR}" "${WORK}/src" > "${WORK}/build.log" 2>&1 || {
  tail -40 "${WORK}/build.log" >&2
  exit 1
}
WHEEL="$(ls "${OUTDIR}"/hpcagent_bench-"${VERSION}"-*.whl)"
SDIST="$(ls "${OUTDIR}"/hpcagent_bench-"${VERSION}".tar.gz)"

echo "=== twine check ==="
"${WORK}/tools/bin/twine" check --strict "${WHEEL}" "${SDIST}"

echo "=== wheel contents ==="
"${PY}" - "${WHEEL}" "${SDIST}" <<'PY'
import os
import sys
import zipfile

BINARY = (".so", ".o", ".a", ".dll", ".dylib", ".db", ".sqlite", ".sif", ".sqsh", ".pkl", ".pt", ".bin")
LARGE = 1 << 20

wheel, sdist = sys.argv[1], sys.argv[2]
infos = zipfile.ZipFile(wheel).infolist()
total = sum(i.file_size for i in infos)
print(f"wheel  {os.path.basename(wheel)}: {os.path.getsize(wheel) / 2**20:.1f} MiB compressed, "
      f"{total / 2**20:.1f} MiB unpacked, {len(infos)} files")
print(f"sdist  {os.path.basename(sdist)}: {os.path.getsize(sdist) / 2**20:.1f} MiB")
by_top: dict[str, int] = {}
for i in infos:
    by_top[i.filename.split("/")[0]] = by_top.get(i.filename.split("/")[0], 0) + 1
print("files per top-level entry:", dict(sorted(by_top.items(), key=lambda kv: -kv[1])))
print("largest files:")
for i in sorted(infos, key=lambda i: -i.file_size)[:8]:
    print(f"  {i.file_size / 1024:8.0f} KiB  {i.filename}")
flagged = [i.filename for i in infos if i.filename.endswith(BINARY) or i.file_size > LARGE]
flagged += [i.filename for i in infos if i.filename.startswith(("tests/", "experiments/", "containers/"))]
kernel_tests = [i for i in infos if os.path.basename(i.filename).startswith("test_")]
print(f"per-kernel test files (shipped with the corpus): {len(kernel_tests)}")
if flagged:
    print("FLAGGED (binary, > 1 MiB, or repo-level tests/experiments/containers):")
    for name in flagged:
        print("  " + name)
    sys.exit(1)
PY

if [ "${SMOKE}" -eq 1 ]; then
  echo "=== smoke: fresh venv + wheel install ==="
  "${PY}" -m venv "${WORK}/smoke"
  SPY="${WORK}/smoke/bin/python"
  "${SPY}" -m pip install --quiet --upgrade pip
  "${SPY}" -m pip install --quiet "${WHEEL}" pytest
  mkdir -p "${WORK}/smoke-tests"
  # Pure tests that need the installed package data (every manifest) and nothing from the repo.
  for t in test_output_args.py test_perf_protocol.py test_distributions.py; do
    cp "${WORK}/src/tests/${t}" "${WORK}/smoke-tests/"
  done
  printf '[pytest]\nmarkers =\n    real_fuzz\nfilterwarnings =\n    error\n' > "${WORK}/smoke-tests/pytest.ini"
  cd "${WORK}/smoke-tests"
  "${SPY}" - "${VERSION}" "${WORK}/smoke" <<'PY'
import sys

import hpcagent_bench
from hpcagent_bench.translators import numpyto_c
from hpcagent_bench.translators import numpyto_common
version, venv = sys.argv[1], sys.argv[2]
for mod in (hpcagent_bench, numpyto_c, numpyto_common):
    assert mod.__file__.startswith(venv), f"{mod.__name__} imported from {mod.__file__}, not the wheel"
assert hpcagent_bench.__version__ == version, (hpcagent_bench.__version__, version)

SOURCE = """#include <stdint.h>
void scaled_add_fp64(const double *restrict x, double *restrict y, const int64_t LEN_1D, const double alpha,
                     uint8_t *restrict workspace, const int64_t workspace_size) {
    for (int64_t i = 0; i < LEN_1D; ++i) y[i] += alpha * x[i];
}
"""
score = hpcagent_bench.verify("scaled_add", SOURCE, language="c", preset="S", baseline="c")
assert score.correct, score
print(f"import + native grade OK: hpcagent_bench {hpcagent_bench.__version__}, scaled_add correct")
PY
  "${WORK}/smoke/bin/hpcagent-bench" --version
  "${WORK}/smoke/bin/hpcagent-bench" --help > /dev/null
  "${WORK}/smoke/bin/numpyto" --help > /dev/null
  "${SPY}" -m pytest -q -p no:cacheprovider .
  cd "${REPO_ROOT}"
fi

if [ -n "${UPLOAD}" ]; then
  echo "=== upload -> ${UPLOAD} ==="
  "${WORK}/tools/bin/twine" upload --repository "${UPLOAD}" "${WHEEL}" "${SDIST}"
else
  echo "dry run OK: ${WHEEL}, ${SDIST} (pass --upload testpypi|pypi to publish)"
fi
