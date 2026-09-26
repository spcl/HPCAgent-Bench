#!/usr/bin/env bash
# THE dace refresh: every job that runs dace moves the image's /opt/dace to HPCAGENT_BENCH_DACE_REF
# at job start, inside the container, before anything imports dace.
#
#   containers/images/dace_refresh.sh             # refresh DACE_DIR (default /opt/dace), print its commit
#   containers/images/dace_refresh.sh --resolve   # print the commit HPCAGENT_BENCH_DACE_REF names now
#
# HPCAGENT_BENCH_DACE_REF is `pinned` (the default: the release's tested commit, pyproject.toml
# [tool.hpcagent-bench] dace-pin), a branch (its tip, e.g. `extended` to try a newer dace) or a full
# commit sha. A job that spans several containers resolves the ref once on the batch host (--resolve)
# and exports the sha, so every rank runs the same commit even if a branch moves.
# Writes land in the container's writable layer, so the image itself never changes.
#
# Without a checkout at DACE_DIR (bare metal, a serving image) it does nothing and exits 0, or 1
# under a pin it cannot honour.
# A branch fetch that fails keeps the baked commit (a working dace) and exits 0; a pinned commit
# that cannot be reached exits 1, since running another commit would break the pin. Concurrent
# calls in one container serialize on /opt/dace.commit. The last line, `dace-refresh: live commit
# <sha>`, is the job's dace provenance; /opt/dace.commit holds the same sha.
ulimit -c 0
DACE_DIR="${DACE_DIR:-/opt/dace}"
DACE_REF="${HPCAGENT_BENCH_DACE_REF:-pinned}"
DACE_URL="https://github.com/spcl/dace.git"
COMMIT_FILE="${DACE_DIR}.commit"
PIN_FILE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/pyproject.toml"

is_sha() { [[ "$1" =~ ^[0-9a-f]{40}$ ]]; }

if [[ "${DACE_REF}" == pinned ]]; then
    DACE_REF="$(sed -n 's/^dace-pin = "\(.*\)"$/\1/p' "${PIN_FILE}")"
    is_sha "${DACE_REF}" || { echo "dace-refresh: ${PIN_FILE} holds no commit sha ('${DACE_REF}')" >&2; exit 2; }
fi

if ! is_sha "${DACE_REF}" && ! git check-ref-format --branch "${DACE_REF}" >/dev/null 2>&1; then
    echo "dace-refresh: HPCAGENT_BENCH_DACE_REF='${DACE_REF}' is not a branch, \`pinned\` or a full commit sha" >&2
    exit 2
fi

if [[ "${1:-}" == --resolve ]]; then
    if is_sha "${DACE_REF}"; then
        echo "${DACE_REF}"
        exit 0
    fi
    sha="$(timeout 120 git ls-remote "${DACE_URL}" "refs/heads/${DACE_REF}" 2>/dev/null | cut -f1)"
    if is_sha "${sha}"; then
        echo "${sha}"
    else
        echo "dace-refresh: could not resolve ${DACE_REF}; each container fetches it itself" >&2
        echo "${DACE_REF}"
    fi
    exit 0
fi

# Bare metal, or an image without a dace checkout: the installed dace is what scripts/install_dace.sh
# put there, and nothing here can move it.
if [[ ! -d "${DACE_DIR}/.git" ]]; then
    if is_sha "${DACE_REF}"; then
        echo "dace-refresh: no dace checkout at ${DACE_DIR} to pin to ${DACE_REF}" >&2
        exit 1
    fi
    echo "dace-refresh: no dace checkout at ${DACE_DIR}; running the installed dace as it is"
    exit 0
fi

exec 9>>"${COMMIT_FILE}"
flock 9

live() { echo "dace-refresh: live commit $(git -C "${DACE_DIR}" rev-parse HEAD)"; }

baked="$(git -C "${DACE_DIR}" rev-parse HEAD)"
if [[ "${baked}" == "${DACE_REF}" ]]; then
    live
    exit 0
fi

# gitretry is the image's retrying wrapper; plain git where it is absent. The timeout bounds a job
# start on an unreachable remote.
git_cmd=(git)
command -v gitretry >/dev/null && git_cmd=(gitretry)
if ! timeout 900 "${git_cmd[@]}" -C "${DACE_DIR}" fetch -q --depth 1 origin "${DACE_REF}"; then
    if is_sha "${DACE_REF}"; then
        echo "dace-refresh: pinned commit ${DACE_REF} is unreachable; refusing to run another commit" >&2
        exit 1
    fi
    echo "dace-refresh: fetch of origin/${DACE_REF} FAILED; staying on the baked commit"
    live
    exit 0
fi

tip="$(git -C "${DACE_DIR}" rev-parse FETCH_HEAD)"
if [[ "${tip}" != "${baked}" ]]; then
    # The image's interpreter, from its EDF; an EDF rendered before the variable existed leaves it
    # empty, and the reinstall below must not silently not run.
    py="${HPCAGENT_BENCH_IMAGE_PYTHON:?dace-refresh: HPCAGENT_BENCH_IMAGE_PYTHON is unset; re-render the EDF (install_edfs.sh)}"
    git -C "${DACE_DIR}" checkout -q FETCH_HEAD
    git -C "${DACE_DIR}" submodule update --init --recursive --depth 1 -q || true
    # --no-deps: a resolver run here could move numpy underneath a running arm, silently
    # invalidating its numbers rather than failing it.
    PIP_BREAK_SYSTEM_PACKAGES=1 "${py}" -m pip install --no-cache-dir --no-deps -q -e "${DACE_DIR}"
    (cd /tmp && "${py}" -c "import dace; print('dace-refresh: import OK, dace', dace.__version__)")
    echo "dace-refresh: ${baked} -> ${tip}"
fi
git -C "${DACE_DIR}" rev-parse HEAD >"${COMMIT_FILE}"
live
