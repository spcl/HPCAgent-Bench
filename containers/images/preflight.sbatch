#!/usr/bin/env bash
# Check, in minutes on a debug node, what build_and_verify.sbatch needs hours into a build: run it from the
# checkout root with the same environment the build gets, before submitting the build.
#
#   sbatch -p debug -A <account> containers/images/preflight.sbatch judge-agent-cuda vllm-cuda
#
# Prints one line per check and PREFLIGHT PASS or PREFLIGHT FAIL; exits 1 on any failure. It builds nothing.
#
#SBATCH --job-name=preflight-image
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out
#SBATCH --no-requeue
set -uo pipefail
ulimit -c 0

REPO="${REPO:-${SLURM_SUBMIT_DIR:?submit from the checkout root or set REPO}}"
(( $# > 0 )) || { echo "usage: preflight.sbatch <image directory under containers/images>..." >&2; exit 2; }
cd "${REPO}" || exit 2
failed=0
fail() { echo "FAIL: $*"; failed=1; }

echo "== node $(hostname): $(nproc) cores, $(awk '/^MemTotal:/ {print int($2 / 1048576)}' /proc/meminfo) GB"
[[ -f containers/images/images.env ]] || fail "${REPO} is not the checkout root"
[[ -n "${SCRATCH:-}" ]] || fail "SCRATCH is not set"
command -v uv >/dev/null && echo "uv $(uv --version)" || fail "uv is not on PATH"
command -v podman >/dev/null && echo "$(podman --version)" || fail "podman is not on PATH"

for image in "$@"; do
    echo "== ${image}"
    bash containers/images/build.sh "${image}" --roles || fail "${image}: no roles"
    # --drift exits 0 (the published image matches: a build pulls it), 1 (it would build) or 2 (an input broke).
    timeout 300 bash containers/images/build.sh "${image}" --drift
    (( $? <= 1 )) || fail "${image}: build inputs do not resolve (build.sh --drift)"
done

echo "== podman store and space"
(
    # shellcheck source=build_common.sh
    source containers/images/build_common.sh && ce_podman_env \
        && podman info --format '{{.Store.GraphRoot}} {{.Store.GraphDriverName}}' \
        && df -h "${CE_IMAGES}" /dev/shm | tail -n 2 && ce_spack_jobs
) || fail "podman environment"

echo "== network"
for url in https://github.com https://pypi.org/simple/ https://mirror.spack.io \
    https://developer.download.nvidia.com/compute/cuda/repos/ https://nvcr.io/v2/ https://registry-1.docker.io/v2/; do
    code="$(curl -s -o /dev/null -m 20 -w '%{http_code}' "${url}")"
    echo "${url} ${code}"
    [[ "${code}" =~ ^(2..|3..|401)$ ]] || fail "cannot reach ${url}"
done

if command -v nvidia-smi >/dev/null; then nvidia-smi -L || fail "GPUs not visible"; fi
if command -v rocm-smi >/dev/null; then rocm-smi --showid >/dev/null || fail "GPUs not visible"; fi

(( failed == 0 )) && { echo "PREFLIGHT PASS"; exit 0; }
echo "PREFLIGHT FAIL"
exit 1
