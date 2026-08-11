#!/usr/bin/env bash
# Rebuild the whole inference image lineage as ONE dependency chain of Slurm jobs:
#
#   J0 phase1-build.sbatch            pack host SDK + podman build + sqsh import + EDF install
#   J1 phase1 gate                    host-loader gate + one-rank RCCL init (run-phase1-one-rank.sh)
#   J2 build-pytorch211-phase1.sbatch phase1.sqsh -> pytorch211-candidate.sqsh
#   J2b add-numpy-pytorch211.sbatch   candidate + numpy (in place, keeps a backup)
#   J3 build-vllm023-pt211.sbatch     candidate -> rocm723-vllm-0.23.0-pytorch211-ofi.sqsh
#
# Each step starts only afterok its predecessor, so one submission rebuilds everything or stops
# at the first failure. Everything lands under VLLM_BUILD_ROOT -- keep that DISTINCT from the
# directory holding the verified production copies: a rebuild has different hashes (timestamps),
# so the checked-in sha256 pins verify provenance of the ORIGINALS, not build reproducibility.
#
#   VLLM_BUILD_ROOT=$SCRATCH/vllm-mi300-rebuild ./build-chain.sh --account=a-g200
#
# Extra arguments are passed to every sbatch verbatim.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ROOT="${VLLM_BUILD_ROOT:?set VLLM_BUILD_ROOT (e.g. \$SCRATCH/vllm-mi300-rebuild)}"

# ROOT is a SELF-CONTAINED work tree in the vllm-mi300-3 layout the upstream sbatch scripts
# assume: they resolve their inner scripts as ${ROOT}/build/*-inner.sh and their artifacts as
# ${ROOT}/phase1-passed / ${ROOT}/containers. Seed it from the repo, then submit everything FROM
# it -- also required because Slurm spools batch scripts, so nothing may rely on BASH_SOURCE.
mkdir -p "${ROOT}"
rsync -a --exclude archive --exclude logs "${INFERENCE_DIR}/" "${ROOT}/"
mkdir -p "${ROOT}/logs" "${ROOT}/phase1-passed" "${ROOT}/containers"

submit() {
    # submit <dependency-jobid-or-empty> <sbatch args...> -> prints the new job id
    local dep="$1"
    shift
    local -a args=(--parsable --export="ALL,VLLM_BUILD_ROOT=${ROOT}")
    if [[ -n "${dep}" ]]; then
        args+=(--dependency="afterok:${dep}")
    fi
    sbatch "${args[@]}" "$@"
}

# Submit from ROOT: the upstream #SBATCH --output=logs/... paths are relative to the submit cwd.
cd "${ROOT}"

j0="$(submit "" "$@" "${ROOT}/build/phase1-build.sbatch")"

# The gate is its own job so a build failure and an interconnect failure are distinguishable.
# ENVIRONMENT is set explicitly: Cray batch sessions export ENVIRONMENT=BATCH, which would
# override the gate script's default EDF name.
j1="$(submit "${j0}" "$@" --job-name=phase1-gate --partition=mi300 --nodes=1 --time=00:20:00 \
    --output="${ROOT}/logs/phase1-gate-%j.out" --error="${ROOT}/logs/phase1-gate-%j.err" \
    --wrap="cd '${ROOT}' && ENVIRONMENT=rocm723-ofi-host-diag VLLM_BUILD_ROOT='${ROOT}' \
        bash beverin-rocm723-host-ofi-phase1/run-phase1-one-rank.sh")"

j2="$(submit "${j1}" "$@" "${ROOT}/build/build-pytorch211-phase1.sbatch")"
j2b="$(submit "${j2}" "$@" "${ROOT}/build/add-numpy-pytorch211.sbatch")"
j3="$(submit "${j2b}" "$@" "${ROOT}/build/build-vllm023-pt211.sbatch")"

cat <<EOF
submitted chain:
  J0  phase1 build   ${j0}
  J1  phase1 gate    ${j1}
  J2  pytorch 2.11   ${j2}
  J2b add numpy      ${j2b}
  J3  vLLM 0.23.0    ${j3}
watch:  squeue --me --format='%.10i %.16j %.8T %.10M %R'
result: ${ROOT}/containers/rocm723-vllm-0.23.0-pytorch211-ofi.sqsh
EOF
