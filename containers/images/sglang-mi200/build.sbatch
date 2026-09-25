#!/usr/bin/env bash
# Build the SGLang MI250X (gfx90a) image on an mi200 node.
#
#   sbatch --chdir=containers/images/sglang-mi200 build.sbatch                  (build only)
#   REPO=$PWD IMAGE_DIR=containers/images/sglang-mi200 sbatch --partition=mi200 \
#     --cpus-per-task=64 --gpus-per-node=8 containers/images/build_and_verify.sbatch
#
# mi200 only, checked in the body: build_and_verify.sbatch runs this body inside its own allocation,
# where the directive below does not apply, and a gfx942 node cannot run the verifier's launch check.
#SBATCH --job-name=build-sglang-mi200-image
#SBATCH --partition=mi200
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 64 PHYSICAL cores: EPYC 7A53, 128 logical CPUs with SMT.
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --no-requeue

set -Eeuxo pipefail
ulimit -c 0

if [[ "${SLURM_JOB_PARTITION:-}" != mi200 ]]; then
    echo "refusing: partition '${SLURM_JOB_PARTITION:-none}' is not mi200" >&2
    exit 2
fi

# Slurm spools the batch script, so BASH_SOURCE is useless here: --chdir or IMAGE_DIR names the dir.
IMAGE_DIR="${IMAGE_DIR:-$PWD}"
test -x "${IMAGE_DIR}/build.sh" || { echo "no build.sh under IMAGE_DIR=${IMAGE_DIR}" >&2; exit 2; }

# shellcheck source=../build_common.sh
source "${IMAGE_DIR}/../build_common.sh"
# shellcheck source=../images.env
source "${IMAGE_DIR}/../images.env"
ce_require_partition sglang-mi200 || exit 2
export OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?}/ce-images/${INFERENCE_SGLANG_MI200_CANDIDATE}}"

ce_refuse_mounted "${OUTPUT_SQSH}" || exit 2

"${IMAGE_DIR}/build.sh"

ls -lh "${OUTPUT_SQSH}"
sha256sum "${OUTPUT_SQSH}" | tee "${OUTPUT_SQSH}.sha256"
echo "SGLANG MI200 IMAGE READY: ${OUTPUT_SQSH}"
echo "NOT VERIFIED: run build_and_verify.sbatch (kernel launch check) and PRESET=mi200 inference/serve-private.sbatch."
