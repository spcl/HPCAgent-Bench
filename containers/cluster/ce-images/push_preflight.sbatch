#!/usr/bin/env bash
# Upload readiness for every built image, WITHOUT pushing anything.
#
#   sbatch --export=ALL,IMAGE_DIR=$PWD push_preflight.sbatch
#
# Answers the only questions that can fail an upload after it has started: is any single layer
# over the registry's 10 GB ceiling, is the image over 100 GB, and is the manifest what we intend
# to publish. Reads the .oci.tar archives, so it needs no rebuild and no credentials -- and
# deliberately runs before anyone hands credentials over.
#
# RAM, not disk: push_image.sh unpacks into a tmpfs graphroot because rootless layer extraction
# onto Lustre cannot create its pivot dir. The judge+agent image needs 60+ GB free, which is why
# this takes a whole node with --mem=0 and does the archives one at a time.
#SBATCH --job-name=push-preflight
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/push-preflight-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/push-preflight-%j.err

set -Eeuo pipefail
ulimit -c 0
IMAGE_DIR="${IMAGE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -x "${IMAGE_DIR}/push_image.sh"
CE="${CE_IMAGES:-/capstor/scratch/cscs/${USER}/x86_64/ce-images}"

rc=0
for archive in "${CE}"/*.oci.tar; do
    [[ -f "${archive}" ]] || continue
    printf '\n===== %s =====\n' "$(basename "${archive}")"
    # One at a time and cleaned between: two 60 GB unpacks in /dev/shm at once fills the node.
    if PUSH_ROOT="/dev/shm/${USER}/preflight-$$" \
       "${IMAGE_DIR}/push_image.sh" --from-archive "${archive}" --check-only; then
        :
    else
        echo "PREFLIGHT FAILED for $(basename "${archive}")" >&2
        rc=1
    fi
    rm -rf "/dev/shm/${USER}/preflight-$$" 2>/dev/null || true
done

printf '\n===== preflight verdict: %s =====\n' "$([ "${rc}" -eq 0 ] && echo READY || echo BLOCKED)"
exit "${rc}"
