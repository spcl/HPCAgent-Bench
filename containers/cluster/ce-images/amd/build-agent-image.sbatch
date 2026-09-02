#!/usr/bin/env bash
# Rebuild the agent/judge CE image from containers/cluster/ce-images/amd/Dockerfile.
#
# Runs on a COMPUTE node: the login node has no podman storage worth the space and the build pulls
# a ~30 GB ROCm base. Everything build_sqsh.sh needs comes from its own defaults; this wrapper only
# supplies the node, the wall clock and a log.
#
# The output name is DELIBERATELY not the live image. Campaign arms mount
# optarena-ce-amd-mi300-v4.sqsh while they run, and overwriting a mounted squashfs is how a live
# arm starts reading a half-written inode table. Build to a candidate, verify, then promote by
# rename when nothing is using it.
#
#   sbatch build-agent-image.sbatch
#   OUTPUT_SQSH=$SCRATCH/ce-images/optarena-ce-amd-mi300-v5.sqsh sbatch ... build-agent-image.sbatch
#
#SBATCH --job-name=build-agent-image
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=logs/build-agent-image-%j.out
#SBATCH --error=logs/build-agent-image-%j.err

set -Eeuxo pipefail

# Slurm SPOOLS the batch script, so BASH_SOURCE points at /var/spool/slurmd -- pass the directory
# rather than deriving it (this is exactly what killed build job 600467).
AMD_DIR="${AMD_DIR:?set AMD_DIR to containers/cluster/ce-images/amd}"
test -x "${AMD_DIR}/build_sqsh.sh"

export OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?}/ce-images/optarena-ce-amd-mi300-v5-candidate.sqsh}"

# Refuse to write over an image some EDF points a running job at. This used to be one hard-coded
# filename. That name was and still is the right one -- the campaign env files select
# AMD_CE_ENV=optarena-amd-mi300-v4, whose toml mounts v4.sqsh -- so this is a generalisation, not
# a repair: the guard held only while somebody remembered to edit it on the day the live image
# moved, and nothing failed if they did not. Ask the EDFs instead, since they are what a job
# actually resolves. The cost is that it also protects images nothing currently mounts, which is
# the safe direction to be wrong in.
edf_images() {
    grep -hoE '^[[:space:]]*image[[:space:]]*=[[:space:]]*"[^"]+"' "${HOME}/.edf"/*.toml 2>/dev/null \
      | sed -E 's/.*"(.*)"/\1/' | sed -E "s|\$\{SCRATCH\}|${SCRATCH:-}|g; s|\$SCRATCH|${SCRATCH:-}|g"
}
if edf_images | grep -qxF "${OUTPUT_SQSH}"; then
    echo "refusing to overwrite ${OUTPUT_SQSH}: an EDF in ${HOME}/.edf mounts it, so a running" >&2
    echo "arm would read a half-written squashfs. Build to a candidate name and promote by rename." >&2
    exit 2
fi

"${AMD_DIR}/build_sqsh.sh"

ls -lh "${OUTPUT_SQSH}"
sha256sum "${OUTPUT_SQSH}" | tee "${OUTPUT_SQSH}.sha256"
echo "AGENT IMAGE READY: ${OUTPUT_SQSH}"
