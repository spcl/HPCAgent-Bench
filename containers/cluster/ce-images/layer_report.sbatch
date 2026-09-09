#!/usr/bin/env bash
# Map every image's layers to the Dockerfile step that made them, biggest first.
#
# A registry rejects a layer over its ceiling (Docker Hub: 10 GB) partway through an upload, so the
# only useful time to know a layer's size is before pushing. `podman history` gives size AND the
# command, which is what turns "layer 23.3 GB" into "that RUN".
#SBATCH --job-name=layer-report
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/layer-report-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/layer-report-%j.err
set -Eeuo pipefail
ulimit -c 0
CE="/capstor/scratch/cscs/${USER}/x86_64/ce-images"
export TMPDIR="/dev/shm/${USER}/tmp"; export XDG_RUNTIME_DIR="/dev/shm/${USER}/xdg"
mkdir -p "$TMPDIR"; mkdir -p -m 0700 "$XDG_RUNTIME_DIR"; unset DBUS_SESSION_BUS_ADDRESS

for a in "${CE}"/optarena-sglang-candidate.oci.tar "${CE}"/optarena-vllm-candidate.oci.tar; do
    [[ -f "$a" ]] || { echo "skip (absent): $a"; continue; }
    printf '\n===== %s =====\n' "$(basename "$a")"
    R="/dev/shm/${USER}/lr-$$"; mkdir -p "$R/root" "$R/run"
    P=(podman --root "$R/root" --runroot "$R/run" --storage-driver overlay
       --storage-opt ignore_chown_errors=true)
    tag="$("${P[@]}" pull "oci-archive:$a" 2>/dev/null | tail -1)" || { echo "  load failed"; continue; }
    "${P[@]}" history --no-trunc --format json "$tag" \
      | python3 -c '
import json, sys
rows = json.load(sys.stdin)
over = 0
for r in sorted(rows, key=lambda r: -(r.get("size") or 0))[:12]:
    gb = (r.get("size") or 0) / 1073741824
    if gb > 10: over += 1
    flag = "  <<< OVER 10 GB" if gb > 10 else ""
    cmd = " ".join((r.get("createdBy") or "").split())[:150]
    print(f"  {gb:7.2f} GB  {cmd}{flag}")
print(f"  layers over 10 GB: {over}")'
    podman unshare rm -rf "$R" 2>/dev/null || true
done
