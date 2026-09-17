#!/usr/bin/env bash
# Print the container launcher a job on this system should use: `ce` (srun --environment, pyxis)
# or `enroot` (scripts/cscs/enroot_srun.sh). A caller's explicit CONTAINER_RUNTIME always wins.
#
#   CONTAINER_RUNTIME="$(scripts/cscs/container_runtime.sh)"
#
# WHY A PROBE AND NOT A SETTING. Since the Sep 2026 scratch migration the site's
# /etc/enroot/enroot.conf names ENROOT_CACHE_PATH under /capstor, which no longer exists, and pyxis
# dies at task_init() for every `srun --environment=`. Nothing user-side reaches pyxis's environment.
# Hardcoding `enroot` into 149 arm files would outlive the fix; probing the one fact that broke
# means the stack returns to pyxis by itself the day CSCS corrects the file.
#
# The rule: pyxis is usable iff the site's ENROOT_CACHE_PATH is a directory this user can write, or
# could CREATE -- its nearest existing ancestor is writable. Not "the filesystem root exists": on
# compute nodes /capstor survives as an empty, non-writable mount point, a root-exists probe said
# `ce`, and job 640052 died in pyxis with "mkdir: cannot create directory '/capstor/scratch/cscs'".
# SITE_ENROOT_CONF overrides the file read, for tests.
set -uo pipefail

if [[ -n "${CONTAINER_RUNTIME:-}" ]]; then
    printf '%s\n' "${CONTAINER_RUNTIME}"
    exit 0
fi

conf="${SITE_ENROOT_CONF:-/etc/enroot/enroot.conf}"
# The whole rest of the line, not $2: the value holds "$(id -nu)", which has a space in it.
cache="$(awk '$1 == "ENROOT_CACHE_PATH" { sub(/^[ \t]*ENROOT_CACHE_PATH[ \t]+/, ""); sub(/[ \t]+$/, ""); print; exit }' "${conf}" 2>/dev/null)"
if [[ -z "${cache}" ]]; then
    # No site setting to be broken: whatever pyxis does, it is not this failure.
    printf 'ce\n'
    exit 0
fi
# The site writes $(id -nu) into the path; expand exactly that and nothing else.
cache="${cache//\$(id -nu)/$(id -nu)}"
probe="${cache}"
while [[ ! -e "${probe}" && "${probe}" != / ]]; do
    probe="$(dirname -- "${probe}")"
done
if [[ -d "${probe}" && -w "${probe}" ]]; then
    printf 'ce\n'
else
    echo "container_runtime: ${conf} puts ENROOT_CACHE_PATH at ${cache}, which this user cannot create" \
        "(${probe} is not a writable directory); pyxis cannot start containers -- using enroot" >&2
    printf 'enroot\n'
fi
