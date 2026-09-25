#!/usr/bin/env bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# A node-local WRITE layer over a shared JIT cache (triton, inductor, vLLM compile cache).
#
#   jit_cache_layer.sh seed    <shared> <local>   copy what the shared cache has into <local>
#   jit_cache_layer.sh publish <local> <shared>   add what <local> compiled to the shared cache
#
# WHY. The shared cache lives on ${SCRATCH}, which is NFS on beverin. Several engines compiling at
# the same moment each rewrite the same content-addressed files there, and NFS turns "a file another
# client just replaced" into ESTALE for the reader (`OSError: [Errno 116] Stale file handle` in
# inductor's autotune: one TP worker down, the engine hung). So an engine never writes the shared tree while it runs: it
# compiles into a node-local copy, and publishes what it added once it is serving.
#
# PUBLISH IS ADD-ONLY AND ATOMIC PER ENTRY. A directory the shared cache lacks is copied under a
# staging name beside its destination and renamed into place, so a reader sees all of it or none
# of it (a triton kernel directory without its binary would otherwise be read as a hit). Inside a
# directory both sides already have, each missing file is staged and renamed the same way. Nothing
# that exists is ever overwritten: entries are content-addressed, so an existing one is either the
# same bytes or another engine's equally valid result. Staging names start with STAGE_PREFIX and
# seed skips them, so a publisher killed mid-copy leaves nothing a later seed would pick up.
set -uo pipefail

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
STAGE_PREFIX=".jit-layer-staging"

# vLLM's compile cache records the ABSOLUTE path of every inductor artifact inside
# vllm_compile_cache.py. An entry another job compiled under a node-local root that no longer
# exists is therefore not a cache MISS but a cache TRAP: the engine reads the manifest, opens the
# recorded path and dies at startup with FileNotFoundError on artifact_compile_range_*, taking the
# whole arm with it. Such an entry is dropped
# after seeding so the engine simply recompiles it.
scrub_dead_entries() {
    local root="$1" manifest dir path
    [[ -d "${root}" ]] || return 0
    while IFS= read -r -d '' manifest; do
        dir="$(dirname -- "${manifest}")"
        while IFS= read -r path; do
            [[ -e "${path}" ]] && continue
            rm -rf -- "${dir}"
            break
        done < <(grep -o "'/[^']*'" "${manifest}" 2>/dev/null | tr -d "'")
    done < <(find "${root}" -name vllm_compile_cache.py -print0 2>/dev/null)
    return 0
}

seed() {
    local shared="$1" local_dir="$2"
    mkdir -p "${local_dir}" || return 1
    [[ -d "${shared}" ]] || return 0
    # -n: a local entry (a previous seed on this node) is never replaced by a shared one mid-run.
    (cd "${shared}" && find . -mindepth 1 -maxdepth 1 ! -name "${STAGE_PREFIX}*" -print0) |
        while IFS= read -r -d '' entry; do
            cp -an -- "${shared}/${entry#./}" "${local_dir}/" 2>/dev/null || true
        done
    scrub_dead_entries "${local_dir}"
    return 0
}

# stage_rename <src> <dst>: copy <src> next to <dst> under a staging name, then rename it into place
# unless <dst> appeared meanwhile.
stage_rename() {
    local src="$1" dst="$2" stage
    stage="$(dirname -- "${dst}")/${STAGE_PREFIX}.$(hostname -s).$$.$(basename -- "${dst}")"
    cp -a -- "${src}" "${stage}" 2>/dev/null || { rm -rf -- "${stage}"; return 0; }
    if [[ -e "${dst}" ]]; then
        rm -rf -- "${stage}"
    else
        mv -nT -- "${stage}" "${dst}" 2>/dev/null
        rm -rf -- "${stage}" 2>/dev/null
    fi
    return 0
}

publish_dir() {  # publish_dir <local-dir> <shared-dir>, both existing
    local src="$1" dst="$2" entry name
    for entry in "${src}"/* "${src}"/.[!.]*; do
        [[ -e "${entry}" ]] || continue
        name="$(basename -- "${entry}")"
        [[ "${name}" == "${STAGE_PREFIX}"* ]] && continue
        if [[ -d "${entry}" && ! -L "${entry}" ]]; then
            if [[ -d "${dst}/${name}" ]]; then
                publish_dir "${entry}" "${dst}/${name}"
            elif [[ ! -e "${dst}/${name}" ]]; then
                stage_rename "${entry}" "${dst}/${name}"
            fi
        elif [[ ! -e "${dst}/${name}" ]]; then
            stage_rename "${entry}" "${dst}/${name}"
        fi
    done
}

publish() {
    local local_dir="$1" shared="$2"
    [[ -d "${local_dir}" ]] || return 0
    mkdir -p "${shared}" || return 1
    publish_dir "${local_dir}" "${shared}"
}

case "${1:-}" in
    seed) seed "${2:?shared}" "${3:?local}" ;;
    publish) publish "${2:?local}" "${3:?shared}" ;;
    *) echo "usage: $0 seed <shared> <local> | publish <local> <shared>" >&2; exit 2 ;;
esac
