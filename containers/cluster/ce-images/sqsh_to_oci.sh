#!/usr/bin/env bash
# Re-export a squashfs image as an OCI archive that push_image.sh --from-archive can publish.
#
#   containers/cluster/ce-images/sqsh_to_oci.sh $SCRATCH/ce-images/hpcagent-bench-sglang-mi300.sqsh
#   -> $SCRATCH/ce-images/hpcagent-bench-sglang-mi300.oci.tar and .oci.tar.sha256
#
# The squashfs is the source of truth (rebuilding gives a different image), so its byte content
# becomes the archive content, cut into LAYER_BYTES slices of disjoint files (a single layer would
# exceed the registry's 10 GB limit) since a squashfs carries no original layer structure.
#
# Image config is read back from what `enroot import` wrote into the rootfs (/etc/environment,
# /etc/rc, /etc/fstab), not guessed from the Dockerfile at HEAD, so it matches what the runs
# actually used. enroot records no User/ExposedPorts/StopSignal/Healthcheck; left unset.
#
# Streams from a squashfuse mount (no unsquash); the mountpoint sits under /dev/shm since FUSE
# refuses to mount over Lustre. Temporary layer blobs go under WORK_DIR and are deleted at exit.
#
#   LAYER_BYTES      uncompressed bytes per layer before a new one starts (default 4 GiB)
#   MAX_LAYER_GB     refuse a compressed layer above this, as push_image.sh does (default 10)
#   GZIP_THREADS     pigz threads (default 16)
#   WORK_DIR         scratch work dir (default ${SCRATCH}/.tmp/sqsh-to-oci-<pid>, else beside the output)
#   MNT              squashfuse mountpoint, not on Lustre (default /dev/shm/${USER}/sqsh-to-oci-<pid>)
set -Eeuo pipefail

ulimit -c 0
SQSH="${1:?usage: sqsh_to_oci.sh <image.sqsh> [<out.oci.tar>]}"
OUT="${2:-${SQSH%.sqsh}.oci.tar}"
LAYER_BYTES="${LAYER_BYTES:-4294967296}"
MAX_LAYER_GB="${MAX_LAYER_GB:-10}"
GZIP_THREADS="${GZIP_THREADS:-16}"
WORK_DIR="${WORK_DIR:-${SCRATCH:-$(dirname -- "${OUT}")}/.tmp/sqsh-to-oci-$$}"
MNT="${MNT:-/dev/shm/${USER}/sqsh-to-oci-$$}"
LAYOUT="${WORK_DIR}/layout"

die() { echo "sqsh_to_oci: $*" >&2; exit 2; }

[[ -f "${SQSH}" ]] || die "no squashfs at ${SQSH}"
# Never over an existing artifact: an archive or checksum beside a live image may be what a
# published tag was made from.
for f in "${OUT}" "${OUT}.sha256"; do
    [[ ! -e "${f}" ]] || die "${f} exists; refusing to overwrite it"
done
for tool in squashfuse_ll fusermount pigz jq gawk sha256sum; do
    command -v "${tool}" >/dev/null || die "${tool} not found"
done

# Fresh directories this run owns, since cleanup deletes them whole.
mkdir -p "$(dirname -- "${WORK_DIR}")" "$(dirname -- "${MNT}")"
mkdir "${WORK_DIR}" || die "${WORK_DIR} exists; pick another WORK_DIR"
mkdir "${MNT}" || die "${MNT} exists; pick another MNT"
mkdir -p "${LAYOUT}/blobs/sha256" "${WORK_DIR}/lists"
cleanup() {
    fusermount -u "${MNT}" 2>/dev/null || fusermount -uz "${MNT}" 2>/dev/null || true
    rmdir "${MNT}" 2>/dev/null || true
    rm -rf "${WORK_DIR}" "${OUT}.partial"
}
trap cleanup EXIT

# The source's identity, checked against the sidecar the build wrote. A mismatch means the file is
# not the one that sidecar describes, and a label naming it would be a false provenance claim.
source_sha256() {
    local sum recorded
    sum="$(sha256sum "${SQSH}" | cut -d' ' -f1)"
    if [[ -f "${SQSH}.sha256" ]]; then
        recorded="$(awk '{print $1; exit}' "${SQSH}.sha256")"
        [[ "${recorded}" == "${sum}" ]] || die "${SQSH} hashes to ${sum}, its .sha256 says ${recorded}"
    fi
    printf '%s' "${sum}"
}

# `exec <words>` from /etc/rc to a JSON array. enroot writes the words single-quoted via ${x[@]@Q};
# anything else is refused before the eval below runs.
exec_words_json() {
    local words="$1" quoted="^([[:space:]]*'[^']*')*[[:space:]]*\$"
    local -a parsed=()
    [[ "${words}" =~ ${quoted} ]] || die "unexpected /etc/rc exec line: ${words}"
    eval "parsed=(${words})"
    # One word per line into jq, not `jq --args`, which would parse a word like -c as its own flag.
    if (( ${#parsed[@]} == 0 )); then
        printf '[]'
    else
        printf '%s\n' "${parsed[@]}" | jq -cRs 'split("\n") | .[:-1]'
    fi
}

# ELF machine of the image's /bin/sh, as an OCI architecture: the squashfs says nothing else about it.
image_arch() {
    local machine
    machine="$(od -An -t x2 -j 18 -N 2 "${MNT}/bin/sh" | tr -d ' ')"
    case "${machine}" in
        003e) printf 'amd64' ;;
        00b7) printf 'arm64' ;;
        *) die "unknown ELF machine ${machine} in /bin/sh" ;;
    esac
}

# The config enroot recorded, as a JSON object of config fields plus the provenance labels.
recorded_config() {
    local rc="${MNT}/etc/rc" env_json labels_json vols_json workdir ep_line all_line ep_json all_json cmd_json
    [[ -f "${rc}" && -f "${MNT}/etc/environment" ]] || die "no enroot /etc/rc or /etc/environment in ${SQSH}"
    env_json="$(jq -cR -s 'split("\n") | map(select(length > 0))' "${MNT}/etc/environment")"
    labels_json="$(sed -n '/^$/q; s/^# //p' "${rc}" \
        | jq -cR -s 'split("\n") | map(select(length > 0) | capture("^(?<key>[^ ]+) (?<value>.*)$")) | from_entries')"
    vols_json="$(awk 'NF >= 2 {print $2}' "${MNT}/etc/fstab" 2>/dev/null \
        | jq -cR -s 'split("\n") | map(select(length > 0) | {(.): {}}) | add // {}')"
    workdir="$(sed -n 's/^cd "\(.*\)" && unset OLDPWD.*/\1/p' "${rc}")"
    # First exec is `exec <entrypoint> "$@"`, second is `exec <entrypoint> <cmd>`.
    ep_line="$(grep -m1 -E '^[[:space:]]*exec ' "${rc}" | sed -E 's/^[[:space:]]*exec //; s/ ?"\$@"[[:space:]]*$//')"
    all_line="$(grep -E '^[[:space:]]*exec ' "${rc}" | sed -n '2{s/^[[:space:]]*exec //;p}')"
    ep_json="$(exec_words_json "${ep_line}")"
    all_json="$(exec_words_json "${all_line}")"
    cmd_json="$(jq -cn --argjson ep "${ep_json}" --argjson all "${all_json}" '$all[($ep | length):]')"
    jq -cn --argjson env "${env_json}" --argjson labels "${labels_json}" --argjson vols "${vols_json}" \
        --arg wd "${workdir}" --argjson ep "${ep_json}" --argjson cmd "${cmd_json}" \
        --arg src "${SQSH##*/}" --arg sha "${SRC_SHA256}" '
        {Env: $env,
         Labels: ($labels + {"hpcagent-bench.reexport.method": "re-exported from squashfs",
                             "hpcagent-bench.reexport.source": $src,
                             "hpcagent-bench.reexport.source.sha256": $sha})}
        + (if $wd != "" then {WorkingDir: $wd} else {} end)
        + (if ($ep | length) > 0 then {Entrypoint: $ep} else {} end)
        + (if ($cmd | length) > 0 then {Cmd: $cmd} else {} end)
        + (if ($vols | length) > 0 then {Volumes: $vols} else {} end)'
}

# Cut the tree into NUL-separated file lists, one per layer, in stable find/pre-order. Each list
# starts with the ancestors of its first entry. A hard-link group stays in one layer (the one its
# first link lands in) so it is still a hard link when stacked, and tar stores its data once.
plan_layers() {
    (cd "${MNT}" && find . -mindepth 1 -printf '%y %s %n %i %P\0') \
    | gawk -v RS='\0' -v max="${LAYER_BYTES}" -v dir="${WORK_DIR}/lists" '
        # Each path at most once per list: a layer tar that names a path twice is refused on unpack.
        function add(c, p) {
            if ((c, p) in emitted) return
            emitted[c, p] = 1
            if (c == chunk) printf "%s\0", p > out
            else late[c] = late[c] p "\0"
        }
        function add_with_ancestors(c, path,    n, parts, i, prefix) {
            n = split(path, parts, "/")
            for (i = 1; i < n; i++) {
                prefix = (i == 1) ? parts[1] : prefix "/" parts[i]
                add(c, prefix)
            }
            add(c, path)
        }
        {
            type = substr($0, 1, 1)
            split(substr($0, 3), field, " ")
            size = (type == "f") ? field[1] + 0 : 0
            path = substr($0, 3 + length(field[1] " " field[2] " " field[3] " "))
            linked = (type == "f" && field[2] > 1)
            if (linked && (field[3] in home)) {
                if (home[field[3]] != chunk) {
                    add_with_ancestors(home[field[3]], path)
                    next
                }
                size = 0
            }
            # Only data starts a new layer, so a link entry never lands apart from its data.
            if (chunk == 0 || (size > 0 && bytes > 0 && bytes + size > max)) {
                if (out != "") close(out)
                out = sprintf("%s/layer-%03d.list", dir, ++chunk)
                bytes = 0
            }
            if (linked && !(field[3] in home)) home[field[3]] = chunk
            add_with_ancestors(chunk, path)
            bytes += size
        }
        END {
            if (out != "") close(out)
            for (c in late) {
                f = sprintf("%s/layer-%03d.list", dir, c)
                printf "%s", late[c] >> f
                close(f)
            }
        }'
}

# One layer: tar the list, gzip it, and hash both streams in the same pass. diff_id is the digest
# of the uncompressed tar, digest is the digest of the blob. Prints "<diff_id> <digest> <size>".
write_layer() {
    local list="$1" raw_fifo="${WORK_DIR}/raw.fifo" gz_fifo="${WORK_DIR}/gz.fifo" tmp="${LAYOUT}/blobs/layer.tmp"
    local raw_pid gz_pid diff_id digest size
    rm -f "${raw_fifo}" "${gz_fifo}"
    mkfifo "${raw_fifo}" "${gz_fifo}"
    sha256sum < "${raw_fifo}" > "${WORK_DIR}/raw.sum" &
    raw_pid=$!
    sha256sum < "${gz_fifo}" > "${WORK_DIR}/gz.sum" &
    gz_pid=$!
    tar --create --file=- --directory="${MNT}" --format=gnu --numeric-owner --owner=0 --group=0 \
        --no-recursion --null --verbatim-files-from --no-unquote --files-from="${list}" \
        | tee "${raw_fifo}" | pigz -n -T -p "${GZIP_THREADS}" | tee "${gz_fifo}" > "${tmp}"
    wait "${raw_pid}" "${gz_pid}"
    diff_id="$(cut -d' ' -f1 "${WORK_DIR}/raw.sum")"
    digest="$(cut -d' ' -f1 "${WORK_DIR}/gz.sum")"
    size="$(stat -c %s "${tmp}")"
    if awk -v b="${size}" -v m="${MAX_LAYER_GB}" 'BEGIN { exit !(b > m * 1073741824) }'; then
        die "${list##*/} compresses to ${size} bytes, over the ${MAX_LAYER_GB} GB layer limit; lower LAYER_BYTES"
    fi
    mv -f "${tmp}" "${LAYOUT}/blobs/sha256/${digest}"
    printf '%s %s %s\n' "${diff_id}" "${digest}" "${size}"
}

# Content-addressed JSON blob; prints "<digest> <size>".
put_json_blob() {
    local file="$1" digest
    digest="$(sha256sum "${file}" | cut -d' ' -f1)"
    mv -f "${file}" "${LAYOUT}/blobs/sha256/${digest}"
    printf '%s %s\n' "${digest}" "$(stat -c %s "${LAYOUT}/blobs/sha256/${digest}")"
}

t0=${SECONDS}
echo "source   ${SQSH}"
echo "output   ${OUT}"
SRC_SHA256="$(source_sha256)"
echo "sha256   ${SRC_SHA256} ($((SECONDS - t0)) s)"

squashfuse_ll "${SQSH}" "${MNT}"
ARCH="$(image_arch)"
CONFIG_FIELDS="$(recorded_config)"
echo "config   $(jq -c '{WorkingDir, Entrypoint, Cmd, env: (.Env | length), labels: (.Labels | length)}' \
    <<< "${CONFIG_FIELDS}")"

plan_layers
mapfile -t LISTS < <(printf '%s\n' "${WORK_DIR}"/lists/layer-*.list)
echo "layers   ${#LISTS[@]} of <= ${LAYER_BYTES} uncompressed bytes"

: > "${WORK_DIR}/layers.txt"
for list in "${LISTS[@]}"; do
    t1=${SECONDS}
    layer="$(write_layer "${list}")"
    printf '%s\n' "${layer}" >> "${WORK_DIR}/layers.txt"
    read -r _ digest size <<< "${layer}"
    printf '  %s  sha256:%s  %s bytes  (%s s)\n' "${list##*/}" "${digest}" "${size}" "$((SECONDS - t1))"
done

# Created = the squashfs mtime, so re-running on the same file gives the same config digest.
CREATED="$(date -u -r "${SQSH}" +%Y-%m-%dT%H:%M:%SZ)"
jq -cn --argjson fields "${CONFIG_FIELDS}" --arg arch "${ARCH}" --arg created "${CREATED}" \
    --arg src "${SQSH##*/}" --rawfile layers "${WORK_DIR}/layers.txt" '
    ($layers | split("\n") | map(select(length > 0) | split(" "))) as $l
    | {created: $created, architecture: $arch, os: "linux", config: $fields,
       rootfs: {type: "layers", diff_ids: [$l[] | "sha256:" + .[0]]},
       history: [range(0; $l | length) as $i
                 | {created: $created,
                    created_by: "re-exported from squashfs \($src), layer \($i + 1)/\($l | length)"}]}' \
    > "${WORK_DIR}/config.json"
read -r config_digest config_size < <(put_json_blob "${WORK_DIR}/config.json")

jq -cn --arg cd "${config_digest}" --argjson cs "${config_size}" --arg created "${CREATED}" \
    --rawfile layers "${WORK_DIR}/layers.txt" '
    {schemaVersion: 2, mediaType: "application/vnd.oci.image.manifest.v1+json",
     config: {mediaType: "application/vnd.oci.image.config.v1+json", digest: ("sha256:" + $cd), size: $cs},
     layers: [$layers | split("\n")[] | select(length > 0) | split(" ")
              | {mediaType: "application/vnd.oci.image.layer.v1.tar+gzip",
                 digest: ("sha256:" + .[1]), size: (.[2] | tonumber)}],
     annotations: {"org.opencontainers.image.created": $created}}' > "${WORK_DIR}/manifest.json"
read -r manifest_digest manifest_size < <(put_json_blob "${WORK_DIR}/manifest.json")

base="${SQSH##*/}"
jq -cn --arg md "${manifest_digest}" --argjson ms "${manifest_size}" --arg ref "localhost/${base%.sqsh}:latest" '
    {schemaVersion: 2, mediaType: "application/vnd.oci.image.index.v1+json",
     manifests: [{mediaType: "application/vnd.oci.image.manifest.v1+json", digest: ("sha256:" + $md), size: $ms,
                  annotations: {"org.opencontainers.image.ref.name": $ref}}]}' > "${LAYOUT}/index.json"
printf '{"imageLayoutVersion":"1.0.0"}' > "${LAYOUT}/oci-layout"

fusermount -u "${MNT}"
# Sorted, fixed owner/mtime: same squashfs gives the same archive bytes every run.
tar --create --file="${OUT}.partial" --directory="${LAYOUT}" --format=gnu --numeric-owner --owner=0 --group=0 \
    --sort=name --mtime="${CREATED}" oci-layout index.json blobs
mv -n -T "${OUT}.partial" "${OUT}"
[[ ! -e "${OUT}.partial" ]] || die "${OUT} appeared while this ran; left it alone"
sha256sum "${OUT}" > "${OUT}.sha256"
echo "manifest sha256:${manifest_digest}"
echo "wrote    ${OUT} ($(stat -c %s "${OUT}") bytes), sha256 $(cut -d' ' -f1 "${OUT}.sha256")"
echo "elapsed  $((SECONDS - t0)) s"
