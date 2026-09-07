#!/usr/bin/env bash
# Record what each built image actually IS, in one file, outside the image.
#
# The pieces already exist and are scattered: build.sh writes <sqsh>.digest and <sqsh>.sha256,
# ce_export_image writes <sqsh>.oci.tar, and the Dockerfile bakes toolchain and library versions
# into /usr/local/share/image-provenance -- INSIDE the image, where nothing can read it without
# mounting it. So the question "what is on scratch, and can it be published" had no single answer.
#
# It also answers the publishing question specifically. An image is uploadable only if its OCI
# archive exists: the squashfs is a flattened filesystem, so an image whose archive is missing has
# to be REBUILT to be pushed, and this is where that shows up as a row saying so.
#
#   ./snapshot_images.sh                 # sidecars only, no jobs, seconds
#   ./snapshot_images.sh --deep          # also reads provenance from inside each image (one srun each)
#
# Writes to ${SCRATCH}/ce-images/SNAPSHOT.txt by default. Deliberately NOT into the repo: it
# describes what is on this filesystem right now, which is not a fact about the source tree.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${SCRATCH:?set SCRATCH}"
CE_IMAGES="${CE_IMAGES:-${SCRATCH}/ce-images}"
OUT="${OUT:-${CE_IMAGES}/SNAPSHOT.txt}"
DEEP=0
[[ "${1:-}" == "--deep" ]] && DEEP=1

sidecar() { [[ -s "$1" ]] && tr -d '\n' < "$1" || printf 'none'; }

# A sidecar left over from a PREVIOUS build of the same name reads as valid to anything that just
# cats it. That is how job 620068 reported an image it had not built, and it happened again here:
# a build failed after writing the squashfs but before its checksum, leaving a four-day-old sha256
# beside a fresh image.
#
# Older is NOT the test, though -- ce_export_image writes the digest, THEN imports the squashfs,
# so a correct .digest is always a little older than the .sqsh. Measured across four images, that
# gap is 101-133 s (the enroot import), while the real staleness was 4 days. So flag only a gap
# large enough that no single build could produce it; anything under the threshold is the normal
# write order, and flagging it made every correctly built image look broken.
STALE_AFTER_S="${STALE_AFTER_S:-21600}"   # 6 h: ~160x the observed within-build gap
stale_note() {
    local sidecar="$1" sqsh="$2" gap
    [[ -s "${sidecar}" && -e "${sqsh}" ]] || return 0
    gap=$(( $(stat -c %Y "${sqsh}") - $(stat -c %Y "${sidecar}") ))
    (( gap > STALE_AFTER_S )) \
      && printf '  <- STALE by %sh: belongs to an earlier build' "$(( gap / 3600 ))"
    return 0
}
human()   { [[ -e "$1" ]] && du -h --apparent-size "$1" 2>/dev/null | cut -f1 || printf '-'; }

{
    printf 'CE image snapshot\n'
    printf 'taken     %s\n' "$(date -u '+%Y-%m-%d %H:%M:%SZ')"
    printf 'host      %s\n' "$(hostname)"
    printf 'tree      %s @ %s\n' "$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || echo '?')" \
                                 "$(cd "${SCRIPT_DIR}" && git rev-parse --short HEAD 2>/dev/null || echo '?')"
    printf 'directory %s\n\n' "${CE_IMAGES}"

    for sqsh in "${CE_IMAGES}"/*.sqsh; do
        [[ -e "${sqsh}" ]] || continue
        base="${sqsh%.sqsh}"
        archive="${base}.oci.tar"
        printf '%s\n' "$(basename "${sqsh}")"
        printf '  size        %s\n' "$(human "${sqsh}")"
        printf '  built       %s\n' "$(date -u -r "${sqsh}" '+%Y-%m-%d %H:%M:%SZ' 2>/dev/null || echo '?')"
        printf '  digest      %s%s\n' "$(sidecar "${sqsh}.digest")" "$(stale_note "${sqsh}.digest" "${sqsh}")"
        printf '  sha256      %s%s\n' "$(sidecar "${sqsh}.sha256" | awk '{print $1}')" "$(stale_note "${sqsh}.sha256" "${sqsh}")"
        if [[ -e "${archive}" ]]; then
            printf '  publishable YES -- push_image.sh --from-archive %s\n' "${archive}"
            printf '  archive     %s\n' "$(human "${archive}")"
        else
            printf '  publishable NO -- no OCI archive; publishing this build needs a REBUILD\n'
        fi
        # An EDF naming it is what makes an image live rather than a candidate.
        # `|| true`: grep exits 1 when an image is mounted by nothing, and under `set -e` with
        # pipefail that ends the snapshot at the first unreferenced candidate -- which is most of
        # them. A candidate with no EDF is the normal case here, not an error.
        edfs="$(grep -lE "image *= *\"[^\"]*$(basename "${sqsh}")\"" "${HOME}"/.edf/*.toml 2>/dev/null \
                | xargs -r -n1 basename | sed 's/\.toml$//' | tr '\n' ' ' || true)"
        printf '  mounted by  %s\n' "${edfs:-nothing (candidate)}"

        if (( DEEP )); then
            edf="${CE_IMAGES}/.tmp-snapshot-$$.toml"
            mkdir -p "$(dirname "${edf}")"
            printf 'image = "%s"\nmounts = ["/capstor/:/capstor/"]\nworkdir = "/"\n' "${sqsh}" > "${edf}"
            prov="$(srun --partition=mi300 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=0 \
                        --time=00:05:00 --environment="${edf}" \
                        cat /usr/local/share/image-provenance 2>/dev/null || true)"
            rm -f "${edf}"
            if [[ -n "${prov}" ]]; then
                printf '  provenance\n'; printf '%s\n' "${prov}" | sed 's/^/    /'
            else
                printf '  provenance  none (image carries no /usr/local/share/image-provenance)\n'
            fi
        fi
        printf '\n'
    done
} > "${OUT}"

cat "${OUT}"
printf 'wrote %s\n' "${OUT}"
