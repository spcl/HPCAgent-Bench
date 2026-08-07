#!/usr/bin/env bash
set -euo pipefail

# Build a compile-only SDK from the currently installed Beverin host stack.
# The SDK is used only in the Containerfile builder stage. The final image
# contains neither libfabric nor libcxi; those must come from the CSCS host hook.

OUTPUT=${1:-beverin-host-sdk.tar.gz}
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

ROOT="$STAGE/opt/beverin-sdk"
mkdir -p "$ROOT/include" "$ROOT/lib64/pkgconfig" "$ROOT/bin" "$ROOT/manifest"

LF_PREFIX=$(readlink -f /opt/cray/libfabric/host)
LF_REAL=$(readlink -f /opt/cray/libfabric/host/lib64/libfabric.so.1)
CXI_REAL=$(readlink -f /usr/lib64/libcxi.so.1)
FI_INFO=$(readlink -f /opt/cray/libfabric/host/bin/fi_info)

for path in "$LF_PREFIX/include" "$LF_REAL" "$CXI_REAL" "$FI_INFO"; do
    test -e "$path" || { echo "Missing required host path: $path" >&2; exit 1; }
done

cp -a "$LF_PREFIX/include/." "$ROOT/include/"
cp -L "$LF_REAL" "$ROOT/lib64/libfabric.so.1.29.1"
ln -s libfabric.so.1.29.1 "$ROOT/lib64/libfabric.so.1"
ln -s libfabric.so.1 "$ROOT/lib64/libfabric.so"
cp -L "$FI_INFO" "$ROOT/bin/fi_info"
chmod 0755 "$ROOT/bin/fi_info"

# Copy the non-glibc dependency closure needed to load the real host libfabric
# during configure/link checks in the Ubuntu 24.04 builder stage.
is_core_runtime() {
    case "$(basename "$1")" in
        libc.so.*|libm.so.*|libpthread.so.*|libdl.so.*|librt.so.*|libresolv.so.*|\
        libutil.so.*|libanl.so.*|libBrokenLocale.so.*|ld-linux-*.so.*)
            return 0 ;;
        *) return 1 ;;
    esac
}

collect_deps() {
    local object=$1
    ldd "$object" | awk '
        /=> \/.*\(/ { print $3 }
        /^[[:space:]]*\/.*ld-linux/ { print $1 }
    '
}

mapfile -t DEPS < <(
    { collect_deps "$LF_REAL"; collect_deps "$CXI_REAL"; } |
        awk 'NF' | sort -u
)

for dep in "${DEPS[@]}"; do
    test -e "$dep" || continue
    is_core_runtime "$dep" && continue
    cp -L "$dep" "$ROOT/lib64/$(basename "$dep")"
done

# Ensure the direct libcxi SONAME is present even if ldd formatting changes.
cp -L "$CXI_REAL" "$ROOT/lib64/libcxi.so.1"

cat > "$ROOT/lib64/pkgconfig/libfabric.pc" <<'PC'
prefix=/opt/beverin-sdk
exec_prefix=${prefix}
libdir=${prefix}/lib64
includedir=${prefix}/include

Name: libfabric
Description: Beverin host libfabric compile-time SDK
Version: 2.3.1
Requires:
Cflags: -I${includedir}
Libs: -L${libdir} -lfabric
Libs.private: -lcxi -lcurl -ljson-c -lm -latomic -lpthread -ldl
Requires.private:
PC

{
    echo 'Beverin host SDK manifest'
    echo "created_utc=$(date -u +%FT%TZ)"
    echo "hostname=$(hostname)"
    echo "libfabric_prefix=$LF_PREFIX"
    echo "libfabric_real=$LF_REAL"
    echo "libcxi_real=$CXI_REAL"
    echo
    rpm -qf "$LF_REAL" "$CXI_REAL" || true
    echo
    /opt/cray/libfabric/host/bin/fi_info --version
    echo
    echo 'Version definitions:'
    readelf --version-info "$LF_REAL" | grep -E 'Name: FABRIC_' || true
    readelf --version-info "$CXI_REAL" | grep -E 'Name: LIBCXI_' || true
} > "$ROOT/manifest/host-abi.txt"

(
    cd "$STAGE"
    find opt/beverin-sdk -type f ! -path '*/manifest/SHA256SUMS' -print0 | sort -z | xargs -0 sha256sum
) > "$ROOT/manifest/SHA256SUMS"

tar -C "$STAGE" -czf "$OUTPUT" opt/beverin-sdk
printf 'Created %s\n' "$(readlink -f "$OUTPUT")"
tar -tzf "$OUTPUT" | sed -n '1,40p'
