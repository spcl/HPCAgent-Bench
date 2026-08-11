#!/usr/bin/env bash
set -euo pipefail
set -x

LF_LINK=/opt/cray/libfabric/host/lib64/libfabric.so.1
LF_REAL=$(readlink -f "${LF_LINK}")

CXI_LINK=/usr/lib64/libcxi.so.1
CXI_REAL=$(readlink -f "${CXI_LINK}")

echo "=== resolved paths ==="
printf 'LF_LINK=%s\nLF_REAL=%s\n' "${LF_LINK}" "${LF_REAL}"
printf 'CXI_LINK=%s\nCXI_REAL=%s\n' "${CXI_LINK}" "${CXI_REAL}"

echo "=== host package identities ==="
rpm -qf "${LF_REAL}" "${CXI_REAL}" || true
sha256sum "${LF_REAL}" "${CXI_REAL}"

echo "=== libfabric ELF identity ==="
file "${LF_REAL}"
readelf -h "${LF_REAL}"
readelf -d "${LF_REAL}"
readelf --version-info "${LF_REAL}"
objdump -T "${LF_REAL}" | grep -E 'FABRIC_[0-9]' | sort -u || true
ldd "${LF_REAL}"

echo "=== libcxi ELF identity ==="
file "${CXI_REAL}"
readelf -h "${CXI_REAL}"
readelf -d "${CXI_REAL}"
readelf --version-info "${CXI_REAL}"
ldd "${CXI_REAL}"

echo "=== libfabric development files ==="
find /opt/cray/libfabric/host/include -maxdepth 3 -type f -print | sort
find /opt/cray/libfabric/host/lib64/pkgconfig -maxdepth 1 \
  -type f -print -exec cat {} \;

echo "=== CXI provider ==="
/opt/cray/libfabric/host/bin/fi_info --version
FI_PROVIDER=cxi /opt/cray/libfabric/host/bin/fi_info -p cxi

echo "=== host stack environment ==="
env | sort | grep -E \
  '^(SLINGSHOT|PMI|PMIX|FI_|CXI_|LD_LIBRARY_PATH|PATH)=' || true
