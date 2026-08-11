set -euxo pipefail

echo "=== OS / libc ==="
cat /etc/os-release
ldd --version
uname -a

echo "=== Host libfabric installation ==="
readlink -f /opt/cray/libfabric/host || true
ls -lah /opt/cray/libfabric/host || true
ls -lah /opt/cray/libfabric/host/lib64 || true
/opt/cray/libfabric/host/bin/fi_info --version || true

echo "=== Injected runtime libraries ==="
find /lib64 /usr/lib64 /opt/cray/libfabric/host \
  -maxdepth 3 \
  \( -name 'libfabric.so*' -o -name 'libcxi.so*' \) \
  -ls 2>/dev/null || true

echo "=== libfabric ABI ==="
LIBFABRIC="$(find /lib64 /usr/lib64 /opt/cray/libfabric/host \
  -name 'libfabric.so.1' -type f 2>/dev/null | head -n1)"

echo "LIBFABRIC=${LIBFABRIC}"
readelf -d "${LIBFABRIC}" || true
readelf --version-info "${LIBFABRIC}" | grep -E 'FABRIC_|GLIBC_' || true
ldd "${LIBFABRIC}" || true

echo "=== libcxi ABI ==="
LIBCXI="$(find /lib64 /usr/lib64 \
  -name 'libcxi.so.1' -type f 2>/dev/null | head -n1)"

echo "LIBCXI=${LIBCXI}"
readelf -d "${LIBCXI}" || true
readelf --version-info "${LIBCXI}" | grep -E 'GLIBC_' || true
ldd "${LIBCXI}" || true

echo "=== CXI provider enumeration ==="
command -v fi_info || true
fi_info --version || true
fi_info -p cxi || true

echo "=== Devices ==="
ls -lah /dev/cxi* /dev/kfd /dev/dri/render* 2>/dev/null || true

echo "=== Environment and loader paths ==="
env | sort | grep -E '^(LD_|FI_|CXI_|ROCR_|HSA_|HIP_|NCCL_|OFI_)' || true
cat /etc/ld.so.conf 2>/dev/null || true
find /etc/ld.so.conf.d -type f -maxdepth 1 -print -exec cat {} \; 2>/dev/null || true