#!/usr/bin/env bash
set -euo pipefail

PLUGIN=${PLUGIN:-/opt/aws-ofi-nccl/lib/librccl-net.so}
FI_INFO=${FI_INFO:-/usr/local/bin/fi_info-host}
LOG_DIR=${LOG_DIR:-/tmp/beverin-loader-gate}
mkdir -p "$LOG_DIR"

section() { printf '\n===== %s =====\n' "$*"; }
fail() { echo "ERROR: $*" >&2; exit 1; }

section 'OS and libc'
cat /etc/os-release

ldd --version 2>&1 | sed -n '1p'

GLIBC_VERSION=$(getconf GNU_LIBC_VERSION)
GLIBC_VERSION=${GLIBC_VERSION##* }

printf 'detected_glibc=%s\n' "$GLIBC_VERSION"

python3 - "$GLIBC_VERSION" <<'PYGLIBC'
import re
import sys

match = re.fullmatch(r"(\d+)\.(\d+)(?:\..*)?", sys.argv[1])
if not match:
    raise SystemExit(f"could not parse glibc version: {sys.argv[1]!r}")

version = tuple(map(int, match.groups()))
if version < (2, 38):
    raise SystemExit(
        f"glibc {sys.argv[1]} is below required 2.38"
    )

print(f"glibc gate passed: {sys.argv[1]}")
PYGLIBC

section 'ROCm and PyTorch'
python3 - <<'PY'
import platform
import torch
print("python", platform.python_version())
print("torch", torch.__version__)
print("torch.version.hip", torch.version.hip)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
assert torch.cuda.is_available(), "ROCm devices are not available"
PY

section 'Plugin identity'
test -e "$PLUGIN" || fail "missing plugin: $PLUGIN"
readelf -d "$PLUGIN"
readelf --version-info "$PLUGIN" | tee "$LOG_DIR/plugin.versions.txt"
if grep -Eq 'FABRIC_1\.(9|[1-9][0-9])|FABRIC_[2-9]' "$LOG_DIR/plugin.versions.txt"; then
    fail 'plugin requires a libfabric symbol newer than host ceiling FABRIC_1.8'
fi

section 'Resolved loader closure'
ldd -r "$PLUGIN" | tee "$LOG_DIR/plugin.ldd.txt"
if grep -Eq 'not found|undefined symbol|GLIBC_[0-9.]+.*not found|FABRIC_[0-9.]+.*not found' \
     "$LOG_DIR/plugin.ldd.txt"; then
    fail 'plugin loader closure is not clean'
fi

FABRIC_PATH=$(awk '/libfabric\.so\.1 =>/ {print $3; exit}' "$LOG_DIR/plugin.ldd.txt")
CXI_PATH=$(awk '/libcxi\.so\.1 =>/ {print $3; exit}' "$LOG_DIR/plugin.ldd.txt")
printf 'resolved_libfabric=%s\nresolved_libcxi=%s\n' "$FABRIC_PATH" "$CXI_PATH"
case "$FABRIC_PATH" in
    /lib64/*|/usr/lib64/*|/opt/cray/libfabric/*) ;;
    *) fail "libfabric did not resolve from a host-hook path: $FABRIC_PATH" ;;
esac
case "$CXI_PATH" in
    /lib64/*|/usr/lib64/*|/opt/cray/*) ;;
    *) fail "libcxi did not resolve from a host-hook path: $CXI_PATH" ;;
esac

section 'CXI provider'
"$FI_INFO" --version
FI_PROVIDER=cxi "$FI_INFO" -p cxi | tee "$LOG_DIR/fi_info-cxi.txt"
for device in cxi0 cxi1 cxi2 cxi3; do
    grep -q "domain: ${device}" "$LOG_DIR/fi_info-cxi.txt" || \
        fail "fi_info did not enumerate ${device}"
done

section 'Direct plugin load'
python3 - "$PLUGIN" <<'PY'
import ctypes
import os
import sys
plugin = sys.argv[1]
ctypes.CDLL(plugin, mode=os.RTLD_NOW | os.RTLD_LOCAL)
print(f"ctypes RTLD_NOW load succeeded: {plugin}")
PY

if [[ ${CAPTURE_LD_DEBUG:-0} == 1 ]]; then
    section 'LD_DEBUG capture'
    LD_DEBUG=libs,versions \
      LD_DEBUG_OUTPUT="$LOG_DIR/ld-debug" \
      python3 -c "import ctypes, os; ctypes.CDLL('$PLUGIN', mode=os.RTLD_NOW)"
    ls -lah "$LOG_DIR"/ld-debug.*
fi

section 'Gate passed'
echo 'Host loader, CXI enumeration, and direct plugin loading are clean.'
