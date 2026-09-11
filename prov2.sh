set -u
echo "--- spack libfabric (the one MPI resolves to) : does it have CXI?"
sp="$(ls -d /opt/spack-install/linux-zen4/libfabric-*/lib 2>/dev/null | head -1)"
echo "  dir: ${sp:-none}"
[ -n "${sp}" ] && { ls "${sp}"/../ 2>/dev/null | head; strings "${sp}/libfabric.so.1" 2>/dev/null | grep -oiE '^(cxi|verbs|tcp|sockets|shm|udp|psm3)$' | sort -u | tr '\n' ' '; echo; }
[ -n "${sp}" ] && { echo "  provider symbols:"; strings "${sp}/libfabric.so.1" 2>/dev/null | grep -icE 'cxi' | sed 's/^/    cxi mentions: /'; }
echo
echo "--- host libfabric injected by the hook : does IT have CXI?"
ls -la /usr/lib64/libfabric.so.1 2>/dev/null
strings /usr/lib64/libfabric.so.1 2>/dev/null | grep -icE 'cxi' | sed 's/^/    cxi mentions: /'
echo "    version symbols:"
readelf --version-info /usr/lib64/libfabric.so.1 2>/dev/null | grep -oE 'FABRIC_[0-9.]+' | sort -u | tr '\n' ' '; echo
echo
echo "--- does libcxi exist anywhere in this image?"
find / -name 'libcxi.so*' -not -path '/proc/*' 2>/dev/null | head
echo
echo "--- what would the RCCL plugin resolve libfabric.so.1 to? (LD_LIBRARY_PATH then default)"
echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
for d in $(echo "${LD_LIBRARY_PATH:-}" | tr ':' ' ') /usr/lib64 /usr/lib/x86_64-linux-gnu; do
    [ -e "${d}/libfabric.so.1" ] && { echo "  FIRST MATCH: ${d}/libfabric.so.1"; break; }
done
