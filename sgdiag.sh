set -u
echo "--- OS / glibc"
. /etc/os-release 2>/dev/null && echo "  ${PRETTY_NAME:-unknown}"
ldd --version 2>/dev/null | head -1
echo
echo "--- what is UNRESOLVED for the in-image libfabric?"
ldd /opt/ofi/lib/libfabric.so.1 2>&1 | grep -E "not found" || echo "  (nothing missing)"
echo
echo "--- and for libcxi / the plugin?"
ldd /opt/ofi/lib/libcxi.so.1 2>&1 | grep -E "not found" || echo "  libcxi: ok"
ldd /opt/ofi/lib/librccl-net.so 2>&1 | grep -E "not found" || echo "  plugin: ok"
echo
echo "--- who normally provides libldap_r-2.4.so.2 here?"
for p in libldap-2.4-2 libldap-common libldap-2.5-0; do
  dpkg -s "$p" >/dev/null 2>&1 && echo "  installed: $p"
done
apt-cache search --names-only 'libldap' 2>/dev/null | head -5 || echo "  (no apt cache)"
echo
echo "--- does anything else in the image already ship it?"
find / -name 'libldap*' -not -path '/proc/*' 2>/dev/null | head -8
echo
echo "--- what pulled it in: does libfabric link curl?"
ldd /opt/ofi/lib/libfabric.so.1 2>&1 | grep -iE "curl|ldap|sasl|gnutls" || echo "  no curl/ldap direct"
