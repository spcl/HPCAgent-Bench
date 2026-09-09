#!/usr/bin/env bash
# WHICH libfabric does MPI actually load, and can RCCL's plugin load the same one?
#
# This is the question behind "make RCCL use whatever MPI is using". MPICH here is ch4:ofi and
# cross-node device-pointer allreduce WORKS (629802), so a libfabric with the CXI provider is
# already present and functioning at run time. It is not the one in the image: this image ships a
# COMPILE-ONLY libfabric (providers disabled) purely to link against. So something injects the
# real one, and this job proves what, from where, and whether the plugin resolves to it too.
#
# Everything here is read-only inspection. No collective, no allocation beyond one node.
#
#SBATCH --job-name=ofi-provenance
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/ofi-prov-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/ofi-prov-%j.err

set -Eeuo pipefail
SCRATCH="${SCRATCH:-/capstor/scratch/cscs/${USER}/x86_64}"
REPO="${REPO:-${SCRATCH}/optarena}"
IMAGE="${IMAGE:-${SCRATCH}/ce-images/optarena-ce-amd-mi300.sqsh}"
HOOKS="${HOOKS:-artifact}"     # artifact (production) | none (control)
EDF="${SCRATCH}/.tmp/ofiprov-${SLURM_JOB_ID}.toml"
mkdir -p "$(dirname "${EDF}")" "${SCRATCH}/ce-images/logs"

{
  echo "image = \"${IMAGE}\""
  echo 'mounts = ["'"${REPO}"':'"${REPO}"'"]'
  echo "workdir = \"${REPO}\""
  echo
  echo '[annotations]'
  # The five-annotation block the production EDFs carry, verbatim. Anything less is not the
  # configuration under test: netstack.source="host" is the OPT-OUT that grafts host paths in and
  # is how a host libcurl needing glibc 2.38 reached a 2.35 image (629822), so it is not offered.
  case "${HOOKS}" in
      artifact) echo 'com.hooks.netstack.source = "artifact"'
                echo 'com.hooks.netstack.version = "26.08.1"'
                echo 'com.hooks.netstack.name = "gpu_rocm7-cxi_13.1.0-ofi_2.6.0-aws_1.20.0"'
                echo 'com.hooks.cxi.enabled = "true"'
                echo 'com.hooks.aws_ofi_nccl.enabled = "true"' ;;
      none)     echo '# no hooks -- control case: shows what the IMAGE alone ships' ;;
      *)        echo "unknown HOOKS=${HOOKS}" >&2; exit 2 ;;
  esac
  echo
  echo '[env]'
  echo 'PATH = "/opt/gcc/bin:/opt/view/bin:/opt/rocm/bin:/opt/venv/bin:/opt/cscs/netstack/bin:/usr/local/bin:/usr/bin:/bin"'
  # No /opt/aws-ofi-nccl/lib: that directory is gone. libfabric, libcxi and librccl-net all come
  # from the pinned netstack artifact the hooks install under /opt/cscs/netstack.
  echo 'LD_LIBRARY_PATH = "/opt/cscs/netstack/lib:/opt/view/lib:/opt/view/lib64:/opt/rocm/lib:/opt/rocm/lib64"'
} > "${EDF}"

echo "=== image ${IMAGE}   hooks=${HOOKS}   node $(hostname)"
rc=0
srun --environment="${EDF}" bash -s <<'INNER' || rc=$?
set -u
# THE SCRIPT PRINTED FINDINGS AND EXITED WITH `rm -f`'s STATUS, which cannot fail -- so nothing
# it said was ever a pass or a fail. Three conditions are hard failures, tracked here.
fail=0

echo
echo "--- 1. does the CXI DEVICE reach the container? (libcxi talks to these)"
if ls -la /dev/cxi* 2>/dev/null; then :; else
    echo "  NO /dev/cxi* -- no CXI device access, nothing can use Slingshot"
    fail=1
fi

echo
echo "--- 2. which libfabric does MPI resolve to? (MPICH is ch4:ofi)"
mpi_so="$(ls /opt/view/lib/libmpi.so* 2>/dev/null | head -1)"
if [ -n "${mpi_so}" ]; then
    echo "  libmpi: ${mpi_so}"
    fab="$(ldd "${mpi_so}" 2>/dev/null | awk '/fabric/ {print $3}' | head -1)"
    if [ -z "${fab}" ]; then
        echo "  (libmpi does not link libfabric directly)"
    else
        echo "  -> ${fab}"
        case "${fab}" in
            /opt/ofi-sdk/*|/opt/spack-install/*)
                echo "  FAIL: that is the compile-only stub, built with providers disabled" >&2
                fail=1 ;;
            /opt/cscs/*) echo "  OK: the netstack artifact" ;;
        esac
    fi
else
    echo "  no libmpi.so under /opt/view/lib"
fi
echo "  resolver's answer for libfabric.so.1:"
ldconfig -p 2>/dev/null | grep -i 'libfabric' || echo "    not in ldconfig cache"
for c in /opt/cscs/netstack /opt/cscs/netstack/lib /opt/view/lib /usr/lib/x86_64-linux-gnu \
         /usr/lib64 /opt/ofi-sdk/lib /opt/ofi/lib; do
    ls -la "${c}"/libfabric.so* 2>/dev/null
done

echo
echo "--- 3. which PROVIDERS does that libfabric offer? cxi is the one that matters"
prov=""
if command -v fi_info >/dev/null 2>&1; then
    prov="$(fi_info -l 2>&1 | head -20)"
else
    echo "  no fi_info on PATH; trying the artifact's and the image's own copies"
    for f in /opt/cscs/netstack/bin/fi_info /opt/ofi-sdk/bin/fi_info /opt/ofi/bin/fi_info; do
        if [ -x "${f}" ]; then echo "  ${f}:"; prov="$("${f}" -l 2>&1 | head -12)"; break; fi
    done
fi
printf '%s\n' "${prov:-  no fi_info anywhere}"
# ABSENCE OF THE TOOL IS NOT ABSENCE OF THE PROVIDER. Job 630034 reported "no cxi provider listed"
# on a node whose RCCL was at that moment running a correct 8-rank collective over cxi -- the
# netstack artifact simply ships no fi_info binary, and neither does the image. Failing on that
# rejects a working stack for a missing diagnostic, which is a worse error than the one this leg
# exists to catch. When fi_info is unavailable, fall back to reading the provider names out of the
# libfabric binary itself, and if even that is not possible say INCONCL and let the live MPI
# provider check in mpi_gpu_check.sh be the authority.
if printf '%s' "${prov}" | grep -qi cxi; then
    echo "  cxi provider present"
elif [ -n "${prov}" ]; then
    echo "  FAIL: fi_info ran and did NOT list cxi -- this libfabric cannot reach Slingshot" >&2
    fail=1
elif lf_art="$(ls /opt/cscs/netstack/libfabric.so.1 /opt/cscs/netstack/lib/libfabric.so.1 \
                  /usr/lib64/libfabric.so.1 2>/dev/null | head -1)" && [ -n "${lf_art}" ] \
     && strings "${lf_art}" 2>/dev/null | grep -q 'cxi_dev_list'; then
    echo "  cxi provider symbols present in ${lf_art} (fi_info is not shipped by the artifact)"
else
    echo "  INCONCL: no fi_info anywhere and could not read provider names from the library;" >&2
    echo "           mpi_gpu_check.sh reports the provider the live process actually selected" >&2
fi

echo
echo "--- 4. can the RCCL plugin resolve against it?"
found=0
for plug in /opt/cscs/netstack/librccl-net.so /opt/cscs/netstack/lib/librccl-net.so \
            /opt/ofi/lib/librccl-net.so; do
    if [ -e "${plug}" ]; then
        found=1
        echo "  ${plug}"
        ldd "${plug}" 2>&1 | grep -iE 'fabric|not found' || echo "    all deps resolved"
    fi
done
[ "${found}" -eq 1 ] || echo "  NO PLUGIN VISIBLE -- RCCL will use its built-in Socket transport"

echo
echo "--- 5. what did the hook actually inject?"
ls -la /opt/cscs/netstack 2>/dev/null | head -12 || echo "  no /opt/cscs/netstack"
mount 2>/dev/null | grep -iE 'cxi|fabric|libfabric' || echo "  no cxi/libfabric bind mounts visible"

exit "${fail}"
INNER
rm -f "${EDF}"

echo
if [ "${rc}" -eq 0 ]; then
    echo "PROVENANCE: PASS"
else
    echo "PROVENANCE: FAIL"
    exit 1
fi
