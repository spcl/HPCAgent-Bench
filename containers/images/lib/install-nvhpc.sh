#!/usr/bin/env bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# NVIDIA HPC SDK (nvc, nvc++, nvfortran), the cc_nvhpc column's driver on a host whose image does not carry it (the AMD
# judge), from NVIDIA's tarball into a user PREFIX without root. TARBALL is nvhpc_<year>_<ver>_Linux_<arch>_cuda_<x>.tar.gz
# as a local path or a URL; the default is NVIDIA's link for VERSION on this machine's arch. Several GB. It unpacks
# only when its digest is CHECKSUM: NVIDIA publishes md5 alone (https://developer.download.nvidia.com/hpc-sdk/<ver>/
# md5sum.txt), so md5 is accepted here and nowhere else; the default tarballs' are pinned below. Run it where the
# column runs (inside the judge image): the installer records the gcc it finds in nvc's localrc. CPU_ONLY=1 (the
# default) leaves the bundled CUDA, math and comm libraries, profilers and examples out of the unpack: a CPU column
# needs only the compilers, and the rest is most of the files.
#
#   install-nvhpc.sh PREFIX [TARBALL]          VERSION=<yy.m> (default 26.11), CHECKSUM=<md5|sha256|...>:<hex>, CPU_ONLY=0|1
set -euo pipefail

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 PREFIX [TARBALL(.tar.gz path or URL)]" >&2
    exit 2
fi
PREFIX=$1
VERSION=${VERSION:-26.11}
arch=$(uname -m)
year=20${VERSION%%.*}
compact=${VERSION/./}
name=nvhpc_${year}_${compact}_Linux_${arch}_cuda_${CUDA:-13.4}
TARBALL=${2:-https://developer.download.nvidia.com/hpc-sdk/${VERSION}/${name}.tar.gz}
# From NVIDIA's md5sum.txt of the release.
case ${name} in
    nvhpc_2026_2611_Linux_x86_64_cuda_13.4) pinned=md5:e85ea5d785cf0497d82b5a402cb30a98 ;;
    nvhpc_2026_2611_Linux_aarch64_cuda_13.4) pinned=md5:35f797000dc0e2adc40f909cc43b5688 ;;
    *) pinned="" ;;
esac
CHECKSUM=${CHECKSUM:-${pinned}}
[[ -n "${CHECKSUM}" ]] || {
    echo "set CHECKSUM=md5:<hex>, NVIDIA's https://developer.download.nvidia.com/hpc-sdk/${VERSION}/md5sum.txt" >&2
    exit 2
}
# shellcheck source=verify_checksum.sh
. "$(dirname -- "${BASH_SOURCE[0]}")/verify_checksum.sh"
work=$(mktemp -d)
trap 'rm -rf "${work}"' EXIT

if [[ -f ${TARBALL} ]]; then
    tarball=${TARBALL}
else
    tarball="${work}/${name}.tar.gz"
    curl -fSL -o "${tarball}" "${TARBALL}"
fi
verify "${tarball}" "md5 sha256 sha384 sha512"
skip=()
if [[ ${CPU_ONLY:-1} == 1 ]]; then
    for part in cuda math_libs comm_libs profilers examples; do
        skip+=("--exclude=*Linux_${arch}/${VERSION}/${part}")
    done
fi
tar -xzf "${tarball}" -C "${work}" "${skip[@]}"
mkdir -p "${PREFIX}"
# The tarball's installer, unattended: a single-system install into PREFIX.
root=${PREFIX}/Linux_${arch}/${VERSION}/compilers
if [[ ${CPU_ONLY:-1} == 1 ]]; then
    # Without the bundled cuda/ the installer's last step, makelocalrc, stops at "cuda directory not found" and
    # writes no localrc (nvc refuses to run without one). Pinning the CUDA version skips that lookup.
    NVHPC_SILENT=true NVHPC_INSTALL_DIR="${PREFIX}" NVHPC_INSTALL_TYPE=single "${work}/${name}/install" || true
    "${root}/bin/makelocalrc" -x "${root}/bin" -cuda "${CUDA:-13.4}" -gcc gcc -gpp g++ -g77 gfortran
else
    NVHPC_SILENT=true NVHPC_INSTALL_DIR="${PREFIX}" NVHPC_INSTALL_TYPE=single "${work}/${name}/install"
fi
[[ -x ${root}/bin/nvc && -f ${root}/bin/localrc ]] || { echo "no nvc or localrc under ${root}/bin" >&2; exit 1; }
"${root}/bin/nvc" --version | grep -m 1 nvc
echo "# Export this before running the cc_nvhpc column (nvc stays off PATH; its libraries are on its own rpath):"
echo "export HPCAGENT_BENCH_NVHPC_ROOT=${root}"
