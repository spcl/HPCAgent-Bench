# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Sourced by the install-*.sh scripts. verify FILE [ALGOS] -- refuse FILE unless its digest is CHECKSUM
# (<algo>:<hex>), as the vendor publishes it. ALGOS lists the accepted algorithms (default: sha256 sha384 sha512);
# a vendor that publishes only md5 (NVIDIA's HPC SDK) passes md5 explicitly.

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
verify() {
    local file=$1 algos=${2:-sha256 sha384 sha512} algo="${CHECKSUM%%:*}" want="${CHECKSUM#*:}" got
    [[ " ${algos} " == *" ${algo} "* ]] || {
        echo "CHECKSUM must be <${algos// /|}>:<hex>, got '${CHECKSUM}'" >&2
        exit 2
    }
    got="$("${algo}sum" "${file}" | cut -d' ' -f1)"
    [[ "${got}" == "${want,,}" ]] || { echo "${file}: ${algo} ${got}, expected ${want}" >&2; exit 1; }
}
