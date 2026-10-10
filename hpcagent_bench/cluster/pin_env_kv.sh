#!/usr/bin/env bash

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
pin_env_kv() {
    local file="$1" kv="$2" key="${2%%=*}"
    [[ -f "${file}" ]] || { echo "pin_env_kv: no such env file ${file}" >&2; return 2; }
    local tmp
    tmp="$(mktemp "${file}.pin.XXXXXX")"
    grep -v "^${key}=" "${file}" >"${tmp}" || true
    printf '%s\n' "${kv}" >>"${tmp}"
    mv -- "${tmp}" "${file}"
}
