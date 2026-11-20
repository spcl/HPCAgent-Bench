# shellcheck shell=bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# The keyed-endpoint helpers of the two launchers that serve off the loopback with a key:
# containers/inference/serve-private.sbatch and hpcagent_bench/cluster/serve-only.sbatch ACCESS=alps
# (docs/serving/private-endpoint.md, section 1). Sourced; every refusal calls the caller's refuse <message>.

# A core dump lands in the crashing process's CWD and Slurm propagates the SUBMITTER's core limit, so the floor has
# to be set here.
ulimit -c 0

# private_key <key file>: sets API_KEY, unexported, so srun --export=ALL never carries it. Refused unless the file is
# mode 600, owned by this user, at least 32 characters of [A-Za-z0-9._~+/=-].
private_key() {
    local file="$1" mode
    [[ -f "${file}" ]] || refuse "no key file ${file}; create it as the header shows"
    mode="$(stat -c %a -- "${file}")"
    [[ "${mode}" == 600 ]] || refuse "key file ${file} is mode ${mode}, not 600"
    [[ "$(stat -c %u -- "${file}")" == "$(id -u)" ]] || refuse "key file ${file} is not owned by $(id -un)"
    API_KEY="$(tr -d '\n' < "${file}")"
    [[ ${#API_KEY} -ge 32 ]] || refuse "key in ${file} is shorter than 32 characters"
    [[ "${API_KEY}" =~ ^[A-Za-z0-9._~+/=-]+$ ]] || refuse "key in ${file} has characters outside [A-Za-z0-9._~+/=-]"
}

# private_refuse_words <name> <word>...: the server words a keyed SGLang may not take from an operator (<name> says
# whose): a bind, the key or a config of its own, a second tokenizer worker (the key is checked with one only) or
# metrics (served without it).
private_refuse_words() {
    local name="$1" word
    shift
    for word in "$@"; do
        case "${word}" in
            --host*|--port*|--api*|--admin*|--config*|--tokenizer-worker*|--enable-metrics*)
                refuse "${name} may not set ${word%%=*}" ;;
        esac
    done
}

# private_dir <dir>: <dir> created mode 700 with scratch's inherited default ACLs stripped. It holds the secrets and
# the server log, which prints the key.
private_dir() {
    local acl
    (umask 077 && mkdir -p -- "$1")
    chmod 700 -- "$1"
    # Scratch default ACLs (group:csstaff r-x) are inherited and beat umask.
    if command -v setfacl >/dev/null 2>&1; then setfacl -b -- "$1" 2>/dev/null || true; fi
    [[ "$(stat -c %a -- "$1")" == 700 ]] || refuse "$1 is not mode 700"
    if command -v getfacl >/dev/null 2>&1; then
        acl="$(getfacl -cp -- "$1" 2>/dev/null || true)"
        if grep -qvE '^(user|group|other)::|^$' <<< "${acl}"; then refuse "$1 still carries extended ACL entries"; fi
    fi
}

# private_secret <file> <line>: printf is a builtin, so the content never reaches argv.
private_secret() {
    (umask 077 && printf '%s\n' "$2" > "$1")
    chmod 600 -- "$1"
    [[ "$(stat -c %a -- "$1")" == 600 ]] || refuse "$1 is not mode 600"
}

# private_secrets <dir>: the three secret files of a keyed SGLang in <dir>, from API_KEY: its --config, the key alone
# (clients' --api-key-file) and the curl header. Never printed; the caller deletes them on exit.
private_secrets() {
    private_secret "$1/sglang-auth.yaml" "api-key: \"${API_KEY}\""
    private_secret "$1/api.key" "${API_KEY}"
    private_secret "$1/auth.header" "Authorization: Bearer ${API_KEY}"
}

# private_status <base url> <body file> [header file]: the HTTP code of one chat completion, 000 when none answers.
private_status() {
    local auth=()
    [[ $# -lt 3 ]] || auth=(-H "@$3")
    curl -s -o /dev/null -w '%{http_code}' --max-time 600 -H 'Content-Type: application/json' "${auth[@]}" \
        --data-binary "@$2" "$1/v1/chat/completions" || true
}

# private_auth_check <base url> <body file> <header file>: one chat completion without the key and one with it,
# printed; true when they answer 401 and 200. Only then may a launcher publish the endpoint.
private_auth_check() {
    local closed open
    closed="$(private_status "$1" "$2")"
    open="$(private_status "$1" "$2" "$3")"
    printf 'unauthenticated POST: %s (want 401)\nauthenticated POST:   %s (want 200)\n' "${closed}" "${open}"
    [[ "${closed}" == 401 && "${open}" == 200 ]]
}
