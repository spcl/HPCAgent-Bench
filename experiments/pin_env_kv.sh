#!/usr/bin/env bash
# pin_env_kv <env-file> <KEY=VALUE> -- make KEY read exactly VALUE, once.
#
# The submitters used to append when the exact line was absent, which is not the same thing: a
# completion wave is COPIED from the previous wave's env, so every re-pin left the old line in
# place and the file grew a new one. A wave-4 env carried three AGENT_TIMEOUT_SECONDS lines
# (28800, 12600, 21600). run_cluster.sh sources under `set -a`, so last-wins made the value right
# by accident -- but nothing else reads these files that way. arm_nodes.sh greps a key with -oP
# and feeds the result to $(( )), so the day a duplicated key is one it reads, the arithmetic gets
# a two-line string and the submit dies with a syntax error instead of a wrong number.
pin_env_kv() {
    local file="$1" kv="$2" key="${2%%=*}"
    [[ -f "${file}" ]] || { echo "pin_env_kv: no such env file ${file}" >&2; return 2; }
    # Drop every existing spelling of the key, then write the wanted one at the end, so the pinned
    # value is also the last one a `set -a` source would take.
    local tmp
    tmp="$(mktemp "${file}.pin.XXXXXX")"
    grep -v "^${key}=" "${file}" >"${tmp}" || true
    printf '%s\n' "${kv}" >>"${tmp}"
    mv -- "${tmp}" "${file}"
}
