#!/usr/bin/env bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Check a private endpoint (serve-private.sbatch) from anywhere you can ssh from. Every check runs ON
# the serving node over ssh, so the endpoint never listens beyond 127.0.0.1 and the key never leaves
# the node's home directory.
#
#   ping-endpoint.sh <node> <preset>                                  # on a Beverin login node
#   JUMP=beverin.alps.cscs.ch ping-endpoint.sh <node> <preset>       # from ela or Daint
#
# <node> is the node the job's "private endpoint is live" line names (squeue -j <jobid> -o %N),
# <preset> the job's PRESET (mi300 or mi200); it picks the key file. Checks, in order: /health
# answers 200; a POST without the key answers 401; a chat completion with it answers 200 with
# content. Prints the completion and its tokens per second. Exit 1 when a check fails.
#
# PORT (default 30000) is the leg's port, KEY_FILE the key on the node (default
# ~/.config/hpcagent-bench/<preset>-endpoint.key), MAX_TOKENS the completion's length.
set -euo pipefail
ulimit -c 0

#: The first leg's port (serve-private.sbatch API_PORT).
DEFAULT_PORT=30000
#: The name the server answers to (serve-private.sbatch SERVED_MODEL).
SERVED_MODEL=hpcagent-bench-vllm
#: A completion long enough to time, short enough to answer in seconds.
DEFAULT_MAX_TOKENS=128
#: Seconds one request may take; a loaded server answers well inside it.
REQUEST_TIMEOUT_S=300

node="${1:?usage: ping-endpoint.sh <node> <preset>}"
preset="${2:?usage: ping-endpoint.sh <node> <preset>}"
[[ "${preset}" =~ ^[a-z0-9]+$ ]] || { echo "ping-endpoint: preset must be a name like mi200, got '${preset}'" >&2; exit 2; }
port="${PORT:-${DEFAULT_PORT}}"
key_file="${KEY_FILE:-~/.config/hpcagent-bench/${preset}-endpoint.key}"
jump=(${JUMP:+-J "${JUMP}"})

ssh "${jump[@]}" -o BatchMode=yes "${node}" bash -s -- \
    "${port}" "${key_file}" "${SERVED_MODEL}" "${MAX_TOKENS:-${DEFAULT_MAX_TOKENS}}" "${REQUEST_TIMEOUT_S}" <<'ON_NODE'
set -euo pipefail
port=$1 key_file=${2/#\~/$HOME} model=$3 max_tokens=$4 timeout=$5
url="http://127.0.0.1:${port}"
[[ -r "${key_file}" ]] || { echo "FAIL: no key file ${key_file} on $(hostname)" >&2; exit 1; }
header=$(mktemp)
trap 'rm -f "${header}"' EXIT
printf 'Authorization: Bearer %s\n' "$(cat "${key_file}")" > "${header}"
body=$(printf '{"model": "%s", "max_tokens": %d, "messages": [{"role": "user", "content": "Explain in two sentences what a stencil kernel is."}], "chat_template_kwargs": {"enable_thinking": false}}' "${model}" "${max_tokens}")

check() {  # check <what> <want> <got>
    printf '%-26s %s (want %s)\n' "$1" "$3" "$2"
    [[ "$3" == "$2" ]] || { echo "FAIL: $1" >&2; exit 1; }
}
check "health" 200 "$(curl -s -o /dev/null -w '%{http_code}' -m "${timeout}" "${url}/health")"
check "POST without the key" 401 "$(curl -s -o /dev/null -w '%{http_code}' -m "${timeout}" \
    -H 'Content-Type: application/json' -d "${body}" "${url}/v1/chat/completions")"
answer=$(mktemp)
trap 'rm -f "${header}" "${answer}"' EXIT
timing=$(curl -s -o "${answer}" -w '%{http_code} %{time_total}' -m "${timeout}" -H 'Content-Type: application/json' \
    -H "@${header}" -d "${body}" "${url}/v1/chat/completions")
check "chat completion with key" 200 "${timing%% *}"
python3 - "${answer}" "${timing##* }" <<'REPORT'
import json, sys
reply = json.load(open(sys.argv[1]))
seconds = float(sys.argv[2])
text = reply["choices"][0]["message"]["content"] or ""
tokens = reply["usage"]["completion_tokens"]
if not text.strip():
    sys.exit("FAIL: the completion is empty")
print(f"completion                 {tokens} tokens in {seconds:.2f} s = {tokens / seconds:.1f} tok/s")
print(text.strip())
REPORT
ON_NODE
