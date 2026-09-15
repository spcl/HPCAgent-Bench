# shellcheck shell=bash
# Source this in YOUR job on another Alps cluster (Daint) to use the ACCESS=alps endpoint that YOUR
# beverin job serves (serve-private.sbatch prints this line with both paths filled in):
#
#   source <checkout>/containers/cluster/ce-images/inference/alps-endpoint.sh <beverin run dir>/endpoint.json
#
# Checks, in order: endpoint.json is readable; the key file it names is yours and mode 600;
# GET /v1/models lists the served model; one POST /v1/chat/completions answers with a choice.
# Then exports VLLM_BASE_URL, VLLM_API_KEY and VLLM_MODEL. Any failure returns non-zero and exports
# nothing. The key reaches curl on stdin (-H @-), never argv, and is never printed.

alps_endpoint_request() {  # alps_endpoint_request <key> <url> [json body]: prints the body, then the HTTP code
    local data=()
    [[ $# -lt 3 ]] || data=(-H 'Content-Type: application/json' --data-binary "$3")
    curl -sS --noproxy '*' --max-time 120 -w '\n%{http_code}' "${data[@]}" -H @- "$2" <<< "Authorization: Bearer $1"
}

alps_endpoint() {  # alps_endpoint <endpoint.json>
    local file="${1:-}" fields url model key_file key reply code body
    if [[ ! -r "${file}" ]]; then
        echo "alps-endpoint: cannot read '${file}': wrong path, or the serving job has ended" >&2
        return 2
    fi
    if ! fields="$(python3 -c 'import json, sys; d = json.load(open(sys.argv[1]))
print(d["url"], d["served_model"], d["key_file"], sep="\n")' "${file}" 2>/dev/null)"; then
        echo "alps-endpoint: ${file} is not an endpoint file written by serve-private.sbatch" >&2
        return 2
    fi
    { read -r url; read -r model; read -r key_file; } <<< "${fields}"
    if [[ ! -f "${key_file}" || "$(stat -c %a -- "${key_file}")" != 600 \
        || "$(stat -c %u -- "${key_file}")" != "$(id -u)" ]]; then
        echo "alps-endpoint: key file ${key_file} must exist, be owned by you and be mode 600" >&2
        return 2
    fi
    key="$(tr -d '\n' < "${key_file}")"

    reply="$(alps_endpoint_request "${key}" "${url}/models")"
    code="${reply##*$'\n'}"
    if [[ "${code}" != 200 ]] || ! python3 -c 'import json, sys
sys.exit(not any(m.get("id") == sys.argv[1] for m in json.loads(sys.argv[2]).get("data", [])))' \
        "${model}" "${reply%$'\n'*}" 2>/dev/null; then
        echo "alps-endpoint: GET ${url}/models answered ${code}, want 200 listing ${model}" \
            "(401: wrong key; 000: unreachable or the serving job has ended)" >&2
        return 1
    fi
    body="{\"model\": \"${model}\", \"max_tokens\": 4,"
    body+=" \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}]}"
    reply="$(alps_endpoint_request "${key}" "${url}/chat/completions" "${body}")"
    code="${reply##*$'\n'}"
    if [[ "${code}" != 200 ]] || ! python3 -c 'import json, sys
sys.exit(not json.loads(sys.argv[1]).get("choices"))' "${reply%$'\n'*}" 2>/dev/null; then
        echo "alps-endpoint: POST ${url}/chat/completions answered ${code}, want 200 with a choice" >&2
        return 1
    fi
    export VLLM_BASE_URL="${url}" VLLM_API_KEY="${key}" VLLM_MODEL="${model}"
    printf 'alps-endpoint: %s serves %s; exported VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL\n' "${url}" "${model}"
}

alps_endpoint "$@"
