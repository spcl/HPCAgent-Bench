#!/usr/bin/env bash
# arm_nodes <env-file> -- how many nodes that arm's .env actually needs.
#
# beverin.sbatch refuses to run when the allocation does not match INFERENCE_NODES + AGENT_NODES +
# JUDGE_NODES exactly ("allocation has N nodes, but .env requests M", exit 2). The submit scripts
# used to hardcode --nodes=6, which was right until 23b9b4b2 sized the judge pool from measured
# grading rates: qwen30b went to 8 judge nodes and oss120b to 6, so both models needed 10 and every
# arm would have died at once. Derive the number from the same file the launcher reads, and the
# two can no longer disagree.
arm_nodes() {
    local env_file="$1" inference agent judge
    [[ -s "${env_file}" ]] || { echo "arm_nodes: missing env file ${env_file}" >&2; return 2; }
    # grep rather than sourcing: these files carry VLLM_EXTRA_ARGS and other values that a shell
    # would word-split or expand, and this needs three integers, not the environment.
    inference="$(grep -oP '^INFERENCE_NODES=\K[0-9]+' "${env_file}" || true)"
    agent="$(grep -oP '^AGENT_NODES=\K[0-9]+' "${env_file}" || true)"
    judge="$(grep -oP '^JUDGE_NODES=\K[0-9]+' "${env_file}" || true)"
    # The launcher's own defaults, for an arm that leaves one unset.
    echo $(( ${inference:-2} + ${agent:-1} + ${judge:-1} ))
}
