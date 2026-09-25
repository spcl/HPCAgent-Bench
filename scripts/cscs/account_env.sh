#!/usr/bin/env bash
# Resolve the Slurm project account ONCE and hand it to sbatch/srun/salloc through Slurm's own
# input environment variables. Source before submitting anything.
#
#   . "${HPCAGENT_BENCH_REPO}/scripts/cscs/account_env.sh"
#
# WHY THE ENVIRONMENT AND NOT -A. Slurm reads SBATCH_ACCOUNT, SLURM_ACCOUNT and SALLOC_ACCOUNT
# natively, so exporting them gives every #SBATCH script an account without any of them naming
# one -- which is what stops a campaign silently splitting across two billing lines. The site
# layer (scripts/site_env.sh, sourced below) supplies the partition the same way (SBATCH_PARTITION).
#
# WHY IT IS DETECTED AND NOT A CONSTANT. An account name is site- and person-specific; writing one
# into the repo makes the benchmark unrunnable for anybody else. The account is therefore read
# from the user's own Slurm associations. `root` is excluded: it is never a project account.
#
# WHY AMBIGUITY IS A HARD ERROR. With more than one candidate, ANY automatic choice can differ
# between two submissions of the same campaign -- pick-by-fairshare in particular re-decides every
# time usage shifts. Half a campaign billed to one account and half to another is unrecoverable
# after the fact, so this refuses to guess and asks once.
#
# Zero candidates returns 2, not fatal: dry runs, tests and hooks source this with no Slurm account.
# Every submitter sources it as `. account_env.sh || exit 2`, so none submits without one.
#
# WHEN SLURM ACCOUNTING DOES NOT ANSWER (weekly maintenance, slurmdbd down) nothing can be checked:
# an exported HPCAGENT_BENCH_ACCOUNT is used unchecked (sbatch refuses an account that is not an
# association), anything else resolves no account. It used to read the silence as "not one of your
# associations" and refuse, which failed every pre-commit hook run through run_hook.sh.
set -uo pipefail

. "$(dirname -- "${BASH_SOURCE[0]}")/../site_env.sh" || return 1 2>/dev/null || exit 1

# A dump lands in the crashing process's CWD (the checkout) and Slurm propagates the SUBMITTER's
# core limit, so the floor has to be set here.
ulimit -c 0
# hpcagent_bench_accounts -- the user's associations but root, one per line; returns 3 when Slurm
# accounting does not answer. No sacctmgr at all (CI, a laptop) is no associations.
hpcagent_bench_accounts() {
    local out
    command -v sacctmgr >/dev/null 2>&1 || return 0
    out="$(timeout 60 sacctmgr -nP show assoc where user="${USER:-$(id -un)}" format=Account 2>/dev/null)" || return 3
    printf '%s\n' "${out}" | sort -u | grep -vx 'root' | grep -v '^$' || true
}

hpcagent_bench_resolve_account() {
    local candidates n rc=0
    candidates="$(hpcagent_bench_accounts)" || rc=$?
    if [ -n "${HPCAGENT_BENCH_ACCOUNT:-}" ]; then
        if [ "${HPCAGENT_BENCH_ACCOUNT}" = root ]; then
            echo "HPCAGENT_BENCH_ACCOUNT=root: root is never a campaign account" >&2
            return 1
        fi
        if [ "${rc}" -eq 3 ]; then
            echo "Slurm accounting does not answer: HPCAGENT_BENCH_ACCOUNT=${HPCAGENT_BENCH_ACCOUNT} used unchecked" >&2
            printf '%s' "${HPCAGENT_BENCH_ACCOUNT}"; return 0
        fi
        # A here-string, not printf | grep -q: under pipefail grep -q exits on the first match, printf
        # dies of SIGPIPE, and the pipeline reads as "no match" -- a listed account refused at random.
        if ! grep -qxF "${HPCAGENT_BENCH_ACCOUNT}" <<<"${candidates}"; then
            echo "HPCAGENT_BENCH_ACCOUNT=${HPCAGENT_BENCH_ACCOUNT} is not one of your associations:" >&2
            printf '%s\n' "${candidates}" | sed 's/^/    /' >&2
            return 1
        fi
        printf '%s' "${HPCAGENT_BENCH_ACCOUNT}"; return 0
    fi
    if [ "${rc}" -eq 3 ]; then
        echo "Slurm accounting does not answer: no account resolved (export HPCAGENT_BENCH_ACCOUNT)" >&2
        return 2
    fi

    n="$(printf '%s\n' "${candidates}" | grep -c . || true)"
    case "${n}" in
        0) echo "no usable Slurm association for ${USER:-$(id -un)} (root is never used)." >&2
           echo "  Ask your site to add you to a project account." >&2
           return 2 ;;
        1) printf '%s' "${candidates}"; return 0 ;;
        *) echo "several project accounts available and none chosen. Pick ONE for the whole" >&2
           echo "  campaign -- splitting it across two is not repairable afterwards:" >&2
           printf '%s\n' "${candidates}" | while read -r a; do
               printf '    %-12s fairshare %s\n' "${a}" \
                   "$(sshare -nP -A "${a}" -u "${USER:-$(id -un)}" -o FairShare 2>/dev/null | tail -1)" >&2
           done
           echo "  export HPCAGENT_BENCH_ACCOUNT=<one of the above>" >&2
           return 1 ;;
    esac
}

# hpcagent_bench_require_account -- a submitter's last gate before sbatch: refuses when no account
# was resolved, since a cluster may run an accountless job on a default account nobody chose.
hpcagent_bench_require_account() {
    [ -n "${SBATCH_ACCOUNT:-}" ] && return 0
    echo "no Slurm account resolved (scripts/cscs/account_env.sh): refusing to submit on the default account" >&2
    return 2
}

# if/elif, not a separate $? check: a caller under `set -e` (run_hook.sh) aborts on a failing assignment.
if _acct="$(hpcagent_bench_resolve_account)"; then
    export HPCAGENT_BENCH_ACCOUNT="${_acct}"
    export SBATCH_ACCOUNT="${_acct}" SLURM_ACCOUNT="${_acct}" SALLOC_ACCOUNT="${_acct}"
elif [ "$?" -ne 2 ]; then
    # ambiguous, or an explicit HPCAGENT_BENCH_ACCOUNT that is not an association: hard fail
    unset _acct
    [ "${BASH_SOURCE[0]}" = "${0}" ] && exit 1
    return 1 2>/dev/null || exit 1
fi
unset _acct
# An `if`, not `[ ... ] && echo`: as the LAST command of a sourced file, a false test is the file's
# exit status, so every `. account_env.sh || exit` refused a resolved account.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "account: ${HPCAGENT_BENCH_ACCOUNT:-<none resolved>}"
fi
