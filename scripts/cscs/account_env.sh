#!/usr/bin/env bash
# Resolve the Slurm project account ONCE and hand it to sbatch/srun/salloc through Slurm's own
# input environment variables. Source before submitting anything.
#
#   . "${HPCAGENT_BENCH_REPO}/scripts/cscs/account_env.sh"
#
# WHY THE ENVIRONMENT AND NOT -A. beverin now REJECTS every accountless job:
#     ERROR: you must specify a project account (-A <account>)
# This repo has 456 #SBATCH directives across 54 files and none carries an account, so as of the
# Sep 2026 migration every campaign submission is refused outright. Slurm reads SBATCH_ACCOUNT,
# SLURM_ACCOUNT and SALLOC_ACCOUNT natively (verified on beverin: SBATCH_ACCOUNT alone produced a
# job recorded under that account), so exporting them here gives all 456 an account without
# editing one of them -- and keeps the rule that no submitter spells an account of its own, which
# is what stops a campaign silently splitting across two billing lines.
#
# WHY IT IS DETECTED AND NOT A CONSTANT. An account name is site- and person-specific; writing one
# into the repo makes the benchmark unrunnable for anybody else. The account is therefore read
# from the user's own Slurm associations. `root` is excluded: every pre-migration job here ran
# under it and it can no longer be submitted to.
#
# WHY AMBIGUITY IS A HARD ERROR. With more than one candidate, ANY automatic choice can differ
# between two submissions of the same campaign -- pick-by-fairshare in particular re-decides every
# time usage shifts. Half a campaign billed to one account and half to another is unrecoverable
# after the fact, so this refuses to guess and asks once.
#
# WHY ZERO CANDIDATES IS NOT ALSO A HARD ERROR HERE. A real user missing a project account still
# needs the message below, but a dry run (SUBMIT=0, never reaches sbatch) and every submitter test
# run this same file with no Slurm reachable at all, or a CI service account that genuinely has
# none -- and neither is trying to bill a campaign to anything. Failing THIS file for them blocked
# submit_common.sh before it ever got to the SUBMIT=0 short circuit (2026-09-17). So zero candidates
# is reported (return 2) but not fatal: the account vars stay unset, and a real submission to
# beverin still gets refused, just by sbatch's own accountless-job error instead of pre-empting it
# here. Ambiguity and an explicit-but-wrong HPCAGENT_BENCH_ACCOUNT stay hard errors (return 1):
# both are misconfiguration, not "nothing to resolve".
set -uo pipefail

hpcagent_bench_accounts() {
    sacctmgr -nP show assoc where user="${USER:-$(id -un)}" format=Account 2>/dev/null \
        | sort -u | grep -vx 'root' | grep -v '^$'
}

hpcagent_bench_resolve_account() {
    local candidates n
    if [ -n "${HPCAGENT_BENCH_ACCOUNT:-}" ]; then
        if ! hpcagent_bench_accounts | grep -qxF "${HPCAGENT_BENCH_ACCOUNT}"; then
            echo "HPCAGENT_BENCH_ACCOUNT=${HPCAGENT_BENCH_ACCOUNT} is not one of your associations:" >&2
            hpcagent_bench_accounts | sed 's/^/    /' >&2
            return 1
        fi
        printf '%s' "${HPCAGENT_BENCH_ACCOUNT}"; return 0
    fi

    candidates="$(hpcagent_bench_accounts)"
    n="$(printf '%s\n' "${candidates}" | grep -c . || true)"
    case "${n}" in
        0) echo "no usable Slurm association for ${USER:-$(id -un)} (only 'root', which beverin" >&2
           echo "  no longer accepts). Ask CSCS to add you to a project account." >&2
           echo "  a real sbatch submission will be refused for lacking one; a dry run is unaffected." >&2
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

# An `if`, not `_acct=...; status=$?; if [ "$status" ...]`: a caller sourcing this under `set -e`
# (run_hook.sh does) aborts on the FIRST simple command that fails, before a separate status check
# ever runs -- only the tested command of an if/elif is exempt from that.
if _acct="$(hpcagent_bench_resolve_account)"; then
    export HPCAGENT_BENCH_ACCOUNT="${_acct}"
    export SBATCH_ACCOUNT="${_acct}" SLURM_ACCOUNT="${_acct}" SALLOC_ACCOUNT="${_acct}"
elif [ "$?" -ne 2 ]; then
    # ambiguous, or an explicit HPCAGENT_BENCH_ACCOUNT that is not one of the associations: both
    # are misconfiguration, not "nothing to resolve" -- hard fail.
    unset _acct
    [ "${BASH_SOURCE[0]}" = "${0}" ] && exit 1
    return 1 2>/dev/null || exit 1
fi
# status 2 (no usable account) falls through here: reported above but not fatal -- see WHY ZERO
# CANDIDATES above.
unset _acct
# An `if`, not `[ ... ] && echo`: as the LAST command of a sourced file, a false test is the file's
# exit status, so every `. account_env.sh || exit` refused a resolved account.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "account: ${HPCAGENT_BENCH_ACCOUNT:-<none resolved>}"
fi
