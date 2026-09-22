# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A submitter's COMMIT BUDGET is declared, consistent, and never acquired by accident.

Single submission used to be the llrblind intervention and nothing else, because every other arm
was compared against multi-submission baselines; a submitter that quietly pinned one submission
turned its arms into a blind variant nobody asked for (git-scicomp and scicomp-dc did, until
2026-09-17). Two things have changed since, and the guard has to say what it now guards:

* ``layers/common.env`` declares three modes over (oracle access x commit budget), and
  commit-single is the DEFAULT -- so "pins single" no longer implies "deviates from the campaign".
* ``submit-mlscale.sh`` is commit-single on purpose: a scaling curve read off the best of many
  commits is a best-of-k statistic rather than that submission's scaling.

What still must not happen is a submitter acquiring a commit budget nobody chose, or declaring one
in two places that disagree. So every submitter names its mode HERE, the two keys it pins must
agree with each other, and the oracle stays on everywhere but llrblind -- an arm goes blind by
disabling the score tool, which is a different key and a different intervention.
"""

import pathlib
import re

import pytest

EXPERIMENTS = pathlib.Path(__file__).resolve().parents[1] / "experiments"

#: Submitters whose arms are commit-single BY DESIGN, and why. Everything else is commit-unbounded
#: and pins the multi pair; adding a name here is a deliberate change to that experiment's contract.
COMMIT_SINGLE: dict[str, str] = {
    "submit-llrblind.sh": "the blind intervention: oracle-none + commit-single",
    "submit-mlscale.sh": "one graded submission per kernel, so a scaling curve is not best-of-k",
}

#: The submitter that also turns the score tool OFF. Blind is oracle access, not commit budget.
BLIND = {"submit-llrblind.sh"}

SINGLE_FLAG = re.compile(r"^[^#]*AGENT_SINGLE_SUBMISSION=1")
MULTI_FLAG = re.compile(r"^[^#]*AGENT_SINGLE_SUBMISSION=0")
SINGLE_PAGE = re.compile(r"^[^#]*AGENT_SUBMISSION_POLICY_FILE=\$?\{?[^#]*submission-single\.md")
MULTI_PAGE = re.compile(r"^[^#]*AGENT_SUBMISSION_POLICY_FILE=\$?\{?[^#]*submission-multi\.md")
SCORE_OFF = re.compile(r"^[^#]*AGENT_SCORE_TOOL=(0|none)")

SUBMITTERS = sorted(p.name for p in EXPERIMENTS.glob("submit*.sh"))


def pins(script: str, pattern: re.Pattern[str]) -> list[str]:
    """Non-comment lines of ``script`` that pin ``pattern``."""
    text = (EXPERIMENTS / script).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if pattern.search(line)]


@pytest.mark.parametrize("script", SUBMITTERS)
def test_a_submitter_is_commit_single_only_by_declaration(script: str) -> None:
    """No submitter acquires a single-submission budget that is not recorded in COMMIT_SINGLE."""
    single = pins(script, SINGLE_FLAG) + pins(script, SINGLE_PAGE)
    if script in COMMIT_SINGLE:
        assert single, (
            f"{script} is listed as commit-single ({COMMIT_SINGLE[script]}) but pins neither "
            f"AGENT_SINGLE_SUBMISSION=1 nor submission-single.md; an inherited default is not a "
            f"declaration, and a later change to layers/common.env would move this experiment"
        )
        return
    assert single == [], (
        f"{script} pins a single-submission budget: {single}. Every other arm is compared against "
        f"commit-unbounded baselines, so either pin the multi pair or add {script} to "
        f"COMMIT_SINGLE with the reason its experiment needs one graded submission."
    )


@pytest.mark.parametrize("script", SUBMITTERS)
def test_the_two_commit_budget_keys_never_disagree(script: str) -> None:
    """``AGENT_SINGLE_SUBMISSION`` is what the driver enforces and the policy page is what the
    agent is told; a submitter pinning one of each is an arm whose prompt and whose limit describe
    different experiments."""
    crossed = (pins(script, SINGLE_FLAG) and pins(script, MULTI_PAGE)) or (
        pins(script, MULTI_FLAG) and pins(script, SINGLE_PAGE)
    )
    assert not crossed, (
        f"{script} pins AGENT_SINGLE_SUBMISSION and AGENT_SUBMISSION_POLICY_FILE to opposite "
        f"modes: {pins(script, SINGLE_FLAG) + pins(script, MULTI_FLAG)} vs "
        f"{pins(script, SINGLE_PAGE) + pins(script, MULTI_PAGE)}"
    )


@pytest.mark.parametrize("script", SUBMITTERS)
def test_only_the_blind_submitter_turns_the_oracle_off(script: str) -> None:
    """Commit-single does not make an arm blind: blind is oracle-none, a different key.

    Only a LITERAL pin is visible here -- llrblind reaches the same key through its packet
    (``AGENT_SCORE_TOOL=${packet_kv[AGENT_SCORE_TOOL]}``, the ``no-score-tool`` packet), which no
    regex over the script can resolve. So this catches a submitter that hardcodes the oracle off,
    not one that composes a packet which does; packet coverage is tests/test_packets.py's.
    """
    off = pins(script, SCORE_OFF)
    if script in BLIND:
        return
    assert off == [], f"{script} disables the score tool: {off}. That is the llrblind intervention."
