# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``token_cost.task_totals``: the TASK TOKEN TOTAL (T2) of one worker directory.

A relaunched task's attempts are separate transcripts -- the driver moves a crashed attempt's log
aside (``claude.attemptN.log``) and starts the next one from an empty context AND an empty workspace
(T5) -- so the task is what its FINAL attempt did and the earlier attempts are reported beside it as
spend. A worker directory the driver never entered must not read as a task that cost 0.
"""

import importlib.util
import json
import os
import pathlib
import sys
from types import ModuleType

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def load_token_cost() -> ModuleType:
    spec = importlib.util.spec_from_file_location("token_cost", REPO / "experiments" / "token_cost.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="token_cost")
def token_cost_fixture() -> ModuleType:
    return load_token_cost()


def assistant_line(message_id: str, input_tokens: int, output_tokens: int) -> str:
    """One claude ``assistant`` event with a whole turn's usage, as the driver's fold reads it."""
    return json.dumps(
        {
            "type": "assistant",
            "message": {
                "id": message_id,
                "usage": {
                    "input_tokens": input_tokens,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": output_tokens,
                },
            },
        }
    )


def result_line(output_tokens: int) -> str:
    """The claude ``result`` event, the only place ``episode_cost`` reads output from."""
    return json.dumps({"type": "result", "usage": {"output_tokens": output_tokens}})


def write_claude_log(path: pathlib.Path, input_tokens: int, output_tokens: int) -> None:
    path.write_text(assistant_line("m1", input_tokens, 0) + "\n" + result_line(output_tokens) + "\n", encoding="utf-8")


def write_attempts(module: ModuleType, folder: pathlib.Path, attempts: list[tuple[int, int, bool]]) -> None:
    """The driver's attempt ledger: one line per ``(attempt, start_ms, crashed)`` (T5)."""
    lines = [
        json.dumps({"attempt": n, "start_ms": start, "end_ms": start + 1, "returncode": 1, "crashed": c, "cleared": c})
        for n, start, c in attempts
    ]
    (folder / module.ATTEMPTS_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def usage_line(call_input: int, output: int) -> str:
    return json.dumps({"input": call_input, "cached_input": 0, "output": output, "reasoning": 0})


def test_a_legacy_dir_without_attempts_jsonl_also_counts_only_its_final_attempt(
    token_cost, tmp_path: pathlib.Path
) -> None:
    """PROPERTY CHANGED on purpose: this asserted the sum over attempts. One rule holds for every
    directory now, old runs included -- the last agent ran the task from nothing to its end, so the
    total is its attempt's and the earlier ones are reported as crashed spend (T2/T5)."""
    write_claude_log(tmp_path / "claude.attempt1.log", input_tokens=1000, output_tokens=100)
    write_claude_log(tmp_path / "claude.log", input_tokens=2000, output_tokens=200)

    totals = token_cost.task_totals(tmp_path)

    assert totals.attempts == 2
    # effective adds the result event's output; billed folds only the per-turn assistant usage,
    # which reports output_tokens: 0 on this endpoint (episode_cost's own docstring) -- so the two
    # totals differ by exactly the output component, which is the point of carrying both (8.1).
    assert totals.tokens_effective == 2000 + 200
    assert totals.tokens_billed == 2000
    assert totals.tokens_effective_crashed == 1000 + 100
    assert totals.tokens_billed_crashed == 1000


def test_a_legacy_dirs_final_attempt_starts_when_the_crash_was_moved_aside(token_cost, tmp_path: pathlib.Path) -> None:
    """Old runs carry no ledger, so the cut X7 applies comes from the rename: the moved-aside
    transcript was renamed while the crash was handled, which is where the final attempt began."""
    crashed = tmp_path / "claude.attempt1.log"
    write_claude_log(crashed, input_tokens=1000, output_tokens=100)
    write_claude_log(tmp_path / "claude.log", input_tokens=2000, output_tokens=200)
    os.utime(crashed, (1_700_000_000, 1_700_000_000))

    assert token_cost.task_totals(tmp_path).final_attempt_start_ms == 1_700_000_000_000


def test_a_fresh_relaunched_task_counts_only_its_final_attempt_and_reports_the_rest_as_crashed(
    token_cost, tmp_path: pathlib.Path
) -> None:
    """T5: the relaunch wiped the workspace, so attempt 1 built no part of what attempt 2 was graded
    on. Its spend is real and is reported as crashed; the ledger states when attempt 2 began."""
    write_claude_log(tmp_path / "claude.attempt1.log", input_tokens=1000, output_tokens=100)
    write_claude_log(tmp_path / "claude.log", input_tokens=2000, output_tokens=200)
    write_attempts(token_cost, tmp_path, [(1, 1_000, True), (2, 2_000, False)])

    totals = token_cost.task_totals(tmp_path)

    assert totals.attempts == 2
    assert (totals.tokens_effective, totals.tokens_billed) == (2000 + 200, 2000)
    assert (totals.tokens_effective_crashed, totals.tokens_billed_crashed) == (1000 + 100, 1000)
    assert totals.final_attempt_start_ms == 2_000


def test_a_single_attempt_tasks_total_is_that_attempts_own(token_cost, tmp_path: pathlib.Path) -> None:
    write_claude_log(tmp_path / "claude.log", input_tokens=500, output_tokens=50)

    totals = token_cost.task_totals(tmp_path)

    assert totals.attempts == 1
    assert totals.tokens_effective == 550
    assert totals.tokens_billed == 500


def test_effective_comes_from_episode_cost_not_a_reimplementation(token_cost, tmp_path: pathlib.Path) -> None:
    """The per-attempt effective figure must be exactly ``episode_cost(log)["effective"]`` summed,
    the one implementation of the cost model -- not a second copy that can drift from it."""
    write_claude_log(tmp_path / "claude.attempt1.log", input_tokens=900, output_tokens=10)
    write_claude_log(tmp_path / "claude.log", input_tokens=1900, output_tokens=20)

    totals = token_cost.task_totals(tmp_path)

    def effective(name: str) -> int:
        return int(token_cost.episode_cost(tmp_path / name)["effective"])

    assert totals.tokens_effective == effective("claude.log")
    assert totals.tokens_effective_crashed == effective("claude.attempt1.log")


def test_billed_comes_from_accumulate_total_tokens_not_a_reimplementation(token_cost, tmp_path: pathlib.Path) -> None:
    """The per-attempt billed figure must be exactly ``accumulate_total_tokens``'s fold, folded
    fresh per attempt -- the same fold the driver's token-budget watcher uses."""
    write_claude_log(tmp_path / "claude.attempt1.log", input_tokens=900, output_tokens=10)
    write_claude_log(tmp_path / "claude.log", input_tokens=1900, output_tokens=20)

    totals = token_cost.task_totals(tmp_path)

    def folded(name: str) -> int:
        return token_cost.accumulate_total_tokens((tmp_path / name).read_text(encoding="utf-8").splitlines(), {})

    assert totals.tokens_billed == folded("claude.log")
    assert totals.tokens_billed_crashed == folded("claude.attempt1.log")


def test_skipping_lines_that_cannot_carry_usage_changes_no_total(token_cost, tmp_path: pathlib.Path) -> None:
    """The fold decodes only lines that can hold usage, to avoid parsing megabytes of tool output. A
    line that merely mentions "assistant" is decoded and rejected, stderr and a half-written tail are
    skipped, and every real turn, thinking delta and result is still read -- so the totals are the
    cost model's own numbers, computed here by hand."""
    lines = [
        assistant_line("m1", 1000, 0),
        json.dumps({"type": "user", "note": "assistant", "message": {"content": "result usage 999999"}}),
        json.dumps({"type": "system", "subtype": "thinking_tokens", "estimated_tokens_delta": 40}),
        assistant_line("m1", 1000, 0),
        assistant_line("m2", 1500, 0),
        "stderr: warning, not json",
        result_line(70),
        '{"type": "assistant", "message": {"id": "m3"',
    ]
    log = tmp_path / "claude.log"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cost = token_cost.episode_cost(log)
    fresh, cached, generated = 1000 + (1500 - 1000), 1000, 70 + 40
    assert (cost["fresh_input"], cost["cached_input"], cost["output"], cost["thinking"]) == (fresh, cached, 70, 40)
    assert cost["effective"] == fresh + token_cost.CACHE_DISCOUNT * cached + generated
    assert token_cost.accumulate_total_tokens(lines, {}) == 1000 + 1500
    totals = token_cost.task_totals(tmp_path)
    assert (totals.tokens_effective, totals.tokens_billed) == (int(cost["effective"]), 1000 + 1500)


def test_a_worker_dir_without_a_transcript_has_no_token_total(token_cost, tmp_path: pathlib.Path) -> None:
    """A worker directory the driver never entered is not a task that cost 0 tokens -- it has no
    measurement at all, same as R7 treats a missing kernel token total."""
    totals = token_cost.task_totals(tmp_path)

    assert totals.attempts == 0
    assert totals.tokens_effective is None
    assert totals.tokens_billed is None


def test_a_runner_harnesss_totals_come_from_its_final_attempts_usage_file(token_cost, tmp_path: pathlib.Path) -> None:
    """A non-Claude harness (mini-SWE, OpenHands, Optimas) leaves ``usage.jsonl`` per attempt, moved
    aside on a crash the same way ``claude.log`` is (harnesses.py's ``records``), and is read under
    the same rule: the final attempt is the task, the earlier ones are crashed spend."""
    (tmp_path / "usage.attempt1.jsonl").write_text(usage_line(300, 30) + "\n", encoding="utf-8")
    (tmp_path / "usage.jsonl").write_text(usage_line(700, 70) + "\n", encoding="utf-8")

    totals = token_cost.task_totals(tmp_path)

    assert totals.attempts == 2
    assert (totals.tokens_effective, totals.tokens_billed) == (700 + 70, 770)
    assert (totals.tokens_effective_crashed, totals.tokens_billed_crashed) == (300 + 30, 330)


@pytest.mark.parametrize(
    ("name", "expected"),
    [("usage.jsonl", True), ("usage.attempt1.log", False), ("claude.log", False), ("usage.attempt3.jsonl", True)],
)
def test_is_usage_transcript_recognizes_a_renamed_crashed_attempt(token_cost, name: str, expected: bool) -> None:
    assert token_cost.is_usage_transcript(pathlib.Path(name)) is expected
