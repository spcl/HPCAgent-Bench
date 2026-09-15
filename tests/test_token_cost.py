# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``token_cost``: the TASK TOKEN TOTAL (T2), and what one episode's OUTPUT is.

A relaunched task's attempts are separate transcripts -- the driver moves a crashed attempt's log
aside (``claude.attemptN.log``) and starts the next one from an empty context (``agent_driver.py``'s
move-aside rule) -- so the task's total has to be the SUM over them, never just the last attempt's,
and a worker directory the driver never entered must not read as a task that cost 0.

The other property here is the OUTPUT RULE (8.1, F8): both engines serve ``/v1/messages`` with an
``output_tokens`` that already counts reasoning, so the streamed thinking estimate is informational
and is never added to anything.
"""

import importlib.util
import json
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


def usage_line(call_input: int, output: int, reasoning: int = 0) -> str:
    """One runner call's usage line, whose four counts are DISJOINT (``runner_common.usage_line``):
    ``output`` is the completion WITHOUT its reasoning, so the call generated their sum."""
    return json.dumps({"input": call_input, "cached_input": 0, "output": output, "reasoning": reasoning})


def load_http_json() -> ModuleType:
    """The container's ``http_json`` tool, loaded the way the container does: by path, stdlib only."""
    path = REPO / "containers" / "agent" / "tools" / "http_json.py"
    spec = importlib.util.spec_from_file_location("http_json", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_relaunched_tasks_totals_are_the_sum_over_attempts_not_the_last(token_cost, tmp_path: pathlib.Path) -> None:
    write_claude_log(tmp_path / "claude.attempt1.log", input_tokens=1000, output_tokens=100)
    write_claude_log(tmp_path / "claude.log", input_tokens=2000, output_tokens=200)

    totals = token_cost.task_totals(tmp_path)

    assert totals.attempts == 2
    # effective adds the result event's output; billed folds only the per-turn assistant usage,
    # which reports output_tokens: 0 on this endpoint (episode_cost's own docstring) -- so the two
    # totals differ by exactly the output component, which is the point of carrying both (8.1).
    assert totals.tokens_effective == 1000 + 100 + 2000 + 200
    assert totals.tokens_billed == 1000 + 2000


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

    want = sum(
        int(token_cost.episode_cost(log)["effective"])
        for log in (tmp_path / "claude.attempt1.log", tmp_path / "claude.log")
    )
    assert totals.tokens_effective == want


def test_billed_comes_from_accumulate_total_tokens_not_a_reimplementation(token_cost, tmp_path: pathlib.Path) -> None:
    """The per-attempt billed figure must be exactly ``accumulate_total_tokens``'s fold, folded
    fresh per attempt -- the same fold the driver's token-budget watcher uses."""
    write_claude_log(tmp_path / "claude.attempt1.log", input_tokens=900, output_tokens=10)
    write_claude_log(tmp_path / "claude.log", input_tokens=1900, output_tokens=20)

    totals = token_cost.task_totals(tmp_path)

    want = sum(
        token_cost.accumulate_total_tokens(log.read_text(encoding="utf-8").splitlines(), {})
        for log in (tmp_path / "claude.attempt1.log", tmp_path / "claude.log")
    )
    assert totals.tokens_billed == want


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
    fresh, cached = 1000 + (1500 - 1000), 1000
    assert (cost["fresh_input"], cost["cached_input"], cost["output"]) == (fresh, cached, 70)
    # The thinking delta is READ and reported, and is not part of any total: the result record's 70
    # output tokens already count it (module docstring, 3).
    assert cost["thinking_estimate"] == 40
    assert cost["effective"] == fresh + token_cost.CACHE_DISCOUNT * cached + 70
    assert token_cost.accumulate_total_tokens(lines, {}) == 1000 + 1500
    totals = token_cost.task_totals(tmp_path)
    assert (totals.tokens_effective, totals.tokens_billed) == (int(cost["effective"]), 1000 + 1500)


def test_the_streamed_thinking_estimate_is_reported_but_never_added_to_the_effective_total(
    token_cost, tmp_path: pathlib.Path
) -> None:
    """Both engines' ``/v1/messages`` fills ``output_tokens`` with EVERY generated token -- reasoning,
    answer text and tool arguments alike -- so adding the client's ``estimated_tokens_delta`` on top
    charged the same reasoning twice (13/F8). Measured on job 636540 problem-0 the estimate was
    34,517 against a server output of 24,153, which is why the old effective ran ~1.4x high.

    The property CHANGED here: before this, ``effective`` was ``fresh + output + thinking``.
    """
    log = tmp_path / "claude.log"
    log.write_text(
        "\n".join(
            [
                assistant_line("m1", 1000, 0),
                json.dumps({"type": "system", "subtype": "thinking_tokens", "estimated_tokens_delta": 900}),
                result_line(300),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cost = token_cost.episode_cost(log)

    assert cost["output"] == 300 and cost["thinking_estimate"] == 900
    assert cost["effective"] == 1000 + 300, "the 900 is the same tokens the 300 already counts"
    assert token_cost.task_totals(tmp_path).tokens_effective == 1300


def test_an_episode_whose_server_never_reported_output_says_so_instead_of_claiming_zero(
    token_cost, tmp_path: pathlib.Path
) -> None:
    """A timed-out episode has no ``result`` record, and the per-turn events carry output_tokens: 0
    on these endpoints -- so its output is unknown, not measured as nothing. ``output_reported``
    separates the two, because a reader that averages a silence in with a measurement reports every
    timeout arm as having generated less than it did."""
    write_claude_log(tmp_path / "claude.log", input_tokens=500, output_tokens=50)
    killed = tmp_path / "claude.attempt1.log"
    killed.write_text(assistant_line("m1", 700, 0) + "\nagent_driver: killed after AGENT_TIMEOUT_SECONDS\n")

    assert token_cost.episode_cost(killed)["output_reported"] == 0.0
    assert token_cost.episode_cost(tmp_path / "claude.log")["output_reported"] == 1.0


def test_the_clis_synthetic_placeholder_turn_is_not_a_turn(token_cost, tmp_path: pathlib.Path) -> None:
    """When the endpoint answers nothing the CLI appends its own assistant message, model
    ``<synthetic>``, carrying a usage block of zeros. Folded as a turn it says the context shrank to
    nothing, so the next real turn's whole prompt is charged as fresh a second time -- and it ends
    the transcript as a turn that generated nothing, hiding the last real one."""
    lines = [
        assistant_line("m1", 1000, 0),
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "id": "synthetic-1",
                    "model": "<synthetic>",
                    "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0},
                    "content": [{"type": "text", "text": "API Error: The operation timed out."}],
                },
            }
        ),
        assistant_line("m2", 1500, 0),
        result_line(70),
    ]
    log = tmp_path / "claude.log"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cost = token_cost.episode_cost(log)

    assert cost["turns"] == 2, "the placeholder is not a model turn"
    # 1000 fresh, then 500 more: without the skip the placeholder resets the prefix and m2's whole
    # 1500 is charged fresh again.
    assert (cost["fresh_input"], cost["cached_input"]) == (1500, 1000)
    assert cost["effective"] == 1500 + 70


def test_a_runner_harnesss_output_counts_its_reasoning_once(token_cost, tmp_path: pathlib.Path) -> None:
    """The runner splits what the chat-completions API reports as one number: ``completion_tokens``
    counts every generated token in both engines, and ``runner_common.usage_line`` writes it out as
    ``output`` (without reasoning) plus ``reasoning``. So the episode's output is their SUM -- the
    same quantity the claude fold reads off ``output_tokens`` -- and reasoning is never a third
    addend."""
    (tmp_path / "usage.jsonl").write_text(usage_line(400, 60, reasoning=40) + "\n", encoding="utf-8")

    cost = token_cost.episode_cost(tmp_path / "usage.jsonl")

    assert cost["output"] == 100 and cost["thinking_estimate"] == 40
    assert cost["effective"] == 400 + 100
    assert cost["naive_total"] == 400 + 100


def test_the_container_tool_and_the_analysis_bill_a_turn_for_the_same_usage_fields(token_cost) -> None:
    """``containers/agent/tools/http_json.py`` ships standalone in the agent image and repeats this
    list rather than importing it (its own comment says so). A copy that drifts makes the token cap
    the agent is killed on and the cost the analysis reports two different quantities, silently."""
    assert load_http_json().USAGE_FIELDS == token_cost.USAGE_FIELDS


def test_a_worker_dir_without_a_transcript_has_no_token_total(token_cost, tmp_path: pathlib.Path) -> None:
    """A worker directory the driver never entered is not a task that cost 0 tokens -- it has no
    measurement at all, same as R7 treats a missing kernel token total."""
    totals = token_cost.task_totals(tmp_path)

    assert totals.attempts == 0
    assert totals.tokens_effective is None
    assert totals.tokens_billed is None


def test_a_runner_harnesss_totals_sum_its_attempt_usage_files(token_cost, tmp_path: pathlib.Path) -> None:
    """A non-Claude harness (mini-SWE, OpenHands, Optimas) leaves ``usage.jsonl`` per attempt, moved
    aside on a crash the same way ``claude.log`` is (harnesses.py's ``records``)."""
    (tmp_path / "usage.attempt1.jsonl").write_text(usage_line(300, 30) + "\n", encoding="utf-8")
    (tmp_path / "usage.jsonl").write_text(usage_line(700, 70) + "\n", encoding="utf-8")

    totals = token_cost.task_totals(tmp_path)

    assert totals.attempts == 2
    assert totals.tokens_effective == 300 + 30 + 700 + 70
    assert totals.tokens_billed == 330 + 770


@pytest.mark.parametrize(
    ("name", "expected"),
    [("usage.jsonl", True), ("usage.attempt1.log", False), ("claude.log", False), ("usage.attempt3.jsonl", True)],
)
def test_is_usage_transcript_recognizes_a_renamed_crashed_attempt(token_cost, name: str, expected: bool) -> None:
    assert token_cost.is_usage_transcript(pathlib.Path(name)) is expected
