# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``token_cost``: the TASK TOKEN TOTAL (T2) of one worker directory, and what one episode's OUTPUT is.

A relaunched task's attempts are separate transcripts -- the driver moves a crashed attempt's log
aside (``claude.attemptN.log``) and starts the next one from an empty context AND an empty workspace
(T5) -- so the task is what its FINAL attempt did and the earlier attempts are reported beside it as
spend. A worker directory the driver never entered must not read as a task that cost 0.

The other property here is the OUTPUT RULE (8.2, F8): both engines serve ``/v1/messages`` with an
``output_tokens`` that already counts reasoning, so the streamed thinking estimate is informational
and is never added to anything.
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


def test_a_legacy_dir_without_attempts_jsonl_also_counts_only_its_final_attempt(
    token_cost: ModuleType, tmp_path: pathlib.Path
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


def test_a_legacy_dirs_final_attempt_starts_when_the_crash_was_moved_aside(
    token_cost: ModuleType, tmp_path: pathlib.Path
) -> None:
    """Old runs carry no ledger, so the cut X7 applies comes from the rename: the moved-aside
    transcript was renamed while the crash was handled, which is where the final attempt began."""
    crashed = tmp_path / "claude.attempt1.log"
    write_claude_log(crashed, input_tokens=1000, output_tokens=100)
    write_claude_log(tmp_path / "claude.log", input_tokens=2000, output_tokens=200)
    os.utime(crashed, (1_700_000_000, 1_700_000_000))

    assert token_cost.task_totals(tmp_path).final_attempt_start_ms == 1_700_000_000_000


def test_a_fresh_relaunched_task_counts_only_its_final_attempt_and_reports_the_rest_as_crashed(
    token_cost: ModuleType, tmp_path: pathlib.Path
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


def test_a_single_attempt_tasks_total_is_that_attempts_own(token_cost: ModuleType, tmp_path: pathlib.Path) -> None:
    write_claude_log(tmp_path / "claude.log", input_tokens=500, output_tokens=50)

    totals = token_cost.task_totals(tmp_path)

    assert totals.attempts == 1
    assert totals.tokens_effective == 550
    assert totals.tokens_billed == 500


def test_effective_comes_from_episode_cost_not_a_reimplementation(
    token_cost: ModuleType, tmp_path: pathlib.Path
) -> None:
    """The per-attempt effective figure must be exactly ``episode_cost(log)["effective"]`` summed,
    the one implementation of the cost model -- not a second copy that can drift from it."""
    write_claude_log(tmp_path / "claude.attempt1.log", input_tokens=900, output_tokens=10)
    write_claude_log(tmp_path / "claude.log", input_tokens=1900, output_tokens=20)

    totals = token_cost.task_totals(tmp_path)

    def effective(name: str) -> int:
        return int(token_cost.episode_cost(tmp_path / name)["effective"])

    assert totals.tokens_effective == effective("claude.log")
    assert totals.tokens_effective_crashed == effective("claude.attempt1.log")


def test_billed_comes_from_accumulate_total_tokens_not_a_reimplementation(
    token_cost: ModuleType, tmp_path: pathlib.Path
) -> None:
    """The per-attempt billed figure must be exactly ``accumulate_total_tokens``'s fold, folded
    fresh per attempt -- the same fold the driver's token-budget watcher uses."""
    write_claude_log(tmp_path / "claude.attempt1.log", input_tokens=900, output_tokens=10)
    write_claude_log(tmp_path / "claude.log", input_tokens=1900, output_tokens=20)

    totals = token_cost.task_totals(tmp_path)

    def folded(name: str) -> int:
        return token_cost.accumulate_total_tokens((tmp_path / name).read_text(encoding="utf-8").splitlines(), {})

    assert totals.tokens_billed == folded("claude.log")
    assert totals.tokens_billed_crashed == folded("claude.attempt1.log")


def test_skipping_lines_that_cannot_carry_usage_changes_no_total(
    token_cost: ModuleType, tmp_path: pathlib.Path
) -> None:
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
    # The three readings of ONE fold (USER 2026-09-16): free (cache reads at 0), provider (cache reads
    # at a tenth, what a hosted service meters) and billed (every prompt in full). 1500 + 0.1 * 1000 + 70.
    assert cost["effective_provider"] == fresh + 0.1 * cached + 70 == 1670
    assert cost["effective"] < cost["effective_provider"] < cost["naive_total"]
    assert token_cost.accumulate_total_tokens(lines, {}) == 1000 + 1500
    totals = token_cost.task_totals(tmp_path)
    assert (totals.tokens_effective, totals.tokens_billed) == (int(cost["effective"]), 1000 + 1500)
    assert totals.tokens_provider == 1670
    # The components a cost card weights ride on the task, and they are NOT recoverable from billed:
    # billed (2500) carries no output here, so billed - effective is not the cached count.
    assert (totals.tokens_fresh_input, totals.tokens_cached_input, totals.tokens_output) == (fresh, cached, 70)
    assert totals.tokens_billed - totals.tokens_effective != cached


def test_the_streamed_thinking_estimate_is_reported_but_never_added_to_the_effective_total(
    token_cost: ModuleType, tmp_path: pathlib.Path
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
    token_cost: ModuleType, tmp_path: pathlib.Path
) -> None:
    """A timed-out episode has no ``result`` record, and the per-turn events carry output_tokens: 0
    on these endpoints -- so its output is unknown, not measured as nothing. ``output_source`` says
    which tier counted it, because a reader that averages a silence in with a measurement reports
    every timeout arm as having generated less than it did."""
    write_claude_log(tmp_path / "claude.log", input_tokens=500, output_tokens=50)
    killed = tmp_path / "claude.attempt1.log"
    killed.write_text(assistant_line("m1", 700, 0) + "\nagent_driver: killed after AGENT_TIMEOUT_SECONDS\n")

    assert token_cost.episode_cost(killed)["output_source"] == "none"
    assert token_cost.episode_cost(tmp_path / "claude.log")["output_source"] == "result"


def message_start(message_id: str) -> str:
    """The CLI's ``stream_event`` envelope around an Anthropic ``message_start``.

    CRAFTED from the documented streaming event shapes rather than captured: these events only
    appear under ``--include-partial-messages``, which no recorded campaign ran, and this repository
    has no cluster to record a new one from.
    """
    return json.dumps(
        {
            "type": "stream_event",
            "event": {"type": "message_start", "message": {"id": message_id, "role": "assistant", "usage": {}}},
            "session_id": "s1",
        }
    )


def message_delta(output_tokens: int) -> str:
    """The ``message_delta`` that closes a request, whose usage is that request's running total.
    Crafted from the documented event shapes, for the same reason as :func:`message_start`."""
    return json.dumps(
        {
            "type": "stream_event",
            "event": {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": output_tokens},
            },
            "session_id": "s1",
        }
    )


def test_per_request_usage_outranks_the_episodes_result_record(token_cost: ModuleType, tmp_path: pathlib.Path) -> None:
    """PRECEDENCE (8.2). ``--include-partial-messages`` makes the server report each REQUEST's
    output as it finishes, which is the only exact count an episode killed at its wall leaves
    behind. Where both exist the per-request sum wins: it is the same server counting the same
    tokens, and it survives the kill that the result record does not."""
    log = tmp_path / "claude.log"
    log.write_text(
        "\n".join(
            [
                message_start("m1"),
                assistant_line("m1", 1000, 0),
                message_delta(120),
                message_start("m2"),
                assistant_line("m2", 1500, 0),
                message_delta(80),
                result_line(999),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cost = token_cost.episode_cost(log)

    assert cost["output"] == 200 and cost["output_source"] == "message_delta"
    assert cost["output_delta_shape"] == "cumulative"
    assert cost["effective"] == 1500 + 200


def test_a_killed_episode_still_has_its_per_request_counts(token_cost: ModuleType, tmp_path: pathlib.Path) -> None:
    """The reason for the flag: no result record, and the output is still the server's own number."""
    log = tmp_path / "claude.log"
    log.write_text(
        "\n".join([message_start("m1"), assistant_line("m1", 700, 0), message_delta(140)]) + "\n",
        encoding="utf-8",
    )

    cost = token_cost.episode_cost(log)

    assert (cost["output"], cost["output_source"]) == (140, "message_delta")


def test_per_chunk_increments_are_summed_and_a_running_total_is_not(
    token_cost: ModuleType, tmp_path: pathlib.Path
) -> None:
    """A message_delta's usage is the running total for its message, so several readings of one
    request are one number and not several. A CLI that streamed per-chunk increments instead would
    be counted correctly by summing -- which is what a non-monotone series gets. The shape that was
    seen is recorded, because the two are told apart by a heuristic and not by the protocol."""
    cumulative = tmp_path / "cumulative.log"
    cumulative.write_text(
        "\n".join([message_start("m1"), assistant_line("m1", 100, 0), message_delta(40), message_delta(90)]) + "\n",
        encoding="utf-8",
    )
    increments = tmp_path / "increments.log"
    increments.write_text(
        "\n".join([message_start("m1"), assistant_line("m1", 100, 0), message_delta(90), message_delta(40)]) + "\n",
        encoding="utf-8",
    )

    grew = token_cost.episode_cost(cumulative)
    fell = token_cost.episode_cost(increments)

    assert (grew["output"], grew["output_delta_shape"]) == (90, "cumulative")
    assert (fell["output"], fell["output_delta_shape"]) == (130, "increment")


def test_the_retokenized_tier_is_reached_only_when_the_server_counted_nothing(
    token_cost: ModuleType, tmp_path: pathlib.Path
) -> None:
    """The model's own tokenizer is the LAST tier: it counts what the transcript says was generated,
    which is the model's work but not the server's arithmetic (2-4 percent low, measured). So it is
    consulted for an attempt with no result record and never allowed to override one."""
    killed = tmp_path / "claude.attempt1.log"
    killed.write_text(assistant_line("m1", 700, 0) + "\n", encoding="utf-8")
    finished = tmp_path / "claude.log"
    write_claude_log(finished, input_tokens=500, output_tokens=50)

    counted = token_cost.episode_cost(killed, lambda events: 321)
    assert (counted["output"], counted["output_source"]) == (321, "retokenized")
    kept = token_cost.episode_cost(finished, lambda events: 321)
    assert (kept["output"], kept["output_source"]) == (50, "result")


def test_a_counter_that_cannot_count_leaves_the_output_unmeasured(
    token_cost: ModuleType, tmp_path: pathlib.Path
) -> None:
    """A model whose tokenizer is not in the offline cache gives None, not 0: an attempt nobody
    could count must keep saying so rather than join the measurements at zero."""
    killed = tmp_path / "claude.log"
    killed.write_text(assistant_line("m1", 700, 0) + "\n", encoding="utf-8")

    cost = token_cost.episode_cost(killed, lambda events: None)

    assert (cost["output"], cost["output_source"]) == (0, "none")


def test_a_result_record_the_transcripts_own_content_overflows_is_flagged_not_replaced(
    token_cost: ModuleType, tmp_path: pathlib.Path
) -> None:
    """F9: on Qwen/SGLang some complete episodes report a result total far below what the transcript
    demonstrably contains -- 6,918 against 32,720 retokenized in the worst measured case, with every
    tool call answered and every message carrying usage. Unexplained, so the record STANDS and the
    row is flagged; substituting the bigger number would be preferring a guess to a measurement."""
    log = tmp_path / "claude.log"
    write_claude_log(log, input_tokens=500, output_tokens=100)

    suspect = token_cost.episode_cost(log, lambda events: 200)
    fine = token_cost.episode_cost(log, lambda events: 110)

    assert suspect["output_suspect"] == 1.0 and suspect["output"] == 100
    assert fine["output_suspect"] == 0.0


def test_the_clis_synthetic_placeholder_turn_is_not_a_turn(token_cost: ModuleType, tmp_path: pathlib.Path) -> None:
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


def test_a_runner_harnesss_output_counts_its_reasoning_once(token_cost: ModuleType, tmp_path: pathlib.Path) -> None:
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


def test_the_budget_fold_reads_a_partial_message_transcript_as_it_read_the_old_one(
    token_cost: ModuleType, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--include-partial-messages`` adds stream_event lines to every new transcript, and the token
    cap the agent is killed on folds that same file inside the container (``http_json``). The new
    lines are not assistant turns and must change no billed total, or every arm's budget moves the
    day the flag lands."""
    lines = [
        message_start("m1"),
        assistant_line("m1", 1000, 0),
        message_delta(120),
        assistant_line("m2", 1500, 0),
        result_line(200),
    ]
    log = tmp_path / "claude.log"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    monkeypatch.delenv("OPTARENA_USAGE_PATH", raising=False)
    monkeypatch.setenv("CLAUDE_LOG_PATH", str(log))

    assert token_cost.accumulate_total_tokens(lines, {}) == 1000 + 1500
    assert load_http_json().transcript_tokens() == 1000 + 1500


def test_the_container_tool_and_the_analysis_bill_a_turn_for_the_same_usage_fields(token_cost: ModuleType) -> None:
    """``containers/agent/tools/http_json.py`` ships standalone in the agent image and repeats this
    list rather than importing it (its own comment says so). A copy that drifts makes the token cap
    the agent is killed on and the cost the analysis reports two different quantities, silently."""
    assert load_http_json().USAGE_FIELDS == token_cost.USAGE_FIELDS


def test_a_worker_dir_without_a_transcript_has_no_token_total(token_cost: ModuleType, tmp_path: pathlib.Path) -> None:
    """A worker directory the driver never entered is not a task that cost 0 tokens -- it has no
    measurement at all, same as R7 treats a missing kernel token total."""
    totals = token_cost.task_totals(tmp_path)

    assert totals.attempts == 0
    assert totals.tokens_effective is None
    assert totals.tokens_billed is None


def test_a_runner_harnesss_totals_come_from_its_final_attempts_usage_file(
    token_cost: ModuleType, tmp_path: pathlib.Path
) -> None:
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
def test_is_usage_transcript_recognizes_a_renamed_crashed_attempt(
    token_cost: ModuleType, name: str, expected: bool
) -> None:
    assert token_cost.is_usage_transcript(pathlib.Path(name)) is expected


#: One optimas call as the OLD writer spelled it (``input`` = the WHOLE prompt, ``cached_input``
#: repeating a part of it) and as the fixed one does (``input`` = the prompt MINUS its cached part).
#: Same call either way: a 1000-token prompt of which 900 came from the prefix cache, 50 generated.
OPTIMAS_OVERLAPPING = {"input": 1000, "cached_input": 900, "output": 50, "reasoning": 0}
OPTIMAS_DISJOINT = {"input": 100, "cached_input": 900, "output": 50, "reasoning": 0, "prompt": 1000}
#: The same call on an EARLY turn: the uncached remainder exceeds the cached prefix, which is what an
#: old overlapping line looks like by magnitude alone. Only the ``prompt`` field tells them apart.
OPTIMAS_DISJOINT_EARLY = {"input": 900, "cached_input": 100, "output": 50, "reasoning": 0, "prompt": 1000}


def write_optimas_usage(worker_dir: pathlib.Path, records: list[dict[str, int]]) -> pathlib.Path:
    """``usage.jsonl`` beside the ``optimas.log`` that tells an offline reader which harness wrote it."""
    worker_dir.mkdir(parents=True, exist_ok=True)
    (worker_dir / "optimas.log").write_text("", encoding="utf-8")
    path = worker_dir / "usage.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def test_the_offline_fold_reads_an_old_optimas_line_and_a_new_one_as_the_same_call(
    token_cost: ModuleType, tmp_path: pathlib.Path
) -> None:
    """``episode.append_usage`` used to write the whole prompt as ``input`` AND repeat its cached
    part in ``cached_input``, breaking the disjoint contract every reader sums under. Lines in both
    spellings are already on disk, so the reader re-derives the disjoint prompt for the old ones --
    without it the overlapping line prices a 1000-token prompt at 1900."""
    old = token_cost.usage_episode_cost(write_optimas_usage(tmp_path / "old", [OPTIMAS_OVERLAPPING]))
    new = token_cost.usage_episode_cost(write_optimas_usage(tmp_path / "new", [OPTIMAS_DISJOINT]))

    early = token_cost.usage_episode_cost(write_optimas_usage(tmp_path / "early", [OPTIMAS_DISJOINT_EARLY]))

    assert old == new == early, "the same call written three ways must cost the same"
    assert old["naive_total"] == 1000 + 50, old
    assert old["effective"] == 1000 + 50, old


def test_only_optimas_gets_the_overlap_rule(token_cost: ModuleType, tmp_path: pathlib.Path) -> None:
    """mini-SWE and OpenHands go through ``runner_common.usage_line`` and never wrote the overlap, so
    a line of theirs whose uncached remainder happens to exceed its cached prefix keeps the contract
    reading. The harness is read off the runner's own log beside the file."""
    miniswe = tmp_path / "miniswe"
    miniswe.mkdir()
    (miniswe / "miniswe.log").write_text("", encoding="utf-8")
    path = miniswe / "usage.jsonl"
    path.write_text(json.dumps(OPTIMAS_OVERLAPPING) + "\n", encoding="utf-8")

    assert token_cost.overlapping_usage_line(path) is False
    assert token_cost.usage_episode_cost(path)["naive_total"] == 1000 + 900 + 50


def test_the_judge_column_reads_an_old_optimas_line_and_a_new_one_as_the_same_call(
    token_cost: ModuleType, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The container-side reader carries the SAME rule as the offline one (it ships standalone and
    cannot import it), keyed on ``$OPTARENA_HARNESS`` since it has no worker directory to look at.
    Both spellings must give one number, or an optimas run's ``tokens`` column jumps the day the
    writer was fixed."""
    http_json = load_http_json()
    old = write_optimas_usage(tmp_path / "old", [OPTIMAS_OVERLAPPING])
    new = write_optimas_usage(tmp_path / "new", [OPTIMAS_DISJOINT])
    early = write_optimas_usage(tmp_path / "early", [OPTIMAS_DISJOINT_EARLY])

    monkeypatch.setenv("OPTARENA_HARNESS", "optimas")
    assert http_json.usage_jsonl_tokens(str(old)) == 1000 + 50
    assert http_json.usage_jsonl_tokens(str(new)) == 1000 + 50
    assert http_json.usage_jsonl_tokens(str(early)) == 1000 + 50, "a fixed early-turn line is not an old one"

    # Another runner never wrote the overlap, so its lines keep the contract reading.
    monkeypatch.setenv("OPTARENA_HARNESS", "miniswe")
    assert http_json.usage_jsonl_tokens(str(old)) == 1000 + 900 + 50


def test_the_container_tool_and_the_analysis_agree_on_the_overlap_rule(
    token_cost: ModuleType, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two copies of one rule, in two files that cannot import each other (the deliberate-duplication
    note on ``USAGE_FIELDS``). Duplication nothing compares is duplication that drifts, and a drift
    here makes the token cap the agent is killed on and the cost the analysis reports disagree.

    Compared on the quantity both derive: the call's whole prompt, counted once.
    """
    http_json = load_http_json()
    monkeypatch.setenv("OPTARENA_HARNESS", "optimas")
    cases = (
        OPTIMAS_OVERLAPPING,
        OPTIMAS_DISJOINT,
        OPTIMAS_DISJOINT_EARLY,
        {"input": 0, "cached_input": 0, "output": 1, "reasoning": 0},
    )
    for index, record in enumerate(cases):
        path = write_optimas_usage(tmp_path / f"case{index}", [record])
        offline = token_cost.usage_prompt_tokens(record, token_cost.overlapping_usage_line(path))
        container = http_json.usage_jsonl_tokens(str(path)) - record["output"] - record["reasoning"]
        assert offline == container, record


def test_a_compaction_charges_the_rebuilt_prompt_as_fresh_and_is_counted(
    token_cost: ModuleType, tmp_path: pathlib.Path
) -> None:
    """USER 2026-09-16: the claude arms compact under a CLAUDE_AUTOCOMPACT wall. After a compaction the
    prompt is SHORTER than the previous one and shares no prefix with it -- a full cache miss -- so the
    whole rebuilt prompt is fresh. The old fold charged it at zero (max(0, 400 - 1500))."""
    lines = [
        assistant_line("m1", 1000, 0),
        assistant_line("m2", 1500, 0),
        assistant_line("m3", 400, 0),
        result_line(70),
    ]
    log = tmp_path / "claude.log"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cost = token_cost.episode_cost(log)

    assert (cost["fresh_input"], cost["cached_input"], cost["compactions"]) == (1000 + 500 + 400, 1000, 1)
    assert cost["naive_total"] == 1000 + 1500 + 400 + 70
    assert cost["effective"] == 1900 + 70
    assert token_cost.fold_prompt(400, 1500) == (400, 0, 1)
    assert token_cost.fold_prompt(1500, 1000) == (500, 1000, 0)
