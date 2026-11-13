# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""hpcagent_bench.results.transcripts: each harness's extract against an episode it recorded.

The fixtures under ``tests/fixtures/transcripts/<harness>/`` are worker directories of recorded episodes (harness20
for Claude Code, mini-SWE-agent and OpenHands, a smoke run for the tuner), trimmed: tool results, prompts and tool
descriptions cut, Claude Code's content-block deltas and tool heartbeats dropped and its runs of ``thinking_tokens``
events merged (their sum kept), the OpenHands state reduced to its usage, site paths and host names replaced. Every
model output (reasoning, text, tool call), every usage count and every record of the harness's own token counts is as
recorded."""

import hashlib
import json
import pathlib
import shutil

import pytest
from hpcagent_agent.driver import harnesses, token_cost

from hpcagent_bench import paths
from hpcagent_bench.results import transcripts

FIXTURES = paths.ROOT / "tests" / "fixtures" / "transcripts"

#: Each fixture's model output in transcript order, read off its files by hand: ``(kind, text prefix)`` for reasoning
#: and text, ``(kind, id, name)`` for a tool call.
EXPECTED: dict[str, list[tuple[str, ...]]] = {
    "claude": [
        ("reasoning", "We need to understand the problem. It's "),
        ("tool_call", "chatcmpl-tool-add5aac6f01ef4b0", "Read"),
        ("reasoning", "We need to pass correct parameters: file"),
        ("tool_call", "chatcmpl-tool-93ebbd9423697f07", "Read"),
        ("reasoning", "The signature defines the function gemm_"),
        ("tool_call", "chatcmpl-tool-8788020b0b60accd", "Read"),
        ("reasoning", "The reference implementation uses malloc"),
        ("tool_call", "chatcmpl-tool-b373b211fbf30e55", "Bash"),
        ("reasoning", "Now run syntax_check on gemm.c.\n\n"),
        ("tool_call", "chatcmpl-tool-a1c83f5a732ca9f8", "mcp__hpcagent_bench__syntax_check"),
        ("reasoning", "Compilation ok, no warnings.\n\nNow test t"),
        ("tool_call", "chatcmpl-tool-85dfd4e63cc5b0dc", "mcp__hpcagent_bench__score"),
        ("reasoning", "The score shows it is correct and speedu"),
        ("tool_call", "chatcmpl-tool-95c194bf1e43085f", "mcp__hpcagent_bench__submit"),
        ("text", "Implemented an optimized GEMM kernel in "),
    ],
    "miniswe": [
        ("reasoning", "We need to optimize segment_reduce_ragge"),
        ("tool_call", "chatcmpl-tool-8098b6134b1cc2ff", "bash"),
        ("reasoning", "Let's view signature.json."),
        ("tool_call", "chatcmpl-tool-8f39dfc09795e073", "bash"),
        ("reasoning", "The function signature is:\n\nvoid segment"),
        ("tool_call", "chatcmpl-tool-99a237d4aa135a06", "bash"),
        ("reasoning", "The reference kernel does a simple reduc"),
        ("tool_call", "chatcmpl-tool-bc718e53a3cede6d", "bash"),
        ("reasoning", "Now run syntax check on this file to ens"),
        ("tool_call", "chatcmpl-tool-8fc7318bb731281a", "bash"),
        ("reasoning", "Compiles cleanly.\n\nNow need to evaluate "),
        ("tool_call", "chatcmpl-tool-b6bcb7916640a5eb", "bash"),
        # A reply the agent refused for its format: kept, then the harness's message.
        ("reasoning", "The result shows correct output and good"),
        ("text", "The kernel has been optimized by paralle"),
        ("reasoning", "We need to ensure we put at least one to"),
        ("tool_call", "chatcmpl-tool-95036b326bf63a72", "bash"),
        ("reasoning", "Great, speedup ~4.79x vs baseline. That'"),
        ("tool_call", "chatcmpl-tool-9a382545d12023aa", "bash"),
        ("reasoning", "The submission succeeded.\n\nNow final ste"),
        ("text", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTP"),
        ("reasoning", "Again tool calls not recognized? The too"),
        ("text", "The system responded with output. Then t"),
        ("tool_call", "chatcmpl-tool-9fdf1de02daa55df", "bash"),
        ("tool_call", "chatcmpl-tool-99fae8fb4c1ba493", "bash"),
    ],
    "openhands": [
        ("reasoning", "We need to optimize seissol_batched_gemm"),
        ("tool_call", "chatcmpl-tool-83d60020de50d7be", "file_editor"),
        ("reasoning", "We see a python reference and a C refere"),
        ("tool_call", "chatcmpl-tool-ae48ad7c539f6200", "file_editor"),
        ("reasoning", "We need to optimize the seissol_batched_"),
        ("tool_call", "chatcmpl-tool-93ffc5593bda0e07", "file_editor"),
        ("reasoning", "We need to optimize seissol_batched_gemm"),
        ("tool_call", "chatcmpl-tool-b2820d42c5ed78ec", "file_editor"),
        ("reasoning", "We need to optimize seissol_batched_gemm"),
        ("tool_call", "chatcmpl-tool-87007a017f4a7a8f", "syntax_check"),
        ("reasoning", "The syntax error arises due to the funct"),
        ("tool_call", "chatcmpl-tool-a4171b69295e7340", "file_editor"),
        ("reasoning", "We need to address the syntax errors. Lo"),
        ("tool_call", "chatcmpl-tool-b26a717b9cb41f1f", "terminal"),
        ("reasoning", "The compilation error says \"expected ')'"),
        ("text", "<|start|>assistant<|channel|>analysis to"),
    ],
    "tuner": [],
}
#: Registered harnesses no recorded episode exists of: refused, never guessed.
NO_SAMPLE = frozenset({"openai", "anthropic", "optimas"})
#: The tool calls (and tool results) of the Claude Code fixture.
CLAUDE_CALLS = 7


def episode(harness: str, tmp_path: pathlib.Path) -> pathlib.Path:
    """A copy of the fixture worker directory of ``harness``."""
    worker = tmp_path / "agents" / "node-0" / "problem-0-worker-0"
    shutil.copytree(FIXTURES / harness, worker)
    return worker


def extracted(worker: pathlib.Path) -> list[transcripts.Record]:
    """The extract of ``worker``, written compressed and read back."""
    harness = transcripts.harness_of(worker)
    path = worker.parent / "extract.zst"
    path.write_bytes(transcripts.compress(transcripts.extract(worker, harness, transcripts.sources(worker, harness))))
    return transcripts.read(path)


def model_output(records: list[transcripts.Record]) -> list[tuple[str, ...]]:
    found: list[tuple[str, ...]] = []
    for record in records:
        if record["kind"] in ("reasoning", "text"):
            found.append((str(record["kind"]), str(record["text"])[:40]))
        elif record["kind"] == "tool_call":
            found.append(("tool_call", str(record["id"]), str(record["name"])))
    return found


@pytest.mark.parametrize("harness", sorted(EXPECTED))
def test_the_extract_holds_every_reasoning_block_and_tool_call_in_order(harness: str, tmp_path: pathlib.Path) -> None:
    """Every reasoning block, text and tool call (with its full input) of a recorded episode is in its extract, in the
    order the model produced them, under the harness's own extractor."""
    worker = episode(harness, tmp_path)
    assert transcripts.harness_of(worker) == harness
    records = extracted(worker)
    assert records[0]["harness"] == harness
    for found, want in zip(model_output(records), EXPECTED[harness], strict=True):
        assert found[0] == want[0]
        assert found[1].startswith(want[1]) if want[0] != "tool_call" else found == want
    assert all(record.get("input") for record in records if record["kind"] == "tool_call")


@pytest.mark.parametrize("harness", ["miniswe", "openhands", "tuner"])
def test_a_runners_extract_usage_sums_to_its_usage_jsonl(harness: str, tmp_path: pathlib.Path) -> None:
    """One usage record per model call, its four counts those the runner wrote to ``usage.jsonl``: the cost fold reads
    the same numbers off the extract alone."""
    worker = episode(harness, tmp_path)
    usage = [record for record in extracted(worker) if record["kind"] == "usage"]
    lines = (worker / harnesses.USAGE_FILE).read_text().splitlines()
    assert sorted(json.dumps({f: u[f] for f in harnesses.CONSUMED_FIELDS}) for u in usage) == sorted(
        json.dumps(json.loads(line)) for line in lines
    )
    as_lines = [json.dumps({f: u[f] for f in harnesses.CONSUMED_FIELDS}) for u in usage]
    assert token_cost.usage_lines_cost(as_lines) == token_cost.usage_episode_cost(worker / harnesses.USAGE_FILE)


def test_claude_token_accounting_runs_on_the_extract_alone(tmp_path: pathlib.Path) -> None:
    """The episode's token totals (T2) and its cost row (cached input under the perfect-prefix fold, output from the
    ``message_delta`` tier, T8/T9) come out the same from the stream logs :func:`transcripts.restore` writes from the
    extract as from the recorded logs, and equal the driver's ``tokens.json``."""
    worker = episode("claude", tmp_path)
    records = extracted(worker)
    restored = tmp_path / "restored"
    restored.mkdir()
    assert [p.name for p in transcripts.restore(records, restored)] == ["claude.log"]
    shutil.copy(worker / token_cost.ATTEMPTS_NAME, restored)
    assert token_cost.episode_totals(restored) == token_cost.episode_totals(worker)
    assert token_cost.episode_cost(restored / "claude.log") == token_cost.episode_cost(worker / "claude.log")
    recorded = json.loads((worker / "tokens.json").read_text())
    row = token_cost.episode_cost(restored / "claude.log")
    totals = token_cost.episode_totals(restored)
    assert (totals.tokens_effective, totals.tokens_billed) == (recorded["tokens_effective"], recorded["tokens_billed"])
    for key in ("fresh_input", "cached_input", "output", "thinking_estimate", "turns", "output_source"):
        assert row[key] == recorded[key], key


def test_claude_usage_records_fold_to_the_drivers_counts(tmp_path: pathlib.Path) -> None:
    """One usage record per request: their prompts folded by the perfect-prefix model and their outputs summed give
    the driver's ``tokens.json`` components."""
    worker = episode("claude", tmp_path)
    usage = [record for record in extracted(worker) if record["kind"] == "usage"]
    fresh = cached = previous = 0
    for record in usage:
        prompt = int(record["input"]) + int(record["cached_input"])
        fold = token_cost.fold_prompt(prompt, previous)
        fresh, cached, previous = fresh + fold.fresh, cached + fold.cached, prompt
    recorded = json.loads((worker / "tokens.json").read_text())
    assert (len(usage), fresh, cached) == (recorded["turns"], recorded["fresh_input"], recorded["cached_input"])
    assert sum(int(record["output"]) for record in usage) == recorded["output"]


def test_claude_stream_and_session_give_one_extract(tmp_path: pathlib.Path) -> None:
    """A turn in both the stream log and the session file is extracted once, from the log; the session adds only
    what no log holds (here the task prompt and the CLI's attachments), and none of its request metadata."""
    worker = episode("claude", tmp_path)
    records = extracted(worker)
    calls = [record["id"] for record in records if record["kind"] == "tool_call"]
    assert len(calls) == len(set(calls)) == CLAUDE_CALLS
    files = [record["path"] for record in records if record["kind"] == "file"]
    assert files[0] == "claude.log"
    assert files[1].startswith(str(transcripts.CLAUDE_SESSIONS))
    session = records[records.index({"kind": "file", "path": files[1]}) + 1 :]
    assert [record["kind"] for record in session] == ["event", "event", "event", "prompt"]
    assert all(record["event"]["type"] == "attachment" for record in session[:3])
    assert not any(
        isinstance(record.get("event"), dict) and record["event"].get("type") in transcripts.SESSION_METADATA
        for record in records
    )


def tool_results(worker: pathlib.Path) -> list[object]:
    """Every tool result of the Claude Code fixture's stream log, as the model received it."""
    found: list[object] = []
    for line in (worker / "claude.log").read_text().splitlines():
        event = json.loads(line) if line.startswith("{") else {}
        if event.get("type") == "user" and isinstance(event["message"]["content"], list):
            found += [block["content"] for block in event["message"]["content"] if block["type"] == "tool_result"]
    return found


def test_a_tool_result_is_kept_as_its_length_and_hash_only(tmp_path: pathlib.Path) -> None:
    """Each tool result is its id, byte length and sha256 in the extract, never its text."""
    worker = episode("claude", tmp_path)
    records = extracted(worker)
    results = [record for record in records if record["kind"] == "tool_result"]
    recorded = tool_results(worker)
    assert len(results) == len(recorded) == CLAUDE_CALLS
    for record, content in zip(results, recorded, strict=True):
        data = content.encode() if isinstance(content, str) else json.dumps(content, separators=(",", ":")).encode()
        assert (record["bytes"], record["sha256"]) == (len(data), hashlib.sha256(data).hexdigest())
    text = transcripts.decompress((worker.parent / "extract.zst").read_bytes()).decode()
    assert not any(isinstance(content, str) and content in text for content in recorded)


def test_claude_attempts_and_resumes_carry_their_ledger_lines(tmp_path: pathlib.Path) -> None:
    """A crashed attempt's log (``claude.attempt1.log``), then a log the driver resumed in: an attempt record at the
    top of each log and at the resume line, each with its ``attempts.jsonl`` line, and the token totals of the restored
    logs the episode's own."""
    worker = episode("claude", tmp_path)
    log = (worker / "claude.log").read_text().splitlines(keepends=True)
    shutil.copy(worker / "claude.log", worker / "claude.attempt1.log")
    resume = json.dumps(
        {"type": "system", "subtype": "driver_resume", "session_id": "s", "returncode": 1, "attempt": 2}
    )
    (worker / "claude.log").write_text("".join([*log[:20], resume + "\n", *log[20:]]))
    ledger = [
        {"attempt": 1, "cleared": True, "start_ms": 1},
        {"attempt": 2, "cleared": False, "resumed": True, "start_ms": 2},
        {"attempt": 3, "cleared": False, "start_ms": 3},
    ]
    (worker / token_cost.ATTEMPTS_NAME).write_text("".join(json.dumps(line) + "\n" for line in ledger))
    records = extracted(worker)
    marks = [record for record in records if record["kind"] == "attempt"]
    assert marks == [{"kind": "attempt", **line} for line in ledger]
    restored = tmp_path / "restored"
    restored.mkdir()
    assert [p.name for p in transcripts.restore(records, restored)] == ["claude.attempt1.log", "claude.log"]
    shutil.copy(worker / token_cost.ATTEMPTS_NAME, restored)
    assert token_cost.episode_totals(restored) == token_cost.episode_totals(worker)


def test_a_tuner_extract_holds_its_attempts_only(tmp_path: pathlib.Path) -> None:
    """A tuner calls no model and keeps no transcript: its extract reads no file and holds its attempt, marked as
    one without a transcript."""
    worker = episode("tuner", tmp_path)
    records = extracted(worker)
    assert records[0]["sources"] == []
    assert [record["kind"] for record in records[1:]] == ["attempt"]
    assert records[1]["transcript"] is False


def test_every_registered_harness_has_an_extractor_or_is_refused(tmp_path: pathlib.Path) -> None:
    """Each harness the driver can launch (a plugin's included) maps to its extractor, but the three no episode was
    ever recorded of; an episode of one of those is refused, not skipped."""
    names = set(harnesses.names())
    extracted_harnesses = set(transcripts.EXTRACTORS) - {""}
    assert extracted_harnesses == names - NO_SAMPLE
    assert NO_SAMPLE <= names
    worker = tmp_path / "w"
    worker.mkdir()
    for harness in NO_SAMPLE:
        (worker / "tokens.json").write_text(json.dumps({"identity": {"harness": harness}}))
        assert transcripts.harness_of(worker) == harness
        with pytest.raises(transcripts.NoExtractorError, match=harness):
            transcripts.sources(worker, harness)


def test_an_autokernel_episode_runs_on_claude_code_and_gets_its_extractor(tmp_path: pathlib.Path) -> None:
    """The AutoKernel workflow is a packet on Claude Code: its episode records harness ``claude`` and packet
    ``autokernel``, and its transcript is extracted by the Claude Code extractor (its ``.experiments/`` and
    ``results.tsv`` are no transcript)."""
    worker = episode("claude", tmp_path)
    tokens = json.loads((worker / "tokens.json").read_text())
    tokens["identity"]["packet"] = "autokernel"
    (worker / "tokens.json").write_text(json.dumps(tokens))
    (worker / ".experiments").mkdir()
    (worker / ".experiments" / "exp-1.c").write_text("void k(void) {}\n")
    (worker / "results.tsv").write_text("exp\tspeedup\n")
    assert transcripts.harness_of(worker) == harnesses.CLAUDE
    assert transcripts.EXTRACTORS[transcripts.harness_of(worker)] is transcripts.EXTRACTORS[harnesses.CLAUDE]
    names = {path.name for path in transcripts.sources(worker, harnesses.CLAUDE)}
    assert not names & {"exp-1.c", "results.tsv"}
    assert model_output(extracted(worker)) == model_output(extracted(episode("claude", tmp_path / "plain")))


def test_the_harness_of_a_worker_without_a_record_is_the_one_whose_log_is_there(tmp_path: pathlib.Path) -> None:
    """With no ``tokens.json`` (an episode killed before its end) the harness's log names it; none names none."""
    worker = tmp_path / "w"
    worker.mkdir()
    assert transcripts.harness_of(worker) == ""
    (worker / "openhands.attempt1.log").write_text("")
    assert transcripts.harness_of(worker) == "openhands"
