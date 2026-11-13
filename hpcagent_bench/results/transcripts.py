# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One episode's transcript extract: what ``collect copy`` keeps of an agent's transcripts unless told to keep them
whole (``--full-transcripts``).

Per worker directory one zstd-compressed JSONL, :data:`EXTRACT_NAME`, read from the harness's own transcript files
(:data:`EXTRACTORS`, each derived from the harness's writer and from recorded episodes) in transcript order. Its first
line is the header (``kind`` ``extract``: :data:`SCHEMA`, the harness, every source file with its size and sha256);
every other line is one record, by ``kind``:

=============  ==========================================================================================
``file``       the records below come from this source file (``path``, relative to the worker directory)
``attempt``    an attempt, or a resumed leg of one, starts here: its ``attempts.jsonl`` line when known;
               ``transcript: false`` for an attempt the harness kept no transcript of
``prompt``     what the model was given: the system prompt, the task, a resume prompt, a harness message
``text``       text the model wrote
``reasoning``  the model's reasoning (``redacted: true`` for an encrypted block)
``tool_call``  ``id``, ``name`` and the full ``input`` as the harness recorded it (an object for Claude Code,
               the arguments' JSON text for the OpenAI-format harnesses)
``tool_result``  ``id``, ``bytes`` and ``sha256`` of the result as the model received it (:func:`digest`), and
               ``error`` when the harness says; never the result itself
``usage``      one model call: ``id``, ``input`` (uncached), ``cached_input``, ``output``, and ``reasoning`` where the
               harness counts it apart from ``output`` (the four disjoint counts of ``usage.jsonl``)
``event``      a harness record kept whole: Claude Code's token-accounting stream events (an ``assistant`` event
               without its content), and records of a kind this module does not split
``log``        a line of a stream log that is not JSON (a driver note, a CLI warning)
=============  ==========================================================================================

Records carry the harness's own timestamp as ``ts`` where it writes one. Claude Code's ``event`` records are every
event :mod:`hpcagent_agent.driver.token_cost` reads, so :func:`restore` gives back stream logs whose token totals equal
the originals'.
"""

import collections
import contextlib
import dataclasses
import hashlib
import json
import pathlib
import shutil
import subprocess
from collections.abc import Callable, Iterator, Sequence
from typing import cast

from hpcagent_agent.driver import harnesses, token_cost
from hpcagent_agent.driver.token_cost import as_block
from hpcagent_agent.harness import run_miniswe, run_openhands, runner_common

from hpcagent_bench.run_layout import CLAUDE_LOG, TOKENS_NAME

__all__ = [
    "CLAUDE_SESSIONS",
    "DROPPED_EVENTS",
    "EXTRACTORS",
    "EXTRACT_NAME",
    "REQUEST_EVENTS",
    "SCHEMA",
    "SESSION_METADATA",
    "THINKING_SUBTYPE",
    "ZSTD_LEVEL",
    "Extractor",
    "NoExtractorError",
    "Record",
    "Request",
    "Requests",
    "as_list",
    "claude_records",
    "claude_sources",
    "claude_usage",
    "compress",
    "content_records",
    "decompress",
    "digest",
    "encode",
    "extract",
    "extractor",
    "file_record",
    "file_sha",
    "harness_of",
    "history",
    "logs",
    "marked",
    "miniswe_records",
    "miniswe_sources",
    "no_records",
    "no_sources",
    "openai_reply",
    "openai_usage",
    "openhands_event",
    "openhands_events",
    "openhands_reasoning",
    "openhands_records",
    "openhands_sources",
    "openhands_usages",
    "parsed",
    "read",
    "restore",
    "session_records",
    "sources",
    "stamped",
    "stream_event_records",
    "stream_logs",
    "stream_records",
    "texts",
    "thinking",
    "unattributed",
    "usage_key",
    "user_records",
    "zstd",
]

#: One line of an extract.
type Record = dict[str, object]

#: The extract's file name in a worker directory of a copy.
EXTRACT_NAME = "transcript.extract.jsonl.zst"
#: The extract format; a reader refuses another.
SCHEMA = 1
#: zstd's level for an extract: written once, read rarely.
ZSTD_LEVEL = 19

#: Where Claude Code keeps its sessions (its subagents' too) in a worker directory.
CLAUDE_SESSIONS = pathlib.PurePosixPath("home/.claude/projects")
#: Claude Code session records of no turn: request parameters with the system prompt and tool schemas
#: (``api-request``, ``api-request-shape``), the file-backup index (``file-history-snapshot``), the queued prompt
#: (``queue-operation``, the task's ``prompt.txt``), the CLI's cost tally (``cost-state``, the stream's ``result``) and
#: its pointers (``atis-latch``, ``last-prompt``).
SESSION_METADATA = frozenset(
    {
        "api-request",
        "api-request-shape",
        "file-history-snapshot",
        "queue-operation",
        "cost-state",
        "atis-latch",
        "last-prompt",
    }
)
#: Stream events of one request that :mod:`token_cost` reads; the content-block deltas only repeat what the
#: ``assistant`` events carry whole.
REQUEST_EVENTS = frozenset({"message_start", "message_delta", "message_stop"})
#: Claude Code's streamed estimate of the thinking so far, one event per few tokens: a run of them is kept as its last
#: event carrying the run's summed ``estimated_tokens_delta`` (the one field :mod:`token_cost` reads).
THINKING_SUBTYPE = "thinking_tokens"
#: Claude Code's tool heartbeats: elapsed time only.
DROPPED_EVENTS = frozenset({"tool_progress"})


class NoExtractorError(Exception):
    """An episode of a harness :data:`EXTRACTORS` has no extractor for."""


def digest(value: object) -> Record:
    """``bytes`` and ``sha256`` of a tool result: its UTF-8 text, or for structured content its compact JSON
    (``ensure_ascii`` off, separators without spaces, key order as recorded)."""
    data = value.encode() if isinstance(value, str) else encode(value).encode() if value is not None else b""
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def as_list(raw: object) -> list[object]:
    """A JSON array with the weakest true statement about its members; anything else reads as empty."""
    return cast("list[object]", raw) if isinstance(raw, list) else []


def encode(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parsed(line: str) -> dict[str, object] | None:
    """``line`` as a JSON object; None for anything else."""
    text = line.strip()
    if not text.startswith("{"):
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return as_block(value) if isinstance(value, dict) else None


def stamped(record: Record, ts: object) -> Record:
    """``record`` with ``ts`` when the harness wrote one."""
    return record if ts is None or ts == "" else {**record, "ts": ts}


def file_record(worker: pathlib.Path, path: pathlib.Path) -> Record:
    return {"kind": "file", "path": path.relative_to(worker).as_posix()}


# Claude Code: the stream-json logs of every attempt, then what only its sessions hold.


def claude_sources(worker: pathlib.Path) -> list[pathlib.Path]:
    """Every attempt's stream log in relaunch order (``claude.attemptN.log``, then ``claude.log``), then every
    session file under :data:`CLAUDE_SESSIONS` (its subagents' included)."""
    logs = token_cost.numbered_attempts(worker.glob("claude.attempt*.log"))
    if (worker / CLAUDE_LOG).is_file():
        logs.append(worker / CLAUDE_LOG)
    sessions = worker / CLAUDE_SESSIONS
    found = (
        sorted(p for p in sessions.rglob("*.jsonl") if p.is_file() and not p.is_symlink()) if sessions.is_dir() else []
    )
    return [*logs, *found]


def content_records(content: object, ts: object, parent: object) -> list[Record]:
    """The records of one assistant message's content blocks, in order."""
    if isinstance(content, str):
        return [stamped({"kind": "text", "text": content}, ts)] if content else []
    records: list[Record] = []
    for raw in as_list(content):
        block = as_block(raw)
        match block.get("type"):
            case "thinking":
                record: Record = {"kind": "reasoning", "text": block.get("thinking", "")}
            case "redacted_thinking":
                record = {"kind": "reasoning", "redacted": True}
            case "text":
                record = {"kind": "text", "text": block.get("text", "")}
            case "tool_use" | "server_tool_use" | "mcp_tool_use":
                record = {
                    "kind": "tool_call",
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": block.get("input"),
                }
            case _:
                record = {"kind": "event", "event": block}
        if parent:
            record["parent"] = parent
        records.append(stamped(record, ts))
    return records


def user_records(message: dict[str, object], ts: object, parent: object) -> list[Record]:
    """The records of one user message: a tool result as its :func:`digest`, a text as a prompt."""
    content = message.get("content")
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else as_list(content)
    records: list[Record] = []
    for raw in blocks:
        block = as_block(raw)
        match block.get("type"):
            case "tool_result":
                record: Record = {"kind": "tool_result", "id": block.get("tool_use_id"), **digest(block.get("content"))}
                if "is_error" in block:
                    record["error"] = block["is_error"]
            case "text":
                record = {"kind": "prompt", "role": "user", "text": block.get("text", "")}
            case other:
                record = {"kind": "prompt", "role": "user", "type": other, **digest(block)}
        if parent:
            record["parent"] = parent
        records.append(stamped(record, ts))
    return records


@dataclasses.dataclass(slots=True)
class Request:
    """One model request of a stream log: what :func:`claude_usage` reads, and the record it follows."""

    started: dict[str, object] = dataclasses.field(default_factory=dict[str, object])
    usage: dict[str, object] = dataclasses.field(default_factory=dict[str, object])
    deltas: list[int] = dataclasses.field(default_factory=list[int])
    after: int = 0
    ts: object = None
    #: Its ``message_stop`` arrived.
    completed: bool = False
    #: In flight at a failure the driver resumed from: :mod:`token_cost` reports it lost, never cost.
    lost: bool = False


def claude_usage(message_id: str, request: Request) -> Record:
    """One request's usage record, read as :func:`token_cost.events_cost` reads it: its input from the last
    ``assistant`` usage (else its ``message_start``), its output from its ``message_delta`` readings (else that
    usage). Claude Code's ``output`` includes the thinking, so there is no ``reasoning``."""
    usage = request.usage or request.started
    output = token_cost.delta_total(request.deltas)[0] if request.deltas else token_cost.count(usage, "output_tokens")
    record: Record = {
        "kind": "usage",
        "id": message_id,
        "input": token_cost.count(usage, "input_tokens") + token_cost.count(usage, "cache_creation_input_tokens"),
        "cached_input": token_cost.count(usage, "cache_read_input_tokens"),
        "output": output,
    }
    if request.lost:
        record["lost"] = True
    return stamped(record, request.ts)


@dataclasses.dataclass(slots=True)
class Requests:
    """The model requests of one stream log as they stream, by message id (the one in flight is ``current``)."""

    by_id: dict[str, Request] = dataclasses.field(default_factory=dict[str, Request])
    current: str = ""

    def assistant(self, message: dict[str, object], after: int, ts: object) -> None:
        """An ``assistant`` event of a request: its usage (the last one counts) and the record it ends at."""
        turn = message.get("id")
        if isinstance(turn, str) and turn and not token_cost.is_synthetic(message):
            request = self.by_id.setdefault(turn, Request())
            request.usage = as_block(message.get("usage")) or request.usage
            request.after, request.ts = after, ts

    def stream(self, event: dict[str, object], after: int) -> None:
        """A ``message_start``, ``message_delta`` or ``message_stop``, grouped under the last ``message_start``."""
        opened = token_cost.stream_message_id(event)
        if opened is not None:
            self.current = opened
            message = as_block(token_cost.stream_event_body(event).get("message"))
            self.by_id.setdefault(opened, Request()).started = as_block(message.get("usage"))
        if not self.current:
            return
        request = self.by_id.setdefault(self.current, Request())
        if (delta := token_cost.delta_usage(event)) is not None:
            request.deltas.append(delta[1])
        request.completed = request.completed or token_cost.request_completed(event)
        request.after = after

    def resume(self) -> None:
        """The driver's resume line: the request in flight, unless it completed, is lost."""
        if self.current and not self.by_id[self.current].completed:
            self.by_id[self.current].lost = True
        self.current = ""

    def placed(self, records: list[Record]) -> list[Record]:
        """``records`` with each request's usage record after the last record of the request."""
        for message_id, request in sorted(self.by_id.items(), key=lambda item: -item[1].after):
            records.insert(request.after, claude_usage(message_id, request))
        return records


def stream_event_records(event: dict[str, object], records: list[Record], requests: Requests) -> None:
    """Append the records of one stream-json event to ``records``: an ``assistant`` event without its content, then
    the content; a user message's records; a request's stream events and every other event whole, but the content-block
    deltas and the tool heartbeats."""
    kind = event.get("type")
    body = token_cost.stream_event_body(event).get("type")
    if kind == "assistant":
        message = as_block(event.get("message"))
        stripped = {key: value for key, value in message.items() if key != "content"}
        records.append({"kind": "event", "event": {**event, "message": stripped}})
        ts = event.get("timestamp")
        records += content_records(message.get("content"), ts, event.get("parent_tool_use_id"))
        requests.assistant(message, len(records), ts)
    elif kind == "user":
        message = as_block(event.get("message"))
        records += user_records(message, event.get("timestamp"), event.get("parent_tool_use_id"))
    elif body in REQUEST_EVENTS:
        records.append({"kind": "event", "event": event})
        requests.stream(event, len(records))
    elif event.get("subtype") == THINKING_SUBTYPE and records and thinking(records[-1]):
        last = as_block(records[-1]["event"])
        delta = token_cost.count(last, "estimated_tokens_delta") + token_cost.count(event, "estimated_tokens_delta")
        records[-1] = {"kind": "event", "event": {**event, "estimated_tokens_delta": delta}}
    elif not (kind == "stream_event" or kind in DROPPED_EVENTS or str(body).startswith("content_block")):
        if token_cost.is_resume(event):
            requests.resume()
            records.append({"kind": "attempt"})
        records.append({"kind": "event", "event": event})


def thinking(record: Record) -> bool:
    """Whether ``record`` holds a ``thinking_tokens`` event."""
    return record.get("kind") == "event" and as_block(record.get("event")).get("subtype") == THINKING_SUBTYPE


def stream_records(log: pathlib.Path, streamed: set[str]) -> list[Record]:
    """The records of one stream-json log; adds every event's ``uuid`` to ``streamed``. Each request's usage
    record follows its last event."""
    records: list[Record] = []
    requests = Requests()
    with log.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            event = parsed(line)
            if event is None:
                if line.strip():
                    records.append({"kind": "log", "text": line})
                continue
            if isinstance(uuid := event.get("uuid"), str):
                streamed.add(uuid)
            stream_event_records(event, records, requests)
    return requests.placed(records)


def session_records(session: pathlib.Path, streamed: set[str]) -> list[Record]:
    """The records of the turns of a Claude Code session file no stream log holds (a record whose ``uuid`` a log
    carried is the same turn): its content and usage, and records of other kinds whole, but not
    :data:`SESSION_METADATA`."""
    records: list[Record] = []
    with session.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            entry = parsed(raw)
            if entry is None:
                if raw.strip():
                    records.append({"kind": "log", "text": raw.rstrip("\n")})
                continue
            if entry.get("uuid") in streamed:
                continue
            kind, ts = entry.get("type"), entry.get("timestamp")
            message = as_block(entry.get("message"))
            if kind == "assistant":
                records += content_records(message.get("content"), ts, None)
                turn = message.get("id")
                if isinstance(turn, str) and turn and not token_cost.is_synthetic(message):
                    records.append(claude_usage(turn, Request(usage=as_block(message.get("usage")), ts=ts)))
            elif kind == "user":
                records += user_records(message, ts, None)
            elif kind not in SESSION_METADATA:
                records.append({"kind": "event", "event": entry})
    return records


def claude_records(worker: pathlib.Path, files: Sequence[pathlib.Path]) -> list[Record]:
    """Claude Code (and every workflow it runs, AutoKernel's included): each stream log, an attempt starting at its
    top and at each of the driver's resume lines, then the turns only the sessions hold. A turn is in the stream log
    and in the session file under one ``uuid``; it is extracted once, from the log."""
    streamed: set[str] = set()
    records: list[Record] = []
    sessions = [path for path in files if path.parent != worker]
    for log in (path for path in files if path.parent == worker):
        records += [file_record(worker, log), {"kind": "attempt"}, *stream_records(log, streamed)]
    for session in sessions:
        if found := session_records(session, streamed):
            records += [file_record(worker, session), *found]
    return records


# mini-SWE-agent: the trajectory (or the checkpoint, when the run died before the trajectory caught up).


def miniswe_sources(worker: pathlib.Path) -> list[pathlib.Path]:
    """The trajectory and the runner's checkpoint, each the whole message history."""
    return [path for path in (worker / run_miniswe.TRAJECTORY, worker / runner_common.CHECKPOINT) if path.is_file()]


def history(path: pathlib.Path, key: str) -> list[object]:
    """The message list under ``key`` of the JSON file ``path``; empty when unreadable."""
    try:
        value = as_block(json.loads(path.read_text(encoding="utf-8", errors="replace"))).get(key)
    except (OSError, ValueError):
        return []
    return as_list(value)


def openai_reply(message: dict[str, object], ts: object) -> list[Record]:
    """An OpenAI-format assistant message: its reasoning, text and tool calls (the arguments' JSON text whole)."""
    records: list[Record] = []
    reasoning = message.get("reasoning_content") or as_block(message.get("provider_specific_fields")).get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        records.append(stamped({"kind": "reasoning", "text": reasoning}, ts))
    content = message.get("content")
    if content:
        records.append(stamped({"kind": "text", "text": content}, ts))
    for raw in as_list(message.get("tool_calls")):
        call = as_block(raw)
        function = as_block(call.get("function"))
        record = {
            "kind": "tool_call",
            "id": call.get("id"),
            "name": function.get("name"),
            "input": function.get("arguments"),
        }
        records.append(stamped(record, ts))
    return records


def openai_usage(response: dict[str, object], ts: object) -> list[Record]:
    """The usage record of one chat-completions response, the line ``run_miniswe`` writes to ``usage.jsonl``."""
    if "usage" not in response:
        return []
    line = runner_common.openai_usage(as_block(response.get("usage")))
    return [stamped({"kind": "usage", "id": response.get("id"), **line}, ts)]


def miniswe_records(worker: pathlib.Path, files: Sequence[pathlib.Path]) -> list[Record]:
    """mini-SWE-agent (``mini-swe-agent-1.1`` trajectories): the longer of the trajectory's ``messages`` and the
    checkpoint's ``history`` (the same messages; the checkpoint is written before each call, the trajectory after a
    step). A format error is a model reply the agent refused: its reply (``extra.response``) and then the harness's
    message. An attempt starts at the top and at each driver prompt after the task (a user message with no
    ``extra``: the resume prompt)."""
    found = [(history(path, "messages" if path.name == run_miniswe.TRAJECTORY else "history"), path) for path in files]
    if not found:
        return unattributed(worker, [])
    messages, source = max(found, key=lambda pair: len(pair[0]))
    records: list[Record] = [file_record(worker, source), {"kind": "attempt"}]
    prompts = 0
    for raw in messages:
        message = as_block(raw)
        role, extra = message.get("role"), as_block(message.get("extra"))
        response = as_block(extra.get("response"))
        if role == "assistant":
            ts = extra.get("timestamp")
            records += [*openai_reply(message, ts), *openai_usage(response, ts)]
        elif role == "tool":
            result = {"kind": "tool_result", "id": message.get("tool_call_id"), **digest(message.get("content"))}
            if "returncode" in extra:
                result["returncode"] = extra["returncode"]
            records.append(stamped(result, extra.get("timestamp")))
        elif role in ("system", "user"):
            if response:
                choices = response.get("choices")
                reply = next((as_block(as_block(choice).get("message")) for choice in as_list(choices)), {})
                records += [
                    *openai_reply(reply, response.get("created")),
                    *openai_usage(response, response.get("created")),
                ]
            elif role == "user":
                prompts += 1
                if prompts > 1:
                    records.append({"kind": "attempt"})
            prompt: Record = {"kind": "prompt", "role": role, "text": message.get("content")}
            if extra.get("interrupt_type"):
                prompt["interrupt"] = extra["interrupt_type"]
            records.append(prompt)
        else:
            records.append({"kind": "event", "event": message})
    return records + unattributed(worker, records)


# OpenHands: the event log, the conversation's persisted events, the usage of each response.


def openhands_sources(worker: pathlib.Path) -> list[pathlib.Path]:
    """The event log, the checkpoint and every file of the conversation's persisted state (its events, its
    ``base_state.json`` with each response's usage)."""
    state = worker / run_openhands.STATE_DIR
    persisted = sorted(p for p in state.rglob("*") if p.is_file() and not p.is_symlink()) if state.is_dir() else []
    named = [path for path in (worker / run_openhands.EVENTS, worker / runner_common.CHECKPOINT) if path.is_file()]
    return [*named, *persisted]


def openhands_events(files: Sequence[pathlib.Path]) -> list[dict[str, object]]:
    """Every event once, in the conversation's order: the persisted ``events/event-NNNNN-*.json`` order, each event
    as the event log wrote it (all fields; the persisted copy leaves out the empty ones), then any logged event not
    persisted."""
    logged: dict[str, dict[str, object]] = {}
    unnamed: list[dict[str, object]] = []
    for log in (path for path in files if path.name == run_openhands.EVENTS):
        with log.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if (event := parsed(line)) is not None:
                    key = event.get("id")
                    if isinstance(key, str):
                        logged.setdefault(key, event)
                    else:
                        unnamed.append(event)
    ordered: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in sorted((p for p in files if p.parent.name == "events" and p.suffix == ".json"), key=lambda p: p.name):
        with contextlib.suppress(OSError, ValueError):
            event = as_block(json.loads(path.read_text(encoding="utf-8", errors="replace")))
            key = str(event.get("id"))
            if key not in seen:
                seen.add(key)
                ordered.append(logged.get(key, event))
    ordered += [event for key, event in logged.items() if key not in seen]
    return ordered + unnamed


def openhands_usages(files: Sequence[pathlib.Path]) -> dict[str, list[Record]]:
    """Each response's usage records by response id, from every ``base_state.json``'s per-LLM ``token_usages`` (the
    agent's and the condenser's), as ``run_openhands.usage_recorder`` writes them to ``usage.jsonl``."""
    found: dict[str, list[Record]] = {}
    for path in (p for p in files if p.name == "base_state.json"):
        with contextlib.suppress(OSError, ValueError):
            state = as_block(json.loads(path.read_text(encoding="utf-8", errors="replace")))
            for llm, metrics in as_block(as_block(state.get("stats")).get("usage_to_metrics")).items():
                last = ""
                for raw in as_list(as_block(metrics).get("token_usages")):
                    usage = as_block(raw)
                    response = str(usage.get("response_id") or "")
                    if response == last:
                        continue  # the recorder writes a response once
                    last = response
                    counts = runner_common.usage_line(
                        token_cost.count(usage, "prompt_tokens"),
                        token_cost.count(usage, "cache_read_tokens"),
                        token_cost.count(usage, "completion_tokens"),
                        token_cost.count(usage, "reasoning_tokens"),
                    )
                    found.setdefault(response, []).append({"kind": "usage", "id": response, "llm": llm, **counts})
    return found


def texts(content: object) -> str:
    """The text of an OpenHands content list (``[{"type": "text", "text": ...}]``), joined."""
    parts = (as_block(block).get("text") for block in as_list(content))
    return "".join(part for part in parts if isinstance(part, str))


def openhands_reasoning(event: dict[str, object], ts: object) -> list[Record]:
    """An agent event's reasoning: ``reasoning_content``, then each thinking block."""
    records: list[Record] = []
    if isinstance(reasoning := event.get("reasoning_content"), str) and reasoning:
        records.append(stamped({"kind": "reasoning", "text": reasoning}, ts))
    for raw in as_list(event.get("thinking_blocks")):
        block = as_block(raw)
        if block.get("type") == "redacted_thinking":
            records.append(stamped({"kind": "reasoning", "redacted": True}, ts))
        else:
            records.append(stamped({"kind": "reasoning", "text": block.get("thinking", "")}, ts))
    return records


def openhands_event(event: dict[str, object], prompts: int) -> list[Record]:
    """The records of one OpenHands event; ``prompts`` counts the user messages before it."""
    ts = event.get("timestamp")
    match event.get("kind"):
        case "SystemPromptEvent":
            system = as_block(event.get("system_prompt"))
            tools = [as_block(tool).get("title") for tool in as_list(event.get("tools"))]
            return [stamped({"kind": "prompt", "role": "system", "text": system.get("text"), "tools": tools}, ts)]
        case "MessageEvent":
            message = as_block(event.get("llm_message"))
            if event.get("source") == "user":
                prompt = stamped({"kind": "prompt", "role": "user", "text": texts(message.get("content"))}, ts)
                return [{"kind": "attempt"}, prompt] if prompts else [prompt]
            text = texts(message.get("content"))
            return [*openhands_reasoning(message, ts), *([stamped({"kind": "text", "text": text}, ts)] if text else [])]
        case "ActionEvent":
            call = as_block(event.get("tool_call"))
            thought = texts(event.get("thought"))
            record = {
                "kind": "tool_call",
                "id": event.get("tool_call_id"),
                "name": event.get("tool_name"),
                "input": call.get("arguments"),
            }
            if event.get("summary"):
                record["summary"] = event["summary"]
            return [
                *openhands_reasoning(event, ts),
                *([stamped({"kind": "text", "text": thought}, ts)] if thought else []),
                stamped(record, ts),
            ]
        case "ObservationEvent":
            observation = as_block(event.get("observation"))
            result = {"kind": "tool_result", "id": event.get("tool_call_id"), **digest(observation.get("content"))}
            if "is_error" in observation:
                result["error"] = observation["is_error"]
            return [stamped(result, ts)]
        case "AgentErrorEvent":
            return [
                stamped(
                    {
                        "kind": "tool_result",
                        "id": event.get("tool_call_id"),
                        **digest(event.get("error")),
                        "error": True,
                    },
                    ts,
                )
            ]
        case "Condensation":
            summary = {
                "kind": "text",
                "text": event.get("summary"),
                "condensation": len(as_list(event.get("forgotten_event_ids"))),
            }
            return [stamped(summary, ts)]
        case _:
            return [{"kind": "event", "event": event}]


def openhands_records(worker: pathlib.Path, files: Sequence[pathlib.Path]) -> list[Record]:
    """OpenHands (``Event.model_dump_json`` per line): each event's records, each response's usage after the first
    event naming it (``llm_response_id``; the condenser's after its ``Condensation``), and usage no event names at the
    end. An attempt starts at each user message after the first (the driver's resume prompt)."""
    usages = openhands_usages(files)
    records: list[Record] = []
    if any(path.name == run_openhands.EVENTS for path in files):
        records.append({"kind": "file", "path": run_openhands.EVENTS})
    records.append({"kind": "attempt"})
    prompts = 0
    for event in openhands_events(files):
        records += openhands_event(event, prompts)
        if event.get("kind") == "MessageEvent" and event.get("source") == "user":
            prompts += 1
        if response := event.get("llm_response_id"):
            records += usages.pop(str(response), [])
    records += [record for found in usages.values() for record in found]
    return records + unattributed(worker, records)


def unattributed(worker: pathlib.Path, records: list[Record]) -> list[Record]:
    """The calls of the runner's ``usage.jsonl`` no record of ``records`` accounts for (a run killed after the call
    was counted but before its transcript was saved), in file order, as usage records without an id: the extract's
    usage then sums to the file's. The driver's resume lines are none."""
    path = worker / harnesses.USAGE_FILE
    with contextlib.suppress(OSError), path.open(encoding="utf-8", errors="replace") as handle:
        lines = [line for line in token_cost.usage_records(list(handle)) if token_cost.USAGE_RESUME_KEY not in line]
        held = collections.Counter(usage_key(record) for record in records if record.get("kind") == "usage")
        missing: list[Record] = []
        for line in lines:
            key = usage_key(line)
            if held[key]:
                held[key] -= 1
            else:
                missing.append(
                    {"kind": "usage", "id": None, **{field: line.get(field, 0) for field in harnesses.CONSUMED_FIELDS}}
                )
        return missing
    return []


def usage_key(record: dict[str, object]) -> tuple[int, ...]:
    return tuple(token_cost.count(record, field) for field in harnesses.CONSUMED_FIELDS)


@dataclasses.dataclass(frozen=True, slots=True)
class Extractor:
    """How one harness's episodes are extracted: the files of a worker directory it reads (which a default copy
    leaves out) and its records."""

    sources: Callable[[pathlib.Path], list[pathlib.Path]]
    records: Callable[[pathlib.Path, Sequence[pathlib.Path]], list[Record]]


def no_sources(_worker: pathlib.Path) -> list[pathlib.Path]:
    return []


def no_records(_worker: pathlib.Path, _files: Sequence[pathlib.Path]) -> list[Record]:
    return []


#: Every harness's extractor by its name (:func:`harnesses.names`). A tuner calls no model and keeps no transcript:
#: its extract holds its attempts only (its ``tuner.log`` is collected as it is). ``""`` is a worker directory no
#: harness wrote its log in. The OpenAI, Anthropic and Optimas runners have none: no episode of theirs was ever
#: recorded to derive their format from, so collecting one without ``--full-transcripts`` is refused.
EXTRACTORS: dict[str, Extractor] = {
    harnesses.CLAUDE: Extractor(claude_sources, claude_records),
    "miniswe": Extractor(miniswe_sources, miniswe_records),
    "openhands": Extractor(openhands_sources, openhands_records),
    harnesses.TUNER: Extractor(no_sources, no_records),
    "": Extractor(no_sources, no_records),
}


def logs() -> dict[str, str]:
    """Every harness's log file name (the first of its records), by harness."""
    return {harnesses.CLAUDE: CLAUDE_LOG, **{name: spec.log_name for name, spec in harnesses.runners().items()}}


def harness_of(worker: pathlib.Path) -> str:
    """The harness of the episode in ``worker``: its ``tokens.json`` identity, else the one whose log (or a crashed
    attempt's) is there, ``+``-joined when several are; ``""`` when none is."""
    with contextlib.suppress(OSError, ValueError):
        identity = as_block(as_block(json.loads((worker / TOKENS_NAME).read_text(encoding="utf-8"))).get("identity"))
        if isinstance(named := identity.get("harness"), str) and named:
            return named
    found = []
    for name, log in logs().items():
        stem, suffix = log.rsplit(".", 1)
        if (worker / log).is_file() or any(worker.glob(f"{stem}.attempt*.{suffix}")):
            found.append(name)
    return "+".join(found)


def extractor(harness: str) -> Extractor:
    if harness not in EXTRACTORS:
        raise NoExtractorError(f"harness {harness!r} has no transcript extractor")
    return EXTRACTORS[harness]


def sources(worker: pathlib.Path, harness: str) -> list[pathlib.Path]:
    """The transcript files of ``harness`` in ``worker`` its extract is read from. Raises NoExtractorError."""
    return extractor(harness).sources(worker)


def marked(records: list[Record], ledger: list[dict[str, object]]) -> list[Record]:
    """``records`` with each ``attempt`` record filled with its ledger line: the last ones to the markers in order
    (an attempt whose transcript the driver removed comes first), the earlier ones at the top with ``transcript``
    false."""
    marks = [index for index, record in enumerate(records) if record.get("kind") == "attempt"]
    kept = ledger[max(0, len(ledger) - len(marks)) :]
    for index, line in zip(marks[len(marks) - len(kept) :], kept, strict=True):
        records[index] = {"kind": "attempt", **line}
    early = ledger[: len(ledger) - len(kept)]
    return [{"kind": "attempt", **line, "transcript": False} for line in early] + records


def extract(worker: pathlib.Path, harness: str, files: Sequence[pathlib.Path]) -> bytes:
    """The extract of the episode in ``worker`` from its transcript ``files`` (:func:`sources`, less what the caller
    may not read), uncompressed. Raises NoExtractorError."""
    records = marked(extractor(harness).records(worker, files), token_cost.attempt_ledger(worker))
    header: Record = {
        "kind": "extract",
        "schema": SCHEMA,
        "harness": harness,
        "sources": [
            {"path": path.relative_to(worker).as_posix(), "bytes": path.stat().st_size, "sha256": file_sha(path)}
            for path in files
        ],
    }
    return "".join(encode(record) + "\n" for record in (header, *records)).encode()


def file_sha(path: pathlib.Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def zstd() -> str:
    found = shutil.which("zstd")
    if found is None:
        raise RuntimeError("zstd is not on PATH; a transcript extract is written and read with it")
    return found


def compress(data: bytes) -> bytes:
    return subprocess.run([zstd(), "-q", f"-{ZSTD_LEVEL}", "-c"], input=data, capture_output=True, check=True).stdout


def decompress(data: bytes) -> bytes:
    return subprocess.run([zstd(), "-q", "-d", "-c"], input=data, capture_output=True, check=True).stdout


def read(path: pathlib.Path) -> list[Record]:
    """The records of the extract at ``path``, header first. Raises ValueError for anything else."""
    try:
        text = decompress(path.read_bytes()).decode()
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: not a transcript extract: {exc}") from exc
    records = [as_block(json.loads(line)) for line in text.splitlines() if line]
    if not records or records[0].get("kind") != "extract" or records[0].get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a transcript extract of schema {SCHEMA}")
    return records


def stream_logs(records: Sequence[Record]) -> Iterator[tuple[str, list[dict[str, object]]]]:
    """``(log name, its event records' events)`` of each Claude Code stream log of an extract."""
    current: list[dict[str, object]] | None = None
    name = ""
    for record in records:
        if record.get("kind") == "file":
            if current is not None:
                yield name, current
            name = str(record.get("path"))
            current = [] if "/" not in name else None
        elif record.get("kind") == "event" and current is not None:
            current.append(as_block(record.get("event")))
    if current is not None:
        yield name, current


def restore(records: Sequence[Record], directory: pathlib.Path) -> list[pathlib.Path]:
    """Write each Claude Code stream log of an extract into ``directory`` under its name, as the events it kept:
    :func:`token_cost.episode_totals` of ``directory`` (with the episode's ``attempts.jsonl``) is then the episode's
    own."""
    written = []
    for name, events in stream_logs(records):
        path = directory / name
        path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        written.append(path)
    return written
