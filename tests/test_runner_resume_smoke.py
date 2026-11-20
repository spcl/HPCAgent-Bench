# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Each LLM runner, as a real process against a stub model server: a call whose connection dies ends the leg as a
resumable ``api_timeout``, ``--resume-prompt`` re-sends exactly the cut-off request plus the resume message, and
``usage.jsonl`` holds each completed call once, split into its four disjoint counts.

The stub answers a leg's requests in order; :data:`DROP` is a request whose response dies mid-body, and it keeps
dropping every retry the client's SDK makes. A runner whose SDK is not installed in this interpreter is skipped:
each runs in its own venv (``agent/pyproject.toml``'s ``harness-<name>`` groups), where CI's integration[0] runs
this file once per runner.
"""

import http.server
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import threading
from collections.abc import Iterator
from typing import Any, ClassVar, NamedTuple

import pytest

RESUME = "The connection dropped; continue."
DROP = "drop"
#: A script entry the stub answers with a 401, the provider's refusal of a credential.
REFUSE = "refuse"
#: The only credential any test here hands a runner.
KEY = "test-credential-0123456789"
MCP_STUB = """
import anyio
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server


async def list_tools(context, params):
    return types.ListToolsResult(tools=[])


async def main():
    server = Server("stub", on_list_tools=list_tools)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


anyio.run(main)
"""
COMMAND = "echo hi > note.txt"
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def sse(kind: str, data: dict[str, Any]) -> bytes:
    return f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode()


def message_start(prompt: int, cached: int) -> bytes:
    usage = {
        "input_tokens": prompt,
        "output_tokens": 1,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
    }
    message = {"id": "msg", "type": "message", "role": "assistant", "model": "stub", "content": [], "stop_reason": None}
    return sse("message_start", {"type": "message_start", "message": {**message, "usage": usage}})


def messages_reply(
    content: dict[str, Any], delta: dict[str, Any], stop: str, usage: tuple[int, int, int]
) -> list[bytes]:
    """One streamed Messages API reply of one content block; ``usage`` is (uncached prompt, cache read, output)."""
    end = {"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": usage[2]}}
    return [
        message_start(usage[0], usage[1]),
        sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": content}),
        sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
        sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        sse("message_delta", end),
        sse("message_stop", {"type": "message_stop"}),
    ]


def anthropic_tool_call(command: str, usage: tuple[int, int, int]) -> list[bytes]:
    call = {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {}}
    arguments = {"type": "input_json_delta", "partial_json": json.dumps({"command": command})}
    return messages_reply(call, arguments, "tool_use", usage)


def anthropic_text(text: str, usage: tuple[int, int, int]) -> list[bytes]:
    return messages_reply({"type": "text", "text": ""}, {"type": "text_delta", "text": text}, "end_turn", usage)


def completion(message: dict[str, Any], finish: str, usage: tuple[int, int, int, int]) -> bytes:
    """One chat completion; ``usage`` is (prompt incl. cached, cached, completion incl. reasoning, reasoning)."""
    counts = {
        "prompt_tokens": usage[0],
        "completion_tokens": usage[2],
        "total_tokens": usage[0] + usage[2],
        "prompt_tokens_details": {"cached_tokens": usage[1]},
        "completion_tokens_details": {"reasoning_tokens": usage[3]},
    }
    choice = {"index": 0, "finish_reason": finish, "message": {"role": "assistant", **message}}
    reply = {"id": "chatcmpl", "object": "chat.completion", "created": 0, "model": "stub", "choices": [choice]}
    return json.dumps({**reply, "usage": counts}).encode()


def chat_tool_call(command: str, usage: tuple[int, int, int, int]) -> bytes:
    function = {"name": "bash", "arguments": json.dumps({"command": command})}
    call = {"id": "call_1", "type": "function", "function": function}
    return completion({"content": "running it", "tool_calls": [call]}, "tool_calls", usage)


def chat_text(text: str, usage: tuple[int, int, int, int]) -> bytes:
    return completion({"content": text}, "stop", usage)


class StubModel(http.server.BaseHTTPRequestHandler):
    """Answers each request with the next entry of ``script``; :data:`DROP` stays, so retries drop too."""

    protocol_version = "HTTP/1.1"
    script: ClassVar[list[Any]] = []
    requests: ClassVar[list[dict[str, Any]]] = []
    #: Each request's credential: the Messages API's header, or the bearer token of chat completions.
    credentials: ClassVar[list[str]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)
        type(self).credentials.append(
            self.headers.get("x-api-key") or self.headers.get("Authorization", "").removeprefix("Bearer ")
        )
        reply = self.script[0]
        if reply != DROP:
            self.script.pop(0)
        if reply == REFUSE:
            payload = json.dumps({"type": "error", "error": {"type": "authentication_error", "message": "no"}}).encode()
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        if self.path.endswith("/messages"):
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for event in [message_start(1, 0)] if reply == DROP else reply:
                self.wfile.write(f"{len(event):x}\r\n".encode() + event + b"\r\n")
            if reply != DROP:
                self.wfile.write(b"0\r\n\r\n")
        else:
            payload = b'{"id": "chatcmpl", "choices": [' if reply == DROP else reply
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload) + (4096 if reply == DROP else 0)))
            self.end_headers()
            self.wfile.write(payload)
        self.wfile.flush()
        if reply == DROP:
            self.close_connection = True

    def log_message(self, *args: object) -> None:
        pass


class Runner(NamedTuple):
    module: str
    package: str
    #: The stub's base URL suffix the runner is handed (the Messages SDK appends ``/v1`` itself).
    suffix: str
    mcp: bool
    first: list[Any]
    second: list[Any]
    #: Each completed call's usage line, (input, cached_input, output, reasoning).
    usage: list[tuple[int, int, int, int]]


RUNNERS = {
    "anthropic": Runner(
        "run_anthropic",
        "anthropic",
        "",
        True,
        [anthropic_tool_call(COMMAND, (100, 40, 10)), DROP],
        [anthropic_text("done", (300, 50, 5))],
        [(100, 40, 10, 0), (300, 50, 5, 0)],
    ),
    "openai": Runner(
        "run_openai",
        "agents",
        "/v1",
        True,
        [chat_tool_call(COMMAND, (100, 40, 30, 20)), DROP],
        [chat_text("done", (300, 50, 9, 4))],
        [(60, 40, 10, 20), (250, 50, 5, 4)],
    ),
    "miniswe": Runner(
        "run_miniswe",
        "minisweagent",
        "/v1",
        False,
        [chat_tool_call(COMMAND, (100, 40, 30, 20)), DROP],
        [chat_tool_call(SUBMIT, (300, 50, 9, 4))],
        [(60, 40, 10, 20), (250, 50, 5, 4)],
    ),
}


@pytest.fixture(name="stub")
def stub_fixture() -> Iterator[str]:
    StubModel.requests = []
    StubModel.credentials = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StubModel)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def run_leg(
    runner: Runner, workdir: pathlib.Path, base_url: str, script: list[Any], *extra: str, **environ: str
) -> int:
    StubModel.script = list(script)
    # mini-swe-agent retries a failed call 10 times with backoff (minutes); two attempts show the same ending.
    env = {
        **os.environ,
        "ANTHROPIC_API_KEY": KEY,
        "OPENAI_API_KEY": KEY,
        "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": "2",
        **environ,
    }
    mcp = ("--mcp-config", str(workdir / "mcp.json")) if runner.mcp else ()
    argv = [
        sys.executable,
        "-m",
        f"hpcagent_agent.harness.{runner.module}",
        *("--workdir", str(workdir), "--prompt", str(workdir / "prompt.txt"), "--base-url", base_url + runner.suffix),
        *("--model", "stub", "--usage", str(workdir / "usage.jsonl"), "--request-timeout", "30", *mcp, *extra),
    ]
    done = subprocess.run(argv, env=env, cwd=workdir, capture_output=True, text=True, timeout=300, check=False)
    print(done.stdout[-3000:], done.stderr[-3000:])
    return done.returncode


def prepared(name: str, workdir: pathlib.Path) -> Runner:
    """The runner ``name`` with its prompt and stub MCP server in ``workdir``; skipped where its SDK is missing."""
    runner = RUNNERS[name]
    # CI names the runner its venv was built for, where a missing SDK is a broken venv, not a skip.
    if importlib.util.find_spec(runner.package) is None and os.environ.get("HPCAGENT_BENCH_RUNNER_SMOKE") != name:
        pytest.skip(f"{runner.package} is installed only in the {name} runner's venv")
    (workdir / "prompt.txt").write_text("the task", encoding="utf-8")
    (workdir / "mcp_stub.py").write_text(MCP_STUB, encoding="utf-8")
    servers = {"stub": {"command": sys.executable, "args": [str(workdir / "mcp_stub.py")]}}
    (workdir / "mcp.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return runner


@pytest.mark.parametrize("name", list(RUNNERS))
def test_a_dropped_call_resumes_with_the_cut_off_request_and_counts_each_call_once(
    name: str, stub: str, tmp_path: pathlib.Path
) -> None:
    runner = prepared(name, tmp_path)

    assert run_leg(runner, tmp_path, stub, runner.first) == 1
    end = json.loads((tmp_path / "harness-end.json").read_text(encoding="utf-8"))
    assert (end["reason"], end["turns"]) == ("api_timeout", 1), end
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "hi\n"
    cut = StubModel.requests[-1]
    StubModel.requests = []

    assert run_leg(runner, tmp_path, stub, runner.second, "--resume-prompt", RESUME) == 0
    (resumed,) = StubModel.requests
    assert resumed["messages"][: len(cut["messages"])] == cut["messages"]
    assert [message["content"] for message in resumed["messages"][len(cut["messages"]) :]] == [RESUME]
    usage = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text(encoding="utf-8").splitlines()]
    fields = ("input", "cached_input", "output", "reasoning")
    assert [tuple(line[field] for field in fields) for line in usage] == runner.usage


@pytest.mark.parametrize("name", ["anthropic", "openai"])
def test_a_refused_credential_ends_the_leg_and_no_record_or_output_holds_the_key(
    name: str, stub: str, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = prepared(name, tmp_path)
    assert run_leg(runner, tmp_path, stub, [REFUSE]) == 1
    (request,) = StubModel.requests
    assert request["model"] == "stub"
    assert {tool.get("name") or tool["function"]["name"] for tool in request["tools"]} == {"bash"}
    assert StubModel.credentials == [KEY]
    end = json.loads((tmp_path / "harness-end.json").read_text(encoding="utf-8"))
    assert end["reason"] == "error", end
    leaks = [path.name for path in tmp_path.iterdir() if path.is_file() and KEY in path.read_text(encoding="utf-8")]
    assert leaks == []
    assert KEY not in "".join(capsys.readouterr())


def test_an_unusable_task_call_is_answered_to_the_openai_model(stub: str, tmp_path: pathlib.Path) -> None:
    runner = prepared("openai", tmp_path)
    agents = tmp_path / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / "helper.md").write_text("---\nname: helper\ndescription: helps\n---\nHelp.\n", encoding="utf-8")
    function = {"name": "task", "arguments": '{"subagent_type": '}
    call = {"id": "call_1", "type": "function", "function": function}
    script = [completion({"tool_calls": [call]}, "tool_calls", (1, 0, 1, 0)), chat_text("done", (1, 0, 1, 0))]
    environ = {"HOME": str(tmp_path), "HPCAGENT_BENCH_PACKET_AGENTS": "helper"}
    assert run_leg(runner, tmp_path, stub, script, **environ) == 0
    answer = StubModel.requests[-1]["messages"][-1]
    assert (answer["role"], answer["tool_call_id"]) == ("tool", "call_1"), answer
    assert "unusable task call" in answer["content"]
