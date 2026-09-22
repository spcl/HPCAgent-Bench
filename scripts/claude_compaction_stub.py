# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end proof that the claude CLI compacts before any request overflows the served window.

Runs the real ``claude`` binary with the argv and environment ``experiments/agent_driver.py`` builds
(claude_command, claude_env) against a stub Anthropic Messages server on localhost. Every main turn
the stub calls Read on fresh files, so the prompt grows by a whole big turn at a time -- the case
compaction has to survive -- and it reports each request's size as its usage, like vLLM/SGLang do.
A request whose size plus its max_tokens passes the window gets vLLM's 400, and the run FAILS when
any request overflowed, when nothing compacted, or when the CLI did not finish its turns.

    python scripts/claude_compaction_stub.py --claude "$(command -v claude)"                # PASS
    python scripts/claude_compaction_stub.py --claude "$(command -v claude)" --without-fix  # FAIL

Needs no GPU, network or judge: the binary and python3 only. Sizes are body characters / 3, the
estimate claude-code 2.1.197 itself applies to a model it does not know.
"""

import argparse
import dataclasses
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import types
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = pathlib.Path(__file__).resolve().parents[1]
MODEL = "hpcagent-bench-vllm"
MAX_OUTPUT_TOKENS = 32768
CHARS_PER_TOKEN = 3
SUMMARY_MARK = "Your task is to create a detailed summary"
#: (served window, Read calls per turn, characters per file): one 262144 arm at two reads of
#: ~16k tokens a turn -- above the largest per-turn growth measured on real transcripts, 32.6k --
#: and oss120b's 131072 at one, the most its smaller window holds without thrashing.
CASES = ((262144, 2, 48000), (131072, 1, 48000))


@dataclasses.dataclass
class Episode:
    """What the stub saw of one CLI run; the handler thread writes it, the main thread reads it."""

    window: int
    reads: int
    files: list[pathlib.Path]
    turns: int
    requests: list[dict[str, object]] = dataclasses.field(default_factory=list)
    main_turns: int = 0
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


def load_driver() -> types.ModuleType:
    path = REPO / "experiments" / "agent_driver.py"
    spec = importlib.util.spec_from_file_location("agent_driver", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def last_user_text(body: dict) -> str:
    for message in reversed(body.get("messages") or []):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            return " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return ""


def sse_events(tokens: int, blocks: list[dict[str, object]], stop: str) -> list[dict[str, object]]:
    """One streamed reply, its usage shaped the way vLLM/SGLang report it (no cache_creation field)."""
    usage = {"input_tokens": 1000, "output_tokens": 0, "cache_read_input_tokens": tokens - 1000}
    start = {"id": f"msg_{uuid.uuid4().hex[:16]}", "type": "message", "role": "assistant", "model": MODEL}
    events: list[dict[str, object]] = [{"type": "message_start", "message": {**start, "content": [], "usage": usage}}]
    for index, block in enumerate(blocks):
        if block["type"] == "tool_use":
            head = {**block, "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        else:
            head = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        events += [
            {"type": "content_block_start", "index": index, "content_block": head},
            {"type": "content_block_delta", "index": index, "delta": delta},
            {"type": "content_block_stop", "index": index},
        ]
    events.append({"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 200}})
    events.append({"type": "message_stop"})
    return events


def reply(episode: Episode, kind: str) -> tuple[list[dict[str, object]], str]:
    """The next reply's content: a summary, the closing text, or this turn's Read calls."""
    if kind != "main":
        return [{"type": "text", "text": "<summary>stub summary of the work so far</summary>"}], "end_turn"
    if episode.main_turns > episode.turns:
        return [{"type": "text", "text": "DONE"}], "end_turn"
    first = (episode.main_turns - 1) * episode.reads
    reads = [
        {"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:16]}", "name": "Read", "input": {"file_path": str(path)}}
        for path in episode.files[first : first + episode.reads]
    ]
    return reads, "tool_use"


def handler_for(episode: Episode) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers.get("content-length", "0")))
            if "count_tokens" in self.path:  # vLLM has no such route; the CLI falls back to its estimate
                self.send_error(404)
                return
            body = json.loads(raw or b"{}")
            compaction = SUMMARY_MARK in last_user_text(body)
            kind = "compact" if compaction else "main" if body.get("tools") else "aux"
            tokens = len(raw) // CHARS_PER_TOKEN
            size = tokens + int(body.get("max_tokens", 0))
            with episode.lock:
                episode.main_turns += kind == "main"
                episode.requests.append(
                    {"kind": kind, "tokens": tokens, "size": size, "overflow": size > episode.window}
                )
                blocks, stop = reply(episode, kind)
            if size > episode.window:
                message = f"This model's maximum context length is {episode.window} tokens. However, you requested {size} tokens."
                payload = json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": message}})
                self.send_response(400)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(payload.encode())
                return
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for event in sse_events(tokens, blocks, stop):
                self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())

    return Handler


def run_case(
    driver: types.ModuleType, args: argparse.Namespace, root: pathlib.Path, case: tuple[int, int, int]
) -> bool:
    window, reads, chars = case
    turns = args.turns
    work = root / f"window-{window}"
    (work / "home").mkdir(parents=True)
    line = "x" * 79 + "\n"
    files = [work / f"big-{index}.txt" for index in range(turns * reads)]
    for path in files:
        path.write_text(line * (chars // len(line)), encoding="utf-8")
    episode = Episode(window=window, reads=reads, files=files, turns=turns)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(episode))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    mcp = work / "mcp.json"
    mcp.write_text('{"mcpServers": {}}', encoding="utf-8")
    base = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(work / "home"),
        "ANTHROPIC_AUTH_TOKEN": "EMPTY",
        "ANTHROPIC_API_KEY": "EMPTY",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(MAX_OUTPUT_TOKENS),
        "CONTEXT_LENGTH": str(window),
    }
    os.environ.update({"CLAUDE_BIN": args.claude, "CLAUDE_MODEL": MODEL, "CLAUDE_MAX_TURNS": "100000"})
    context = types.SimpleNamespace(
        prompt="Read the files the tools name.",
        mcp_config=mcp,
        replica_root=f"http://127.0.0.1:{server.server_address[1]}",
        workdir=work,
    )
    environment = driver.claude_env(context, base)
    if args.without_fix:  # the environment every episode ran with before claude_context_env
        for name in driver.claude_context_env(base):
            environment.pop(name)
    log = work / "claude.log"
    with log.open("w", encoding="utf-8") as out:
        argv = driver.claude_command(context)
        process = subprocess.run(
            argv, cwd=work, env=environment, stdin=subprocess.DEVNULL, stdout=out, timeout=900, check=False
        )
    rc = process.returncode
    server.shutdown()
    compactions = sum(request["kind"] == "compact" for request in episode.requests)
    overflows = [request for request in episode.requests if request["overflow"]]
    largest = max(int(request["size"]) for request in episode.requests)
    print(
        f"window {window}: rc={rc} requests={len(episode.requests)} main_turns={episode.main_turns} "
        f"compactions={compactions} largest input+max_tokens={largest} overflows={len(overflows)} "
        f"env={ {key: value for key, value in environment.items() if 'COMPACT' in key or 'CONTEXT' in key} }"
    )
    return rc == 0 and compactions > 0 and not overflows and episode.main_turns > turns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--claude", default=os.environ.get("CLAUDE_BIN", "claude"), help="the claude binary")
    parser.add_argument("--turns", type=int, default=16, help="Read turns per case before the stub says DONE")
    parser.add_argument("--without-fix", action="store_true", help="drop claude_context_env: must FAIL")
    args = parser.parse_args()
    driver = load_driver()
    with tempfile.TemporaryDirectory(prefix="claude-compaction-") as root:
        passed = [run_case(driver, args, pathlib.Path(root), case) for case in CASES]
    print("PASS" if all(passed) else "FAIL")
    return 0 if all(passed) else 1


if __name__ == "__main__":
    sys.exit(main())
