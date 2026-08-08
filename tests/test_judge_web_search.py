#!/usr/bin/env python3
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Network-free test for containers/judge/tools/web_search.py."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "containers" / "judge" / "tools" / "web_search.py"


class FakeHandler(BaseHTTPRequestHandler):
    serpapi_payload: dict[str, Any] = {
        "organic_results": [
            {
                "title": "Crawl4AI docs",
                "link": "https://example.test/crawl4ai",
                "snippet": "AsyncWebCrawler converts pages to markdown.",
            },
            {
                "title": "vLLM docs",
                "link": "https://example.test/vllm",
                "snippet": "vLLM serves OpenAI-compatible chat completions.",
            },
        ]
    }
    llm_payload: dict[str, Any] = {
        "choices": [{
            "message": {
                "content":
                "Use AsyncWebCrawler to collect markdown, then summarize with vLLM.\n\nSources:\n- https://example.test/crawl4ai"
            }
        }]
    }

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path.startswith("/search.json"):
            self._json(self.serpapi_payload)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            assert body["model"] == "fake-vllm"
            assert "messages" in body
            self._json(self.llm_payload)
            return
        self.send_error(404)

    def _json(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    env = os.environ.copy()
    env.update({
        "SERPAPI_API_KEY":
        "fake",
        "SERPAPI_URL":
        f"http://127.0.0.1:{port}/search.json",
        "WEBSEARCH_LLM_BASE_URL":
        f"http://127.0.0.1:{port}/v1",
        "WEBSEARCH_LLM_MODEL":
        "fake-vllm",
        "WEBSEARCH_FAKE_CRAWL_JSON":
        json.dumps({
            "https://example.test/crawl4ai": "Crawl4AI provides AsyncWebCrawler and markdown output.",
            "https://example.test/vllm": "vLLM has OpenAI-compatible chat completions endpoints.",
        }),
    })

    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(TOOL), "--query", "How should web search work?", "--max-results", "2", "--max-pages", "2"
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    finally:
        server.shutdown()

    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        return proc.returncode

    payload = json.loads(proc.stdout)
    assert payload["query"] == "How should web search work?"
    assert "AsyncWebCrawler" in payload["answer"]
    assert len(payload["search_results"]) == 2
    assert len(payload["crawled_pages"]) == 2
    assert payload["crawled_pages"][0]["success"] is True
    print("web_search fake SerpAPI/vLLM test passed")
    return 0


def test_web_search_tool_answers_via_fake_serpapi_crawl_and_llm() -> None:
    """containers/judge/tools/web_search.py end to end, against a local fake HTTP server --
    no real network egress, no SERPAPI_API_KEY, no crawl4ai import (WEBSEARCH_FAKE_CRAWL_JSON
    short-circuits the real Crawl4AI dependency in the tool itself)."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
