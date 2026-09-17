#!/usr/bin/env python3
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Network-free test for containers/judge/tools/web_search.py."""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "containers" / "judge" / "tools" / "web_search.py"


def load_web_search() -> types.ModuleType:
    """``web_search.py`` by path, exactly as ``experiments/judge_service.py`` loads it (TOOLS_DIR on
    ``sys.path``), so ``isinstance(exc, web_search.NotProvisionedError)`` checks the same class the
    router would catch."""
    spec = importlib.util.spec_from_file_location("web_search_contract_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        "choices": [
            {
                "message": {
                    "content": "Use AsyncWebCrawler to collect markdown, then summarize with vLLM.\n\nSources:\n- https://example.test/crawl4ai"
                }
            }
        ]
    }

    def log_message(self, fmt: str, *args: object) -> None:
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
    env.update(
        {
            "SERPAPI_API_KEY": "fake",
            "SERPAPI_URL": f"http://127.0.0.1:{port}/search.json",
            "WEBSEARCH_LLM_BASE_URL": f"http://127.0.0.1:{port}/v1",
            "WEBSEARCH_LLM_MODEL": "fake-vllm",
            "WEBSEARCH_FAKE_CRAWL_JSON": json.dumps(
                {
                    "https://example.test/crawl4ai": "Crawl4AI provides AsyncWebCrawler and markdown output.",
                    "https://example.test/vllm": "vLLM has OpenAI-compatible chat completions endpoints.",
                }
            ),
        }
    )

    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--query",
                "How should web search work?",
                "--max-results",
                "2",
                "--max-pages",
                "2",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()

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


#: Every ``experiments/.env.*`` ships this empty as of 2026-09-17 (98/98 arms), so this is the
#: config every campaign arm actually runs search under today.
UNPROVISIONED_ENV = {"SERPAPI_API_KEY": "", "WEBSEARCH_LLM_BASE_URL": "", "WEBSEARCH_LLM_MODEL": ""}


def test_missing_serpapi_key_raises_not_provisioned_not_a_bare_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``experiments/judge_service.py`` tells a 'never configured' 503 apart from a 'this call
    failed' 502 by ``isinstance(exc, NotProvisionedError)`` -- so a config gap must raise THAT
    class, not a plain :class:`RuntimeError` a real SerpAPI/crawl/LLM failure also raises."""
    web_search = load_web_search()
    for key, value in UNPROVISIONED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("SERPAPI_URL", raising=False)
    with pytest.raises(web_search.NotProvisionedError):
        web_search.call_serpapi("how do I use crawl4ai", 3, 5.0)
    # It IS a RuntimeError too: main()'s existing except clause must keep catching it.
    with pytest.raises(RuntimeError):
        web_search.call_serpapi("how do I use crawl4ai", 3, 5.0)


@pytest.mark.parametrize("missing", ["WEBSEARCH_LLM_BASE_URL", "WEBSEARCH_LLM_MODEL"])
def test_missing_llm_config_raises_not_provisioned(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    """The LLM synthesis leg is provisioning too: an arm with a SerpAPI key but no configured
    judge-local LLM is just as unprovisioned as one with neither."""
    web_search = load_web_search()
    monkeypatch.setenv("WEBSEARCH_LLM_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("WEBSEARCH_LLM_MODEL", "fake-model")
    monkeypatch.setenv(missing, "")
    with pytest.raises(web_search.NotProvisionedError):
        web_search.call_llm("query", [], 5.0)


def test_a_real_search_failure_is_still_a_plain_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The distinction only means something if a genuine failure of a PROVISIONED search does NOT
    also raise ``NotProvisionedError`` -- else every 502 would misreport as a 503. A key and an LLM
    endpoint are both configured here; what fails is SerpAPI itself answering no usable results,
    exactly the shape ``run_web_search`` raises for a real, mid-run search failure."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original_payload = FakeHandler.serpapi_payload
    try:
        FakeHandler.serpapi_payload = {"organic_results": []}
        web_search = load_web_search()
        monkeypatch.setenv("SERPAPI_API_KEY", "fake")
        monkeypatch.setenv("SERPAPI_URL", f"http://127.0.0.1:{server.server_address[1]}/search.json")
        monkeypatch.setenv("WEBSEARCH_LLM_BASE_URL", "http://127.0.0.1:1/v1")
        monkeypatch.setenv("WEBSEARCH_LLM_MODEL", "fake-model")
        with pytest.raises(RuntimeError) as excinfo:
            web_search.run_web_search("query", max_results=3, timeout=5.0)
        assert not isinstance(excinfo.value, web_search.NotProvisionedError)
    finally:
        FakeHandler.serpapi_payload = original_payload
        server.shutdown()
        server.server_close()


def test_cli_marks_a_not_provisioned_refusal_with_a_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI (a hand-run debugging path, same contract as the service) must carry the same
    distinction into its JSON error body, not just the exit code."""
    env = os.environ.copy()
    env.update(UNPROVISIONED_ENV)
    env.pop("SERPAPI_URL", None)
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--query", "anything"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    body = json.loads(proc.stderr)
    assert body["ok"] is False
    assert body["cause"] == "not_provisioned"


if __name__ == "__main__":
    raise SystemExit(main())
