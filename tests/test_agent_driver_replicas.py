# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""agent_driver.py: what a vLLM replica that is not ready yet costs the run.

The driver used to wait on replicas one after another with a hard failure, which made the slowest
replica the deadline for all of them and turned one laggard into a dead arm: on llr4, oss
589511/512/513/516 lost all 242 agents and wrote zero judge rows because a replica was still
capturing CUDA graphs when its wait expired. A replica that misses the deadline is usually late
rather than dead, and LiteLLM keeps every upstream in rotation regardless of what the driver saw,
so the run must proceed on whatever answered -- while still refusing to start with nothing.

The returned order is pinned too. Completion order is a race between replicas, and a run's logs
should not differ between two identical runs.
"""

import http.server
import importlib.util
import pathlib
import sys
import threading
import time
from types import ModuleType
from typing import ClassVar

import pytest

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "experiments"


def load_example_module(name: str) -> ModuleType:
    """``sys.modules`` must carry the module BEFORE exec, matching tests/test_validate_run.py."""
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="driver")
def driver_fixture() -> ModuleType:
    return load_example_module("agent_driver")


REPLICAS = ["http://a:8000", "http://b:8000", "http://c:8000"]


def test_a_late_replica_does_not_abort_the_run(driver, monkeypatch) -> None:

    def probe(name: str, replica: str, timeout: float, headers: dict[str, str] | None = None) -> None:
        if replica.startswith("http://b:"):
            raise TimeoutError("still capturing CUDA graphs")

    monkeypatch.setattr(driver, "wait_for_engine", probe)
    assert driver.wait_for_ready_replicas(REPLICAS, 1.0, {}) == ["http://a:8000", "http://c:8000"]


def test_no_ready_replica_is_still_a_failure(driver, monkeypatch) -> None:
    """Proceeding on a subset must not become proceeding on nothing: with every replica down there
    is no endpoint to serve the agents, and starting anyway would burn the allocation producing
    242 identical connection errors."""

    def probe(name: str, replica: str, timeout: float, headers: dict[str, str] | None = None) -> None:
        raise TimeoutError("never came up")

    monkeypatch.setattr(driver, "wait_for_engine", probe)
    with pytest.raises(TimeoutError, match="no vLLM replica became ready"):
        driver.wait_for_ready_replicas(REPLICAS, 1.0, {})


def test_ready_replicas_come_back_in_replica_order(driver, monkeypatch) -> None:
    """The first replica is made the slowest, so completion order is the REVERSE of replica order
    and a version that returned as_completed order would fail here."""

    def probe(name: str, replica: str, timeout: float, headers: dict[str, str] | None = None) -> None:
        if replica.startswith("http://a:"):
            time.sleep(0.2)

    monkeypatch.setattr(driver, "wait_for_engine", probe)
    assert driver.wait_for_ready_replicas(REPLICAS, 1.0, {}) == REPLICAS


class FakeEngineHandler(http.server.BaseHTTPRequestHandler):
    """A GET-only vLLM/sglang stand-in: ``/v1/models`` always answers, ``/health`` tracks warmup."""

    health_status: ClassVar[int] = 200

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            status, body = 200, b"{}"
        elif self.path == "/health":
            status = type(self).health_status
            body = b"OK" if status < 400 else b"warming up"
        else:
            status, body = 404, b""
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def start_fake_engine(health_status: int) -> tuple[http.server.ThreadingHTTPServer, str]:
    """Start a real HTTP server standing in for one vLLM/sglang replica; returns (server, its base url)."""
    handler = type("FakeEngine", (FakeEngineHandler,), {"health_status": health_status})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


def test_a_replica_stuck_before_health_is_not_returned_as_ready(driver) -> None:
    """A replica that answers /v1/models but whose /health still 503s (mid warmup, the failure mode
    behind the qwen38 2026-09-17 23:00 incident) must not be handed agents, even though /v1/models
    alone would have looked ready under the old single-phase gate."""
    ready_server, ready_url = start_fake_engine(200)
    stuck_server, stuck_url = start_fake_engine(503)
    try:
        assert driver.wait_for_ready_replicas([ready_url, stuck_url], 1.0, {}) == [ready_url]
    finally:
        for server in (ready_server, stuck_server):
            server.shutdown()
            server.server_close()
