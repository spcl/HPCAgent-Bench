# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""agent_driver.py: the single-stream generation-throughput probe.

The number this measures is per-request decode tok/s -- the one a 1.49 tok/s serve regression shows
up in and an aggregate figure hides. It runs once, before the agents start, because 40 concurrent
workers saturate the endpoint and there is no single-stream number left to take.

It is measurement, so it must never be able to end a run: a replica that errors, a reply with no
usage block, an unwritable run dir all cost the sample and nothing else.
"""

import importlib.util
import json
import pathlib
import sys
import urllib.error
from types import ModuleType

import pytest

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "containers/cluster/example-script"


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


class FakeResponse:
    """Just enough of urlopen's context manager for ``json.load`` to read a body."""

    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def reply(completion_tokens: int, prompt_tokens: int = 40) -> dict:
    return {
        "choices": [{"message": {"content": "x"}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def test_the_probe_counts_the_servers_tokens_not_its_own_request(driver, monkeypatch):
    """max_tokens is what was ASKED for; completion_tokens is what came back. A model that stops
    early must be scored on what it produced, or every early stop reads as a throughput drop."""
    seen: list[dict] = []
    clock = iter([0.0, 2.0])

    def fake_urlopen(request, timeout=None):
        seen.append(json.loads(request.data))
        return FakeResponse(reply(completion_tokens=100))

    monkeypatch.setattr(driver.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(driver.time, "monotonic", lambda: next(clock))
    monkeypatch.setenv("VLLM_SERVED_MODEL", "optarena-vllm")

    samples = driver.throughput_probe("http://vllm:8000/v1", {}, 1)

    assert len(samples) == 1
    assert samples[0]["decode_tok_s"] == pytest.approx(50.0)  # 100 tokens / 2 s, not PROBE_MAX_TOKENS
    body = seen[0]
    assert body["model"] == "optarena-vllm"  # vLLM answers its served name and nothing else
    assert body["stream"] is False, "a streamed reply has no usage block to count from"
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == driver.PROBE_MAX_TOKENS


def test_the_probe_posts_to_the_chat_route_of_the_replica_it_was_given(driver, monkeypatch):
    """The replica URL already ends in /v1 -- the probe must not rebuild it or invent a host."""
    urls: list[str] = []

    def fake_urlopen(request, timeout=None):
        urls.append(request.full_url)
        assert request.get_method() == "POST"
        return FakeResponse(reply(completion_tokens=10))

    monkeypatch.setattr(driver.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(driver.time, "monotonic", iter([0.0, 1.0]).__next__)

    driver.throughput_probe("http://nid002994:8000/v1", {}, 1)
    assert urls == ["http://nid002994:8000/v1/chat/completions"]


def test_a_failing_replica_costs_the_sample_and_not_the_run(driver, monkeypatch):
    """A probe that raises would kill the driver before a single agent started -- the exact trade
    the measurement is not worth. The bad request is dropped and the good ones still land."""
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("connection refused")
        if calls["n"] == 2:
            return FakeResponse({"choices": []})  # served, but no usage block
        return FakeResponse(reply(completion_tokens=60))

    monkeypatch.setattr(driver.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(driver.time, "monotonic", iter([0.0, 1.0, 2.0, 3.0, 4.0, 6.0]).__next__)

    samples = driver.throughput_probe("http://vllm:8000/v1", {}, 3)
    assert len(samples) == 1 and samples[0]["completion_tokens"] == 60


def test_the_report_takes_the_median_and_writes_the_raw_samples(driver, monkeypatch, tmp_path, capsys):
    """Median, not mean: the first request after readiness pays for whatever is still cold, and one
    such outlier moves a five-sample mean enough to argue about. The raw samples are kept so the
    outlier stays visible instead of being averaged away."""
    monkeypatch.setenv("RUN_DIR", str(tmp_path))
    monkeypatch.delenv("AGENT_NODE_RANK", raising=False)
    monkeypatch.setenv("SLURM_PROCID", "0")

    samples = [
        {"elapsed_s": 1.0, "completion_tokens": rate, "prompt_tokens": 40.0, "decode_tok_s": rate}
        for rate in (1.5, 40.0, 41.0, 42.0, 43.0)
    ]
    driver.report_throughput(samples)

    written = json.loads((tmp_path / "throughput-node0.json").read_text(encoding="utf-8"))
    assert written["median_decode_tok_s"] == pytest.approx(41.0)  # not the mean, which the 1.5 drags to 33.5
    assert len(written["samples"]) == 5
    assert "median=41.00 tok/s" in capsys.readouterr().out


def test_an_unwritable_run_dir_does_not_raise(driver, monkeypatch, tmp_path, capsys):
    """Reporting is the last thing between readiness and the agents; a bad path must not stop them."""
    monkeypatch.setenv("RUN_DIR", str(tmp_path / "does" / "not" / "exist"))
    driver.report_throughput(
        [{"elapsed_s": 1.0, "completion_tokens": 10.0, "prompt_tokens": 5.0, "decode_tok_s": 10.0}]
    )
    assert "could not write" in capsys.readouterr().out


def test_no_samples_reports_nothing_and_writes_nothing(driver, monkeypatch, tmp_path, capsys):
    """Every request failing is a real outcome. It must be SAID, not silently written as a zero that
    later reads as a measured throughput of zero."""
    monkeypatch.setenv("RUN_DIR", str(tmp_path))
    driver.report_throughput([])
    assert "no samples" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []
