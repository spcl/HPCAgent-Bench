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

import importlib.util
import pathlib
import sys
import time
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


REPLICAS = ["http://a:8000", "http://b:8000", "http://c:8000"]


def test_a_late_replica_does_not_abort_the_run(driver, monkeypatch):

    def probe(name: str, url: str, timeout: float, headers=None) -> None:
        if url.startswith("http://b:"):
            raise TimeoutError("still capturing CUDA graphs")

    monkeypatch.setattr(driver, "wait_for_json", probe)
    assert driver.wait_for_ready_replicas(REPLICAS, 1.0, {}) == ["http://a:8000", "http://c:8000"]


def test_no_ready_replica_is_still_a_failure(driver, monkeypatch):
    """Proceeding on a subset must not become proceeding on nothing: with every replica down there
    is no endpoint to serve the agents, and starting anyway would burn the allocation producing
    242 identical connection errors."""

    def probe(name: str, url: str, timeout: float, headers=None) -> None:
        raise TimeoutError("never came up")

    monkeypatch.setattr(driver, "wait_for_json", probe)
    with pytest.raises(TimeoutError, match="no vLLM replica became ready"):
        driver.wait_for_ready_replicas(REPLICAS, 1.0, {})


def test_ready_replicas_come_back_in_replica_order(driver, monkeypatch):
    """The first replica is made the slowest, so completion order is the REVERSE of replica order
    and a version that returned as_completed order would fail here."""

    def probe(name: str, url: str, timeout: float, headers=None) -> None:
        if url.startswith("http://a:"):
            time.sleep(0.2)

    monkeypatch.setattr(driver, "wait_for_json", probe)
    assert driver.wait_for_ready_replicas(REPLICAS, 1.0, {}) == REPLICAS
