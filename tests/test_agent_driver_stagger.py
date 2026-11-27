# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The agent start stagger: agents must not all initialize their MCP servers at once.

With every agent submitted to the pool at the same instant, most come up with mcp_servers status
"failed", and an agent without its MCP server has no submit tool at all.
"""

import types

import pytest

from tests.fresh_module import fresh


def load_driver(monkeypatch: pytest.MonkeyPatch, **env: str) -> types.ModuleType:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return fresh("agent_driver")


def test_the_stagger_is_on_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_START_STAGGER_SECONDS", raising=False)
    driver = load_driver(monkeypatch)
    assert driver.AGENT_START_STAGGER_SECONDS > 0, "agents would all start their MCP servers at once"


def test_only_the_first_wave_waits_its_slot_after_the_driver_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Worker 3 of a driver that started 5 s ago sleeps the 25 s left of its 30 s slot; an episode a worker picks
    up after the stagger window starts at once."""
    driver = load_driver(monkeypatch, AGENT_START_STAGGER_SECONDS="10", AGENT_START_STAGGER_MAX_SECONDS="600")
    slept: list[float] = []
    monkeypatch.setattr(driver.time, "sleep", slept.append)
    monkeypatch.setattr(driver.time, "time", lambda: 1000.0)
    driver.stagger_start(3, {driver.DRIVER_STARTED_ENV: "995"})
    driver.stagger_start(70, {driver.DRIVER_STARTED_ENV: "0"})
    assert slept == [25.0, 0.0]
