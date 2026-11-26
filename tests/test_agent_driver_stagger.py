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


