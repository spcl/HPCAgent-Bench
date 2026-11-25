# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The agent start stagger: agents must not all initialize their MCP servers at once.

With every agent submitted to the pool at the same instant, most come up with mcp_servers status
"failed", and an agent without its MCP server has no submit tool at all.
"""

import types

import pytest

from tests.fresh_module import DRIVER_DIR, fresh


def load_driver(monkeypatch: pytest.MonkeyPatch, **env: str) -> types.ModuleType:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return fresh("agent_driver")


def test_the_stagger_is_on_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_START_STAGGER_SECONDS", raising=False)
    driver = load_driver(monkeypatch)
    assert driver.AGENT_START_STAGGER_SECONDS > 0, "agents would all start their MCP servers at once"


def test_both_mcp_budgets_are_raised() -> None:
    """An agent whose MCP server reports "failed" has no submit tool and records nothing. Claude
    Code has TWO budgets and the connect one defaults to 5 s -- raising only the 30 s startup
    budget leaves the tighter of the pair in place."""
    source = (DRIVER_DIR / "agent_driver.py").read_text()
    assert 'environment.setdefault("MCP_TIMEOUT"' in source
    assert 'environment.setdefault("MCP_CONNECT_TIMEOUT_MS"' in source
