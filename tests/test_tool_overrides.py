# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A plugin tool declared ``overrides=True`` replaces a core tool only for the packets that carry its name."""

from typing import Any

import pydantic
import pytest

from hpcagent_agent.tools import registry


class NoArgs(pydantic.BaseModel):
    pass


def core_run(payload: dict[str, Any]) -> dict[str, Any]:
    return {"from": "core"}


def plugin_run(payload: dict[str, Any]) -> dict[str, Any]:
    return {"from": "plugin"}


def test_an_override_replaces_the_core_tool_only_for_its_packet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "REGISTRY", {})
    monkeypatch.setattr(registry, "OVERRIDES", {})
    registry.tool(spec=registry.Spec("core", NoArgs), name="submit")(core_run)
    registry.tool(spec=registry.Spec("plugin", NoArgs), name="submit", overrides=True)(plugin_run)
    plain = registry.Setup.read({})
    assert registry.served(plain)["submit"].run({})["from"] == "core"
    packet = registry.Setup.read({registry.PACKET_TOOLS_ENV: "submit"})
    assert registry.served(packet)["submit"].run({})["from"] == "plugin"


def test_a_preview_override_stays_unserved_in_blind_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = registry.Tool("score", registry.Spec("plugin", NoArgs), plugin_run, core=False, preview=True)
    blind = registry.Setup.read({registry.PACKET_TOOLS_ENV: "score", "AGENT_SUBMISSION_MODE": "blind"})
    assert registry.offered(entry, blind) is blind.preview_served
