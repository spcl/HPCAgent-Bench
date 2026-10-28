# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A tool's hints reach ``tools/list`` as MCP annotations; a tool that says nothing carries none."""

import pydantic
from hpcagent_agent.tools import mcp_server, registry


class NoInput(pydantic.BaseModel):
    pass


def served(hints: registry.Hints) -> registry.Served:
    return registry.Served("look", "Look around.", {"type": "object"}, "", lambda payload: {"ok": True}, hints)


def test_a_read_only_tool_is_listed_with_its_hints_and_only_those() -> None:
    sut = mcp_server.tool_definition("look", served(registry.Hints(read_only=True, idempotent=True)))

    annotations = sut.model_dump(by_alias=True, exclude_none=True)["annotations"]
    assert annotations == {"readOnlyHint": True, "idempotentHint": True}


def test_a_tool_without_hints_is_listed_without_annotations() -> None:
    sut = mcp_server.tool_definition("look", served(registry.Hints()))

    assert sut.annotations is None
