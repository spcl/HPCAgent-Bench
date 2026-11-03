# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A tool call the judge keeps busy sends progress until it answers, so a client's idle timeout never ends it."""

import time
import types as builtin_types
from typing import Any

import anyio
import pytest
from mcp import types

from hpcagent_agent.tools import mcp_server


class Session:
    """Records the progress notifications a handler sends."""

    def __init__(self) -> None:
        self.progress: list[tuple[object, float]] = []

    async def send_progress_notification(self, token: object, progress: float, message: str = "") -> None:
        self.progress.append((token, progress))


def slow_answer(offered: object, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
    time.sleep(0.35)
    return types.CallToolResult.model_validate({"content": [{"type": "text", "text": name}], "isError": False})


@pytest.mark.parametrize(("meta", "beats"), [({"progress_token": 7}, True), (None, False)])
def test_a_slow_call_sends_progress_until_it_answers(
    monkeypatch: pytest.MonkeyPatch, meta: dict[str, object] | None, beats: bool
) -> None:
    monkeypatch.setattr(mcp_server, "answer", slow_answer)
    monkeypatch.setattr(mcp_server, "HEARTBEAT_SECONDS", 0.02)
    session = Session()
    context = builtin_types.SimpleNamespace(meta=meta, session=session)
    params = builtin_types.SimpleNamespace(name="score", arguments={})
    result = anyio.run(mcp_server.call_handler({}), context, params)
    assert result.content[0].text == "score"
    assert (len(session.progress) >= 2) is beats
    assert all(token == 7 for token, _ in session.progress)


if __name__ == "__main__":
    with pytest.MonkeyPatch.context() as patch:
        test_a_slow_call_sends_progress_until_it_answers(patch, {"progress_token": 7}, True)
    with pytest.MonkeyPatch.context() as patch:
        test_a_slow_call_sends_progress_until_it_answers(patch, None, False)
