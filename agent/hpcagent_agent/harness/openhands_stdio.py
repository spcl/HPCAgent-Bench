# Copyright 2026 the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The OpenHands runner's stdio MCP client factory: one handle per ``mcpServers`` entry.

The driver's server runs on the driver's interpreter (the ``command`` its mcp.json carries), so the runner
needs only a client. A handle is constructed without connecting, and the ``mcp`` imports stay inside the
functions: the package exists only in the OpenHands venv (``agent/harness/openhands``, mcp 1.x, while the
agent package pins mcp >= 2.3 for the server), so an interpreter without it must still import this module
and collect the tests that only build a handle."""

from collections.abc import Mapping
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, AsyncIterator

__all__ = ["SERVER_KEYS", "StdioClient", "server_parameters"]

#: The entry keys a stdio launch takes; a Claude-format config entry may carry transport keys a stdio
#: client must not see (``url`` and friends), and :func:`server_parameters` drops them.
SERVER_KEYS = ("command", "args", "env", "cwd")


def server_parameters(server: Mapping[str, Any]) -> Any:  # noqa: ANN401
    """One ``mcpServers`` entry as the stdio launch parameters: the interpreter to run the server on and
    its argv (the driver's mcp.json names the driver's own interpreter and server module)."""
    from mcp import StdioServerParameters

    return StdioServerParameters(**{key: server[key] for key in SERVER_KEYS if key in server})


class StdioClient:
    """A stdio client handle for one ``mcpServers`` entry: the launch parameters at construction, the
    connection and session only inside :meth:`connect`."""

    def __init__(self, server: Mapping[str, Any]) -> None:
        self.params = server_parameters(server)

    @property
    def command(self) -> list[str]:
        """The server's launch argv: the interpreter that runs it and the module with its arguments."""
        return [str(self.params.command), *(str(argument) for argument in self.params.args or ())]

    @asynccontextmanager
    async def connect(
        self, read_timeout_seconds: float | timedelta | None = None
    ) -> AsyncIterator[Any]:  # noqa: ANN401
        """The connected ``ClientSession`` over stdio: the transport, then the session whose per-call wait
        is ``read_timeout_seconds`` (spelled per mcp generation by ``run_openhands.read_timeout``)."""
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        async with stdio_client(self.params) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=read_timeout_seconds) as session:
                yield session
