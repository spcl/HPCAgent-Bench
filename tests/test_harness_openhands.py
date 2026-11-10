# Copyright 2026 the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The OpenHands harness's own uv project and lock (``agent/harness/openhands``) and its stdio MCP client.

The agent package pins mcp >= 2.3 for the benchmark's MCP server while openhands-sdk 1.58.2 resolves
fastmcp 3, and so mcp < 2, so the harness venv is synced from its own lock like TileLang's overlay and
never from the ``harness-`` groups: static checks here, the client against the real server this
interpreter can start, and the built venv where ``launch_venv.sh`` made one (network)."""

import asyncio
import datetime
import importlib.metadata
import json
import os
import pathlib
import subprocess
import sys
import tomllib

import pytest

from hpcagent_agent.harness import openhands_stdio, run_openhands, runner_common

from hpcagent_bench import paths

from tests.mcp_stdio import STRIPPED

ROOT: pathlib.Path = paths.ROOT
PROJECT: pathlib.Path = ROOT / "agent" / "harness" / "openhands"
HOOK: pathlib.Path = ROOT / "containers" / "images" / "lib" / "launch_venv.sh"
#: The venv the launch hook syncs inside the launch venv; a host that never launched an agent step has none.
VENV: pathlib.Path = pathlib.Path(sys.prefix) / "harness" / "openhands"

#: The core tools every setup is served (tests/test_packet_wiring.py asserts the same set over raw stdio).
CORE_TOOLS = {"score", "submit", "profile", "syntax_check"}


def project() -> dict:
    """The parsed ``pyproject.toml`` of the OpenHands project."""
    return tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))


def lock(path: pathlib.Path) -> dict:
    """A parsed ``uv.lock``."""
    return tomllib.loads(path.read_text(encoding="utf-8"))


def locked(lock_data: dict, name: str) -> dict:
    """One locked package's entry."""
    return next(package for package in lock_data["package"] if package["name"] == name)


def hook() -> str:
    return HOOK.read_text(encoding="utf-8")


def openhands_block() -> str:
    """The launch hook's own-project block: the ``if`` naming the lock through its ``fi``."""
    lines = hook().splitlines()
    start = next(index for index, line in enumerate(lines) if '"${harness}/openhands/uv.lock"' in line)
    end = next(index for index in range(start, len(lines)) if lines[index].strip() == "fi")
    return "\n".join(lines[start : end + 1])


def test_the_openhands_project_stays_out_of_the_agent_package() -> None:
    """The own project and a ``harness-`` group are two ways to the same venv: the group would drag the
    agent package's mcp >= 2.3 into a lock openhands-sdk cannot sign, so exactly one exists."""
    agent = tomllib.loads((ROOT / "agent" / "pyproject.toml").read_text(encoding="utf-8"))
    assert "harness-openhands" not in agent.get("dependency-groups", {})
    data = project()
    assert data["project"]["name"] == "hpcagent-bench-openhands"
    dependencies = data["project"]["dependencies"]
    assert dependencies == ["openhands-sdk==1.58.2", "openhands-tools==1.58.2", "pydantic>=2.12"]
    assert not any("hpcagent-agent" in dependency for dependency in dependencies)
    assert data["tool"]["uv"]["package"] is False


def test_the_workspace_lock_stays_single_mcp() -> None:
    """The agent package's lock carries one mcp, 2.3 or later, and no OpenHands stack: dual 1.x/2.x
    support was removed again, so only the OpenHands project's own lock may hold an older one."""
    root = lock(ROOT / "uv.lock")
    assert int(locked(root, "mcp")["version"].partition(".")[0]) >= 2
    assert locked(root, "mcp")["version"] >= "2.3"
    names = {package["name"] for package in root["package"]}
    assert not names & {"openhands-sdk", "openhands-tools", "fastmcp"}


def test_the_openhands_lock_freezes_the_project_at_the_repos_lock_revision() -> None:
    """The image's pinned uv reads revision <root lock's>: a newer revision fails ``uv sync --frozen`` at
    launch, hours after anyone edited the project."""
    openhands = lock(PROJECT / "uv.lock")
    assert openhands["revision"] == lock(ROOT / "uv.lock")["revision"]
    assert locked(openhands, "openhands-sdk")["version"] == "1.58.2"
    assert locked(openhands, "openhands-tools")["version"] == "1.58.2"
    assert locked(openhands, "mcp")["version"].partition(".")[0] == "1"
    names = {package["name"] for package in openhands["package"]}
    assert not names & {"hpcagent-agent", "hpcagent-bench"}


def test_the_launch_hook_syncs_the_openhands_project_beside_the_groups() -> None:
    """The hook syncs the own lock like TileLang's, puts agent/ on the venv's path with a .pth, and gates
    the build on the venv's mcp being 1.x and hpcagent_bench unreachable -- before any agent starts."""
    block = openhands_block()
    assert 'UV_PROJECT_ENVIRONMENT="${venv}/harness/openhands"' in block
    assert "uv sync -q --frozen --inexact --python" in block
    assert '"${workspace}/agent" > "$(site "${venv}/harness/openhands/bin/python")/zz-agent.pth"' in block
    # The agent package (mcp >= 2.3) must never be installed into this venv.
    assert "--package hpcagent-agent" not in block
    assert "the openhands venv pins mcp < 2" in block
    assert "openhands can import hpcagent_bench" in block
    table = hook().split("done <<EOF\n", 1)[1].split("\nEOF", 1)[0]
    assert "openhands" not in table


def test_the_read_timeout_spells_each_mcp_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ClientSession``'s ``read_timeout_seconds`` takes seconds on mcp 2.x and a ``timedelta`` on the
    1.x the OpenHands venv resolves; the runner is the only client that may see either."""
    monkeypatch.setattr(run_openhands.importlib.metadata, "version", lambda _name: "1.30.0")
    assert run_openhands.read_timeout(5) == datetime.timedelta(seconds=5)
    monkeypatch.setattr(run_openhands.importlib.metadata, "version", lambda _name: "2.3.0")
    assert run_openhands.read_timeout(5) == 5


def mcp_major() -> int:
    """The installed ``mcp`` major: the driver's server is mcp 2.x, the OpenHands venv's client 1.x."""
    return int(importlib.metadata.version("mcp").partition(".")[0])


def test_the_stdio_factory_imports_without_the_mcp_package() -> None:
    """``mcp`` lives only in the OpenHands venv, so the factory's imports wait for the first handle: an
    interpreter without the package still imports the module, and collection with it never breaks."""
    done = subprocess.run(
        [sys.executable, "-c", "import sys; import hpcagent_agent.harness.openhands_stdio; print('mcp' in sys.modules)"],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
        cwd=str(ROOT),
    )
    assert done.stdout.strip() == "False"


def test_the_stdio_factory_builds_the_drivers_server_command_without_connecting(tmp_path: pathlib.Path) -> None:
    """The handle is the launch parameters only: the driver's interpreter and its server module from the
    mcp.json entry, with the entry's non-stdio transport keys dropped -- construction starts no process."""
    entry = {
        "command": sys.executable,
        "args": ["-m", "hpcagent_agent.tools.mcp_server"],
        "env": {"HPCAGENT_BENCH_TEST": "1"},
        "cwd": str(tmp_path),
        "url": "http://mcp.invalid/sse",
    }
    handle = openhands_stdio.StdioClient(entry)
    assert handle.command == [sys.executable, "-m", "hpcagent_agent.tools.mcp_server"]
    assert handle.params.env == {"HPCAGENT_BENCH_TEST": "1"}
    assert handle.params.cwd == str(tmp_path)


@pytest.mark.skipif(
    mcp_major() < 2,
    reason="the driver's MCP server is written against mcp 2.x (agent/pyproject.toml); this interpreter resolves 1.x",
)
def test_an_openhands_client_lists_the_bench_tools(tmp_path: pathlib.Path) -> None:
    """The runner's stdio client lists the core tools of the real server this interpreter starts: the
    server runs on the driver's interpreter, so a client that cannot list its tools ends the attempt at
    the runner's probe, before the SDK's first model call."""
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "hpcagent_bench": {"command": sys.executable, "args": ["-m", "hpcagent_agent.tools.mcp_server"]}
                }
            }
        ),
        encoding="utf-8",
    )
    environ = {key: value for key, value in os.environ.items() if key not in STRIPPED}
    offered = asyncio.run(run_openhands.bench_tools(runner_common.mcp_servers(config, environ, tmp_path)))
    assert CORE_TOOLS <= set(offered["hpcagent_bench"])


@pytest.mark.skipif(
    not (VENV / "bin" / "python").is_file(),
    reason="the openhands venv is synced at container launch from its own lock (network); none here",
)
def test_the_openhands_venv_pins_mcp_1_without_the_bench() -> None:
    """The harness venv's mcp is 1.x, it imports the runner's common module through the launch hook's .pth,
    and it cannot import the bench: the driver's judge stack stays out of the agent's clients."""
    probe = (
        "import importlib.metadata, importlib.util\n"
        "import hpcagent_agent.harness.runner_common\n"
        "print(importlib.metadata.version('mcp'), importlib.util.find_spec('hpcagent_bench') is None)"
    )
    done = subprocess.run(
        [str(VENV / "bin" / "python"), "-c", probe], capture_output=True, text=True, timeout=120, check=True
    )
    version, no_bench = done.stdout.split()
    assert version.startswith("1."), version
    assert no_bench == "True"
