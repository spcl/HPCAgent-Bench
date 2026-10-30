# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A method packet reaches the agent through the MCP server and the driver; a setup sees only the core tools plus
its own packet's, never another packet's (e.g. ``canonical_parallel_form``)."""

import json
import subprocess
import sys
from types import ModuleType

import pytest

from hpcagent_bench import paths, plugins, prompt_build
from tests.fresh_module import fresh
from tests.mcp_stdio import served_tools, server_env, tool_names

REPO = paths.ROOT
WORKFLOW = REPO / "plugins" / "autokernel" / "prompt" / "workflow.md"
#: What the autokernel packet exports: its tool.
AUTOKERNEL = {"HPCAGENT_BENCH_PACKET_TOOLS": "experiment"}
#: ``search`` is excluded: it is not a core tool (only the ``web-search`` packet carries it).
CORE_TOOLS = {"score", "submit", "profile", "syntax_check"}

#: The env the launcher exports from the resolved packet's ``tools`` (``hpcagent_bench.packets.PACKET_TOOLS_ENV``):
#: what makes ``canonical_parallel_form`` or ``search`` a tool of THAT setup and of no other.
TOOLS_ENV = "HPCAGENT_BENCH_PACKET_TOOLS"
CPF_TOOLS = {TOOLS_ENV: "canonical_parallel_form"}
SEARCH_TOOLS = {TOOLS_ENV: "search"}

#: The skills packet's env: it stages pages and carries no tool.
SKILLS_ENV: dict[str, str] = {}


def registry_view(**env: str) -> dict[str, object]:
    """The offered tools and the rendered prompt tool list, as ``mcp_server --describe`` answers the driver under
    ``env``: a fresh process, the way the driver asks and the container spawns the server."""
    result = subprocess.run(
        [sys.executable, "-m", "hpcagent_agent.tools.mcp_server", "--describe"],
        env=server_env(env),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    described = json.loads(result.stdout)
    return {"allowed": described["allowed_tools"], "prompt": described["prompt"]}


def load_driver() -> ModuleType:
    """agent/hpcagent_agent/driver/agent_driver.py as a module; it imports its sibling harnesses.py by bare name."""
    return fresh("agent_driver")


def test_without_a_packet_the_server_serves_the_core_tools_only() -> None:
    """The default setup's tool list must not change because a packet directory exists."""
    result = served_tools()
    assert result.returncode == 0, result.stderr
    assert tool_names(result) == CORE_TOOLS


def test_the_autokernel_packet_adds_the_study_tool_to_the_core_tools() -> None:
    """The plugin's tool is served next to the unchanged core set."""
    result = served_tools(**AUTOKERNEL)
    assert result.returncode == 0, result.stderr
    assert tool_names(result) == CORE_TOOLS | {"experiment"}


def test_the_cpf_page_packet_is_the_only_setup_served_the_canonical_parallel_form_tool() -> None:
    """The bare setup and the skills setup must not see a tool whose whole answer there is
    ``unavailable``; the cpf setup, whose packet pins the rendered view, must."""
    assert tool_names(served_tools()) == CORE_TOOLS
    assert tool_names(served_tools(**SKILLS_ENV)) == CORE_TOOLS
    assert tool_names(served_tools(**CPF_TOOLS)) == CORE_TOOLS | {"canonical_parallel_form"}


def test_search_is_off_by_default_because_a_run_must_not_have_internet_access() -> None:
    """``search`` reaches the real internet (SerpAPI + a page crawl), and this benchmark's runs must not have
    internet access -- so it is no core tool: only the ``web-search`` packet carries it, and no shipped experiment
    uses that packet. A default setup, a packet setup and a no-score setup must all omit it from ``tools/list``."""
    assert "search" not in tool_names(served_tools())
    assert "search" not in tool_names(served_tools(**SKILLS_ENV))
    assert "search" not in tool_names(served_tools(**CPF_TOOLS))
    assert "search" not in tool_names(served_tools(AGENT_SUBMISSION_MODE="blind"))


def test_the_web_search_packet_serves_it_and_nothing_else_changes() -> None:
    """The ``web-search`` packet is the only way ``search`` appears -- exercised so the opt-in path itself is
    proven live, not just the default-off path."""
    assert tool_names(served_tools(**SEARCH_TOOLS)) == CORE_TOOLS | {"search"}


def test_search_is_excluded_from_allowed_tools_and_the_prompt_when_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Like ``score`` in blind mode, an unprovisioned ``search`` must be invisible everywhere: not offered
    is not offered, not merely unusable."""
    monkeypatch.delenv(TOOLS_ENV, raising=False)
    bare = registry_view()
    assert "search" not in bare["allowed"]
    assert "`search`" not in bare["prompt"]
    opted_in = registry_view(**SEARCH_TOOLS)
    assert "search" in opted_in["allowed"]
    assert "`search`" in opted_in["prompt"]


def test_the_allowed_list_and_the_prompt_follow_the_packet_the_setup_carries() -> None:
    """``--allowedTools`` is built from the same set as ``tools/list``, so a tool this setup's packet
    does not carry is invisible rather than merely unusable. The PROMPT text is identical in all
    three setups: canonical_parallel_form never had a bullet, so gating it moves no recorded prompt."""
    bare = registry_view()
    skills = registry_view(**SKILLS_ENV)
    cpf = registry_view(**CPF_TOOLS)
    assert "canonical_parallel_form" not in bare["allowed"]
    assert "canonical_parallel_form" not in skills["allowed"]
    assert "canonical_parallel_form" in cpf["allowed"]
    assert bare["prompt"] == skills["prompt"] == cpf["prompt"]


def test_every_tool_a_packet_declares_is_a_gated_tool_of_the_server() -> None:
    """The agent image carries no ``hpcagent_bench``, so the packets' ``tools`` and the server's ``core=False`` tools
    are held together by this test only. A drift here is a tool served to every setup although a packet carries
    it, a packet tool the server has no module for, or a gated tool no packet can switch on."""
    from hpcagent_bench import plugins, study_tags

    declared = {tool for definition in study_tags.registry().packet_defs.values() for tool in definition.tools}
    from hpcagent_agent.tools import registry

    fresh("mcp_server").load_tools()
    gated = {name for name, entry in registry.REGISTRY.items() if not entry.core}
    # next_phase is switched on by a plugin packet with an agent phase (tests/test_plugin_phases.py)
    assert declared | {plugins.NEXT_PHASE} == gated


def test_the_http_loop_prompt_documents_the_packet_tool_only_where_the_run_serves_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """miniswe reaches the same judge by curl, and their prompt's tool section is
    built from ``hpcagent_bench/tools/*.md`` rather than from the MCP registry -- so the withdrawal
    above did not reach them and every setup was handed the ``canonical_parallel_form`` curl line. The
    route answers ``unavailable`` without a view, which is the turn the MCP gate exists to save."""
    from hpcagent_bench.harness import prompts

    monkeypatch.setenv(TOOLS_ENV, "")
    assert "tools/canonical_parallel_form.md" not in prompts.tool_fragments()
    monkeypatch.setenv(TOOLS_ENV, "canonical_parallel_form")
    assert "tools/canonical_parallel_form.md" in prompts.tool_fragments()


def test_a_packet_tool_page_is_staged_by_that_packet_and_by_no_other() -> None:
    """One setup, one packet. A packet staging a cpf-tool page -- the manual for a tool only the cpf
    setup is served -- has its agents read instructions for a tool they do not have, and every call
    answers ``unavailable``."""
    from hpcagent_bench import packets

    assert packets.tool_pages() == {"cpf-tool-cpu", "cpf-tool-gpu", "cpf-tool-cpu-kinds", "cpf-tool-gpu-kinds"}
    staged = set(packets.resolve("lang-skills", "c").pages)
    assert not staged & packets.tool_pages()
    assert staged, "the skills packet must still stage the language and method pages"
    assert set(packets.resolve("cpf-tool", "c", environ={"HPCAGENT_BENCH_CPF_CACHE": "/views/cpf"}).pages) == {
        "cpf-tool-cpu"
    }


def test_an_unknown_packet_tool_stops_the_server() -> None:
    """A misnamed packet tool must fail the MCP start instead of serving a setup that lacks it."""
    result = served_tools(HPCAGENT_BENCH_PACKET_TOOLS="nosuchtool")
    assert result.returncode != 0
    assert "nosuchtool" in result.stderr


def test_without_a_packet_the_driver_adds_no_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """No packet: no packet tool is allowed."""
    monkeypatch.delenv(TOOLS_ENV, raising=False)
    driver = load_driver()
    assert "experiment" not in driver.agent_tools()


def test_the_driver_allows_the_packet_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """The autokernel packet: claude's allow-list gains its tool."""
    for name, value in AUTOKERNEL.items():
        monkeypatch.setenv(name, value)
    driver = load_driver()
    assert driver.agent_tools().count("experiment") == 1, "allowed once, as the server serves it"


def workflow_vars(packet: str) -> dict[str, str]:
    """The slots the resolved agent prompt of ``packet`` fills in its workflow block."""
    prompt = prompt_build.resolve_layers([], packet)
    return next(dict(block.vars) for block in prompt.blocks if block.id == prompt_build.WORKFLOW_BLOCK)


def test_without_a_packet_the_resolved_prompt_fills_no_workflow() -> None:
    assert workflow_vars("") == {}


def test_the_packets_workflow_fills_the_workflow_slot_of_the_resolved_prompt() -> None:
    """The staged prompt.md carries the workflow, so the driver has no slot left to fill."""
    assert workflow_vars("autokernel") == {"WORKFLOW": WORKFLOW.read_text(encoding="utf-8").rstrip("\n")}


def test_an_overlay_its_plugin_does_not_ship_is_refused() -> None:
    """An unknown overlay stops the prompt's resolution rather than launching agents without it."""
    with pytest.raises(ValueError, match="ships no prompt/nosuch.yaml"):
        prompt_build.load_overlay("autokernel:prompt/nosuch.yaml")
