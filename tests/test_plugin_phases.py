# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A plugin packet's phases and prompt table: the agent is served the tools of the phases it runs, the workflow names
every phase in order with who runs it, ``extends`` starts a packet from another, and the table compiles to an overlay
both prompts apply like a YAML one. A malformed manifest is refused before anything registers."""

import contextlib
import pathlib
from collections.abc import Iterator
from http import HTTPMethod, HTTPStatus

import pydantic
import pytest
from hpcagent_agent import plugins as agent_plugins
from hpcagent_agent import prompt_blocks

from hpcagent_bench import plugins, prompt_build
from hpcagent_bench.api import RunConfig
from hpcagent_bench.harness import prompt_sections
from hpcagent_bench.harness.judge_wire import JudgeAnswer, JudgeError
from hpcagent_bench.harness.routes import ROUTES, Route, RouteRequest, next_phase
from hpcagent_bench.registry import Kind
from hpcagent_bench.skill_frontmatter import PacketBlock
from hpcagent_bench.vocabulary import PacketDef

MANIFEST = """
name = "phased"

[phases.shape]
title = "Shape"
summary = "The default shape."
text = "prompt/shape.md"
tools = ["shape_tool"]
budget = 100

[phases.build]
title = "Build"
summary = "The default build."
text = "prompt/build.md"
tools = ["build_tool"]
mode = "default"
default = { tool = "build_tool", args = { level = 2 } }

[packets.phased]
name = "Phased"
phases = ["shape", "build"]
tools = ["measure"]

[packets.phased.prompt]
about = "prompt/about.md"
off.agent = ["build", "syntax_check"]
off.task = ["build_flags"]
replace.agent = { end_to_end = "prompt/agent/end_to_end.md" }
replace.task = { delivery = "prompt/task/delivery.md" }

[packets.phased-full]
extends = "phased"
name = "Phased (all phases)"
modes = { build = "agent" }

[packets.phased-full.prompt]
off.task = ["build_flags", "optimizations"]
"""

TEXTS = {
    "prompt/shape.md": "Shape it with `shape_tool`.\n",
    "prompt/build.md": "Build it with `build_tool`.\n",
    "prompt/about.md": "## Phased\nPhased shapes the program.\n",
    "prompt/agent/end_to_end.md": "## End to end\n\n1. Shape, then build.\n{{MODE:closing}}\n",
    "prompt/task/delivery.md": "## Delivery\nThe library is `{{ lib_name }}`.\n",
}


def write_plugin(root: pathlib.Path, manifest: str = MANIFEST, texts: dict[str, str] | None = None) -> pathlib.Path:
    """A plugin folder ``root/phased`` with ``manifest`` and ``texts`` (:data:`TEXTS` by default)."""
    folder = root / "phased"
    for relative, text in (TEXTS if texts is None else texts).items():
        (folder / relative).parent.mkdir(parents=True, exist_ok=True)
        (folder / relative).write_text(text, encoding="utf-8")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "plugin.toml").write_text(manifest, encoding="utf-8")
    return folder


@pytest.fixture
def packets(monkeypatch: pytest.MonkeyPatch) -> Kind[PacketDef]:
    """An empty packet vocabulary the registrations of a test go to."""
    kind: Kind[PacketDef] = Kind("packets")
    monkeypatch.setattr(plugins, "PACKETS", kind)
    return kind


def registered(root: pathlib.Path, packets: Kind[PacketDef], manifest: str = MANIFEST) -> Kind[PacketDef]:
    plugins.register_packets(agent_plugins.find([write_plugin(root, manifest)]))
    return packets


def test_the_agent_is_served_the_tools_of_the_phases_it_runs_and_the_packets_own(
    tmp_path: pathlib.Path, packets: Kind[PacketDef]
) -> None:
    sut = registered(tmp_path, packets)

    assert sut["phased"].tools == ("shape_tool", "measure", "next_phase")
    assert sut["phased-full"].tools == ("shape_tool", "build_tool", "measure", "next_phase")


def test_the_workflow_names_every_phase_in_order_with_who_runs_it(
    tmp_path: pathlib.Path, packets: Kind[PacketDef]
) -> None:
    registered(tmp_path, packets)

    sut = plugins.compiled_overlay("plugin:phased/phased")[0]["agent"]["set"]["workflow"]["vars"]["WORKFLOW"]

    assert sut == (
        "## Workflow\n\nThe work runs in these phases, in order.\n\n### 1. Shape (you)\n\n"
        "Shape it with `shape_tool`.\n\n### 2. Build (default)\n\nThe default build.\n"
    )


def test_an_extending_packet_inherits_the_prompt_table_and_overrides_what_it_names(
    tmp_path: pathlib.Path, packets: Kind[PacketDef]
) -> None:
    registered(tmp_path, packets)

    sut = plugins.compiled_overlay("plugin:phased/phased-full")[0]

    assert sut["agent"]["set"]["build"] == {"enabled": False}, "off.agent is inherited"
    assert sut["task"]["set"]["optimizations"] == {"enabled": False}, "off.task is the extending packet's own"
    assert "### 2. Build (you)" in sut["agent"]["set"]["workflow"]["vars"]["WORKFLOW"]


def test_the_compiled_table_changes_the_agent_prompt_like_an_overlay(
    tmp_path: pathlib.Path, packets: Kind[PacketDef]
) -> None:
    registered(tmp_path, packets)
    base = prompt_blocks.parse((prompt_build.AGENT / "prompt.md").read_text(encoding="utf-8"))

    sut = prompt_blocks.apply(base, prompt_build.load_overlay("plugin:phased/phased").agent)

    blocks = {block.id: block for block in sut.blocks}
    ids = [block.id for block in sut.blocks]
    assert not blocks["build"].enabled and not blocks["syntax_check"].enabled
    assert ids.index("phased") == ids.index("intro") + 1, "about follows the intro"
    assert blocks["end_to_end"].text.startswith("## End to end\n\n1. Shape, then build.")
    assert "### 1. Shape (you)" in prompt_blocks.render(sut)


def test_the_compiled_table_changes_the_task_prompt_like_an_overlay(
    tmp_path: pathlib.Path, packets: Kind[PacketDef]
) -> None:
    registered(tmp_path, packets)

    sut = prompt_sections.resolve([prompt_build.load_overlay("plugin:phased/phased").task])

    assert not sut.on("build_flags")
    assert sut.on("phased") and sut.on("phased_workflow")
    assert "`{{ lib_name }}`" in sut.sources[prompt_sections.INLINE_PREFIX + "delivery"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            ("[packets.phased.prompt]\n", '[packets.phased.prompt]\nworkflow = "prompt/shape.md"\n'),
            "has phases and a workflow",
        ),
        (('phases = ["shape", "build"]', 'phases = ["shape", "nope"]'), r"names phases \['nope'\]"),
        (('modes = { build = "agent" }', 'modes = { nope = "agent" }'), r"sets modes of phases \['nope'\]"),
        (('extends = "phased"', 'extends = "phased-full"'), "extends itself"),
        (
            ('off.task = ["build_flags"]', 'off.task = ["build_flags", "delivery"]'),
            r"switches task blocks \['delivery'\] off and replaces them",
        ),
    ],
)
def test_a_malformed_packet_is_refused_and_adds_nothing(
    tmp_path: pathlib.Path, packets: Kind[PacketDef], change: tuple[str, str], message: str
) -> None:
    plugin = write_plugin(tmp_path, MANIFEST.replace(*change), TEXTS)

    with pytest.raises(ValueError, match=message):
        plugins.register_packets(agent_plugins.find([plugin]))
    assert not packets.entries


def test_template_syntax_in_a_text_bound_for_the_agent_prompt_is_refused(
    tmp_path: pathlib.Path, packets: Kind[PacketDef]
) -> None:
    texts = {**TEXTS, "prompt/agent/end_to_end.md": "## End to end\nThe library is `{{ lib_name }}`.\n"}
    plugin = agent_plugins.find([write_plugin(tmp_path, texts=texts)])

    with pytest.raises(ValueError, match="template syntax the agent prompt never renders"):
        plugins.register_packets(plugin)


def test_a_phase_the_agent_runs_needs_a_text(tmp_path: pathlib.Path, packets: Kind[PacketDef]) -> None:
    manifest = MANIFEST.replace('text = "prompt/build.md"\n', "")

    with pytest.raises(ValueError, match="lets the agent run phase 'build', which has no text"):
        registered(tmp_path, packets, manifest)


def test_a_phase_text_outside_the_plugin_folder_is_refused(tmp_path: pathlib.Path, packets: Kind[PacketDef]) -> None:
    manifest = MANIFEST.replace('text = "prompt/shape.md"', 'text = "../outside.md"')

    with pytest.raises(ValueError, match="phase 'shape'"):
        registered(tmp_path, packets, manifest)


def test_a_skill_page_packet_takes_no_phases() -> None:
    with pytest.raises(pydantic.ValidationError, match="phases"):
        PacketBlock.model_validate({"name": "page", "phases": ["shape"]})


class FakeJudge:
    """The part of ``service.JudgeHandler`` a route uses; keeps the last answer."""

    def __init__(self) -> None:
        self.cfg = RunConfig()
        self.code = 0
        self.answer: dict[str, object] = {}

    def reply(self, code: int, payload: JudgeAnswer) -> None:
        self.code, self.answer = code, payload.to_json()

    @contextlib.contextmanager
    def device_slot(self) -> Iterator[object]:
        yield object()


class BuildAnswer(JudgeAnswer):
    built: dict[str, object]


class PluginRoute:
    """The plugin's ``POST /phased/<tool>`` route: records each call, answers ``status``."""

    def __init__(self, status: int = HTTPStatus.OK) -> None:
        self.status = status
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, judge: FakeJudge, request: RouteRequest) -> None:
        self.calls.append((request.parts, dict(request.body)))
        if self.status != HTTPStatus.OK:
            return judge.reply(self.status, JudgeError(error="no build"))
        return judge.reply(HTTPStatus.OK, BuildAnswer(built=dict(request.body)))


def phased_run(
    root: pathlib.Path, packets: Kind[PacketDef], monkeypatch: pytest.MonkeyPatch, packet: str = "phased", **route: int
) -> PluginRoute:
    """Packet ``packet`` registered as this run's phases, the plugin route answering, and no episode remembered."""
    phases = registered(root, packets)[packet].phases
    monkeypatch.setattr(next_phase, "run_phases", lambda: phases)
    monkeypatch.setattr(next_phase, "EPISODES", next_phase.EpisodeStore(lambda episode: next_phase.Episode()))
    plugin = PluginRoute(**route)
    monkeypatch.setitem(ROUTES, "phased", Route(plugin, frozenset({HTTPMethod.POST})))
    return plugin


def step(episode: str = "episode", tokens: int = 0) -> FakeJudge:
    judge = FakeJudge()
    body = {"kernel": "gemm", "tokens": tokens}
    next_phase.serve(judge, RouteRequest(HTTPMethod.POST, ["next_phase"], {}, body, episode))
    return judge


def test_a_phase_carries_its_default_call_and_budget_to_the_judge(
    tmp_path: pathlib.Path, packets: Kind[PacketDef]
) -> None:
    sut = registered(tmp_path, packets)["phased"].phases

    assert [(phase.name, phase.mode, phase.route) for phase in sut] == [
        ("shape", "agent", "phased"),
        ("build", "default", "phased"),
    ]
    assert (sut[0].budget, sut[1].default_tool, sut[1].default_args) == (100, "build_tool", (("level", 2),))


def test_next_phase_runs_the_default_phases_after_the_agents_and_starts_a_new_round(
    tmp_path: pathlib.Path, packets: Kind[PacketDef], monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin = phased_run(tmp_path, packets, monkeypatch)

    sut = step()

    assert sut.code == HTTPStatus.OK
    assert sut.answer == {
        "phase": "shape",
        "round": 2,
        "ran": [
            {"phase": "build", "tool": "build_tool", "status": 200, "answer": {"built": {"level": 2, "kernel": "gemm"}}}
        ],
    }
    assert plugin.calls == [(["phased", "build_tool"], {"level": 2, "kernel": "gemm"})]
    assert next_phase.current("episode", "gemm") == "shape"


def test_a_failed_default_phase_answers_its_error_and_keeps_the_phase(
    tmp_path: pathlib.Path, packets: Kind[PacketDef], monkeypatch: pytest.MonkeyPatch
) -> None:
    phased_run(tmp_path, packets, monkeypatch, status=HTTPStatus.BAD_REQUEST)

    sut = step()

    assert sut.code == HTTPStatus.BAD_REQUEST
    assert "phase 'build': its default build_tool failed (no build); the kernel stays in 'shape'" in str(sut.answer)


def test_a_tool_of_a_later_agent_phase_is_refused_until_next_phase_reaches_it(
    tmp_path: pathlib.Path, packets: Kind[PacketDef], monkeypatch: pytest.MonkeyPatch
) -> None:
    phased_run(tmp_path, packets, monkeypatch, packet="phased-full")
    body: dict[str, object] = {"kernel": "gemm", "tokens": 0}

    refused = next_phase.refusal(["phased", "build_tool"], body, "episode")
    step()
    sut = next_phase.refusal(["phased", "build_tool"], body, "episode")

    assert refused == "build_tool is a tool of phase 'build' and this kernel is in phase 'shape': call next_phase"
    assert sut is None
    assert next_phase.refusal(["phased", "shape_tool"], body, "episode") is None, "an earlier phase's tool runs"


def test_a_phase_that_spent_its_token_budget_refuses_its_tools_until_the_next_round(
    tmp_path: pathlib.Path, packets: Kind[PacketDef], monkeypatch: pytest.MonkeyPatch
) -> None:
    phased_run(tmp_path, packets, monkeypatch)
    next_phase.refusal(["phased", "shape_tool"], {"kernel": "gemm", "tokens": 50}, "episode")

    within = next_phase.refusal(["phased", "shape_tool"], {"kernel": "gemm", "tokens": 150}, "episode")
    spent = next_phase.refusal(["phased", "shape_tool"], {"kernel": "gemm", "tokens": 151}, "episode")
    step(tokens=160)
    renewed = next_phase.refusal(["phased", "shape_tool"], {"kernel": "gemm", "tokens": 260}, "episode")

    assert (within, renewed) == (None, None)
    assert spent == "phase 'shape' spent 101 tokens of its budget of 100: call next_phase"


def test_a_tool_no_phase_owns_and_a_run_without_phases_are_never_gated(
    tmp_path: pathlib.Path, packets: Kind[PacketDef], monkeypatch: pytest.MonkeyPatch
) -> None:
    phased_run(tmp_path, packets, monkeypatch)
    body: dict[str, object] = {"kernel": "gemm", "tokens": 10**6}

    owned_by_none = next_phase.refusal(["phased", "measure"], body, "episode")
    monkeypatch.setattr(next_phase, "run_phases", tuple)
    unphased = next_phase.refusal(["phased", "shape_tool"], body, "episode")

    assert (owned_by_none, unphased) == (None, None)
    assert next_phase.current("episode", "gemm") == ""
    assert step().code == HTTPStatus.NOT_FOUND
