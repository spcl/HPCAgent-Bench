# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A plugin packet's prompt table (``plugin.toml``) shapes the prompts of the setups that run it: the task prompt's
sections are turned off or replaced, its ``about`` text follows the intro of both prompts, and the user's own overlays
still win. The agent prompt takes it through the ``prompt.md`` the launcher stages."""

import pathlib
from collections.abc import Iterator

import pytest
from hpcagent_agent import plugins as agent_plugins
from hpcagent_agent import prompt_blocks

from hpcagent_bench import config, packets, plugins, prompt_build
from hpcagent_bench import study_tags as tags
from hpcagent_bench.categories import Language, SourceMode
from hpcagent_bench.harness import prompts
from hpcagent_bench.harness.prompts import PromptConfig, build_prompt
from hpcagent_bench.harness.task import Task
from hpcagent_bench.vocabulary import PACKETS
from tests.fresh_module import fresh
from tests.problem_facts import rendered

TASK = Task("gemm", SourceMode.RESTRICTED, Language.C)

#: The texts every probe plugin ships under ``prompt/``.
TEXTS = {
    "timing.md": "## Timing\nThe plugin times {{ kernel }} itself.\n",
    "timing_two.md": "## Timing\nThe second plugin times {{ kernel }}.\n",
    "about.md": "## Probe\nThe plugin also counts the kernel's calls.\n",
    "keep.md": "## Keep going\nApply one more move.\n",
}


def registered_plugin(root: pathlib.Path, name: str, table: str) -> pathlib.Path:
    """Register a plugin whose packet ``name`` has the prompt keys ``table`` (TOML lines of ``[packets.<name>]``) and
    ships :data:`TEXTS` under ``prompt/``."""
    folder = root / name
    (folder / "prompt").mkdir(parents=True)
    for file, text in TEXTS.items():
        (folder / "prompt" / file).write_text(text, encoding="utf-8")
    (folder / "plugin.toml").write_text(
        f'name = "{name}"\n\n[packets.{name}]\nname = "{name}"\n{table}\n',
        encoding="utf-8",
    )
    plugins.register_packets(agent_plugins.find([folder]))
    tags.registry.cache_clear()
    # Each test registers its own table under the same reference, which keys the task prompt's cache.
    prompts.task_blocks.cache_clear()
    return folder


#: A packet that replaces the task prompt's timing and drops its scoring.
TIMING_OFF_SCORING = 'replace = { timing = "prompt/timing.md" }\noff = ["scoring"]'
#: A packet that says what it does after the intro.
ABOUT = 'about = "prompt/about.md"'


@pytest.fixture
def plugin_root(tmp_path: pathlib.Path) -> Iterator[pathlib.Path]:
    """A folder for plugins registered into the packet vocabulary, which is left as it was found."""
    tags.registry()  # the shipped and installed plugins register first, once
    entries, orders = dict(PACKETS.entries), dict(PACKETS.orders)
    yield tmp_path
    PACKETS.entries.clear()
    PACKETS.entries.update(entries)
    PACKETS.orders.clear()
    PACKETS.orders.update(orders)
    tags.registry.cache_clear()


def test_a_packet_replaces_and_turns_off_sections_only_in_its_own_setups(plugin_root: pathlib.Path) -> None:
    registered_plugin(plugin_root, "promptprobe", TIMING_OFF_SCORING)
    without = build_prompt(TASK)

    with config.overridden("record.packet", "promptprobe"):
        sut = build_prompt(TASK)

    assert "The plugin times gemm itself." in sut
    assert "## Scoring" not in sut
    assert "The plugin times gemm itself." not in without
    assert "## Scoring" in without


def test_the_users_own_overlays_win_over_the_packet(plugin_root: pathlib.Path, tmp_path: pathlib.Path) -> None:
    registered_plugin(plugin_root, "promptprobe", TIMING_OFF_SCORING)
    mine = tmp_path / "mine.yaml"
    mine.write_text("task:\n  set:\n    scoring: {enabled: true}\n", encoding="utf-8")

    with config.overridden("prompt.overlays", (str(mine),)), config.overridden("record.packet", "promptprobe"):
        sut = build_prompt(TASK)

    assert "## Scoring" in sut
    assert "The plugin times gemm itself." in sut


def test_two_packets_apply_their_tables_in_spec_order(plugin_root: pathlib.Path) -> None:
    registered_plugin(plugin_root, "promptone", 'replace = { timing = "prompt/timing.md" }')
    registered_plugin(plugin_root, "prompttwo", 'replace = { timing = "prompt/timing_two.md" }')

    with config.overridden("record.packet", "promptone+prompttwo"):
        sut = build_prompt(TASK)

    assert "The second plugin times gemm." in sut
    assert "The plugin times gemm itself." not in sut


def test_a_gpu_setup_takes_the_device_form_of_its_packet_in_both_prompts(
    plugin_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A device-language setup applies the packet's GPU form (packets.by_device) to the task prompt, the same layers
    the agent prompt takes (prompt_build.env_layers)."""
    registered_plugin(plugin_root, "promptprobe", TIMING_OFF_SCORING)
    registered_plugin(plugin_root, "promptprobegpu", ABOUT)
    monkeypatch.setattr(packets, "GPU_ALIASES", {"promptprobe": "promptprobegpu"})
    environ = {"HPCAGENT_BENCH_RECORD_PACKET": "promptprobe", "LANGUAGE": "hip"}

    with config.overridden("record.packet", "promptprobe"), config.overridden("record.language", "hip"):
        sut = build_prompt(TASK)
        overlays = PromptConfig.resolve().overlays
        agent_packet = prompt_build.env_layers(environ)[1]

    assert overlays == packets.prompt_layers(agent_packet) == packets.prompt_layers("promptprobegpu")
    assert "The plugin also counts the kernel's calls." in sut
    assert "The plugin times gemm itself." not in sut


def test_a_packets_about_text_follows_the_intro_only_in_its_own_setups(plugin_root: pathlib.Path) -> None:
    registered_plugin(plugin_root, "promptprobe", ABOUT)
    without = build_prompt(TASK)

    with config.overridden("record.packet", "promptprobe"):
        sut = build_prompt(TASK)

    assert sut.index("The plugin also counts the kernel's calls.") < sut.index("## Timing")
    assert "The plugin also counts" not in without


def test_a_packet_reshapes_the_cluster_prompt_of_its_setups(
    plugin_root: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered_plugin(
        plugin_root,
        "promptprobe",
        'off = ["build"]\nreplace = { keep_improving = "prompt/keep.md" }\nabout = "prompt/about.md"',
    )
    material = tmp_path / "material"
    material.mkdir()
    (material / "prompt.md").write_text(
        prompt_blocks.resolved(prompt_build.resolve_layers([], "promptprobe")), encoding="utf-8"
    )
    driver = fresh("agent_driver")
    monkeypatch.setenv(driver.MATERIAL_DIR_ENV, str(material))

    sut = rendered(driver, "gemm")

    assert "The `compiler` field" not in sut
    assert "## Keep going\nApply one more move." in sut
    assert "Parallelize first" not in sut
    assert sut.index("The plugin also counts the kernel's calls.") < sut.index("## End to end")


def test_the_cluster_prompt_without_a_packet_keeps_every_section_and_no_section_line() -> None:
    sut = rendered(fresh("agent_driver"), "gemm")

    assert "@@section" not in sut
    assert "The `compiler` field" in sut
    assert "Parallelize first" in sut


def test_a_block_no_prompt_has_is_refused(plugin_root: pathlib.Path) -> None:
    with pytest.raises(ValueError, match=r"names blocks \['nosuch'\] that no prompt has"):
        registered_plugin(plugin_root, "promptprobe", 'off = ["nosuch"]')


def test_the_users_own_inserted_text_follows_the_packets(plugin_root: pathlib.Path, tmp_path: pathlib.Path) -> None:
    registered_plugin(plugin_root, "promptprobe", ABOUT)
    mine = tmp_path / "mine.yaml"
    mine.write_text(
        'task:\n  insert:\n    - {after: promptprobe, block: {id: own, text: "The user reads {{ kernel }}\'s log.\\n\\n"}}\n',
        encoding="utf-8",
    )

    with config.overridden("prompt.overlays", (str(mine),)), config.overridden("record.packet", "promptprobe"):
        sut = build_prompt(TASK)

    assert sut.index("The plugin also counts the kernel's calls.") < sut.index("The user reads gemm's log.")


def test_a_plugin_packets_skill_page_is_indexed_in_the_task_prompt(
    plugin_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plugin ships its page in its own folder, which is on no template root; the task prompt still finds it."""
    monkeypatch.setenv("HPCAGENT_BENCH_USER_PLUGIN_DIRS", str(plugin_root / "promptprobe"))
    page = plugin_root / "promptprobe" / "skills" / "promptprobe" / "SKILL.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "---\nname: promptprobe\ndescription: Probe page of the plugin.\nwhen: you drive the probe.\n---\n\nBody.\n",
        encoding="utf-8",
    )
    registered_plugin(plugin_root, "promptprobe", ABOUT)

    with config.overridden("record.packet", "promptprobe"):
        sut = build_prompt(TASK)

    assert "promptprobe" in sut
    assert "you drive the probe" in sut
