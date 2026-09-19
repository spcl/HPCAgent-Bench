# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Full-registry isolation matrix: every registered packet key (the "" control included), checked
against every OTHER surface an agent can read a gated capability off -- the MCP tool list, the
task-text announcement, and the shared task material -- not just the two or three arms a targeted
test already pins.

The 2026-09-15 leak (``canonical_parallel_form`` served to every arm, not only ``cpf``'s) was fixed
and pinned for THREE arms (bare, ``lang-skills``, ``cpf``) in tests/test_packet_wiring.py. That
leaves the other 19 registered keys unchecked on the same surface: a packet added later, or a packet
whose own env happens to collide with ``PACKET_TOOL_SWITCH``'s value, has no test that would catch
it. This file parametrizes over :func:`hpcagent_bench.experiment_tags.registry`'s ``packet_defs``
directly, so a new registry entry is covered the day it is added, with no matching edit here.

Every assertion goes through the real rendering function it is checking (``packets.resolve``, the
live ``mcp_server.py`` process, ``make_problems.packet_note``, ``materialize_shared.sh``) -- none of
it is reimplemented, only cross-checked against :func:`hpcagent_bench.packets.reached_keys`, which is
itself the module's own oracle for "what does this spec compose".
"""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
from types import ModuleType

import pytest

from hpcagent_bench import cpf_cache, experiment_tags as tags, packets

REPO = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"
MCP_SERVER = REPO / "containers" / "agent" / "tools" / "mcp_server.py"
MATERIALIZE = EXPERIMENTS / "materialize_shared.sh"
KERNEL = "loop_level_reasoning/argmax_value/argmax_value"

CPF_TOOL_SWITCH = "HPCAGENT_BENCH_SERVICE_CANONICAL_PARALLEL_FORM_DIR"
CORE_TOOLS = {"score", "submit", "profile", "syntax_check"}

DEFINITIONS = tags.registry().packet_defs
REGISTERED_KEYS = tuple(sorted(DEFINITIONS))


def effective_device(key: str) -> str:
    """``key``'s own ``device:``, or the one a packet it COMPOSES sets (``all-in-amd`` names no
    device of its own; ``perf-playbook-amd``, which it reaches through ``packets:``, does)."""
    return next(
        (DEFINITIONS[sub].device for sub in packets.reached_keys(key, DEFINITIONS) if DEFINITIONS[sub].device), ""
    )


#: One (language, image) pair every registered key can resolve under -- the language its EFFECTIVE
#: device runs (:mod:`hpcagent_bench.packets`.DEVICE_LANGUAGES, read through composition), "c" for
#: every device-neutral key. Read off the registry rather than hand-picked, so a new ``device:``
#: packet fails loudly here instead of silently picking the wrong language below.
ARM_FOR_KEY: dict[str, tuple[str, str | None]] = {
    key: {"amd": ("hip", "amd"), "nvidia": ("cuda", "nvidia")}.get(effective_device(key), ("c", None))
    for key in DEFINITIONS
}

#: cpf_cache.DIALECT names c, c++/cpp and hip; a CPF-composing key resolved for any other language
#: (only all-in-nvidia, pinned to cuda by its device) cannot render a drop-in at all and every
#: consumer of it (packet_note, materialize_shared.sh) must refuse rather than serve nothing.
DIALECT_LANGUAGES = frozenset({"c", "cpp", "hip"})


def resolved(key: str) -> packets.Packet:
    language, image = ARM_FOR_KEY[key]
    return packets.resolve(
        key, language, environ={"CPF_VIEW": "/views/dummy", "REPO_LAYOUT_PYTHON": sys.executable}, image=image
    )


def reaches(key: str, target: str) -> bool:
    return target in packets.reached_keys(key, DEFINITIONS)


# ---------------------------------------------------------------------------------------------
# A: the resolved env itself -- the ground truth every other surface below reads its gate from.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", REGISTERED_KEYS)
def test_the_cpfsrc_dropin_switch_is_set_only_by_a_spec_that_reaches_cpfsrc(key: str) -> None:
    env = dict(resolved(key).env)
    assert ("CPF_DROPIN_DIR" in env) == reaches(key, "cpfsrc"), (key, env)


@pytest.mark.parametrize("key", REGISTERED_KEYS)
def test_the_cpf_tool_switch_is_set_only_by_a_spec_that_reaches_cpf(key: str) -> None:
    env = dict(resolved(key).env)
    assert (CPF_TOOL_SWITCH in env) == reaches(key, "cpf"), (key, env)


@pytest.mark.parametrize("key", REGISTERED_KEYS)
def test_a_tool_manual_page_is_staged_only_by_a_spec_that_reaches_a_packet_declaring_that_tool(key: str) -> None:
    """Generalizes the single cpf/lang-skills/bare check in test_packet_wiring.py to all 22 keys:
    the manual for a gated MCP tool must never ride on a packet that does not carry the tool."""
    gated = packets.tool_pages()
    owners = {sub for sub in packets.reached_keys(key, DEFINITIONS) if DEFINITIONS[sub].tools}
    expected = {page for owner in owners for page in DEFINITIONS[owner].skills} & gated
    assert set(resolved(key).pages) & gated == expected, key


# ---------------------------------------------------------------------------------------------
# B: the MCP surface -- a fresh server process per key, the same way the container spawns one.
# ---------------------------------------------------------------------------------------------


def tool_names(env: dict[str, str]) -> set[str]:
    """A fresh ``tools/list`` reply from ``mcp_server.py`` under exactly ``env`` layered on a base
    that strips every packet/tool switch the current process might carry, so a developer's local
    export can never leak into what a "clean" arm is believed to serve."""
    stripped = {"AGENT_PACKET", "AGENT_SCORE_TOOL", CPF_TOOL_SWITCH, "AGENT_SEARCH_TOOL"}
    base = {k: v for k, v in os.environ.items() if k not in stripped}
    request = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n"
    result = subprocess.run(
        [sys.executable, str(MCP_SERVER)],
        input=request,
        env={**base, "PYTHONSAFEPATH": "1", **env},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    answer = json.loads(result.stdout.splitlines()[0])
    return {tool["name"] for tool in answer["result"]["tools"]}


def method_tool_stems(agent_packet: str) -> set[str]:
    """The extra tool stems ``AGENT_PACKET=<agent_packet>`` adds: every ``*.py`` under its
    ``containers/agent/packets/<name>/`` directory, the same glob ``mcp_server.py`` runs."""
    directory = REPO / "containers" / "agent" / "packets" / agent_packet
    return {p.stem for p in directory.glob("*.py")} if directory.is_dir() else set()


@pytest.mark.parametrize("key", REGISTERED_KEYS)
def test_the_mcp_tool_list_matches_exactly_what_this_key_declares(key: str) -> None:
    """One arm, one packet: the served ``tools/list`` must be the core set (minus ``score`` under
    ``AGENT_SCORE_TOOL=0``), plus a declared MCP tool, plus a method packet's own modules -- never
    a tool belonging to a DIFFERENT registered key."""
    env = dict(resolved(key).env)
    served = tool_names(env)
    expected_core = CORE_TOOLS - {"score"} if env.get("AGENT_SCORE_TOOL") == "0" else CORE_TOOLS
    declared = {tool for sub in packets.reached_keys(key, DEFINITIONS) for tool in DEFINITIONS[sub].tools}
    method_tools = method_tool_stems(env["AGENT_PACKET"]) if "AGENT_PACKET" in env else set()
    assert served == expected_core | declared | method_tools, (key, served)


def test_no_registered_key_other_than_cpf_ever_serves_the_canonical_parallel_form_tool() -> None:
    """The exact 2026-09-15 leak, restated as a universal negative: every OTHER key's own env,
    resolved for real, must never make the judge's cpf route answer anything but absent."""
    leaking = [
        key
        for key in REGISTERED_KEYS
        if not reaches(key, "cpf") and "canonical_parallel_form" in tool_names(dict(resolved(key).env))
    ]
    assert not leaking, leaking


# ---------------------------------------------------------------------------------------------
# C: the task text -- make_problems.py's CPFSRC_NOTE, gated on the same CPF_DROPIN_DIR switch.
# ---------------------------------------------------------------------------------------------


def load_make_problems() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_problems_isolation_matrix", EXPERIMENTS / "make_problems.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_problems = load_make_problems()


@pytest.mark.parametrize("key", REGISTERED_KEYS)
def test_the_task_text_announces_the_cpf_dropin_only_for_a_spec_that_reaches_cpfsrc(key: str) -> None:
    """Generalizes test_cpfsrc_carries_the_note_into_every_packet_that_composes_it (3 keys) to all
    22: a control or an unrelated packet must never read a task claiming a file it was not given,
    and a cpfsrc-composing key in a dialect the CPF renderer cannot serve must refuse outright
    rather than silently rendering a task with no note for a file materialize_shared.sh will try
    (and fail) to stage."""
    language, _ = ARM_FOR_KEY[key]
    if reaches(key, "cpfsrc") and language not in DIALECT_LANGUAGES:
        with pytest.raises(ValueError, match="not for"):
            make_problems.packet_note(key, language)
        return
    note = make_problems.packet_note(key, language)
    assert ("DROP-IN" in note) == reaches(key, "cpfsrc"), (key, note)


# ---------------------------------------------------------------------------------------------
# D: the shared task folder -- materialize_shared.sh actually staging (or not staging) the file,
# spot-checked across the reaches/does-not-reach split so A-C's env-level gate is proven to reach
# disk, not just asserted about the env that is supposed to drive it.
# ---------------------------------------------------------------------------------------------


@pytest.fixture(name="repo")
def repo_fixture(tmp_path):
    kernel_dir = tmp_path / "hpcagent_bench/benchmarks/loop_level_reasoning/argmax_value"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "argmax_value_numpy.py").write_text("def argmax_value(a): return a.max()\n")
    (kernel_dir / "argmax_value.yaml").write_text("benchmark: {}\n")
    prompt = tmp_path / "containers/agent"
    prompt.mkdir(parents=True)
    (prompt / "prompt.md").write_text("base rules\n{{HINTS}}\n\nTask:\n\n{{TASK}}\n")
    return tmp_path


def problems_file(path: pathlib.Path, kernels: list[str]) -> pathlib.Path:
    path.write_text("".join(json.dumps({"id": i, "kernel": k, "task": "opt"}) + "\n" for i, k in enumerate(kernels)))
    return path


def materialize_arm(repo, shared, problems, **arm: str) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if key not in ("CPF_DROPIN_DIR", "AGENT_LANGUAGE")}
    env.update(
        PYTHONPATH=f"{REPO}:{REPO / 'hpcagent_bench' / 'numpy_translators' / 'src'}",
        REPO_LAYOUT_PYTHON=sys.executable,
        **arm,
    )
    return subprocess.run(
        [str(MATERIALIZE), str(repo), str(shared), str(problems)], capture_output=True, text=True, env=env
    )


#: Spot-check across the full split rather than all 22 (a dialect-rendered CPF view per language
#: is real filesystem setup, not a free parametrize row): one key that reaches cpfsrc through a
#: direct skill (cpfsrc itself), one that reaches it through composition (all-in-cpu), and two that
#: never touch it at all -- one with no env of its own (autokernel) and one that sets a DIFFERENT
#: env switch entirely (no-score-tool), so "no env" and "some other env" both prove out.
STAGES_DROPIN = ("cpfsrc", "all-in-cpu")
STAGES_NOTHING = ("autokernel", "no-score-tool", "")


@pytest.mark.parametrize("key", STAGES_DROPIN)
def test_the_shared_task_folder_carries_the_dropin_for_a_cpfsrc_composing_key(tmp_path, repo, key: str) -> None:
    from tests.test_cpf_cache import view_with

    view = view_with(tmp_path, "argmax_value")
    shared = tmp_path / "shared"
    env = dict(resolved(key).env)
    materialize_arm(
        repo,
        shared,
        problems_file(tmp_path / "problems.jsonl", [KERNEL]),
        CPF_DROPIN_DIR=str(view),
        **{k: v for k, v in env.items() if k != "CPF_DROPIN_DIR"},
    )
    staged = sorted(p.name for p in (shared / "tasks/argmax_value").iterdir())
    assert "argmax_value.c" in staged, (key, staged)


@pytest.mark.parametrize("key", STAGES_NOTHING)
def test_the_shared_task_folder_carries_no_dropin_for_a_key_that_does_not_reach_cpfsrc(
    tmp_path, repo, key: str
) -> None:
    from tests.test_cpf_cache import view_with

    view_with(tmp_path, "argmax_value")  # a view exists on disk; a non-cpfsrc arm must not find it
    shared = tmp_path / "shared"
    env = dict(resolved(key).env)
    assert "CPF_DROPIN_DIR" not in env
    materialize_arm(repo, shared, problems_file(tmp_path / "problems.jsonl", [KERNEL]), **env)
    staged = sorted(p.name for p in (shared / "tasks/argmax_value").iterdir())
    assert not [name for name in staged if name.split(".", 1)[0] == "argmax_value"], (key, staged)


def test_every_registered_key_is_covered_by_the_reaches_split_above() -> None:
    """The spot-check lists above are hand-picked FOR the split; this pins the split itself so a
    future packet cannot silently join a class it was never assigned to."""
    reaches_cpfsrc = {key for key in REGISTERED_KEYS if reaches(key, "cpfsrc")}
    assert set(STAGES_DROPIN) <= reaches_cpfsrc
    assert set(STAGES_NOTHING) <= (set(REGISTERED_KEYS) - reaches_cpfsrc)
