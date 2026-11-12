# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every agent harness installs at launch from a checked-in pin (``agent/harness``): the launch hook and
``harnesses.py`` agree on where, and no image recipe carries one. Static."""

import functools
import json
import os
import pathlib
import re
import sys
import tomllib

import pytest

from hpcagent_bench import paths
from tests.fresh_module import module_at

ROOT: pathlib.Path = paths.ROOT
HARNESS: pathlib.Path = ROOT / "agent" / "harness"
CE_IMAGES: pathlib.Path = ROOT / "containers" / "images"
JUDGE_AGENT_DOCKERFILES: tuple[str, ...] = (
    "judge-agent-amd/Dockerfile",
    "judge-agent-cuda/Dockerfile",
    "judge-agent-cpu/Dockerfile",
)
IMAGE_DOCKERFILES: tuple[str, ...] = tuple(
    sorted(str(path.relative_to(CE_IMAGES)) for path in CE_IMAGES.rglob("Dockerfile"))
)

#: The CLI agent_driver's claude settings (flags, tool set, compaction variables) were checked against.
CLAUDE_CODE_VERSION: str = "2.1.303"
#: Python harness venv -> the import the launch hook's gate runs: every module its runner imports.
PYTHON_HARNESSES: dict[str, str] = {
    "miniswe": (
        "import yaml, minisweagent.agents.default, minisweagent.environments.local, minisweagent.models.litellm_model"
    ),
    "optimas": (
        "import agents, agents.exceptions, agents.mcp, agents.models.openai_chatcompletions, agents.run_config, "
        "litellm, openai, openai.types.shared, optimas.arch.base, optimas.optim.opro, optimas.wrappers.example"
    ),
    "anthropic": "import anthropic, anthropic.lib.tools, anthropic.lib.tools.mcp, mcp, mcp.client.stdio",
    "openai": (
        "import agents, agents.mcp, agents.models.openai_chatcompletions, agents.run_config, openai, "
        "openai.types.shared"
    ),
}
PYPROJECT: pathlib.Path = ROOT / "agent" / "pyproject.toml"
#: Python harness names, from agent/pyproject.toml's harness-<name> dependency groups.
HARNESS_GROUP_NAMES: tuple[str, ...] = tuple(
    sorted(
        name.removeprefix("harness-")
        for name in tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["dependency-groups"]
        if name.startswith("harness-")
    )
)
#: npm packages whose CLI is a per-platform native binary package, as <prefix>-linux-<arch>.
NATIVE_BINARY_PREFIXES: tuple[str, ...] = ("@anthropic-ai/claude-code",)
#: npm cpu names of the two image architectures: judge-agent-amd x86_64, judge-agent-cuda and judge-agent-cpu
#: either.
IMAGE_NPM_ARCHES: tuple[str, ...] = ("x64", "arm64")
PINS_ENV_SHAPES: dict[str, str] = {
    "HARNESS_PYTHON": r"3\.\d+",
    "UV_VERSION": r"\d+\.\d+\.\d+",
    "UV_SHA256_X86_64": r"[0-9a-f]{64}",
    "UV_SHA256_AARCH64": r"[0-9a-f]{64}",
    "NODE_VERSION": r"\d+\.\d+\.\d+",
    "NODE_SHA256_X64": r"[0-9a-f]{64}",
    "NODE_SHA256_ARM64": r"[0-9a-f]{64}",
}

EXACT_VERSION: re.Pattern[str] = re.compile(r"\d+\.\d+\.\d+")
EXACT_NPM_SPEC: re.Pattern[str] = re.compile(r"(?:@[^/\s]+/)?[^@\s]+@\d+\.\d+\.\d+")
UNPINNED_FETCHES: tuple[re.Pattern[str], ...] = (re.compile(r"releases/latest"), re.compile(r"setup_\d+\.x"))
NPM_GLOBAL_INSTALL: re.Pattern[str] = re.compile(r"npm\s+(?:install|i)\s+(?:-g|--global)\s+([^;&|\\\n]+)")


def json_object(value: object, where: str) -> dict[str, object]:
    assert isinstance(value, dict), f"{where} is not a JSON object"
    return {str(key): member for key, member in value.items()}


def dependencies() -> dict[str, str]:
    package = json_object(json.loads((HARNESS / "node" / "package.json").read_text(encoding="utf-8")), "package.json")
    return {name: str(version) for name, version in json_object(package["dependencies"], "dependencies").items()}


def locked_packages() -> dict[str, dict[str, object]]:
    lock = json_object(json.loads((HARNESS / "node" / "package-lock.json").read_text(encoding="utf-8")), "lock")
    packages = json_object(lock["packages"], "packages")
    return {path: json_object(meta, path) for path, meta in packages.items()}


def pins_env() -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in (HARNESS / "pins.env").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            key, sep, value = line.partition("=")
            assert sep, f"pins.env line is not KEY=VALUE: {line!r}"
            pins[key] = value
    return pins


def recipe(dockerfile: str) -> str:
    return (CE_IMAGES / dockerfile).read_text(encoding="utf-8")


def unpinned_installs(text: str) -> list[str]:
    """Each fetch in a recipe that resolves to whatever upstream publishes on the day it runs."""
    found = [pattern.pattern for pattern in UNPINNED_FETCHES if pattern.search(text)]
    for match in NPM_GLOBAL_INSTALL.finditer(text):
        for token in match.group(1).split():
            spec = token.strip("\"'")
            if not spec.startswith("-") and EXACT_NPM_SPEC.fullmatch(spec) is None:
                found.append(spec)
    return found


@pytest.mark.parametrize("package", sorted(dependencies()))
def test_every_npm_cli_is_pinned_to_an_exact_version(package: str) -> None:
    """A range lets the lock move to a different CLI the next time anyone regenerates it."""
    version = dependencies()[package]
    assert EXACT_VERSION.fullmatch(version), f"{package} is pinned as {version!r}, not an exact version"


def test_claude_code_is_the_version_the_drivers_claude_settings_were_checked_against() -> None:
    assert dependencies()["@anthropic-ai/claude-code"] == CLAUDE_CODE_VERSION


def test_the_npm_lock_was_generated_from_package_json() -> None:
    """npm ci rejects a lock that disagrees with package.json, but only at build time, hours in."""
    packages = locked_packages()
    assert packages[""]["dependencies"] == dependencies(), packages[""]["dependencies"]
    locked = {name: packages[f"node_modules/{name}"]["version"] for name in dependencies()}
    assert locked == dependencies(), locked


def test_every_locked_npm_package_is_a_registry_tarball_with_its_integrity() -> None:
    floating = [
        path
        for path, meta in locked_packages().items()
        if path
        and not (
            str(meta.get("resolved", "")).startswith("https://registry.npmjs.org/")
            and str(meta.get("integrity", "")).startswith("sha512-")
        )
    ]
    assert floating == [], floating


@pytest.mark.parametrize("arch", IMAGE_NPM_ARCHES)
@pytest.mark.parametrize("prefix", NATIVE_BINARY_PREFIXES)
def test_the_npm_lock_carries_each_native_cli_binary_for_both_image_architectures(prefix: str, arch: str) -> None:
    """Both images npm ci from one lock; a lock missing the arm64 binary builds a CUDA image with no CLI."""
    meta = locked_packages().get(f"node_modules/{prefix}-linux-{arch}")
    assert meta is not None, f"package-lock.json has no {prefix}-linux-{arch}"
    assert (meta.get("os"), meta.get("cpu")) == (["linux"], [arch]), meta


@pytest.mark.parametrize("key", sorted(PINS_ENV_SHAPES))
def test_pins_env_names_an_exact_tool_version_or_checksum(key: str) -> None:
    value = pins_env().get(key, "")
    assert re.fullmatch(PINS_ENV_SHAPES[key], value), f"pins.env {key}={value!r}"


def test_every_python_harness_has_exactly_one_group() -> None:
    assert tuple(sorted(PYTHON_HARNESSES)) == HARNESS_GROUP_NAMES, HARNESS_GROUP_NAMES


@pytest.mark.parametrize("dockerfile", JUDGE_AGENT_DOCKERFILES)
def test_a_judge_agent_image_bakes_uv_and_node_and_no_harness(dockerfile: str) -> None:
    text = recipe(dockerfile)
    assert "COPY agent/harness/pins.env agent/harness/install_tools.sh" in text
    assert "sh /opt/harness/install_tools.sh /usr/local" in text
    assert "npm ci" not in text and "HARNESS_" not in text and "package-lock" not in text


@pytest.mark.parametrize("name", sorted(PYTHON_HARNESSES))
def test_the_launch_hook_gates_the_import_of_each_python_harness(name: str) -> None:
    hook = (CE_IMAGES / "lib" / "launch_venv.sh").read_text(encoding="utf-8").replace(" \\\n", " ")
    assert f"\n{name} {PYTHON_HARNESSES[name].removeprefix('import ')}\n" in hook


@pytest.mark.parametrize("dockerfile", IMAGE_DOCKERFILES)
def test_no_image_recipe_fetches_a_tool_version_chosen_on_build_day(dockerfile: str) -> None:
    assert unpinned_installs(recipe(dockerfile)) == []


@pytest.mark.parametrize(
    ("text", "flagged"),
    [
        ("curl -fsSL https://deb.nodesource.com/setup_20.x | bash -", [r"setup_\d+\.x"]),
        ("wget -qO uv.tgz https://github.com/astral-sh/uv/releases/latest/download/uv.tar.gz", ["releases/latest"]),
        ('npm install -g "@anthropic-ai/claude-code"; claude --version', ["@anthropic-ai/claude-code"]),
        ("npm i --global example-cli@latest", ["example-cli@latest"]),
        ('npm install -g "@anthropic-ai/claude-code@${v}"', ["@anthropic-ai/claude-code@${v}"]),
        ('npm install -g "@anthropic-ai/claude-code@2.1.197" example-cli@1.18.30', []),
    ],
)
def test_the_unpinned_install_scan_flags_each_floating_fetch(text: str, flagged: list[str]) -> None:
    assert unpinned_installs(text) == flagged


def module(path: pathlib.Path, name: str) -> object:
    """Load a repo script by file location; neither directory is an importable package."""
    return module_at(path, name)


@functools.lru_cache(maxsize=1)
def harnesses() -> object:
    """agent/hpcagent_agent/driver/harnesses.py, which puts its own directory on sys.path for ``import effort``."""
    return module(ROOT / "agent" / "hpcagent_agent" / "driver" / "harnesses.py", "harness_pins_harnesses")


#: The prompt layer each built-in runner's setups read: what submit.sh's HARNESS_LAYER map said before the registry
#: carried it, so no recorded setup's prompt moves.
BUILT_IN_LAYERS: dict[str, str] = {
    "miniswe": "harness-cli",
    "openhands": "harness-openhands",
    "optimas": "harness-native",
    "anthropic": "harness-native",
    "openai": "harness-native",
    "tuner": "",
}


def test_every_built_in_runner_keeps_the_prompt_layer_its_recorded_setups_read() -> None:
    assert {name: spec.layer for name, spec in harnesses().RUNNERS.items()} == BUILT_IN_LAYERS


def test_the_submitter_asks_the_drivers_registry_for_a_harness_and_its_layer() -> None:
    """One registry: a setup spec the submitter accepts and the driver rejects would die after the nodes are held."""
    text = (ROOT / "hpcagent_bench" / "cluster" / "submit.sh").read_text(encoding="utf-8")
    body = text[text.index("prompt_layers_of() {") : text.index("\n}\n", text.index("prompt_layers_of() {"))]
    assert "-m hpcagent_agent.driver.harnesses layer" in body
    assert "HARNESS_LAYER" not in text


@pytest.mark.parametrize("name", sorted(PYTHON_HARNESSES))
def test_a_setup_may_override_a_runner_interpreter_without_editing_the_driver(name: str) -> None:
    """A relocated venv is a setup env change; the fallback is the launch venv's, which the seal keeps."""
    variable = f"{name.upper()}_PYTHON"
    previous = os.environ.pop(variable, None)
    try:
        assert harnesses().interpreter(name) == f"{sys.prefix}/harness/{name}/bin/python"
        os.environ[variable] = "/tmp/other/bin/python"
        assert harnesses().interpreter(name) == "/tmp/other/bin/python"
    finally:
        os.environ.pop(variable, None)
        if previous is not None:
            os.environ[variable] = previous
