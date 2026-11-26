# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The user-plugin surface stays stable: a plugin written against one release works on the next without any edit to
this repository. Three pins: the names a plugin imports (``hpcagent_bench.plugin_api`` and the agent-side tool API),
the ``plugin.toml`` schema (additive only) and a sample plugin that installs, checks and serves a tool.
Regenerate the snapshots after a deliberate, additive change with ``python -c "from tests.test_plugin_stability import write_snapshots; write_snapshots()"``."""

import dataclasses
import enum
import importlib
import inspect
import json
import os
import pathlib
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import pydantic
from hpcagent_agent import plugins as agent_plugins

from hpcagent_bench import plugin_api

TESTS = pathlib.Path(__file__).parent
SURFACE_SNAPSHOT = TESTS / "plugin_surface.json"
MANIFEST_SNAPSHOT = TESTS / "plugin_manifest_schema.json"
SAMPLE = TESTS / "sample_plugin"
UPDATE = 'python -c "from tests.test_plugin_stability import write_snapshots; write_snapshots()"'
#: The modules a plugin imports from, and the names of each that count as the pinned surface (``None``: its ``__all__``).
PINNED: dict[str, tuple[str, ...] | None] = {
    "hpcagent_bench.plugin_api": None,
    "hpcagent_agent.tools.registry": None,
    "hpcagent_agent.tools.http_json": None,
    "hpcagent_agent.coded": None,
}
SCALARS = (str, int, float, bool, type(None))


def signature_text(value: Callable[..., object]) -> str:
    """``value``'s parameters as ``name[:annotation][=]``, a default marked but never spelled: a default's repr can
    carry the process environment or an address."""
    parts = [
        f"{parameter.name}{'' if parameter.annotation is parameter.empty else f':{parameter.annotation}'}"
        f"{'' if parameter.default is parameter.empty else '='}"
        for parameter in inspect.signature(value).parameters.values()
    ]
    return f"({','.join(parts)})"


def fingerprint(value: object) -> str:
    """A cheap spelling of what a plugin can rely on in ``value``: the kind, and the signature, fields or members."""
    if inspect.isclass(value):
        if issubclass(value, pydantic.BaseModel):
            return "model:" + ",".join(value.model_fields)
        if dataclasses.is_dataclass(value):
            return "dataclass:" + ",".join(field.name for field in dataclasses.fields(value))
        if issubclass(value, enum.Enum):
            return "enum:" + ",".join(member.name for member in value)
        try:
            return f"class{signature_text(value)}"
        except (ValueError, TypeError):
            return "class"
    if callable(value):
        try:
            return f"callable{signature_text(value)}"
        except (ValueError, TypeError):
            return "callable"
    if isinstance(value, SCALARS):
        return f"const:{value!r}"
    if isinstance(value, tuple) and all(isinstance(item, SCALARS) for item in value):
        return f"const:{value!r}"
    return f"value:{type(value).__name__}"


def surface() -> dict[str, dict[str, str]]:
    """Every pinned module's exported names, fingerprinted."""
    snapshot: dict[str, dict[str, str]] = {}
    for module_name, names in PINNED.items():
        namespace = vars(importlib.import_module(module_name))
        exported = names if names is not None else tuple(namespace["__all__"])
        snapshot[module_name] = {name: fingerprint(namespace[name]) for name in sorted(exported)}
    return {"api_version": {"plugin_api": str(plugin_api.API_VERSION)}, **snapshot}


def surface_problems(old: dict[str, dict[str, str]], new: dict[str, dict[str, str]]) -> list[str]:
    """What moved between two surface snapshots: a removed or changed name is breaking, an added one needs a regen."""
    problems: list[str] = []
    for module_name in sorted(old.keys() | new.keys()):
        before, after = old.get(module_name, {}), new.get(module_name, {})
        problems += [f"{module_name}.{name}: removed" for name in sorted(before.keys() - after.keys())]
        problems += [
            f"{module_name}.{name}: {before[name]} -> {after[name]}"
            for name in sorted(before.keys() & after.keys())
            if before[name] != after[name]
        ]
        problems += [f"{module_name}.{name}: added" for name in sorted(after.keys() - before.keys())]
    return problems


def resolve(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """``node`` with a ``$ref`` followed into ``defs``."""
    while "$ref" in node:
        node = defs[node["$ref"].rpartition("/")[2]]
    return node


def alternatives(node: dict[str, Any], defs: dict[str, Any]) -> list[dict[str, Any]]:
    """The alternatives of a node: its ``anyOf`` members, or itself."""
    node = resolve(node, defs)
    return [resolve(item, defs) for item in node["anyOf"]] if "anyOf" in node else [node]


def accepted_values(node: dict[str, Any]) -> list[Any] | None:
    """The finite set a node allows (``const`` / ``enum``), else ``None``."""
    if "const" in node:
        return [node["const"]]
    return node.get("enum")


def narrowing(
    old: dict[str, Any],
    new: dict[str, Any],
    old_defs: dict[str, Any],
    new_defs: dict[str, Any],
    where: str,
    seen: set[tuple[str, str]],
) -> list[str]:
    """Why ``new`` accepts less than ``old`` at ``where`` (empty when everything ``old`` accepted is still accepted)."""
    key = (json.dumps(old, sort_keys=True), json.dumps(new, sort_keys=True))
    if key in seen:
        return []
    seen.add(key)
    problems: list[str] = []
    new_options = alternatives(new, new_defs)
    for old_option in alternatives(old, old_defs):
        attempts = [narrowing_of_option(old_option, option, old_defs, new_defs, where, seen) for option in new_options]
        if not any(not attempt for attempt in attempts):
            problems += (
                attempts[0] if len(attempts) == 1 else [f"{where}: no alternative of the new type accepts {old_option}"]
            )
    return problems


def narrowing_of_option(
    old: dict[str, Any],
    new: dict[str, Any],
    old_defs: dict[str, Any],
    new_defs: dict[str, Any],
    where: str,
    seen: set[tuple[str, str]],
) -> list[str]:
    """:func:`narrowing` for two single (non-union) schema nodes."""
    problems: list[str] = []
    old_values, new_values = accepted_values(old), accepted_values(new)
    if new_values is not None and (
        old_values is None or not set(map(json.dumps, old_values)) <= set(map(json.dumps, new_values))
    ):
        problems.append(f"{where}: allowed values narrowed to {new_values}")
    if "type" in new and new.get("type") != old.get("type") and old_values is None:
        problems.append(f"{where}: type {old.get('type')} -> {new['type']}")
    if new.get("pattern") not in (None, old.get("pattern")):
        problems.append(f"{where}: pattern now {new['pattern']}")
    old_props, new_props = old.get("properties", {}), new.get("properties", {})
    for name, old_prop in old_props.items():
        if name not in new_props:
            problems.append(f"{where}.{name}: removed")
        else:
            problems += narrowing(old_prop, new_props[name], old_defs, new_defs, f"{where}.{name}", seen)
    for name in sorted(set(new.get("required", [])) - set(old.get("required", []))):
        problems.append(f"{where}.{name}: now required")
    if "items" in old and "items" in new:
        problems += narrowing(old["items"], new["items"], old_defs, new_defs, f"{where}[]", seen)
    old_extra, new_extra = old.get("additionalProperties"), new.get("additionalProperties")
    if isinstance(old_extra, dict) and isinstance(new_extra, dict):
        problems += narrowing(old_extra, new_extra, old_defs, new_defs, f"{where}{{}}", seen)
    elif new_extra is False and old_extra not in (False, None):
        problems.append(f"{where}: additional properties no longer accepted")
    return problems


def api_values(schema: dict[str, Any]) -> list[Any]:
    """The versions ``Manifest.api`` accepts."""
    node = schema["properties"]["api"]
    return sorted(
        value for option in alternatives(node, schema.get("$defs", {})) for value in accepted_values(option) or []
    )


def manifest_problems(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Where the new manifest schema rejects a ``plugin.toml`` the old one accepted; none when ``api`` moved."""
    if api_values(old) != api_values(new):
        return []
    return narrowing(old, new, old.get("$defs", {}), new.get("$defs", {}), "Manifest", set())


def manifest_schema() -> dict[str, Any]:
    """The current manifest schema, as plain JSON."""
    return json.loads(json.dumps(agent_plugins.Manifest.model_json_schema()))


def write_snapshots() -> None:
    """Regenerate both snapshots from the code."""
    SURFACE_SNAPSHOT.write_text(json.dumps(surface(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    MANIFEST_SNAPSHOT.write_text(json.dumps(manifest_schema(), indent=1, sort_keys=True) + "\n", encoding="utf-8")


def test_the_names_a_plugin_imports_keep_their_signatures() -> None:
    """Every exported name of the plugin API and the agent-side tool API is still there with the same signature;
    removing or changing one fails unless ``API_VERSION`` moved, and a new name needs a regenerated snapshot."""
    old = json.loads(SURFACE_SNAPSHOT.read_text(encoding="utf-8"))
    new = surface()
    problems = surface_problems(old, new)
    version_moved = old["api_version"] != new["api_version"]
    breaking = [problem for problem in problems if not problem.endswith(": added")]
    assert not (breaking and not version_moved), "user-plugin surface broke without an API_VERSION bump:\n" + "\n".join(
        breaking
    )
    assert not problems, f"the plugin surface changed ({'; '.join(problems)}); regenerate with: {UPDATE}"


def test_the_manifest_schema_only_grows() -> None:
    """A ``plugin.toml`` the pinned schema accepted is still accepted: new optional fields pass (after a regenerated
    snapshot), a removed or renamed field, a newly required one or a narrowed type fails unless ``api`` moved."""
    old = json.loads(MANIFEST_SNAPSHOT.read_text(encoding="utf-8"))
    new = manifest_schema()
    problems = manifest_problems(old, new)
    assert not problems, "plugin.toml schema narrowed:\n" + "\n".join(problems)
    assert old == new, f"the manifest schema grew; regenerate with: {UPDATE}"


def test_the_manifest_comparison_tells_growth_from_breakage() -> None:
    """The structural comparison accepts an added optional field and rejects a removed field, a newly required one
    and a narrowed type, so the pin above cannot pass by accident."""
    old = manifest_schema()
    grown = json.loads(json.dumps(old))
    grown["properties"]["extra_field"] = {"type": "string", "default": ""}
    removed = json.loads(json.dumps(old))
    del removed["properties"]["description"]
    required = json.loads(json.dumps(old))
    required["required"] = [*required.get("required", []), "description"]
    narrowed = json.loads(json.dumps(old))
    narrowed["properties"]["openmp"] = {"const": "libgomp"}

    assert manifest_problems(old, grown) == []
    assert manifest_problems(old, removed)
    assert manifest_problems(old, required)
    assert manifest_problems(old, narrowed)


def run_module(args: list[str], folder: pathlib.Path, cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    """``python <args>`` as a plugin repository's CI runs it, with only ``folder`` exported as a user plugin."""
    env = {**os.environ, agent_plugins.USER_PLUGIN_DIRS_ENV: str(folder)}
    return subprocess.run(
        [sys.executable, *args], env=env, cwd=cwd, capture_output=True, text=True, check=False, timeout=300
    )


def test_a_sample_plugin_installs_checks_and_serves_its_tool(tmp_path: pathlib.Path) -> None:
    """A plugin folder that lives outside this repository passes ``plugins install`` and ``plugins check`` and its
    agent tool is served to a setup whose packet lists it, and answers a call. No stub episode: the sample ships no
    tuner, and the episode path (``tests/test_user_plugin_episodes.py``) needs a plugin's ``[ci]`` block. Its own MCP
    server (``[mcp.sample]``) is exported by its packet and answers a tool call over stdio, launched as the agent's
    mcp.json names it."""
    for command in ("install", "check"):
        done = run_module(["-m", "hpcagent_bench.plugins", command], SAMPLE, tmp_path)
        assert done.returncode == 0, done.stdout + done.stderr
    assert "plugins are fine" in done.stdout, done.stdout
    script = (
        "import os\n"
        "from hpcagent_agent import plugins\n"
        "from hpcagent_agent.tools import registry\n"
        "plugins.import_tools(os.environ)\n"
        "served = registry.served(registry.Setup.read(os.environ))\n"
        "print(served['sample_echo'].run({'text': 'hi'}))\n"
    )
    env_packet = {registry_env: "sample_echo" for registry_env in ("HPCAGENT_BENCH_PACKET_TOOLS",)}
    done = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, agent_plugins.USER_PLUGIN_DIRS_ENV: str(SAMPLE), **env_packet},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert done.returncode == 0 and "'echo': 'hi'" in done.stdout, done.stdout + done.stderr
    done = subprocess.run(
        [sys.executable, "-c", SERVE_PLUGIN_MCP],
        env={**os.environ, agent_plugins.USER_PLUGIN_DIRS_ENV: str(SAMPLE)},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert done.returncode == 0 and '{"shout": "HI"}' in done.stdout, done.stdout + done.stderr


#: Resolves the sample packet, checks it exports its MCP server, launches the server the way the agent's mcp.json
#: names it, and calls its tool over stdio.
SERVE_PLUGIN_MCP = """
import sys
import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from hpcagent_agent import plugins
from hpcagent_bench import packets
from hpcagent_bench import plugins as bench_plugins
bench_plugins.register()
packet = packets.resolve("sample", "cpp")
assert packet.mcp == ("sample",), packet
assert f"{plugins.PACKET_MCP_ENV}=sample" in packets.packet_lines(packet, None, None)
entry = plugins.mcp_launches(packet.mcp, sys.executable)["sample"]

async def main():
    params = StdioServerParameters(command=entry["command"], args=entry["args"], env=entry["env"] or None)
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        print((await session.call_tool("shout", {"text": "hi"})).content[0].text)

anyio.run(main)
"""
