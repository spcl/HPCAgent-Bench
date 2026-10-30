# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The one packet format: a ``[packets.<key>]`` table (:class:`PacketTable`), shipped in :data:`SHIPPED` and declared by
a plugin in its ``plugin.toml``. The shipped tables register here in file order (each packet's slot); a plugin's after
them (:mod:`hpcagent_bench.plugins`). A recorded key's definition is immutable: a changed meaning is a NEW key."""

import pathlib
import tomllib
from collections.abc import Collection, Mapping
from typing import Any

import pydantic

from hpcagent_bench import paths
from hpcagent_bench.vocabulary import PACKETS, PacketDef, PacketDevice, PacketPhase

__all__ = ["SHIPPED", "SHIPPED_KEYS", "PacketTable", "definition", "merged_table", "register_shipped", "shipped_pages"]

#: The shipped packets' tables.
SHIPPED = paths.PACKAGE / "packets.toml"


class PacketTable(pydantic.BaseModel):
    """One ``[packets.<key>]`` table (``docs/extending/packets.md``). ``skills`` default to the page named like the key
    when its source ships one; ``overlays`` are prompt overlay references (``docs/prompts.md``); ``extends`` names a
    table of the same file whose fields this one starts from; ``agent`` names the plugin phases the agent runs."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    name: str = pydantic.Field(min_length=1)
    aliases: tuple[str, ...] = ()
    extends: str = ""
    skills: tuple[str, ...] = ()
    packets: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    agents: tuple[str, ...] = ()
    env: dict[str, str] = pydantic.Field(default_factory=dict[str, str])
    overlays: tuple[str, ...] = ()
    agent: tuple[str, ...] = ()
    device: PacketDevice | None = None
    frozen: str = ""
    marker: str = ""
    short: str = ""


def merged_table(
    tables: Mapping[str, Mapping[str, Any]], key: str, where: str, chain: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Table ``key`` of ``tables`` with the fields of the one it ``extends`` under it: tables merge key by key, any
    other value replaces; ``aliases`` are never inherited, being names of one packet.

    :raises ValueError: ``extends`` names no table of ``tables``, or the chain loops.
    """
    raw = dict(tables[key])
    base = raw.get("extends", "")
    if not base:
        return raw
    if not isinstance(base, str) or base not in tables:
        raise ValueError(f"{where}: packet {key!r} extends {base!r}, which it does not declare")
    if base in (*chain, key):
        raise ValueError(f"{where}: packet {key!r} extends itself through {[*chain, key, base]}")
    merged = merged_table(tables, base, where, (*chain, key))
    merged.pop("aliases", None)
    return deep_merged(merged, raw)


def deep_merged(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """``over`` on top of ``base``: tables merge key by key, any other value (a list, a string) replaces."""
    out = dict(base)
    for key, value in over.items():
        inherited = out.get(key)
        out[key] = deep_merged(inherited, value) if isinstance(value, dict) and isinstance(inherited, dict) else value
    return out


def validated(tables: Mapping[str, Mapping[str, Any]], where: str) -> dict[str, PacketTable]:
    """Every table of ``tables``, merged with what it extends and validated, in file order."""
    out: dict[str, PacketTable] = {}
    for key in tables:
        try:
            out[key] = PacketTable.model_validate(merged_table(tables, key, where))
        except pydantic.ValidationError as exc:
            raise ValueError(f"{where}: packet {key!r}: {exc}") from exc
    return out


def definition(
    key: str,
    table: PacketTable,
    pages: Collection[str],
    *,
    tools: tuple[str, ...] | None = None,
    phases: tuple[PacketPhase, ...] = (),
    workflow: str = "",
) -> PacketDef:
    """The packet ``key``'s table declares; ``pages`` are the pages its source ships (the default ``skills``), and
    ``tools``, ``phases`` and ``workflow`` what a plugin's phases derive (``tools`` None: the table's own)."""
    skills = table.skills or ((key,) if key in pages else ())
    return PacketDef(
        name=table.name,
        skills=skills,
        packets=table.packets,
        env=tuple(table.env.items()),
        prompt_overlays=table.overlays,
        tools=table.tools if tools is None else tools,
        agents=table.agents,
        device=table.device,
        frozen=table.frozen,
        marker=table.marker,
        short=table.short,
        phases=phases,
        workflow=workflow,
    )


def shipped_pages() -> frozenset[str]:
    """The shipped skill pages, by directory name."""
    return frozenset(page.parent.name for page in (paths.PACKAGE / "skills").glob("*/SKILL.md"))


def register_shipped(path: pathlib.Path = SHIPPED) -> tuple[str, ...]:
    """Register the no-packet control, then every table of ``path`` in file order; the keys registered."""
    tables = tomllib.loads(path.read_text(encoding="utf-8")).get("packets", {})
    checked = validated(tables, str(path))
    if phased := sorted(key for key, table in checked.items() if table.agent):
        raise ValueError(f"{path}: packets {phased} name agent phases, which only a plugin declares")
    pages = shipped_pages()
    PACKETS.add("", PacketDef(name="No Skill Packet"), slotted=False)
    for key, table in checked.items():
        PACKETS.add(key, definition(key, table, pages), aliases=table.aliases)
    return tuple(checked)


#: The shipped packet keys, registered on import.
SHIPPED_KEYS = register_shipped()
