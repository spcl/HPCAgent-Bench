# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The vocabulary of a figure and a results database: LLMs, standalone optimizers, agent harnesses,
languages, devices and skill packets, each registered by a class decorator.

The decorated classes live in :mod:`hpcagent_bench.models` (the first five) and
:mod:`hpcagent_bench.skill_packets`. This module owns the decorators, what a decorated class must
provide, and the registries they fill (:mod:`hpcagent_bench.registry` has the mechanism and the
rules every kind shares). ``docs/extending/registry.md`` is the author's guide.

THE SLOTS. ``order`` is the explicit slot of an entry. The colour ramp and the marker pool are
indexed by it (:mod:`hpcagent_bench.stats.palette`): a model takes marker ``order`` of the pool, an
optimizer ``order`` counted from its end, a harness and a packet the next free shape of the treatment
pool in slot order, and every kind takes hue ``order`` of the ramp. So an entry's colour and marker
change if and only if its ``order`` does: never reuse or renumber one, and give a new entry
:meth:`~hpcagent_bench.registry.Kind.next_order`. ``tests/test_vocabulary.py`` pins every slot.
"""

import dataclasses
from collections.abc import Callable, Iterable
from typing import Any

from hpcagent_bench.registry import Field, Kind, RegistryError

__all__ = [
    "DEVICES",
    "HARNESSES",
    "KINDS",
    "LANGUAGES",
    "MODELS",
    "OPTIMIZERS",
    "PACKETS",
    "ModelEntry",
    "PacketDef",
    "check_vocabulary",
    "device",
    "harness",
    "language",
    "llm",
    "optimizer",
    "packet",
]


@dataclasses.dataclass(frozen=True, slots=True)
class ModelEntry:
    """A model's display name and the checkpoint it is expected to serve.

    The checkpoint is recorded so an experiment that swaps one cannot silently keep the old name on an
    axis; ``tests/test_display_names.py`` checks it against what the setups really ran."""

    name: str
    serves: str


@dataclasses.dataclass(frozen=True, slots=True)
class PacketDef:
    """One packet's definition, before ``${VAR}`` placeholders are filled or ``lang``/``*`` are expanded
    into concrete skill pages -- see :mod:`hpcagent_bench.packets`, the resolver that reads this."""

    name: str
    skills: tuple[str, ...]
    packets: tuple[str, ...]
    env: tuple[tuple[str, str], ...]
    method: str
    #: MCP tools this packet CARRIES -- served by agent/tools/mcp_server.py only in its setups (its
    #: ``PACKET_TOOL_SWITCH``). Its ``skills`` pages are then that tool's manual, which is why ``*`` does
    #: not expand to them (:func:`hpcagent_bench.packets.tool_pages`).
    tools: tuple[str, ...] = ()
    #: Whose tools the pages teach (cpu, amd, nvidia); "" for a device-neutral packet.
    device: str = ""
    #: Why a recorded key takes no new submissions; "" while it still does.
    frozen: str = ""
    #: The shape this packet wears instead of the next free one from the pool ("": the pool's).
    marker: str = ""
    #: A figure's short spelling, for a column naming delivery and packet at once ("C-Skills").
    short: str = ""


NAME = Field(str, doc="the display name a figure prints")


def named(key: str, attrs: dict[str, Any]) -> str:
    """The display name of an entry that is a name and nothing else."""
    return str(attrs["name"])


def packet_def(key: str, attrs: dict[str, Any]) -> PacketDef:
    """A packet's :class:`PacketDef`; its env mapping becomes ordered ``(key, value)`` pairs."""
    return PacketDef(
        name=attrs["name"],
        skills=attrs["skills"],
        packets=attrs["packets"],
        env=tuple((str(name), str(value)) for name, value in attrs["env"].items()),
        method=attrs["method"],
        tools=attrs["tools"],
        device=attrs["device"],
        frozen=attrs["frozen"],
        marker=attrs["marker"],
        short=attrs["short"],
    )


MODELS = Kind(
    "models",
    {
        "name": NAME,
        "serves": Field(str, doc="the checkpoint the tag is expected to serve"),
    },
    lambda key, attrs: ModelEntry(attrs["name"], attrs["serves"]),
)
OPTIMIZERS = Kind("optimizers", {"name": NAME}, named)
HARNESSES = Kind("harnesses", {"name": NAME}, named)
LANGUAGES = Kind("languages", {"name": NAME}, named)
DEVICES = Kind("devices", {"name": NAME}, named)
PACKETS = Kind(
    "packets",
    {
        "name": NAME,
        "skills": Field(tuple, (), "skill page directories to stage; `lang` and `*` expand at resolve time"),
        "packets": Field(tuple, (), "other registered packet keys this one composes"),
        "env": Field(dict, {}, "KEY -> value env switches; a value may hold ${VAR}"),
        "method": Field(str, "", "a directory under agent/packets/, at most one per resolved packet"),
        "tools": Field(tuple, (), "MCP tools this packet carries"),
        "device": Field(str, "", "cpu, amd or nvidia: whose tools the pages teach"),
        "frozen": Field(str, "", "why the key takes no new submissions"),
        "marker": Field(str, "", "the shape this packet wears instead of the pool's next free one"),
        "short": Field(str, "", "a figure's short spelling"),
    },
    packet_def,
)

#: Every vocabulary kind by its plural name, the spelling a figure and the display-name lookups use.
KINDS: dict[str, Kind[Any]] = {kind.name: kind for kind in (MODELS, OPTIMIZERS, HARNESSES, LANGUAGES, DEVICES, PACKETS)}


def llm(key: str, *, order: int | None, aliases: Iterable[str] = ()) -> Callable[[type], type]:
    """Register an LLM under ``key``, the tag its setups record (``qwen38``).

    The class must provide ``name`` (str: the display name) and ``serves`` (str: the checkpoint the tag
    is expected to serve). ``order`` is the slot of its marker (:func:`~hpcagent_bench.registry.Kind.next_order`
    for a new one); ``aliases`` are other spellings that resolve to it and take no slot of their own."""
    return MODELS.register(key, order=order, aliases=aliases)


def optimizer(key: str, *, order: int | None, aliases: Iterable[str] = ()) -> Callable[[type], type]:
    """Register a standalone optimizer under ``key``: a compiler or pipeline that stands where an LLM
    stands on a figure (``dace``, ``cpf``, ``pluto``).

    The class must provide ``name`` (str). Its marker is slot ``order`` counted from the END of the pool,
    so registering another LLM never repaints it; device variants are ``aliases``."""
    return OPTIMIZERS.register(key, order=order, aliases=aliases)


def harness(key: str, *, order: int | None, aliases: Iterable[str] = ()) -> Callable[[type], type]:
    """Register an agent harness under ``key``, the ``harness`` column value (``claude``).

    The class must provide ``name`` (str). Harnesses take the clearest shapes of the treatment pool,
    ahead of every packet, in slot order."""
    return HARNESSES.register(key, order=order, aliases=aliases)


def language(key: str, *, order: int | None, aliases: Iterable[str] = ()) -> Callable[[type], type]:
    """Register a language under ``key``, the language a setup asked for (``c``, ``hip``).

    The class must provide ``name`` (str), the proper name (``C++``, not ``Cpp``). A GPU-only language
    is a language here, not a framework: it is what the agent was asked to write."""
    return LANGUAGES.register(key, order=order, aliases=aliases)


def device(key: str, *, order: int | None, aliases: Iterable[str] = ()) -> Callable[[type], type]:
    """Register a device under ``key``, the ``device`` column value the schema constrains
    (``cpu``, ``gpu``, ``cpu-multinode``, ``gpu-multinode``).

    The class must provide ``name`` (str)."""
    return DEVICES.register(key, order=order, aliases=aliases)


def packet(key: str, *, order: int | None, aliases: Iterable[str] = ()) -> Callable[[type], type]:
    """Register a skill packet under ``key``, the canonical ``packet`` column value.

    A packet names a set of skills, tools and env switches an agent is given, never a device and never
    a programming model. The class must provide ``name`` (str) and may provide ``skills``, ``packets``,
    ``tools`` (tuples of str), ``env`` (a dict of KEY -> value), ``method``, ``device``, ``frozen``,
    ``marker`` and ``short`` (str); :mod:`hpcagent_bench.packets` resolves them. The key ``""`` is the
    no-packet control: it takes ``order=None`` and a neutral colour of its own.

    A registered key's definition is immutable once a results database has recorded it: a changed
    meaning goes under a NEW key, and a rename without a change of meaning is an alias. ``order`` is
    the slot of the packet's hue and of its shape; ``frozen`` keeps a retired key resolving for the
    records that hold it."""
    return PACKETS.register(key, order=order, aliases=aliases)


#: The devices a packet's ``device`` may name: whose tools its pages teach.
PACKET_DEVICES = frozenset({"", "cpu", "amd", "nvidia"})


def check_vocabulary() -> None:
    """The cross-entry rules no single decorator can check, run once the whole vocabulary is registered:
    every kind but the packets gives every entry a slot, only the no-packet control has none, a composed
    packet and a ``device`` name something that exists."""
    for kind in KINDS.values():
        slotless = sorted(key for key, order in kind.orders.items() if order is None and (kind is not PACKETS or key))
        if slotless:
            raise RegistryError(f"{kind.name}: {slotless} have no order, so no colour or marker slot")
    if PACKETS.orders.get("", 0) is not None or "" not in PACKETS.entries:
        raise RegistryError("packets: the no-packet control (key '') must be registered with order=None")
    for key, definition in PACKETS.entries.items():
        for part in definition.packets:
            if PACKETS.canonical(part) not in PACKETS.entries:
                raise RegistryError(f"packets {key!r}: composes {part!r}, which is not registered")
        if definition.device not in PACKET_DEVICES:
            raise RegistryError(f"packets {key!r}: device {definition.device!r} is not one of {sorted(PACKET_DEVICES)}")
