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
import re
from collections.abc import Callable, Iterable
from typing import Any, NotRequired, TypedDict

from hpcagent_bench.precision import Precision
from hpcagent_bench.registry import Field, Kind, RegistryError

__all__ = [
    "DEVICES",
    "FRAMEWORKS",
    "HARNESSES",
    "KINDS",
    "LANGUAGES",
    "MODELS",
    "OPTIMIZERS",
    "PACKETS",
    "RETIRED_FRAMEWORKS",
    "FrameworkMeta",
    "ModelEntry",
    "PacketDef",
    "check_vocabulary",
    "device",
    "framework",
    "framework_slots",
    "harness",
    "language",
    "llm",
    "optimizer",
    "packet",
    "retired_framework",
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
    #: MCP tools this packet CARRIES -- served by agent/hpcagent_agent/tools/mcp_server.py only in its setups (its
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


#: One framework column's descriptor. A TypedDict rather than a dataclass because these entries are read by
#: SUBSCRIPT across the repo (the CLI, preflight, the flavor tests) and ``Framework.info`` is one of them
#: with ``simple_name`` added. :func:`framework` builds it from a decorated class.
class FrameworkMeta(TypedDict):
    display: str
    adapter: str
    base: str
    sweep_deterministic: bool
    full_name: str
    postfix: str
    arch: str
    precisions: frozenset[Precision]
    pipelines: NotRequired[tuple[str, ...]]
    column: NotRequired[str]
    flavor: NotRequired[str]
    language: NotRequired[str]
    emit_language: NotRequired[str]
    compiler: NotRequired[str]
    flags: NotRequired[str]
    autopar_gate: NotRequired[str]
    transform: NotRequired[str]
    simple_name: NotRequired[str]


NAME = Field(str, doc="the display name a figure prints")
ADAPTER_PATTERN = re.compile(r"^[a-z_][a-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
FRAMEWORK_ARCHS = frozenset({"cpu", "gpu"})
#: The optional descriptor fields; one a class leaves out is absent from its entry, not ``None``.
FRAMEWORK_OPTIONAL = (
    "pipelines",
    "column",
    "flavor",
    "language",
    "emit_language",
    "compiler",
    "flags",
    "autopar_gate",
    "transform",
)


def framework_meta(key: str, attrs: dict[str, Any]) -> FrameworkMeta:
    """A framework column's :class:`FrameworkMeta`; refuses an ``arch`` that is not cpu/gpu, an ``adapter`` that
    is not ``package.module:Class``, an empty precision set and a ``column`` without its ``flavor``."""
    if attrs["arch"] not in FRAMEWORK_ARCHS:
        raise RegistryError(f"frameworks {key!r}: arch must be one of {sorted(FRAMEWORK_ARCHS)}, got {attrs['arch']!r}")
    if not ADAPTER_PATTERN.match(attrs["adapter"]):
        raise RegistryError(f"frameworks {key!r}: adapter {attrs['adapter']!r} must read 'package.module:Class'")
    if not attrs["precisions"]:
        raise RegistryError(f"frameworks {key!r}: precisions is empty, so the column can execute nothing")
    if (attrs["column"] is None) != (attrs["flavor"] is None):
        raise RegistryError(f"frameworks {key!r}: a flavor entry declares both column and flavor, or neither")
    required = [name for name in FrameworkMeta.__required_keys__]
    meta = {name: attrs[name] for name in required}
    meta.update({name: attrs[name] for name in FRAMEWORK_OPTIONAL if attrs[name] is not None})
    return meta  # type: ignore[return-value]


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
FRAMEWORKS = Kind(
    "frameworks",
    {
        "display": Field(str, doc="the name a figure prints: the compiler or library, not the flavor"),
        "adapter": Field(str, doc="'package.module:Class' of the Framework subclass, imported on first use"),
        "base": Field(str, doc="the backend this column is a flavor of; columns sharing a base share an adapter"),
        "full_name": Field(str, doc="the long name a table prints"),
        "postfix": Field(str, doc="selects the kernel impl file, <module>_<postfix>.py"),
        "arch": Field(str, doc="cpu or gpu"),
        "sweep_deterministic": Field(bool, doc="a deterministic (unjudged, no-agent) sweep may select it"),
        "precisions": Field(frozenset, doc="the Precision set it can execute; else the sweep records skip"),
        "pipelines": Field(tuple, None, "the SDFG pipelines a DaCe flavor compiles and scores"),
        "column": Field(str, None, "with flavor: the DB column this name groups under"),
        "flavor": Field(str, None, "with column: the optimizer; the name must be <column>_<flavor>"),
        "language": Field(str, None, "what the native/pluto column compiles"),
        "emit_language": Field(str, None, "the language of the translator output the sources start from"),
        "compiler": Field(str, None, "the compilers.yaml block the build forces"),
        "flags": Field(str, None, "the hpcagent_bench.flags preset appended to the baseline"),
        "autopar_gate": Field(str, None, "the flags.<probe>() that must read OK before the column builds"),
        "transform": Field(str, None, "pluto or ppcg: the source-to-source tool whose output it compiles"),
    },
    framework_meta,
)
#: A column no setup builds any more, kept so a recorded row still resolves to a name and its hue slot is
#: never reused (a removal would repaint every figure already drawn). It shares the slot namespace of
#: :data:`FRAMEWORKS`; see :func:`framework_slots`.
RETIRED_FRAMEWORKS = Kind(
    "retired frameworks",
    {"display": Field(str, doc="the name a figure prints"), "reason": Field(str, doc="why it was retired")},
    lambda key, attrs: attrs["display"],
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
        "method": Field(str, "", "a directory under agent/hpcagent_agent/packets/, at most one per resolved packet"),
        "tools": Field(tuple, (), "MCP tools this packet carries"),
        "device": Field(str, "", "cpu, amd or nvidia: whose tools the pages teach"),
        "frozen": Field(str, "", "why the key takes no new submissions"),
        "marker": Field(str, "", "the shape this packet wears instead of the pool's next free one"),
        "short": Field(str, "", "a figure's short spelling"),
    },
    packet_def,
)

#: Every vocabulary kind by its plural name, the spelling a figure and the display-name lookups use.
KINDS: dict[str, Kind[Any]] = {
    kind.name: kind for kind in (MODELS, OPTIMIZERS, HARNESSES, LANGUAGES, DEVICES, PACKETS, FRAMEWORKS)
}


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


def framework(key: str, *, order: int | None, aliases: Iterable[str] = ()) -> Callable[[type], type]:
    """Register a framework column under ``key``, the ``framework`` column value and the CLI name (``cc``,
    ``dace_cpu_autoopt``). The class must provide every field of :data:`FRAMEWORKS` that has no default
    (see ``docs/extending/registry.md``): ``display``, ``adapter``, ``base``, ``full_name``, ``postfix``,
    ``arch``, ``sweep_deterministic`` and ``precisions``; the rest describe a native, pluto or DaCe flavor.

    ``adapter`` names the :class:`~hpcagent_bench.frameworks.framework.Framework` subclass as
    ``package.module:Class`` and is imported on first use, so registering a column never imports its
    backend. ``order`` is the hue slot, append-only like every kind; ``aliases`` are other spellings of
    the name. A flavor (``column`` + ``flavor``) must be named ``<column>_<flavor>``, and its column must
    be a registered framework: both are checked in :func:`check_vocabulary`."""
    return FRAMEWORKS.register(key, order=order, aliases=aliases)


def retired_framework(key: str, *, order: int) -> Callable[[type], type]:
    """Register a framework column no setup builds any more. The class must provide ``display`` and ``reason`` (str).
    The key keeps its hue slot and resolves to its name for rows already recorded."""
    return RETIRED_FRAMEWORKS.register(key, order=order)


def framework_slots() -> dict[str, int]:
    """Every framework key, live and retired, with its hue slot."""
    slots = {key: order for key, order in RETIRED_FRAMEWORKS.orders.items() if order is not None}
    slots.update({key: order for key, order in FRAMEWORKS.orders.items() if order is not None})
    return slots


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
    taken = [order for kind in (FRAMEWORKS, RETIRED_FRAMEWORKS) for order in kind.orders.values()]
    if len(taken) != len(set(taken)) or set(FRAMEWORKS.entries) & set(RETIRED_FRAMEWORKS.entries):
        raise RegistryError("frameworks: a live and a retired column share a name or a hue slot")
    for base in {meta["base"] for meta in FRAMEWORKS.entries.values()}:
        adapters = {meta["adapter"] for meta in FRAMEWORKS.entries.values() if meta["base"] == base}
        if len(adapters) != 1:
            raise RegistryError(f"frameworks: base {base!r} names adapters {sorted(adapters)}; it needs exactly one")
    if PACKETS.orders.get("", 0) is not None or "" not in PACKETS.entries:
        raise RegistryError("packets: the no-packet control (key '') must be registered with order=None")
    for key, definition in PACKETS.entries.items():
        for part in definition.packets:
            if PACKETS.canonical(part) not in PACKETS.entries:
                raise RegistryError(f"packets {key!r}: composes {part!r}, which is not registered")
        if definition.device not in PACKET_DEVICES:
            raise RegistryError(f"packets {key!r}: device {definition.device!r} is not one of {sorted(PACKET_DEVICES)}")
