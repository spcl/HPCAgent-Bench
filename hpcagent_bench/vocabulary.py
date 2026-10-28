# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The vocabulary of a figure and a results database (LLMs, optimizers, harnesses, languages, devices, packets,
columns): one :class:`~hpcagent_bench.registry.Kind` each and their entry types. Entries are added in
:mod:`models`, :mod:`columns`, :mod:`skill_packets` and by plugins; ``docs/extending/registry.md``."""

import dataclasses
import enum
import re

from hpcagent_agent.coded import Coded

from hpcagent_bench.precision import Precision
from hpcagent_bench.registry import Kind, RegistryError

__all__ = [
    "ADAPTER_PATTERN",
    "DEVICES",
    "FRAMEWORKS",
    "HARNESSES",
    "KINDS",
    "LANGUAGES",
    "MODELS",
    "OPTIMIZERS",
    "PACKETS",
    "Arch",
    "Column",
    "DaceStage",
    "ModelEntry",
    "PacketDef",
    "PacketDevice",
    "Transform",
    "check_vocabulary",
]


class Arch(Coded):
    """The device a framework column runs on, so the device anti-cheats a grade meets."""

    CPU = enum.auto()
    GPU = enum.auto()


class DaceStage(Coded):
    """A DaCe flavor's pipeline stage."""

    CANON = enum.auto()
    PARALLEL = enum.auto()
    AUTOOPT = enum.auto()


class Transform(Coded):
    """The source-to-source tool whose output a pluto-family column compiles."""

    PLUTO = enum.auto()
    PPCG = enum.auto()
    POLYGEIST = enum.auto()


class PacketDevice(Coded):
    """Whose tools a packet's pages teach."""

    CPU = enum.auto()
    AMD = enum.auto()
    NVIDIA = enum.auto()


@dataclasses.dataclass(frozen=True, slots=True)
class ModelEntry:
    """A model's display name and the checkpoint it is expected to serve (checked against what setups ran)."""

    name: str
    serves: str


@dataclasses.dataclass(frozen=True, slots=True)
class PacketDef:
    """One packet's definition before placeholders are filled and ``lang``/``*`` expanded (:mod:`packets`).

    Immutable once a results database recorded the key: a changed meaning goes under a NEW key."""

    #: The display name.
    name: str
    #: Skill page directories to stage; ``lang`` and ``*`` expand at resolve time.
    skills: tuple[str, ...] = ()
    #: Other registered packet keys this one composes.
    packets: tuple[str, ...] = ()
    #: ``KEY=value`` switches, in order; a value may hold ``${VAR}``.
    env: tuple[tuple[str, str], ...] = ()
    #: Prompt overlays its setups apply (:mod:`hpcagent_bench.prompt_build`), in order.
    prompt_overlays: tuple[str, ...] = ()
    #: MCP tools this packet CARRIES, served only in its setups; its ``skills`` are then that tool's manual.
    tools: tuple[str, ...] = ()
    #: Subagents this packet offers: a plugin's ``agents/<name>.md`` (:mod:`hpcagent_agent.harness.subagents`).
    agents: tuple[str, ...] = ()
    #: Whose tools the pages teach; None for a device-neutral packet.
    device: PacketDevice | None = None
    #: Why a recorded key takes no new submissions; "" while it still does.
    frozen: str = ""
    #: The shape this packet wears instead of the next free one from the pool ("": the pool's).
    marker: str = ""
    #: A figure's short spelling, for a column naming delivery and packet at once ("C-Skills").
    short: str = ""


#: ``package.module:Class``.
ADAPTER_PATTERN = re.compile(r"^[a-z_][a-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")


@dataclasses.dataclass(frozen=True, slots=True)
class Column:
    """One framework column (or flavor of one): what runs it, where, and how a figure names it."""

    #: The name a figure prints: the compiler or library, not the flavor.
    display: str
    #: ``package.module:Class`` of the Framework (or, for an optimizer flavor, its Agent), imported on first use.
    adapter: str
    #: The backend this column is a flavor of; columns sharing a base share an adapter.
    base: str
    #: The long name a table prints.
    full_name: str
    #: Selects the kernel impl file, ``<module>_<postfix>.py``.
    postfix: str
    arch: Arch
    #: Whether a deterministic (unjudged, no-agent) sweep may select it.
    sweep_deterministic: bool
    #: The precisions it can execute; ``run-framework`` does not enforce it.
    precisions: frozenset[Precision]
    #: The pipelines the adapter compiles and scores for this column (a DaCe flavor's SDFG pipeline).
    pipelines: tuple[str, ...] = ()
    #: The framework's own pipeline stage this flavor stops at (:class:`DaceStage` for DaCe).
    stage: Coded | None = None
    #: An optimizer (an Agent, often a plugin's) runs this flavor, not a Framework; ``adapter`` names it.
    optimizer: bool = False
    #: With ``flavor``: the DB column this name groups under; the name must be ``<column>_<flavor>``.
    column: str | None = None
    flavor: str | None = None
    #: What the native/pluto column compiles.
    language: str | None = None
    #: The language of the translator output the sources start from.
    emit_language: str | None = None
    #: The ``compilers.yaml`` block the build forces.
    compiler: str | None = None
    #: The :mod:`hpcagent_bench.flags` preset appended to the baseline.
    flags: str | None = None
    #: The ``flags.<probe>()`` that must read OK before the column builds.
    autopar_gate: str | None = None
    transform: Transform | None = None

    def __post_init__(self) -> None:
        if not ADAPTER_PATTERN.match(self.adapter):
            raise RegistryError(
                f"frameworks {self.display!r}: adapter {self.adapter!r} must read 'package.module:Class'"
            )
        if not self.precisions:
            raise RegistryError(f"frameworks {self.display!r}: precisions is empty, so the column can execute nothing")
        if (self.column is None) != (self.flavor is None):
            raise RegistryError(f"frameworks {self.display!r}: a flavor declares both column and flavor, or neither")


MODELS: Kind[ModelEntry] = Kind("models")
#: Standalone optimizers by display name: a compiler or pipeline that stands where an LLM stands on a figure.
OPTIMIZERS: Kind[str] = Kind("optimizers")
#: Agent harnesses (the loop driving the model) by display name.
HARNESSES: Kind[str] = Kind("harnesses")
#: The language a setup asked for, by display name.
LANGUAGES: Kind[str] = Kind("languages")
#: Where a kernel was timed, by display name.
DEVICES: Kind[str] = Kind("devices")
PACKETS: Kind[PacketDef] = Kind("packets")
FRAMEWORKS: Kind[Column] = Kind("frameworks")

#: Every vocabulary kind by its plural name, the spelling a figure and the display-name lookups use.
KINDS: dict[str, Kind[ModelEntry] | Kind[str] | Kind[PacketDef] | Kind[Column]] = {
    kind.name: kind for kind in (MODELS, OPTIMIZERS, HARNESSES, LANGUAGES, DEVICES, PACKETS, FRAMEWORKS)
}


def check_vocabulary() -> None:
    """The cross-entry rules, run once the vocabulary is registered: slots, one adapter per base, composed packets
    exist."""
    for kind in KINDS.values():
        slotless = sorted(key for key, order in kind.orders.items() if order is None and (kind is not PACKETS or key))
        if slotless:
            raise RegistryError(f"{kind.name}: {slotless} have no slot, so no colour or marker")
    for base in {column.base for column in FRAMEWORKS.entries.values()}:
        adapters = {
            column.adapter for column in FRAMEWORKS.entries.values() if column.base == base and not column.optimizer
        }
        if len(adapters) != 1:
            raise RegistryError(
                f"frameworks: base {base!r} names Framework adapters {sorted(adapters)}; it needs exactly one"
            )
    if PACKETS.orders.get("", 0) is not None or "" not in PACKETS.entries:
        raise RegistryError("packets: the no-packet control (key '') must be registered with slotted=False")
    for key, definition in PACKETS.entries.items():
        for part in definition.packets:
            if PACKETS.canonical(part) not in PACKETS.entries:
                raise RegistryError(f"packets {key!r}: composes {part!r}, which is not registered")
