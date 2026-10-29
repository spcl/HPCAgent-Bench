# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""How a recorded entity is SPELLED in a figure: display names from the registered vocabulary
(:mod:`hpcagent_bench.vocabulary`), whose ``order`` also fixes colours. Every lookup FALLS BACK to the tag unchanged
rather than raising (``tests/test_display_names.py`` guards that)."""

import dataclasses
import functools
import re
from typing import Annotated

import pydantic
import yaml

from hpcagent_bench import columns, external_skills, models, paths, plugins, skill_packets, spec
from hpcagent_bench.kernel_key import kernel_stem
from hpcagent_bench.vocabulary import (
    KINDS,
    MODELS,
    PACKETS,
    Column,
    ModelEntry,
    PacketDef,
    check_vocabulary,
)

__all__ = [
    "COMPACT_NAMES",
    "COMPACT_NAME_MAX",
    "KEYED_FIELDS",
    "OFFLOAD_DELIVERY_NAME",
    "OFFLOAD_SETUP_TOKEN",
    "REGISTRY",
    "SHORT_NAME_MAX",
    "STUDIES",
    "SUITE_PREFIXES",
    "VOCABULARY_MODULES",
    "BaselineSpec",
    "ExperimentEntry",
    "Marker",
    "Names",
    "Registry",
    "RegistryFile",
    "StudiesFile",
    "Text",
    "canonical",
    "dash_spellings",
    "display_name",
    "entry_name",
    "framework_name",
    "harness_name",
    "kernel_compact_display_name",
    "kernel_display_name",
    "kernel_names",
    "kernel_short_display_name",
    "language_name",
    "language_of",
    "manifest_names",
    "model_checkpoint",
    "model_name",
    "model_of",
    "names",
    "order",
    "packet_name",
    "packet_of",
    "packet_parts",
    "packet_short_name",
    "packet_spellings",
    "registered",
    "registry",
    "setup_delivery_name",
    "setup_suffix",
    "slot",
    "token_of",
]

#: The pools (neutral colour, marker shapes, lightness step) every figure draws from.
REGISTRY = paths.ENVS_DIR / "registry.yaml"
#: Run-level data: studies, experiments, baselines and the spellings older setups recorded under.
STUDIES = paths.ENVS_DIR / "studies.yaml"
#: The modules whose import registers the vocabulary; importing this module is what loads them.
VOCABULARY_MODULES = (models, skill_packets, columns)

#: One entity kind's tag -> display name. Key ORDER is the colour and marker order.
Names = dict[str, str]


#: A YAML key or value forced to text (YAML reads ``on`` as True and a bare version as a float).
Text = Annotated[str, pydantic.BeforeValidator(str)]


class BaselineSpec(pydantic.BaseModel):
    """The canon-sweep columns a study is drawn against: one ``denominator`` per study, and ``comparators`` drawn as
    their own series (never the denominator)."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    denominator: Text
    comparators: tuple[Text, ...] = ()


class ExperimentEntry(pydantic.BaseModel):
    """One launcher's job-name prefix: its study, device and served tag (empty = none); ``name`` is finer than the
    study. ``study``, ``name`` and ``prefix`` fall back to the entry's key."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    study: Text = ""
    name: Text = ""
    device: Text = ""
    tag: Text = ""
    #: The setup-name prefix it owns (the key, unless set) and, when set, the one name token its setups
    #: carry (``llr40-<model>-<lang>-blind``): such an experiment takes those setups from its prefix's.
    prefix: Text = ""
    suffix: Text = ""
    #: The ``experiments/experiments.toml`` experiment its setups were staged from (``submit.sh`` ``BASE``).
    base: Text = ""


#: The :class:`ExperimentEntry` fields an experiment that leaves them out takes from its key.
KEYED_FIELDS = ("study", "name", "prefix")

#: A matplotlib marker: a code such as ``"s"``, or ``(sides, style, angle)``.
Marker = str | tuple[int, int, float]


class RegistryFile(pydantic.BaseModel):
    """:data:`REGISTRY`: the pools every figure draws from."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    control_color: str = "#4d4d4d"
    markers: tuple[str, ...] = ()
    #: The treatments' shape pool: a marker code or a ``[sides, style, angle]`` list.
    shapes: tuple[Marker, ...] = ()
    #: How many of the pool's free shapes, from the front, are the harnesses'; packets take the rest.
    harness_shapes: int = pydantic.Field(default=8, ge=0)
    lightness_step: float = 0.13


class StudiesFile(pydantic.BaseModel):
    """:data:`STUDIES`: the studies, the experiments that run them, their baselines and owed run roots."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    studies: dict[Text, Text] = pydantic.Field(default_factory=dict[str, str])
    experiments: dict[Text, ExperimentEntry] = pydantic.Field(default_factory=dict[str, ExperimentEntry])
    owed_run_roots: dict[Text, tuple[Text, ...]] = pydantic.Field(default_factory=dict[str, tuple[str, ...]])
    dropped_setups: str = ""
    study_baselines: dict[Text, BaselineSpec] = pydantic.Field(default_factory=dict[str, BaselineSpec])

    @pydantic.field_validator("experiments")
    @classmethod
    def keyed_defaults(cls, experiments: dict[str, ExperimentEntry]) -> dict[str, ExperimentEntry]:
        """Each entry's :data:`KEYED_FIELDS` that the file leaves out, from its key."""
        return {
            key: entry.model_copy(update={field: key for field in KEYED_FIELDS if field not in entry.model_fields_set})
            for key, entry in experiments.items()
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Registry:
    """The parsed registry: the YAML validated ONCE into the typed fields consumers read."""

    control_color: str
    markers: tuple[str, ...]
    #: The treatments' shape pool (harnesses take its front, packets the rest): a marker code or a
    #: (sides, style, angle) tuple, see :mod:`hpcagent_bench.stats.palette`.
    shapes: tuple["Marker", ...]
    #: The front of the pool's free shapes reserved for harnesses (:func:`hpcagent_bench.stats.palette.shape_table`).
    harness_shapes: int
    lightness_step: float
    studies: Names
    models: dict[str, ModelEntry]
    #: Standalone optimizers (DaCe, CPF): shapes after the models, see :func:`optimizer_name`.
    optimizers: Names
    packets: Names
    packet_defs: dict[str, PacketDef]
    devices: Names
    languages: Names
    frameworks: Names
    harnesses: Names
    #: job-name prefix -> the experiment it names. Longest prefix wins; see :mod:`hpcagent_bench.experiments`.
    experiments: dict[str, ExperimentEntry]
    #: One regex matching every setup the user retired from the studies.
    dropped_setups: str
    #: study -> the canon columns it is scored against and drawn beside.
    study_baselines: dict[str, "BaselineSpec"]
    #: kind -> {spelling: the tag it names}, so an alias never takes its own colour slot.
    aliases: dict[str, Names]
    #: study -> the run-root prefixes its fused owed waves write (:func:`owed_run_roots_of`).
    owed_run_roots: dict[str, tuple[str, ...]] = dataclasses.field(default_factory=dict[str, tuple[str, ...]])


def entry_name(entry: ModelEntry | PacketDef | Column | str) -> str:
    """The name a figure prints for a vocabulary entry."""
    match entry:
        case str():
            return entry
        case Column():
            return entry.display
        case ModelEntry() | PacketDef():
            return entry.name


def registered(kind: str) -> Names:
    """``{key: display name}`` of a vocabulary kind, in slot order (the no-packet control first)."""
    block = KINDS[kind]
    return {key: entry_name(block.entries[key]) for key in block}


@functools.lru_cache(maxsize=1, typed=True)
def registry() -> Registry:
    """The code-registered vocabulary plus the two yaml files' pools and run-level data (cached); external packets
    join the packet vocabulary."""
    external_skills.register()
    plugins.register()
    check_vocabulary()
    pools = RegistryFile.model_validate(yaml.safe_load(REGISTRY.read_text(encoding="utf-8")) or {})
    doc = StudiesFile.model_validate(yaml.safe_load(STUDIES.read_text(encoding="utf-8")) or {})
    return Registry(
        control_color=pools.control_color,
        markers=pools.markers,
        shapes=pools.shapes,
        harness_shapes=pools.harness_shapes,
        lightness_step=pools.lightness_step,
        studies=dict(doc.studies),
        models={key: MODELS.entries[key] for key in MODELS},
        optimizers=registered("optimizers"),
        packets=registered("packets"),
        packet_defs={key: PACKETS.entries[key] for key in PACKETS},
        devices=registered("devices"),
        languages=registered("languages"),
        frameworks=registered("frameworks"),
        harnesses=registered("harnesses"),
        experiments=dict(doc.experiments),
        dropped_setups=doc.dropped_setups,
        study_baselines=dict(doc.study_baselines),
        aliases={kind: dict(block.aliases) for kind, block in KINDS.items()},
        owed_run_roots=dict(doc.owed_run_roots),
    )


def slot(kind: str, tag: str) -> int | None:
    """The slot ``tag`` takes within ``kind``: :meth:`~hpcagent_bench.registry.Kind.slot` of a vocabulary kind,
    else (the yaml kinds) its position; ``None`` for an unregistered tag and for the no-packet control."""
    resolved = canonical(kind, tag)
    block = KINDS.get(kind)
    if block is not None:
        return block.slot(resolved) if resolved in block.keys() else None
    known = tuple(names(kind))
    return known.index(resolved) if resolved in known else None


def canonical(kind: str, tag: str) -> str:
    """``tag`` with an alias resolved (so a spelling never gets its own colour or legend entry); unregistered passes."""
    return registry().aliases.get(kind, {}).get(str(tag), str(tag))


#: Entity kind -> the registry field holding its names; an unknown kind gets an empty block (figures keep drawing).
def names(kind: str) -> Names:
    """The ordered ``{tag: name}`` block for one entity kind. Key order IS channel order."""
    reg = registry()
    blocks: dict[str, Names] = {
        "studies": reg.studies,
        "models": {tag: entry.name for tag, entry in reg.models.items()},
        "optimizers": reg.optimizers,
        "packets": reg.packets,
        "devices": reg.devices,
        "languages": reg.languages,
        "frameworks": reg.frameworks,
        "harnesses": reg.harnesses,
    }
    return blocks.get(kind, {})


def order(kind: str) -> tuple[str, ...]:
    """The tags of ``kind`` in registry order -- the order that assigns colours and markers."""
    return tuple(names(kind))


def display_name(tag: str) -> str:
    """The name to put on a figure for a study tag. Falls back to the tag itself."""
    if not tag:
        return ""
    known = names("studies")
    if tag in known:
        return known[tag]
    # A setup rather than a study ("llr40-qwen38-c-skills"): title it by its study.
    return known.get(tag.split("-", 1)[0], tag)


def model_name(model: str) -> str:
    """The display spelling of a model. Unknown ones pass through unchanged."""
    entry = registry().models.get(canonical("models", str(model).lower()))
    return entry.name if entry is not None else str(model)


def model_checkpoint(model: str) -> str:
    """The checkpoint a model tag is expected to serve, or "" (checked by ``tests/test_display_names.py``)."""
    entry = registry().models.get(canonical("models", str(model).lower()))
    return entry.serves if entry is not None else ""


def packet_short_name(packet: str) -> str:
    """A packet's short figure spelling (its registry ``short:``), else its display name."""
    definition = registry().packet_defs.get(canonical("packets", str(packet)))
    return definition.short if definition is not None and definition.short else packet_name(packet)


def packet_name(packet: str) -> str:
    """The display spelling of a skill packet: "" is the control; a combination joins its parts with " + "."""
    known = names("packets")
    key = canonical("packets", str(packet))
    if key in known:
        return known[key]
    found = [p for p in packet_parts(key) if p]
    return " + ".join(known.get(p, p) for p in found) if found else known.get("", "No Skill Packet")


def packet_parts(packet: str) -> tuple[str, ...]:
    """The packets in a ``packet`` value, aliases resolved, empties dropped (``()`` for the control); splits on ``+``
    and ``;``. Shared by the palette and the labels."""
    found = [canonical("packets", p) for p in re.split(r"[+;]+", str(packet))]
    return tuple(dict.fromkeys(p for p in found if p))


@functools.lru_cache(maxsize=None, typed=True)
def dash_spellings(kind: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """``(tag, its dash-bounded spellings)`` of every ``kind`` entry in registry order, aliases included."""
    aliases = registry().aliases.get(kind, {})
    return tuple(
        (tag, tuple(f"-{name}-" for name in (tag, *(a for a, target in aliases.items() if target == tag))))
        for tag in order(kind)
    )


def token_of(setup: str, kind: str, unknown: str) -> str:
    """The first ``kind`` tag (registry order) whose spelling is a whole dash-delimited token of ``setup``."""
    padded = f"-{setup}-"
    for tag, spellings in dash_spellings(kind):
        if any(spelling in padded for spelling in spellings):
            return tag
    return unknown


def model_of(setup: str, unknown: str = "other") -> str:
    """The model tag a setup name carries as a whole dash-delimited token (registry order, aliases resolved);
    ``unknown`` when none. A last resort: prefer the recorded identity columns."""
    return token_of(setup, "models", unknown)


def language_of(setup: str, unknown: str = "") -> str:
    """The language tag a setup name carries as a whole dash-delimited token; ``unknown`` when none. A last resort,
    like :func:`model_of`."""
    return token_of(setup, "languages", unknown)


def setup_suffix(setup: str) -> str:
    """The dash-padded part of a setup name after its model token (``-c-cpf-``); "" without a registered model. A
    packet is read only from here (a study prefix can spell one)."""
    padded = f"-{setup}-"
    ends = [padded.find(s) + len(s) - 1 for _, spellings in dash_spellings("models") for s in spellings if s in padded]
    return padded[min(ends) :] if ends else ""


@functools.lru_cache(maxsize=1, typed=True)
def packet_spellings() -> tuple[tuple[str, str], ...]:
    """``(packet, dash-bounded spelling)`` for every packet and packet alias, longest spelling first."""
    aliases = registry().aliases.get("packets", {})
    rows = [(alias, target) for alias, target in aliases.items() if target]
    rows += [(packet, packet) for packet in order("packets") if packet]
    return tuple((target, f"-{spelling}-") for spelling, target in sorted(rows, key=lambda row: -len(row[0])))


def packet_of(setup: str, unknown: str = "") -> str:
    """The packet a setup name carries after its model token; ``unknown`` when none (the caller decides whether that
    is the control). A last resort, like :func:`model_of`."""
    suffix = setup_suffix(setup)
    for packet, spelling in packet_spellings():
        if spelling in suffix:
            return packet
    return unknown


#: The most characters a manifest's ``short-name`` may hold (a rotated tick on a 40-kernel axis).
SHORT_NAME_MAX: int = 14


@functools.lru_cache(maxsize=1, typed=True)
def manifest_names() -> tuple[Names, Names]:
    """``(names, short_names)`` keyed by kernel name, from ONE light YAML pass; an unparseable manifest is left out
    (its axis label is its stem)."""
    found: Names = {}
    short: Names = {}
    for key in spec.KERNELS:
        path = spec.KERNELS[key]
        kernel = kernel_stem(key)
        try:
            raw = spec.load_yaml(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 -- a broken manifest just keeps its stem on the axis
            continue
        title = raw.get("name")
        if isinstance(title, str) and title:
            found[kernel] = title
        abbreviated = raw.get("short-name")
        if isinstance(abbreviated, str) and abbreviated:
            short[kernel] = abbreviated
    return found, short


def kernel_names() -> Names:
    """``short_name -> the manifest's own ``name``, for every benchmark in the corpus (the kernel-axis labels)."""
    return manifest_names()[0]


def kernel_display_name(kernel: str) -> str:
    """The kernel-axis name for ``kernel`` (a manifest short_name); falls back to the identifier unchanged."""
    return kernel_names().get(str(kernel), str(kernel))


def kernel_short_display_name(kernel: str) -> str:
    """The manifest's ``short-name`` for a compact axis, else its ``name``."""
    return manifest_names()[1].get(str(kernel), kernel_display_name(kernel))


#: Longest compact name (:func:`kernel_compact_display_name`) on a paper-width per-kernel axis.
COMPACT_NAME_MAX: int = 10

#: A suite prefix a compact name drops.
SUITE_PREFIXES: tuple[str, ...] = ("TSVC ",)

#: Compact names for kernels whose short name exceeds :data:`COMPACT_NAME_MAX`. DISPLAY ONLY.
COMPACT_NAMES: dict[str, str] = {
    "argmax_with_index": "Argmax",
    "ext_break_capture": "Early Brk",
    "ext_war_unit": "WAR Unit",
    "fuse_diamond": "Diamond",
    "fuse_move_ifs": "Hoist Ifs",
    "fuse_stencil_through_transient": "Stencil",
    "quasi_affine_reduce_odd": "Quasi-Aff",
    "scan_affine_decay": "Aff. Scan",
    "scatter_accum_dup": "Scatter",
    "segment_reduce_ragged": "Ragged",
    "versioned_distance_update": "Dist Upd",
    "wf_diff_skew": "Wave Skew",
    "wf_triangular": "Tri Wave",
}


def kernel_compact_display_name(kernel: str) -> str:
    """The kernel's name for a paper-width axis: :data:`COMPACT_NAMES` where one is set, else its
    short name without a :data:`SUITE_PREFIXES` prefix (``TSVC s2710`` -> ``s2710``)."""
    if str(kernel) in COMPACT_NAMES:
        return COMPACT_NAMES[str(kernel)]
    name = kernel_short_display_name(kernel)
    return next((name.removeprefix(prefix) for prefix in SUITE_PREFIXES if name.startswith(prefix)), name)


def language_name(language: str) -> str:
    """The display spelling of a language. Unknown ones pass through unchanged."""
    key = canonical("languages", str(language).lower())
    return names("languages").get(key, str(language))


#: What a GPU C setup delivered (recorded language ``c``, written as ``#pragma omp target``). DISPLAY ONLY.
OFFLOAD_DELIVERY_NAME: str = "OpenMP Offload"

#: The setup-name token those setups carry, for a caller holding a name and no device column.
OFFLOAD_SETUP_TOKEN: str = "-c-openmp-"


def setup_delivery_name(setup: str) -> str:
    """The display spelling of what a setup DELIVERED, from its NAME: its language, except a GPU C setup
    (:data:`OFFLOAD_SETUP_TOKEN`), which is :data:`OFFLOAD_DELIVERY_NAME`."""
    name = str(setup)
    if OFFLOAD_SETUP_TOKEN in f"-{name}-":
        return OFFLOAD_DELIVERY_NAME
    return language_name(language_of(name))


def framework_name(framework: str) -> str:
    """The display spelling of a compiler or framework through its alias (``c-autopar`` and ``cc_autopar`` agree)."""
    key = canonical("frameworks", str(framework).lower())
    return names("frameworks").get(key, str(framework))


def harness_name(harness: str) -> str:
    """The display spelling of an agent harness. Unknown ones pass through unchanged."""
    return names("harnesses").get(str(harness).lower(), str(harness))
