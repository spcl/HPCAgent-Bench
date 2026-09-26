# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One colour and one shape per entity, decided by its identity and by nothing else.

A reader carries a colour between figures whether or not we meant them to, so a colour has to be a
property of the entity rather than of its position in whatever list one figure happened to hold:
dropping a series must not repaint the survivors.

**Shape is always the MODEL, colour whatever the figure VARIES -- except the packet efficacy
panels, which invert this.** There are a few models and there will be many packets, and colour
separates more values than shape does, so the model takes the shape and keeps it everywhere, EXCEPT
in :mod:`hpcagent_bench.stats.figures.efficacy` and
:mod:`hpcagent_bench.stats.figures.signed`: those panels already split one packet per
panel (or per pair), so colour is spent there for nothing, while the few models sharing that one
panel need telling apart at a glance when their summary marks overlap -- checked by rendering both
orders on the same llr40 figure; a shared hue with only a circle-vs-square edge to tell two
overlapping marks apart read far worse than two distinct hues with a shape that does not have to
carry any information. Those two modules read colour off :func:`model_color` and shape off
:func:`packet_marker` instead of :func:`color` and :func:`marker`.

Four things get coloured, and a figure varies exactly one of them, so they never compete for the
ramp: an INTERVENTION (a skill packet, or a scope such as `kernel`/`repo`/`no-score`), a HARNESS (an
agent-harness comparison), a FRAMEWORK (a compiler/library comparison, which has no agent in it), or
a MODEL (a figure whose only axis is which LLM ran).

ONE GLOBAL PALETTE: matplotlib's ``tab20``, extended by ``tab20b`` once its twenty slots are spent,
and nothing else. Every entity a figure colours -- packet, framework, model, harness, language --
takes a SLOT decided by its position in
``envs/registry.yaml``, so a colour is looked up in exactly one table and no figure, script or
registry entry carries a hex literal of its own.

A packet's colour is :func:`color`: its LEAD packet's slot and one lightness step per additional
packet, so ``cpfsrc`` and ``cpfsrc+lang-skills`` read as the same treatment family at two
strengths, and neutral grey for the no-packet control.

THE VOCABULARY AND THE ORDER ARE DATA, in ``envs/registry.yaml``, beside the display names: one
registry for one vocabulary.
"""

import colorsys
import functools
import logging
import zlib
from collections.abc import Iterable

import matplotlib
import matplotlib.colors

from hpcagent_bench import packets
from hpcagent_bench.experiment_tags import Registry, canonical, order, registry

__all__ = [
    "COMBINED_GRID",
    "COMBINED_L",
    "COMBINED_MIN_CHROMA",
    "COMBINED_SLOTS",
    "CONTROL_MARKER",
    "CONTROL_SHADE",
    "LOG",
    "TAB20",
    "TAB20B",
    "TAB20B_ORDER",
    "TAB20_ORDER",
    "color",
    "colormap_slot",
    "colors",
    "combined_ramp",
    "combined_slot",
    "control_color",
    "fixed_packet_markers",
    "framework_color",
    "framework_colors",
    "harness_marker",
    "hue_order",
    "hues",
    "in_order",
    "language_marker",
    "lighten",
    "marker",
    "markers",
    "model_color",
    "model_language_color",
    "model_markers",
    "model_shade",
    "oklab",
    "ordered_color",
    "packet_marker",
    "shape_table",
    "slot_color",
    "tab20_slot",
    "treatment_shape",
    "warn_on_collision",
]

LOG = logging.getLogger(__name__)

#: The colormap every figure in this repo draws from. The user's global palette decision.
TAB20: str = "tab20"

#: tab20 slots in DARK-FIRST order: its ten saturated slots, then their ten light twins. tab20 is
#: laid out as light/dark PAIRS, so reading it straight through would spend the second colour of a
#: figure on a pale wash of its first; taken this way the entities plotted most take the ten
#: colours that stay apart at 4pt, and the light twin of a hue only comes back once the dark ones
#: are spent.
TAB20_ORDER: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19)

#: Slots 21-40: the 21st entity extends the palette; it never wraps or shares a colour.
#: tab20b is five hues of four shades each; taken darkest shade of every hue first, then the next
#: shade, so its first entries stay as far apart as tab20's dark half.
TAB20B: str = "tab20b"
TAB20B_ORDER: tuple[int, ...] = (0, 4, 8, 12, 16, 2, 6, 10, 14, 18, 1, 5, 9, 13, 17, 3, 7, 11, 15, 19)


def colormap_slot(name: str, slot: int) -> str:
    """Entry ``slot`` of the colormap ``name`` as ``#rrggbb``. The ONE place a colour value enters
    this repo."""
    colormap = matplotlib.colormaps[name]
    return matplotlib.colors.to_hex(colormap(slot % colormap.N))


def tab20_slot(slot: int) -> str:
    """tab20 entry ``slot`` as ``#rrggbb``."""
    return colormap_slot(TAB20, slot)


def hues() -> tuple[str, ...]:
    """The categorical ramp: tab20 in :data:`TAB20_ORDER`, then tab20b in :data:`TAB20B_ORDER`."""
    return (
        *(tab20_slot(slot) for slot in TAB20_ORDER),
        *(colormap_slot(TAB20B, slot) for slot in TAB20B_ORDER),
    )


def markers() -> tuple[str, ...]:
    """The marker shapes, in assignment order."""
    return registry().markers


def control_color() -> str:
    """The colour of the no-packet control."""
    return registry().control_color


def hue_order(kind: str) -> tuple[str, ...]:
    """``kind``'s tags in the order that assigns hues.

    The control packet is dropped: it is a registered name with its own neutral colour, and leaving
    it in the ramp would shift every treatment one hue and repaint every figure already drawn."""
    return tuple(tag for tag in order(kind) if tag)


def slot_color(kind: str, name: str) -> str:
    """``name``'s tab20 slot within one entity ``kind``, silently -- an unregistered name gets a
    stable CRC slot. :func:`ordered_color` is this plus the warning; a caller that has already
    warned for ``name`` (:func:`color`) uses this one so one unknown entity logs once.

    CRC, never ``hash()``: ``hash`` is salted by PYTHONHASHSEED and would hand the same entity a
    different colour in two runs of the same script."""
    known, ramp = hue_order(kind), hues()
    resolved = canonical(kind, name)
    if resolved in known:
        return ramp[known.index(resolved) % len(ramp)]
    return ramp[zlib.crc32(str(name).encode()) % len(ramp)]


def ordered_color(kind: str, name: str) -> str:
    """``name``'s tab20 slot within one entity ``kind``, warning when the registry does not name it."""
    if canonical(kind, name) not in hue_order(kind):
        LOG.warning("palette: %s %r is not in registry.yaml; using a hash colour", kind, name)
    return slot_color(kind, name)


def warn_on_collision(chosen: dict[str, str], kind: str) -> dict[str, str]:
    """Pass ``chosen`` through, warning when two entities in ONE figure draw the same colour.

    There are more entities than hues, so the ramp wraps. Harmless until two that wrap onto each
    other appear together, where they read as one series. Warned rather than raised: a flagged plot
    beats no plot, and the fix is another hue, not a dropped series."""
    seen: dict[str, str] = {}
    for name, hue in chosen.items():
        entity = canonical(f"{kind}s", name)
        if hue in seen and seen[hue] != entity:
            LOG.warning("palette: %s %r and %r both draw %s; extend hues", kind, seen[hue], name, hue)
        seen[hue] = entity
    return chosen


def lighten(hex_color: str, steps: int) -> str:
    """``hex_color`` moved ``steps`` toward white in HLS, capped short of white so it stays visible.

    What makes a combination read as its lead packet's family rather than as a fourth treatment."""
    if steps <= 0:
        return hex_color
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    lightness = min(0.88, lightness + steps * registry().lightness_step)
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return matplotlib.colors.to_hex((r, g, b))


def color(packet: str) -> str:
    """The one colour ``packet`` wears, in every figure and every process: the control grey for the
    control, otherwise the lead part's tab20 slot lightened one step per extra part.

    Warns for each part registry.yaml does not name, since an unregistered part draws in a hash slot."""
    known = set(hue_order("packets"))
    for part in packets.spec_parts(packet):
        if part not in known:
            LOG.warning("palette: packets %r is not in registry.yaml; using a hash colour", part)
    parts = packets.spec_parts(packet)
    if not parts:
        return control_color()
    return lighten(slot_color("packets", packets.lead(parts)), len(parts) - 1)


def colors(names: Iterable[str]) -> dict[str, str]:
    """``{packet: colour}`` for one figure."""
    return warn_on_collision({p: color(p) for p in dict.fromkeys(names)}, "packet")


def in_order(names: Iterable[str], kind: str = "models") -> list[str]:
    """``names`` in registry order, unregistered ones last and alphabetical among themselves.

    The draw order of a figure's series, so a dodge or a legend is the same in every figure. Sorting
    by name alone would reorder every panel the day a model is renamed, and an unregistered name
    needs a tiebreak or two of them land in whatever order the frame happened to hold."""
    known = order(kind)

    def rank(name: str) -> tuple[int, str]:
        resolved = canonical(kind, name)
        return (known.index(resolved), "") if resolved in known else (len(known), str(name))

    return sorted(dict.fromkeys(names), key=rank)


def marker(model: str) -> str:
    """The one SHAPE an optimizer wears: an LLM (``models``) or a standalone optimizer
    (``optimizers``, e.g. DaCe or CPF). Models take shapes from the FRONT of the sequence and
    standalone optimizers from the BACK, so registering another model never repaints a figure that
    already carries DaCe or CPF. Pairs with :func:`color` so identity is never colour alone."""
    shapes = markers()
    models = order("models")
    resolved = canonical("models", model)
    if resolved in models:
        return shapes[models.index(resolved) % len(shapes)]
    standalone = order("optimizers")
    resolved = canonical("optimizers", model)
    if resolved in standalone:
        return shapes[-1 - (standalone.index(resolved) % len(shapes))]
    LOG.warning("palette: optimizer %r is not in registry.yaml; using a hash marker", model)
    return shapes[zlib.crc32(str(model).encode()) % len(shapes)]


def language_marker(language: str) -> str:
    """The one SHAPE a delivery language wears in a figure whose colour is the model (the transfer
    scatter): the registry's ``markers`` in ``languages`` order, so appending a language never
    reshapes another."""
    shapes = markers()
    languages = order("languages")
    resolved = canonical("languages", str(language).lower())
    if resolved in languages:
        return shapes[languages.index(resolved) % len(shapes)]
    LOG.warning("palette: language %r is not in registry.yaml; using a hash marker", language)
    return shapes[zlib.crc32(str(language).encode()) % len(shapes)]


#: The control's shape, reserved: no packet or harness is ever assigned it, and the control is drawn
#: hollow in its model's colour, so "no packet" reads the same in every figure.
CONTROL_MARKER: str = "o"


@functools.lru_cache(maxsize=1, typed=True)
def shape_table() -> dict[tuple[str, str], object]:
    """``(kind, key) -> shape`` for every registered treatment: harnesses, then packets, each taking
    the next free shape of the registry's pool in file order, or its packet entry's own ``marker:``.
    The control's circle is never handed out. Registering a treatment therefore gives it a shape of
    its own without reshaping any other; a pool too small, or two treatments on one shape, is a
    registry error raised here rather than two treatments drawn alike."""
    reg = registry()
    # Harnesses first: there are few of them and each is drawn in every harness comparison, so they
    # take the pool's clearest filled shapes; packets follow in file order.
    entities = [("harnesses", key) for key in hue_order("harnesses")] + [
        ("packets", key) for key in hue_order("packets")
    ]
    fixed = fixed_packet_markers(reg)
    free = [shape for shape in reg.shapes if shape not in fixed.values() and shape != CONTROL_MARKER]
    unfixed = [entity for entity in entities if entity not in fixed]
    if len(unfixed) > len(free):
        raise ValueError(f"registry: {len(entities)} treatments outgrow the {len(reg.shapes)}-shape pool")
    table = dict(zip(unfixed, free)) | fixed
    return {entity: table[entity] for entity in entities}


def fixed_packet_markers(reg: Registry) -> dict[tuple[str, str], object]:
    """``("packets", key) -> shape`` for every packet whose registry entry names its own ``marker:``;
    raises when two share one or one takes the control's :data:`CONTROL_MARKER`."""
    fixed = {("packets", key): d.marker for key, d in reg.packet_defs.items() if key and d.marker}
    taken = list(fixed.values())
    if CONTROL_MARKER in taken or len(set(taken)) != len(taken):
        raise ValueError(f"registry: packet markers must be distinct and never {CONTROL_MARKER!r}: {fixed}")
    return fixed


def treatment_shape(kind: str, name: str) -> object:
    """``name``'s registered shape among ``kind`` (packets, harnesses); an unregistered one warns and
    takes a stable pool slot by CRC."""
    resolved = canonical(kind, name)
    table = shape_table()
    if (kind, resolved) in table:
        return table[(kind, resolved)]
    LOG.warning("palette: %s %r is not in registry.yaml; using a hash marker", kind, name)
    pool = [shape for shape in registry().shapes if shape != CONTROL_MARKER]
    return pool[zlib.crc32(str(name).encode()) % len(pool)]


def packet_marker(packet: str) -> object:
    """The one SHAPE ``packet`` wears (:func:`shape_table`): its lead part's, or the control's hollow
    circle for no packet. Colour is spent on the model (:func:`model_color`), so shape alone tells
    treatments apart, and no two registered treatments share one."""
    parts = packets.spec_parts(packet)
    if not parts:
        return CONTROL_MARKER
    return treatment_shape("packets", packets.lead(parts))


def harness_marker(harness: str) -> object:
    """The one SHAPE an agent HARNESS wears (:func:`shape_table`), from the same pool as the packets,
    so a harness and a packet in one figure never share a shape."""
    return treatment_shape("harnesses", harness)


def model_markers(models: Iterable[str]) -> dict[str, str]:
    """``{model: marker}`` for one figure."""
    chosen = {m: marker(m) for m in dict.fromkeys(models)}
    seen: dict[str, str] = {}
    for name, shape in chosen.items():
        entity = canonical("models", name)
        if shape in seen and seen[shape] != entity:
            LOG.warning("palette: models %r and %r both draw %r; extend markers", seen[shape], name, shape)
        seen[shape] = entity
    return chosen


def framework_color(name: str) -> str:
    """The one colour a framework wears. Frameworks never share a figure with a packet."""
    return ordered_color("frameworks", name)


def framework_colors(names: Iterable[str]) -> dict[str, str]:
    """``{framework: colour}`` for one figure."""
    return warn_on_collision({n: framework_color(n) for n in dict.fromkeys(names)}, "framework")


def model_color(name: str) -> str:
    """The one colour a model wears: a figure whose ONLY axis is which model ran, or a packet
    efficacy panel (:func:`packet_marker`'s own docstring), where shape is spent on the packet.

    Shape identifies the model everywhere else; this exists because a figure that varies nothing
    else would otherwise draw four series in one grey."""
    return ordered_color("models", name)


def model_shade(name: str, step: int) -> str:
    """The ``step``-th close shade of a model's colour (0 = the colour itself): the rule for ONE
    model drawn several times in one figure -- with and without a packet, on several devices, in
    several pairs -- so the series stay the same model at a glance yet tell apart where their marks
    or intervals overlap. One step is :data:`registry().lightness_step`."""
    return lighten(model_color(name), step)


#: The shade a model's CONTROL (no packet) wears beside its treated setups: one step lighter.
CONTROL_SHADE: int = 1
