# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One colour and one shape per entity, decided by its identity (its registered ``order``), so dropping a series
never repaints the others. Shape is the MODEL and colour what the figure varies, except the packet efficacy panels
(colour = model, shape = packet). One palette: ``tab20``, extended by ``tab20b``."""

import colorsys
import functools
import logging
import zlib
from collections.abc import Iterable
from typing import NamedTuple

import matplotlib as mpl
from matplotlib.colors import to_hex
from matplotlib.markers import MarkerStyle

from hpcagent_bench import packets
from hpcagent_bench.study_tags import Marker, Registry, canonical, order, registry, slot
from hpcagent_bench.vocabulary import KINDS

__all__ = [
    "CONTROL_MARKER",
    "CONTROL_SHADE",
    "LOG",
    "Look",
    "TAB20",
    "TAB20B",
    "TAB20B_ORDER",
    "TAB20_ORDER",
    "color",
    "colormap_slot",
    "colors",
    "cycled",
    "control_color",
    "fixed_packet_markers",
    "framework_color",
    "framework_marker",
    "harness_marker",
    "hue_order",
    "hues",
    "in_order",
    "darken",
    "lighten",
    "look",
    "marker",
    "marker_style",
    "markers",
    "model_color",
    "model_markers",
    "model_shade",
    "ordered_color",
    "packet_marker",
    "shape_table",
    "slot_color",
    "slot_look",
    "tab20_slot",
    "treatment_shape",
    "warn_on_collision",
]

LOG = logging.getLogger(__name__)

#: The colormap every figure draws from.
TAB20: str = "tab20"

#: tab20 slots in DARK-FIRST order (tab20 is light/dark pairs): the ten saturated colours, then their light twins.
TAB20_ORDER: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19)

#: Slots 21-40 (never wrapping): tab20b, darkest shade of every hue first.
TAB20B: str = "tab20b"
TAB20B_ORDER: tuple[int, ...] = (0, 4, 8, 12, 16, 2, 6, 10, 14, 18, 1, 5, 9, 13, 17, 3, 7, 11, 15, 19)


#: The control's shape, reserved: no packet or harness is ever assigned it, and the control is drawn
#: hollow in its model's colour, so "no packet" reads the same in every figure.
CONTROL_MARKER: str = "o"


def colormap_slot(name: str, slot: int) -> str:
    """Entry ``slot`` of the colormap ``name`` as ``#rrggbb``. The ONE place a colour value enters
    this repo."""
    colormap = mpl.colormaps[name]
    return to_hex(colormap(slot % colormap.N))


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
    """``kind``'s tags in hue order, the control packet dropped (it has its own neutral colour)."""
    return tuple(tag for tag in order(kind) if tag)


class Look(NamedTuple):
    """The colour and marker one entry is drawn with."""

    colour: str
    marker: Marker


def slot_look(slot: int) -> Look:
    """The look of registry slot ``slot``: colour ``hues()[slot % H]``, marker ``markers()[(slot // H) % M]`` for
    ``H`` hues and ``M`` markers. Defined for every non-negative slot, so a new entry never raises or goes grey."""
    ramp, shapes = hues(), markers()
    return Look(ramp[slot % len(ramp)], shapes[(slot // len(ramp)) % len(shapes)])


def look(kind: str, name: str) -> Look:
    """``name``'s look within the registry kind ``kind`` (``"frameworks"``, ``"models"``, ...): its
    :meth:`~hpcagent_bench.registry.Kind.slot` through :func:`slot_look`; the control's grey circle for an entry
    without a slot. An unregistered name takes a stable CRC slot (never ``hash()``, which PYTHONHASHSEED salts)."""
    entries = KINDS.get(kind)
    key = canonical(kind, name)
    if entries is not None and key in entries.orders:
        index = entries.slot(key)
        return slot_look(index) if index is not None else Look(control_color(), CONTROL_MARKER)
    return slot_look(zlib.crc32(str(name).encode()) % (len(hues()) * len(markers())))


def slot_color(kind: str, name: str) -> str:
    """``name``'s colour within ``kind`` (:func:`look`), silently. :func:`ordered_color` is this plus the warning
    for an unregistered name."""
    return look(kind, name).colour


def ordered_color(kind: str, name: str) -> str:
    """``name``'s tab20 slot within one entity ``kind``, warning when the registry does not name it."""
    entries = KINDS.get(kind)
    if entries is None or canonical(kind, name) not in entries.orders:
        LOG.warning("palette: %s %r is not registered; using a hash colour", kind, name)
    return slot_color(kind, name)


def warn_on_collision(chosen: dict[str, str], kind: str) -> dict[str, str]:
    """Pass ``chosen`` through, warning when two entities in ONE figure draw the same colour (the ramp wraps)."""
    seen: dict[str, str] = {}
    for name, hue in chosen.items():
        entity = canonical(f"{kind}s", name)
        if hue in seen and seen[hue] != entity:
            LOG.warning("palette: %s %r and %r both draw %s; extend hues", kind, seen[hue], name, hue)
        seen[hue] = entity
    return chosen


def darken(hex_color: str, steps: int) -> str:
    """``hex_color`` moved ``steps`` toward black in HLS, floored short of black so its hue still reads:
    a statistic drawn over its own series' marks (an interval over the runs)."""
    if steps <= 0:
        return hex_color
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    lightness = max(0.15, lightness - steps * registry().lightness_step)
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return to_hex((r, g, b))


def lighten(hex_color: str, steps: int) -> str:
    """``hex_color`` moved ``steps`` toward white in HLS, capped short of white (a combination's family shade)."""
    if steps <= 0:
        return hex_color
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    lightness = min(0.88, lightness + steps * registry().lightness_step)
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return to_hex((r, g, b))


def color(packet: str) -> str:
    """The one colour ``packet`` wears, in every figure and every process: the control grey for the
    control, otherwise the lead part's tab20 slot lightened one step per extra part.

    Warns for each part the vocabulary does not register, since an unregistered part draws in a hash slot."""
    known = set(hue_order("packets"))
    for part in packets.spec_parts(packet):
        if part not in known:
            LOG.warning("palette: packets %r is not registered; using a hash colour", part)
    parts = packets.spec_parts(packet)
    if not parts:
        return control_color()
    return lighten(slot_color("packets", packets.lead(parts)), len(parts) - 1)


def colors(names: Iterable[str]) -> dict[str, str]:
    """``{packet: colour}`` for one figure."""
    return warn_on_collision({p: color(p) for p in dict.fromkeys(names)}, "packet")


def in_order(names: Iterable[str], kind: str = "models") -> list[str]:
    """``names`` in registry order, unregistered ones last and alphabetical (every figure's series order)."""
    known = order(kind)

    def rank(name: str) -> tuple[int, str]:
        resolved = canonical(kind, name)
        return (known.index(resolved), "") if resolved in known else (len(known), str(name))

    return sorted(dict.fromkeys(names), key=rank)


def marker(model: str) -> str:
    """The one SHAPE an optimizer wears: models from the FRONT of the sequence by slot, standalone optimizers from
    the BACK (so a new model never reshapes them); an unregistered name warns and takes a stable CRC shape."""
    shapes = markers()
    index = slot("models", model)
    if index is not None:
        return shapes[index % len(shapes)]
    index = slot("optimizers", model)
    if index is not None:
        return shapes[-1 - (index % len(shapes))]
    LOG.warning("palette: optimizer %r is not registered; using a hash marker", model)
    return shapes[zlib.crc32(str(model).encode()) % len(shapes)]


def framework_marker(name: str) -> Marker:
    """The marker a framework column wears: a standalone optimizer's shape (:func:`marker`) when the column is one,
    else its slot's (:func:`look`), so a column registered at runtime or by a plugin needs no plot-code edit."""
    if canonical("optimizers", name) in KINDS["optimizers"].keys():
        return marker(name)
    return look("frameworks", name).marker


@functools.lru_cache(maxsize=1, typed=True)
def shape_table() -> dict[tuple[str, str], Marker]:
    """``(kind, key) -> shape`` for every registered treatment from one pool: the first ``harness_shapes`` free
    shapes are the harnesses', the rest the packets' (slot order unless a packet names ``marker:``). The control's
    circle is never handed out; an overfull part wraps with a warning."""
    reg = registry()
    fixed = fixed_packet_markers(reg)
    free = [shape for shape in reg.shapes if shape not in fixed.values() and shape != CONTROL_MARKER]
    harness_part, packet_part = free[: reg.harness_shapes], free[reg.harness_shapes :]
    harnesses = [("harnesses", key) for key in hue_order("harnesses")]
    packets_left = [("packets", key) for key in hue_order("packets") if ("packets", key) not in fixed]
    if len(harnesses) > len(harness_part) or len(packets_left) > len(packet_part):
        LOG.warning(
            "palette: %d harnesses share %d harness shapes and %d packets %d packet shapes; extend shapes",
            len(harnesses), len(harness_part), len(packets_left), len(packet_part),
        )  # fmt: skip
    return cycled(harnesses, harness_part or free) | cycled(packets_left, packet_part or free) | fixed


def cycled(keys: list[tuple[str, str]], shapes: list[Marker]) -> dict[tuple[str, str], Marker]:
    """``keys`` matched to ``shapes`` in order, wrapping when there are more keys than shapes."""
    return {key: shapes[index % len(shapes)] for index, key in enumerate(keys)} if shapes else {}


def fixed_packet_markers(reg: Registry) -> dict[tuple[str, str], Marker]:
    """``("packets", key) -> shape`` for every packet whose registry entry names its own ``marker:``;
    raises when two share one or one takes the control's :data:`CONTROL_MARKER`."""
    fixed: dict[tuple[str, str], Marker] = {
        ("packets", key): d.marker for key, d in reg.packet_defs.items() if key and d.marker
    }
    taken = list(fixed.values())
    if CONTROL_MARKER in taken or len(set(taken)) != len(taken):
        raise ValueError(f"registry: packet markers must be distinct and never {CONTROL_MARKER!r}: {fixed}")
    return fixed


def treatment_shape(kind: str, name: str) -> Marker:
    """``name``'s registered shape among ``kind`` (packets, harnesses); an unregistered one warns and
    takes a stable pool slot by CRC."""
    resolved = canonical(kind, name)
    table = shape_table()
    if (kind, resolved) in table:
        return table[(kind, resolved)]
    LOG.warning("palette: %s %r is not registered; using a hash marker", kind, name)
    pool = [shape for shape in registry().shapes if shape != CONTROL_MARKER]
    return pool[zlib.crc32(str(name).encode()) % len(pool)]


def marker_style(marker: Marker) -> MarkerStyle:
    """A registry marker as the object matplotlib draws: a code, or a ``(sides, style, angle)`` polygon."""
    if isinstance(marker, str):
        return MarkerStyle(marker)
    # The stubs type the angle as int; matplotlib accepts any float degrees.
    return MarkerStyle(marker)  # pyright: ignore[reportArgumentType]


def packet_marker(packet: str) -> Marker:
    """The one SHAPE ``packet`` wears (:func:`shape_table`): its lead part's, or the control's hollow
    circle for no packet. Colour is spent on the model (:func:`model_color`), so shape alone tells
    treatments apart, and no two registered treatments share one."""
    parts = packets.spec_parts(packet)
    if not parts:
        return CONTROL_MARKER
    return treatment_shape("packets", packets.lead(parts))


def harness_marker(harness: str) -> Marker:
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


def model_color(name: str) -> str:
    """The one colour a model wears (a model-only figure, or a packet efficacy panel where shape is the packet)."""
    return ordered_color("models", name)


def model_shade(name: str, step: int) -> str:
    """The ``step``-th close shade of a model's colour (0 = itself), for one model drawn several times in one
    figure; one step is :data:`registry().lightness_step`."""
    return lighten(model_color(name), step)


#: The shade a model's CONTROL (no packet) wears beside its treated setups: one step lighter.
CONTROL_SHADE: int = 1
