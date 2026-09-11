# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One colour and one shape per entity, decided by its identity and by nothing else.

A reader carries a colour between figures whether or not we meant them to, so a colour has to be a
property of the entity rather than of its position in whatever list one figure happened to hold:
dropping a series must not repaint the survivors.

**Colour is the SKILL PACKET, shape is the MODEL.** There are a few models and there will be many
packets, and colour separates more values than shape does. A packet combination takes its LEAD
packet's hue and one lightness step per additional packet, so ``cpfsrc`` and ``cpfsrc+lang-skills``
read as the same treatment family at two strengths. The no-packet control is neutral grey: it is
the reference every treatment is read against, not one more colour among them.

Hues are Okabe-Ito, which is designed for deuteranopia, protanopia and tritanopia; it is
categorical, so it is the right ramp for identity (viridis is sequential and would imply an order).
"""

from __future__ import annotations

import colorsys
import logging
import zlib
from collections.abc import Iterable

LOG = logging.getLogger(__name__)

#: Okabe-Ito, minus black (reserved for ink) and yellow (illegible on white). Order matters: the
#: packets named earliest below get the most-separated hues. APPEND ONLY.
HUES: tuple[str, ...] = (
    "#0072b2",  # blue
    "#e69f00",  # orange
    "#009e73",  # bluish green
    "#cc79a7",  # reddish purple
    "#d55e00",  # vermillion
    "#56b4e9",  # sky blue
)

#: The no-packet control. Neutral on purpose, and not drawn from HUES.
CONTROL_COLOR = "#4d4d4d"

#: Packets in hue-assignment order, and the order that picks the LEAD of a combination: the lead of
#: ``cpfsrc+lang-skills`` is cpfsrc, so a CPF figure's arms stay one family. APPEND ONLY -- inserting
#: a name repaints every packet after it.
#:
#: The first six take the six hues. Past that the ramp wraps, which is safe only for packets that
#: never share a figure with their twin: `no-score-tool` is its campaign's only packet and
#: `profiling` only ever rides along with divide-and-conquer, so neither is ever a lead beside the
#: packet it wraps onto. :func:`colors` warns if that ever stops being true.
PACKET_ORDER: tuple[str, ...] = (
    "cpfsrc",
    "cpf",
    "lang-skills",
    "divide-and-conquer",
    "openmp-offload",
    "repo",
    "no-score-tool",
    "profiling",
)

#: Models, in shape-assignment order. APPEND ONLY.
MODEL_ORDER: tuple[str, ...] = ("qwen38", "oss120b", "kimi27sglang", "glm53")

#: Marker shapes. A filled circle, square, triangle and diamond stay apart at 4pt where a pentagon
#: and a hexagon do not, so identity survives a column-width figure and a greyscale print.
MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")

#: How much lighter each packet BEYOND the lead makes the colour. Large enough to read as a step,
#: small enough that three steps stay clearly the lead's hue.
LIGHTNESS_STEP = 0.13


def parts(packet: str) -> tuple[str, ...]:
    """The packets in a canonical ``packet`` value; ``()`` for the control."""
    return tuple(p for p in packet.split("+") if p)


def lead(packet: str) -> str:
    """The packet that decides the hue: the earliest of ``packet``'s parts in :data:`PACKET_ORDER`.

    An unregistered part sorts after every registered one, and by name among themselves, so the
    lead is a pure function of the value rather than of the order the parts were written in."""

    def rank(name: str) -> tuple[int, str]:
        return (PACKET_ORDER.index(name), "") if name in PACKET_ORDER else (len(PACKET_ORDER), name)

    found = parts(packet)
    return min(found, key=rank) if found else ""


def lighten(hex_color: str, steps: int) -> str:
    """``hex_color`` moved ``steps`` toward white in HLS, capped short of white so it stays visible."""
    if steps <= 0:
        return hex_color
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    h, lightness, s = colorsys.rgb_to_hls(r, g, b)
    lightness = min(0.88, lightness + steps * LIGHTNESS_STEP)
    r, g, b = colorsys.hls_to_rgb(h, lightness, s)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def hue_of(name: str) -> str:
    """The base hue of one packet NAME. Unregistered names get a stable CRC hue and a warning.

    CRC, never ``hash()``: ``hash`` is salted by PYTHONHASHSEED and would hand the same packet a
    different colour in two runs of the same script."""
    if name in PACKET_ORDER:
        return HUES[PACKET_ORDER.index(name) % len(HUES)]
    LOG.warning("palette: packet %r is not registered in PACKET_ORDER; using a hash colour", name)
    return HUES[zlib.crc32(name.encode()) % len(HUES)]


def color(packet: str) -> str:
    """The one colour ``packet`` wears, in every figure and every process."""
    found = parts(packet)
    if not found:
        return CONTROL_COLOR
    return lighten(hue_of(lead(packet)), len(found) - 1)


def colors(packets: Iterable[str]) -> dict[str, str]:
    """``{packet: colour}`` for one figure, warning when two of its packets share a colour.

    Warned rather than raised: a flagged plot beats no plot, and the fix is another hue or a
    lightness step, not a dropped series."""
    chosen = {p: color(p) for p in dict.fromkeys(packets)}
    seen: dict[str, str] = {}
    for name, hue in chosen.items():
        if hue in seen:
            LOG.warning("palette: packets %r and %r both draw %s; extend HUES", seen[hue], name, hue)
        seen[hue] = name
    return chosen


def marker(model: str) -> str:
    """The one SHAPE ``model`` wears. Pairs with :func:`color` so identity is never colour alone."""
    if model in MODEL_ORDER:
        return MARKERS[MODEL_ORDER.index(model) % len(MARKERS)]
    LOG.warning("palette: model %r is not registered in MODEL_ORDER; using a hash marker", model)
    return MARKERS[zlib.crc32(model.encode()) % len(MARKERS)]


def markers(models: Iterable[str]) -> dict[str, str]:
    """``{model: marker}`` for one figure, warning when two of its models share a shape."""
    chosen = {m: marker(m) for m in dict.fromkeys(models)}
    seen: dict[str, str] = {}
    for name, shape in chosen.items():
        if shape in seen:
            LOG.warning("palette: models %r and %r both draw %r; extend MARKERS", seen[shape], name, shape)
        seen[shape] = name
    return chosen
