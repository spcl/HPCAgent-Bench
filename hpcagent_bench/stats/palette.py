# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One colour and one shape per entity, decided by its identity and by nothing else.

A reader carries a colour between figures whether or not we meant them to, so a colour has to be a
property of the entity rather than of its position in whatever list one figure happened to hold:
dropping a series must not repaint the survivors.

**Shape is always the MODEL. Colour is whatever the figure VARIES.** There are a few models and
there will be many packets, and colour separates more values than shape does, so the model takes
the shape and keeps it everywhere.

Four things get coloured, and a figure varies exactly one of them, so they never compete for the
ramp: an INTERVENTION (a skill packet, or a scope such as `kernel`/`repo`/`no-score`), a HARNESS (an
agent-harness comparison), a FRAMEWORK (a compiler/library comparison, which has no agent in it), or
a MODEL (a figure whose only axis is which LLM ran).

ONE GLOBAL PALETTE: matplotlib's ``tab20``, and nothing else. Every entity a figure colours --
packet, framework, model, harness, language -- takes a tab20 SLOT decided by its position in
``envs/registry.yaml``, so a colour is looked up in exactly one table and no figure, script or
registry entry carries a hex literal of its own. The ramp it used to carry was hand-picked
Okabe-Ito with per-packet hex overrides bolted on wherever six hues wrapped, which is two palettes
pretending to be one.

A packet's colour is :func:`color`: its LEAD packet's slot and one lightness step per additional
packet, so ``cpfsrc`` and ``cpfsrc+lang-skills`` read as the same treatment family at two
strengths, and neutral grey for the no-packet control.

THE VOCABULARY AND THE ORDER ARE DATA, in ``envs/registry.yaml``, beside the display names. They
were tuples here and names there, which is two registries for one vocabulary -- and the failure
mode is silent, because a packet missing from one of them still draws, in a hash colour, under a
raw-string label.
"""

import colorsys
import logging
import zlib
from collections.abc import Iterable

import matplotlib
import matplotlib.colors

from hpcagent_bench import packets
from hpcagent_bench.experiment_tags import canonical, order, registry

LOG = logging.getLogger(__name__)

#: The colormap every figure in this repo draws from. The user's global palette decision.
TAB20: str = "tab20"

#: tab20 slots in DARK-FIRST order: its ten saturated slots, then their ten light twins. tab20 is
#: laid out as light/dark PAIRS, so reading it straight through would spend the second colour of a
#: figure on a pale wash of its first; taken this way the entities plotted most take the ten
#: colours that stay apart at 4pt, and the light twin of a hue only comes back once the dark ones
#: are spent. Twenty slots for twenty packets, so nothing wraps and no entity needs an override.
TAB20_ORDER: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19)


def tab20_slot(slot: int) -> str:
    """tab20 entry ``slot`` as ``#rrggbb``. The ONE place a colour value enters this repo."""
    colormap = matplotlib.colormaps[TAB20]
    return matplotlib.colors.to_hex(colormap(slot % colormap.N))


def hues() -> tuple[str, ...]:
    """The categorical ramp: tab20 in :data:`TAB20_ORDER`."""
    return tuple(tab20_slot(slot) for slot in TAB20_ORDER)


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
    (``optimizers``, e.g. DaCe or CPF), which take the shapes after the models. Pairs with
    :func:`color` so identity is never colour alone."""
    shapes = markers()
    models = order("models")
    resolved = canonical("models", model)
    if resolved in models:
        return shapes[models.index(resolved) % len(shapes)]
    standalone = order("optimizers")
    resolved = canonical("optimizers", model)
    if resolved in standalone:
        return shapes[(len(models) + standalone.index(resolved)) % len(shapes)]
    LOG.warning("palette: optimizer %r is not in registry.yaml; using a hash marker", model)
    return shapes[zlib.crc32(str(model).encode()) % len(shapes)]


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
    """The one colour a model wears in a figure whose ONLY axis is which model ran.

    Shape identifies the model in every figure; this exists because a figure that varies nothing
    else would otherwise draw four series in one grey."""
    return ordered_color("models", name)


def model_colors(names: Iterable[str]) -> dict[str, str]:
    """``{model: colour}`` for one figure."""
    return warn_on_collision({n: model_color(n) for n in dict.fromkeys(names)}, "model")


def harness_color(name: str) -> str:
    """The one colour an agent HARNESS wears. A harness figure varies the harness and the model,
    so the harness takes the colour channel a packet figure spends on the packet."""
    return ordered_color("harnesses", name)


def harness_colors(names: Iterable[str]) -> dict[str, str]:
    """``{harness: colour}`` for one figure."""
    return warn_on_collision({n: harness_color(n) for n in dict.fromkeys(names)}, "harness")


def language_colors(names: Iterable[str]) -> dict[str, str]:
    """``{language: colour}`` for a figure whose ONLY axis is which language an arm asked for."""
    return warn_on_collision({n: ordered_color("languages", n) for n in dict.fromkeys(names)}, "language")
