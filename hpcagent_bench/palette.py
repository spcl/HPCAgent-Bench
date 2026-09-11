# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One colour per entity, decided by its NAME and by nothing else.

Every figure in this repo draws the same things -- frameworks and models -- and a reader carries a
colour between figures whether or not we meant them to. So a colour has to be a property of the
entity, not of the figure: ``plotting.py`` used to index the palette by a framework's position in
whatever list that figure happened to hold, so dropping one column repainted every survivor, and
the same framework wore one hue in the heatmap grid and another in the speed-up chart.

**Registration assigns both a colour and a MARKER SHAPE**, and it is idempotent: registering a
name that already has a slot returns the slot it has. The orders below are the pre-registered
seed, so every name this repo has ever plotted keeps the identity it already had, and
:func:`register` is how a new one joins.

Shape carries the identity alongside colour because colour alone is not enough. Two series that
survive a CVD check on screen can still be indistinguishable in a greyscale print, in a figure
shrunk to a column width, or to a reader with a display that mangles hue -- and identity encoded
twice costs nothing. It is also the difference between a legend a reader must trust and one they
can verify against the marks.

A colour remains a pure function of the NAME rather than of a figure's series order, which is the
property that matters: dropping a column must not repaint the survivors, and the same framework
must wear one hue in the heatmap and in the speed-up chart.

The orders are APPEND ONLY. Inserting a name shifts every colour after it and silently re-colours
every figure already published.
"""

from __future__ import annotations

import logging
import zlib
from collections.abc import Iterable

LOG = logging.getLogger(__name__)

#: Categorical hues, colourblind-safe and checked for pairwise separation. Order matters: the
#: entities named earliest in the orders below get the most-separated hues.
PALETTE: tuple[str, ...] = (
    "#2a78d6",
    "#e07a2b",
    "#1baf7a",
    "#d64550",
    "#7a5cc0",
    "#b5892b",
    "#4aada6",
    "#c65b9b",
    "#6b8f3a",
    "#8a8a86",
    "#3f6fb0",
    "#c0522b",
)

#: Frameworks, most-plotted first so the common ones get the most distinguishable hues.
FRAMEWORK_ORDER: tuple[str, ...] = (
    "numpy",
    "cc",
    "fortran",
    "cpp",
    "dace_cpu",
    "numba",
    "pythran",
    "llvm",
    "pluto",
    "polly",
    "jax",
    "tvm",
    "cc_autopar",
    "cc_llvm",
    "cc_llvm_autopar",
    "cc_nvhpc",
    "cc_nvhpc_autopar",
    "cc_oneapi",
    "flang",
    "fortran_autopar",
    "dace_cpu_autoopt",
    "dace_cpu_canonicalize",
    "dace_gpu",
    "dace_gpu_autoopt",
    "dace_gpu_canonicalize",
    "cupy",
    "triton",
    "ppcg",
    "ppcg_cuda",
    "ppcg_hip",
    "tvm_cpu",
)

#: Models. A separate order from frameworks on purpose -- they are never series in one chart, so
#: both may use the head of the palette and each gets the well-separated hues.
MODEL_ORDER: tuple[str, ...] = (
    "oss120b",
    "qwen38",
    "kimi27sglang",
    "gpt-oss-120b",
    "qwen3.8",
    "kimi-k2.7",
    # APPEND ONLY -- glm53 goes here rather than beside the other three campaign models, because
    # inserting it there would shift every alias after it onto a different hue.
    "glm53",
    "glm-5.3",
)

#: Marker shapes, in assignment order. Chosen to stay distinct at small sizes and in print: a
#: filled circle, square, triangle and diamond read apart at 4pt where a pentagon and a hexagon do
#: not. Shorter than PALETTE on purpose -- shape repeats sooner than colour, and the PAIR is what
#: identifies a series.
MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")

#: The arm CONDITIONS, in assignment order. A condition is what a figure varies WITHIN a model --
#: no packet, the language packet, the CPF page, the CPF drop-in source -- so it is an entity kind
#: of its own: a figure carrying three of them needs three shapes that are stable across runs and
#: across campaigns, and taking them off the model order would repaint the models.
#: APPEND ONLY, for the same reason the model order is.
CONDITION_ORDER: tuple[str, ...] = ("plain", "skills", "cpf", "cpfsrc")

#: What an entity IS. Two entities of different kinds may share a hue; two of the same kind in one
#: figure may not, which is what :func:`colors` checks.
KINDS: tuple[str, ...] = ("framework", "model", "condition")

#: The live registry: the frozen orders above as a seed, plus whatever :func:`register` adds. A
#: list rather than the tuple so registration can append; the seed prefix is never reordered, so
#: an already-published figure cannot be repainted by a later registration.
ORDERS: dict[str, list[str]] = {
    "framework": list(FRAMEWORK_ORDER),
    "model": list(MODEL_ORDER),
    "condition": list(CONDITION_ORDER),
}


def register(kind: str, name: str) -> tuple[str, str]:
    """Give ``name`` its colour and marker, and return them. Idempotent.

    A name already known keeps the slot it has -- so registering twice, or registering something
    the frozen order already lists, cannot move it. A NEW name is appended, which makes its slot
    depend on registration order; register at import time from a fixed list if two processes must
    agree, or add the name to the order above, which is what "registered" ultimately means.
    """
    if kind not in ORDERS:
        raise ValueError(f"unknown entity kind {kind!r}; expected one of {KINDS}")
    order = ORDERS[kind]
    if name not in order:
        order.append(name)
    return color(kind, name), marker(kind, name)


def slot(kind: str, name: str) -> int:
    """Where ``name`` sits. Registered names get their position; unknown ones a stable CRC slot.

    The fallback is a CRC of the name, never ``hash()``: ``hash`` is salted by PYTHONHASHSEED and
    would hand the same entity different colours in two runs of the same script.
    """
    if kind not in ORDERS:
        raise ValueError(f"unknown entity kind {kind!r}; expected one of {KINDS}")
    order = ORDERS[kind]
    try:
        return order.index(name)
    except ValueError:
        return len(order) + zlib.crc32(name.encode()) % len(PALETTE)


def marker(kind: str, name: str) -> str:
    """The one SHAPE ``name`` wears, everywhere. Pairs with :func:`color` to identify a series."""
    return MARKERS[slot(kind, name) % len(MARKERS)]


def markers(kind: str, names: Iterable[str]) -> dict[str, str]:
    """``{name: marker}`` for one figure."""
    return {name: marker(kind, name) for name in dict.fromkeys(names)}


def color(kind: str, name: str) -> str:
    """The one colour ``name`` wears, in every figure and every process.

    A name the order does not know still gets a stable colour rather than an exception: a new
    framework must not break a plot. It is derived from a CRC of the name, not ``hash()``, whose
    value depends on ``PYTHONHASHSEED`` and would hand the same entity different colours in two
    runs of the same script.
    """
    return PALETTE[slot(kind, name) % len(PALETTE)]


def colors(kind: str, names: Iterable[str]) -> dict[str, str]:
    """``{name: colour}`` for one figure, warning when two of its entities share a hue.

    There are more entities than palette entries, so the order wraps. Harmless until two that wrap
    onto each other appear in the SAME figure, where they read as one series. Warned rather than
    raised: a flagged plot beats no plot, and the fix is another hue, not a dropped series.
    """
    chosen = {name: color(kind, name) for name in dict.fromkeys(names)}
    seen: dict[str, str] = {}
    for name, hue in chosen.items():
        if hue in seen:
            LOG.warning("palette: %s and %s share %s in one figure; extend PALETTE", seen[hue], name, hue)
        seen[hue] = name
    return chosen


def model_of(arm: str, unknown: str = "other") -> str:
    """The model an arm ran. Arms are ``<experiment>-<model>-<language>[-skills]``.

    Lived in three plotting scripts as three copies that had already drifted -- one knew about
    ``glm53`` and the others did not, so the same CSV produced a different set of series depending
    on which figure read it. One copy, keyed on the same order that decides the colour, so a model
    that has a hue is also a model the readers can find.
    """
    for name in MODEL_ORDER:
        if f"-{name}-" in arm or arm.endswith(f"-{name}"):
            return name
    return unknown
