# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One colour per entity, decided by its NAME and by nothing else.

Every figure in this repo draws the same things -- frameworks and models -- and a reader carries a
colour between figures whether or not we meant them to. So a colour has to be a property of the
entity, not of the figure: ``plotting.py`` used to index the palette by a framework's position in
whatever list that figure happened to hold, so dropping one column repainted every survivor, and
the same framework wore one hue in the heatmap grid and another in the speed-up chart.

**There is deliberately no mutable registry.** A colour is a pure function of the name, so it is
already the same in every process, on every machine, in every re-run, with no file to write, load,
lock, or fall out of date -- which is a stronger guarantee than "assigned once and remembered", and
it cannot be broken by running two plots concurrently or by shipping a figure without its registry.
Registering an entity means adding its name to the frozen order below.

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
)

#: What an entity IS. Two entities of different kinds may share a hue; two of the same kind in one
#: figure may not, which is what :func:`colors` checks.
KINDS: tuple[str, ...] = ("framework", "model")

ORDERS: dict[str, tuple[str, ...]] = {"framework": FRAMEWORK_ORDER, "model": MODEL_ORDER}


def color(kind: str, name: str) -> str:
    """The one colour ``name`` wears, in every figure and every process.

    A name the order does not know still gets a stable colour rather than an exception: a new
    framework must not break a plot. It is derived from a CRC of the name, not ``hash()``, whose
    value depends on ``PYTHONHASHSEED`` and would hand the same entity different colours in two
    runs of the same script.
    """
    if kind not in ORDERS:
        raise ValueError(f"unknown entity kind {kind!r}; expected one of {KINDS}")
    order = ORDERS[kind]
    try:
        index = order.index(name)
    except ValueError:
        index = len(order) + zlib.crc32(name.encode()) % len(PALETTE)
    return PALETTE[index % len(PALETTE)]


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
