# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The one visual style every figure in this repo is drawn in.

Figures from one project that do not look like one project make a reader work out, per figure,
what is ink and what is data. This fixes the parts that are never data -- type sizes, tick and
spine weight, grid colour, the neutral inks -- so a plot only has to decide what it is actually
showing. Colour is NOT here: it belongs to the entity, and :mod:`hpcagent_bench.palette` owns it.

Neutrals carry a slight cool bias rather than being a pure grey, so they sit under the palette's
blues without looking like a different rendering of the page.
"""

from __future__ import annotations

import textwrap

import matplotlib

#: Ink, in decreasing emphasis. Text NEVER takes a series colour: a coloured mark beside a label
#: carries the identity, and a coloured label just makes the text harder to read.
INK: str = "#1c1c1e"
MUTED: str = "#6b6b70"
RULE: str = "#d6d6da"
#: The zero/parity reference. Darker than the grid because it is a statement, not a guide.
REFERENCE: str = "#3a3a3e"

#: Type scale, in points.
TITLE_PT: float = 12.0
SUBTITLE_PT: float = 8.5
LABEL_PT: float = 9.0
TICK_PT: float = 8.0
ANNOTATION_PT: float = 7.5


def apply() -> None:
    """Set the process-wide rcParams. Idempotent; call it before creating a figure."""
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": RULE,
            "axes.labelcolor": MUTED,
            "axes.labelsize": LABEL_PT,
            "axes.titlesize": LABEL_PT + 1,
            "axes.titlecolor": INK,
            "axes.grid": False,  # each plot opts in on ONE axis; a full grid is noise
            "axes.axisbelow": True,  # data over guides, never the reverse
            "grid.color": RULE,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": TICK_PT,
            "ytick.labelsize": TICK_PT,
            "legend.frameon": False,
            "legend.fontsize": TICK_PT,
            "text.color": INK,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,  # embed as TrueType so the PDF's text stays selectable
            "ps.fonttype": 42,
        }
    )


def despine(ax, keep: tuple[str, ...] = ("left", "bottom")) -> None:
    """Drop the spines that only box the data in."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
        if side in keep:
            ax.spines[side].set_color(RULE)


#: Characters per subtitle line before it wraps. A subtitle that runs the full width of a wide
#: figure is a paragraph, and nobody reads a paragraph above a chart.
SUBTITLE_WRAP: int = 110


def title(fig, text: str, subtitle: str = "") -> float:
    """Left-aligned title with the how-to-read sentence under it; returns the top of the plot area.

    The caller must pass that value to ``tight_layout(rect=...)``. Placing both at fixed fractions
    was wrong for any figure that was not one specific height: on a short figure the subtitle
    landed ON the title, and on a tall one it floated halfway to the axes.
    """
    lines = textwrap.wrap(subtitle, SUBTITLE_WRAP) if subtitle else []
    height = fig.get_size_inches()[1]
    # Work in inches, then convert: a fraction of a 4-inch figure is a different gap than the same
    # fraction of a 12-inch one, which is what made the fixed offsets collide.
    top = 1.0 - (0.30 / height)
    fig.text(0.01, top, text, fontsize=TITLE_PT, color=INK, ha="left", va="top")
    cursor = top - (0.26 / height)
    for line in lines:
        fig.text(0.01, cursor, line, fontsize=SUBTITLE_PT, color=MUTED, ha="left", va="top")
        cursor -= 0.17 / height
    return max(0.5, cursor - 0.12 / height)
