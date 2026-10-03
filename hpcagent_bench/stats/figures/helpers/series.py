# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A series' look from the registry: COLOUR is the model, SHAPE the packet, the control hollow.

One (packet, model) pair is one series in every figure module that overlays setups; this is the one
place that turns the pair into matplotlib style keywords (``color``, ``marker`` and the two marker
faces), so two figures never disagree on how a setup looks."""

from typing import TypedDict

from matplotlib.markers import MarkerStyle

from hpcagent_bench.stats import palette

__all__ = ["TORCH_DIST_MARKER", "TORCH_DIST_SETUP", "SeriesStyle", "series_style", "torch_dist_style"]

#: The pseudo-setup (and model) of the torch.distributed baseline curve's rows.
TORCH_DIST_SETUP: str = "torch_dist"
TORCH_DIST_MARKER: str = "x"


class SeriesStyle(TypedDict):
    """The matplotlib keywords one series is drawn with."""

    color: str
    marker: MarkerStyle
    markerfacecolor: str
    markeredgecolor: str


def torch_dist_style() -> SeriesStyle:
    """The torch.distributed baseline curve's look: the control's grey, its own marker, filled."""
    grey = palette.control_color()
    return {"color": grey, "marker": MarkerStyle(TORCH_DIST_MARKER), "markerfacecolor": grey, "markeredgecolor": grey}


def series_style(packet: str, model: str) -> SeriesStyle:
    """Colour from the model, shape from the packet; the control's mark is hollow. The
    torch.distributed baseline (:data:`TORCH_DIST_SETUP`) wears :func:`torch_dist_style`."""
    if model == TORCH_DIST_SETUP:
        return torch_dist_style()
    # Two setups of one model share its hue; the control takes a lighter shade so their marks and
    # intervals stay apart where they overlap.
    hue = palette.model_shade(model, 0 if packet else palette.CONTROL_SHADE)
    face = hue if packet else "none"
    return {
        "color": hue,
        "marker": palette.marker_style(palette.packet_marker(packet)),
        "markerfacecolor": face,
        "markeredgecolor": hue,
    }
