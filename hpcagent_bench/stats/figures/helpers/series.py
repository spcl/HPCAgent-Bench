# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A series' look from the registry: COLOUR is the model, SHAPE the packet, the control hollow.

One (packet, model) pair is one series in every figure module that overlays setups; this is the one
place that turns the pair into matplotlib style keywords (``color``, ``marker`` and the two marker
faces), so two figures never disagree on how a setup looks."""

from hpcagent_bench.stats import palette

__all__ = ["TORCH_DIST_ARM", "TORCH_DIST_MARKER", "series_style", "torch_dist_style"]

#: The pseudo-arm (and model) of the torch.distributed baseline curve's rows.
TORCH_DIST_ARM: str = "torch_dist"
TORCH_DIST_MARKER: str = "x"


def torch_dist_style() -> dict[str, object]:
    """The torch.distributed baseline curve's look: the control's grey, its own marker, filled."""
    grey = palette.control_color()
    return {"color": grey, "marker": TORCH_DIST_MARKER, "markerfacecolor": grey, "markeredgecolor": grey}


def series_style(packet: str, model: str) -> dict[str, object]:
    """Colour from the model, shape from the packet; the control's mark is hollow. The
    torch.distributed baseline (:data:`TORCH_DIST_ARM`) wears :func:`torch_dist_style`."""
    if model == TORCH_DIST_ARM:
        return torch_dist_style()
    # Two setups of one model share its hue; the control takes a lighter shade so their marks and
    # intervals stay apart where they overlap.
    hue = palette.model_shade(model, 0 if packet else palette.CONTROL_SHADE)
    face = hue if packet else "none"
    return {"color": hue, "marker": palette.packet_marker(packet), "markerfacecolor": face, "markeredgecolor": hue}
