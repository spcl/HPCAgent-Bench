# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The one series look every overlay figure reads: colour from the model, shape from the packet,
the control hollow, the torch.distributed baseline grey. The control case is pinned in
``test_palette.py``; these cover the treated setup and the baseline."""

from hpcagent_bench.stats import palette
from hpcagent_bench.stats.figures import scaling
from hpcagent_bench.stats.figures.helpers import series


def test_a_treated_setup_wears_its_models_full_colour_and_its_packets_filled_shape() -> None:
    packet = palette.hue_order("packets")[0]
    style = series.series_style(packet, "qwen38")
    hue = palette.model_shade("qwen38", 0)
    assert style == {
        "color": hue,
        "marker": palette.packet_marker(packet),
        "markerfacecolor": hue,
        "markeredgecolor": hue,
    }


def test_the_torch_dist_baseline_is_the_control_grey_filled_x_whatever_its_packet() -> None:
    grey = palette.control_color()
    expected = {"color": grey, "marker": series.TORCH_DIST_MARKER, "markerfacecolor": grey, "markeredgecolor": grey}
    assert series.series_style("", series.TORCH_DIST_ARM) == expected
    assert series.series_style("anything", series.TORCH_DIST_ARM) == expected


def test_the_scaling_figures_read_the_shared_series_look_not_a_copy() -> None:
    assert scaling.series_style is series.series_style
    assert scaling.torch_dist_style is series.torch_dist_style
    assert scaling.TORCH_DIST_ARM == series.TORCH_DIST_ARM
