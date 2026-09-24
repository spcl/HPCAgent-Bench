# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The cost-weighting wrap figure: drawn at its placed width, on the shared print type scale."""

import math

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from hpcagent_bench.stats import style
from hpcagent_bench.stats.figures import cost_weighting

PAIRS = (
    ("harness20-qwen38-openhands", "harness20-qwen38-claude"),
    ("gpu-llr-focus40-oss120b-hip-skills", "gpu-llr-focus40-oss120b-hip"),
)


def points() -> pd.DataFrame:
    """rho_C per pair and card, one pair without an interval (too few kernels)."""
    rows = []
    for index, (treated, control) in enumerate(PAIRS):
        for card in cost_weighting.DEFAULT_CARDS:
            low, high = (0.8, 1.6) if index == 0 else (math.nan, math.nan)
            rows.append(
                {
                    "arm_a": treated,
                    "arm_b": control,
                    "card": card,
                    "n": 20,
                    "rho_c": 1.1,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows, columns=list(cost_weighting.COLUMNS))


def test_the_cost_figure_is_as_wide_as_its_wrap_and_prints_on_the_shared_scale() -> None:
    """It sits beside the ML scaling wrap figure; different widths or type would show on the page."""
    labels = {PAIRS[0][0]: "OpenHands (Harness20)", PAIRS[1][0]: "HIP skills (LLR40)"}
    fig = cost_weighting.figure_cost_points(points(), labels)
    assert fig is not None
    try:
        assert float(fig.get_size_inches()[0]) == pytest.approx(style.ICLR_WRAP_WIDTH_IN)
        assert style.print_type_violations(fig) == []
    finally:
        plt.close(fig)


def test_the_key_names_each_model_colour_and_how_many_shades_it_wears() -> None:
    """Colour is the model; a reader must find which hue is which model without the caption."""
    extra = ("cpf-llr-focus40-qwen38-c-cpfsrc-v2", "cpf-llr-focus40-qwen38-c")
    table = pd.concat([points(), points().assign(arm_a=extra[0], arm_b=extra[1])], ignore_index=True)
    fig = cost_weighting.figure_cost_points(table, {})
    assert fig is not None
    try:
        texts = [text.get_text() for legend in fig.legends for text in legend.get_texts()]
        assert "Qwen3.8-27B (2 shades)" in texts and "GPT-OSS-120B" in texts, texts
    finally:
        plt.close(fig)


def test_the_usd_slot_is_explained_in_the_legend() -> None:
    """``USD w_m`` is a per-model price vector, not a weight triple; a note under the key says so."""
    fig = cost_weighting.figure_cost_points(points(), {})
    assert fig is not None
    try:
        assert cost_weighting.USD_NOTE in [text.get_text() for text in fig.texts]
    finally:
        plt.close(fig)
