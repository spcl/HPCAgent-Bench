# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The cost-weighting wrap figure: drawn at its placed width, on the shared print type scale."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms
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
        for card in (*cost_weighting.TOKEN_CARDS, cost_weighting.Card.USD):
            low, high = (0.8, 1.6) if index == 0 else (math.nan, math.nan)
            rows.append(
                {
                    "arm_a": treated,
                    "arm_b": control,
                    "card": card.value,
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


def test_the_key_lists_one_row_per_pair_and_no_model_swatches() -> None:
    """USER 2026-09-25: the model is the hue and the caption names it; a swatch row per model
    doubled a key that sits in a 2.1in wrap."""
    extra = ("cpf-llr-focus40-qwen38-c-cpfsrc-v2", "cpf-llr-focus40-qwen38-c")
    table = pd.concat([points(), points().assign(arm_a=extra[0], arm_b=extra[1])], ignore_index=True)
    labels = {PAIRS[0][0]: "OpenHands", PAIRS[1][0]: "HIP skills", extra[0]: "CPF"}
    fig = cost_weighting.figure_cost_points(table, labels)
    assert fig is not None
    try:
        texts = [text.get_text() for legend in fig.legends for text in legend.get_texts()]
        assert sorted(texts) == ["CPF", "HIP skills", "OpenHands"], texts
    finally:
        plt.close(fig)


def test_every_slot_is_ticked_by_its_weighting_name() -> None:
    """Short weight ticks were unreadable in the wrap; one row per weighting carries its full name, and
    the paper text gives each weight vector, so the figure carries no note."""
    fig = cost_weighting.figure_cost_points(points(), {})
    assert fig is not None
    try:
        ticks = [label.get_text() for label in fig.axes[0].get_yticklabels()]
        assert ticks == ["Effective", "Billed", "Total", "USD"][: len(ticks)], ticks
        assert fig.texts == []
    finally:
        plt.close(fig)


def test_the_axes_ink_fills_the_placed_width() -> None:
    """USER 2026-09-25: no empty band left of the Y label; the ink runs to the pad on both sides."""
    fig = cost_weighting.figure_cost_points(points(), {})
    assert fig is not None
    try:
        renderer = fig.canvas.get_renderer()
        ink = matplotlib.transforms.Bbox.union([ax.get_tightbbox(renderer) for ax in fig.axes])
        width = float(fig.get_size_inches()[0])
        assert ink.x0 / fig.dpi == pytest.approx(style.PLACED_SIDE_PAD_IN, abs=0.01)
        assert width - ink.x1 / fig.dpi == pytest.approx(style.PLACED_SIDE_PAD_IN, abs=0.01)
    finally:
        plt.close(fig)


def test_the_usd_slot_is_drawn_only_when_every_model_has_a_price_card(tmp_path: Path) -> None:
    """USER 2026-09-25: four slots when every model in the figure has a ``usd-<model>`` card,
    three otherwise -- a dollar slot for some models only would compare a subset of the marks."""
    priced = [cost_weighting.Pair(*PAIRS[0]), cost_weighting.Pair(*PAIRS[1])]
    assert cost_weighting.figure_cards(priced) == (*cost_weighting.TOKEN_CARDS, cost_weighting.Card.USD)
    unpriced = [*priced, cost_weighting.Pair("harness20-unpriced99-openhands", "harness20-unpriced99-claude")]
    assert cost_weighting.figure_cards(unpriced) == cost_weighting.TOKEN_CARDS
    extra = tmp_path / "cards.yaml"
    card = cost_weighting.usd_card_name(unpriced[-1].model)
    extra.write_text(f"{card}:\n  fresh_input: 1.0\n  cached_input: 0.1\n  output: 2.0\n", encoding="utf-8")
    assert cost_weighting.Card.USD in cost_weighting.figure_cards(unpriced, extra)


def test_a_three_slot_figure_has_no_usd_tick() -> None:
    """The dollar tick goes with the USD slot."""
    table = points()
    table = table[table.card != cost_weighting.Card.USD.value]
    fig = cost_weighting.figure_cost_points(table, {})
    assert fig is not None
    try:
        ticks = [label.get_text() for label in fig.axes[0].get_yticklabels()]
        assert cost_weighting.TICKS[cost_weighting.Card.USD] not in ticks and len(ticks) == 3, ticks
    finally:
        plt.close(fig)


def test_pairs_come_from_a_toml_file_and_the_command_line(tmp_path: Path) -> None:
    """A list of model + intervention pairs is data, not a command line: the script reads [[pair]] tables."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "plot_cost_weighting", pathlib.Path(__file__).parents[1] / "statistics" / "plot_cost_weighting.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    toml = tmp_path / "pairs.toml"
    toml.write_text('[[pair]]\ntreated = "a-qwen38-x"\ncontrol = "a-qwen38-y"\nlabel = "X"\n', encoding="utf-8")
    assert script.pairs_file(toml) == [cost_weighting.Pair("a-qwen38-x", "a-qwen38-y", "X")]
    assert script.parse_pair("t,c,Label, with comma") == cost_weighting.Pair("t", "c", "Label, with comma")


def test_rho_c_is_control_over_treated_on_the_shared_kernels() -> None:
    """Control spends 200 on k1-k6; treated 100 on k1-k5 and 400 on k6: rho_C = GM(2 x5, 0.5)
    = 2^(4/6) = 1.5874 (above 1: the treatment is cheaper); treated/control would read 0.63."""
    pair = cost_weighting.Pair("treated-arm", "control-arm")
    tokens = {("control-arm", f"k{i}"): 200.0 for i in range(1, 7)}
    tokens |= {("treated-arm", f"k{i}"): 100.0 for i in range(1, 6)} | {("treated-arm", "k6"): 400.0}

    row = cost_weighting.ratio_row(tokens, pair, "billed")

    assert row["n"] == 6
    assert row["rho_c"] == pytest.approx(2.0 ** (4.0 / 6.0))
    assert row["ci_low"] < row["rho_c"] < row["ci_high"]


def test_each_card_prices_the_same_tasks_into_its_own_rho_c() -> None:
    """Both arms read 100 fresh tokens per kernel; the control also re-reads 1000 cached. Billed
    (1, 0.1, 1): 200 / 100 = 2; effective (1, 0, 1): 100 / 100 = 1; total (1, 1, 1): 1100 / 100 = 11."""
    rows = [
        {
            "run_root": "r",
            "job": "j",
            "run_id": f"{arm}-{kernel}",
            "arm": arm,
            "benchmark": kernel,
            "record": "task",
            "ts_ms": 1,
            "tokens": 100.0,
            "tokens_fresh_input": 100.0,
            "tokens_cached_input": cached,
            "tokens_output": 0.0,
        }  # fmt: skip
        for arm, cached in (("harness20-qwen38-openhands", 0.0), ("harness20-qwen38-claude", 1000.0))
        for kernel in ("k1", "k2", "k3")
    ]

    table = cost_weighting.pair_cost_ratios(pd.DataFrame(rows), [PAIRS[0]], cards=("billed", "effective", "total"))

    assert table.set_index("card").rho_c.to_dict() == pytest.approx({"billed": 2.0, "effective": 1.0, "total": 11.0})
