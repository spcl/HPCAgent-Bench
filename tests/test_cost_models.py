# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cost cards price a task from its recorded token components, never from engine cache hits."""

import math
import pathlib

import pandas as pd
import pytest

from hpcagent_bench.stats import cost, population


def task_rows(fresh: float, cached: float, output: float) -> pd.DataFrame:
    """One task row and one judge row, the shape an extraction writes."""
    return pd.DataFrame(
        {
            "record": [population.TASK_RECORD, "submission"],
            "tokens": [fresh + output, 5.0],
            "tokens_fresh_input": [fresh, None],
            "tokens_cached_input": [cached, None],
            "tokens_output": [output, None],
        }
    )


def test_the_paper_cards_weight_cache_reads_and_output_as_stated() -> None:
    cards = cost.shipped_cards()
    assert (cards["effective"].fresh_input, cards["effective"].cached_input, cards["effective"].output) == (1, 0, 1)
    assert (cards["billed"].fresh_input, cards["billed"].cached_input, cards["billed"].output) == (1, 0.1, 1)
    assert (cards["api-priced"].cached_input, cards["api-priced"].output) == (0.1, 5)
    assert (cards["total"].cached_input, cards["total"].output) == (1, 1)


def test_billed_card_charges_a_tenth_of_every_re_sent_prefix() -> None:
    priced = cost.priced(task_rows(fresh=1000.0, cached=20000.0, output=300.0), cost.resolve("billed"))
    assert priced["tokens"].iloc[0] == pytest.approx(1000 + 2000 + 300)


def test_api_priced_card_weights_output_five_times() -> None:
    priced = cost.priced(task_rows(fresh=1000.0, cached=20000.0, output=300.0), cost.resolve("api-priced"))
    assert priced["tokens"].iloc[0] == pytest.approx(1000 + 2000 + 1500)


def test_a_judge_row_keeps_its_own_tokens() -> None:
    priced = cost.priced(task_rows(fresh=1.0, cached=1.0, output=1.0), cost.resolve("total"))
    assert priced["tokens"].iloc[1] == pytest.approx(5.0)


def test_effective_card_leaves_the_frame_untouched_even_without_components() -> None:
    frame = pd.DataFrame({"record": [population.TASK_RECORD], "tokens": [42.0]})
    assert cost.priced(frame, cost.resolve("effective")) is frame


def test_a_card_that_needs_missing_components_refuses_instead_of_mispricing() -> None:
    frame = pd.DataFrame({"record": [population.TASK_RECORD], "tokens": [42.0]})
    with pytest.raises(ValueError, match="re-extract"):
        cost.priced(frame, cost.resolve("billed"))


def test_inline_weights_are_a_card() -> None:
    card = cost.resolve("fresh_input=1,cached_input=0.25,output=4")
    assert (card.fresh_input, card.cached_input, card.output) == (1.0, 0.25, 4.0)


def test_a_user_file_adds_cards_and_a_bad_weight_is_refused(tmp_path: pathlib.Path) -> None:
    good = tmp_path / "cards.yaml"
    good.write_text("kimi:\n  name: Kimi List Price\n  fresh_input: 1\n  cached_input: 0.17\n  output: 4.2\n")
    assert cost.resolve("kimi", good).output == pytest.approx(4.2)
    bad = tmp_path / "bad.yaml"
    bad.write_text("broken:\n  fresh_input: 1\n  cached_input: -1\n  output: 1\n")
    with pytest.raises(ValueError, match="non-negative cached_input"):
        cost.resolve("broken", bad)
    with pytest.raises(ValueError, match="unknown cost model"):
        cost.resolve("no-such-card")


def test_a_missing_component_prices_the_task_as_no_measurement() -> None:
    """R7: a task row that does not state a component cannot be priced; NaN, never its raw
    ``tokens`` passed off as a billed total. The judge row keeps its own count."""
    frame = task_rows(fresh=1.0, cached=1.0, output=1.0)
    frame.loc[0, "tokens_output"] = math.nan
    priced = cost.priced(frame, cost.resolve("billed"))
    assert math.isnan(priced["tokens"].iloc[0])
    assert priced["tokens"].iloc[1] == pytest.approx(5.0)


def card_price(card: str, fresh: float, cached: float, output: float) -> float:
    """One task's tokens under the shipped ``card``."""
    return float(cost.priced(task_rows(fresh, cached, output), cost.resolve(card))["tokens"].iloc[0])


def test_the_fold_and_the_cards_agree_on_every_proxy(tmp_path: pathlib.Path) -> None:
    """``experiments/token_cost.py`` ships stdlib-only inside the agent image and so spells its three
    readings inline; this pins them to the cards, so the two definitions cannot drift apart."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "token_cost_for_cards", pathlib.Path(__file__).resolve().parents[1] / "experiments" / "token_cost.py"
    )
    assert spec is not None and spec.loader is not None
    token_cost = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(token_cost)
    usage = tmp_path / "usage.jsonl"
    usage.write_text(
        '{"input": 900, "cached_input": 0, "output": 40, "reasoning": 10}\n'
        '{"input": 300, "cached_input": 900, "output": 60, "reasoning": 0}\n',
        encoding="utf-8",
    )
    row = token_cost.usage_episode_cost(usage)
    parts = (float(row["fresh_input"]), float(row["cached_input"]), float(row["output"]))
    assert row["effective"] == pytest.approx(card_price("effective", *parts))
    assert row["effective_provider"] == pytest.approx(card_price("billed", *parts))
    assert row["naive_total"] == pytest.approx(card_price("total", *parts))
