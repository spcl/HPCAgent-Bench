# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The SC15 benchmarking rules, as checks a figure cannot get past silently.

From Hoefler and Belli, "Scientific Benchmarking of Parallel Computing Systems", SC15. A rule that
lives in a style guide is a rule a figure breaks without anyone noticing: the plot still renders and
the numbers still look plausible. These pin that each rule fails loudly, and that it fails on the
thing it is actually about rather than on the absence of data.
"""

import pandas as pd
import pytest

from hpcagent_bench.stats import rules


def ratios(**extra: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"speedup": [1.4, 2.0], **extra})


def test_rule_4_refuses_a_ratio_with_no_costs_behind_it() -> None:
    """1.4x on a 3 ms kernel and 1.4x on a 3 s one are different results, and the ratio cannot tell
    them apart."""
    with pytest.raises(rules.RuleViolation, match="Rule 4"):
        rules.require_costs(ratios(), "speedup", ("baseline_ms", "candidate_ms"))


def test_rule_4_refuses_a_cost_column_that_is_entirely_empty() -> None:
    """A declared but never-filled cost column is the same failure wearing a header."""
    frame = ratios(baseline_ms=[float("nan"), float("nan")], candidate_ms=[1.0, 2.0])
    with pytest.raises(rules.RuleViolation, match="entirely missing"):
        rules.require_costs(frame, "speedup", ("baseline_ms", "candidate_ms"))


def test_rule_5_refuses_nondeterministic_points_with_no_interval_columns() -> None:
    with pytest.raises(rules.RuleViolation, match="Rule 5"):
        rules.require_interval(ratios(), "speedup", "speedup_low", "speedup_high")


def test_rule_5_refuses_a_table_where_no_plotted_point_has_an_interval() -> None:
    frame = ratios(speedup_low=[float("nan")] * 2, speedup_high=[float("nan")] * 2)
    with pytest.raises(rules.RuleViolation, match="empty interval"):
        rules.require_interval(frame, "speedup", "speedup_low", "speedup_high")


def test_rule_5_passes_rows_whose_interval_was_withheld_for_too_few_values() -> None:
    """Below ``summary.MIN_INTERVAL_SAMPLES`` values the interval is withheld and the values drawn as dots; a row
    with enough values and no interval is still refused."""
    nan = float("nan")
    frame = ratios(speedup_low=[nan, nan], speedup_high=[nan, nan], n=[3, 5])
    assert rules.require_interval(frame, "speedup", "speedup_low", "speedup_high") is frame
    frame = ratios(speedup_low=[nan, nan], speedup_high=[nan, nan], n=[3, 6])
    with pytest.raises(rules.RuleViolation, match="empty interval"):
        rules.require_interval(frame, "speedup", "speedup_low", "speedup_high")


def test_rule_5_catches_interval_ends_the_wrong_way_round() -> None:
    frame = ratios(speedup_low=[2.0, 1.0], speedup_high=[1.0, 2.0])
    with pytest.raises(rules.RuleViolation, match="wrong way round"):
        rules.require_interval(frame, "speedup", "speedup_low", "speedup_high")


def test_declaring_the_data_deterministic_is_the_only_way_past_rule_5() -> None:
    """The paper's own escape hatch, and it is an ASSERTION about the measurement rather than a
    way around the check -- the caption then has to say so too."""
    assert rules.require_interval(ratios(), "speedup", "low", "high", deterministic=True) is not None
