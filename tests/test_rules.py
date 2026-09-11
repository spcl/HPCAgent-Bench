# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
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


def test_rule_4_passes_a_table_that_reports_both_costs() -> None:
    frame = ratios(baseline_ms=[3.0, 4.0], candidate_ms=[2.0, 2.0])
    assert rules.require_costs(frame, "speedup", ("baseline_ms", "candidate_ms")) is frame


def test_rule_5_refuses_nondeterministic_points_with_no_interval_columns() -> None:
    with pytest.raises(rules.RuleViolation, match="Rule 5"):
        rules.require_interval(ratios(), "speedup", "speedup_low", "speedup_high")


def test_rule_5_refuses_a_table_where_no_plotted_point_has_an_interval() -> None:
    frame = ratios(speedup_low=[float("nan")] * 2, speedup_high=[float("nan")] * 2)
    with pytest.raises(rules.RuleViolation, match="empty interval"):
        rules.require_interval(frame, "speedup", "speedup_low", "speedup_high")


def test_rule_5_allows_a_single_degenerate_row_beside_real_intervals() -> None:
    """One cell with too few repetitions to bound is a fact about the run, not a broken figure."""
    frame = ratios(speedup_low=[1.2, float("nan")], speedup_high=[1.6, float("nan")])
    assert rules.require_interval(frame, "speedup", "speedup_low", "speedup_high") is frame


def test_rule_5_catches_interval_ends_the_wrong_way_round() -> None:
    frame = ratios(speedup_low=[2.0, 1.0], speedup_high=[1.0, 2.0])
    with pytest.raises(rules.RuleViolation, match="wrong way round"):
        rules.require_interval(frame, "speedup", "speedup_low", "speedup_high")


def test_declaring_the_data_deterministic_is_the_only_way_past_rule_5() -> None:
    """The paper's own escape hatch, and it is an ASSERTION about the measurement rather than a
    way around the check -- the caption then has to say so too."""
    assert rules.require_interval(ratios(), "speedup", "low", "high", deterministic=True) is not None


def test_rule_7_only_non_overlap_is_a_statement() -> None:
    assert rules.separated(1.0, 2.0, 3.0, 4.0)
    assert rules.separated(3.0, 4.0, 1.0, 2.0)
    # Overlap implies NOTHING, which is the half readers get wrong; it is reported as "not separated".
    assert not rules.separated(1.0, 3.0, 2.0, 4.0)
    assert not rules.separated(1.0, float("nan"), 2.0, 4.0)


def test_rule_12_refuses_a_line_across_an_axis_with_no_order() -> None:
    """Two conditions are not a trend, and a line between them claims interpolation across a gap
    that does not exist."""
    with pytest.raises(rules.RuleViolation, match="Rule 12"):
        rules.require_ordered_x([0.0, 0.0], connected=True)
    with pytest.raises(rules.RuleViolation, match="Rule 12"):
        rules.require_ordered_x([2.0, 1.0], connected=True)
    with pytest.raises(rules.RuleViolation, match="fewer than two"):
        rules.require_ordered_x([1.0], connected=True)


def test_rule_12_allows_an_ordered_axis_and_any_unconnected_one() -> None:
    rules.require_ordered_x([1.0, 2.0, 8.0], connected=True)
    rules.require_ordered_x([3.0, 1.0, 1.0], connected=False)


def test_a_control_to_treatment_link_is_drawn_as_a_difference() -> None:
    """The segment sits at ONE x and its length is the effect, so there is no horizontal run for
    the eye to extrapolate along."""
    xs, ys = rules.difference_segment(1.0, 1.6, at=3.0)
    assert xs == (3.0, 3.0) and ys == (1.0, 1.6)


def test_every_enforced_rule_quotes_the_paper() -> None:
    """The error message carries the rule's own words and the citation, so a reader who has not
    read the paper still learns what is being asked and where to check it."""
    assert set(rules.RULE_TEXT) == {4, 5, 7, 12}
    assert "SC15" in rules.CITATION and "Hoefler" in rules.CITATION
    message = str(rules.RuleViolation(12, "detail"))
    assert "Only connect measurements by lines if they indicate trends" in message
    assert rules.CITATION in message
