# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Multiplicity corrections: Holm and Benjamini-Hochberg against hand-worked values."""

import math

import pytest

from hpcagent_bench.harness import efficacy
from hpcagent_bench.stats import inference


def test_adjusted_pvalues_never_shrink_below_the_raw_ones() -> None:
    raw = [0.001, 0.01, 0.03, 0.2, 0.5, 0.9]
    for method in ("fdr_bh", "holm"):
        adjusted = inference.adjust_pvalues(raw, method=method)
        assert all(a >= p - 1e-12 for a, p in zip(adjusted, raw)), method
        assert all(0.0 <= a <= 1.0 for a in adjusted), method


def test_holm_matches_the_textbook_step_down_values() -> None:
    """Worked by hand: n=4, sorted p = .01 .02 .03 .04 -> .04 .06 .06 .06 after monotonicity."""
    assert inference.adjust_pvalues([0.01, 0.02, 0.03, 0.04], method="holm") == pytest.approx([0.04, 0.06, 0.06, 0.06])


def test_benjamini_hochberg_matches_the_step_up_values_by_hand() -> None:
    """m = 4, p = .01 .02 .03 .04: p * m / rank = .04 each, so every q is .04."""
    assert inference.adjust_pvalues([0.01, 0.02, 0.03, 0.04], method="fdr_bh") == pytest.approx([0.04] * 4)


def test_benjamini_hochberg_takes_the_running_minimum_from_the_top() -> None:
    """p = .01 .04 .03 .20 (input order): ranked .01 .03 .04 .20 give p * 4 / rank = .04 .06 .0533 .20;
    the step-up minimum from the top turns the .06 into .0533, so q = .04 .0533 .0533 .20."""
    assert inference.adjust_pvalues([0.01, 0.04, 0.03, 0.20], method="fdr_bh") == pytest.approx(
        [0.04, 0.16 / 3.0, 0.16 / 3.0, 0.20]
    )


def test_a_family_member_without_a_p_does_not_count_in_m() -> None:
    """correct_family: the NaN leg is underpowered and m stays 4, giving the same q as above."""
    verdicts = efficacy.correct_family([0.01, math.nan, 0.04, 0.03, 0.20])
    assert [v.adjusted for v in verdicts] == pytest.approx([0.04, math.nan, 0.16 / 3.0, 0.16 / 3.0, 0.20], nan_ok=True)
    assert [v.label for v in verdicts] == [
        "significant",
        "underpowered",
        "not-significant",
        "not-significant",
        "not-significant",
    ]


def test_adjust_pvalues_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown multiple-comparison method"):
        inference.adjust_pvalues([0.1, 0.2], method="bonferroni-ish")


def test_adjust_pvalues_handles_an_empty_corpus() -> None:
    assert inference.adjust_pvalues([], method="fdr_bh") == []
    assert inference.adjust_pvalues([], method="holm") == []
