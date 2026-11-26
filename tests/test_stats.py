# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for hpcagent_bench.stats: robust outlier rejection + median bootstrap CI."""

import warnings

import numpy as np
import pytest

from hpcagent_bench.stats.summary import drop_outliers, median_ci


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ([1.0, 1.0, 1.01, 0.99, 1.02, 0.98, 1.0, 1.0, 10.0], [10.0]),
        # Tight cluster (MAD == 0 because >= half are identical) + one hiccup: the mean-abs-dev fallback still
        # flags it.
        ([1.0] * 20 + [50.0], [50.0]),
    ],
    ids=["slow-hiccup", "mad-zero-fallback"],
)
def test_drop_removes_slow_hiccup(sample: list[float], expected: list[float]) -> None:
    kept, dropped = drop_outliers(sample, warn=False)
    assert dropped.tolist() == expected
    assert not set(expected) & set(kept.tolist())


def test_keeps_ordinary_jitter() -> None:
    kept, dropped = drop_outliers([1.0, 1.05, 0.95, 1.1, 0.9, 1.02, 0.98], warn=False)
    assert dropped.size == 0
    assert kept.size == 7


def test_lower_sample_is_kept() -> None:
    # A fast run is real signal (never below the hardware minimum), so the low side is never trimmed.
    around_ten = [10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9]
    kept, dropped = drop_outliers([*around_ten, 1.0], warn=False)
    assert dropped.size == 0
    assert 1.0 in kept.tolist()


def test_drop_warns_and_names_values() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        drop_outliers([1.0] * 8 + [10.0], warn=True, label="kern@dace")
    assert caught
    assert "outlier" in str(caught[0].message)
    assert "kern@dace" in str(caught[0].message)
    assert "10.0" in str(caught[0].message)


def test_degenerate_and_tiny_samples_no_drop() -> None:
    assert drop_outliers([2.0, 2.0], warn=False)[1].size == 0  # < 3 samples
    assert drop_outliers([5.0] * 10, warn=False)[1].size == 0  # all identical: no robust scale


@pytest.mark.parametrize(
    ("sample", "n_dropped", "point", "collapsed"),
    [
        (np.random.default_rng(0).normal(100.0, 5.0, 200).tolist(), 0, None, False),
        ([1.0] * 49 + [100.0], 1, 1.0, False),  # the hiccup is dropped and counted
        ([3.0] * 10, None, 3.0, True),  # no spread: the interval collapses onto the median
    ],
    ids=["brackets-median", "drops-and-reports-count", "point-ci-on-no-spread"],
)
def test_median_ci_brackets_median(
    sample: list[float], n_dropped: int | None, point: float | None, collapsed: bool
) -> None:
    med, lo, hi, n_drop = median_ci(sample, warn=False)
    assert lo <= med <= hi
    if n_dropped is not None:
        assert n_drop == n_dropped
    if collapsed:
        assert (med, lo, hi) == (point, point, point)
    elif point is not None:
        assert abs(med - point) < 1e-9
