# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for hpcagent_bench.stats: robust outlier rejection + median bootstrap CI."""
import warnings

import numpy as np

from hpcagent_bench.stats import DEFAULT_MAD_Z, drop_outliers, median_ci


def test_drop_removes_slow_hiccup() -> None:
    tight = [1.0, 1.0, 1.01, 0.99, 1.02, 0.98, 1.0, 1.0]
    kept, dropped = drop_outliers(tight + [10.0], warn=False)
    assert dropped.tolist() == [10.0]
    assert 10.0 not in kept.tolist()


def test_keeps_ordinary_jitter() -> None:
    kept, dropped = drop_outliers([1.0, 1.05, 0.95, 1.1, 0.9, 1.02, 0.98], warn=False)
    assert dropped.size == 0
    assert kept.size == 7


def test_lower_sample_is_kept() -> None:
    # A fast run is real signal (never below the hardware minimum), so the low side is never trimmed.
    around_ten = [10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9]
    kept, dropped = drop_outliers(around_ten + [1.0], warn=False)
    assert dropped.size == 0
    assert 1.0 in kept.tolist()


def test_drop_warns_and_names_values() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        drop_outliers([1.0] * 8 + [10.0], warn=True, label="kern@dace")
    assert caught and "outlier" in str(caught[0].message)
    assert "kern@dace" in str(caught[0].message) and "10.0" in str(caught[0].message)


def test_degenerate_and_tiny_samples_no_drop() -> None:
    assert drop_outliers([2.0, 2.0], warn=False)[1].size == 0  # < 3 samples
    assert drop_outliers([5.0] * 10, warn=False)[1].size == 0  # all identical: no robust scale


def test_mad_zero_fallback_catches_outlier() -> None:
    # Tight cluster (MAD == 0 because >= half are identical) + one hiccup: the mean-abs-dev
    # fallback still flags it.
    kept, dropped = drop_outliers([1.0] * 20 + [50.0], warn=False)
    assert dropped.tolist() == [50.0]


def test_threshold_is_the_documented_default() -> None:
    assert DEFAULT_MAD_Z == 5.0


def test_median_ci_brackets_median() -> None:
    x = np.random.default_rng(0).normal(100.0, 5.0, 200).tolist()
    med, lo, hi, n_drop = median_ci(x, warn=False)
    assert lo <= med <= hi
    assert n_drop == 0


def test_median_ci_drops_and_reports_count() -> None:
    med, lo, hi, n_drop = median_ci([1.0] * 49 + [100.0], warn=False)
    assert n_drop == 1
    assert abs(med - 1.0) < 1e-9


def test_median_ci_point_ci_on_no_spread() -> None:
    med, lo, hi, n_drop = median_ci([3.0] * 10, warn=False)
    assert (med, lo, hi) == (3.0, 3.0, 3.0)
