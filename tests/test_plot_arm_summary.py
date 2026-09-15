# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``scripts/plot_arm_summary.py`` -- where an arm's point lands on the speed-up axis.

Speed-up is a ratio, so its "overall" value is the GEOMETRIC MEAN over kernels
(:func:`hpcagent_bench.stats.population.kernel_medians`), the same rule every other "overall
speed-up" in this repo follows (:class:`~hpcagent_bench.stats.population.ArmAggregate`). A median
of per-kernel speed-ups equals the geomean only when the per-kernel values happen to be symmetric,
so the two statistics have to be told apart by an asymmetric fixture, not merely computed and
compared against each other.
"""

import importlib.util
import math
import pathlib
import sys

import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    """Import ``scripts/plot_arm_summary.py`` as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("plot_arm_summary", REPO / "scripts" / "plot_arm_summary.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plot = load_script()

#: Per-kernel speed-ups whose geomean and median disagree, at (or above) the interval floor
#: (summary.MIN_INTERVAL_SAMPLES = 5) so ``rules.require_interval`` does not reject the whole
#: table for being too thin to say anything either way: four kernels flat at 1.0x, one at 1000x.
#: Median = 1.0x (log2 = 0); geomean = 1000**0.2 (log2 = 3.32...).
ASYMMETRIC_SPEEDUPS: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1000.0)


def arm_frame(speedups: tuple[float, ...]) -> pd.DataFrame:
    """One arm, one kernel per speed-up, one episode each -- the shape ``arm_points`` groups over."""
    rows = []
    for index, value in enumerate(speedups):
        kernel = f"k{index}"
        run = f"w{index}"
        common = {
            "arm": "demo-arm",
            "model": "qwen38",
            "language": "c",
            "condition": "",
            "benchmark": kernel,
            "run_root": run,
            "job": run,
            "run_id": run,
            "attempt_index": 1,
            "baseline": "numba",
            "suspect": 0,
        }
        rows.append(
            {
                **common,
                "record": "submission",
                "speedup": value,
                "ts_ms": 1,
                "tokens": None,
                "baseline_ns": 1000.0,
                "native_ns": 1000.0 / value,
            }
        )
        rows.append(
            {
                **common,
                "record": "call",
                "speedup": value,
                "ts_ms": 2,
                "tokens": 100.0,
                "baseline_ns": 0.0,
                "native_ns": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_an_arm_points_speed_up_is_the_geomean_over_kernels_not_the_median() -> None:
    frame = arm_frame(ASYMMETRIC_SPEEDUPS)
    table = plot.arm_points(frame)
    assert len(table) == 1
    row = table.iloc[0]
    expected_geomean = math.prod(ASYMMETRIC_SPEEDUPS) ** (1.0 / len(ASYMMETRIC_SPEEDUPS))
    assert row.log2_speedup == pytest.approx(math.log2(expected_geomean))
    median_log2 = math.log2(sorted(ASYMMETRIC_SPEEDUPS)[len(ASYMMETRIC_SPEEDUPS) // 2])
    assert row.log2_speedup != pytest.approx(median_log2)


def test_an_arm_points_tokens_stay_the_median_over_kernels() -> None:
    """Tokens are not a ratio, so the summary rule that moved speed-up to the geomean does not
    apply here -- the spend axis stays the median :func:`hpcagent_bench.stats.population.kernel_medians`
    already reported."""
    frame = arm_frame(ASYMMETRIC_SPEEDUPS)
    table = plot.arm_points(frame)
    row = table.iloc[0]
    assert row.tokens == pytest.approx(100.0)
