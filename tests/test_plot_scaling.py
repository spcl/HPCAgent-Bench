# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The scaling figures: eta from known times, the geomean over P, the holes, and the axes.

A SYNTHETIC frame throughout -- no cluster data, no judge database. Every number below is one a
reader can check by hand against the formula in the docstring of
:func:`hpcagent_bench.harness.metric.scaling_point`, which is the point: the figure module must not
have a second definition of eta, and these tests fail if it grows one.
"""

import math
from collections.abc import Callable

import pandas as pd
import pytest
from matplotlib.figure import Figure

# hpcagent_bench.stats.figures selects the headless backend in its own __init__, before any
# submodule binds pyplot, so this import needs no matplotlib.use of its own.
from hpcagent_bench.harness import metric
from hpcagent_bench.stats.figures import scaling

#: The sweep both the weak and the strong fixtures are built over.
RANKS: tuple[int, ...] = (1, 2, 4, 8, 16)


def row(
    arm: str,
    kernel: str,
    mode: str,
    ranks: int,
    ranked_ns: float,
    single_rank_ns: float = 1000.0,
    work_ratio: float = float("nan"),
    note: str = "",
    ts_ms: int = 10,
) -> dict[str, object]:
    """One per-P scaling row in the shape the extractor is required to write."""
    return {
        "row_kind": scaling.SCALING_RECORD,
        "arm": arm,
        "benchmark": kernel,
        "scaling_mode": mode,
        "scaling_ranks": ranks,
        "scaling_nodes": -(-ranks // scaling.RANKS_PER_NODE),
        "scaling_ranked_ns": ranked_ns,
        "scaling_single_rank_ns": single_rank_ns,
        "scaling_work_ratio": work_ratio,
        "scaling_note": note,
        "ts_ms": ts_ms,
    }


def perfect_strong(arm: str, kernel: str, t1: float = 4096.0) -> list[dict[str, object]]:
    """A strong curve on the ideal: T(P) = T(1)/P, so eta(P) = 1. T(1) is a power of two so
    every T(P) is a whole nanosecond and no point is off the ideal by a truncated clock."""
    return [row(arm, kernel, "strong", p, t1 / p, single_rank_ns=t1) for p in RANKS]


def perfect_weak(arm: str, kernel: str, t1: float = 4096.0) -> list[dict[str, object]]:
    """A weak curve that hits the ideal: the P-times-larger problem takes the base time, eta = 1."""
    return [row(arm, kernel, "weak", p, t1, single_rank_ns=t1, work_ratio=float(p)) for p in RANKS]


def frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def only(curves: list[scaling.Curve]) -> scaling.Curve:
    assert len(curves) == 1, [(c.arm, c.kernel, c.mode) for c in curves]
    return curves[0]


def test_strong_efficiency_is_t1_over_p_times_tp() -> None:
    """eta(P) = T(1)/(P*T(P)): halving the time on four ranks is an efficiency of exactly 0.5."""
    rows = [row("mlscale-strong-qwen38-hip", "dist_softmax", "strong", 4, 500.0, single_rank_ns=1000.0)]
    curve = only(scaling.curves(frame(rows)))
    point = curve.points[0]
    assert point.ranks == 4
    assert point.achieved_speedup == pytest.approx(2.0)  # 1000 / 500
    assert point.ideal_speedup == pytest.approx(4.0)  # strong: sigma* = P
    assert point.efficiency == pytest.approx(0.5)  # 2 / 4
    # and it is the GRADER's number, not a second one computed here
    graded = metric.scaling_point("strong", 4, 1000, 500)
    assert point.efficiency == graded.efficiency


def test_weak_efficiency_divides_by_p_over_the_realized_work_ratio() -> None:
    """Weak eta = r*T(1)/(P*T(P)). Four times the work in 1.25x the time is eta = 0.8."""
    rows = [row("mlscale-weak-qwen38-hip", "dist_sdpa", "weak", 4, 1250.0, single_rank_ns=1000.0, work_ratio=4.0)]
    point = only(scaling.curves(frame(rows))).points[0]
    assert point.ideal_speedup == pytest.approx(1.0)  # P / r = 4 / 4
    assert point.efficiency == pytest.approx(0.8)  # (1000/1250) / 1.0
    # the work-SCALED speedup is what the speedup panel draws, and its ideal is P
    assert point.value("speedup") == pytest.approx(3.2)  # 4 * 1000 / 1250
    assert point.value("efficiency") == pytest.approx(0.8)


def test_a_weak_row_without_a_work_ratio_is_read_as_exact_growth() -> None:
    """An absent work_ratio means r = P (metric.ideal_speedup's own None), never r = 0 or a drop."""
    rows = [row("mlscale-weak-qwen38-hip", "dist_mlp_tp", "weak", 8, 1000.0, single_rank_ns=1000.0)]
    point = only(scaling.curves(frame(rows))).points[0]
    assert point.work_ratio == pytest.approx(8.0)
    assert point.ideal_speedup == pytest.approx(1.0)
    assert point.efficiency == pytest.approx(1.0)


def test_the_curve_score_is_the_geomean_over_p_and_not_the_mean() -> None:
    """geomean_P eta(P) over {1, 0.5}: sqrt(0.5) = 0.7071, where the arithmetic mean says 0.75."""
    arm = "mlscale-strong-qwen38-hip"
    rows = [
        row(arm, "dist_layer_norm", "strong", 2, 500.0, single_rank_ns=1000.0),  # eta = 1
        row(arm, "dist_layer_norm", "strong", 4, 500.0, single_rank_ns=1000.0),  # eta = 0.5
    ]
    curve = only(scaling.curves(frame(rows)))
    assert curve.mean_efficiency() == pytest.approx(math.sqrt(0.5))
    assert curve.mean_efficiency() != pytest.approx(0.75)


def test_a_perfect_sweep_scores_one_at_every_p_in_both_modes() -> None:
    """The fixtures are on the ideal, so every eta and every geomean is 1.0 -- weak and strong."""
    rows = perfect_strong("mlscale-strong-qwen38-hip", "dist_softmax")
    rows += perfect_weak("mlscale-weak-qwen38-hip", "dist_softmax")
    for curve in scaling.curves(frame(rows)):
        assert [p.efficiency for p in curve.points] == pytest.approx([1.0] * len(RANKS))
        assert curve.mean_efficiency() == pytest.approx(1.0)


def test_a_failed_p_is_dropped_with_its_reason_and_leaves_a_hole_not_a_zero() -> None:
    """A P the sweep could not measure has no point, is named with the judge's note, and is counted."""
    arm = "mlscale-strong-qwen38-hip"
    rows = perfect_strong(arm, "dist_moe_dispatch")
    rows = [r for r in rows if r["scaling_ranks"] != 8]
    rows.append(row(arm, "dist_moe_dispatch", "strong", 8, 0.0, note="P=8: mpi build failed"))
    curve = only(scaling.curves(frame(rows)))
    assert curve.ranks == (1, 2, 4, 16)  # the hole is a hole
    assert 8 not in [p.ranks for p in curve.points]
    assert curve.dropped == ((8, "P=8: mpi build failed"),)
    assert scaling.dropped_points([curve]) == [(arm, "dist_moe_dispatch", "strong", 8, "P=8: mpi build failed")]
    assert all(p.efficiency > 0 for p in curve.points)  # nothing was filled in at zero


def test_a_point_with_no_recorded_note_still_says_why_it_is_absent() -> None:
    """Every drop carries a reason; a row with none gets a plain statement rather than an empty cell."""
    rows = [row("mlscale-weak-qwen38-hip", "dist_gemm_add_relu", "weak", 16, 0.0)]
    curve = only(scaling.curves(frame(rows)))
    assert curve.points == ()
    assert curve.dropped[0][1] == "no measured time recorded at this P"


def test_a_single_point_curve_draws_nothing_and_is_counted() -> None:
    """One point is a measurement, not a curve: it is excluded from the figures and reported."""
    arm = "mlscale-strong-qwen38-hip"
    rows = perfect_strong(arm, "dist_softmax")
    rows += [row(arm, "dist_sdpa", "strong", 4, 250.0)]
    curves = scaling.curves(frame(rows))
    assert {c.kernel for c in scaling.drawable(curves)} == {"dist_softmax"}
    assert [c.kernel for c in scaling.single_point_curves(curves)] == ["dist_sdpa"]
    dropped = scaling.dropped_table(curves)
    assert (dropped.benchmark == "dist_sdpa").any()
    assert "fewer than" in dropped[dropped.benchmark == "dist_sdpa"].reason.iloc[0]


def test_a_kernel_only_one_model_solved_is_out_of_the_overlay_and_named() -> None:
    """The shared panels use the kernels EVERY arm has; the small multiples keep the solo one."""
    rows = perfect_strong("mlscale-strong-qwen38-hip", "dist_softmax")
    rows += perfect_strong("mlscale-strong-kimi27sglang-hip", "dist_softmax")
    rows += perfect_strong("mlscale-strong-qwen38-hip", "dist_moe_dispatch")
    curves = scaling.curves(frame(rows))
    assert scaling.common_kernels(curves, "strong") == {"dist_softmax"}
    assert {c.kernel for c in scaling.restrict(curves, "strong", scaling.common_kernels(curves, "strong"))} == {
        "dist_softmax"
    }
    figure = scaling.figure_per_kernel(curves, "strong")
    assert figure is not None
    titles = {ax.get_title() for ax in figure.axes if ax.get_visible()}
    assert titles == {"softmax", "moe_dispatch"}  # the shared dist_ prefix separates nothing


def test_a_panel_title_drops_only_a_prefix_every_kernel_shares() -> None:
    """``dist_sdpa`` reads as ``sdpa`` beside ``dist_softmax``, and is left whole beside ``gemm``."""
    assert scaling.panel_title("dist_sdpa", ["dist_sdpa", "dist_softmax"]) == "sdpa"
    assert scaling.panel_title("dist_sdpa", ["dist_sdpa", "gemm_add_relu"]) == "dist_sdpa"
    assert scaling.panel_title("dist_sdpa", ["dist_sdpa"]) == "dist_sdpa"  # one panel names itself


def test_the_rank_axis_is_log2_and_ticks_exactly_the_measured_rank_counts() -> None:
    """x is log2 with a tick at 1, 2, 4, 8, 16 -- the P that ran, not every power of two in the span."""
    rows = perfect_strong("mlscale-strong-qwen38-hip", "dist_softmax")
    curves = scaling.curves(frame(rows))
    assert scaling.rank_axis(curves) == RANKS
    figure = scaling.figure_efficiency(curves)
    assert figure is not None
    ax = figure.axes[0]
    assert ax.get_xscale() == "log"
    assert ax.xaxis.get_transform().base == 2  # pyright: ignore[reportAttributeAccessIssue]
    assert [int(t) for t in ax.get_xticks()] == list(RANKS)
    assert [t.get_text() for t in ax.get_xticklabels()] == [str(p) for p in RANKS]
    assert list(ax.xaxis.get_minorticklocs()) == []  # no minor grid (plotting.md rule 4)


def test_the_efficiency_panel_draws_the_ideal_at_one_and_the_speedup_panel_at_p() -> None:
    """Both figures carry their own ideal reference, and neither is left to the reader to imagine."""
    rows = perfect_strong("mlscale-strong-qwen38-hip", "dist_softmax")
    efficiency = scaling.figure_efficiency(scaling.curves(frame(rows)))
    assert efficiency is not None
    assert any(line.get_ydata()[0] == 1.0 for line in efficiency.axes[0].lines if len(line.get_ydata()) > 0)
    speedup = scaling.figure_speedup(scaling.curves(frame(rows)))
    assert speedup is not None
    ideal = [line for line in speedup.axes[0].lines if list(line.get_xdata()) == list(line.get_ydata())]
    # The bound runs border to border (USER 2026-09-25), not only between the measured P.
    assert ideal and tuple(ideal[0].get_xdata()) == pytest.approx(speedup.axes[0].get_xlim())


def test_every_figure_carries_one_legend_on_the_figure_and_none_on_an_axes() -> None:
    """plotting.md rule 5: one key, below the whole figure, never per panel."""
    rows = perfect_strong("mlscale-strong-qwen38-hip", "dist_softmax")
    rows += perfect_weak("mlscale-weak-qwen38-hip", "dist_softmax")
    curves = scaling.curves(frame(rows))
    for figure in (
        scaling.figure_efficiency(curves),
        scaling.figure_speedup(curves),
        scaling.figure_summary(curves),
        scaling.figure_per_kernel(curves, "weak"),
    ):
        assert figure is not None
        assert len(figure.legends) == 1
        assert all(ax.get_legend() is None for ax in figure.axes)


def test_the_summary_panel_is_the_geomean_over_kernels_and_withholds_a_two_kernel_interval() -> None:
    """One mark per arm per mode, at the geomean of its per-kernel geomean eta, n on the label; two
    kernels are below the 6-kernel floor, so the mark carries no interval."""
    arm = "mlscale-strong-qwen38-hip"
    rows = perfect_strong(arm, "dist_softmax")  # eta = 1 everywhere
    rows += [row(arm, "dist_sdpa", "strong", p, 1000.0 / p * 2.0, single_rank_ns=1000.0) for p in RANKS]  # eta = 0.5
    summary_rows = scaling.summary_rows(scaling.curves(frame(rows)))
    assert len(summary_rows) == 1
    arm, model, mode, interval, n_kernels = summary_rows[0]
    assert arm.endswith("qwen38-hip") and model == "qwen38"
    assert mode == "strong"
    assert n_kernels == 2
    assert interval.point == pytest.approx(math.sqrt(1.0 * 0.5))
    assert math.isnan(interval.low) and math.isnan(interval.high)


def test_only_the_latest_grade_of_a_kernel_enters_a_curve() -> None:
    """A re-graded (arm, kernel, mode) keeps its newest sweep; the superseded one is not pooled in."""
    arm = "mlscale-strong-qwen38-hip"
    rows = [row(arm, "dist_softmax", "strong", 4, 1000.0, ts_ms=1)]  # eta = 0.25, superseded
    rows += [row(arm, "dist_softmax", "strong", p, 1000.0 / p, ts_ms=99) for p in (2, 4)]
    curve = only(scaling.curves(frame(rows)))
    assert curve.ranks == (2, 4)
    assert curve.mean_efficiency() == pytest.approx(1.0)


def test_the_mode_falls_back_to_the_arm_name_when_a_row_does_not_state_it() -> None:
    """A CSV extracted before the column exists still splits weak from strong, by the arm key."""
    assert scaling.mode_of("mlscale-weak-qwen38-hip") == "weak"
    assert scaling.mode_of("mlscale-strong-kimi27sglang-hip") == "strong"
    assert scaling.mode_of("llr-focus40-qwen38-c") == ""
    assert scaling.mode_of("mlscale-weak-qwen38-hip", "strong") == "strong"  # a stated mode wins


def test_a_recorded_efficiency_that_the_times_do_not_give_is_reported() -> None:
    """The disclosure column is CHECKED against metric.scaling_point, never trusted over it."""
    rows = [row("mlscale-strong-qwen38-hip", "dist_softmax", "strong", 4, 500.0)]
    good = frame(rows)
    good["scaling_point_efficiency"] = [0.5]
    assert scaling.disagreements(good) == []
    bad = frame(rows)
    bad["scaling_point_efficiency"] = [0.9]
    assert scaling.disagreements(bad) == [("mlscale-strong-qwen38-hip", "dist_softmax", 4, 0.9, 0.5)]


def test_an_empty_frame_draws_nothing_and_raises_nothing() -> None:
    """No scaling sweep yet is a normal state of the observations table, not a crash."""
    for empty in (pd.DataFrame(), pd.DataFrame(columns=list(scaling.REQUIRED_COLUMNS)), frame([])):
        assert scaling.curves(empty) == []
        assert scaling.disagreements(empty) == []
    assert scaling.figure_efficiency([]) is None
    assert scaling.figure_speedup([]) is None
    assert scaling.figure_summary([]) is None
    assert scaling.figure_per_kernel([], "weak") is None
    assert scaling.points_table([]).empty
    assert scaling.dropped_table([]).empty


def test_a_frame_of_grade_rows_alone_holds_no_scaling_rows() -> None:
    """The per-P rows are selected by ``record``, so an ordinary observations CSV yields no curves."""
    grades = pd.DataFrame(
        [{"row_kind": "submission", "arm": "mlscale-weak-qwen38-hip", "benchmark": "dist_softmax", "speedup": 2.0}]
    )
    assert scaling.curves(grades) == []


@pytest.mark.parametrize("builder", [scaling.figure_summary, scaling.figure_efficiency])
def test_a_print_size_scaling_figure_keeps_every_text_on_the_print_scale(builder: Callable[..., Figure]) -> None:
    """The ML figure sits in a wrap beside the cost figure: both must print at the same type sizes and width."""
    from hpcagent_bench.stats import style

    arm = "mlscale-strong-qwen38-hip"
    rows = perfect_strong(arm, "dist_softmax") + perfect_weak("mlscale-weak-qwen38-hip", "dist_softmax")
    fig = builder(scaling.curves(frame(rows)), width=style.ICLR_WRAP_WIDTH_IN, type_=style.PRINT_SCALE)
    assert fig is not None
    try:
        assert style.print_type_violations(fig) == []
        assert float(fig.get_size_inches()[0]) == pytest.approx(style.ICLR_WRAP_WIDTH_IN)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


@pytest.mark.parametrize("geomean_panel", [True, False])
def test_the_mode_grid_is_one_row_per_law_one_column_per_picked_kernel_and_the_geomean(geomean_panel: bool) -> None:
    """The paper's scaling figure: weak above strong, the picked kernels in the picked order, the
    geomean column last only when asked, one key under the figure and one shared Y label."""
    import matplotlib.pyplot as plt

    from hpcagent_bench import experiment_tags

    rows = []
    for kernel in ("dist_softmax", "dist_sdpa", "dist_matmul"):
        rows += perfect_strong("mlscale-strong-qwen38-hip", kernel) + perfect_weak("mlscale-weak-qwen38-hip", kernel)
    fig = scaling.figure_mode_grid(
        scaling.curves(frame(rows)), ["dist_sdpa", "dist_softmax"], geomean_panel=geomean_panel
    )
    assert fig is not None
    try:
        columns = 2 + int(geomean_panel)
        assert len(fig.axes) == 2 * columns
        top, bottom = fig.axes[:columns], fig.axes[columns:]
        assert [ax.get_ylabel() for ax in (top[0], bottom[0])] == ["Weak Scaling", "Strong Scaling"]
        names = [experiment_tags.kernel_short_display_name(k) for k in ("dist_sdpa", "dist_softmax")]
        assert [ax.get_title() for ax in top] == names + [scaling.GEOMEAN_LABEL] * geomean_panel
        assert len(fig.legends) == 1 and all(ax.get_legend() is None for ax in fig.axes)
        assert [text.get_text() for text in fig.texts].count(scaling.SPEEDUP_LABEL) == 1
    finally:
        plt.close(fig)


def test_the_band_at_a_rank_count_is_the_log_t_interval_of_the_kernels_geomean() -> None:
    """Six kernels at P = 2 with efficiency 1, 0.5, 0.25 twice: GM = 0.5, log2 sd = sqrt(0.8),
    t(0.975, 5) = 2.5706, so the band is 0.5 * 2^(-/+ 0.93865) = 0.26086 .. 0.95836 (scipy t quantile, worked by hand)."""
    arm = "mlscale-strong-qwen38-hip"
    t1 = 4096.0
    rows = [
        row(arm, f"k{index}", "strong", p, t1 / p / (eta if p == 2 else 1.0), single_rank_ns=t1)
        for index, eta in enumerate((1.0, 0.5, 0.25, 1.0, 0.5, 0.25))
        for p in RANKS
    ]

    band = scaling.series(scaling.curves(frame(rows)), "efficiency")[2]

    assert (band.point, band.low, band.high) == pytest.approx((0.5, 0.2608615520422396, 0.958362771526864))
    assert band.method == "log-t"
