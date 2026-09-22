# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``hpcagent_bench.stats.figures.per_kernel`` and ``statistics/plot_per_kernel.py`` -- the per-kernel
speed-up and tokens figure: ci/box style, the log2 speed-up axis, the summary column and the
separate/stacked layout.
"""

import argparse
import importlib.util
import math
import pathlib
import sys
import types

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.lines
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection

from hpcagent_bench import experiment_tags
from hpcagent_bench.stats import population, style
from hpcagent_bench.stats.figures import per_kernel as pk

REPO = pathlib.Path(__file__).resolve().parents[1]


def load_script() -> types.ModuleType:
    """Import ``statistics/plot_per_kernel.py`` as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("plot_per_kernel", REPO / "statistics" / "plot_per_kernel.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plot = load_script()


def speed_metric(cells: list[pk.KernelCell], color: str = "#1155cc") -> pk.Metric:
    """A one-series speed-up panel, the shape ``statistics/plot_per_kernel.py`` builds."""
    return pk.speedup_series_metric([pk.Series("", tuple(cells), color)], "Speed-Up")


def token_metric(cells: list[pk.KernelCell], color: str = "#cc5511") -> pk.Metric:
    """A one-series token panel, the shape ``statistics/plot_per_kernel.py`` builds."""
    return pk.token_series_metric([pk.Series("", tuple(cells), color)], "Tokens")


def episode(kernel: str, run: str, speedup: float, tokens: float) -> list[dict]:
    """One episode as the judge records it: a graded row and a call row, the shape
    :func:`hpcagent_bench.stats.population.graded_episode_rows`/``episode_tokens`` read."""
    common = {
        "arm": "demo-arm",
        "benchmark": kernel,
        "run_root": run,
        "job": run,
        "run_id": run,
        "attempt_index": 1,
        "ts_ms": 1,
        "suspect": 0,
        "timing_reduction": "mwd-v2",
    }
    return [
        {**common, "record": "submission", "speedup": speedup, "tokens": None},
        {**common, "record": "task", "speedup": None, "tokens": tokens},
    ]


def frame_of(cells: dict[str, list[tuple[float, float]]]) -> pd.DataFrame:
    """``{kernel: [(speedup, tokens), ...]}`` -> one row set, one episode per pair."""
    rows: list[dict] = []
    for kernel, pairs in cells.items():
        for index, (speedup, tokens) in enumerate(pairs):
            rows += episode(kernel, f"{kernel}-w{index}", speedup, tokens)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reduction: which population speedup_cells / token_cells read.


def test_speedup_cells_keeps_every_episodes_own_final_speedup() -> None:
    """A kernel run by 3 episodes (git-scicomp) must carry all 3 values in its cell, not the best
    or the mean of them -- the box style draws their raw spread."""
    frame = frame_of({"k1": [(1.0, 10.0)], "k2": [(1.5, 20.0), (3.0, 30.0), (2.2, 25.0)]})
    cells = {cell.kernel: cell for cell in pk.speedup_cells(frame)}
    assert cells["k1"].episodes == (1.0,)
    assert sorted(cells["k2"].episodes) == [1.5, 2.2, 3.0]


def test_token_cells_keeps_every_episodes_own_total_not_the_kernel_sum() -> None:
    frame = frame_of({"k1": [(1.0, 10.0)], "k2": [(1.5, 20.0), (3.0, 30.0), (2.2, 25.0)]})
    cells = {cell.kernel: cell for cell in pk.token_cells(frame)}
    assert cells["k1"].episodes == (10.0,)
    assert sorted(cells["k2"].episodes) == [20.0, 25.0, 30.0]


# ---------------------------------------------------------------------------
# The log2 speed-up axis.


@pytest.mark.parametrize(
    "ratio, label",
    [
        pytest.param(0.125, "0.125x", id="eighth"),
        pytest.param(0.25, "0.25x", id="quarter"),
        pytest.param(0.5, "0.5x", id="half"),
        pytest.param(1.0, "1x", id="unity"),
        pytest.param(2.0, "2x", id="double"),
        pytest.param(4.0, "4x", id="quadruple"),
    ],
)
def test_speedup_tick_label_reads_a_log2_ratio_back_as_a_ratio(ratio: float, label: str) -> None:
    assert style.ratio_tick_label(ratio) == label


def test_speedup_yticks_always_spans_at_least_a_quarter_to_four_x() -> None:
    """A panel of narrow-range wins (1.1x .. 1.8x) must still show the 1x line inside a band wide
    enough to read a slow-down and a speed-up the same distance from it."""
    cells = [pk.KernelCell("k1", (1.1,)), pk.KernelCell("k2", (1.8,))]
    ticks = pk.speedup_yticks(cells)
    assert 0.25 in ticks and 4.0 in ticks and 1.0 in ticks


def test_speedup_yticks_grows_to_cover_a_wide_range() -> None:
    cells = [pk.KernelCell("k1", (0.1,)), pk.KernelCell("k2", (20.0,))]
    ticks = pk.speedup_yticks(cells)
    assert min(ticks) <= 0.1 and max(ticks) >= 20.0


# ---------------------------------------------------------------------------
# ci vs box drawing.


@pytest.mark.parametrize(
    "n, boxed",
    [
        pytest.param(1, False, id="one-episode-is-a-point"),
        pytest.param(2, False, id="two-episodes-is-a-point"),
        pytest.param(3, True, id="three-episodes-is-a-box"),
        pytest.param(5, True, id="five-episodes-is-a-box"),
    ],
)
def test_box_style_draws_a_real_box_only_at_or_above_the_spread_floor(n: int, boxed: bool) -> None:
    cell = pk.KernelCell("k1", tuple(float(i + 1) for i in range(n)))
    fig, ax = plt.subplots()
    try:
        pk.draw_box(ax, pk.Series("", (cell,), "#1155cc"), {"k1": 0})
        assert (len(ax.patches) > 0) == boxed
    finally:
        plt.close(fig)


def test_ci_style_never_draws_a_box_patch() -> None:
    """``ci`` draws a point and a whisker, never a boxplot artist -- even at a spread-worthy n."""
    cell = pk.KernelCell("k1", (1.0, 2.0, 4.0, 8.0, 16.0))
    fig, ax = plt.subplots()
    try:
        pk.draw_ci(ax, pk.Series("", (cell,), "#1155cc"), {"k1": 0}, log2_space=True)
        assert len(ax.patches) == 0
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# The --summary column.


def test_summary_point_speedup_is_the_geomean_over_the_plotted_kernels_own_medians() -> None:
    """Speed-up is a ratio, so its overall value is the GEOMETRIC MEAN over kernels -- never a
    median, which equals the geomean only when the per-kernel medians happen to be symmetric."""
    cells = [pk.KernelCell("k1", (1.0,)), pk.KernelCell("k2", (1.0,)), pk.KernelCell("k3", (1000.0,))]
    point, _, _ = pk.summary_point_speedup(cells)
    expected_geomean = (1.0 * 1.0 * 1000.0) ** (1.0 / 3.0)
    assert point == pytest.approx(expected_geomean)
    assert point != pytest.approx(1.0)  # the median of (1, 1, 1000) is 1.0 -- must not equal it


def test_summary_point_tokens_is_the_median_over_the_plotted_kernels_own_medians() -> None:
    """Tokens are not a ratio, so the summary column keeps the median -- the geomean rule above
    does not apply to a count."""
    cells = [pk.KernelCell("k1", (10.0,)), pk.KernelCell("k2", (20.0,)), pk.KernelCell("k3", (1000.0,))]
    point, _, _ = pk.summary_point_tokens(cells)
    assert point == pytest.approx(20.0)


def test_draw_panel_labels_the_summary_column_with_its_own_statistic() -> None:
    """The annotation above the summary marker names the statistic it draws (Geomean for
    speed-up, Median for tokens), so a reader is not left to assume it matches the per-kernel
    style."""
    speed = speed_metric([pk.KernelCell("k1", (2.0,))])
    fig, ax = plt.subplots()
    try:
        pk.draw_panel(ax, speed, ["k1"], "ci", True, True)
        labels = [text.get_text() for text in ax.texts]
    finally:
        plt.close(fig)
    assert "Geomean" in labels


def test_the_top_panel_of_a_stacked_figure_still_names_its_own_summary_statistic() -> None:
    """Kernel names are hidden on every panel but the bottom (shared x axis, shown once) -- but
    the two panels' summary columns carry DIFFERENT statistics (geomean, median), and a plain x
    TICK label would be silently overwritten by whichever panel's axis drew second, since a
    stacked figure's two axes share one set of tick labels (see draw_summary_column). The
    annotation above each panel's own marker is not shared, so both survive."""
    speed = speed_metric([pk.KernelCell("k1", (2.0,))])
    tokens = token_metric([pk.KernelCell("k1", (100.0,))])
    fig = pk.figure_panels([speed, tokens], ["k1"], "ci", True, "demo")
    try:
        top_ax, bottom_ax = fig.axes
        top_texts = [text.get_text() for text in top_ax.texts]
        bottom_texts = [text.get_text() for text in bottom_ax.texts]
        assert "Geomean" in top_texts
        assert "Median" in bottom_texts
        assert "Geomean" not in bottom_texts
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# --layout separate vs stacked, and which files a run writes.


def build_metrics(frame: pd.DataFrame) -> tuple[pk.Metric, pk.Metric, list[pk.KernelCell], list[pk.KernelCell]]:
    speed_cells = pk.speedup_cells(frame)
    token_cells = pk.token_cells(frame)
    return speed_metric(speed_cells), token_metric(token_cells), speed_cells, token_cells


def render_args(tmp_path: pathlib.Path, layout: str, style_: str = "ci", summary: bool = False) -> argparse.Namespace:
    return argparse.Namespace(layout=layout, style=style_, summary=summary, out=tmp_path / "figures" / "per_kernel.pdf")


def test_layout_separate_writes_one_file_per_metric(tmp_path: pathlib.Path) -> None:
    frame = frame_of({"k1": [(2.0, 100.0)], "k2": [(0.5, 200.0)]})
    speed, tokens, speed_cells, token_cells = build_metrics(frame)
    written = plot.render(speed, tokens, speed_cells, token_cells, render_args(tmp_path, "separate"), "demo")
    names = sorted(p.name for p in written)
    assert names == ["per_kernel-speedup", "per_kernel-tokens"]


def test_layout_stacked_writes_one_file_for_both_metrics(tmp_path: pathlib.Path) -> None:
    frame = frame_of({"k1": [(2.0, 100.0)], "k2": [(0.5, 200.0)]})
    speed, tokens, speed_cells, token_cells = build_metrics(frame)
    written = plot.render(speed, tokens, speed_cells, token_cells, render_args(tmp_path, "stacked"), "demo")
    assert [p.name for p in written] == ["per_kernel"]


def test_layout_stacked_refuses_when_one_metric_has_no_cells(tmp_path: pathlib.Path) -> None:
    """A stacked figure shares one kernel axis between two panels; with only one metric present
    there is nothing for the second panel to share it with."""
    frame = frame_of({"k1": [(2.0, 100.0)]})
    frame.loc[frame.record == "task", "tokens"] = None  # drop every token cell
    speed, tokens, speed_cells, token_cells = build_metrics(frame)
    assert token_cells == []
    with pytest.raises(SystemExit, match="stacked"):
        plot.render(speed, tokens, speed_cells, token_cells, render_args(tmp_path, "stacked"), "demo")


def test_stacked_layouts_two_panels_share_the_kernel_axis() -> None:
    """Column ``i`` must name the SAME kernel in both panels, even when one metric is missing a
    kernel the other has -- an empty slot at that column, never a re-packed one."""
    speed_cells = [pk.KernelCell("a", (2.0,)), pk.KernelCell("b", (1.0,))]
    token_cells = [pk.KernelCell("b", (50.0,)), pk.KernelCell("c", (80.0,))]
    kernels = pk.shared_kernel_order(speed_cells, token_cells)
    assert kernels == ["b", "a", "c"]  # speed-up's own order (b before a), then tokens-only "c"

    speed = speed_metric(speed_cells)
    tokens = token_metric(token_cells)
    fig = pk.figure_panels([speed, tokens], kernels, "ci", False, "demo")
    try:
        top_ax, bottom_ax = fig.axes
        assert top_ax.get_xlim() == bottom_ax.get_xlim()
        assert list(top_ax.get_xticks()) == list(bottom_ax.get_xticks())
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Reproducibility.


def test_a_rerun_writes_byte_identical_png_and_pdf(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    speed_cells = [pk.KernelCell("k1", (1.0, 2.0, 4.0)), pk.KernelCell("k2", (0.5,))]
    speed = speed_metric(speed_cells)
    for epoch, folder in (("0", "first"), ("86400", "second")):
        monkeypatch.setenv("SOURCE_DATE_EPOCH", epoch)
        fig = pk.figure_one(speed, pk.ordered_kernels(speed_cells), "box", True, "demo")
        pk.save(fig, tmp_path / folder / "figure.pdf")
    for name in ("figure.pdf", "figure.png"):
        first, second = (tmp_path / folder / name for folder in ("first", "second"))
        assert first.read_bytes() == second.read_bytes(), f"{name} depends on when it was saved"


def test_the_speedup_panel_carries_a_major_grid_and_a_minor_one_on_the_value_axis_only() -> None:
    """The measured axis is ruled at the pinned powers of two and, lighter, at the shared minors
    between them (user, 2026-09-22); the kernel axis carries names and no line at all."""
    fig, ax = plt.subplots()
    try:
        pk.style_speedup_axis(ax, [pk.KernelCell("k1", (1.0, 2.0))])
        assert any(line.get_visible() for line in ax.yaxis.get_gridlines())
        assert [tick for tick in ax.yaxis.get_minor_ticks() if tick.gridline.get_visible()]
        assert not [tick for tick in ax.xaxis.get_major_ticks() if tick.gridline.get_visible()]
        assert not [tick for tick in ax.xaxis.get_minor_ticks() if tick.gridline.get_visible()]
    finally:
        plt.close(fig)


def test_the_token_panel_puts_its_measured_value_on_a_log_y_axis_with_a_major_and_a_minor_grid() -> None:
    """A token count is a measured quantity, so it is on Y; the kernel names are the x categories."""
    fig, ax = plt.subplots()
    try:
        pk.style_token_axis(ax, [pk.KernelCell("k1", (120.0,)), pk.KernelCell("k2", (90000.0,))])
        assert ax.get_yscale() == "log"
        assert any(line.get_visible() for line in ax.yaxis.get_gridlines())
        assert [tick for tick in ax.yaxis.get_minor_ticks() if tick.gridline.get_visible()]
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("value", "want"),
    [
        (0.125, "0.125x"),
        (0.5, "0.5x"),
        (1.0, "1x"),
        (2.0, "2x"),
        # Not a power of two, and not a whole reciprocal: the old 1/n spelling rounded this to
        # "1/1x", a ratio of one marking a point 30% below it.
        (0.5**0.5, "0.707x"),
        (0.35, "0.35x"),
    ],
)
def test_a_ratio_below_one_prints_as_a_decimal(value: float, want: str) -> None:
    assert style.ratio_tick_label(value) == want


# ---------------------------------------------------------------------------
# The one per-kernel API every per-kernel figure draws through: cells, ticks, marks, summary, canvas.


def answer_rows(kernel: str, run: str, ts_ms: int, speedup: float) -> dict[str, object]:
    """One graded submission of one run, the columns ``population.kernel_answers`` reads."""
    return {
        "arm": "demo-arm", "benchmark": kernel, "run_root": run, "job": run, "run_id": run, "record": "submission",
        "speedup": speedup, "baseline_ns": 1000.0, "native_ns": 1000.0 / speedup, "baseline": "numba",
        "suspect": 0, "ts_ms": ts_ms, "attempt_index": 1, "timing_reduction": "mwd-v2",
    }  # fmt: skip


def test_answer_cells_takes_a_rerun_kernels_latest_run_not_its_first() -> None:
    """A rerun supersedes the run it replaced: a stale first answer drawn beside the rerun's would
    credit the arm with a result its latest run did not deliver."""
    frame = pd.DataFrame([answer_rows("k1", "first", 10, 8.0), answer_rows("k1", "rerun", 30, 2.0)])
    assert [(cell.kernel, cell.episodes) for cell in pk.answer_cells(frame)] == [("k1", (2.0,))]


@pytest.mark.parametrize(
    "low, high",
    [
        pytest.param(1.1, 1.8, id="narrow"),
        pytest.param(0.1, 20.0, id="eight-octaves"),
        pytest.param(1.0 / 64.0, 1024.0, id="sixteen-octaves"),
        pytest.param(2.0, 4096.0, id="wins-only"),
    ],
)
def test_speedup_yticks_stay_within_the_tick_budget_and_always_carry_1x(low: float, high: float) -> None:
    """Twelve octaves labelled one by one printed on top of each other; and the 1x line is the
    reference every other tick is read against, so thinning must never drop it."""
    ticks = pk.speedup_yticks([pk.KernelCell("a", (low,)), pk.KernelCell("b", (high,))])
    assert len(ticks) <= pk.MAX_SPEEDUP_TICKS, ticks
    assert 1.0 in ticks, ticks
    assert ticks[0] <= low and ticks[-1] >= high, ticks


def test_kernel_medians_leave_out_undelivered_pending_and_flagged_cells() -> None:
    """A placeholder's 1x, a kernel not run yet and a disowned claim are drawn but are not measured
    values; entering any of them would move the summary by attrition or by the exploit."""
    cells = [
        pk.KernelCell("solved", (4.0,)),
        pk.KernelCell("unanswered", (1.0,), delivered=False),
        pk.KernelCell("waiting", (1.0,), delivered=False, pending=True),
        pk.KernelCell("exploited", (1007.0,), flagged=True),
    ]
    assert list(pk.kernel_medians(cells)) == [4.0]


def test_kernel_cells_fill_a_missing_speedup_at_the_placeholder_and_drop_a_missing_cost() -> None:
    """A kernel with no verified answer still happened (1x, crossed); a kernel with no token total
    is no measurement, and a mark at the axis edge would read as the smallest spend."""
    speed = pk.kernel_cells({"k1": 3.0}, ["k1", "k2"])
    tokens = pk.kernel_cells({"k1": 900.0}, ["k1", "k2"], fill=False)
    assert [(c.kernel, c.episodes, c.delivered) for c in speed] == [
        ("k1", (3.0,), True),
        ("k2", (population.NOT_DELIVERED,), False),
    ]
    assert [c.kernel for c in tokens] == ["k1"]


def test_kernel_cells_keep_a_present_placeholders_own_value_and_flag() -> None:
    """A ratio one side of which never delivered is a number but not a measurement: it keeps its
    value (the cross goes where the ratio is) and never counts as delivered."""
    (cell,) = pk.kernel_cells({"k1": 4.0}, ["k1"], delivered={"k1": False})
    assert cell.episodes == (4.0,) and not cell.delivered


@pytest.mark.parametrize(
    "low, high, want",
    [
        pytest.param(90.0, 120.0, (90.0, 120.0), id="a-real-range"),
        pytest.param(100.0, 100.0, None, id="one-task-no-range"),
        pytest.param(math.nan, 120.0, None, id="half-a-range"),
    ],
)
def test_kernel_cells_carry_an_interval_only_when_it_has_width(
    low: float, high: float, want: tuple[float, float] | None
) -> None:
    (cell,) = pk.kernel_cells({"k1": 100.0}, ["k1"], fill=False, low={"k1": low}, high={"k1": high})
    assert cell.interval == want


def test_kernel_cells_turn_a_pending_kernel_into_a_pending_placeholder_whatever_its_value() -> None:
    """A kernel not attempted yet must not read as a failure, nor as whatever stale value a map
    still holds for it."""
    (cell,) = pk.kernel_cells({"k1": 5.0}, ["k1"], pending=frozenset({"k1"}))
    assert cell.pending and not cell.delivered and cell.episodes == (population.NOT_DELIVERED,)


def test_an_undelivered_cell_draws_hollow_and_crossed_at_its_own_value() -> None:
    """The cross is what separates a placeholder from a measured value at the same height; it goes
    on the cell's own value, which for a one-sided ratio is not 1x."""
    fig, ax = plt.subplots()
    try:
        pk.draw_ci(ax, pk.Series("", (pk.KernelCell("k1", (4.0,), delivered=False),), "#1155cc"), {"k1": 0}, True)
        at = [c for c in ax.collections if isinstance(c, PathCollection) and list(c.get_offsets()[0]) == [0.0, 4.0]]
    finally:
        plt.close(fig)
    assert len(at) == 3, at  # the white halo, the hollow mark, the cross
    assert any(len(c.get_facecolors()) == 0 for c in at)


@pytest.mark.parametrize(
    "count, want",
    [
        pytest.param(0, [], id="no-series-no-offset"),
        pytest.param(1, [0.0], id="one-series-on-the-column"),
        pytest.param(3, [-0.31, 0.0, 0.31], id="three-over-the-span"),
    ],
)
def test_dodge_offsets_centre_the_series_on_their_column(count: int, want: list[float]) -> None:
    assert pk.dodge_offsets(count) == pytest.approx(want)


def test_dodge_offsets_with_no_span_stack_every_series_on_the_column() -> None:
    """``--offset 0`` on the compiler figure: optimizers differ by shape, all at the kernel's x."""
    assert pk.dodge_offsets(4, 0.0) == [0.0, 0.0, 0.0, 0.0]


def test_mark_size_shrinks_with_crowding_down_to_its_floor() -> None:
    """A crowded column shrinks its marks rather than merging them, never below the size at which
    the undelivered cross still reads."""
    roomy = pk.mark_size(0.3, 2)
    crowded = pk.mark_size(0.3, 12)
    assert roomy == pytest.approx(pk.MARK_PT**2)
    assert crowded < roomy
    assert pk.mark_size(0.01, 40) == pytest.approx(pk.MIN_MARK_PT**2)


def test_an_undodged_mark_is_sized_to_the_column_not_to_a_zero_gap() -> None:
    """Stacked series are one column apart from their neighbours, not zero: a zero gap would floor
    every mark on the compiler figure."""
    assert pk.mark_size(0.3, 6, span=0.0) == pytest.approx(pk.MARK_PT**2)


def test_status_handles_key_only_the_status_marks_a_figure_draws() -> None:
    """An entry for a mark that is not on the figure is one more thing to read and find nowhere."""
    solved = speed_metric([pk.KernelCell("k1", (2.0,))])
    failed = speed_metric([pk.KernelCell("k1", (1.0,), delivered=False)])
    waiting = speed_metric([pk.KernelCell("k1", (1.0,), delivered=False, pending=True)])
    assert pk.status_handles([solved]) == []
    assert [h.get_label() for h in pk.status_handles([failed])] == [style.NOT_DELIVERED_LABEL]
    assert [h.get_label() for h in pk.status_handles([waiting])] == [style.PENDING_LABEL]


def test_the_undelivered_key_entry_shows_the_cross() -> None:
    """Hollow is this repo's spelling for a control, so a key showing only the hollow shape names
    the wrong thing; the cross is what separates a placeholder from a measurement."""
    (handle,) = pk.status_handles([speed_metric([pk.KernelCell("k1", (1.0,), delivered=False)])])
    assert handle.get_marker() == "x"


def three_series_metric() -> pk.Metric:
    return pk.speedup_series_metric(
        [
            pk.Series("a", (pk.KernelCell("k1", (2.0,)), pk.KernelCell("k2", (4.0,))), "#1155cc"),
            pk.Series("b", (pk.KernelCell("k1", (3.0,)), pk.KernelCell("k2", (1.0,), delivered=False)), "#cc5511"),
            pk.Series("c", (pk.KernelCell("k1", (0.5,)),), "#11cc55", "^"),
        ],
        "Speed-Up",
    )


def test_every_series_gets_one_summary_slot_and_one_settled_value_label() -> None:
    """Summaries that agree to a few percent, drawn in one column, hid all but the top mark; and the
    value a caption quotes has to be on the figure, tagged so it settles clear of the marks."""
    fig = pk.figure_one(three_series_metric(), ["k1", "k2"], "ci", True, "")
    try:
        (ax,) = fig.axes
        separator = pk.summary_separator_x(2)
        slots = sorted(
            {round(float(x), 6) for c in ax.collections if isinstance(c, PathCollection) for x, _ in c.get_offsets()
             if x > separator}
        )  # fmt: skip
        labels = [t for t in ax.texts if t.get_gid() == style.CLEAR_GID]
    finally:
        plt.close(fig)
    assert slots == [round(pk.summary_slot_x(2, slot), 6) for slot in range(3)], slots
    assert len(labels) == 3, [t.get_text() for t in labels]


def test_a_summary_value_label_prints_the_geomean_over_solved_kernels_only() -> None:
    """Series b solved k1 at 3x and failed k2: its printed value is 3x, not the geomean with the
    placeholder's 1x (1.7x)."""
    fig = pk.figure_one(three_series_metric(), ["k1", "k2"], "ci", True, "")
    try:
        texts = [t.get_text() for t in fig.axes[0].texts if t.get_gid() == style.CLEAR_GID]
    finally:
        plt.close(fig)
    assert texts == [style.ratio_label(8.0**0.5), style.ratio_label(3.0), style.ratio_label(0.5)], texts


def test_kernel_tick_label_prints_the_manifest_short_name() -> None:
    assert pk.kernel_tick_label("argmax_with_index") == experiment_tags.kernel_short_display_name("argmax_with_index")
    assert pk.kernel_tick_label("argmax_with_index") != experiment_tags.kernel_display_name("argmax_with_index")


@pytest.mark.parametrize(
    "kernel",
    [
        pytest.param("conv2d_group_norm_tanh_hardswish_residual_add_logsumexp", id="hyphenated"),
        pytest.param("quasi_affine_floor_div_scatter", id="spaced"),
    ],
)
def test_kernel_tick_label_folds_a_long_fallback_name_without_dropping_a_character(kernel: str) -> None:
    """A kernel with no short name falls back to its full name; unfolded, one long rotated name
    deepens the whole band, and cut ("2-D Jacobi stencil..") it no longer names one kernel."""
    name = experiment_tags.kernel_display_name(kernel)
    assert len(name) > experiment_tags.SHORT_NAME_MAX
    lines = pk.kernel_tick_label(kernel).split("\n")
    assert len(lines) > 1, lines
    assert "".join(lines).replace(" ", "") == name.replace(" ", ""), lines
    assert all(len(line) <= experiment_tags.SHORT_NAME_MAX or " " not in line for line in lines), lines


def ink_box_in(fig: matplotlib.figure.Figure, artists: list) -> list:
    renderer = fig.canvas.get_renderer()
    return [a.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted()) for a in artists]


def test_a_key_grows_the_canvas_and_never_overprints_the_kernel_names() -> None:
    """The key's band is measured and added under the names: a fixed band put a long rotated name
    straight through the key on a narrow page, and a band too short pushed both off the canvas."""
    kernels = [f"k{i}" for i in range(12)]
    metric = speed_metric([pk.KernelCell(k, (2.0,)) for k in kernels])
    key = [matplotlib.lines.Line2D([], [], marker="o", linestyle="none", label=f"series {i}") for i in range(9)]
    bare = pk.figure_one(metric, kernels, "ci", False, "", width_in=3.3)
    keyed = pk.figure_one(metric, kernels, "ci", False, "", width_in=3.3, legend=key)
    try:
        assert keyed.get_size_inches()[1] > bare.get_size_inches()[1]
        keyed.canvas.draw()
        names = ink_box_in(keyed, [t for t in keyed.axes[0].get_xticklabels() if t.get_text()])
        (legend,) = ink_box_in(keyed, list(keyed.legends))
        width = float(keyed.get_size_inches()[0])
    finally:
        plt.close(bare)
        plt.close(keyed)
    assert min(box.y0 for box in names) >= legend.y1, (min(box.y0 for box in names), legend.y1)
    assert legend.y0 >= 0.0 and all(0.0 <= box.x0 and box.x1 <= width for box in names)


def test_every_panel_of_a_stack_ends_where_the_widest_summary_does() -> None:
    """A panel with fewer series (a compiler column spends no tokens) must not end short of the one
    above it, or the shared axis clips the upper panel's last summary slot."""
    speed = three_series_metric()
    tokens = pk.token_series_metric([pk.Series("a", (pk.KernelCell("k1", (100.0,)),), "#1155cc")], "Tokens")
    fig = pk.figure_panels([speed, tokens], ["k1", "k2"], "ci", True, "")
    try:
        top, bottom = fig.axes
        assert top.get_xlim() == bottom.get_xlim()
        assert top.get_xlim()[1] > pk.summary_slot_x(2, 2)
    finally:
        plt.close(fig)


def test_a_stated_pitch_sizes_the_canvas_to_the_kernel_axis() -> None:
    """A standalone render asks for room per kernel; the canvas grows to it instead of squeezing
    forty names into a page width."""
    kernels = [f"k{i}" for i in range(40)]
    metric = speed_metric([pk.KernelCell(k, (2.0,)) for k in kernels])
    fig = pk.figure_panels([metric], kernels, "ci", True, "", pitch_in=0.3)
    try:
        assert pk.column_pitch_in(fig.axes[0]) == pytest.approx(0.3)
    finally:
        plt.close(fig)


def test_the_speedup_axis_is_pinned_just_past_its_outermost_ticks() -> None:
    """The limits come from the cells, not from autoscaling, so the chrome is measured before any
    mark is drawn and a position means the same ratio wherever the figure is read."""
    fig, ax = plt.subplots()
    try:
        cells = [pk.KernelCell("k1", (0.3,)), pk.KernelCell("k2", (40.0,))]
        pk.style_speedup_axis(ax, cells)
        ticks = pk.speedup_yticks(cells)
        low, high = ax.get_ylim()
    finally:
        plt.close(fig)
    assert low == pytest.approx(ticks[0] / 2.0**pk.VALUE_PAD_OCTAVES)
    assert high == pytest.approx(ticks[-1] * 2.0**pk.VALUE_PAD_OCTAVES)


@pytest.mark.parametrize(
    "values",
    [
        pytest.param((120.0,), id="one-value"),
        pytest.param((100.0, 110.0), id="a-narrow-window"),
        pytest.param((15_000.0, 150_000.0), id="one-decade"),
        pytest.param((100.0, 900_000.0), id="four-decades"),
    ],
)
def test_token_limits_hold_every_value_and_at_least_two_labelled_ticks(values: tuple[float, ...]) -> None:
    """The token axis is pinned before any mark is drawn, so its limits have to hold every value;
    and a window labelling one tick (or none) leaves nothing to read a value against."""
    low, high = pk.token_limits([pk.KernelCell(f"k{i}", (v,)) for i, v in enumerate(values)])
    assert low < min(values) and max(values) < high, (low, high)
    assert len(pk.grid_125(low, high)) >= 2, (low, high)


def test_a_wide_interval_never_stretches_the_value_axis() -> None:
    """A two-repeat t-interval can span thirty octaves; pinning the axis to its ends printed ticks
    like 4.29497e+09x and flattened every mark onto one line. The whisker is cut by the frame."""
    wide = pk.KernelCell("k1", (4.0,), interval=(1e-6, 1e9))
    assert pk.speedup_yticks([wide]) == pk.speedup_yticks([pk.KernelCell("k1", (4.0,))])


def test_a_y_label_taller_than_its_panel_is_fitted_to_the_panel() -> None:
    """A rotated label taller than its panel runs past both ends of the frame, and in a stack the
    two panels' labels printed over each other in the gap."""
    long = "Speed-Up Ratio (Repository Formulation / Bare Kernel)"
    metric = pk.speedup_series_metric([pk.Series("", (pk.KernelCell("k1", (2.0,)),), "#1155cc")], long)
    fig = pk.figure_panels([metric, metric], ["k1"], "ci", True, "")
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        heights = [ax.yaxis.label.get_window_extent(renderer).height / fig.dpi for ax in fig.axes]
        words = [" ".join(ax.get_ylabel().split()) for ax in fig.axes]
    finally:
        plt.close(fig)
    assert all(height <= pk.PANEL_HEIGHT_IN for height in heights), heights
    assert words == [long, long]
