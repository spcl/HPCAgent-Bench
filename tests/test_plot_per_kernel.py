# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``hpcagent_bench.stats.figures.per_kernel`` and ``statistics/plot_per_kernel.py`` -- the per-kernel
speed-up and tokens figure: ci/box style, the log2 speed-up axis, the summary column and the
separate/stacked layout.
"""

import argparse
import importlib.util
import pathlib
import sys
import types

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from hpcagent_bench.stats import style
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
        pk.draw_box(ax, [cell], {"k1": 0}, "#1155cc")
        assert (len(ax.patches) > 0) == boxed
    finally:
        plt.close(fig)


def test_ci_style_never_draws_a_box_patch() -> None:
    """``ci`` draws a point and a whisker, never a boxplot artist -- even at a spread-worthy n."""
    cell = pk.KernelCell("k1", (1.0, 2.0, 4.0, 8.0, 16.0))
    fig, ax = plt.subplots()
    try:
        pk.draw_ci(ax, [cell], {"k1": 0}, "#1155cc", log2_space=True)
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
    speed = pk.speedup_metric([pk.KernelCell("k1", (2.0,))], "Speed-Up", "#1155cc")
    fig, ax = plt.subplots()
    try:
        pk.draw_panel(
            ax,
            speed.cells,
            ["k1"],
            "ci",
            speed.color,
            speed.log2_space,
            speed.ylabel,
            speed.summary_reducer,
            speed.summary_label,
            True,
            True,
        )
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
    speed = pk.speedup_metric([pk.KernelCell("k1", (2.0,))], "Speed-Up", "#1155cc")
    tokens = pk.token_metric([pk.KernelCell("k1", (100.0,))], "Tokens", "#cc5511")
    fig = pk.figure_stacked(speed, tokens, ["k1"], "ci", True, "demo")
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
    return (
        pk.speedup_metric(speed_cells, "Speed-Up", "#1155cc"),
        pk.token_metric(token_cells, "Tokens", "#cc5511"),
        speed_cells,
        token_cells,
    )


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

    speed = pk.speedup_metric(speed_cells, "Speed-Up", "#1155cc")
    tokens = pk.token_metric(token_cells, "Tokens", "#cc5511")
    fig = pk.figure_stacked(speed, tokens, kernels, "ci", False, "demo")
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
    speed = pk.speedup_metric(speed_cells, "Speed-Up", "#1155cc")
    for epoch, folder in (("0", "first"), ("86400", "second")):
        monkeypatch.setenv("SOURCE_DATE_EPOCH", epoch)
        fig = pk.figure_one(speed, pk.ordered_kernels(speed_cells), "box", True, "demo")
        pk.save(fig, tmp_path / folder / "figure.pdf")
    for name in ("figure.pdf", "figure.png"):
        first, second = (tmp_path / folder / name for folder in ("first", "second"))
        assert first.read_bytes() == second.read_bytes(), f"{name} depends on when it was saved"


def test_the_speedup_panel_carries_a_major_grid_and_no_minor_one() -> None:
    """Major grid only, on the measured axis. The ticks are pinned to powers of two here, so the
    grid is drawn beside them rather than through ``value_axis``, which would relocate them."""
    fig, ax = plt.subplots()
    try:
        pk.style_speedup_axis(ax, [pk.KernelCell("k1", (1.0, 2.0))])
        assert any(line.get_visible() for line in ax.yaxis.get_gridlines())
        assert not [tick for tick in ax.yaxis.get_minor_ticks() if tick.gridline.get_visible()]
        assert not [tick for tick in ax.xaxis.get_major_ticks() if tick.gridline.get_visible()]
    finally:
        plt.close(fig)


def test_the_token_panel_puts_its_measured_value_on_a_log_y_axis_with_a_major_grid() -> None:
    """A token count is a measured quantity, so it is on Y; the kernel names are the x categories."""
    fig, ax = plt.subplots()
    try:
        pk.style_token_axis(ax)
        assert ax.get_yscale() == "log"
        assert any(line.get_visible() for line in ax.yaxis.get_gridlines())
        assert not [tick for tick in ax.yaxis.get_minor_ticks() if tick.gridline.get_visible()]
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
