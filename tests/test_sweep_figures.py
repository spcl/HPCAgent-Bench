# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Sweep figures: what a slide reader relies on without being told (no title, a log speedup axis, bars that start at 1x)."""

import pathlib

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib.patches import Rectangle

from hpcagent_bench.stats.figures import sweeps


def two_panels() -> list[sweeps.Panel]:
    xs = (1e4, 1e6, 1e8)
    up = sweeps.Series("Fast", xs, (0.5, 4.0, 20.0), "#1f77b4", "o")
    down = sweeps.Series("Slow", xs, (0.3, 1.5, 3.0), "#d62728", "s", "--")
    return [
        sweeps.Panel("K = 1", (up, down), (sweeps.VLine(1e6, "Crossover", "#000000"),)),
        sweeps.Panel("K = 4", (up, down)),
    ]


def test_a_speedup_row_has_log_axes_and_no_title():
    fig = sweeps.speedup_panels(two_panels(), xlabel="N (elements)", ylabel="Speedup", foot="hardware line")
    axes = fig.axes[:2]
    assert [ax.get_xscale() for ax in axes] == ["log", "log"]
    assert [ax.get_yscale() for ax in axes] == ["log", "log"]
    assert all(ax.get_title() == "" for ax in axes)
    assert fig._suptitle is None


def test_the_legend_lists_each_label_once_across_panels():
    fig = sweeps.speedup_panels(two_panels(), xlabel="N", ylabel="Speedup")
    legend = fig.legends[0]
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["Fast", "Slow", "Crossover"], labels


def test_a_series_with_an_empty_label_stays_out_of_the_key():
    hidden = sweeps.Series("", (1e4, 1e6), (1.0, 2.0), "#000000", "D", line=False)
    fig = sweeps.speedup_panels([sweeps.Panel("P", (two_panels()[0].series[0], hidden))], xlabel="N", ylabel="S")
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["Fast"]


def test_the_footer_names_the_hardware_and_wraps_to_the_canvas():
    text = "AMD Instinct MI300A " + "x" * 400
    fig = sweeps.speedup_panels(two_panels(), xlabel="N", ylabel="S", foot=text)
    foot = [t for t in fig.texts if t.get_text().replace("\n", "").startswith("AMD Instinct")]
    assert foot, [t.get_text() for t in fig.texts]
    assert "\n" in foot[0].get_text()


def test_bars_start_at_one_because_a_log_bar_cannot_start_at_zero():
    panels = [sweeps.BarPanel("CPU", (("a", (2.0, 4.0)), ("b", (None, 3.0))))]
    fig = sweeps.tile_bars(panels, ["16", "32"], ylabel="Speedup", legend_title="Tile size")
    bars = [p for p in fig.axes[0].patches if isinstance(p, Rectangle)]
    assert len(bars) == 3, "a value of None draws no bar"
    assert all(p.get_y() == pytest.approx(1.0) for p in bars)
    assert sorted(round(p.get_y() + p.get_height(), 6) for p in bars) == [2.0, 3.0, 4.0]


def test_a_wavefront_cell_is_labelled_with_its_anti_diagonal():
    fig = sweeps.wavefront_grid(4, ["line one", "line two"])
    labels = sorted(t.get_text() for t in fig.axes[0].texts if t.get_text().isdigit())
    want = sorted(str(i + j) for i in range(4) for j in range(4))
    assert labels == want


def test_the_score_histogram_holds_every_task_once_and_marks_one_x():
    values = [0.0] * 90 + [0.3] * 5 + [-3.0] * 5
    fig = sweeps.score_histogram(values, xlabel="Task score", note="false credit 10%")
    ax = fig.axes[0]
    shares = sum(p.get_height() for p in ax.patches)
    assert shares == pytest.approx(100.0), shares
    assert any(np.isclose(line.get_xdata()[0], 0.0) for line in ax.lines)


def test_curve_panels_give_each_panel_its_own_y_label():
    panels = [
        sweeps.CurvePanel("a", "Spread", (sweeps.Series("m = 4", (10, 100), (0.2, 0.1), "#1f77b4"),)),
        sweeps.CurvePanel("b", "Rate", (sweeps.Series("m = 4", (10, 100), (0.2, 0.1), "#1f77b4"),), ylim=(0.0, 0.5)),
    ]
    fig = sweeps.curve_panels(panels, xlabel="Runs per answer", x_ticks=(10, 100), x_formatter=lambda v, p=0: f"{v:g}")
    assert [ax.get_ylabel() for ax in fig.axes[:2]] == ["Spread", "Rate"]
    assert fig.axes[1].get_ylim() == (0.0, 0.5)


def test_saving_writes_a_png_for_the_deck_and_a_pdf_for_the_paper(tmp_path: pathlib.Path):
    fig = sweeps.speedup_panels(two_panels(), xlabel="N", ylabel="S")
    png, pdf = sweeps.save_figure(fig, tmp_path / "sub" / "fig")
    assert png.suffix == ".png" and png.stat().st_size > 1000
    assert pdf.suffix == ".pdf" and pdf.read_bytes()[:4] == b"%PDF"
