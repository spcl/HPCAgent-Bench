# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The per-arm figures draw from the tables stats.arms writes, in registered colours."""

import logging
import pathlib

import matplotlib.figure
import pandas as pd
import pytest

from hpcagent_bench.stats import palette
from hpcagent_bench.stats import style as plotstyle
from hpcagent_bench.stats.figures import arms


def arm_table() -> pd.DataFrame:
    """The columns ``stats.arms.per_arm_summary`` writes that the figure reads, keyed as it keys them."""
    rows = [
        ("llr40-qwen38-c", "c", 4.0, 3.0, 2.0, 4.5, 40, 30),
        ("llr40-qwen38-fortran", "fortran", 2.0, 1.5, 1.2, 2.4, 40, 22),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "arm",
            "language",
            "geomean_solved",
            "geomean_served",
            "geomean_solved_low",
            "geomean_solved_high",
            "n_served",
            "n_solved",
        ],
    )
    return frame.assign(baseline="numba").set_index(["arm", "baseline"])


def paired_table() -> pd.DataFrame:
    """``stats.arms.per_language_kernel`` keyed on ``(baseline, benchmark)``, one kernel with no Fortran answer."""
    frame = pd.DataFrame(
        {
            "baseline": "numba",
            "benchmark": ["k1", "k2", "k3"],
            "c_best_su": [2.0, 5.0, 1.5],
            "fortran_best_su": [3.0, 4.0, None],
        }
    )
    return frame.set_index(["baseline", "benchmark"])


def test_both_arm_figures_render_in_registered_language_colours(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=palette.LOG.name):
        arms.figure_arms(arm_table(), "numba", tmp_path)
        arms.figure_paired(paired_table(), "numba", tmp_path)
    assert "registry.yaml" not in caplog.text
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == [
        "per_arm_geomean_numba.pdf",
        "per_arm_geomean_numba.png",
        "per_kernel_c_vs_fortran_numba.pdf",
        "per_kernel_c_vs_fortran_numba.png",
    ]


def long_paired_table() -> pd.DataFrame:
    """:func:`paired_table` with kernel names long enough to force a wide rotated tick label --
    the bottom band a fixed fraction used to clip must grow to hold them instead."""
    frame = pd.DataFrame(
        {
            "baseline": "numba",
            "benchmark": [f"a-rather-long-canonical-kernel-name-{i:02d}" for i in range(3)],
            "c_best_su": [2.0, 5.0, 1.5],
            "fortran_best_su": [3.0, 4.0, None],
        }
    )
    return frame.set_index(["baseline", "benchmark"])


def captured_paired_figure(
    monkeypatch: pytest.MonkeyPatch, paired: pd.DataFrame, tmp_path: pathlib.Path
) -> matplotlib.figure.Figure:
    """:func:`~hpcagent_bench.stats.figures.arms.figure_paired` with the write intercepted, so a
    test can measure the canvas ``finish`` actually built instead of the file it would have
    written."""
    import matplotlib.pyplot as plt

    captured: list[plt.Figure] = []

    def fake_save(fig: plt.Figure, stem: pathlib.Path, formats: tuple = ("pdf", "png"), fixed: bool = False):
        del formats, fixed
        captured.append(fig)
        return stem

    monkeypatch.setattr(arms.style, "save", fake_save)
    arms.figure_paired(paired, "numba", tmp_path)
    return captured[0]


def test_long_kernel_names_grow_the_canvas_instead_of_clipping_the_tick(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """``finish`` used to reserve a fixed ``bottom=0.34`` band for the rotated kernel names; a name
    long enough to need more than that printed off the bottom of the canvas. The band is measured
    off the rendered tick labels now, so a longer name grows the canvas instead."""
    import matplotlib.pyplot as plt

    short = captured_paired_figure(monkeypatch, paired_table(), tmp_path)
    short_height = float(short.get_size_inches()[1])
    plt.close(short)
    long_fig = captured_paired_figure(monkeypatch, long_paired_table(), tmp_path)
    long_height = float(long_fig.get_size_inches()[1])
    plt.close(long_fig)
    assert long_height > short_height, (short_height, long_height)


def test_the_note_and_the_legend_never_overlap_the_tick_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The caveat note and the legend used to be placed inside one fixed ``bottom=0.34`` band with
    the rotated kernel names; nothing stopped the three from printing through one another. Each
    band is measured now, so no two of them may overlap."""
    import matplotlib.pyplot as plt

    fig = captured_paired_figure(monkeypatch, paired_table(), tmp_path)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    tick_boxes = [label.get_window_extent(renderer) for label in ax.get_xticklabels() if label.get_text()]
    note_boxes = [text.get_window_extent(renderer) for text in fig.texts if text.get_text()]
    legend_box = fig.legends[0].get_window_extent(renderer)
    assert note_boxes, "figure_paired always draws its UNVETTED caveat note"
    for note_box in note_boxes:
        assert not legend_box.overlaps(note_box)
        assert not any(note_box.overlaps(tick_box) for tick_box in tick_boxes)
    assert not any(legend_box.overlaps(tick_box) for tick_box in tick_boxes)
    plt.close(fig)


def drawn_figures(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> list:
    """Both per-arm figures, captured just before ``style.save`` would close them."""
    import matplotlib.pyplot as plt

    captured: list[plt.Figure] = []

    def fake_save(
        fig: plt.Figure, stem: pathlib.Path, formats: tuple = ("pdf", "png"), fixed: bool = False
    ) -> pathlib.Path:
        del formats, fixed
        captured.append(fig)
        return stem

    monkeypatch.setattr(arms.style, "save", fake_save)
    arms.figure_arms(arm_table(), "numba", tmp_path)
    arms.figure_paired(paired_table(), "numba", tmp_path)
    return captured


def test_the_measured_speedup_is_on_the_y_axis_of_both_figures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Rule one: a speed-up is a measured quantity and stays on Y, log-scaled. X carries the
    CATEGORY (the arm or the kernel), which is why it is linear and ticked with names."""
    import matplotlib.pyplot as plt

    figures = drawn_figures(monkeypatch, tmp_path)
    try:
        assert len(figures) == 2
        for fig in figures:
            ax = fig.axes[0]
            assert ax.get_yscale() == "log"
            assert ax.get_xscale() == "linear"
            labels = [tick.get_text() for tick in ax.get_xticklabels()]
            assert labels and all(not label.replace(".", "").isdigit() for label in labels)
    finally:
        for fig in figures:
            plt.close(fig)


def test_neither_figure_enables_a_minor_grid_or_an_axes_legend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Major grid only (rule four), and one legend on the FIGURE, never ``ax.legend`` (rule five)."""
    import matplotlib.pyplot as plt

    figures = drawn_figures(monkeypatch, tmp_path)
    try:
        for fig in figures:
            ax = fig.axes[0]
            assert any(line.get_visible() for line in ax.yaxis.get_gridlines())
            assert not [tick for tick in ax.yaxis.get_minor_ticks() if tick.gridline.get_visible()]
            assert ax.get_legend() is None
            assert len(fig.legends) == 1
    finally:
        for fig in figures:
            plt.close(fig)


def test_tick_and_label_type_comes_from_the_shared_style_scale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """No fontsize literal: every text size drawn traces back to a ``style`` constant."""
    import matplotlib.pyplot as plt

    known = {plotstyle.TITLE_PT, plotstyle.SUBTITLE_PT, plotstyle.LABEL_PT, plotstyle.TICK_PT, plotstyle.ANNOTATION_PT}
    figures = drawn_figures(monkeypatch, tmp_path)
    try:
        for fig in figures:
            ax = fig.axes[0]
            for tick in ax.get_xticklabels():
                assert tick.get_fontsize() in known
    finally:
        for fig in figures:
            plt.close(fig)
