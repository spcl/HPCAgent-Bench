# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``statistics/plot_repo_vs_kernel.py`` -- the repo-against-bare-kernel ratio figure.

This figure used to live in the paper repository as two scripts of its own, with their own geomean,
their own served/undelivered policy, their own two hues and their own model names, and they read the
BILLED per-turn token count where every other figure reads the task row's EFFECTIVE total. What is
asserted here is that the library decides all five: the drawing conventions
(``docs/plotting.md``), the served policy (``population.kernel_answers``), the token quantity
(``population.kernel_tokens``), and the colour and the name (``palette``, ``experiment_tags``).
"""

import importlib.util
import math
import pathlib
import sys
import types

import matplotlib.axes
import matplotlib.collections
import matplotlib.colors
import matplotlib.figure
import pandas as pd
import pytest

from hpcagent_bench import experiment_tags
from hpcagent_bench.stats import palette, population
from hpcagent_bench.stats import style as plotstyle
from hpcagent_bench.stats.figures import kernel_comparison

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Models registered in the tag registry, so the palette resolves them instead of hashing them.
TREATED_ARM: str = "git-scicomp-qwen38-c-repo"
CONTROL_ARM: str = "git-scicomp-qwen38-c-kernel"
SECOND_TREATED: str = "git-scicomp-kimi27sglang-c-repo"
SECOND_CONTROL: str = "git-scicomp-kimi27sglang-c-kernel"


def load_script() -> types.ModuleType:
    """Import ``statistics/plot_repo_vs_kernel.py`` as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("plot_repo_vs_kernel", REPO / "statistics" / "plot_repo_vs_kernel.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plot = load_script()


def submission_rows(arm: str, speedups: dict[str, float]) -> list[dict[str, object]]:
    """One graded episode per (arm, kernel), the columns ``population.kernel_answers`` reads."""
    return [
        {
            "run_root": "j1",
            "job": "j1",
            "run_id": f"{arm}-{kernel}",
            "arm": arm,
            "record": "submission",
            "benchmark": kernel,
            "speedup": speedup,
            "baseline_ns": 1000.0,
            "native_ns": 1000.0 / speedup,
            "baseline": "c-autopar",
            "suspect": 0,
            "ts_ms": 1,
            "attempt_index": 1,
            "timing_reduction": "mwd-v2",
        }
        for kernel, speedup in speedups.items()
    ]


def task_rows(arm: str, tokens: dict[str, float]) -> list[dict[str, object]]:
    """One ``record=task`` row per (arm, kernel): the task's own EFFECTIVE total (spec T1-T4)."""
    return [
        {
            "run_root": "j1",
            "job": "j1",
            "run_id": f"{arm}-{kernel}",
            "arm": arm,
            "record": "task",
            "benchmark": kernel,
            "tokens": total,
            "ts_ms": 1,
        }
        for kernel, total in tokens.items()
    ]


def call_rows(arm: str, tokens: dict[str, float]) -> list[dict[str, object]]:
    """One ``record=call`` row per (arm, kernel): a running count mid-task, never a cost (T4)."""
    return [
        {
            "run_root": "j1",
            "job": "j1",
            "run_id": f"{arm}-{kernel}",
            "arm": arm,
            "record": "call",
            "benchmark": kernel,
            "tokens": running,
            "ts_ms": 1,
            "attempt_index": 1,
        }
        for kernel, running in tokens.items()
    ]


def one_pair_frame() -> pd.DataFrame:
    """Two arms over three kernels. ``k3`` is SERVED by both and DELIVERED by neither, which is the
    non-delivery every aggregate has to enter at 1x rather than drop."""
    rows: list[dict[str, object]] = []
    rows += submission_rows(CONTROL_ARM, {"k1": 2.0, "k2": 4.0})
    rows += submission_rows(TREATED_ARM, {"k1": 4.0, "k2": 2.0})
    rows += task_rows(CONTROL_ARM, {"k1": 1000.0, "k2": 1000.0, "k3": 1000.0})
    rows += task_rows(TREATED_ARM, {"k1": 2000.0, "k2": 500.0, "k3": 1000.0})
    return pd.DataFrame(rows)


def one_pair_series() -> tuple[pd.DataFrame, list[kernel_comparison.Series], dict[str, dict[str, bool]]]:
    frame = one_pair_frame()
    return frame, *plot.build_series(frame, [(TREATED_ARM, CONTROL_ARM)], "repo", "latest")


def test_a_served_kernel_neither_arm_delivered_enters_the_ratio_at_one_and_is_flagged() -> None:
    """The two paper-repo scripts disagreed about this number: the extractor wrote 0.0 for a
    non-delivery and the plotter read 1.0. ``population.kernel_answers`` under the served policy is
    the one place it is decided -- 1.0, which is what the failed episode left standing -- and the
    flag is what puts the cross on the mark."""
    frame, series_list, flags = one_pair_series()
    series = series_list[0]

    assert series.values["k3"] == pytest.approx(1.0)
    assert flags[TREATED_ARM]["k3"] is False
    assert flags[TREATED_ARM]["k1"] is True


def test_the_token_ratio_is_the_effective_task_total_not_the_billed_per_turn_count() -> None:
    """Every other figure costs a kernel by its task row (spec T1-T4, ``population.kernel_tokens``).
    The paper-repo script read ``tokens.json``'s ``tokens``, the per-turn sum, which charges a
    173-turn episode for its prompt 173 times -- a different quantity from the one beside it."""
    rows = list(one_pair_frame().to_dict("records"))
    rows += call_rows(TREATED_ARM, {"k1": 999999.0})
    rows += call_rows(CONTROL_ARM, {"k1": 3.0})

    series_list, flags = plot.build_series(pd.DataFrame(rows), [(TREATED_ARM, CONTROL_ARM)], "repo", "latest")

    assert series_list[0].tokens["k1"] == pytest.approx(2.0)


def test_the_series_wears_the_registry_hue_and_the_registry_model_name() -> None:
    """Colour is the MODEL and shape is the INTERVENTION, both from ``registry.yaml`` -- the
    inversion :mod:`hpcagent_bench.stats.figures.efficacy` draws under (one series is already one
    intervention, so colour is free for the models sharing it). The paper-repo copy carried two
    hues of its own and a name map that called ``kimi27`` "Kimi K2.7" where the registry says
    "Kimi-K2.7-Code"."""
    frame, series_list, flags = one_pair_series()
    series = series_list[0]

    assert series.color == palette.model_color("qwen38")
    assert series.marker == palette.packet_marker("repo")
    assert series.label == experiment_tags.model_name("qwen38")


def one_pair_figure() -> tuple[matplotlib.figure.Figure, list[kernel_comparison.Series], list[str]]:
    frame, series_list, flags = one_pair_series()
    kernels = plot.kernels_of(series_list)
    return plot.figure(series_list, flags, kernels, "title", "repo", "kernel"), series_list, kernels


def test_the_measured_value_is_on_the_y_axis_of_both_panels() -> None:
    """Rule one: the ratio is the measured quantity and is on Y; X carries the kernel NAMES, which
    are categories, so it stays linear and unscaled."""
    import matplotlib.pyplot as plt_local

    fig, series_list, kernels = one_pair_figure()
    try:
        axes = fig.axes
        assert [ax.get_yscale() for ax in axes] == ["log", "log"]
        assert [ax.get_xscale() for ax in axes] == ["linear", "linear"]
        assert all("Ratio" in ax.get_ylabel() for ax in axes)
        assert [tick.get_text() for tick in axes[-1].get_xticklabels()] == list(kernels)
        assert all(tick.get_rotation() == 90 for tick in axes[-1].get_xticklabels())
    finally:
        plt_local.close(fig)


def test_neither_panel_enables_a_minor_grid() -> None:
    """Major grid only, on the value axis alone. A minor line is a second grid at a second weight,
    and a reduced panel then reads as a texture the marks sit on."""
    import matplotlib.pyplot as plt_local

    fig, series_list, kernels = one_pair_figure()
    try:
        for ax in fig.axes:
            assert any(line.get_visible() for line in ax.yaxis.get_gridlines())
            assert not [tick for tick in ax.yaxis.get_minor_ticks() if tick.gridline.get_visible()]
            assert not [tick for tick in ax.xaxis.get_minor_ticks() if tick.gridline.get_visible()]
    finally:
        plt_local.close(fig)


def test_the_legend_is_drawn_once_on_the_figure_and_never_on_an_axes() -> None:
    """One key for the whole figure (``style.legend_below``); a key per panel invites reading the
    two panels as different sets of series when they draw the same ones."""
    import matplotlib.pyplot as plt_local

    fig, series_list, kernels = one_pair_figure()
    try:
        assert len(fig.legends) == 1
        assert all(ax.get_legend() is None for ax in fig.axes)
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
    finally:
        plt_local.close(fig)
    assert plotstyle.NOT_DELIVERED_LABEL in labels, labels
    assert experiment_tags.model_name("qwen38") in labels, labels
    assert sum(1 for label in labels if label.startswith("Geomean")) == len(plot.PANELS), labels


def marks_at(ax: matplotlib.axes.Axes, x: float, y: float) -> list[matplotlib.collections.PathCollection]:
    """Every scatter collection drawn at one (x, y): the white halo, the mark, and any cross."""
    from matplotlib.collections import PathCollection

    return [
        collection
        for collection in ax.collections
        if isinstance(collection, PathCollection)
        for offset_x, offset_y in collection.get_offsets()
        if offset_x == pytest.approx(x) and offset_y == pytest.approx(y)
    ]


def test_a_point_no_arm_delivered_carries_a_cross_at_its_own_ratio() -> None:
    """Rule six. The placeholder is a real RATIO here (a non-delivery over a real answer is not
    1.0), so the cross goes on a DRAWN mark rather than on an absent one -- the placeholder would
    otherwise be indistinguishable from a measured 1.0x, and dropping the mark would hide the
    failure entirely. A delivered kernel in the same panel draws no cross."""
    import matplotlib.markers
    import matplotlib.pyplot as plt_local

    fig, series_list, kernels = one_pair_figure()
    try:
        assert kernels == ["k1", "k2", "k3"]
        placeholder = marks_at(fig.axes[0], 2.0, 1.0)
        measured = marks_at(fig.axes[0], 0.0, 2.0)
        cross = matplotlib.markers.MarkerStyle("x").get_path()
        drawn = [collection.get_paths()[0].vertices.shape for collection in placeholder]
        hues = {matplotlib.colors.to_hex(rgba) for collection in placeholder for rgba in collection.get_facecolor()}
    finally:
        plt_local.close(fig)
    # the white halo, the mark itself and the cross style.point_mark lays over it
    assert len(placeholder) == 3, placeholder
    assert cross.vertices.shape in drawn, drawn
    assert palette.model_color("qwen38") in hues
    # the delivered kernel keeps the halo and the mark, and gains nothing
    assert len(measured) == 2, measured


def test_the_table_carries_every_drawn_ratio_and_the_geomean_behind_the_summary_mark() -> None:
    """A number quoted from a chart cannot be checked against the chart, so every ratio and every
    geomean leaves as a row -- with the interval method and the n the geomean was taken over."""
    frame, series_list, flags = one_pair_series()

    table = plot.table_rows(series_list, flags)

    speed = table[(table.panel == plot.SPEEDUP_PANEL) & (table.arm == TREATED_ARM)]
    kernels = speed[speed.kernel != "GEOMEAN"]
    assert sorted(kernels.kernel) == ["k1", "k2", "k3"]
    # 2.0 * 0.5 * 1.0, geometric mean 1.0
    geomean = speed[speed.kernel == "GEOMEAN"].iloc[0]
    assert float(geomean.ratio) == pytest.approx(1.0)
    assert int(geomean.n) == 3
    assert str(geomean.method)
    assert not bool(kernels[kernels.kernel == "k3"].iloc[0].delivered)


def test_the_script_writes_the_table_beside_the_figure(tmp_path: pathlib.Path) -> None:
    """``plotstyle.save`` decides the stem, and the CSV is written against it, so the three files
    of one figure cannot drift apart."""
    obs = tmp_path / "obs.csv"
    one_pair_frame().to_csv(obs, index=False)
    out = tmp_path / "figures" / "repo_vs_kernel.pdf"

    rc = plot.run([obs], [(TREATED_ARM, CONTROL_ARM)], "repo", "kernel", "", out, False, "latest")

    assert rc == 0
    assert out.exists()
    assert out.with_suffix(".png").exists()
    written = out.with_suffix(".csv")
    assert written.exists()
    assert written.read_text().startswith("#")


def test_two_pairs_draw_in_registry_model_order_with_two_hues_and_one_shape() -> None:
    """Colour separates the models; shape stays the intervention's, so the figure spends one channel
    on one entity. Registry order, never the order the pairs happened to be typed in."""
    rows = list(one_pair_frame().to_dict("records"))
    rows += submission_rows(SECOND_CONTROL, {"k1": 2.0})
    rows += submission_rows(SECOND_TREATED, {"k1": 6.0})
    rows += task_rows(SECOND_CONTROL, {"k1": 1000.0})
    rows += task_rows(SECOND_TREATED, {"k1": 1000.0})
    pairs = [(SECOND_TREATED, SECOND_CONTROL), (TREATED_ARM, CONTROL_ARM)]

    series_list, flags = plot.build_series(pd.DataFrame(rows), pairs, "repo", "latest")

    assert [series.model for series in series_list] == palette.in_order(["kimi27sglang", "qwen38"], "models")
    assert len({series.color for series in series_list}) == 2
    assert len({series.marker for series in series_list}) == 1


def test_an_unusable_pair_is_named_on_stderr_and_skipped(capsys: pytest.CaptureFixture[str]) -> None:
    """A pair with no rows is reported, never dropped in silence."""
    series_list, flags = plot.build_series(one_pair_frame(), [("no-such-arm", CONTROL_ARM)], "repo", "latest")

    assert series_list == []
    assert "no-such-arm" in capsys.readouterr().err


def test_the_interval_note_names_the_method_and_the_n_it_is_over() -> None:
    """Two differently derived intervals drawn the same way are two claims a reader cannot separate,
    so the method and the n go in the key (``summary.geomean_interval``)."""
    note = plot.interval_note([2.0, 0.5, 1.0])

    assert "n=3" in note
    assert "bootstrap" in note or "log-t" in note
    assert "95%" in note


def test_the_non_delivery_value_is_the_populations_own_constant() -> None:
    """1.0 is not a number this figure chose; it is what the served policy enters."""
    assert population.NOT_DELIVERED == 1.0
    assert plot.kernel_comparison.MISSING_MARKER_Y == population.NOT_DELIVERED
    assert not math.isnan(population.NOT_DELIVERED)
