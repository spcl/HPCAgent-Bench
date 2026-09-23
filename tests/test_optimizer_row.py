# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The score-only optimizer row, and the shared drawing fixes it was built with.

The row puts LLM arms beside standalone compilers on one track, so the properties that matter are
the ones that keep the two comparable: both are scored over the SAME roster with an unanswered
kernel at 1x, a compiler's label says which device it ran on, and the numbers behind every mark
ship beside the figure.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from hpcagent_bench.stats import population, style
from hpcagent_bench.stats.figures import optimizers, signed

ROSTER: tuple[str, ...] = ("k1", "k2", "k3")


def canon_table(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"run": "r1", "column": column, "kernel": kernel, "median_ms": ms, "validated": "True"}
         for column, kernel, ms in rows]
    )  # fmt: skip


def episode_row(arm: str, benchmark: str, speedup: float) -> dict[str, object]:
    return {
        "run_root": "j1", "job": "j1", "run_id": f"{arm}-{benchmark}-1", "arm": arm, "record": "submission",
        "benchmark": benchmark, "speedup": speedup, "baseline_ns": 1.0e6, "native_ns": 1.0e6 / speedup,
        "baseline": "numba", "suspect": 0, "ts_ms": 1, "attempt_index": 1, "timing_reduction": "mwd-v2",
    }  # fmt: skip


@pytest.fixture(name="canon")
def canon_fixture() -> pd.DataFrame:
    """numba times every kernel; the CPU DaCe column all three, the GPU one two, Pluto one."""
    return canon_table(
        [
            ("numba", "k1", 100.0), ("numba", "k2", 200.0), ("numba", "k3", 100.0),
            ("dace_cpu_canonicalize", "k1", 10.0), ("dace_cpu_canonicalize", "k2", 20.0),
            ("dace_cpu_canonicalize", "k3", 50.0),
            ("dace_gpu_canonicalize", "k1", 1.0), ("dace_gpu_canonicalize", "k2", 2.0),
            ("pluto", "k1", 50.0),
        ]
    )  # fmt: skip


@pytest.fixture(name="observations")
def observations_fixture() -> pd.DataFrame:
    """One arm that answered k1 and k2 and has no row at all for k3."""
    arm = "cpf-llr-focus40-qwen38-c"
    return pd.DataFrame([episode_row(arm, "k1", 4.0), episode_row(arm, "k2", 16.0)])


def test_an_arm_is_scored_over_the_roster_with_a_missing_kernel_at_1x(observations: pd.DataFrame) -> None:
    """A compiler that declined a kernel enters at 1x over the full roster; an LLM arm that never
    touched one has to enter the same way, or the LLM is averaged over the kernels it happened to
    answer and reads better than the compiler on the same forty."""
    mark = optimizers.arm_mark(observations, "cpf-llr-focus40-qwen38-c", "qwen38", ROSTER)
    assert set(mark.ratios) == set(ROSTER)
    assert mark.ratios["k3"] == population.NOT_DELIVERED
    assert mark.delivered["k3"] is False
    assert mark.solved == 2
    assert mark.interval().point == pytest.approx((4.0 * 16.0 * 1.0) ** (1.0 / 3.0))


def test_a_compiler_mark_is_labelled_with_its_device(canon: pd.DataFrame) -> None:
    """The same optimizer appears on the CPU and the GPU track. Labelled by its optimizer name both
    marks would read "Canonical Parallel Form" and the legend could not say which is which."""
    cpu = optimizers.compiler_mark(canon, "dace_cpu_canonicalize", ROSTER, "numba")
    gpu = optimizers.compiler_mark(canon, "dace_gpu_canonicalize", ROSTER, "numba")
    assert cpu.label != gpu.label
    assert "CPU" in cpu.label and "GPU" in gpu.label


def test_a_compiler_that_declined_kernels_is_counted_not_dropped(canon: pd.DataFrame) -> None:
    """Pluto timed one of three kernels. Its geomean is over three, with two at 1x -- the figure
    must not report Pluto's one success as its track score."""
    mark = optimizers.compiler_mark(canon, "pluto", ROSTER, "numba")
    assert mark.solved == 1 and len(mark.ratios) == 3
    assert mark.interval().point == pytest.approx(2.0 ** (1.0 / 3.0))


def test_the_row_writes_the_figure_and_the_numbers_behind_it(
    canon: pd.DataFrame, observations: pd.DataFrame, tmp_path: pathlib.Path
) -> None:
    """A figure that scores an unanswered kernel at 1x hides how many it answered, so the table
    beside it has to carry the solved count for every mark."""
    panel = optimizers.OptimizerPanel(
        "Loop Reasoning CPU (LLR)", "Numba",
        (
            optimizers.arm_mark(observations, "cpf-llr-focus40-qwen38-c", "qwen38", ROSTER),
            optimizers.compiler_mark(canon, "pluto", ROSTER, "numba"),
        ),
    )  # fmt: skip
    # Into a directory that does not exist yet: the table used to be written before anything
    # created it, so the documented command failed on a fresh checkout.
    stem = optimizers.figure_optimizer_row([panel, panel, panel], tmp_path / "figures" / "row.pdf")
    assert stem.with_suffix(".pdf").is_file() and stem.with_suffix(".png").is_file()
    table = pd.read_csv(stem.with_suffix(".csv"))
    assert list(table.solved[:2]) == [2, 1]
    assert set(table.kernels) == {3}


def test_two_device_variants_of_one_optimizer_get_distinct_row_labels(canon: pd.DataFrame) -> None:
    """The compiler per-kernel figure labels a canon row by its optimizer, and the registry aliases
    both DaCe device variants to one. Drawn together they printed one name twice in the legend."""
    rows = signed.llr40_rows(canon, None, ROSTER, canon_columns=("dace_cpu_canonicalize", "dace_gpu_canonicalize"))
    labels = [row.label for row in rows]
    assert len(set(labels)) == 2, labels


def test_one_device_variant_keeps_its_optimizer_name(canon: pd.DataFrame) -> None:
    """The fallback fires only on a collision; a figure drawing one variant is unchanged."""
    (row,) = signed.llr40_rows(canon, None, ROSTER, canon_columns=("dace_cpu_canonicalize",))
    assert row.label == "Canonical Parallel Form"


@pytest.mark.parametrize(
    ("value", "want"), [(6.27, "6.3x"), (32.45, "32.5x"), (0.928, "0.9x"), (1.0, "1.0x"), (0.04, "0.04x")]
)
def test_a_speedup_value_is_printed_to_one_decimal(value: float, want: str) -> None:
    """One decimal is what a reader quotes; below 0.1x one decimal would print a real slowdown as
    0.0x, so those keep a significant figure."""
    assert style.ratio_label(value) == want


def test_an_undelivered_placeholder_is_drawn_hollow() -> None:
    """A cross in the series' colour on a FILLED mark of that colour is invisible: 34 of 40 PPCG
    placeholders read as measured 1x results. The placeholder's face must be empty."""
    fig, ax = plt.subplots()
    try:
        style.point_mark(ax, 0.0, 1.0, "#cc79a7", "v", True, delivered=False)
        mark = ax.collections[1]  # the white halo is [0], the mark itself [1]
        assert len(mark.get_facecolors()) == 0 or mark.get_facecolors()[0][3] == 0.0
    finally:
        plt.close(fig)


def optimizer_mark(key: str, ratio: float) -> optimizers.OptimizerMark:
    """A one-kernel mark at ``ratio``, for tests that only care about the axis it forces."""
    return optimizers.OptimizerMark(key, key, key, "#009e73", "o", {"k1": ratio}, {"k1": True})


def captured_optimizer_row(monkeypatch: pytest.MonkeyPatch, panels: list, tmp_path: pathlib.Path) -> plt.Figure:
    """``figure_optimizer_row`` with the write intercepted, so a test can measure the canvas
    :func:`~hpcagent_bench.stats.figures.optimizers.figure_optimizer_row` actually built instead of
    the file it would have written."""
    captured: list[plt.Figure] = []

    def fake_save(fig: plt.Figure, stem: pathlib.Path, formats: tuple = ("pdf", "png"), fixed: bool = False):
        del formats, fixed
        captured.append(fig)
        return stem

    monkeypatch.setattr(optimizers.style, "save", fake_save)
    optimizers.figure_optimizer_row(panels, tmp_path / "row.pdf")
    return captured[0]


def test_a_wide_ranging_axis_widens_the_left_margin_instead_of_clipping_its_ticks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The left margin used to be ``config.left_chrome_in``, one fixed inch regardless of what the
    shared log2 axis ended up ticked in. A geomean near 1x prints short ticks ("1x", "2x"); one
    fourteen octaves down prints "0.0000610x" -- the margin has to grow to hold it or the number
    prints outside the canvas."""
    narrow = optimizers.OptimizerPanel("T", "Numba", (optimizer_mark("m", 1.02),))
    wide = optimizers.OptimizerPanel("T", "Numba", (optimizer_mark("m", 2.0**-14),))
    narrow_fig = captured_optimizer_row(monkeypatch, [narrow], tmp_path)
    narrow_left = narrow_fig.axes[0].get_position().x0 * narrow_fig.get_size_inches()[0]
    plt.close(narrow_fig)
    wide_fig = captured_optimizer_row(monkeypatch, [wide], tmp_path)
    wide_left = wide_fig.axes[0].get_position().x0 * wide_fig.get_size_inches()[0]
    plt.close(wide_fig)
    assert wide_left > narrow_left, (narrow_left, wide_left)


def test_the_legend_never_overlaps_a_panels_tick_names(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """The legend used to hang below the canvas edge inside a fixed ``AXIS_BAND_IN`` band; a panel
    with several optimizers and a long baseline name could overrun it and print the key through the
    "1x = baseline" tick row instead of under it."""
    panel = optimizers.OptimizerPanel(
        "A Long Panel Subtitle", "A Rather Long Baseline Name For The 1x Note",
        tuple(optimizer_mark(f"m{i}", 1.0 + 0.1 * i) for i in range(5)),
    )  # fmt: skip
    fig = captured_optimizer_row(monkeypatch, [panel, panel], tmp_path)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = fig.legends[0].get_window_extent(renderer)
    for ax in fig.axes:
        for label in ax.get_xticklabels():
            if label.get_text():
                assert not legend_box.overlaps(label.get_window_extent(renderer))
    plt.close(fig)
