# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The canon speed-up figure: baseline selection, missing-kernel reporting, and reproducibility."""

import contextlib
import csv
import importlib.util
import pathlib
import sqlite3
import sys

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")  # before any pyplot import -- a headless test must never touch a display
import matplotlib.pyplot as plt  # noqa: E402

from hpcagent_bench import experiment_tags, paths

SPEC = importlib.util.spec_from_file_location("plot_canon_speedup", paths.ROOT / "statistics" / "plot_canon_speedup.py")
plot_canon_speedup = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plot_canon_speedup
SPEC.loader.exec_module(plot_canon_speedup)

#: The ``canon`` table's columns, in the order scripts/collect_canon.py writes them.
CANON_FIELDS = ("run", "column", "kernel", "preset", "datatype", "median_ms", "validated")


def make_db(db_path: pathlib.Path, rows: list[tuple]) -> None:
    """A minimal ``canon`` table, built directly rather than through collect_canon.py: the plot
    script's contract is the table shape, not the collector that happens to produce it."""
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "CREATE TABLE canon (run TEXT, column TEXT, kernel TEXT, preset TEXT, datatype TEXT, "
            "median_ms REAL, validated TEXT)"
        )
        conn.executemany(f"INSERT INTO canon ({', '.join(CANON_FIELDS)}) VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        conn.commit()


def row(column: str, kernel: str, median_ms: float, validated: str = "True") -> tuple:
    return ("run1", column, kernel, "fuzzed", "float64", median_ms, validated)


@pytest.mark.parametrize(
    ("baseline", "column", "expected"),
    [
        ("numba", "cc", [0.5, 2.0]),  # numba/cc: 100/200, 50/25
        ("cc", "numba", [2.0, 0.5]),  # cc/numba: 200/100, 25/50
    ],
)
def test_the_speedup_table_uses_the_chosen_baseline_as_the_divisor(
    baseline: str, column: str, expected: list[float]
) -> None:
    """Swapping --baseline must swap which column is the ratio's numerator, not just relabel the
    same numbers: numba-over-cc and cc-over-numba are reciprocal, not identical, ratios."""
    times = {"numba": {"k1": 100.0, "k2": 50.0}, "cc": {"k1": 200.0, "k2": 25.0}}

    got = plot_canon_speedup.speedups(times, baseline, column)

    assert got == expected


def test_a_kernel_the_baseline_measured_but_the_column_missed_is_warned_about_and_dropped() -> None:
    """A crashed or unrun kernel must be named in a warning, not silently excluded from the ratio
    the way an ordinary set intersection would drop it."""
    times = {"numba": {"k1": 10.0, "k2": 20.0}, "cc": {"k1": 5.0}}

    with pytest.warns(UserWarning, match="k2"):
        got = plot_canon_speedup.speedups(times, "numba", "cc")

    assert got == [2.0]  # only k1, the kernel both measured


def test_an_unvalidated_row_is_excluded_from_every_statistic() -> None:
    """A row that did not validate is not a slow result, it is not a result -- crediting it would
    let a wrong answer count as a speed-up."""
    frame = pd.DataFrame([row("numba", "k1", 100.0), row("cc", "k1", 50.0, validated="False")], columns=CANON_FIELDS)

    times = plot_canon_speedup.read_times(frame)

    assert "k1" not in times.get("cc", {})
    assert times["numba"]["k1"] == 100.0


def test_a_rerun_of_the_plot_writes_byte_identical_png_pdf_and_table(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A published figure is regenerated and diffed against the committed one; a file that stamps
    the time of its write, or a table whose row order depends on a dict's iteration, differs on
    every rerun even when nothing about the data changed."""
    db_path = tmp_path / "canon.db"
    make_db(
        db_path,
        [
            row("numba", "k1", 100.0),
            row("numba", "k2", 40.0),
            row("cc", "k1", 200.0),
            row("cc", "k2", 20.0),
        ],
    )

    for epoch, folder in (("0", "first"), ("86400", "second")):
        monkeypatch.setenv("SOURCE_DATE_EPOCH", epoch)
        rc = plot_canon_speedup.run(db_path, tmp_path / folder, "numba", False)
        assert rc == 0

    for name in ("canon_speedup.pdf", "canon_speedup.png", "canon_speedup.csv"):
        first = (tmp_path / "first" / name).read_bytes()
        second = (tmp_path / "second" / name).read_bytes()
        assert first == second, f"{name} depends on when it was rendered"


def test_baseline_cc_works_on_a_native_only_db_with_no_numba_or_dace(tmp_path: pathlib.Path) -> None:
    """A compiler-baseline sweep's db has no numba and no dace_* column at all. --baseline cc must
    still draw a figure (the cc bar, at minimum) rather than crash on the absent columns."""
    db_path = tmp_path / "canon.db"
    make_db(db_path, [row("cc", "k1", 200.0), row("cc", "k2", 25.0), row("cpp", "k1", 150.0), row("cpp", "k2", 20.0)])

    rc = plot_canon_speedup.run(db_path, tmp_path / "out", "cc", False)

    assert rc == 0
    table = (tmp_path / "out" / "canon_speedup.csv").read_text()
    assert "cc" in table


def test_the_default_baseline_missing_fails_clearly_instead_of_crashing(tmp_path: pathlib.Path) -> None:
    """A native-only db has no numba column, the default --baseline. Without --baseline cc this
    must exit non-zero with a message naming the missing column, not raise."""
    db_path = tmp_path / "canon.db"
    make_db(db_path, [row("cc", "k1", 200.0), row("cpp", "k1", 150.0)])

    rc = plot_canon_speedup.run(db_path, tmp_path / "out", "numba", False)

    assert rc == 1
    assert not (tmp_path / "out").exists()


def test_the_written_table_names_the_baseline_row_as_the_baseline(tmp_path: pathlib.Path) -> None:
    """The CSV written beside the figure is the printed table verbatim -- a reader of the CSV must
    see the same "(baseline)" marker the printed rows and the figure's y-axis labels carry."""
    db_path = tmp_path / "canon.db"
    make_db(db_path, [row("numba", "k1", 100.0), row("cc", "k1", 200.0)])

    plot_canon_speedup.run(db_path, tmp_path / "out", "numba", False)

    table = (tmp_path / "out" / "canon_speedup.csv").read_text().splitlines()
    assert any("numba" in line and "baseline" in line for line in table)


def test_read_status_keeps_an_unvalidated_row_as_false_not_dropped() -> None:
    """Unlike read_times, read_status must keep every ATTEMPTED kernel -- a failed one reads False,
    never vanishes -- so a caller can report validated/failed counts without a second table scan."""
    frame = pd.DataFrame(
        [row("dace_cpu", "k1", 10.0), row("dace_cpu", "k2", 0.0, validated="False")], columns=CANON_FIELDS
    )

    status = plot_canon_speedup.read_status(frame)

    assert status["dace_cpu"] == {"k1": True, "k2": False}


def test_columns_option_draws_a_column_not_in_the_default_draw_set(tmp_path: pathlib.Path) -> None:
    """dace_cpu (the non-canonicalized DaCe column) is collected by collect_canon.py but not in the
    default DRAW set; --columns must be able to add it back for a sweep that wants it drawn."""
    db_path = tmp_path / "canon.db"
    make_db(db_path, [row("numba", "k1", 100.0), row("dace_cpu", "k1", 25.0)])

    rc = plot_canon_speedup.run(db_path, tmp_path / "out", "numba", False, columns=["dace_cpu"])

    assert rc == 0
    table = (tmp_path / "out" / "canon_speedup.csv").read_text()
    assert "dace_cpu" in table


def test_stem_option_renames_every_output_file(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "canon.db"
    make_db(db_path, [row("numba", "k1", 100.0), row("cc", "k1", 200.0)])

    rc = plot_canon_speedup.run(db_path, tmp_path / "out", "numba", False, stem="llr_full_speedup")

    assert rc == 0
    assert (tmp_path / "out" / "llr_full_speedup.csv").exists()
    assert (tmp_path / "out" / "llr_full_speedup.png").exists()


def test_per_kernel_csv_option_reports_validated_failed_counts_and_a_failed_kernel_row(
    tmp_path: pathlib.Path,
) -> None:
    """A column that failed a kernel the baseline validated must show up as 'failed' in the
    per-kernel csv and counted in the aggregate table's failed_n, never silently dropped."""
    db_path = tmp_path / "canon.db"
    make_db(
        db_path,
        [
            row("numba", "k1", 100.0),
            row("numba", "k2", 50.0),
            row("cc", "k1", 200.0),
            row("cc", "k2", 0.0, validated="False"),
        ],
    )

    rc = plot_canon_speedup.run(db_path, tmp_path / "out", "numba", False, columns=["cc"], per_kernel_csv=True)

    assert rc == 0
    aggregate = list(csv.reader((tmp_path / "out" / "canon_speedup.csv").read_text().splitlines()))
    header, cc_row = aggregate[0], aggregate[1]
    assert header[-2:] == ["validated_n", "failed_n"]
    assert cc_row[header.index("validated_n")] == "1"
    assert cc_row[header.index("failed_n")] == "1"
    per_kernel = (tmp_path / "out" / "canon_speedup_per_kernel.csv").read_text().splitlines()
    header, *lines = per_kernel
    assert header == "kernel,cc"
    assert "k1,0.500000" in lines  # numba(baseline)/cc: 100/200
    assert "k2,failed" in lines


def test_the_default_title_is_the_canon_llr40_headline() -> None:
    """Every reproduce.sh that never passes --title (every one committed so far) must keep drawing
    exactly this string -- the byte-identical-rerun test only catches a change on canon's own db."""
    rows = plot_canon_speedup.rows_for({"numba": {"k1": 1.0}, "cc": {"k1": 2.0}}, "numba", ["cc"])

    fig, ax = plot_canon_speedup.draw(rows, "numba", False)

    assert ax.get_title(loc="left") == plot_canon_speedup.DEFAULT_TITLE
    plt.close(fig)


def test_a_custom_title_replaces_the_default() -> None:
    """A 248-kernel sweep must not draw the 40-kernel canon headline -- --title exists so a caller
    with a different kernel count or sweep name can say so."""
    rows = plot_canon_speedup.rows_for({"numba": {"k1": 1.0}, "cc": {"k1": 2.0}}, "numba", ["cc"])

    fig, ax = plot_canon_speedup.draw(rows, "numba", False, title="Speed-up over Numba, llr-full (248 kernels)")

    assert ax.get_title(loc="left") == "Speed-up over Numba, llr-full (248 kernels)"
    plt.close(fig)


def test_distribution_gives_a_compiler_baseline_a_different_color_than_a_dace_column() -> None:
    """cc and dace_cpu_canonicalize wrap onto the SAME slot of the palette's 6-hue ramp
    (registry.yaml has 30 frameworks); drawn as two lines of one color they would read as one
    series. draw_distribution must tell them apart (a neutral grey for the non-dace column)."""
    times = {
        "numba": {"k1": 10.0, "k2": 20.0},
        "cc": {"k1": 20.0, "k2": 5.0},
        "dace_cpu_canonicalize": {"k1": 40.0, "k2": 5.0},
    }

    fig, ax = plot_canon_speedup.draw_distribution(times, "numba", ["cc", "dace_cpu_canonicalize"], False)

    lines_by_label = {
        line.get_label().split(" ")[0]: line.get_color() for line in ax.lines if not line.get_label().startswith("_")
    }
    plt.close(fig)
    assert len(set(lines_by_label.values())) == len(lines_by_label), lines_by_label


def test_columns_option_draws_dace_gpu_with_its_own_label(tmp_path: pathlib.Path) -> None:
    """``--columns dace_gpu`` must both draw and label the column -- a label lookup silently falling
    back to the raw column name would print ``dace_gpu`` instead of a readable framework name."""
    db_path = tmp_path / "canon.db"
    make_db(db_path, [row("numba", "k1", 100.0), row("dace_gpu", "k1", 10.0)])

    rc = plot_canon_speedup.run(db_path, tmp_path / "out", "numba", False, columns=["dace_gpu"])

    assert rc == 0
    label = experiment_tags.framework_name("dace_gpu")
    assert label != "dace_gpu"
    table = (tmp_path / "out" / "canon_speedup.csv").read_text()
    assert label in table


def test_the_measured_speedup_is_on_the_y_axis_not_the_x_axis() -> None:
    """Rule one: a speed-up is a measured quantity and stays on Y, log-scaled. X carries the
    CATEGORY (the compiler/framework column), which is why it is linear and ticked with names."""
    rows = plot_canon_speedup.rows_for({"numba": {"k1": 1.0, "k2": 1.0}, "cc": {"k1": 2.0, "k2": 0.5}}, "numba", ["cc"])

    fig, ax = plot_canon_speedup.draw(rows, "numba", False)

    try:
        assert ax.get_yscale() == "log"
        assert ax.get_xscale() == "linear"
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert labels == [f"{experiment_tags.framework_name('cc')}  (n=2)"]
    finally:
        plt.close(fig)


def test_both_figures_draw_one_figure_legend_and_the_shared_minor_grid() -> None:
    """One legend on the FIGURE, never ``ax.legend`` (rule five); the log2 speed-up axis carries the
    shared minor ruling (rule four) like every other ratio axis, although this script draws it outside
    ``style.value_axis``."""
    rows = plot_canon_speedup.rows_for({"numba": {"k1": 1.0}, "cc": {"k1": 2.0}}, "numba", ["cc"])
    times = {"numba": {"k1": 1.0, "k2": 1.0}, "cc": {"k1": 2.0, "k2": 0.5}}

    bar_fig, bar_ax = plot_canon_speedup.draw(rows, "numba", False)
    dist_fig, dist_ax = plot_canon_speedup.draw_distribution(times, "numba", ["cc"], False)
    try:
        for fig, ax in ((bar_fig, bar_ax), (dist_fig, dist_ax)):
            assert ax.get_legend() is None
            assert len(fig.legends) == 1
            assert any(line.get_visible() for line in ax.yaxis.get_gridlines())
            assert [tick for tick in ax.yaxis.get_minor_ticks() if tick.gridline.get_visible()]
    finally:
        plt.close(bar_fig)
        plt.close(dist_fig)


def test_distribution_option_draws_a_second_figure(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "canon.db"
    make_db(
        db_path, [row("numba", "k1", 100.0), row("numba", "k2", 50.0), row("cc", "k1", 200.0), row("cc", "k2", 25.0)]
    )

    rc = plot_canon_speedup.run(db_path, tmp_path / "out", "numba", False, columns=["cc"], distribution=True)

    assert rc == 0
    assert (tmp_path / "out" / "canon_speedup_distribution.png").exists()
    assert (tmp_path / "out" / "canon_speedup_distribution.pdf").exists()
