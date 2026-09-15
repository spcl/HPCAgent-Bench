# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The canon speed-up figure: baseline selection, missing-kernel reporting, and reproducibility."""

import contextlib
import importlib.util
import pathlib
import sqlite3
import sys

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")  # before any pyplot import -- a headless test must never touch a display

from hpcagent_bench import paths

SPEC = importlib.util.spec_from_file_location("plot_canon_speedup", paths.ROOT / "scripts" / "plot_canon_speedup.py")
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
