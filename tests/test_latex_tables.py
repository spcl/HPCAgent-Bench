# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A table that ships beside a figure must compile, and must carry the numbers the figure cannot.

The solve rate is the case: every speedup the efficacy figure draws is a geomean over the kernels
an arm was SERVED, with an unanswered kernel held at 1x, so two arms sit at the same height having
solved twelve kernels and thirty. The table is the only place that difference is visible, which
makes "it compiles and it says what the frame said" a property worth pinning.
"""

import pathlib

import pandas as pd
import pytest

from hpcagent_bench.stats import latex


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Model": "Qwen3.8-27B", "Delivery": "HIP", "No Skill Packet": "13/40", "Packet": "28/40"},
            {"Model": "Kimi-K2.7-Code", "Delivery": "OpenMP Offload", "No Skill Packet": "37/40", "Packet": "27/40"},
        ]
    )


def test_the_table_is_booktabs_with_three_rules_and_no_grid() -> None:
    """Three horizontal rules, no verticals: the house style of every venue this repo targets. A
    ruled grid reads as a spreadsheet, and ``\\hline`` between rows is what produces one."""
    text = latex.booktabs(frame())
    assert text.count(r"\toprule") == 1
    assert text.count(r"\midrule") == 1
    assert text.count(r"\bottomrule") == 1
    assert r"\hline" not in text
    assert "|" not in text.split(r"\begin{tabular}")[1].split("}")[0]


def test_a_caption_sits_above_the_tabular() -> None:
    """A table's caption goes above it and a figure's below it; the two conventions are not
    interchangeable, and a caption emitted after the tabular renders in the wrong place."""
    text = latex.booktabs(frame(), caption="Kernels answered.", label="tab:solve")
    assert text.index(r"\caption") < text.index(r"\begin{tabular}")
    assert text.index(r"\label{tab:solve}") < text.index(r"\begin{tabular}")


@pytest.mark.parametrize(("raw", "want"), [("a_b", r"a\_b"), ("50%", r"50\%"), ("A&B", r"A\&B"), ("$x", r"\$x")])
def test_a_cell_that_would_end_the_compile_is_escaped(raw: str, want: str) -> None:
    """Model and delivery names carry underscores and percent signs. One of them unescaped does not
    render badly -- it ends the LaTeX run."""
    assert latex.escape(raw) == want


def test_a_backslash_is_left_alone_so_a_cell_can_carry_a_macro() -> None:
    """The escaping is deliberately small: a caller that wants ``\\textbf{28/40}`` in a cell has to
    be able to write it."""
    assert latex.escape(r"\textbf{x}") == r"\textbf{x}"


def test_a_numeric_column_is_right_aligned_and_a_text_column_left() -> None:
    """A count read down a column is compared digit by digit, which only works right-aligned."""
    mixed = pd.DataFrame({"Model": ["a"], "Solved": [28]})
    assert latex.column_spec(mixed) == "lr"


def test_the_writer_emits_the_numbers_beside_the_table(tmp_path: pathlib.Path) -> None:
    """A table that ships without its own numbers in machine-readable form is one nobody can audit,
    so the CSV is written every time, not on request."""
    out = latex.write(frame(), tmp_path / "solve.tex", caption="c", label="tab:s")
    assert out.read_text().startswith(r"\begin{table}")
    beside = out.with_suffix(".csv")
    assert beside.is_file()
    assert pd.read_csv(beside).equals(frame())
