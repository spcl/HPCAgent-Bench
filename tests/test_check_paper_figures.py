# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The paper-figure placement check: a figure placed at its drawn width passes, a rescaled one fails."""

import importlib.util
import pathlib
import sys

import matplotlib
import pytest

matplotlib.use("Agg")  # before any pyplot import -- a headless test must never touch a display
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from hpcagent_bench import paths  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "check_paper_figures", paths.ROOT / "statistics" / "check_paper_figures.py"
)
check = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check
SPEC.loader.exec_module(check)

#: The fake paper's \textwidth and the one figure's drawn width: 0.5\textwidth places it exactly.
TEXT_WIDTH_IN: float = 6.0
DRAWN_WIDTH_IN: float = 3.0

SECTION = r"""\section{Results}
\includegraphics[width=0.5\textwidth]{exact.pdf}
% \includegraphics[width=0.3\textwidth]{commented.pdf}
\includegraphics[width=0.8\linewidth]{scaled.pdf}
"""


@pytest.fixture
def paper(tmp_path: pathlib.Path) -> pathlib.Path:
    """A paper root with one section and one 3in-wide PDF, included once exactly and once rescaled."""
    (tmp_path / "sections").mkdir()
    (tmp_path / "sections" / "results.tex").write_text(SECTION)
    figures = tmp_path / "figures"
    figures.mkdir()
    fig, ax = plt.subplots(figsize=(DRAWN_WIDTH_IN, 2.0))
    ax.plot([0, 1], [0, 1])
    # An explicit box: a tight bbox left in rcParams by an earlier figure would crop the page.
    fig.savefig(figures / "exact.pdf", bbox_inches=Bbox.from_bounds(0.0, 0.0, DRAWN_WIDTH_IN, 2.0))
    plt.close(fig)
    (figures / "scaled.pdf").write_bytes((figures / "exact.pdf").read_bytes())
    return tmp_path


def test_placements_reads_every_uncommented_inclusion_with_its_line(paper: pathlib.Path) -> None:
    assert check.placements(paper) == [("results.tex:2", 0.5, "exact.pdf"), ("results.tex:4", 0.8, "scaled.pdf")]


def test_pdf_width_is_the_drawn_figure_width(paper: pathlib.Path) -> None:
    assert check.pdf_width_in(paper / "figures" / "exact.pdf") == pytest.approx(DRAWN_WIDTH_IN)


def test_a_figure_placed_at_its_drawn_width_passes(paper: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    status = check.main([str(paper), "--text-width", str(TEXT_WIDTH_IN), "--ignore", "scaled.pdf"])
    assert status == 0
    assert capsys.readouterr().out.startswith("ok   exact.pdf: placed 3.000in, drawn 3.000in, type scaled x1.000")


def test_a_rescaled_figure_fails_and_is_named(paper: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    status = check.main([str(paper), "--text-width", str(TEXT_WIDTH_IN)])
    lines = capsys.readouterr().out.splitlines()
    assert status == 1
    assert lines[0].startswith("ok   exact.pdf")
    assert lines[1] == "FAIL scaled.pdf: placed 4.800in, drawn 3.000in, type scaled x1.600 (results.tex:4)"
