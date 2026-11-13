# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read the repeat study's markers out of the paper's PDF figure (the A/A study, Figure ``fig:temperature``): every
tick label and every small filled round marker of the first page, as JSON for :mod:`repeat_decode`.

    python -m helpers.scripts.paper_analysis.repeat_extract <figure.pdf> <raw.json>

Needs PyMuPDF (``fitz``), which is no dependency of the benchmark: install it into the environment that runs this."""

import collections
import importlib
import json
import pathlib
import sys
from collections.abc import Sequence


def markers(
    pdf: pathlib.Path,
) -> tuple[list[tuple[str, float, float]], list[tuple[float, float, tuple[float, ...], float]]]:
    """The tick labels ``(text, x, y)`` and markers ``(x, y, rgb, width)`` of the first page of ``pdf``."""
    # PyMuPDF is optional: imported by name so the module imports without it.
    fitz = importlib.import_module("fitz")
    page = fitz.open(pdf)[0]
    ticks = [(str(w[4]), (w[0] + w[2]) / 2, (w[1] + w[3]) / 2) for w in page.get_text("words")]
    marks: list[tuple[float, float, tuple[float, ...], float]] = []
    for drawing in page.get_drawings():
        rect, fill = drawing["rect"], drawing.get("fill")
        small = rect.width < 5 and rect.height < 5 and abs(rect.width - rect.height) < 0.6
        if fill and small and any(item[0] == "c" for item in drawing["items"]):
            rgb = tuple(round(component, 3) for component in fill)
            marks.append(((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2, rgb, round(rect.width, 2)))
    return ticks, marks


def main(argv: Sequence[str] | None = None) -> int:
    pdf, out = (pathlib.Path(arg) for arg in (sys.argv[1:] if argv is None else argv))
    ticks, marks = markers(pdf)
    out.write_text(json.dumps({"ticks": ticks, "marks": marks}), encoding="utf-8")
    styles = collections.Counter((mark[2], mark[3]) for mark in marks)
    print(f"{len(marks)} markers; most common styles: {styles.most_common(5)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
