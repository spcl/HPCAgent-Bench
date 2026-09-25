# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Check that every figure a paper includes is placed at the width it was drawn at.

A figure drawn at print size (:data:`hpcagent_bench.stats.style.PRINT_SCALE`, saved with
``style.save(..., width_in=...)``) prints its type at the sizes set there only when ``\\includegraphics``
does not rescale it. This reads every ``\\includegraphics[width=<f>\\textwidth]{<file>}`` under the
paper's ``sections/``, the PDF's own width from its MediaBox, and reports the scale the page applies.
A scale off 1.0 by more than :data:`hpcagent_bench.stats.style.PLACED_WIDTH_RTOL` is a figure whose
type differs from its neighbours'.

    python statistics/check_paper_figures.py ../agentbench-paper --text-width 5.5 --ignore agentbench_v11_is.pdf
"""

import argparse
import pathlib
import re
import sys

from hpcagent_bench.stats import style

INCLUDE = re.compile(r"\\includegraphics\[width=([0-9.]*)\\(?:text|linewidth|columnwidth)[^\]]*\]\{([^}]+)\}")
MEDIA_BOX = re.compile(rb"/MediaBox\s*\[\s*([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s*\]")


def pdf_width_in(path: pathlib.Path) -> float:
    """The first page's MediaBox width, in inches."""
    found = MEDIA_BOX.search(path.read_bytes())
    if found is None:
        raise ValueError(f"{path}: no MediaBox")
    return (float(found.group(3)) - float(found.group(1))) / 72.0


def placements(paper: pathlib.Path) -> list[tuple[str, float, str]]:
    """``(tex file:line, placed fraction of the text width, figure name)`` for every inclusion."""
    found: list[tuple[str, float, str]] = []
    for tex in sorted(paper.glob("sections/*.tex")):
        for number, line in enumerate(tex.read_text().splitlines(), start=1):
            if line.lstrip().startswith("%"):
                continue
            for fraction, name in INCLUDE.findall(line):
                found.append((f"{tex.name}:{number}", float(fraction or 1.0), name))
    return found


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper", type=pathlib.Path, help="the paper's root (holds sections/ and figures/)")
    parser.add_argument("--text-width", type=float, default=style.ICLR_TEXT_WIDTH_IN, help="\\textwidth, inches")
    parser.add_argument("--figures", default="figures", help="the graphics path, relative to the paper root")
    parser.add_argument("--ignore", action="append", default=[], help="a figure not drawn by this API")
    args = parser.parse_args(argv)
    failed = 0
    for where, fraction, name in placements(args.paper):
        if name in args.ignore:
            continue
        path = args.paper / args.figures / name
        placed = fraction * args.text_width
        scale = placed / pdf_width_in(path)
        ok = abs(scale - 1.0) <= style.PLACED_WIDTH_RTOL
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: placed {placed:.3f}in, drawn {placed / scale:.3f}in, "
              f"type scaled x{scale:.3f} ({where})")  # fmt: skip
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
