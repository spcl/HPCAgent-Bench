# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One intervention impact table (spec section 10) as the markdown a README carries.

The artifact's READMEs quote the numbers their own tables hold, and a number typed twice is a
number that will disagree with itself. This renders the committed CSV, so a README is regenerated
rather than edited whenever the data changes.

Two rows per pair, because the two legs answer different questions and a reader needs both beside
the relaunch rate the final-attempt rule (T2) makes them conditional on:
speedup and tokens (geomean ratio, log-t interval, BH verdict).

    python3 scripts/impact_markdown.py tables/impact_lang_skills_cpu.csv
    python3 scripts/impact_markdown.py --readme experiments/perf-playbook/README.md

``--readme`` rewrites the file in place: each ``<!--TABLE name-->`` line is followed by the render
of ``tables/name.csv`` beside that README, and re-running replaces what the last run wrote. The
marker stays, so the README is regenerated rather than edited whenever the data changes.
"""

import argparse
import math
import pathlib
import sys

import pandas as pd

#: Column headers of the rendered table, in order.
HEADERS = (
    "model",
    "language",
    "packet",
    "n",
    "attempts/task",
    "relaunched",
    "speedup ratio",
    "q",
    "cost ratio (control/treated)",
    "q",
)


def ratio(estimate: float, low: float, high: float) -> str:
    """``1.61 [1.09, 2.36]``, or the estimate alone when the interval was withheld (spec P4)."""
    if not math.isfinite(estimate):
        return "--"
    if not (math.isfinite(low) and math.isfinite(high)):
        return f"{estimate:.2f} (no interval)"
    return f"{estimate:.2f} [{low:.2f}, {high:.2f}]"


def verdict(adjusted: float, label: str) -> str:
    """The corrected p and what it was called; a leg that carried no test says so instead (M1)."""
    if not math.isfinite(adjusted):
        return str(label) if isinstance(label, str) and label else "--"
    star = "*" if str(label) == "significant" else ""
    return f"{adjusted:.3f}{star}"


def row_cells(row: "pd.Series[object]") -> list[str]:
    """One treatment row of the impact table as the cells of one markdown row."""
    return [
        str(row.model),
        str(row.language) if isinstance(row.language, str) and row.language else "--",
        str(row.packet) if isinstance(row.packet, str) and row.packet else "none",
        f"{int(row.speedup_n)}/{int(row.token_n)}" if math.isfinite(float(row.token_n)) else str(row.speedup_n),
        f"{float(row.attempts_per_task):.2f}",
        f"{float(row.share_relaunched):.0%}",
        ratio(float(row.speedup_ratio), float(row.speedup_ci_low), float(row.speedup_ci_high)),
        verdict(float(row.speedup_p_adjusted), row.speedup_verdict),
        ratio(float(row.token_ratio), float(row.token_ci_low), float(row.token_ci_high)),
        verdict(float(row.token_p_adjusted), row.token_verdict),
    ]


def markdown(frame: pd.DataFrame) -> str:
    """``frame``'s treatment rows as a markdown table; a control row carries no ratio and is skipped."""
    treatments = frame[frame.control.notna() & (frame.control.astype(str) != "")]
    lines = ["| " + " | ".join(HEADERS) + " |", "|" + "---|" * len(HEADERS)]
    for row in treatments.itertuples(index=False):
        lines.append("| " + " | ".join(row_cells(row)) + " |")
    return "\n".join(lines)


#: What opens a generated block, with the table name; everything up to the next blank line is ours.
MARKER = "<!--TABLE "


def rendered_readme(text: str, tables: pathlib.Path) -> str:
    """``text`` with every ``<!--TABLE name-->`` block replaced by the render of ``tables/name.csv``.

    A block is the marker plus the lines up to the next blank line, so the previous render is
    replaced instead of accumulating. A marker naming a missing CSV raises: a README that silently
    keeps a stale table is the failure this exists to prevent.
    """
    out: list[str] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        out.append(line)
        index += 1
        if not line.startswith(MARKER):
            continue
        while index < len(lines) and lines[index].strip():
            index += 1
        name = line[len(MARKER) :].rstrip("->").strip()
        out.append(markdown(pd.read_csv(tables / f"{name}.csv")))
    return "\n".join(out) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--table", type=pathlib.Path, help="an impact CSV written by statistics/paired_arms.py")
    source.add_argument("--readme", type=pathlib.Path, help="a README whose <!--TABLE name--> blocks to rewrite")
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.table is not None:
        print(markdown(pd.read_csv(args.table)))
        return 0
    readme = args.readme
    readme.write_text(rendered_readme(readme.read_text(encoding="utf-8"), readme.parent / "tables"), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
