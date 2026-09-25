# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A table for a paper, written from the same frame a figure is drawn from.

A NUMBER A FIGURE CANNOT CARRY still belongs beside it. A solve rate is the case this exists for:
it is a count out of a roster, it moves in whole kernels, and a mark on a speedup axis cannot say
it -- but a reader who does not have it will read every speedup as if the arms solved the same
kernels. So the table comes out of the same reduction the figure does, in the same run, rather than
being typed into the paper from a printout and drifting from the figure a revision later.

BOOKTABS, no vertical rules and no ``\\hline`` between rows: three horizontal rules is the house
style of every venue this repo targets, and a ruled grid reads as a spreadsheet. The escaping is
deliberately small -- the columns here carry model names, delivery names and counts, not arbitrary
text -- and what it does escape is the five characters that would otherwise end the compile
(:data:`ESCAPES`).
"""

import pathlib
from collections.abc import Sequence

import pandas as pd

#: The characters LaTeX reads as markup, and what they are written as. ``\\`` is not among them: a
#: caller that wants a macro in a cell has to be able to write one.
ESCAPES: dict[str, str] = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_"}


def escape(value: object) -> str:
    """One cell's text, with the characters that would end the compile written out."""
    text = str(value)
    for character, replacement in ESCAPES.items():
        text = text.replace(character, replacement)
    return text


def column_spec(frame: pd.DataFrame, numeric: str = "r", text: str = "l") -> str:
    """``llrr``: a text column is left-aligned, a numeric one right-aligned on its own digits."""
    return "".join(numeric if pd.api.types.is_numeric_dtype(frame[name]) else text for name in frame.columns)


def row(values: Sequence[object]) -> str:
    """One body line, escaped and rule-free."""
    return " & ".join(escape(value) for value in values) + r" \\"


def booktabs(
    frame: pd.DataFrame,
    caption: str = "",
    label: str = "",
    headers: Sequence[str] = (),
    align: str = "",
    placement: str = "t",
) -> str:
    """``frame`` as a complete ``table`` environment in booktabs style.

    ``headers`` renames the columns for the page without touching the frame; ``align`` overrides
    :func:`column_spec`. The caption sits ABOVE the tabular, which is where a table's caption goes
    -- a figure's goes below it, and the two conventions are not interchangeable.
    """
    names = list(headers) if headers else [str(name) for name in frame.columns]
    lines = [
        f"\\begin{{table}}[{placement}]",
        r"  \centering",
        *([f"  \\caption{{{caption}}}"] if caption else []),
        *([f"  \\label{{{label}}}"] if label else []),
        f"  \\begin{{tabular}}{{{align or column_spec(frame)}}}",
        r"    \toprule",
        "    " + row(names),
        r"    \midrule",
        *["    " + row(record) for record in frame.itertuples(index=False)],
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def write(frame: pd.DataFrame, out: pathlib.Path, **kwargs: object) -> pathlib.Path:
    """:func:`booktabs` to ``out``, and the frame itself to ``out`` with a ``.csv`` suffix.

    Both, always: the ``.tex`` is what the paper includes and the ``.csv`` is what a reader checks
    it against, and a table that ships without its own numbers in machine-readable form is a table
    nobody can audit.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(booktabs(frame, **kwargs))  # pyright: ignore[reportArgumentType]
    frame.to_csv(out.with_suffix(".csv"), index=False)
    return out
