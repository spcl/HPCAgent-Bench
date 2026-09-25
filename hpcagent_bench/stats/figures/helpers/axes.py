# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Axis steps several figure scripts share."""

from collections.abc import Iterable

#: Advance of one character, as a fraction of its type size: a proportional face averages ~0.6 em.
CHAR_EM: float = 0.6

#: Characters assumed when there is no label to measure, so an empty axis keeps a usable margin.
EMPTY_LABEL_CHARS: int = 8


def rotated_labels_in(labels: Iterable[str], size_pt: float) -> float:
    """Inches the longest of ``labels`` reaches below the axis when printed rotated 90 degrees at
    ``size_pt``: an estimate before drawing, so a fixed margin can grow with the longest name."""
    longest = max((len(label) for label in labels), default=EMPTY_LABEL_CHARS)
    return longest * size_pt * CHAR_EM / 72.0
