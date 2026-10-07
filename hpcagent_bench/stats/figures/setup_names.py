# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Setup names of a per-kernel figure: which setups it draws, their (model, condition), tokens and tag.

The condition comes from the SETUP NAME (``<experiment>-<model>-<language>[-<condition>]``), not the
``language``/``packet`` columns: every row of a setup agrees on its name. :func:`setup_pattern` is both
the setup selector and the (model, condition) parser.
"""

import re
from collections.abc import Sequence

import pandas as pd

from hpcagent_bench import packets, study_tags
from hpcagent_bench.stats import population

__all__ = [
    "candidate_setups",
    "condition_label",
    "parse_setup",
    "rank_condition",
    "setup_pattern",
    "setup_tokens",
    "tag_of",
]


def setup_pattern(experiment: str, language: str = "c") -> re.Pattern[str]:
    """The setups of ``experiment`` in ``language``: ``<experiment>-<model>-<language>`` is the
    control (condition ``""``), ``...-<language>-<condition>`` a treatment."""
    return re.compile(
        rf"^{re.escape(experiment)}-(?P<model>[a-z0-9]+)-{re.escape(language)}(?:-(?P<condition>[a-z0-9-]+))?$"
    )


def parse_setup(setup: str, pattern: re.Pattern[str]) -> tuple[str, str] | None:
    """``setup``'s (model, condition), or ``None`` when ``pattern`` does not name it. A condition spelling a
    packet alias reads as the packet (``skills`` is ``lang-skills``)."""
    match = pattern.fullmatch(setup)
    if match is None:
        return None
    return match.group("model"), study_tags.canonical("packets", match.group("condition") or "")


def candidate_setups(frame: pd.DataFrame, pattern: re.Pattern[str]) -> dict[str, tuple[str, str]]:
    """Every distinct setup of ``frame`` that ``pattern`` names: setup -> (model, condition)."""
    out: dict[str, tuple[str, str]] = {}
    for setup in frame["setup"].dropna().astype(str).unique():
        parsed = parse_setup(str(setup), pattern)
        if parsed is not None:
            out[str(setup)] = parsed
    return out


def setup_tokens(frame: pd.DataFrame, setup: str) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """``setup``'s per-kernel token total (:func:`population.kernel_tokens`), plus the minimum and maximum
    over its slots' tasks.

    Tokens come only from ``row_kind = episode`` rows; a kernel with no episode total is absent, never entered at
    any stand-in value. With one slot one task IS the kernel's value, so its range is empty.
    """
    subset = frame.loc[frame["setup"].astype(str) == setup]
    totals = population.kernel_tokens(subset, ("setup", "kernel"))
    values = {str(kernel): float(value) for kernel, value in totals.droplevel(0).items() if value > 0}
    episodes = population.episode_tokens(population.latest_episodes(subset), ("setup", "kernel"))
    grouped = episodes.groupby("kernel").tokens
    spread = grouped.size() > 1
    low = {str(kernel): float(value) for kernel, value in grouped.min()[spread].items() if str(kernel) in values}
    high = {str(kernel): float(value) for kernel, value in grouped.max()[spread].items() if str(kernel) in values}
    return values, low, high


def tag_of(canon_frame: pd.DataFrame) -> list[str]:
    """Every kernel the canon sweep names, sorted: the tag when none is given."""
    return sorted({str(k) for k in canon_frame["kernel"].dropna().unique()})


def condition_label(condition: str, treatments: Sequence[str]) -> str:
    """The display text of a condition: the control's wording follows the ``treatments`` drawn beside it
    (:func:`hpcagent_bench.packets.control_label`), a treatment its registry display name."""
    if condition == "":
        return packets.control_label(treatments)
    return study_tags.names("packets").get(condition, condition)


def rank_condition(condition: str) -> tuple[int, str]:
    """Registry order (the control first), then unregistered conditions alphabetically."""
    known = study_tags.order("packets")
    return (known.index(condition), "") if condition in known else (len(known), condition)
