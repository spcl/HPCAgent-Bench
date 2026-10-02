# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""llr40 setups: which setups a figure draws, their (model, condition), their tokens and roster.

The condition comes from the SETUP NAME, not the ``language``/``packet`` columns: the pre-regrade
extraction records those inconsistently for the same setup, while every row of a setup agrees on its
name. :data:`SETUP_PATTERN` is both the setup selector and the (model, condition) parser.
"""

import re
from collections.abc import Sequence

import pandas as pd

from hpcagent_bench import study_tags, packets
from hpcagent_bench.stats import population

__all__ = [
    "SETUP_PATTERN",
    "CANON_BASELINE",
    "CONDITION_ORDER",
    "setup_tokens",
    "candidate_setups",
    "condition_label",
    "parse_setup",
    "rank_condition",
    "roster_of",
]

#: A setup an llr40 figure may draw, and its (model, condition) in one match: ``-c`` is the control
#: (condition ``""``), ``-c-cpf`` the CPF page, ``-c-cpfsrc`` CPF as source; the control is ``llr40-``
#: (or its recorded ``llr-focus40-`` spelling) and a CPF setup keeps the ``cpf-llr-focus40-`` prefix. C only
#: -- Fortran has no CPF spelling (mpr-artifacts/experiments/llr-focus40-cpf/README.md).
SETUP_PATTERN: re.Pattern[str] = re.compile(
    r"^(?:llr40|(?:cpf-)?llr-focus40)-(?P<model>[a-z0-9]+)-c(?:-(?P<condition>cpf|cpfsrc))?$"
)

#: Draw order within one model's own slot, control first.
CONDITION_ORDER: tuple[str, ...] = ("", "cpf", "cpfsrc")

#: What every canon column is measured against.
CANON_BASELINE: str = "numba"


def parse_setup(setup: str, pattern: re.Pattern[str] = SETUP_PATTERN) -> tuple[str, str] | None:
    """``setup``'s (model, condition), or ``None`` when ``pattern`` does not name it."""
    match = pattern.fullmatch(setup)
    if match is None:
        return None
    return match.group("model"), match.group("condition") or ""


def candidate_setups(frame: pd.DataFrame, pattern: re.Pattern[str] = SETUP_PATTERN) -> dict[str, tuple[str, str]]:
    """Every distinct setup of ``frame`` that ``pattern`` names: setup -> (model, condition)."""
    out: dict[str, tuple[str, str]] = {}
    for setup in frame["setup"].dropna().astype(str).unique():
        parsed = parse_setup(str(setup), pattern)
        if parsed is not None:
            out[str(setup)] = parsed
    return out


def setup_tokens(
    frame: pd.DataFrame, setup: str, repeats: population.RepeatPolicy = population.RepeatPolicy.LATEST
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """``setup``'s per-kernel token total under ``repeats`` (:func:`population.kernel_tokens`), plus
    the minimum and maximum over the tasks when ``repeats="median"`` (R5).

    Tokens come only from ``record = task`` rows (T4); a kernel with no task total is absent, never
    entered at any stand-in value. Under ``latest`` one task IS the
    kernel's value, so the range dicts come back empty -- there is nothing to bracket.
    """
    subset = frame[frame["setup"].astype(str) == setup]
    totals = population.kernel_tokens(subset, ("setup", "benchmark"), repeats=repeats)
    values = {str(kernel): float(value) for kernel, value in totals.droplevel(0).items() if value > 0}
    if population.repeat_policy(repeats) != population.RepeatPolicy.MEDIAN or not values:
        return values, {}, {}
    episodes = population.episode_tokens(subset, ("setup", "benchmark"))
    grouped = episodes.groupby("benchmark").tokens
    low = {str(kernel): float(value) for kernel, value in grouped.min().items() if str(kernel) in values}
    high = {str(kernel): float(value) for kernel, value in grouped.max().items() if str(kernel) in values}
    return values, low, high


def roster_of(canon_frame: pd.DataFrame) -> list[str]:
    """The 40 llr40 kernels: every kernel the canon sweep names, sorted -- the same order
    :func:`hpcagent_bench.stats.canon.speedups` already reduces its ratios in."""
    return sorted({str(k) for k in canon_frame["kernel"].dropna().unique()})


def condition_label(condition: str) -> str:
    """The display text for a condition tag (``""`` control, ``cpf``, ``cpfsrc``).

    The control reads "No Packet", never the registry's "No Skill Packet": the llr40 treatments
    (CPF page, CPF as source) are not skills, and borrowing the skills studies' wording for
    the control names the wrong thing (:func:`hpcagent_bench.packets.control_label`, gated on the
    treatment set rather than hardcoded here or in the registry).
    """
    if condition == "":
        return packets.control_label(CONDITION_ORDER[1:])
    return study_tags.names("packets").get(condition, condition)


def rank_condition(condition: str, order: Sequence[str] = CONDITION_ORDER) -> tuple[int, str]:
    """``order``'s conditions first, in their declared order, then anything else alphabetically.

    A condition axis need not be a skill packet: gitscicomp10's setup names carry ``kernel``/``repo``,
    neither of which is in :data:`CONDITION_ORDER`. ``CONDITION_ORDER.index`` would raise on those;
    this is the same "known order first, unregistered last" tiebreak :func:`palette.in_order` and
    :mod:`statistics.plot_setup_summary`'s ``condition_order`` already use for model and packet axes.
    """
    return (order.index(condition), "") if condition in order else (len(order), condition)
