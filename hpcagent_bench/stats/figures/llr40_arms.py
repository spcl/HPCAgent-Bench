# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""llr-focus40 arms: which arms a figure draws, their (model, condition), their tokens and roster.

The condition comes from the ARM NAME, not the ``language``/``packet`` columns: the pre-regrade
extraction records those inconsistently for the same arm, while every row of an arm agrees on its
name. :data:`ARM_PATTERN` is both the arm selector and the (model, condition) parser.
"""

import re
from collections.abc import Sequence

import pandas as pd

from hpcagent_bench import experiment_tags, packets
from hpcagent_bench.stats import population

#: An arm an llr-focus40 figure may draw, and its (model, condition) in one match: ``-c`` is the control
#: (condition ``""``), ``-c-cpf`` the CPF page, ``-c-cpfsrc`` CPF as source. C only -- Fortran has no
#: CPF spelling (mpr-artifacts/experiments/llr-focus40-cpf/README.md).
ARM_PATTERN: re.Pattern[str] = re.compile(r"^cpf-llr-focus40-(?P<model>[a-z0-9]+)-c(?:-(?P<condition>cpf|cpfsrc))?$")

#: Draw order within one model's own slot, control first.
CONDITION_ORDER: tuple[str, ...] = ("", "cpf", "cpfsrc")

#: The canon column this figure draws by default, and what it is measured against.
CANON_COLUMN: str = "dace_cpu_canonicalize"
CANON_BASELINE: str = "numba"


def parse_arm(arm: str, pattern: re.Pattern[str] = ARM_PATTERN) -> tuple[str, str] | None:
    """``arm``'s (model, condition), or ``None`` when ``pattern`` does not name it."""
    match = pattern.fullmatch(arm)
    if match is None:
        return None
    return match.group("model"), match.group("condition") or ""


def candidate_arms(frame: pd.DataFrame, pattern: re.Pattern[str] = ARM_PATTERN) -> dict[str, tuple[str, str]]:
    """Every distinct arm of ``frame`` that ``pattern`` names: arm -> (model, condition)."""
    out: dict[str, tuple[str, str]] = {}
    for arm in frame["arm"].dropna().astype(str).unique():
        parsed = parse_arm(str(arm), pattern)
        if parsed is not None:
            out[str(arm)] = parsed
    return out


def arm_tokens(
    frame: pd.DataFrame, arm: str, repeats: population.RepeatPolicy = population.RepeatPolicy.LATEST
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """``arm``'s per-kernel token total under ``repeats`` (:func:`population.kernel_tokens`), plus
    the minimum and maximum over the tasks when ``repeats="median"`` (R5).

    Tokens come only from ``record = task`` rows (T4); a kernel with no task total is absent, never
    entered at any stand-in value. Under ``latest`` one task IS the
    kernel's value, so the range dicts come back empty -- there is nothing to bracket.
    """
    subset = frame[frame["arm"].astype(str) == arm]
    totals = population.kernel_tokens(subset, ("arm", "benchmark"), repeats=repeats)
    values = {str(kernel): float(value) for kernel, value in totals.droplevel(0).items() if value > 0}
    if population.repeat_policy(repeats) != population.RepeatPolicy.MEDIAN or not values:
        return values, {}, {}
    episodes = population.episode_tokens(subset, ("arm", "benchmark"))
    grouped = episodes.groupby("benchmark").tokens
    low = {str(kernel): float(value) for kernel, value in grouped.min().items() if str(kernel) in values}
    high = {str(kernel): float(value) for kernel, value in grouped.max().items() if str(kernel) in values}
    return values, low, high


def roster_of(canon_frame: pd.DataFrame) -> list[str]:
    """The 40 llr-focus40 kernels: every kernel the canon sweep names, sorted -- the same order
    :func:`hpcagent_bench.stats.canon.speedups` already reduces its ratios in."""
    return sorted({str(k) for k in canon_frame["kernel"].dropna().unique()})


def condition_label(condition: str) -> str:
    """The display text for a condition tag (``""`` control, ``cpf``, ``cpfsrc``).

    The control reads "No Packet", never the registry's "No Skill Packet": the llr-focus40 treatments
    (CPF page, CPF as source) are not skills, and borrowing the skills experiments' wording for
    the control names the wrong thing (:func:`hpcagent_bench.packets.control_label`, gated on the
    treatment set rather than hardcoded here or in the registry).
    """
    if condition == "":
        return packets.control_label(CONDITION_ORDER[1:])
    return experiment_tags.names("packets").get(condition, condition)


def rank_condition(condition: str, order: Sequence[str] = CONDITION_ORDER) -> tuple[int, str]:
    """``order``'s conditions first, in their declared order, then anything else alphabetically.

    A condition axis need not be a skill packet: git-scicomp's arm names carry ``kernel``/``repo``,
    neither of which is in :data:`CONDITION_ORDER`. ``CONDITION_ORDER.index`` would raise on those;
    this is the same "known order first, unregistered last" tiebreak :func:`palette.in_order` and
    :mod:`statistics.plot_arm_summary`'s ``condition_order`` already use for model and packet axes.
    """
    return (order.index(condition), "") if condition in order else (len(order), condition)
