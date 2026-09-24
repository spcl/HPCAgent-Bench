# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""agent_driver.judge_ranks: every judge gets an even share of each kernel level.

A level-3 kernel holds its judge for many minutes per grade. Striping problems by index alone put
every level-3 kernel of a wave on the same judge whenever they sat at the same index modulo the
judge count, while the other judges idled.
"""

import collections
import importlib.util
import json
import pathlib
import subprocess
import sys
from types import ModuleType

import pytest

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "experiments"


@pytest.fixture(name="driver")
def driver_fixture() -> ModuleType:
    spec = importlib.util.spec_from_file_location("agent_driver", EXAMPLE / "agent_driver.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def problems_with(levels: list[int]) -> list[dict[str, object]]:
    return [{"id": index, "kernel": f"k{index}", "level": level} for index, level in enumerate(levels)]


def per_judge(levels: list[int], ranks: list[int]) -> dict[int, collections.Counter[int]]:
    counts: dict[int, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    for level, rank in zip(levels, ranks, strict=True):
        counts[rank][level] += 1
    return counts


def test_level_three_kernels_on_one_stripe_are_spread_over_every_judge(driver: ModuleType) -> None:
    """Every 4th problem is level 3 with 4 judges: the index stripe stacks all ten on judge 0."""
    levels = [3 if index % 4 == 0 else 1 for index in range(40)]
    ranks = driver.judge_ranks(problems_with(levels), 4)
    heavy = [per_judge(levels, ranks)[rank][3] for rank in range(4)]
    assert sorted(heavy) == [2, 2, 3, 3]


def test_each_level_differs_by_at_most_one_problem_between_judges(driver: ModuleType) -> None:
    levels = [1, 2, 3, 3, 2, 1, 1, 3, 2, 2, 1, 3, 3, 1, 2, 1, 2, 3, 1, 1, 2, 3, 1]
    ranks = driver.judge_ranks(problems_with(levels), 3)
    counts = per_judge(levels, ranks)
    for level in (1, 2, 3):
        shares = [counts[rank][level] for rank in range(3)]
        assert max(shares) - min(shares) <= 1, (level, shares)
    loads = [sum(level * n for level, n in counts[rank].items()) for rank in range(3)]
    assert max(loads) - min(loads) <= 3


def test_every_rank_is_a_valid_judge_and_the_deal_is_deterministic(driver: ModuleType) -> None:
    problems = problems_with([2, 3, 1, 3, 1])
    ranks = driver.judge_ranks(problems, 2)
    assert set(ranks) <= {0, 1}
    assert ranks == driver.judge_ranks(problems, 2)


def test_a_file_without_levels_keeps_the_index_stripe(driver: ModuleType) -> None:
    """Rendered before make_problems stamped ``level``: one missing level keeps the old stripe."""
    problems = problems_with([3, 3, 3, 1, 1])
    del problems[4]["level"]
    assert driver.judge_ranks(problems, 2) == [0, 1, 0, 1, 0]


def test_make_problems_stamps_the_manifest_level() -> None:
    out = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "make_problems.py"),
            "--track",
            "loop_level_reasoning",
            "--kernel",
            "loop_level_reasoning/argmax_value/argmax_value",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(out.stdout.strip())["level"] in {1, 2, 3}
