# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The regrade planner's packing: regrade.sbatch deals a worklist round-robin over its four tasks, so a
worklist whose ``items[shard::4]`` is not the slot the planner sized would run a slot past its wall."""

import importlib.util
import pathlib
import types

import pytest

from hpcagent_bench.harness import regrade

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "regrade_rest.py"


@pytest.fixture(scope="module")
def planner() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("regrade_rest", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def item(index: int) -> regrade.Item:
    return regrade.Item("db", f"arm.n0.p{index}.w0", f"k{index}", index, "arm", "c", "restricted", "src", "", True, {})


@pytest.mark.parametrize(
    "minutes",
    [
        [5.0] * 37,
        [200.0, 180.0, 150.0, 150.0, 150.0, 90.0, 60.0, 60.0, 30.0] + [7.0] * 41,
        [236.0] * 5 + [4.0, 3.0],
        [10.0],
    ],
)
def test_round_robin_deals_each_slot_as_packed(planner: types.ModuleType, minutes: list[float]) -> None:
    """Every item lands in exactly one job, ``items[shard::4]`` of each job's worklist is one planned
    slot, and no slot exceeds its job's cap (the budget, or its own longest item)."""
    pairs = [(m, item(i)) for i, m in enumerate(minutes)]
    jobs = planner.pack(pairs, 150.0)
    seen = []
    for slots in jobs:
        order = planner.worklist_order(slots)
        dealt = [{id(it) for it in order[shard :: planner.SLOTS]} for shard in range(planner.SLOTS)]
        planned = [{id(it) for _, it in slot} for slot in sorted(slots, key=len, reverse=True)]
        planned += [set()] * (planner.SLOTS - len(planned))
        assert dealt == planned
        cap = max(150.0, max(m for slot in slots for m, _ in slot))
        assert all(planner.load(slot) <= cap for slot in slots)
        seen.extend(order)
    assert sorted(it.ts_ms for it in seen) == list(range(len(minutes)))


def test_wall_is_three_hours_within_budget_and_longer_past_it(planner: types.ModuleType) -> None:
    """A job within budget keeps the 3 h wall; one whose slot is past budget gets +20 % and start-up."""
    assert planner.wall([[(150.0, item(0))]], 150.0) == "03:00:00"
    assert planner.wall([[(236.0, item(0))]], 150.0) == "05:00:00"
