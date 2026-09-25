# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One context policy for every harness (USER 2026-09-22): no episode may die on the window.

``harnesses.context_policy`` gives the runners L = min(served window, 262144), the reply cap
R = min(launcher cap, L // 8) and the compaction trigger T = L - R - round(0.12 * L);
``agent_driver.claude_context_env`` gives claude the same three numbers through the CLI's own
variables. Two computations of one policy, so this holds them equal on every committed arm and on
the reply caps a launcher may configure: a harness comparison must not also compare compaction
points.
"""

import importlib.util
import math
import pathlib
import sys
from types import ModuleType

import pytest

from tests.env_render import BASES, rendered

EXPERIMENTS = pathlib.Path(__file__).resolve().parents[1] / "experiments"

#: (served window, launcher reply cap) -> (L, R, T).
POLICY = {
    (262144, 32768): (262144, 32768, 197919),
    (262144, 16384): (262144, 16384, 214303),
    (131072, 32768): (131072, 16384, 98959),
    (1048576, 32768): (262144, 32768, 197919),
}


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="driver", scope="module")
def driver_fixture() -> ModuleType:
    return load("agent_driver")


@pytest.fixture(name="harnesses", scope="module")
def harnesses_fixture() -> ModuleType:
    return load("harnesses")


def env_values(path: str) -> dict[str, str]:
    """The flat KEY=VALUE environment a job sources for base ``path``, quotes stripped, with the launcher's
    reply cap (run_cluster.sh exports it)."""
    values = {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32768"}
    for line in rendered(path).splitlines():
        key, _, value = line.partition("=")
        values[key] = value.strip().strip('"')
    return values


def claude_policy(driver: ModuleType, environment: dict[str, str]) -> tuple[int, int, int]:
    """L, R and the trigger claude-code compacts at, from the variables claude is given: 2.1.197
    compacts at floor((window - min(R, 20000)) * pct / 100)."""
    claude = driver.claude_context_env(environment)
    limit, reply = int(claude["CLAUDE_CODE_MAX_CONTEXT_TOKENS"]), int(claude["CLAUDE_CODE_MAX_OUTPUT_TOKENS"])
    effective = int(claude["CLAUDE_CODE_AUTO_COMPACT_WINDOW"]) - min(reply, 20000)
    return limit, reply, math.floor(effective * float(claude["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"]) / 100)


@pytest.mark.parametrize(("served", "configured"), sorted(POLICY))
def test_the_policy_leaves_the_reply_and_one_turn_under_the_capped_window(
    harnesses: ModuleType, served: int, configured: int
) -> None:
    environment = {"CONTEXT_LENGTH": str(served), "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(configured)}
    assert tuple(harnesses.context_policy(environment)) == POLICY[served, configured]


@pytest.mark.parametrize("path", BASES)
def test_every_arm_gives_every_harness_claudes_window_reply_and_trigger(
    driver: ModuleType, harnesses: ModuleType, path: str
) -> None:
    """The window is read from the same keys; claude's percentage is truncated to 4 decimals, so its
    trigger may land a token or two before the runners', never after."""
    environment = env_values(path)
    policy = harnesses.context_policy(environment)
    limit, reply, trigger = claude_policy(driver, environment)
    assert harnesses.served_context(environment) == driver.served_context(environment)
    assert (policy.limit, policy.reply) == (limit, reply)
    assert policy.trigger - 2 <= trigger <= policy.trigger


def test_the_smallest_window_any_source_names_wins(harnesses: ModuleType) -> None:
    environment = {
        "CONTEXT_LENGTH": "262144",
        "SGLANG_EXTRA_ARGS": "--trust-remote-code --context-length 262144",
        "VLLM_EXTRA_ARGS": "--dtype auto --max-model-len=131072 --gpu-memory-utilization 0.70",
    }
    assert harnesses.served_context(environment) == 131072


def test_an_arm_naming_no_window_gets_the_policy_cap(harnesses: ModuleType) -> None:
    assert harnesses.context_policy({}) == (262144, 32768, 197919)
