# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""C6-live: ``experiments/harnesses.call_tokens`` / ``accumulate_usage_tokens`` are the LIVE
``AGENT_MAX_TOKENS`` enforcement path for the three non-claude harnesses (mini-SWE, OpenHands,
optimas) -- ``agent_driver.watch_token_budget`` calls ``harness.fold_tokens``, which for a runner
IS ``accumulate_usage_tokens`` (``harnesses.runner``). Nothing exercised either function directly
before this file; ``agent_driver.py``'s own stream-json fold (``accumulate_total_tokens``,
``usage_total``) has a full suite in ``tests/test_agent_driver_budget.py`` and this one mirrors it
for the runner side, which reads a different four fields from a different file shape (one JSON
object per LINE, one line per model call, no ``message.id`` to dedupe by at all).
"""

import importlib.util
import json
import pathlib
import sys
from types import ModuleType

import pytest

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "experiments"


def load_example_module(name: str) -> ModuleType:
    """``sys.modules`` must carry the module BEFORE exec, matching tests/test_validate_run.py."""
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="harnesses")
def harnesses_fixture() -> ModuleType:
    return load_example_module("harnesses")


# call_tokens: one usage.jsonl line's own count


def test_call_tokens_sums_all_four_disjoint_fields(harnesses) -> None:
    """The four fields (uncached prompt, cached prompt, completion, reasoning) are declared disjoint
    (harnesses.py's module docstring); a call that carries all four is charged their sum, not just
    the ones a caller might expect to matter (output alone, say)."""
    record = {"input": 1000, "cached_input": 20000, "output": 300, "reasoning": 150}
    assert harnesses.call_tokens(record) == 1000 + 20000 + 300 + 150


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({}, None),
        ({"output": 7}, 7),
        ({"input": 10, "output": 7}, 17),
        ({"input": "lots", "output": 7}, 7),
        ({"input": True, "output": 7}, 7),  # bool is an int subclass; count() excludes it explicitly
        ({"model": "qwen38"}, None),
    ],
)
def test_call_tokens_treats_missing_and_junk_fields_as_absent_not_zero(harnesses, record, expected) -> None:
    """A record with none of the four fields is ``None`` -- a line worth skipping, not a call that
    cost nothing -- exactly the distinction ``agent_driver.usage_total`` draws for the claude side
    (``tests/test_agent_driver_budget.py::test_usage_total_treats_missing_and_junk_fields_as_zero``).
    A junk or boolean field is dropped rather than crashing the fold or being counted as its truthy
    integer value."""
    assert harnesses.call_tokens(record) == expected


def test_call_tokens_ignores_fields_outside_the_declared_four(harnesses) -> None:
    """A field this contract does not name (e.g. a future ``retries`` or ``latency_ms``) must not be
    swept into the count just because it happens to be numeric -- only CONSUMED_FIELDS are tokens."""
    record = {"input": 100, "output": 50, "latency_ms": 900, "retries": 3}
    assert harnesses.call_tokens(record) == 150


# accumulate_usage_tokens: the running total over usage.jsonl lines


def line(record: dict) -> str:
    return json.dumps(record)


def test_accumulate_usage_tokens_sums_every_call_line(harnesses) -> None:
    total_by_call: dict[str, int] = {}
    lines = [line({"input": 100, "output": 20}), line({"input": 200, "cached_input": 50, "output": 30})]
    assert harnesses.accumulate_usage_tokens(lines, total_by_call) == 120 + 280


def test_accumulate_usage_tokens_keys_calls_by_position_not_by_content(harnesses) -> None:
    """A claude transcript's turns share a ``message.id`` and the driver dedupes on it
    (``agent_driver.accumulate_total_tokens``); usage.jsonl has no such id -- ONE LINE IS ONE CALL --
    so two identical lines are two calls, each counted, never folded into one by their being equal.
    A runner that re-wrote the same call twice (a retried request logged both times, say) would be
    double-counted here, which is the behaviour a future ``message.id``-style dedupe must not add
    silently: usage.jsonl has nothing to dedupe ON."""
    total_by_call: dict[str, int] = {}
    duplicate = line({"input": 500, "output": 10})
    assert harnesses.accumulate_usage_tokens([duplicate, duplicate], total_by_call) == 2 * 510
    assert len(total_by_call) == 2, "two lines are two calls, keyed by position, not deduplicated"


def test_accumulate_usage_tokens_running_total_survives_across_polls(harnesses) -> None:
    """The watcher polls ``usage.jsonl`` repeatedly as it grows (agent_driver.watch_token_budget),
    handing this fold only the NEW lines each time but the SAME ``total_by_call`` dict, so the
    running total must accumulate across calls rather than resetting to just the latest batch."""
    total_by_call: dict[str, int] = {}
    first_total = harnesses.accumulate_usage_tokens([line({"input": 1000, "output": 100})], total_by_call)
    assert first_total == 1100
    second_total = harnesses.accumulate_usage_tokens(
        [line({"input": 2000, "cached_input": 500, "output": 300})], total_by_call
    )
    assert second_total == 1100 + 2800


def test_accumulate_usage_tokens_skips_non_object_and_malformed_lines(harnesses) -> None:
    """A half-written tail while the runner is mid-append, and a stray non-JSON line, cost nothing
    and do not raise -- the same tolerance ``agent_driver.accumulate_total_tokens`` has for a
    partially flushed claude transcript."""
    total_by_call: dict[str, int] = {}
    lines = [
        "",
        "not json at all",
        '{"input": 100, "outpu',  # truncated tail
        json.dumps([1, 2, 3]),  # valid JSON, not an object
        line({"input": 400, "output": 40}),
    ]
    assert harnesses.accumulate_usage_tokens(lines, total_by_call) == 440
    assert len(total_by_call) == 1, "only the one well-formed call line counted"


def test_a_line_with_none_of_the_four_fields_is_not_recorded_as_a_zero_cost_call(harnesses) -> None:
    """``call_tokens`` returning ``None`` for a field-less line must not enter ``total_by_call`` at
    all: a recorded 0 there is indistinguishable from a call that legitimately cost nothing, and
    ``len(total_by_call)`` is what other tests here use to check how many calls were actually seen."""
    total_by_call: dict[str, int] = {}
    harnesses.accumulate_usage_tokens([line({"model": "qwen38"})], total_by_call)
    assert total_by_call == {}
