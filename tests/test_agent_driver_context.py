# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""agent_driver.py: the context-window death, the one budget the CLI spends without saying so.

Campaign 594529 lost agents at ~60 turns to vLLM refusing a prompt longer than the served window.
The CLI closes such a run with subtype ``success`` and exit 0 -- the refusal appears only as
``is_error`` plus the served text in ``result`` -- so without this check the driver records the
death as a finished run and the arm reads as complete.
"""

import importlib.util
import pathlib
import sys
from types import ModuleType

import pytest

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "containers/cluster/example-script"

#: The closing event of a killed agent, verbatim in shape from a 594529 claude.log.
OVERFLOW = (
    '{"type":"result","subtype":"success","is_error":true,"num_turns":61,'
    '"result":"API Error: 500 Input length (66001) exceeds model\'s maximum context length (65536)"}\n'
)


def load_example_module(name: str) -> ModuleType:
    """``sys.modules`` must carry the module BEFORE exec, matching tests/test_validate_run.py."""
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="driver")
def driver_fixture() -> ModuleType:
    return load_example_module("agent_driver")


def test_a_context_overflow_death_is_not_a_success(driver, tmp_path):
    log = tmp_path / "claude.log"
    log.write_text(
        '{"type":"assistant","message":{"id":"a","usage":{"output_tokens":5}}}\n' + OVERFLOW, encoding="utf-8"
    )
    assert driver.context_overflow(log) is True
    # ...and the subtype the CLI reports is exactly the one that made this invisible
    assert driver.final_result(log) == ("success", 61)


@pytest.mark.parametrize(
    "closing",
    [
        '{"type":"result","subtype":"success","is_error":false,"num_turns":12,"result":"submitted"}\n',
        '{"type":"result","subtype":"error_max_turns","is_error":true,"num_turns":40}\n',
        '{"type":"result","subtype":"success","num_turns":9,"result":"context length is 65536 tokens per the task"}\n',
        "",
    ],
)
def test_every_other_ending_is_left_alone(driver, tmp_path, closing):
    """A finished run, the turn cap, and an agent that merely WROTE about context lengths."""
    log = tmp_path / "claude.log"
    log.write_text(
        '{"type":"assistant","message":{"id":"a","usage":{"output_tokens":5}}}\n' + closing, encoding="utf-8"
    )
    assert driver.context_overflow(log) is False
    assert driver.context_overflow(tmp_path / "absent.log") is False
