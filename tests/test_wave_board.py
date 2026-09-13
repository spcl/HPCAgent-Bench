# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The wave board's arm identity, status rule and data embedding: the persistent experiment-status page
reports these, and a wrong one shows a finished experiment as owed or an owed one as finished."""

import importlib.util
import json
import pathlib
import sys
import types

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "wave_board.py"
MODELS = ("kimi27sglang", "oss120b", "qwen38", "glm53")


@pytest.fixture(scope="module")
def board() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("wave_board", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        ("gpu-llr-focus40-kimi27sglang-c-openmp-skills", ("gpu-llr-focus40", "kimi27sglang", "c-openmp-skills")),
        ("harness-focus20-smoke-qwen38-claude-autokernel", ("harness-focus20-smoke", "qwen38", "claude-autokernel")),
        ("llrsingle-oss120b-c", ("llrsingle", "oss120b", "c")),
        ("gpusmoke5-hip-cpf", ("gpusmoke5", "", "hip-cpf")),
    ],
)
def test_an_arm_name_splits_into_its_campaign_model_and_variant(
    board: types.ModuleType, arm: str, expected: tuple[str, str, str]
) -> None:
    assert board.split_arm(arm, MODELS) == expected


def test_a_job_outside_every_campaign_has_no_campaign(board: types.ModuleType) -> None:
    """Agent gate jobs share the queue; the board must not invent an experiment for them."""
    assert board.campaign_of("tc-suite-ab") == ""


@pytest.mark.parametrize(
    ("done", "roster", "states", "void", "expected"),
    [
        (40, 40, ["COMPLETED", "PENDING"], False, "running"),
        (0, 40, ["COMPLETED", "RUNNING"], True, "running"),
        (40, 40, ["FAILED", "COMPLETED"], False, "complete"),
        (39, 40, ["COMPLETED"], False, "incomplete"),
        (40, 40, ["COMPLETED"], True, "void"),
        (0, 0, ["COMPLETED"], False, "incomplete"),
    ],
)
def test_an_arm_is_running_before_void_before_its_coverage_decides(
    board: types.ModuleType, done: int, roster: int, states: list[str], void: bool, expected: str
) -> None:
    assert board.arm_status(done, roster, states, void) == expected


def test_the_embedded_data_survives_a_value_that_closes_a_script_element(board: types.ModuleType) -> None:
    """A note carrying ``</script>`` would end the data block early and the page would parse nothing."""
    data = {"generated": "now", "cluster": "beverin", "arms": [{"arm": "a", "void": "</script><b>x</b>"}]}
    page = board.render(data)
    body = page.split('<script type="application/json" id="data">', 1)[1].split("</script>", 1)[0]
    assert json.loads(body) == data
