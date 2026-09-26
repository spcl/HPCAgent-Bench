# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""scripts/ci_replay.py renders tests.yml the way GitHub does for a push and runs its test steps."""

import pathlib
import re
import subprocess
import sys

import pytest
import yaml


import ci_replay  # pyright: ignore[reportMissingImports]

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("github.event_name == 'workflow_dispatch' && '0' || '1'", "1"),
        ("matrix.shard == 0 && 'all' || 'oneapi'", "all"),
        ("!cancelled()", True),
        ("matrix.leg.select", "tree/"),
        ("contains('a b', 'b')", True),
        ("inputs.missing", ""),
    ],
)
def test_expressions_evaluate_as_on_a_push(expression: str, expected: object) -> None:
    contexts = {
        "github": ci_replay.Context(event_name="push"),
        "matrix": ci_replay.Context(shard=0, leg={"select": "tree/"}),
        "inputs": ci_replay.Context(),
    }
    assert ci_replay.evaluate(expression, contexts) == expected


def test_render_substitutes_every_expression() -> None:
    contexts = {"matrix": ci_replay.Context(shard=2)}
    assert (
        ci_replay.render("awk 'NR % 3 == ${{ matrix.shard }}' ${{ matrix.shard }}", contexts) == "awk 'NR % 3 == 2' 2"
    )


def test_a_matrix_crosses_its_lists_and_appends_include() -> None:
    assert ci_replay.matrix_combinations({"matrix": {"a": [1, 2], "b": ["x"]}}) == [
        {"a": 1, "b": "x"},
        {"a": 2, "b": "x"},
    ]
    assert ci_replay.matrix_combinations({"matrix": {"include": [{"k": 1}, {"k": 2}]}}) == [{"k": 1}, {"k": 2}]
    assert ci_replay.matrix_combinations(None) == [{}]


def test_the_workflow_expands_into_legs_with_rendered_steps(tmp_path: pathlib.Path) -> None:
    """Every test step of tests.yml, no ``${{`` left, the unit shards dealt and setup steps dropped."""
    workflow = yaml.safe_load(ci_replay.WORKFLOW.read_text())
    legs = list(ci_replay.legs(workflow, tmp_path, 1.0))
    labels = {leg.label for leg in legs}
    unit = sorted(leg.label for leg in legs if leg.job == "unit")
    assert len(unit) == len(ci_replay.matrix_combinations(workflow["jobs"]["unit"]["strategy"]))
    assert "coverage" not in {leg.job for leg in legs}, "the coverage job has no test step"
    assert "mpi" in labels
    for leg in legs:
        for step in leg.steps:
            assert "${{" not in step.script and "${{" not in step.name, (leg.label, step.name)
            assert "pytest" in step.script or "python -c" in step.script
            assert step.timeout_s > 0
    shard_one = next(leg for leg in legs if leg.label.startswith("unit[") and "[shard=1]" in leg.label)
    assert re.search(r"awk 'NR % \d+ == 1'", shard_one.steps[-1].script), "shard 1 is not dealt its slice"


def test_list_mode_runs_nothing_and_honours_skip() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "ci_replay.py"), "--list", "--skip", "mpi/Phase 2c", "--jobs", "mpi"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines and all(line.startswith("mpi :: ") for line in lines)
    assert [line for line in lines if "Phase 2c" in line and line.endswith("(skipped)")]
