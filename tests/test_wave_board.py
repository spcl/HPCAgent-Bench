# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The wave board's arm identity, status rule and data embedding: the persistent experiment-status page
reports these, and a wrong one shows a finished experiment as owed or an owed one as finished."""

import importlib.util
import json
import pathlib
import sqlite3
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
        ("llrblind-kimi27sglang-fortran-skills", ("llrblind", "kimi27sglang", "fortran-skills")),
        ("scicomp-dc-qwen38-cpfsrc", ("scicomp-dc", "qwen38", "cpfsrc")),
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
    ("done", "roster", "states", "expected"),
    [
        (40, 40, ["COMPLETED", "PENDING"], "running"),
        (0, 40, ["COMPLETED", "RUNNING"], "running"),
        (40, 40, ["FAILED", "COMPLETED"], "complete"),
        (39, 40, ["COMPLETED"], "incomplete"),
        (0, 0, ["COMPLETED"], "incomplete"),
    ],
)
def test_an_arm_is_running_before_its_coverage_decides(
    board: types.ModuleType, done: int, roster: int, states: list[str], expected: str
) -> None:
    assert board.arm_status(done, roster, states) == expected


def test_the_embedded_data_survives_a_value_that_closes_a_script_element(board: types.ModuleType) -> None:
    """A value carrying ``</script>`` would end the data block early and the page would parse nothing."""
    data = {"generated": "now", "cluster": "beverin", "arms": [{"arm": "</script><b>x</b>"}]}
    page = board.render(data)
    body = page.split('<script type="application/json" id="data">', 1)[1].split("</script>", 1)[0]
    assert json.loads(body) == data


@pytest.mark.parametrize(
    ("campaign", "variant", "experiment"),
    [
        ("cpf-llr-focus40", "c-cpf", "cpf-llr"),
        ("cpf-llr-focus40", "c-cpfsrc", "cpf-llr"),
        ("cpf-llr-focus40", "c-skills", "llr-focus40"),
        ("scicomp-dc", "cpf", "cpf-scicomp"),
        ("scicomp-dc", "dc-cpfsrc", "cpf-scicomp"),
        ("scicomp-dc", "plain", "scicomp-focus40"),
        ("llrblind", "c-cpf", "llr-focus40-blind"),
    ],
)
def test_a_cpf_arm_is_its_own_experiment_on_the_board(
    board: types.ModuleType, campaign: str, variant: str, experiment: str
) -> None:
    """CPF-LLR and CPF-SciComp are reported apart from the campaigns their arms ran in."""
    assert board.board_campaign(campaign, variant).experiment == experiment


def test_the_dropped_gpu_smoke_is_not_on_the_board(board: types.ModuleType) -> None:
    """GPU Smoke@5 was dropped from reporting; its arms must not come back as an experiment window."""
    assert board.campaign_of("gpusmoke5-hip-cpf") == ""


def job_dir_with_rows(root: pathlib.Path, job_id: str, benchmarks: list[str]) -> pathlib.Path:
    """A run directory whose one judge shard holds a submissions row per name in ``benchmarks``."""
    shard = root / job_id / "judge" / "rank-0"
    shard.mkdir(parents=True)
    conn = sqlite3.connect(shard / "hpcagent_bench.db")
    with conn:
        conn.execute("create table submissions (benchmark text)")
        conn.executemany("insert into submissions values (?)", [(name,) for name in benchmarks])
    conn.close()
    return root / job_id


@pytest.mark.parametrize(
    ("rows", "done", "status"),
    [
        ({"100": ["a", "b"]}, 2, "incomplete"),
        ({"100": ["a"], "200": ["b", "c"]}, 3, "complete"),
        ({"100": ["a", "b"], "200": ["b"]}, 2, "incomplete"),
    ],
)
def test_an_arms_coverage_is_the_union_of_every_jobs_rows(
    board: types.ModuleType, tmp_path: pathlib.Path, rows: dict[str, list[str]], done: int, status: str
) -> None:
    """A complement wave grades only what the first wave left, so reading one job reports finished kernels owed."""
    arm = "cpf-llr-focus40-oss120b-c-cpfsrc"
    dirs = {job_id: job_dir_with_rows(tmp_path, job_id, names) for job_id, names in rows.items()}
    jobs = [board.Job(job_id, arm, "COMPLETED", 3, "", "") for job_id in rows]
    row = board.arm_row(arm, jobs, dirs, ["a", "b", "c"], MODELS)
    assert (row["done"], row["status"]) == (done, status), row
