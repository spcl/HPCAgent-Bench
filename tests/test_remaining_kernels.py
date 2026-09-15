# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What ``experiments/remaining_kernels.py`` says an arm still owes, and what a clean re-run owes.

The owed list is what the next wave runs, so an arm credited with a superseded wave's coverage never
re-runs those kernels and the clean arm stays permanently partial -- while the analysis, which drops
the superseded rows (spec X9), reports it as missing them. The two readings have to agree.
"""

import importlib.util
import pathlib
import sqlite3
import sys
import types

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "remaining_kernels.py"
ARM = "cpf-llr-focus40-qwen38-c-cpf"
ROSTER = ["a", "b", "c"]


@pytest.fixture(name="module", scope="module")
def module_fixture() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("remaining_kernels", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def job_dir_with_rows(root: pathlib.Path, job_id: str, benchmarks: list[str]) -> None:
    """A run directory whose one judge shard holds a submissions row per name in ``benchmarks``."""
    shard = root / job_id / "judge" / "rank-0"
    shard.mkdir(parents=True)
    conn = sqlite3.connect(shard / "hpcagent_bench.db")
    with conn:
        conn.execute("create table submissions (benchmark text)")
        conn.executemany("insert into submissions values (?)", [(name,) for name in benchmarks])
    conn.close()


def owed_lists(
    module: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, jobs: dict[str, str]
) -> dict[str, list[str]]:
    """Run the script over ``{job id: arm name}`` and read back the ``<arm>.txt`` files it wrote."""
    root, out = tmp_path / "runs", tmp_path / "owed"
    monkeypatch.setattr(module, "roster", lambda tag, opt: list(ROSTER))
    monkeypatch.setattr(module, "job_name", jobs.__getitem__)
    monkeypatch.setattr(
        sys, "argv", ["remaining_kernels.py", "--run-root", str(root), "--tag", "t", "--out-dir", str(out)]
    )
    assert module.main() == 0
    return {path.stem: path.read_text(encoding="utf-8").split() for path in sorted(out.glob("*.txt"))}


def test_a_clean_rerun_owes_every_kernel_its_own_jobs_have_no_row_for(
    module: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The superseded wave's rows are dropped at read (spec X9), so crediting them here would leave
    those kernels measured by nothing and never re-run."""
    job_dir_with_rows(tmp_path / "runs", "100", ["a", "b"])
    job_dir_with_rows(tmp_path / "runs", "200", ["a"])
    owed = owed_lists(module, monkeypatch, tmp_path, {"100": ARM, "200": f"{ARM}-clean"})
    assert owed == {ARM: ["c"], f"{ARM}-clean": ["b", "c"]}


def test_a_clean_arm_that_covered_the_roster_owes_nothing(
    module: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """An arm owing nothing must leave NO list behind: the wave driver submits one arm per list it
    finds, and a stale one gives every kernel on it a second agent."""
    job_dir_with_rows(tmp_path / "runs", "200", ROSTER)
    owed = owed_lists(module, monkeypatch, tmp_path, {"200": f"{ARM}-clean"})
    assert owed == {}
