# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``hpcagent-bench job submit``: a field is its flag, else its environment variable, else the system's entry."""

import pathlib

import pytest

from hpcagent_bench.cluster import systems

SHIPPED = systems.load_systems({})


def test_the_shipped_systems_name_every_shape_field_a_job_needs() -> None:
    for name, entry in SHIPPED.items():
        assert {"partition", "ntasks_per_node", "cpus_per_task"} <= set(entry), name
        assert set(entry) & set(systems.GPU_FIELDS), f"{name} names no GPU count"


def test_a_system_supplies_the_shape_when_nothing_overrides_it() -> None:
    name, values = systems.resolve("daint.alps", {}, {})
    assert name == "daint.alps"
    assert values["cpus_per_task"] == str(SHIPPED["daint.alps"]["cpus_per_task"])
    assert values["partition"] == SHIPPED["daint.alps"]["partition"]


def test_the_environment_overrides_the_system_and_a_flag_overrides_the_environment() -> None:
    environ = {"HPCAGENT_BENCH_JOB_CPUS_PER_TASK": "7", "SBATCH_PARTITION": "envpart"}
    assert systems.resolve("beverin", {}, environ)[1]["cpus_per_task"] == "7"
    assert systems.resolve("beverin", {"cpus_per_task": "9"}, environ)[1]["cpus_per_task"] == "9"
    assert systems.resolve("beverin", {}, environ)[1]["partition"] == "envpart"


def test_an_explicit_gpus_per_task_displaces_the_systems_gpus_per_node() -> None:
    _name, values = systems.resolve("beverin", {"gpus_per_task": "1"}, {})
    assert values["gpus_per_task"] == "1" and "gpus_per_node" not in values
    _name, values = systems.resolve("beverin", {}, {})
    assert "gpus_per_node" in values and "gpus_per_task" not in values


def test_the_cluster_name_picks_the_system_and_an_unnamed_one_falls_back() -> None:
    cluster = SHIPPED["daint.alps"]["cluster"]
    assert systems.resolve(None, {}, {"SLURM_CLUSTER_NAME": str(cluster)})[0] == "daint.alps"
    assert systems.resolve(None, {}, {"HPCAGENT_BENCH_SYSTEM": "beverin-mi200"})[0] == "beverin-mi200"
    assert systems.resolve(None, {}, {})[0] == next(iter(SHIPPED))


def test_a_systems_file_adds_a_system(tmp_path: pathlib.Path) -> None:
    extra = tmp_path / "systems.yaml"
    extra.write_text("mycluster:\n  partition: gpu\n  ntasks_per_node: 2\n  cpus_per_task: 32\n  gpus_per_task: 1\n")
    environ = {systems.SYSTEMS_FILE: str(extra)}
    name, values = systems.resolve("mycluster", {}, environ)
    assert (name, values["partition"], values["gpus_per_task"]) == ("mycluster", "gpu", "1")
    assert systems.resolve("beverin", {}, environ)[0] == "beverin"  # the shipped ones stay


def test_an_unknown_system_names_the_known_ones() -> None:
    with pytest.raises(SystemExit, match="beverin"):
        systems.resolve("nowhere", {}, {})


def test_the_sbatch_command_puts_the_options_before_the_script_and_keeps_its_arguments() -> None:
    command = systems.sbatch_command(
        "job.sbatch", {"cpus_per_task": "5", "gpus_per_task": "1"}, ["a", "--aa"], nice="100"
    )
    assert command == ["sbatch", "--cpus-per-task=5", "--gpus-per-task=1", "--nice=100", "job.sbatch", "a", "--aa"]


def test_a_dry_run_prints_the_command_and_submits_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(systems, "site_environment", lambda environ: {})
    monkeypatch.setattr(systems.subprocess, "run", lambda *a, **k: pytest.fail("submitted"))
    assert (
        systems.main(["--system", "daint.alps", "--time", "01:00:00", "--dry-run", "job.sbatch", "w.jsonl", "out"]) == 0
    )
    err = capsys.readouterr().err
    assert "--time=01:00:00" in err and "job.sbatch w.jsonl out" in err and "--partition=" in err
