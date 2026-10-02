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
    resolved = systems.resolve("daint.alps", {}, {})
    assert resolved.system == "daint.alps"
    assert resolved.values["cpus_per_task"] == str(SHIPPED["daint.alps"]["cpus_per_task"])
    assert resolved.values["partition"] == SHIPPED["daint.alps"]["partition"]


def test_the_environment_overrides_the_system_and_a_flag_overrides_the_environment() -> None:
    environ = {"HPCAGENT_BENCH_JOB_CPUS_PER_TASK": "7", "SBATCH_PARTITION": "envpart"}
    assert systems.resolve("beverin", {}, environ).values["cpus_per_task"] == "7"
    assert systems.resolve("beverin", {"cpus_per_task": "9"}, environ).values["cpus_per_task"] == "9"
    assert systems.resolve("beverin", {}, environ).values["partition"] == "envpart"


def test_an_explicit_gpus_per_task_displaces_the_systems_gpus_per_node() -> None:
    values = systems.resolve("beverin", {"gpus_per_task": "1"}, {}).values
    assert values["gpus_per_task"] == "1" and "gpus_per_node" not in values
    values = systems.resolve("beverin", {}, {}).values
    assert "gpus_per_node" in values and "gpus_per_task" not in values


def test_the_cluster_name_picks_the_system_and_an_unknown_cluster_picks_none() -> None:
    cluster = SHIPPED["daint.alps"]["cluster"]
    assert systems.resolve(None, {}, {"SLURM_CLUSTER_NAME": str(cluster)}).system == "daint.alps"
    assert systems.resolve(None, {}, {"HPCAGENT_BENCH_SYSTEM": "beverin-mi200"}).system == "beverin-mi200"
    assert systems.resolve(None, {}, {"SLURM_CLUSTER_NAME": "elsewhere"}).system == ""


def test_a_cluster_with_no_entry_runs_from_flags_and_the_environment_alone() -> None:
    resolved = systems.resolve(
        None, {"gpus_per_node": "2"}, {"SBATCH_ACCOUNT": "proj", "SBATCH_PARTITION": "gpu"}, systems.JOB_FIELDS
    )
    assert (resolved.system, resolved.values) == ("", {"partition": "gpu", "account": "proj", "gpus_per_node": "2"})
    assert resolved.extras == {"hardware": "", "max_time_hours": ""}
    assert systems.resolve(None, {}, {}).values == {}


def test_a_systems_file_adds_a_system(tmp_path: pathlib.Path) -> None:
    extra = tmp_path / "systems.yaml"
    extra.write_text("mycluster:\n  partition: gpu\n  ntasks_per_node: 2\n  cpus_per_task: 32\n  gpus_per_task: 1\n")
    environ = {systems.SYSTEMS_FILE: str(extra)}
    resolved = systems.resolve("mycluster", {}, environ)
    assert (resolved.system, resolved.values["partition"], resolved.values["gpus_per_task"]) == (
        "mycluster",
        "gpu",
        "1",
    )
    assert systems.resolve("beverin", {}, environ).system == "beverin"  # the shipped ones stay


def test_an_unknown_system_names_the_known_ones() -> None:
    with pytest.raises(SystemExit, match="beverin"):
        systems.resolve("nowhere", {}, {})


def test_a_missing_required_field_names_its_flag_and_its_variable() -> None:
    with pytest.raises(SystemExit, match=r"no account: pass --account or set \$SBATCH_ACCOUNT"):
        systems.resolve(None, {"gpus_per_node": "4"}, {}, systems.JOB_FIELDS, systems.JOB_REQUIRED)
    with pytest.raises(
        SystemExit, match=r"no gpus per node: pass --gpus-per-node or set \$HPCAGENT_BENCH_JOB_GPUS_PER_NODE"
    ):
        systems.resolve(None, {"account": "p"}, {}, systems.JOB_FIELDS, systems.JOB_REQUIRED)


def test_the_experiment_job_takes_only_its_own_fields_and_a_systems_gpu_count_fills_the_rest() -> None:
    resolved = systems.resolve(
        "beverin-mi200",
        {"account": "p", "cpus_per_task": "99"},
        {"HPCAGENT_BENCH_JOB_NODES": "3"},
        systems.JOB_FIELDS,
        systems.JOB_REQUIRED,
    )
    assert resolved.values == {"partition": "mi200", "gpus_per_node": "8", "account": "p"}
    assert resolved.hardware == "mi200"


def test_the_hardware_is_its_flag_then_its_variable_then_the_systems() -> None:
    assert systems.resolve("beverin", {}, {}).hardware == "mi300"
    assert systems.resolve("beverin", {}, {"HPCAGENT_BENCH_HARDWARE": "mi200"}).hardware == "mi200"
    assert systems.resolve("beverin", {"hardware": "x"}, {"HPCAGENT_BENCH_HARDWARE": "mi200"}).hardware == "x"
    assert systems.resolve(None, {}, {}).hardware == ""


def test_the_longest_time_limit_is_the_systems_unless_set_and_has_no_default() -> None:
    assert systems.resolve("beverin", {}, {}).extras["max_time_hours"] == "23"
    assert systems.resolve("beverin", {"max_time_hours": "5"}, {}).extras["max_time_hours"] == "5"
    assert systems.resolve(None, {}, {"HPCAGENT_BENCH_MAX_TIME_HOURS": "7"}).extras["max_time_hours"] == "7"
    assert systems.resolve(None, {}, {}).extras["max_time_hours"] == ""


def test_the_options_command_prints_one_option_per_line_or_the_hardware(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(systems, "site_environment", lambda environ: {"SBATCH_ACCOUNT": "proj"})
    assert systems.options_main(["--system", "beverin", "--partition", "gpu"]) == 0
    assert capsys.readouterr().out.splitlines() == ["--partition=gpu", "--account=proj", "--gpus-per-node=4"]
    assert systems.options_main(["--system", "beverin-mi200", "--print", "hardware"]) == 0
    assert capsys.readouterr().out == "mi200\n"
    assert systems.options_main(["--print", "max_time_hours"]) == 0
    assert capsys.readouterr().out == "\n"
    with pytest.raises(SystemExit, match="--gpus-per-node"):
        systems.options_main([])


def test_print_names_one_resolved_value_and_require_refuses_it_unset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--print` answers one value (empty when unset); `--require` makes unset an error naming flag and variable."""
    monkeypatch.setattr(systems, "site_environment", lambda environ: {})
    assert systems.options_main(["--print", "gpus_per_node"]) == 0
    assert capsys.readouterr().out == "\n"
    with pytest.raises(SystemExit, match=r"--gpus-per-node or set \$HPCAGENT_BENCH_JOB_GPUS_PER_NODE"):
        systems.options_main(["--print", "gpus_per_node", "--require", "gpus_per_node"])
    assert (
        systems.options_main(["--system", "beverin-mi200", "--print", "gpus_per_node", "--require", "gpus_per_node"])
        == 0
    )
    assert capsys.readouterr().out == "8\n"


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
