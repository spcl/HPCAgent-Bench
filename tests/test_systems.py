# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``hpcagent-bench job submit`` and ``job options``: a field is its flag, else Slurm's own ``SBATCH_*`` variable,
else the job script's ``#SBATCH`` line, else its other variable, else the system's entry."""

import os
import pathlib
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping

import pytest

from hpcagent_bench.cluster import systems

SHIPPED = systems.load_systems({})
ROOT = pathlib.Path(__file__).resolve().parents[1]


def site_environment(values: dict[str, str]) -> Callable[[Mapping[str, str]], dict[str, str]]:
    """A stand-in for ``systems.site_environment`` whose environment and site layer hold ``values`` alone."""
    return lambda environ: dict(values)


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
    assert values["gpus_per_task"] == "1"
    assert "gpus_per_node" not in values
    values = systems.resolve("beverin", {}, {}).values
    assert "gpus_per_node" in values
    assert "gpus_per_task" not in values


def test_the_cluster_name_picks_the_system_and_an_unknown_cluster_picks_none() -> None:
    cluster = SHIPPED["daint.alps"]["cluster"]
    assert systems.resolve(None, {}, {"SLURM_CLUSTER_NAME": str(cluster)}).system == "daint.alps"
    assert systems.resolve(None, {}, {"HPCAGENT_BENCH_SYSTEM": "beverin-mi200"}).system == "beverin-mi200"
    assert systems.resolve(None, {}, {"SLURM_CLUSTER_NAME": "elsewhere"}).system == ""
    assert systems.resolve(None, {"partition": "mi200"}, {"SLURM_CLUSTER_NAME": "beverin"}).system == "beverin-mi200"
    unserved = systems.resolve(None, {"partition": "debug"}, {"HPCAGENT_BENCH_SYSTEM": "beverin"})
    assert (unserved.system, unserved.values["gpus_per_node"]) == ("beverin", "4")


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
    monkeypatch.setattr(systems, "site_environment", site_environment({"SBATCH_ACCOUNT": "proj"}))
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
    monkeypatch.setattr(systems, "site_environment", site_environment({}))
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
    command = systems.sbatch_command("job.sbatch", {"cpus_per_task": "5", "nice": "100"}, ["a", "--aa"])
    assert command == ["sbatch", "--cpus-per-task=5", "--nice=100", "job.sbatch", "a", "--aa"]


def dry_run(tmp_path: pathlib.Path, *args: str, **knobs: str) -> tuple[str, list[str]]:
    """The system and the sbatch command ``hpcagent-bench job submit --dry-run <args>`` prints under the CSCS site
    layer (system beverin) and account ``proj``, with no Slurm or HPCAgent-Bench value inherited from the caller. An
    ``sbatch`` that fails stands first on PATH, so a launcher that submitted would fail the test, not queue a job."""
    (tmp_path / "sbatch").write_text("#!/bin/sh\necho 'sbatch called' >&2\nexit 3\n", encoding="utf-8")
    (tmp_path / "sbatch").chmod(0o755)
    inherited = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SBATCH_", "SLURM_", "HPCAGENT_BENCH_")) and key != "ROLE"
    }
    environ = inherited | {
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "HPCAGENT_BENCH_SITE_ENV": str(ROOT / "experiments" / "layers" / "site-cscs.env"),
        "SBATCH_ACCOUNT": "proj",
        **knobs,
    }
    done = subprocess.run(
        [sys.executable, "-m", "hpcagent_bench", "job", "submit", "--dry-run", *args],
        capture_output=True,
        text=True,
        check=False,
        env=environ,
        cwd=ROOT,
    )
    assert done.returncode == 0, done.stderr
    system, command = done.stderr.splitlines()[-1].removeprefix("# system ").split(": ", 1)
    words = shlex.split(command)
    # every job's Slurm output goes to the scratch logs, never to wherever sbatch ran
    logs = str(ROOT / ".scratch" / "logs")
    assert {f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err"} <= set(words), words
    return system, [word for word in words if not word.startswith(("--output=", "--error="))]


def job_script(tmp_path: pathlib.Path, *header: str) -> str:
    script = tmp_path / "job.sbatch"
    script.write_text("\n".join(["#!/bin/bash", "# a job", *header, "echo start", "#SBATCH --gpus-per-task=1"]))
    return str(script)


def test_a_script_keeps_the_fields_its_header_pins_and_the_system_fills_the_rest(tmp_path: pathlib.Path) -> None:
    """One task of 96 cores stays one task of 96 cores under Beverin's 4 x 24; an #SBATCH line past the first
    command is no header line, so the system's GPUs per node still apply."""
    script = job_script(tmp_path, "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task 96", "#SBATCH -t 02:00:00  # short")
    assert dry_run(tmp_path, script) == (
        "beverin",
        ["sbatch", "--partition=mi300", "--account=proj", "--gpus-per-node=4", "--nice=100", script],
    )


def test_a_flag_beats_the_header_which_beats_our_variables_but_not_slurms(tmp_path: pathlib.Path) -> None:
    script = job_script(tmp_path, "#SBATCH --partition=hdr", "#SBATCH --cpus-per-task=96")
    assert "--cpus-per-task=8" in dry_run(tmp_path, "--cpus-per-task", "8", script)[1]
    system, command = dry_run(tmp_path, script, HPCAGENT_BENCH_JOB_CPUS_PER_TASK="7")
    assert system == "beverin"
    assert not [word for word in command if word.startswith(("--cpus", "--partition"))]
    assert "--partition=envpart" in dry_run(tmp_path, script, SBATCH_PARTITION="envpart")[1]


def test_a_partition_picks_the_system_of_the_cluster_that_serves_it(tmp_path: pathlib.Path) -> None:
    """Without --system, mi200 on Beverin is beverin-mi200's shape; an explicit --system keeps its own."""
    script = job_script(tmp_path)
    mi200 = ["--partition=mi200", "--account=proj", "--ntasks-per-node=4", "--cpus-per-task=16", "--gpus-per-node=8"]
    assert dry_run(tmp_path, "--partition", "mi200", script) == (
        "beverin-mi200",
        ["sbatch", *mi200, "--nice=100", script],
    )
    assert dry_run(tmp_path, script, SBATCH_PARTITION="mi200")[0] == "beverin-mi200"
    system, command = dry_run(tmp_path, "--system", "beverin", "--partition", "mi200", script)
    assert system == "beverin", command
    assert {"--cpus-per-task=24", "--gpus-per-node=4"} <= set(command), command


def test_an_image_job_runs_on_its_roles_partition_unless_a_flag_or_a_system_names_another(
    tmp_path: pathlib.Path,
) -> None:
    build = "containers/images/build_and_verify.sbatch"
    assert dry_run(tmp_path, build, "vllm") == (
        "beverin",
        ["sbatch", "--partition=mi300", "--account=proj", "--gpus-per-node=4", "--nice=100", build, "vllm"],
    )
    system, command = dry_run(tmp_path, "containers/images/verify_image.sbatch", ROLE="judge-mi200")
    assert system == "beverin-mi200", command
    assert {"--partition=mi200", "--gpus-per-node=8"} <= set(command), command
    assert "--partition=p" in dry_run(tmp_path, "--partition", "p", build, "vllm")[1]
    system, command = dry_run(tmp_path, "--system", "beverin-mi200", build, "judge-agent-amd")
    assert system == "beverin-mi200", command
    assert "--partition=mi200" in command, command
