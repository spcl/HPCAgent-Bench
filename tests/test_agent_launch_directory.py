# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""An agent step runs from a per-job launch directory and loads its tools from the payload bound at launch.

run_cluster.sh used to bind all of experiments/ into the agent container, because run_cluster.sh and
agent_driver.py live there -- and with them every arm's .env and problems file. stage_agent_launch now
copies only what the step executes, and agent_driver.py takes its tools, packets and prompts from
``$HPCAGENT_BENCH_AGENT_DIR`` (the checkout's containers/agent, bound by the launcher) instead of probing for
a copy baked into the image. The shell function is cut out of the shipped script and run as-is.
"""

import ast
import importlib.util
import json
import os
import pathlib
import re
import shlex
import stat
import subprocess
import sys
from types import ModuleType

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"
RUN_CLUSTER = EXPERIMENTS / "run_cluster.sh"
LAUNCH_FILES_RE = re.compile(r"^AGENT_LAUNCH_FILES=\(([^)]*)\)$", re.MULTILINE)
PROBLEMS = "problems-arm-c.jsonl"


def shell_function(name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{$.*?^\}}$", RUN_CLUSTER.read_text(), re.MULTILINE | re.DOTALL)
    assert match, f"{name}() not found in {RUN_CLUSTER}"
    return match.group(0)


def launch_files() -> tuple[str, ...]:
    match = LAUNCH_FILES_RE.search(RUN_CLUSTER.read_text())
    assert match, "AGENT_LAUNCH_FILES not found in run_cluster.sh"
    return tuple(match.group(1).split())


def stage(script_dir: pathlib.Path, launch: pathlib.Path, env_file: pathlib.Path, problems: str) -> None:
    script = "\n".join(
        [
            "set -euo pipefail",
            LAUNCH_FILES_RE.search(RUN_CLUSTER.read_text()).group(0),
            f"SCRIPT_DIR={shlex.quote(str(script_dir))}",
            f"AGENT_LAUNCH_DIR={shlex.quote(str(launch))}",
            shell_function("stage_agent_launch"),
            f"stage_agent_launch {shlex.quote(str(env_file))} {shlex.quote(problems)}",
        ]
    )
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr


def staged_checkout(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """The real experiments/ files, an arm env, its problems file, and another arm's files beside them."""
    launch = tmp_path / "runs" / ".agent-launch" / "7"
    env_file = tmp_path / ".env.arm-c"
    env_file.write_text("CAMPAIGN_ARM=arm-c\nPROBLEMS_FILE=problems-arm-c.jsonl\n")
    (tmp_path / PROBLEMS).write_text('{"id": 0, "kernel": "k", "language": "c", "task": "t"}\n')
    stage(EXPERIMENTS, launch, env_file, str(tmp_path / PROBLEMS))
    return launch, env_file


def test_the_launch_directory_holds_what_an_agent_step_executes_and_nothing_else(tmp_path: pathlib.Path) -> None:
    """A file staged here is readable by every agent of the job; another arm's .env or problems file
    names kernels and treatments this arm must not see."""
    scripts = tmp_path / "experiments"
    scripts.mkdir()
    for name in (*launch_files(), ".env.other-arm", "problems-other-arm.jsonl", "submit-other.sh"):
        (scripts / name).write_text(f"# {name}\n")
    env_file = scripts / ".env.arm-c"
    env_file.write_text("CAMPAIGN_ARM=arm-c\n")
    (scripts / PROBLEMS).write_text("{}\n")
    launch = tmp_path / "launch"
    stage(scripts, launch, env_file, str(scripts / PROBLEMS))
    assert sorted(path.name for path in launch.iterdir()) == sorted((*launch_files(), ".env", PROBLEMS))


def test_the_staged_env_names_the_staged_problems_file(tmp_path: pathlib.Path) -> None:
    """The step re-sources .env inside the container, where a problems path with a directory would
    point outside the only experiments files it can read."""
    launch, _ = staged_checkout(tmp_path)
    assert (launch / ".env").read_text().splitlines()[-1] == f"PROBLEMS_FILE={PROBLEMS}"


def test_staged_files_are_read_only(tmp_path: pathlib.Path) -> None:
    launch, _ = staged_checkout(tmp_path)
    writable = [path.name for path in launch.iterdir() if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP)]
    assert not writable, writable


def test_restaging_the_same_job_replaces_the_launch_directory(tmp_path: pathlib.Path) -> None:
    """A requeued job stages again into its own id; read-only files from the first attempt must not stop it."""
    launch, env_file = staged_checkout(tmp_path)
    stage(EXPERIMENTS, launch, env_file, "")
    assert PROBLEMS not in {path.name for path in launch.iterdir()}


def test_concurrent_staging_of_the_same_job_never_leaves_a_readonly_partial_env(tmp_path: pathlib.Path) -> None:
    """Every role of a job (inference, agent, judge) runs its own run_cluster.sh and each one calls
    stage_agent_launch on the SAME AGENT_LAUNCH_DIR (keyed by job id, not role). The in-place
    rm-rf + populate + chmod version let one caller's chmod a-w land between another caller's cp
    and its later `>>` append, so the append hit a file it no longer had permission to write --
    "Permission denied", the whole job dead before any agent work (643180/643181/643182,
    2026-09-19). Runs several stagers of the SAME arm in parallel; every one must still exit 0 and
    the launch directory must end up complete and read-only, not truncated mid-write."""
    scripts = tmp_path / "experiments"
    scripts.mkdir()
    for name in launch_files():
        (scripts / name).write_text(f"# {name}\n")
    env_file = scripts / ".env.arm-c"
    env_file.write_text("CAMPAIGN_ARM=arm-c\n")
    (scripts / PROBLEMS).write_text("{}\n")
    launch = tmp_path / "runs" / ".agent-launch" / "7"
    script = "\n".join(
        [
            "set -euo pipefail",
            LAUNCH_FILES_RE.search(RUN_CLUSTER.read_text()).group(0),
            f"SCRIPT_DIR={shlex.quote(str(scripts))}",
            f"AGENT_LAUNCH_DIR={shlex.quote(str(launch))}",
            shell_function("stage_agent_launch"),
            f"stage_agent_launch {shlex.quote(str(env_file))} {shlex.quote(str(scripts / PROBLEMS))}",
        ]
    )
    procs = [
        subprocess.Popen(["bash", "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(8)
    ]
    results = [(proc.wait(), *proc.communicate()) for proc in procs]
    failed = [err for code, _, err in results if code != 0]
    assert not failed, failed
    assert sorted(path.name for path in launch.iterdir()) == sorted((*launch_files(), ".env", PROBLEMS))
    assert (launch / ".env").read_text().splitlines()[-1] == f"PROBLEMS_FILE={PROBLEMS}"
    writable = [path.name for path in launch.iterdir() if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP)]
    assert not writable, writable


def test_every_file_the_agent_step_reaches_beside_itself_is_staged() -> None:
    """A file the agent step opens next to run_cluster.sh or agent_driver.py that is missing from the
    launch directory fails only on a compute node, minutes into a job."""
    staged = set(launch_files())
    shell_refs = set(re.findall(r"\$\{SCRIPT_DIR\}/([\w.-]+)", shell_function("run_agent_node")))
    python_refs = set()
    for name in sorted(staged):
        if name.endswith(".py"):
            for node in ast.walk(ast.parse((EXPERIMENTS / name).read_text())):
                modules = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                    if isinstance(node, ast.ImportFrom) and node.level == 0
                    else []
                )
                python_refs |= {f"{module}.py" for module in modules if (EXPERIMENTS / f"{module}.py").is_file()}
    assert shell_refs | python_refs <= staged, sorted((shell_refs | python_refs) - staged)


def test_the_driver_runs_from_a_staged_launch_directory(tmp_path: pathlib.Path) -> None:
    """Imported with no checkout on sys.path: its sibling modules and its problems file resolve from the
    launch directory alone, as they must inside an agent container."""
    launch, _ = staged_checkout(tmp_path)
    probe = "\n".join(
        [
            "import importlib.util, pathlib, sys",
            f"launch = pathlib.Path({str(launch)!r})",
            "spec = importlib.util.spec_from_file_location('agent_driver', launch / 'agent_driver.py')",
            "driver = importlib.util.module_from_spec(spec)",
            "sys.modules['agent_driver'] = driver",
            "spec.loader.exec_module(driver)",
            "import_dirs = {pathlib.Path(driver.harness_module().__file__).parent}",
            "import token_cost, promote_unsubmitted",
            "import_dirs |= {pathlib.Path(m.__file__).parent for m in (token_cost, promote_unsubmitted)}",
            f"assert driver.resolve_problems_path({PROBLEMS!r}) == launch / {PROBLEMS!r}",
            "assert import_dirs == {launch}, import_dirs",
        ]
    )
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
    # The agent step's cwd is RUN_DIR, which holds no problems file of its own.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    done = subprocess.run(
        [sys.executable, "-P", "-c", probe], cwd=run_dir, env=env, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


def load_driver(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / "agent_driver.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_the_driver_reads_the_payload_the_launcher_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setenv("HPCAGENT_BENCH_AGENT_DIR", str(tmp_path))
    assert load_driver("agent_driver_bound").agent_runtime() == tmp_path


def test_without_a_bound_payload_the_driver_reads_its_own_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A driver run from a checkout (tests, a host run) has no launcher; no image copy may stand in."""
    monkeypatch.delenv("HPCAGENT_BENCH_AGENT_DIR", raising=False)
    assert load_driver("agent_driver_checkout").agent_runtime() == REPO / "containers" / "agent"


def test_a_bound_directory_without_tools_stops_the_driver_before_any_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A missing bind would otherwise surface as an agent with no tools that exits 0."""
    monkeypatch.setenv("HPCAGENT_BENCH_AGENT_DIR", str(tmp_path))
    with pytest.raises(SystemExit, match="HPCAGENT_BENCH_AGENT_DIR"):
        load_driver("agent_driver_unbound").tool_registry()


def test_a_read_only_snapshot_env_is_staged_with_its_problems_line(tmp_path: pathlib.Path) -> None:
    """Every job gets a read-only snapshot as its env (env_layers.sh snapshot_env). Copied with its
    mode, the staged .env refused the PROBLEMS_FILE line and the job died at launch (643180, 643222)."""
    env_file = tmp_path / "snapshot.env"
    env_file.write_text("CAMPAIGN_ARM=arm-c\n")
    env_file.chmod(0o400)
    (tmp_path / PROBLEMS).write_text("{}\n")
    launch = tmp_path / "launch"
    stage(EXPERIMENTS, launch, env_file, str(tmp_path / PROBLEMS))
    assert (launch / ".env").read_text().splitlines()[-1] == f"PROBLEMS_FILE={PROBLEMS}"
    assert not os.access(launch / ".env", os.W_OK), "the staged env is still read-only once written"


def test_a_fused_waves_setups_are_staged_beside_the_env(tmp_path: pathlib.Path) -> None:
    """The resolved overlays, split envs and problems of every setup: the driver and judge read them there."""
    run_dir = tmp_path / "run"
    setups = run_dir / "setups"
    setups.mkdir(parents=True)
    for suffix in (".resolved", ".env", ".jsonl", ".keys"):
        (setups / f"arm-c-clean{suffix}").write_text("x\n")
    env_file = tmp_path / "job.env"
    env_file.write_text("CAMPAIGN_ARM=owed-w1\n")
    (tmp_path / PROBLEMS).write_text("{}\n")
    launch = tmp_path / "launch"
    script = "\n".join(
        [
            "set -euo pipefail",
            LAUNCH_FILES_RE.search(RUN_CLUSTER.read_text()).group(0),
            f"SCRIPT_DIR={shlex.quote(str(EXPERIMENTS))}",
            f"AGENT_LAUNCH_DIR={shlex.quote(str(launch))}",
            f"RUN_DIR={shlex.quote(str(run_dir))}",
            shell_function("stage_agent_launch"),
            f"stage_agent_launch {shlex.quote(str(env_file))} {shlex.quote(str(tmp_path / PROBLEMS))}",
        ]
    )
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert sorted(path.name for path in (launch / "setups").iterdir()) == [
        "arm-c-clean.env",
        "arm-c-clean.jsonl",
        "arm-c-clean.resolved",
    ]


def test_a_snapshot_problems_path_resolves_to_the_staged_copy(tmp_path: pathlib.Path) -> None:
    """A snapshot env names its problems file `.rendered/<stem>.jsonl`, relative to experiments/. The
    agent step sees that value (not the launch .env's basename line) and no experiments/ at all, so
    the driver must find the staged copy in its own launch directory (643226 died on it)."""
    launch, _ = staged_checkout(tmp_path)
    probe = "\n".join(
        [
            "import importlib.util, pathlib, sys",
            f"launch = pathlib.Path({str(launch)!r})",
            "spec = importlib.util.spec_from_file_location('agent_driver', launch / 'agent_driver.py')",
            "driver = importlib.util.module_from_spec(spec)",
            "sys.modules['agent_driver'] = driver",
            "spec.loader.exec_module(driver)",
            f"assert driver.resolve_problems_path('.rendered/{PROBLEMS}') == launch / {PROBLEMS!r}",
            "assert driver.resolve_problems_path('/elsewhere/absent.jsonl') == pathlib.Path('/elsewhere/absent.jsonl')",
        ]
    )
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    done = subprocess.run(
        [sys.executable, "-P", "-c", probe], cwd=run_dir, env=env, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


def test_a_fused_waves_worker_reads_its_problems_through_the_whole_launch_plan(tmp_path: pathlib.Path) -> None:
    """End to end, dry: submit-owed-wave.sh's files -> the read-only snapshot a job gets
    (snapshot_env) -> the batch step's own env and problems path (run_cluster.sh) -> the staged launch
    directory and the PROBLEMS_FILE the agent step inherits -> the staged driver, run from RUN_DIR,
    loading every problem with its setup. Both ways the step can see its env: the batch step's export
    as it stands (643226, 643245-643248 died there on `.rendered/<stem>.jsonl`), and after re-sourcing
    the staged .env."""
    owed_spec = importlib.util.spec_from_file_location("owed_wave_launch_plan", EXPERIMENTS / "owed_wave.py")
    assert owed_spec is not None and owed_spec.loader is not None
    owed = importlib.util.module_from_spec(owed_spec)
    sys.modules[owed_spec.name] = owed
    owed_spec.loader.exec_module(owed)
    env = (("AGENTS_PER_NODE", "40"), ("INFERENCE_NODES", "1"), ("HPCAGENT_BENCH_RECORD_MODEL", "qwen38"))
    cpu = owed.Setup("arm-c-clean", "arm-c-clean", "llr-focus40", (*env, ("CAMPAIGN_ARM", "arm-c-clean")))
    hip = owed.Setup("arm-hip-clean", "arm-hip-clean", "llr-focus40", (*env, ("CAMPAIGN_ARM", "arm-hip-clean")))
    items = [owed.Owed(setup, {"kernel": f"k{i}", "task": "t"}, "infra") for i, setup in enumerate((cpu, hip, hip))]
    submit_dir = tmp_path / "experiments"
    submit_dir.mkdir()
    wave_env = owed.write_wave(owed.build_wave("owed-w1", items, "${SCRATCH:?}/runs/owed"), tmp_path / "wave")
    snapshot = subprocess.run(
        ["bash", str(EXPERIMENTS / "env_layers.sh"), "snapshot", str(wave_env), "owed-w1"],
        cwd=submit_dir,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert "PROBLEMS_FILE=.rendered/" in (submit_dir / snapshot).read_text(), "the shape that broke the agent step"
    launch = tmp_path / "runs" / ".agent-launch" / "7"
    run_dir = tmp_path / "runs" / "7"
    run_dir.mkdir(parents=True)
    batch = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(submit_dir))}",
            f"export SCRATCH={shlex.quote(str(tmp_path))}",
            f"set -a; . {shlex.quote(str(submit_dir / snapshot))}; set +a",
            LAUNCH_FILES_RE.search(RUN_CLUSTER.read_text()).group(0),
            f"SCRIPT_DIR={shlex.quote(str(EXPERIMENTS))}",
            f"AGENT_LAUNCH_DIR={shlex.quote(str(launch))}",
            f"RUN_DIR={shlex.quote(str(run_dir))}",
            'problems_file="${PROBLEMS_FILE:-}"',
            'if [[ -n "${problems_file}" && ! -f "${problems_file}" ]]; then problems_file="${SCRIPT_DIR}/${problems_file}"; fi',
            shell_function("stage_agent_launch"),
            shell_function("export_staged_problems"),
            'stage_agent_launch "$(pwd)/' + snapshot + '" "${problems_file}"',
            'export_staged_problems "${problems_file}"',
            "env -0",
        ]
    )
    done = subprocess.run(["bash", "-c", batch], capture_output=True, check=False)
    assert done.returncode == 0, done.stderr.decode()
    step_env = dict(item.split("=", 1) for item in done.stdout.decode().split("\0") if "=" in item)
    step_env.pop("PYTHONPATH", None)
    load = "\n".join(
        [
            "import importlib.util, json, sys",
            f"spec = importlib.util.spec_from_file_location('agent_driver', {str(launch / 'agent_driver.py')!r})",
            "driver = importlib.util.module_from_spec(spec)",
            "sys.modules['agent_driver'] = driver",
            "spec.loader.exec_module(driver)",
            "print(json.dumps([(p['kernel'], p['setup']) for p in driver.load_problems()]))",
        ]
    )
    expected = [["k0", "arm-c-clean"], ["k1", "arm-hip-clean"], ["k2", "arm-hip-clean"]]
    for resource in ("", f"set -a; . {shlex.quote(str(launch / '.env'))}; set +a; "):
        agent = subprocess.run(
            ["bash", "-c", f'{resource}exec "$0" -P -c "$1"', sys.executable, load],
            cwd=run_dir,
            env=step_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert agent.returncode == 0, (resource, agent.stderr)
        assert json.loads(agent.stdout) == expected, resource
