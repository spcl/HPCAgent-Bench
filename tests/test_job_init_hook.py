# Copyright 2026 the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The plugin ``[hooks]`` ``job_init``: manifest parsing, the runner against stub modules, and the entrypoint
lines that call it.

The runner is tested in-process against stub plugin folders (no container, no venv build); the entrypoint's
lines are lifted from the file and run through bash, the way the launch scripts are tested elsewhere
(test_cluster_launch_scripts). What needs a real container is skipped here by design: an image build and an
EDF run are not reproducible in a unit suite.
"""

import json
import pathlib
import re
import subprocess
import sys
from collections.abc import Iterator

import pytest
from hpcagent_agent import plugins as agent_plugins

from hpcagent_bench import paths

LAUNCH = paths.ROOT / "containers" / "images" / "lib" / "launch_venv.sh"


@pytest.fixture(autouse=True)
def fresh_plugin_packages() -> Iterator[None]:
    """The runner imports each plugin folder as ``hpcagent_bench_plugins.<name>`` and caches it in
    ``sys.modules``; drop those, so one test's stub never serves another test's plugin of the same name."""
    yield
    for name in [held for held in sys.modules if held.startswith(agent_plugins.PACKAGE)]:
        del sys.modules[name]


def plugin_folder(root: pathlib.Path, name: str, manifest: str, module: str = "") -> pathlib.Path:
    """``root/<name>`` with the given ``plugin.toml`` and, when given, a ``preload.py`` stub module beside it."""
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "plugin.toml").write_text(manifest, encoding="utf-8")
    if module:
        (folder / "preload.py").write_text(module, encoding="utf-8")
    return folder


def manifest_with(name: str, target: str) -> str:
    return f'name = "{name}"\n\n[hooks]\njob_init = "{target}"\n'


@pytest.mark.parametrize(
    ("manifest", "expected"),
    [
        ('name = "probeplugin"\n\n[hooks]\njob_init = "pkg.mod:fn"\n', "pkg.mod:fn"),
        ('name = "probeplugin"\n\n[hooks]\n', None),
        ('name = "probeplugin"\n', None),
    ],
)
def test_a_hooks_table_parses_its_target_or_stays_absent(
    tmp_path: pathlib.Path, manifest: str, expected: str | None
) -> None:
    sut = agent_plugins.read(plugin_folder(tmp_path, "probeplugin", manifest)).manifest.hooks
    if expected is None:
        assert sut is None or sut.job_init is None
    else:
        assert sut is not None
        assert sut.job_init == expected


@pytest.mark.parametrize(
    "target",
    ["pkgmod", "pkg.mod", "pkg.mod:", ":fn", "1pkg.mod:fn", "pkg.mod:1fn", "pkg mod:fn", "pkg.mod:fn:x"],
)
def test_a_malformed_job_init_target_is_refused_by_the_manifest(tmp_path: pathlib.Path, target: str) -> None:
    with pytest.raises(ValueError, match=r"plugin\.toml"):
        agent_plugins.read(plugin_folder(tmp_path, "probeplugin", manifest_with("probeplugin", target)))


def test_an_unknown_key_inside_hooks_is_refused(tmp_path: pathlib.Path) -> None:
    manifest = manifest_with("probeplugin", "pkg.mod:fn") + 'when = "later"\n'
    with pytest.raises(ValueError, match=r"plugin\.toml"):
        agent_plugins.read(plugin_folder(tmp_path, "probeplugin", manifest))


def test_an_unknown_top_level_table_is_still_refused(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match=r"plugin\.toml"):
        agent_plugins.read(plugin_folder(tmp_path, "probeplugin", 'name = "probeplugin"\n\n[wat]\nkey = "x"\n'))


#: The stub a plugin folder ships as ``preload.py``: it records every call (a marker + the context it got)
#: into a JSON-lines file, so a test counts invocations and inspects the context. It is importable under the
#: plugin's package namespace only, never as a top-level module -- the path the runner resolves last.
STUB = """
import json
import pathlib

CALLS = pathlib.Path({calls!r})


def run(context):
    with CALLS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({{"marker": {marker!r}, "context": context}}) + "\\n")
    {extra}
"""


def stub_module(calls: pathlib.Path, marker: str, extra: str = "") -> str:
    return STUB.format(calls=str(calls), marker=marker, extra=extra)


def run_hook_cli(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, folders: list[pathlib.Path]
) -> tuple[int, pathlib.Path]:
    """The CLI's ``job-init`` against ``folders``, into a fresh launch home; returns (code, home)."""
    home = tmp_path / "launch-home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv(agent_plugins.PLUGIN_DIRS_ENV, ":".join(str(folder) for folder in folders))
    code = agent_plugins.main(
        [
            "job-init",
            "--home",
            str(home),
            "--venv",
            str(tmp_path / "venv"),
            "--workspace",
            str(tmp_path / "workspace"),
            "--role",
            "agent",
        ]
    )
    return code, home


def test_the_hook_runs_once_with_the_paths_and_env_of_the_launch(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = tmp_path / "calls.jsonl"
    folder = plugin_folder(
        tmp_path, "probeplugin", manifest_with("probeplugin", "preload:run"), stub_module(calls, "probeplugin")
    )
    monkeypatch.setenv("PROBE_MARKER_ENV", "set-by-the-test")

    code, home = run_hook_cli(tmp_path, monkeypatch, [folder])

    assert code == 0
    lines = calls.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, "one runner invocation runs the hook exactly once"
    record = json.loads(lines[0])
    assert record["marker"] == "probeplugin"
    context = record["context"]
    assert context["plugin"] == "probeplugin"
    assert context["root"] == str(folder)
    assert context["home"] == str(home)
    assert context["venv"] == str(tmp_path / "venv")
    assert context["workspace"] == str(tmp_path / "workspace")
    assert context["role"] == "agent"
    assert context["env_file"] == str(home / agent_plugins.JOB_INIT_ENV)
    assert context["environ"]["PROBE_MARKER_ENV"] == "set-by-the-test"
    err = capsys.readouterr().err
    assert "job-init[probeplugin]: preload:run: start" in err
    assert re.search(r"preload:run: ok in \d+\.\d{3} s", err), "start, end and duration are logged"
    assert f"1 ok, 0 failed of 1; env file {home / agent_plugins.JOB_INIT_ENV}" in err


def test_the_stamp_and_the_exported_names_record_what_ran(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = plugin_folder(
        tmp_path, "probeplugin", manifest_with("probeplugin", "preload:run"), stub_module(tmp_path / "c", "x")
    )

    code, home = run_hook_cli(tmp_path, monkeypatch, [folder])

    assert code == 0
    stamp = json.loads((home / agent_plugins.JOB_INIT_STAMP).read_text(encoding="utf-8"))
    assert stamp["role"] == "agent"
    assert len(stamp["hooks"]) == 1
    entry = stamp["hooks"][0]
    assert entry["plugin"] == "probeplugin"
    assert entry["target"] == "preload:run"
    assert entry["status"] == "ok"
    assert 0.0 <= entry["seconds"] < 60.0
    assert stamp["seconds"] >= entry["seconds"]
    assert (home / agent_plugins.JOB_INIT_PLUGINS).read_text(encoding="utf-8") == "probeplugin\n"


def test_a_dropped_env_file_lands_where_the_entrypoint_sources_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drop = 'pathlib.Path(context["env_file"]).write_text("export PROBE_PRELOAD=1\\n")'
    folder = plugin_folder(
        tmp_path,
        "probeplugin",
        manifest_with("probeplugin", "preload:run"),
        stub_module(tmp_path / "c", "x", "import pathlib\n    " + drop),
    )

    code, home = run_hook_cli(tmp_path, monkeypatch, [folder])

    assert code == 0
    assert (home / agent_plugins.JOB_INIT_ENV).read_text(encoding="utf-8") == "export PROBE_PRELOAD=1\n"


def test_a_failing_hook_is_logged_and_the_next_one_still_runs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = tmp_path / "calls.jsonl"
    boom = plugin_folder(
        tmp_path,
        "boomplugin",
        manifest_with("boomplugin", "preload:run"),
        stub_module(calls, "boom", "raise RuntimeError('boom')"),
    )
    fine = plugin_folder(tmp_path, "fineplugin", manifest_with("fineplugin", "preload:run"), stub_module(calls, "fine"))

    code, home = run_hook_cli(tmp_path, monkeypatch, [boom, fine])

    assert code == 1, "a failed hook is the runner's nonzero exit, the entrypoint's cue to log and continue"
    markers = [json.loads(line)["marker"] for line in calls.read_text(encoding="utf-8").splitlines()]
    assert markers == ["boom", "fine"], "a hook's failure never takes the other plugins' hooks down"
    stamp = json.loads((home / agent_plugins.JOB_INIT_STAMP).read_text(encoding="utf-8"))
    assert [entry["status"] for entry in stamp["hooks"]] == ["failed", "ok"]
    assert (home / agent_plugins.JOB_INIT_PLUGINS).read_text(encoding="utf-8") == "boomplugin,fineplugin\n"
    err = capsys.readouterr().err
    assert "boomplugin]: preload:run: failed in " in err
    assert "RuntimeError('boom')" in err
    assert "1 ok, 1 failed of 2" in err


@pytest.mark.parametrize("target", ["no.such.module:fn", "preload:no_such_name", "preload.run:fn"])
def test_an_unresolvable_target_is_a_failed_hook(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    target: str,
) -> None:
    folder = plugin_folder(
        tmp_path, "probeplugin", manifest_with("probeplugin", target), stub_module(tmp_path / "c", "x")
    )

    code, home = run_hook_cli(tmp_path, monkeypatch, [folder])

    assert code == 1
    stamp = json.loads((home / agent_plugins.JOB_INIT_STAMP).read_text(encoding="utf-8"))
    assert [entry["status"] for entry in stamp["hooks"]] == ["failed"]
    assert "failed in" in capsys.readouterr().err


def test_without_a_hook_nothing_runs_and_nothing_is_written(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plain = plugin_folder(tmp_path, "plainplugin", 'name = "plain"\n')

    code, home = run_hook_cli(tmp_path, monkeypatch, [plain])

    assert code == 0
    assert not (home / agent_plugins.JOB_INIT_STAMP).exists()
    assert not (home / agent_plugins.JOB_INIT_PLUGINS).exists()
    assert not (home / agent_plugins.JOB_INIT_ENV).exists()
    assert capsys.readouterr().err == ""


# The entrypoint's side: the invocation sits inside the venv build's once-per-node gate, the sourcing after it.


def entrypoint_lines() -> list[str]:
    return LAUNCH.read_text(encoding="utf-8").splitlines()


def env_and_names_lines() -> tuple[str, str]:
    """The exact sourcing lines of the entrypoint, for a bash run without a container."""
    env_line = next(
        line for line in entrypoint_lines() if line.startswith(f'[ ! -f "${{home}}/{agent_plugins.JOB_INIT_ENV}" ]')
    )
    names_line = next(line for line in entrypoint_lines() if agent_plugins.JOB_INIT_ENV_VAR in line)
    return env_line, names_line


def test_the_invocation_sits_inside_the_venvs_once_gate_and_before_the_service_starts() -> None:
    text = LAUNCH.read_text(encoding="utf-8")
    invocation = text.index("-m hpcagent_agent.plugins job-init")
    ready = text.index('touch "${home}/ready"')
    lock_released = text.rindex(') 9>"${home}/lock"')
    sourced = text.index(
        f'[ ! -f "${{home}}/{agent_plugins.JOB_INIT_ENV}" ] || . "${{home}}/{agent_plugins.JOB_INIT_ENV}"'
    )
    assert invocation < ready, "the hook runs under the build lock, before the venv is declared ready"
    assert ready < lock_released < sourced, "every step sources the dropped env file after the once gate"
    guard = next(
        line for line in entrypoint_lines() if 'if [ -n "${sources}" ] && [ -z "${HPCAGENT_BENCH_LAUNCH_ROOT' in line
    )
    assert guard, "an image build gate is no job start: the hook never runs there"


def test_the_entrypoint_sources_a_dropped_env_into_the_service_environment(tmp_path: pathlib.Path) -> None:
    env_line, names_line = env_and_names_lines()
    home = tmp_path / "home"
    home.mkdir()
    (home / agent_plugins.JOB_INIT_ENV).write_text("export PROBE_PRELOAD=profiler\n", encoding="utf-8")
    (home / agent_plugins.JOB_INIT_PLUGINS).write_text("probeplugin\n", encoding="utf-8")

    script = (
        f'home="{home}"\n'
        f"{env_line}\n"
        f"{names_line}\n"
        f'printf \'%s|%s\\n\' "$PROBE_PRELOAD" "${{{agent_plugins.JOB_INIT_ENV_VAR}}}"'
    )
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)

    assert done.returncode == 0, done.stderr
    assert done.stdout == "profiler|probeplugin\n"


def test_without_a_dropped_env_the_entrypoint_starts_anyway(tmp_path: pathlib.Path) -> None:
    env_line, names_line = env_and_names_lines()
    home = tmp_path / "home"
    home.mkdir()

    done = subprocess.run(
        ["bash", "-c", f'home="{home}"\n{env_line}\n{names_line}\nprintf \'%s|\\n\' "${{HPCAGENT_BENCH_JOB_INIT:-}}"'],
        capture_output=True,
        text=True,
        check=False,
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout == "|\n"


def test_the_entrypoint_script_parses() -> None:
    for argv in (["bash", "-n"], ["sh", "-n"]):
        done = subprocess.run([*argv, str(LAUNCH)], capture_output=True, text=True, check=False)
        assert done.returncode == 0, done.stderr
