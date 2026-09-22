# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The host-side srun relay (scripts/cscs/gang_relay.py) and mpi_gang's relay mode, on a fake srun."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hpcagent_bench import paths
from hpcagent_bench.harness import mpi_gang

RELAY = paths.ROOT / "scripts" / "cscs" / "gang_relay.py"
GANG_ENV = {
    "HPCAGENT_BENCH_MPI_GANG_NODELIST": "nid001,nid002,nid003,nid004",
    "HPCAGENT_BENCH_MPI_GANG_EDF": "/run/edf/judge.judge-node.toml",
}
#: Prints its argv one per line and exits 3: the relay must hand back both.
FAKE_SRUN = '#!/bin/sh\nfor a in "$@"; do echo "$a"; done\necho "stderr-line" >&2\nexit 3\n'


def load_relay():
    spec = importlib.util.spec_from_file_location("gang_relay", RELAY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def host_python() -> str:
    """The relay runs on the batch host's python3 (3.6 on Beverin); test with it when present."""
    return "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable


def test_a_gang_launch_through_the_relay_returns_the_steps_status_and_output(monkeypatch, tmp_path, capsys) -> None:
    bindir, relay_dir = tmp_path / "bin", tmp_path / "relay"
    bindir.mkdir()
    (bindir / "srun").write_text(FAKE_SRUN)
    (bindir / "srun").chmod(0o755)
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}"}
    relay = subprocess.Popen([host_python(), str(RELAY), str(relay_dir)], env=env)
    try:
        for key, value in GANG_ENV.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv(mpi_gang.RELAY_DIR_ENV, str(relay_dir))
        monkeypatch.setenv("HPCAGENT_BENCH_MPI_GANG_LOCK", str(tmp_path / "gang.lock"))
        monkeypatch.setenv("SLURM_STEP_ID", "3")
        monkeypatch.setenv("HWLOC_COMPONENTS", "-opencl")
        rc = mpi_gang.main(["-n", "8", "/run/bench", "in", "out"])
    finally:
        relay.kill()
        relay.wait()
    out, err = capsys.readouterr()
    argv = out.splitlines()
    assert rc == 3
    assert "--nodelist=nid001,nid002" in argv and "--environment=/run/edf/judge.judge-node.toml" in argv
    assert argv[-3:] == ["/run/bench", "in", "out"]
    assert "/usr/bin/env" in argv and "HWLOC_COMPONENTS=-opencl" in argv
    assert not any(a.startswith("SLURM_") for a in argv)  # the rank's own step sets those
    assert "stderr-line" in err
    assert sorted(p.name for p in relay_dir.iterdir()) == [], "the judge cleans its request files"


def test_the_relay_kills_a_step_whose_judge_stopped_waiting(monkeypatch, tmp_path) -> None:
    """mpi_call's timeout SIGKILLs the judge-side launcher; its step must not run on."""
    relay = load_relay()
    monkeypatch.setattr(relay, "HEARTBEAT_S", 0.2)
    (tmp_path / "abc.req").write_text(json.dumps({"argv": [shutil.which("sleep"), "60"]}))
    running: dict = {}
    relay.step(str(tmp_path), running)
    assert "abc" in running
    time.sleep(0.4)
    relay.step(str(tmp_path), running)
    assert not running and (tmp_path / "abc.rc").read_text().strip() == "137"


def test_an_unstartable_request_gets_an_exit_status_not_silence(tmp_path) -> None:
    relay = load_relay()
    (tmp_path / "bad.req").write_text("{not json")
    relay.step(str(tmp_path), {})
    assert (tmp_path / "bad.rc").read_text().strip() == "127"
    assert "cannot start" in (tmp_path / "bad.err").read_text()


def test_the_relay_env_prefix_keeps_the_judges_env_but_not_slurm_pmi_or_devices() -> None:
    prefix = mpi_gang.relay_env_prefix(
        {"A": "1", "SLURM_JOB_ID": "9", "PMI_RANK": "0", "PMIX_RANK": "0", "ROCR_VISIBLE_DEVICES": "0"}
    )
    assert prefix == ["/usr/bin/env", "A=1"]


def test_the_relay_exits_with_its_parent(tmp_path) -> None:
    """Started from the batch shell, it must die with the job rather than outlive it."""
    script = f'"{host_python()}" "{RELAY}" "{tmp_path}" >/dev/null 2>&1 & echo $!; sleep 1'
    with subprocess.Popen(["/bin/sh", "-c", script], stdout=subprocess.PIPE, text=True) as shell:
        pid = int(shell.stdout.readline())
        shell.wait()
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    os.kill(pid, 9)
    pytest.fail("relay outlived its parent")
