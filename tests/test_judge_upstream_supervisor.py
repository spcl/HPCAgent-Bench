# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""hpcagent_bench/cluster/judge_upstream.py: a judge rank that loses its upstream gets it back.

The upstream must not be a bare background child of the judge step's shell: when the OOM killer
takes it, the router in front of it keeps answering /health with 200 and every grade behind it
comes back 502 -- for hours, with no recorded row, while the sibling judges on the same node keep
working.
"""

import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable

import pytest

from hpcagent_bench import paths
from hpcagent_bench.cluster import judge_upstream

REPO = paths.ROOT
SUPERVISOR = REPO / "hpcagent_bench" / "cluster" / "judge_upstream.py"
RUN_CLUSTER = REPO / "hpcagent_bench" / "cluster" / "run_cluster.sh"

#: An upstream that records every start and then dies the way the OOM killer ends one.
OOM_KILLED = "import os, signal, sys; open(sys.argv[1], 'a').write('start\\n'); os.kill(os.getpid(), signal.SIGKILL)"
#: An upstream that stays up and can be found by pid, the way a healthy judge does.
LONG_LIVED = "import os, sys, time; open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(300)"
#: An upstream that ignores the shutdown signal, the way one stuck in native code does.
IGNORES_TERM = (
    "import os, signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(300)"
)
#: An upstream that fails immediately every time, the way a judge with a taken port or a refused seal does.
CRASH_LOOP = "import sys; sys.exit(2)"


def supervise(
    script: str, *args: str, label: str = "rank=4", uptime: str = "0", quick: str = "3"
) -> "subprocess.Popen[str]":
    """The supervisor running ``python -c script args``, with its output captured."""
    return subprocess.Popen(
        [
            sys.executable,
            str(SUPERVISOR),
            "--label",
            label,
            "--min-uptime-seconds",
            uptime,
            "--max-quick-restarts",
            quick,
            "--backoff-seconds",
            "0.05",
            "--",
            sys.executable,
            "-c",
            script,
            *args,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def finish(proc: "subprocess.Popen[str]") -> str:
    """The stopped supervisor's output. One that does not stop is killed: left behind, it respawns its child for ever."""
    try:
        return proc.communicate(timeout=30)[0]
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise


def wait_for(predicate: Callable[[], bool], timeout: float = 30.0) -> bool:
    """Poll ``predicate`` until it holds or the timeout runs out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_an_oom_killed_upstream_is_started_again(tmp_path: pathlib.Path) -> None:
    """The OOM killer takes ONE process and the node has its memory back a second later; a rank
    that does not come back loses every grade for the rest of the allocation."""
    starts = tmp_path / "starts"
    proc = supervise(OOM_KILLED, str(starts))
    try:
        assert wait_for(lambda: starts.exists() and len(starts.read_text().splitlines()) >= 3)
    finally:
        proc.terminate()
        output = finish(proc)
    assert proc.returncode == 0
    assert "signal=SIGKILL" in output, output


def test_stopping_the_supervisor_takes_the_upstream_with_it(tmp_path: pathlib.Path) -> None:
    """The judge step's teardown kills the supervisor. A child left behind would hold the loopback
    port and its grading children for the rest of the allocation."""
    pidfile = tmp_path / "pid"
    proc = supervise(LONG_LIVED, str(pidfile))
    assert wait_for(lambda: pidfile.exists() and pidfile.read_text().strip().isdigit())
    child = int(pidfile.read_text().strip())
    proc.send_signal(signal.SIGTERM)
    finish(proc)
    assert proc.returncode == 0
    assert wait_for(lambda: not process_alive(child), timeout=15)


def test_stopping_escalates_when_the_upstream_ignores_sigterm(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(judge_upstream, "TERM_GRACE_S", 0.2)
    pidfile = tmp_path / "pid"
    supervisor = judge_upstream.Supervisor([sys.executable, "-c", IGNORES_TERM, str(pidfile)], "rank=4", 60.0, 3, 0.05)
    results: list[int] = []
    runner = threading.Thread(target=lambda: results.append(supervisor.run()), daemon=True)
    runner.start()
    assert wait_for(lambda: pidfile.exists() and pidfile.read_text().strip().isdigit())
    supervisor.stop(signal.SIGTERM, None)
    runner.join(timeout=5)
    child = supervisor.child
    if runner.is_alive() and child is not None:
        child.kill()
    assert results == [0]
    assert child is not None
    assert child.returncode == -signal.SIGKILL


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_an_upstream_that_cannot_start_ends_the_supervisor_instead_of_looping() -> None:
    """A taken port or a refused seal fails in seconds every time. Restarting that forever would
    hide a judge that never came up, which the launcher's readiness loop reads off this exit."""
    proc = supervise(CRASH_LOOP, uptime="30", quick="2")
    output = proc.communicate(timeout=60)[0]
    assert proc.returncode == 1, output
    assert "giving up" in output, output
    assert output.count("rc=2") == 2, output


@pytest.mark.parametrize("needle", ["judge_upstream.py", "--min-uptime-seconds", "--max-quick-restarts"])
def test_the_launcher_starts_the_upstream_through_the_supervisor(needle: str) -> None:
    """run_cluster.sh must not go back to a bare background `hpcagent_bench serve`."""
    text = RUN_CLUSTER.read_text(encoding="utf-8")
    assert needle in text


def test_a_fatal_signal_in_the_judge_leaves_a_traceback() -> None:
    """A rank killed by a fatal signal ends its log mid-line and says nothing. faulthandler is what
    turns that into evidence instead of a guess."""
    probe = (
        "import faulthandler, os, signal;"
        "from hpcagent_bench.harness.service import enable_crash_traces;"
        "enable_crash_traces();"
        "print(faulthandler.is_enabled(), flush=True);"
        "os.kill(os.getpid(), signal.SIGSEGV)"
    )
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)
    assert done.stdout.strip() == "True", done.stdout
    assert "Fatal Python error" in done.stderr, done.stderr
