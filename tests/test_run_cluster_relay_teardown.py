# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``run_cluster.sh`` must end promptly with the gang relay running.

The relay (``experiments/gang_relay.py``) runs as a background child of the batch shell and exits only
once that shell is gone. ``cleanup_steps_on_exit`` / ``cleanup_steps_on_signal`` end in a bare
``wait``, which reaps every child, the relay included: without stopping the relay first the two wait
on each other and the job idles to its time limit (mlscale jobs 649109-649111 and 649795, 2.5-5 h).

The script cannot be sourced, so the relay start block and both cleanup handlers are lifted from its
shipped text and run against the real relay, the way ``test_run_cluster_job_env_trap.py`` does.
"""

import os
import pathlib
import signal
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "experiments" / "run_cluster.sh").read_text()

TEARDOWN_BLOCK = TEXT[TEXT.index("# FROZEN TREE REMOVAL.") : TEXT.index(': "${SLURM_JOB_ID:?')]
CLEANUP_END = "trap cleanup_steps_on_signal INT TERM\n"
CLEANUP_BLOCK = TEXT[TEXT.index("step_pids=()\n") : TEXT.index(CLEANUP_END) + len(CLEANUP_END)]
RELAY_START = "if gang_judge; then\n    export HPCAGENT_BENCH_GANG_RELAY_DIR="
RELAY_BLOCK = TEXT[TEXT.index(RELAY_START) : TEXT.index("fi\n", TEXT.index(RELAY_START)) + len("fi\n")]

#: Far below the relay's own lifetime (unbounded while its parent lives), far above a clean teardown.
DEADLINE_S = 20.0


def build(tmp_path: pathlib.Path, tail: str) -> pathlib.Path:
    """A script: the real cleanup handlers, a stand-in role step, the real relay start, then ``tail``."""
    script = tmp_path / "relay_teardown.sh"
    script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"SCRIPT_DIR={REPO / 'experiments'}\nRUN_DIR={tmp_path}\ngang_judge() {{ true; }}\n"
        f'{TEARDOWN_BLOCK}{CLEANUP_BLOCK}sleep 300 & step_pids+=("$!")\n{RELAY_BLOCK}'
        'echo "relay ${gang_relay_pid}"\n'
        f"{tail}\n"
    )
    return script


def relay_env() -> dict[str, str]:
    """The batch host's interpreter for the relay."""
    return {"PATH": "/usr/bin:/bin", "HPCAGENT_BENCH_HOST_PYTHON": sys.executable}


def wait_for_relay(tmp_path: pathlib.Path) -> None:
    """Block until the relay published its heartbeat, i.e. it is serving."""
    deadline = time.monotonic() + DEADLINE_S
    while not (tmp_path / "gang-relay" / "relay.alive").exists():
        assert time.monotonic() < deadline, (tmp_path / "gang-relay.log").read_text()
        time.sleep(0.1)


def relay_pid(stdout: str) -> int:
    """The relay pid the script printed."""
    return int(next(line for line in stdout.splitlines() if line.startswith("relay ")).split()[1])


def assert_gone(pid: int) -> None:
    """``pid`` no longer runs (reaped, or at worst a zombie of an already-dead parent)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    stat = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    assert stat == "Z", f"relay {pid} outlived the job shell (state {stat})"


def test_a_plain_exit_stops_the_relay_and_returns(tmp_path: pathlib.Path) -> None:
    """The job's normal end: the EXIT trap returns within seconds and the relay is gone."""
    script = build(tmp_path, "while [[ ! -e ${RUN_DIR}/gang-relay/relay.alive ]]; do sleep 0.1; done")
    result = subprocess.run(
        ["bash", str(script)], env=relay_env(), capture_output=True, text=True, check=True, timeout=DEADLINE_S
    )
    assert_gone(relay_pid(result.stdout))


def test_a_sigterm_stops_the_relay_and_returns(tmp_path: pathlib.Path) -> None:
    """scancel / time limit: the INT/TERM handler's ``wait`` returns too, then the EXIT trap runs."""
    script = build(tmp_path, 'wait "${step_pids[0]}" || true')
    with subprocess.Popen(["bash", str(script)], env=relay_env(), stdout=subprocess.PIPE, text=True) as proc:
        try:
            wait_for_relay(tmp_path)
            proc.send_signal(signal.SIGTERM)
            stdout, _ = proc.communicate(timeout=DEADLINE_S)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    assert_gone(relay_pid(stdout))
