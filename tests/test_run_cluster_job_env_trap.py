# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``experiments/run_cluster.sh`` sets an EXIT trap where it creates ``JOB_ENV_FILE`` (the tmpfs
copy of the job env that podman/docker read; it carries the inference key) and a SECOND, unrelated
EXIT trap later where it defines ``cleanup_steps``. Bash keeps only the LAST trap registered for a
given signal, so the second trap silently replaced the first one and the mktemp'd env file was never
removed -- on a real job or a plain successful exit. The fix folds the removal into ``cleanup_steps``
itself, guarded for ``set -u``. These tests lift the exact creation lines and the exact
``cleanup_steps``/``trap`` lines straight out of the file (never retyped) and run only those, the
same "read the real text, run only the real text" approach ``test_run_cluster_cache_env.py`` and
``test_run_cluster_frozen_tree.py`` already take for this file."""

import pathlib
import signal
import subprocess
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "experiments" / "run_cluster.sh").read_text()

CREATE_START = 'job_env_dir="${XDG_RUNTIME_DIR:-}"'
CREATE_END = 'chmod 600 "${JOB_ENV_FILE}"\n'
CREATE_BLOCK = TEXT[TEXT.index(CREATE_START) : TEXT.index(CREATE_END) + len(CREATE_END)]

CLEANUP_START = "step_pids=()\ncleanup_steps() {"
CLEANUP_END = "trap cleanup_steps EXIT INT TERM\n"
CLEANUP_BLOCK = TEXT[TEXT.index(CLEANUP_START) : TEXT.index(CLEANUP_END) + len(CLEANUP_END)]

# JOB_ENV_FILE is never removed by its own creation-site trap any more (see docstring); this test
# pins that the old, now-dead `trap ... EXIT` line stays gone rather than quietly creeping back in
# and shadowing cleanup_steps's removal again.
assert "trap 'rm -f " not in CREATE_BLOCK, "a creation-site EXIT trap on JOB_ENV_FILE reappeared"
assert 'rm -f "${JOB_ENV_FILE:-}"' in CLEANUP_BLOCK, "cleanup_steps no longer removes JOB_ENV_FILE"


def build(tmp_path: pathlib.Path, tail: str) -> pathlib.Path:
    script = tmp_path / "trap_check.sh"
    script.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{CREATE_BLOCK}{CLEANUP_BLOCK}\n{tail}\n")
    script.chmod(0o755)
    return script


def env_files(runtime_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(runtime_dir.glob("job.env.*"))


def test_a_plain_exit_removes_the_job_env_file(tmp_path: pathlib.Path) -> None:
    """Normal completion: cleanup_steps's EXIT trap fires and the mktemp'd file is gone."""
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    script = build(tmp_path, 'echo "created: ${JOB_ENV_FILE}"')
    result = subprocess.run(
        ["bash", str(script)],
        env={"PATH": "/usr/bin:/bin", "XDG_RUNTIME_DIR": str(runtime_dir)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "created: " in result.stdout, result.stdout
    assert env_files(runtime_dir) == [], "JOB_ENV_FILE survived a plain exit"


def test_a_sigterm_still_removes_the_job_env_file(tmp_path: pathlib.Path) -> None:
    """The trap runs on TERM too (scancel, or the job's time limit), not only on a clean exit.

    ``sleep`` is backgrounded and waited on, not run in the foreground: bash only runs a caught
    signal's trap between commands (or when a `wait` is interrupted), never while it is itself
    blocked in `waitpid` for a FOREGROUND child, so a foreground `sleep 30` would just eat the
    signal for the full 30s and defeat what this test is checking. It goes into step_pids (the
    real script's own bookkeeping for its role steps) so cleanup_steps's kill loop -- not just its
    trailing `rm` -- actually reaps it instead of the later bare `wait` blocking on an orphan.
    """
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    script = build(tmp_path, 'echo "created: ${JOB_ENV_FILE}"\nsleep 30 & step_pids+=("$!")\nwait "${step_pids[-1]}"')
    proc = subprocess.Popen(
        ["bash", str(script)],
        env={"PATH": "/usr/bin:/bin", "XDG_RUNTIME_DIR": str(runtime_dir)},
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not env_files(runtime_dir) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert env_files(runtime_dir), "JOB_ENV_FILE was never created before the deadline"
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert env_files(runtime_dir) == [], "JOB_ENV_FILE survived SIGTERM"
