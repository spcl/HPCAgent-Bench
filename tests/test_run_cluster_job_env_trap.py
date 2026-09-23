# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``experiments/run_cluster.sh`` sets an EXIT trap where it creates ``JOB_ENV_FILE`` (the tmpfs
copy of the job env that podman/docker read; it carries the inference key) and a SECOND, unrelated
EXIT trap later where it defined ``cleanup_steps``. Bash keeps only the LAST trap registered for a
given signal, so the second trap silently replaced the first one and the mktemp'd env file was never
removed -- on a real job or a plain successful exit. The fix folded the removal into that later
trap.

``cleanup_steps`` has since split into two: ``cleanup_steps_on_exit`` (EXIT only) and
``cleanup_steps_on_signal`` (INT/TERM). JOB_ENV_FILE is removed ONLY on EXIT: an INT/TERM here falls
through into the mandatory token-record extraction further down in the real file instead of exiting,
and that extraction's containerized call still needs the file (podman/docker only) to exist at that
point, so removing it from the signal path would recreate a version of the very bug this test file
exists to catch -- just moved from "never removed" to "removed too early". These tests lift the
exact creation lines and the exact ``cleanup_steps_on_exit`` / ``cleanup_steps_on_signal`` / `trap`
lines straight out of the file (never retyped) and run only those, the same "read the real text, run
only the real text" approach ``test_run_cluster_cache_env.py`` and ``test_run_cluster_frozen_tree.py``
already take for this file."""

import pathlib
import signal
import subprocess
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "experiments" / "run_cluster.sh").read_text()

CREATE_START = 'job_env_dir="${XDG_RUNTIME_DIR:-}"'
CREATE_END = 'chmod 600 "${JOB_ENV_FILE}"\n'
CREATE_BLOCK = TEXT[TEXT.index(CREATE_START) : TEXT.index(CREATE_END) + len(CREATE_END)]

CLEANUP_START = "step_pids=()\n"
CLEANUP_END = "trap cleanup_steps_on_signal INT TERM\n"
CLEANUP_BLOCK = TEXT[TEXT.index(CLEANUP_START) : TEXT.index(CLEANUP_END) + len(CLEANUP_END)]

# JOB_ENV_FILE is never removed by its own creation-site trap any more (see docstring); this test
# pins that the old, now-dead `trap ... EXIT` line stays gone rather than quietly creeping back in
# and shadowing cleanup_steps_on_exit's removal again.
assert "trap 'rm -f " not in CREATE_BLOCK, "a creation-site EXIT trap on JOB_ENV_FILE reappeared"
assert 'rm -f "${JOB_ENV_FILE:-}"' in CLEANUP_BLOCK, "cleanup_steps_on_exit no longer removes JOB_ENV_FILE"
# The removal must stay EXIT-only: an INT/TERM falls through into extraction in the real file, which
# still needs the file to exist. Pin this by checking the two trap registrations land on the
# functions this test expects, and that the removal line appears strictly before the INT/TERM trap
# is registered (i.e. inside cleanup_steps_on_exit, not cleanup_steps_on_signal).
assert "trap cleanup_steps_on_exit EXIT\n" in CLEANUP_BLOCK, "the EXIT trap no longer names cleanup_steps_on_exit"
_rm_index = CLEANUP_BLOCK.index('rm -f "${JOB_ENV_FILE:-}"')
_exit_trap_index = CLEANUP_BLOCK.index("trap cleanup_steps_on_exit EXIT\n")
assert _rm_index < _exit_trap_index, "JOB_ENV_FILE's removal moved out of cleanup_steps_on_exit"


def build(tmp_path: pathlib.Path, tail: str) -> pathlib.Path:
    script = tmp_path / "trap_check.sh"
    script.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{CREATE_BLOCK}{CLEANUP_BLOCK}\n{tail}\n")
    script.chmod(0o755)
    return script


def env_files(runtime_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(runtime_dir.glob("job.env.*"))


def test_a_plain_exit_removes_the_job_env_file(tmp_path: pathlib.Path) -> None:
    """Normal completion: cleanup_steps_on_exit's EXIT trap fires and the mktemp'd file is gone."""
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
    """The file is still gone once the process actually exits after a TERM (scancel, or the job's
    time limit), not only on a clean exit -- even though cleanup_steps_on_signal (the INT/TERM trap)
    itself no longer removes it directly (see docstring). Bash still runs the EXIT trap
    (cleanup_steps_on_exit) once the script falls off its own end, which is what actually removes
    the file here, the same way falling through to the real file's mandatory extraction ends in its
    own `exit` and the same EXIT trap.

    ``sleep`` is backgrounded and waited on, not run in the foreground: bash only runs a caught
    signal's trap between commands (or when a `wait` is interrupted), never while it is itself
    blocked in `waitpid` for a FOREGROUND child, so a foreground `sleep 30` would just eat the
    signal for the full 30s and defeat what this test is checking. It goes into step_pids (the
    real script's own bookkeeping for its role steps) so cleanup_steps_on_signal's kill loop --
    not just cleanup_steps_on_exit's trailing `rm` -- actually reaps it instead of a later bare
    `wait` blocking on an orphan.
    """
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    script = build(tmp_path, 'echo "created: ${JOB_ENV_FILE}"\nsleep 30 & step_pids+=("$!")\nwait "${step_pids[-1]}"')
    # The with-block closes the stdout pipe on the way out, whichever way the body leaves.
    with subprocess.Popen(
        ["bash", str(script)],
        env={"PATH": "/usr/bin:/bin", "XDG_RUNTIME_DIR": str(runtime_dir)},
        stdout=subprocess.PIPE,
        text=True,
    ) as proc:
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
