# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""C7: the two backstops ``run_agent`` enforces do not scope the same way (docs/token_accounting.md,
"``AGENT_MAX_TOKENS`` is a PER-ATTEMPT cap"), and nothing before this file exercised either scoping
rule through a real relaunch:

* ``AGENT_TIMEOUT_SECONDS`` is the PROBLEM's deadline: ``deadline`` is computed once, before the
  attempt loop, and a relaunch's ``process.wait(timeout=...)`` gets what is left of it -- never a
  fresh full budget.
* ``AGENT_MAX_TOKENS`` is the ATTEMPT's cap: the watcher folds only the attempt's own (freshly
  truncated) transcript, so a cap the two attempts' COMBINED spend would trip must not fire on a
  final attempt that never spent it alone.

Both tests drive ``agent_driver.run_agent`` through the golden-fixture harness
(``tests/fixtures/claude_driver_golden/regen.py``) with a crash-then-success attempt pair, the same
shape ``test_agent_driver_fresh_relaunch.py`` uses, and a purpose-built fake process for each
property: one that reports back the exact ``timeout=`` it was given, one whose ``wait()`` genuinely
blocks so a live watcher thread has real wall-clock time to poll before the attempt ends.
"""

import importlib.util
import json
import pathlib
import subprocess
import sys
import time
import types
from collections.abc import Callable
from types import ModuleType
from typing import TextIO

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "claude_driver_golden"
DRIVER = REPO / "experiments" / "agent_driver.py"


def load_capture() -> ModuleType:
    """The golden capture harness beside the fixtures: env/paths shared with the fresh-relaunch test."""
    spec = importlib.util.spec_from_file_location("relaunch_budgets_capture", FIXTURES / "regen.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {FIXTURES / 'regen.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


capture = load_capture()


def usage(input_tokens: int = 0, cache_creation: int = 0, cache_read: int = 0, output_tokens: int = 0) -> dict:
    """A ``message.usage`` block with all four consumed-token fields the CLI reports."""
    return {
        "input_tokens": input_tokens,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
        "output_tokens": output_tokens,
    }


def assistant_line(message_id: str, usage_block: dict) -> str:
    return json.dumps({"type": "assistant", "message": {"id": message_id, "role": "assistant", "usage": usage_block}})


#: The MCP init event every attempt's transcript needs, or ``start_agent`` reads the missing status
#: as a failed server and retries the launch instead of running the attempt this test wants to see.
CONNECTED_INIT = json.dumps(
    {"type": "system", "subtype": "init", "mcp_servers": [{"name": "hpcagent-bench", "status": "connected"}]}
)


def run_env(**overrides: str) -> tuple[tuple[str, str], ...]:
    env = dict(capture.BASE_ENV)
    env.update(overrides)
    return tuple(env.items())


# ---------------------------------------------------------------------------
# C7a: the wall clock is the PROBLEM's, shared across attempts.
# ---------------------------------------------------------------------------


class TimingProcess:
    """A recorded process that reports exactly what ``timeout=`` it was waited with.

    ``wait`` never actually blocks -- the clock here is the monkeypatched ``time.monotonic`` below,
    not real time -- so every attempt's own ``remaining`` is captured for inspection afterwards.
    """

    def __init__(self, code: int) -> None:
        self.code = code
        self.returncode: int | None = None
        self.pid = 0
        self.wait_timeouts: list[float | None] = []

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            self.returncode = self.code
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


#: One crash, then a clean run: exactly the sequence that exercises one relaunch.
CRASH_LOG = "\n".join(
    [
        CONNECTED_INIT,
        assistant_line("a_msg_1", usage(input_tokens=3000, output_tokens=100)),
        "Error: read ECONNRESET",  # no closing result event: an unrecorded nonzero exit is a crash
    ]
)
SUCCESS_LOG = "\n".join(
    [
        CONNECTED_INIT,
        assistant_line("b_msg_1", usage(input_tokens=4000, output_tokens=200)),
        json.dumps({"type": "result", "subtype": "success", "num_turns": 1}),
    ]
)


def timing_spawner(
    processes: list[TimingProcess], clock: list[float], advance_at_attempt: dict[int, float]
) -> Callable[[list[str], pathlib.Path, dict[str, str], TextIO, int], TimingProcess]:
    """A spawner whose SECOND (or later) launch ratchets the fake clock forward by a fixed amount,
    modelling wall time the previous, now-finished attempt actually spent -- not a value picked to
    make the assertion true, but the same amount ``time.monotonic`` reports to every caller from
    that point on, including the ``remaining`` computation this test reads."""

    def spawn(command: list[str], cwd: pathlib.Path, env: dict[str, str], stdout: TextIO, stderr: int) -> TimingProcess:
        index = len(processes)
        if index in advance_at_attempt:
            clock[0] += advance_at_attempt[index]
        log = CRASH_LOG if index == 0 else SUCCESS_LOG
        stdout.write(log + "\n")
        stdout.flush()
        process = TimingProcess(1 if index == 0 else 0)
        processes.append(process)
        return process

    return spawn


def test_a_relaunch_waits_only_the_remaining_wall_clock_not_a_fresh_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """docs/token_accounting.md: ``AGENT_TIMEOUT_SECONDS`` is ONE deadline shared by every attempt of
    a problem. If a relaunch instead started its own fresh clock, three relaunches would hold a
    worker for three times the wall the arm was sized against -- the exact regression this pins.
    """
    import shutil

    shutil.copytree(FIXTURES / "templates", tmp_path / "shared")
    processes: list[TimingProcess] = []
    clock = [0.0]
    # Attempt 1 starts at clock=0; by the time attempt 2 is spawned (a relaunch, after attempt 1
    # crashed), 40s of the shared 100s budget are already gone.
    spawn = timing_spawner(processes, clock, {1: 40.0})

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    with capture.isolated(tmp_path, run_env(AGENT_TIMEOUT_SECONDS="100")):
        driver = capture.load_driver(DRIVER)
        driver.subprocess = types.SimpleNamespace(
            Popen=spawn,
            STDOUT=subprocess.STDOUT,
            TimeoutExpired=subprocess.TimeoutExpired,
            SubprocessError=subprocess.SubprocessError,
            run=subprocess.run,
        )
        driver.agent_cpus = lambda worker_index, agents: []
        driver.claude_supports_flag = lambda binary, flag: True
        problem = {"id": 7, "kernel": capture.KERNEL, "language": "c", "task": capture.TASK}
        returncode = driver.run_agent(problem, 2, capture.NODE_DIR, list(capture.JUDGES), 7, 3)

    assert len(processes) == 2, "the crash must have relaunched exactly once"
    assert returncode == 0
    # Attempt 1 starts the full 100s budget: nothing has been spent on the problem's clock yet.
    assert processes[0].wait_timeouts == [100.0]
    # Attempt 2 gets what is LEFT of the same 100s deadline (100 - 40 = 60), not a fresh 100 --
    # the bug this pins is a relaunch that recomputes `deadline` instead of reusing it.
    assert processes[1].wait_timeouts == [60.0]
    assert processes[1].wait_timeouts[0] < 100.0, "a relaunch must not receive a fresh full timeout"


# ---------------------------------------------------------------------------
# C7b: the token cap is the ATTEMPT's, reset on every relaunch.
# ---------------------------------------------------------------------------


class SlowProcess:
    """A recorded process whose ``wait()`` genuinely blocks in real time.

    An instantly-returning fake process (as ``TimingProcess`` above, or the golden fixtures'
    ``RecordedProcess``) can let the main thread finish ``process.wait()`` before the watcher
    thread's first ``time.sleep(TOKEN_POLL_SECONDS)`` even elapses, so the watcher may never poll
    the transcript at all -- a token-cap test built on one would pass whether or not the cap
    actually reset, because the fold underneath it never ran. Blocking here for a real, short
    interval gives the watcher thread the same kind of window ``test_agent_driver_budget.py``'s real
    subprocess gives it, without spawning one.
    """

    def __init__(self, code: int, hold_s: float = 0.2) -> None:
        self.code = code
        self.returncode: int | None = None
        self.pid = 0
        self._hold_s = hold_s

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            time.sleep(self._hold_s)
            self.returncode = self.code
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


#: Attempt 1's own total is 3100 (a_msg_1: 3000 input + 100 output); attempt 2's own total is 4200
#: (b_msg_1: 4000 input + 200 output). Neither exceeds 5000 alone; their SUM (7300) does. A cap of
#: 5000 therefore trips only if a relaunch's watcher is charged for what the crashed attempt spent.
SLOW_CRASH_LOG = "\n".join(
    [CONNECTED_INIT, assistant_line("a_msg_1", usage(input_tokens=3000, output_tokens=100)), "Error: read ECONNRESET"]
)
SLOW_SUCCESS_LOG = "\n".join(
    [
        CONNECTED_INIT,
        assistant_line("b_msg_1", usage(input_tokens=4000, output_tokens=200)),
        json.dumps({"type": "result", "subtype": "success", "num_turns": 1}),
    ]
)
TOKEN_RESET_CAP = 5000


def slow_spawner(launches: list[int]) -> Callable[[list[str], pathlib.Path, dict[str, str], TextIO, int], SlowProcess]:
    def spawn(command: list[str], cwd: pathlib.Path, env: dict[str, str], stdout: TextIO, stderr: int) -> SlowProcess:
        index = len(launches)
        launches.append(index)
        log = SLOW_CRASH_LOG if index == 0 else SLOW_SUCCESS_LOG
        stdout.write(log + "\n")
        stdout.flush()
        return SlowProcess(1 if index == 0 else 0)

    return spawn


def test_a_relaunch_starts_the_token_cap_from_zero_not_the_crashed_attempts_count(tmp_path: pathlib.Path) -> None:
    """docs/token_accounting.md: ``AGENT_MAX_TOKENS`` resets on every relaunch because the watcher
    reads the transcript it is handed, and a relaunch writes a NEW one. Attempt 1 alone (3100) and
    attempt 2 alone (4200) both stay under the 5000 cap here -- IF and only if attempt 2's watcher
    is not also charged attempt 1's 3100. A watcher that let its running total carry across attempts
    would see 3100 + 4200 = 7300 > 5000 and kill a final attempt that never overspent on its own.
    """
    import shutil

    shutil.copytree(FIXTURES / "templates", tmp_path / "shared")
    launches: list[int] = []
    with capture.isolated(tmp_path, run_env(AGENT_MAX_TOKENS=str(TOKEN_RESET_CAP))):
        driver = capture.load_driver(DRIVER)
        driver.subprocess = types.SimpleNamespace(
            Popen=slow_spawner(launches),
            STDOUT=subprocess.STDOUT,
            TimeoutExpired=subprocess.TimeoutExpired,
            SubprocessError=subprocess.SubprocessError,
            run=subprocess.run,
        )
        driver.agent_cpus = lambda worker_index, agents: []
        driver.claude_supports_flag = lambda binary, flag: True
        driver.TOKEN_POLL_SECONDS = 0.01
        problem = {"id": 7, "kernel": capture.KERNEL, "language": "c", "task": capture.TASK}
        returncode = driver.run_agent(problem, 2, capture.NODE_DIR, list(capture.JUDGES), 7, 3)

    assert launches == [0, 1], "the crash must have relaunched exactly once"
    assert returncode == 0, (
        "the final attempt spent only 4200 of its OWN transcript and must finish clean; "
        f"got rc={returncode} (125 is RC_TOKEN_BUDGET -- the cap fired on combined spend)"
    )
