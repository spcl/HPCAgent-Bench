# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Restarting an agent whose MCP server did not connect, and capping how many start at once.

A failed MCP server is not a crash. The agent keeps its built-in tools, loses score/submit/task
entirely, burns its whole budget and exits reporting success -- measured on qwen 604475, where one
such agent ran 36 minutes over 54 turns and called a `Submit` tool that does not exist. The harness
records rc=0 and the data point is simply gone, so the driver has to notice and relaunch.
"""

import importlib
import json
import pathlib
import sys
import threading
import time

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "containers/cluster/example-script"


def load_driver(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sys.path.insert(0, str(EXAMPLE))
    try:
        return importlib.reload(importlib.import_module("agent_driver"))
    finally:
        sys.path.remove(str(EXAMPLE))


def init_line(status):
    return json.dumps({"subtype": "init", "mcp_servers": [{"name": "optarena", "status": status}]}) + "\n"


def fake_popen_class(statuses, seen=None):
    """Popen stand-in that writes one init event per spawn, taking each status in turn."""

    class FakePopen:
        spawned = 0
        live = 0

        def __init__(self, command, cwd=None, env=None, stdout=None, stderr=None):
            cls = type(self)
            cls.spawned += 1
            cls.live += 1
            if seen is not None:
                seen.append(cls.live)
            self.returncode = None
            status = statuses[min(cls.spawned - 1, len(statuses) - 1)]
            stdout.write(init_line(status))
            stdout.flush()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = self.returncode or 0
            return self.returncode

        def terminate(self):
            type(self).live -= 1
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    return FakePopen


#: The CPU share start_agent pins the child to. Empty is its own "leave the mask alone" path,
#: which is what these tests want: they drive a fake Popen with no real pid, and the pinning
#: itself is covered against a live process in test_agent_driver_affinity.py.
_UNPINNED: list[int] = []


def start(driver, monkeypatch, tmp_path, statuses, seen=None):
    monkeypatch.setattr(driver.subprocess, "Popen", fake_popen_class(statuses, seen))
    log_path = tmp_path / "claude.log"
    with log_path.open("w", encoding="utf-8") as log:
        process, attempts = driver.start_agent(["claude"], tmp_path, {}, log, log_path, _UNPINNED)
    return process, attempts, log_path


def test_a_connected_server_is_not_retried(monkeypatch, tmp_path):
    driver = load_driver(monkeypatch)
    _, attempts, _ = start(driver, monkeypatch, tmp_path, ["connected"])
    assert attempts == 1


def test_a_failed_server_is_relaunched_until_it_connects(monkeypatch, tmp_path):
    driver = load_driver(monkeypatch)
    _, attempts, log_path = start(driver, monkeypatch, tmp_path, ["failed", "failed", "connected"])
    assert attempts == 3
    assert driver.mcp_failed(log_path) is False, "the surviving transcript is the connected attempt"


def test_the_retries_are_bounded_and_the_agent_still_runs(monkeypatch, tmp_path):
    """Exhausting the attempts must not lose the agent -- a crippled run still beats no run, and the
    log has to say which one this was."""
    driver = load_driver(monkeypatch, AGENT_MCP_ATTEMPTS="2")
    process, attempts, log_path = start(driver, monkeypatch, tmp_path, ["failed"])
    assert attempts == 2
    assert process is not None
    assert "MCP still not connected" in log_path.read_text(encoding="utf-8")


def test_a_retry_leaves_no_half_transcript(monkeypatch, tmp_path):
    """Downstream readers take the LAST result event; a dead attempt's output must not linger."""
    driver = load_driver(monkeypatch)
    _, _, log_path = start(driver, monkeypatch, tmp_path, ["failed", "connected"])
    assert log_path.read_text(encoding="utf-8").count('"subtype": "init"') == 1


def test_mcp_failed_distinguishes_not_yet_from_connected(monkeypatch, tmp_path):
    driver = load_driver(monkeypatch)
    log_path = tmp_path / "claude.log"
    log_path.write_text("", encoding="utf-8")
    assert driver.mcp_failed(log_path) is None, "no init event yet is not the same as a good one"
    log_path.write_text(init_line("connected"), encoding="utf-8")
    assert driver.mcp_failed(log_path) is False
    log_path.write_text(init_line("failed"), encoding="utf-8")
    assert driver.mcp_failed(log_path) is True


def test_only_so_many_agents_start_at_once(monkeypatch, tmp_path):
    """The whole point: 120 pool threads must not put 120 python3 servers on one node at once.

    The stand-in blocks inside the spawn, which is where the real one sits too -- the gate is held
    across the spawn AND the wait for the init event, so a slow startup holds its slot.
    """
    driver = load_driver(monkeypatch, AGENT_START_CONCURRENCY="3")
    release = threading.Event()
    spawned: list[int] = []
    lock = threading.Lock()

    class BlockingPopen:
        def __init__(self, command, cwd=None, env=None, stdout=None, stderr=None):
            with lock:
                spawned.append(1)
            self.returncode = None
            release.wait(timeout=30)
            stdout.write(init_line("connected"))
            stdout.flush()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(driver.subprocess, "Popen", BlockingPopen)
    threads = []
    for index in range(12):
        workdir = tmp_path / str(index)
        workdir.mkdir()

        def run(workdir=workdir):
            log_path = workdir / "claude.log"
            with log_path.open("w", encoding="utf-8") as log:
                driver.start_agent(["claude"], workdir, {}, log, log_path, _UNPINNED)

        threads.append(threading.Thread(target=run))
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 5
    while len(spawned) < 3 and time.monotonic() < deadline:
        time.sleep(0.05)
    held = len(spawned)
    release.set()
    for thread in threads:
        thread.join(timeout=30)
    assert held == 3, f"{held} agents were inside startup at once, cap is 3"
    assert len(spawned) == 12, "every agent must still get its turn"


def test_a_crash_is_a_fault_but_a_budget_is_not(monkeypatch, tmp_path):
    """604475/604476 ended 69 of 240 agents on the wall clock and nothing else. A timed-out agent
    spent what it was given and keeps every submission it made; relaunching it would hand it a
    second budget its peers never had."""
    driver = load_driver(monkeypatch)
    log = tmp_path / "claude.log"
    log.write_text("", encoding="utf-8")
    assert driver.crashed(1, log) is True, "a bare nonzero exit with no result event is a fault"
    for code in (0, driver.RC_TIMEOUT, driver.RC_TOKEN_BUDGET, driver.RC_CONTEXT):
        assert driver.crashed(code, log) is False, f"rc={code} is a budget, not a fault"


def test_a_reported_result_is_the_cli_verdict_not_a_crash(monkeypatch, tmp_path):
    """A nonzero exit AFTER the CLI wrote its result is that run's answer; relaunching overwrites it."""
    driver = load_driver(monkeypatch)
    log = tmp_path / "claude.log"
    log.write_text(json.dumps({"type": "result", "subtype": "error_during_execution"}) + "\n", encoding="utf-8")
    assert driver.crashed(1, log) is False


def test_crash_retries_are_bounded(monkeypatch):
    monkeypatch.delenv("AGENT_CRASH_ATTEMPTS", raising=False)
    driver = load_driver(monkeypatch)
    assert 2 <= driver.AGENT_CRASH_ATTEMPTS <= 5


def crashing_popen_class(exit_codes):
    """Popen stand-in whose Nth spawn exits with ``exit_codes[N-1]`` and marks its own transcript.

    Each spawn writes a connected init event -- the MCP path is not what these exercise -- plus a
    line naming its attempt, which is how the test tells the preserved transcripts apart.
    """

    class FakePopen:
        spawned = 0

        def __init__(self, command, cwd=None, env=None, stdout=None, stderr=None):
            cls = type(self)
            cls.spawned += 1
            self.attempt = cls.spawned
            self.returncode = None
            stdout.write(init_line("connected"))
            stdout.write(f"ATTEMPT {self.attempt}\n")
            stdout.flush()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = exit_codes[min(self.attempt - 1, len(exit_codes) - 1)]
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    return FakePopen


def supervise(driver, monkeypatch, tmp_path, exit_codes):
    """Drive run_agent over a fake CLI, returning the worker directory it wrote into."""
    monkeypatch.setattr(driver.subprocess, "Popen", crashing_popen_class(exit_codes))
    monkeypatch.setattr(driver, "vllm_urls", lambda: ["http://127.0.0.1:8000"])
    monkeypatch.setattr(driver, "write_cost_record", lambda *a, **k: None)
    node_dir = tmp_path / "node-0"
    node_dir.mkdir()
    problem = {"id": 0, "kernel": "loop_level_reasoning/k/k", "language": "c", "task": "do it"}
    driver.run_agent(problem, 0, node_dir, ["http://127.0.0.1:8800"], 0, 1)
    return node_dir / "problem-0-worker-0"


def test_a_relaunch_keeps_the_transcript_of_the_crash_it_followed(monkeypatch, tmp_path):
    """The retry used to reopen claude.log with "w", so the crash it was relaunching -- and the
    note saying a relaunch had happened -- were deleted by the attempt that replaced them. The
    only surviving trace was crash_attempts= on the summary line, which says a crash happened and
    nothing about why."""
    monkeypatch.setenv("HPCAGENT_BENCH_SHARED_DIR", str(tmp_path / "shared"))
    driver = load_driver(monkeypatch, AGENT_CRASH_ATTEMPTS="3", AGENT_TIMEOUT_SECONDS="0")
    workdir = supervise(driver, monkeypatch, tmp_path, [1, 0])
    kept = workdir / "claude.attempt1.log"
    assert kept.exists(), "the crashed attempt's transcript was thrown away by its relaunch"
    first = kept.read_text(encoding="utf-8")
    assert "ATTEMPT 1" in first, f"the preserved file is not attempt 1's: {first[:120]!r}"
    assert "agent crashed (rc=1)" in first, "the preserved file does not say why it was relaunched"
    assert "ATTEMPT 2" in (workdir / "claude.log").read_text(encoding="utf-8"), (
        "claude.log must hold the run that actually finished"
    )


def test_every_attempt_shares_one_wall_clock(monkeypatch, tmp_path):
    """A relaunch that started its own AGENT_TIMEOUT_SECONDS made a crash cost another full budget,
    so three of them held one worker for three times the wall clock the arm was sized against."""
    monkeypatch.setenv("HPCAGENT_BENCH_SHARED_DIR", str(tmp_path / "shared"))
    driver = load_driver(monkeypatch, AGENT_CRASH_ATTEMPTS="3", AGENT_TIMEOUT_SECONDS="600")
    waits: list[float | None] = []
    popen = crashing_popen_class([1, 1, 0])
    original = popen.wait

    def recording_wait(self, timeout=None):
        waits.append(timeout)
        # Spend a measurable slice of the budget, so "what is left" is unambiguously smaller on
        # the next attempt rather than smaller by a clock tick.
        time.sleep(0.05)
        return original(self, timeout=timeout)

    popen.wait = recording_wait
    monkeypatch.setattr(driver.subprocess, "Popen", popen)
    monkeypatch.setattr(driver, "vllm_urls", lambda: ["http://127.0.0.1:8000"])
    monkeypatch.setattr(driver, "write_cost_record", lambda *a, **k: None)
    node_dir = tmp_path / "node-0"
    node_dir.mkdir()
    driver.run_agent(
        {"id": 0, "kernel": "loop_level_reasoning/k/k", "language": "c", "task": "x"},
        0,
        node_dir,
        ["http://127.0.0.1:8800"],
        0,
        1,
    )
    assert len(waits) == 3, f"expected three attempts, got {len(waits)}"
    assert all(w is not None for w in waits), "the wall-clock cap must stay armed across relaunches"
    # Strictly decreasing, not merely non-increasing: a relaunch handed a FRESH budget produces
    # [600, 600, 600], which is non-increasing too, and that is exactly the bug.
    assert waits[2] < waits[1] < waits[0], (
        f"each attempt must inherit what is LEFT of the budget, not a fresh one: {waits}"
    )
    assert all(w < 600 for w in waits), f"no attempt may be given the whole budget again: {waits}"
