# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Restarting an agent whose MCP server did not connect, and capping how many start at once.

A failed MCP server is not a crash. The agent keeps its built-in tools, loses score/submit/task
entirely, burns its whole budget and exits reporting success -- in one qwen run, one
such agent ran 36 minutes over 54 turns and called a `Submit` tool that does not exist. The harness
records rc=0 and the data point is simply gone, so the driver has to notice and relaunch.
"""

import io
import json
import pathlib
import threading
import time
import types
from typing import Self

import pytest

from tests.fresh_module import fresh
from tests.problem_facts import problem as problem_line


def load_driver(monkeypatch: pytest.MonkeyPatch, **env: str) -> types.ModuleType:
    # The fake processes have no pid to cap.
    monkeypatch.setenv("AGENT_MEMORY_CAP_GB", "0")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    driver = fresh("agent_driver")
    # The tool registry is a real ``mcp_server.py --describe`` subprocess: ask it before a test fakes Popen.
    driver.tool_registry()
    return driver


def init_line(status: str) -> str:
    return json.dumps({"subtype": "init", "mcp_servers": [{"name": "hpcagent-bench", "status": status}]}) + "\n"


def fake_popen_class(statuses: list[str], seen: list[int] | None = None) -> type:
    """Popen stand-in that writes one init event per spawn, taking each status in turn."""

    class FakePopen:
        spawned = 0
        live = 0

        def __init__(
            self,
            command: list[str],
            cwd: pathlib.Path | None = None,
            env: dict[str, str] | None = None,
            stdout: io.TextIOWrapper | None = None,
            stderr: int | None = None,
        ) -> None:
            cls = type(self)
            cls.spawned += 1
            cls.live += 1
            if seen is not None:
                seen.append(cls.live)
            self.returncode = None
            self.pid = 0
            status = statuses[min(cls.spawned - 1, len(statuses) - 1)]
            stdout.write(init_line(status))
            stdout.flush()

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: int | None = None) -> int:
            self.returncode = self.returncode or 0
            return self.returncode

        def terminate(self) -> None:
            type(self).live -= 1
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    return FakePopen


#: The CPU share start_agent pins the child to. Empty is its own "leave the mask alone" path,
#: which is what these tests want: they drive a fake Popen with no real pid, and the pinning
#: itself is covered against a live process in test_agent_driver_affinity.py.
_UNPINNED: list[int] = []


def start(
    driver: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    statuses: list[str],
    seen: list[int] | None = None,
) -> tuple[object, int, pathlib.Path]:
    monkeypatch.setattr(driver.subprocess, "Popen", fake_popen_class(statuses, seen))
    log_path = tmp_path / "claude.log"
    with log_path.open("w", encoding="utf-8") as log:
        process, attempts = driver.start_agent(["claude"], tmp_path, {}, log, log_path, _UNPINNED, 0)
    return process, attempts, log_path


def test_a_connected_server_is_not_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    driver = load_driver(monkeypatch)
    _, attempts, _ = start(driver, monkeypatch, tmp_path, ["connected"])
    assert attempts == 1


def test_a_failed_server_is_relaunched_until_it_connects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    driver = load_driver(monkeypatch)
    _, attempts, log_path = start(driver, monkeypatch, tmp_path, ["failed", "failed", "connected"])
    assert attempts == 3
    assert driver.mcp_failed(log_path) is False, "the surviving transcript is the connected attempt"


def test_the_retries_are_bounded_and_an_agent_without_its_tools_never_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """An agent without the judge tools finishes rc 0 and grades nothing it was meant to, so spent
    attempts end it loudly, and the log says why."""
    driver = load_driver(monkeypatch, AGENT_MCP_ATTEMPTS="2")
    with pytest.raises(driver.McpUnavailable, match="after 2 attempt"):
        start(driver, monkeypatch, tmp_path, ["failed"])
    assert "MCP still not connected" in (tmp_path / "claude.log").read_text(encoding="utf-8")


def test_a_retry_leaves_no_half_transcript(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Downstream readers take the LAST result event; a dead attempt's output must not linger."""
    driver = load_driver(monkeypatch)
    _, _, log_path = start(driver, monkeypatch, tmp_path, ["failed", "connected"])
    assert log_path.read_text(encoding="utf-8").count('"subtype": "init"') == 1


def test_mcp_failed_distinguishes_not_yet_from_connected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    driver = load_driver(monkeypatch)
    log_path = tmp_path / "claude.log"
    log_path.write_text("", encoding="utf-8")
    assert driver.mcp_failed(log_path) is None, "no init event yet is not the same as a good one"
    log_path.write_text(init_line("connected"), encoding="utf-8")
    assert driver.mcp_failed(log_path) is False
    log_path.write_text(init_line("failed"), encoding="utf-8")
    assert driver.mcp_failed(log_path) is True


def test_only_so_many_agents_start_at_once(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """The whole point: 120 pool threads must not put 120 python3 servers on one node at once.

    The stand-in blocks inside the spawn, which is where the real one sits too -- the gate is held
    across the spawn AND the wait for the init event, so a slow startup holds its slot.
    """
    driver = load_driver(monkeypatch, AGENT_START_CONCURRENCY="3")
    release = threading.Event()
    spawned: list[int] = []
    lock = threading.Lock()

    class BlockingPopen:
        def __init__(
            self,
            command: list[str],
            cwd: pathlib.Path | None = None,
            env: dict[str, str] | None = None,
            stdout: io.TextIOWrapper | None = None,
            stderr: int | None = None,
        ) -> None:
            with lock:
                spawned.append(1)
            self.returncode = None
            self.pid = 0
            release.wait(timeout=30)
            stdout.write(init_line("connected"))
            stdout.flush()

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            self.returncode = 0
            return 0

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(driver.subprocess, "Popen", BlockingPopen)
    threads = []
    for index in range(12):
        workdir = tmp_path / str(index)
        workdir.mkdir()

        def run(workdir: pathlib.Path = workdir) -> None:
            log_path = workdir / "claude.log"
            with log_path.open("w", encoding="utf-8") as log:
                driver.start_agent(["claude"], workdir, {}, log, log_path, _UNPINNED, 0)

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


def test_a_crash_is_a_fault_but_a_budget_is_not(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Many agents end on the wall clock and nothing else. A timed-out agent
    spent what it was given and keeps every submission it made; relaunching it would hand it a
    second budget its peers never had."""
    driver = load_driver(monkeypatch)
    log = tmp_path / "claude.log"
    log.write_text("", encoding="utf-8")
    assert driver.closing_crashed(1, driver.transcript_closing(log)) is True, (
        "a bare nonzero exit with no result event is a fault"
    )
    for code in (0, driver.RC_TIMEOUT, driver.RC_TOKEN_BUDGET, driver.RC_CONTEXT):
        assert driver.closing_crashed(code, driver.transcript_closing(log)) is False, (
            f"rc={code} is a budget, not a fault"
        )


def test_a_reported_result_is_the_cli_verdict_not_a_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A nonzero exit AFTER the CLI wrote its result is that run's answer; relaunching overwrites it."""
    driver = load_driver(monkeypatch)
    log = tmp_path / "claude.log"
    log.write_text(json.dumps({"type": "result", "subtype": "error_during_execution"}) + "\n", encoding="utf-8")
    assert driver.closing_crashed(1, driver.transcript_closing(log)) is False


def test_crash_retries_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_CRASH_ATTEMPTS", raising=False)
    driver = load_driver(monkeypatch)
    assert 2 <= driver.AGENT_CRASH_ATTEMPTS <= 5


def crashing_popen_class(exit_codes: list[int]) -> type:
    """Popen stand-in whose Nth spawn exits with ``exit_codes[N-1]`` and marks its own transcript.

    Each spawn writes a connected init event -- the MCP path is not what these exercise -- plus a
    line naming its attempt, which is how the test tells the preserved transcripts apart.
    """

    class FakePopen:
        spawned = 0

        def __init__(
            self,
            command: list[str],
            cwd: pathlib.Path | None = None,
            env: dict[str, str] | None = None,
            stdout: io.TextIOWrapper | int | None = None,
            stderr: int | None = None,
            text: bool | None = None,
            **kwargs: object,
        ) -> None:
            self.args = command
            # Recorded so a test can assert on exactly what production passed to Popen.
            self.kwargs = {"cwd": cwd, "env": env, "stdout": stdout, "stderr": stderr, "text": text, **kwargs}
            self.returncode = None
            self.pid = 0
            self.captured = ""
            if isinstance(stdout, int) or stdout is None:
                # claude_supports_flag's --help probe goes through subprocess.run(capture_output=True,
                # text=True), which pipes stdout instead of handing a real log file. It is not one of
                # the agent spawns this fake numbers, so it must not shift the ATTEMPT count.
                self.attempt = None
                return
            cls = type(self)
            cls.spawned += 1
            self.attempt = cls.spawned
            stdout.write(init_line("connected"))
            stdout.write(f"ATTEMPT {self.attempt}\n")
            stdout.flush()

        def communicate(self, input: str | None = None, timeout: float | None = None) -> tuple[str, str]:  # noqa: A002 -- the overridden method's own signature
            return self.captured, ""

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exc_info: object) -> bool:
            return False

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            self.returncode = exit_codes[min(self.attempt - 1, len(exit_codes) - 1)]
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    return FakePopen


def supervise(
    driver: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, exit_codes: list[int]
) -> pathlib.Path:
    """Drive run_agent over a fake CLI, returning the worker directory it wrote into."""
    monkeypatch.setattr(driver.subprocess, "Popen", crashing_popen_class(exit_codes))
    monkeypatch.setattr(driver, "vllm_urls", lambda: ["http://127.0.0.1:8000"])
    monkeypatch.setattr(driver, "write_cost_record", lambda *a, **k: None)
    node_dir = tmp_path / "node-0"
    node_dir.mkdir()
    driver.run_agent(problem_line(0, "gemm", "Optimize gemm in c."), 0, node_dir, ["http://127.0.0.1:8800"], 0, 1)
    return node_dir / "problem-0-worker-0"


def test_a_relaunch_keeps_the_transcript_of_the_crash_it_followed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Reopening claude.log with "w" would delete the crash being relaunched -- and the note saying
    a relaunch happened -- leaving only crash_attempts= on the summary line, which says a crash
    happened and nothing about why."""
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


def test_spent_mcp_attempts_end_that_agent_with_a_record_not_the_whole_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """McpUnavailable used to propagate out of run_agent into main(), whose ``future.result()``
    re-raised it: the node died with a traceback after waiting for the others, skipped its summary
    and exit policy, and the failing worker left no tokens.json and no ledger line."""
    monkeypatch.setenv("HPCAGENT_BENCH_SHARED_DIR", str(tmp_path / "shared"))
    driver = load_driver(monkeypatch, AGENT_MCP_ATTEMPTS="2", AGENT_TIMEOUT_SECONDS="0")
    # Rendered before Popen is faked: the facts come from a real make_problems subprocess.
    problem = problem_line(0, "gemm", "Optimize gemm in c.")
    monkeypatch.setattr(driver.subprocess, "Popen", fake_popen_class(["failed"]))
    monkeypatch.setattr(driver, "vllm_urls", lambda: ["http://127.0.0.1:8000"])
    monkeypatch.setattr(driver, "claude_supports_flag", lambda binary, flag: True)
    monkeypatch.setattr(driver, "promote_at_agent_exit", lambda *args, **kwargs: "")
    monkeypatch.setattr(driver, "agent_cpus", lambda worker_index, agents: [])
    node_dir = tmp_path / "node-0"
    node_dir.mkdir()
    rc = driver.run_agent(problem, 0, node_dir, ["http://127.0.0.1:8800"], 0, 1)
    assert rc == driver.RC_MCP_UNAVAILABLE
    assert rc not in driver.CLEAN_ENDS, "an agent that never ran is a failure, not an end"
    workdir = node_dir / "problem-0-worker-0"
    record = json.loads((workdir / "tokens.json").read_text(encoding="utf-8"))
    assert record["returncode"] == driver.RC_MCP_UNAVAILABLE
    ledger = [json.loads(line) for line in (workdir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [line["returncode"] for line in ledger] == [driver.RC_MCP_UNAVAILABLE]
    assert "MCP still not connected" in (workdir / "claude.log").read_text(encoding="utf-8")


def test_an_agent_thread_that_raises_costs_the_node_one_rc_not_its_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any other exception out of one agent thread is that agent's failure: main() records
    RC_AGENT_FAULT for it and the node's exit policy still runs over every agent."""
    import concurrent.futures

    driver = load_driver(monkeypatch)

    def boom() -> int:
        raise RuntimeError("one agent's fault")

    def clean() -> int:
        return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(boom): {"id": 3}, executor.submit(clean): {"id": 4}}
        rcs = sorted(driver.agent_returncode(future, problem) for future, problem in futures.items())
    assert rcs == sorted([driver.RC_AGENT_FAULT, 0])
    assert driver.node_exit_status(rcs) == 0, "one failed agent must not fail the node"


def test_a_submission_the_watcher_missed_is_not_relaunched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """submit.py writes the marker, then the CLI dies (rc 1, no closing event) inside one watcher
    poll. The attempt read as a crash and was relaunched; the kept marker killed the relaunch at its
    first poll, and the relaunch's start became final_attempt_start_ms -- the X7 cut that drops
    every judge row before it, the episode's only real submission included."""
    monkeypatch.setenv("HPCAGENT_BENCH_SHARED_DIR", str(tmp_path / "shared"))
    driver = load_driver(
        monkeypatch, AGENT_CRASH_ATTEMPTS="3", AGENT_TIMEOUT_SECONDS="0", AGENT_SUBMISSION_MODE="single"
    )
    problem = problem_line(0, "gemm", "Optimize gemm in c.")
    node_dir = tmp_path / "node-0"
    node_dir.mkdir()
    marker = node_dir / "problem-0-worker-0" / driver.SUBMISSION_MARKER
    base = crashing_popen_class([1, 0])

    class SubmitThenDie(base):
        def wait(self, timeout: float | None = None) -> int:
            # The submission lands as the process exits, after the watcher's first look.
            if self.attempt == 1:
                marker.write_text('{"correct": true}\n', encoding="utf-8")
            return super().wait(timeout)

    monkeypatch.setattr(driver.subprocess, "Popen", SubmitThenDie)
    monkeypatch.setattr(driver, "vllm_urls", lambda: ["http://127.0.0.1:8000"])
    monkeypatch.setattr(driver, "agent_cpus", lambda worker_index, agents: [])
    monkeypatch.setattr(driver, "promote_at_agent_exit", lambda *args, **kwargs: "")
    monkeypatch.setattr(driver, "TOKEN_POLL_SECONDS", 0.5)
    rc = driver.run_agent(problem, 0, node_dir, ["http://127.0.0.1:8800"], 0, 1)
    workdir = node_dir / "problem-0-worker-0"
    ledger = [json.loads(line) for line in (workdir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(ledger) == 1, f"the submitted attempt was relaunched: {ledger}"
    assert rc == driver.RC_SUBMITTED
    record = json.loads((workdir / "tokens.json").read_text(encoding="utf-8"))
    assert record["final_attempt_start_ms"] == ledger[0]["start_ms"]
