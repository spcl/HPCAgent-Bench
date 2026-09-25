# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""An enum-typed value leaves the process as its ``.value`` (a DB row, a CLI argument) and reads back
to the same member, so the on-disk and command-line spellings are the enum values themselves."""

import contextlib
import sqlite3
from collections.abc import Iterator

import pytest

from hpcagent_bench import config, harbor
from hpcagent_bench.cli import Execution, build_parser
from hpcagent_bench.harness import recording
from hpcagent_bench.harness.task import RecordDevice


@pytest.fixture
def judge_db() -> Iterator[sqlite3.Connection]:
    """An in-memory judge DB with every recording table."""
    with contextlib.closing(sqlite3.connect(":memory:")) as conn:
        for ddl in recording.TABLES.values():
            conn.execute(ddl)
        yield conn


@pytest.mark.parametrize("device", list(RecordDevice), ids=lambda device: device.value)
def test_the_recorded_device_reads_back_to_its_member(
    device: RecordDevice, judge_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``runs.device`` holds the member's value, and parsing it gives the member back."""
    monkeypatch.setenv("HPCAGENT_BENCH_RECORD_DEVICE", device.value)
    config.reload()
    recording.upsert_run(judge_db, "run-1", 1)
    (stored,) = judge_db.execute("SELECT device FROM runs WHERE run_id = 'run-1'").fetchone()
    assert stored == device.value
    assert RecordDevice(stored) is device


@pytest.mark.parametrize("execution", list(Execution), ids=lambda execution: execution.value)
def test_the_execution_argument_is_the_member_value(execution: Execution) -> None:
    """``hpcagent-bench agent --execution <value>`` accepts every member's value and nothing else."""
    args = build_parser().parse_args(["agent", "stub", "--execution", execution.value])
    assert Execution(args.execution) is execution
    with pytest.raises(SystemExit):
        build_parser().parse_args(["agent", "stub", "--execution", repr(execution)])


@pytest.mark.parametrize(("flag", "kind"), [("--group", harbor.Group), ("--layout", harbor.Layout)])
def test_harbor_generate_parses_group_and_layout_values(flag: str, kind: type[harbor.Group | harbor.Layout]) -> None:
    """Each member's value is accepted and converts back to the member."""
    for member in kind:
        args = harbor.build_parser().parse_args(["generate", "--out", "x", flag, member.value])
        assert kind(getattr(args, flag.removeprefix("--"))) is member
