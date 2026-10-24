# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""An enum-typed value leaves the process as its ``.value`` (a DB row, a CLI argument) and reads back
to the same member, so the on-disk and command-line spellings are the enum values themselves."""

import contextlib
import pathlib
import sqlite3
from collections.abc import Iterator

import pytest

from hpcagent_bench import config
from hpcagent_bench.export import harbor
from hpcagent_bench.categories import RecordDevice
from hpcagent_bench.cli import build_parser
from hpcagent_bench.config import Execution
from hpcagent_bench.harness import recording


@pytest.fixture
def judge_db(tmp_path: pathlib.Path) -> Iterator[sqlite3.Connection]:
    """A fresh results DB."""
    with contextlib.closing(recording.connect(str(tmp_path / "r.db"))) as conn:
        yield conn


@pytest.mark.parametrize("device", list(RecordDevice), ids=lambda device: device.text)
def test_the_recorded_device_reads_back_to_its_member(
    device: RecordDevice, judge_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``setups.device`` holds the member's text, and parsing it gives the member back."""
    monkeypatch.setenv("HPCAGENT_BENCH_RECORD_DEVICE", device.text)
    config.reload()
    recording.open_episode(judge_db, "setup.n0.p0.w0")
    (stored,) = judge_db.execute("SELECT device FROM setups WHERE setup = 'setup'").fetchone()
    assert stored == device.text
    assert RecordDevice.parse(stored) is device


@pytest.mark.parametrize("execution", list(Execution), ids=lambda execution: execution.text)
def test_the_execution_argument_is_the_member_text(execution: Execution) -> None:
    """``hpcagent-bench agent --execution <text>`` accepts every member's spelling and nothing else."""
    assert Execution.texts() == ("native", "container", "harbor")
    args = build_parser().parse_args(["agent", "stub", "--execution", execution.text])
    assert args.execution is execution
    with pytest.raises(SystemExit):
        build_parser().parse_args(["agent", "stub", "--execution", repr(execution)])


@pytest.mark.parametrize(("flag", "kind"), [("--group", harbor.Group), ("--layout", harbor.Layout)])
def test_harbor_generate_parses_group_and_layout_spellings(flag: str, kind: type[harbor.Group | harbor.Layout]) -> None:
    """Each member's spelling is accepted and parses back to the member."""
    for member in kind:
        args = harbor.build_parser().parse_args(["generate", "--out", "x", flag, member.text])
        assert kind.parse(vars(args)[flag.removeprefix("--")]) is member
