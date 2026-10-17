# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``results_engine`` builds the framework-sweep ``results`` table and holds no connection open.

Every rank of a sweep writes through its own shard file, and the first writes of four ranks race the
CREATE TABLE of one file; the plot loader reads the rows back, folding ``flavor`` and ``build`` into the
series name and partitioning by machine.
"""

import multiprocessing
import multiprocessing.queues
import multiprocessing.synchronize
import pathlib
import sqlite3

import pytest
from sqlmodel import Session, select

from hpcagent_bench import osinfo
from hpcagent_bench.frameworks.schema import Result, results_engine
from tests.sqlite_closing import connect


def create_schema_race_worker(
    path: str,
    start: multiprocessing.synchronize.Barrier,
    outcome: multiprocessing.queues.Queue[str],
) -> None:
    """One rank's ``results_engine(path)`` call, synchronized to start with its siblings so the
    CREATE TABLE race is real OS-level file contention, not a simulated ordering."""
    start.wait()
    try:
        results_engine(path)
        outcome.put("ok")
    except Exception as exc:  # noqa: BLE001 -- ANY exception here is the race under test
        outcome.put(f"{type(exc).__name__}: {exc}")


@pytest.mark.skipif(not osinfo.IS_LINUX, reason="fork start method is Linux-only")
def test_four_ranks_racing_the_first_write_to_one_shard_do_not_crash(tmp_path: pathlib.Path) -> None:
    """Every column's ranks write results through ONE shard file each (recording.db_path, sharded
    by SLURM_PROCID); ``results_engine``'s ``create_all`` reads ``sqlite_master`` and then issues
    CREATE TABLE -- check-then-act, not atomic -- so the first kernel a fresh rank writes (cholesky)
    races that first CREATE against a sibling rank's, and the loser must not raise
    ``sqlalchemy.exc.OperationalError: ... table results already exists``. Real forked processes
    race ONE not-yet-existing file here, matching the production shape exactly."""
    path = str(tmp_path / "hpcagent_bench0.db")
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(4)
    outcome: multiprocessing.queues.Queue[str] = ctx.Queue()
    workers = [ctx.Process(target=create_schema_race_worker, args=(path, barrier, outcome)) for rank in range(4)]
    for worker in workers:
        worker.start()
    outcomes = [outcome.get(timeout=30) for worker in workers]
    for worker in workers:
        worker.join(timeout=30)
    assert outcomes == ["ok"] * 4, outcomes
    with connect(path) as conn:
        names = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
    assert names == set(Result.__table__.columns.keys())


def test_a_fresh_db_gets_the_whole_model(tmp_path: pathlib.Path) -> None:
    engine = results_engine(str(tmp_path / "fresh.db"))
    with engine.connect() as conn:
        names = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(results)")}
    assert set(Result.__table__.columns.keys()) == names


def test_a_results_engine_holds_no_open_connection_once_its_session_closes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine outlives its Session here and still owns no open sqlite handle: a pooled one would be
    garbage-collected later as a ResourceWarning (Python 3.13+)."""
    opened: list[sqlite3.Connection] = []
    real_connect = sqlite3.dbapi2.connect  # the entry SQLAlchemy's pysqlite dialect opens its connections through

    def recording_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        conn = real_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3.dbapi2, "connect", recording_connect)
    engine = results_engine(str(tmp_path / "r.db"))
    with Session(engine) as session:
        assert session.exec(select(Result)).all() == []

    assert opened
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")
