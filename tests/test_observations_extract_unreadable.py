# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A results DB the extractor cannot read is named on stderr, never dropped in silence.

A run root holds every rank's shard and job DB; one the extraction skips without a word reads as if
that rank had graded nothing. A DB of another schema version and one that fails mid-read are both such
skips. A SQLite file that holds no results tables at all is not a results DB and stays silent.
"""

import pathlib

import pytest

from hpcagent_bench.results import extract
from hpcagent_bench.harness import results_db
from tests.sqlite_closing import connect

#: A schema version no reader opens: neither the current one nor a legacy one.
FOREIGN_VERSION = results_db.SCHEMA_VERSION + 1


def results_file(path: pathlib.Path) -> pathlib.Path:
    """An empty results DB of the current schema at ``path``."""
    results_db.open_db(path).close()
    return path


def test_a_results_db_of_another_schema_version_is_named_on_stderr(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = results_file(tmp_path / "results.db")
    with connect(db) as conn:
        conn.execute(f"PRAGMA user_version = {FOREIGN_VERSION}")
    assert not extract.results_database(db)
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert str(db) in err


def test_a_sqlite_file_with_no_results_tables_is_skipped_in_silence(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "other.db"
    with connect(db) as conn:
        conn.execute("CREATE TABLE unrelated (x INTEGER)")
    assert not extract.results_database(db)
    assert capsys.readouterr().err == ""


def test_a_db_that_fails_mid_read_is_named_and_contributes_no_row(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file stamped with the current version but missing the grade tables passes the schema check and
    fails on the first query: it is named on stderr, and no setup-less row reaches the observations."""
    db = tmp_path / "results.db"
    with connect(db) as conn:
        conn.execute(f"PRAGMA user_version = {results_db.SCHEMA_VERSION}")
        conn.execute("CREATE TABLE unrelated (x INTEGER)")
    database = extract.Database(db, "root", tmp_path, "1")
    assert extract.read_db(database, "", frozenset(), 0).observations[0]["row_kind"].startswith(extract.UNREADABLE)
    observations, sources = extract.read_all([database], extract.Options(runs=(), benchmarks=tmp_path, threads=1))
    assert (observations, sources) == ([], [])
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert str(db) in err
