# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``disqualifications``, a table no code ever wrote, left the schema in user_version 36 (results_db.DROPPED): a new
file has no such table, an older file with it empty drops it when opened for writing and merges as any older file
does, and a file that somehow holds a row of it is refused, unchanged, rather than lose the row."""

import contextlib
import pathlib
import shutil
import sqlite3

import pytest

from hpcagent_bench.harness import results_db
from tests import results_seed

#: The version that dropped the table, and the one before it.
DROPPED_IN = 36
BEFORE = DROPPED_IN - 1


def tables(path: pathlib.Path) -> set[str]:
    with contextlib.closing(sqlite3.connect(path)) as conn:
        return results_db.table_names(conn)


def version(path: pathlib.Path) -> int:
    with contextlib.closing(sqlite3.connect(path)) as conn:
        return results_db.schema_version(conn)


def old_file(path: pathlib.Path, withdrawn: bool = False) -> pathlib.Path:
    """A user_version 35 file of one graded /score, with an empty ``disqualifications`` table, or one row in it."""
    grade_id = results_seed.score(path, "setup.n0.p1.w1", "gemm", 1, 4.0, source="int x;")
    results_seed.as_version(path, BEFORE)
    if withdrawn:
        with contextlib.closing(sqlite3.connect(path)) as conn:
            conn.execute("INSERT INTO disqualifications VALUES (?, 'audit', 1)", (grade_id,))
            conn.commit()
    return path


def grade_count(path: pathlib.Path) -> int:
    with results_db.reading(path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM grades").fetchone()[0])


def test_an_older_file_holds_the_empty_table_and_reads_as_it_is(tmp_path: pathlib.Path) -> None:
    """Read-only, an older file is never upgraded: the table stays, and the grades read."""
    path = old_file(tmp_path / "results.db")

    assert grade_count(path) == 1
    assert "disqualifications" in tables(path) and version(path) == BEFORE


def test_opening_an_older_file_for_writing_drops_the_empty_table(tmp_path: pathlib.Path) -> None:
    path = old_file(tmp_path / "results.db")

    with contextlib.closing(results_db.open_db(path)) as conn:
        assert results_db.schema_version(conn) == results_db.SCHEMA_VERSION

    assert "disqualifications" not in tables(path)
    assert grade_count(path) == 1


def test_an_older_source_with_the_empty_table_merges_into_a_new_file(tmp_path: pathlib.Path) -> None:
    source, merged = old_file(tmp_path / "shard.db"), tmp_path / "merged.db"

    counts = results_db.merge(merged, [source])

    assert counts["grades"] == 1 and "disqualifications" not in counts
    assert "disqualifications" not in tables(merged)
    assert "disqualifications" in tables(source) and version(source) == BEFORE  # a source is read, never upgraded


def test_a_new_source_merges_into_an_older_file(tmp_path: pathlib.Path) -> None:
    dest, source = old_file(tmp_path / "results.db"), tmp_path / "shard.db"
    results_seed.score(source, "setup.n0.p1.w2", "gemm", 2, 3.0)

    results_db.merge(dest, [source])

    assert grade_count(dest) == 2
    assert "disqualifications" not in tables(dest) and version(dest) == results_db.SCHEMA_VERSION


def test_an_older_file_with_a_withdrawn_grade_is_refused_and_left_as_it_was(tmp_path: pathlib.Path) -> None:
    path = old_file(tmp_path / "results.db", withdrawn=True)
    before = path.read_bytes()

    with pytest.raises(results_db.SchemaVersionError, match="disqualifications"):
        results_db.open_db(path)

    assert version(path) == BEFORE and "disqualifications" in tables(path)
    with contextlib.closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM disqualifications").fetchone()[0] == 1
    assert grade_count(path) == 1 and len(path.read_bytes()) == len(before)


def test_a_source_with_a_withdrawn_grade_is_refused_and_the_destination_unchanged(tmp_path: pathlib.Path) -> None:
    source, merged = old_file(tmp_path / "shard.db", withdrawn=True), tmp_path / "merged.db"
    results_seed.score(merged, "setup.n0.p1.w2", "gemm", 2, 3.0)
    kept = tmp_path / "kept.db"
    shutil.copy(merged, kept)

    with pytest.raises(results_db.SchemaVersionError, match="disqualifications"):
        results_db.merge(merged, [source])

    assert results_db.held_rows(merged) == results_db.held_rows(kept)
