# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A plot leg that finds no data must say which leg actually failed.

``sqlite3.connect`` CREATES an absent file. So when a run records nothing -- the child died, the
selector matched no kernel, the writer used a different path -- the reader opens a valid, empty
database and only discovers the problem one query later, as ``no such table: results``, naming
neither the file it opened nor the shards it looked beside. That message sends the reader to the
plotting code, which is the one place that is working.

The sharding rule is what makes a good message possible: a run records into its OWN shard
(``hpcagent_bench<N>.db``, :func:`recording.db_path`) and the base is the cache rebuilt from those
(:func:`recording.ensure_aggregated`). So "no table, and no shard beside it" is not ambiguous --
it means the run leg wrote nothing.
"""

import pathlib
import sqlite3


from hpcagent_bench.harness import recording


def test_table_exists_is_false_for_a_file_connect_would_create(tmp_path: pathlib.Path) -> None:
    """The primitive the guard rests on. ``sqlite3.connect`` would happily make this file."""
    missing = tmp_path / "hpcagent_bench.db"
    assert not missing.exists()
    assert not recording.table_exists(str(missing), "results")


def test_table_exists_is_false_for_an_empty_database(tmp_path: pathlib.Path) -> None:
    """And an empty DB is not the same as an absent one: a previous reader may already have
    created it, which is exactly how the confusing case arises."""
    empty = tmp_path / "hpcagent_bench.db"
    sqlite3.connect(empty).close()
    assert empty.exists(), "connect() did not create the file, so this test no longer covers its case"
    assert not recording.table_exists(str(empty), "results")
    assert not recording.table_exists(str(tmp_path / "absent.db"), "results")
    assert not (tmp_path / "absent.db").exists(), "table_exists created the file it looked for"


def write_row(shard: pathlib.Path, framework: str, build: str, time: float) -> None:
    """One validated sample into ``shard``, stamped with ``build``."""
    from sqlmodel import Session

    from hpcagent_bench.frameworks.schema import Result, results_engine

    with Session(results_engine(str(shard))) as session:
        session.add(
            Result(
                timestamp=0,
                kernel="gemm",
                domain="LinAlg",
                preset="S",
                framework=framework,
                build=build,
                agent=None,
                validated=True,
                cpu="test-cpu",
                time=time,
                native_time=None,
                datatype="float64",
                variant=None,
                execution="native",
            )
        )
        session.commit()
