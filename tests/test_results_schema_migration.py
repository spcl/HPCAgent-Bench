# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A results DB written before a column existed must still accept inserts.

``SQLModel.metadata.create_all`` is CREATE TABLE IF NOT EXISTS: it builds the table when absent and
does nothing whatsoever when present. Every deployment carries a results DB -- it is a persistent
artifact, one per rank of every past run -- so declaring a new field on ``Result`` without
reconciling the table breaks the next INSERT on every one of them, with a message
("table results has no column named X") that names the symptom and not the cause.
"""
import sqlite3

import pytest
from sqlmodel import Session, select

from hpcagent_bench.frameworks.schema import Result, results_engine

#: The table as it stood before ``flavor`` / ``build``: enough columns to insert a legacy row.
LEGACY_DDL = """
CREATE TABLE results (
    id INTEGER PRIMARY KEY,
    timestamp INTEGER NOT NULL,
    benchmark VARCHAR NOT NULL,
    domain VARCHAR,
    preset VARCHAR NOT NULL,
    framework VARCHAR NOT NULL,
    agent VARCHAR,
    validated BOOLEAN NOT NULL,
    time FLOAT NOT NULL,
    native_time FLOAT,
    datatype VARCHAR,
    variant VARCHAR,
    prompt_hash VARCHAR,
    execution VARCHAR NOT NULL
)
"""


@pytest.fixture(name="legacy_db")
def _legacy_db(tmp_path):
    """A results DB one column behind the model, with a row already in it."""
    path = tmp_path / "hpcagent_bench.db"
    with sqlite3.connect(path) as conn:
        conn.execute(LEGACY_DDL)
        conn.execute("INSERT INTO results (timestamp, benchmark, preset, framework, validated, time, execution) "
                     "VALUES (1, 'gemm', 'S', 'numpy', 1, 1.5, 'native')")
    return str(path)


def test_a_legacy_db_accepts_a_row_carrying_the_new_column(legacy_db):
    engine = results_engine(legacy_db)
    with Session(engine) as session:
        session.add(
            Result(timestamp=2,
                   benchmark="gemm",
                   preset="S",
                   framework="dace_cpu",
                   flavor="parallel",
                   validated=True,
                   time=0.5,
                   build="extended"))
        session.commit()
    with Session(engine) as session:
        rows = {r.framework: (r.flavor, r.build) for r in session.exec(select(Result))}
    # The pre-existing row reads back as NULL, which is what "this run predates the axis" means --
    # it must not be backfilled with a guess that would make it comparable to a labelled row.
    assert rows == {"numpy": (None, None), "dace_cpu": ("parallel", "extended")}


def test_reconciling_twice_is_a_no_op(legacy_db):
    results_engine(legacy_db)
    results_engine(legacy_db)  # ADD COLUMN is not idempotent in SQLite; the guard must be
    with sqlite3.connect(legacy_db) as conn:
        names = [row[1] for row in conn.execute("PRAGMA table_info(results)")]
    assert names.count("flavor") == 1
    assert names.count("build") == 1


def test_a_fresh_db_gets_the_whole_model(tmp_path):
    engine = results_engine(str(tmp_path / "fresh.db"))
    with engine.connect() as conn:
        names = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(results)")}
    assert set(Result.__table__.columns.keys()) == names


def test_the_plot_loader_folds_flavor_and_build_back_into_one_series(tmp_path, monkeypatch):
    """Stored apart, plotted together. Without the fold, dace_cpu's three optimizers -- and the
    same optimizer measured on two DaCe trees -- average into one silently wrong line."""
    import pandas as pd

    from hpcagent_bench import plotting

    path = str(tmp_path / "hpcagent_bench.db")
    engine = results_engine(path)
    rows = [("numpy", None, None), ("dace_cpu", "parallel", "main"), ("dace_cpu", "parallel", "extended"),
            ("dace_cpu", "canonicalize", "extended"), ("dace_cpu", None, None)]
    with Session(engine) as session:
        for framework, flavor, build in rows:
            session.add(
                Result(timestamp=1,
                       benchmark="gemm",
                       domain="LinAlg",
                       preset="S",
                       framework=framework,
                       flavor=flavor,
                       build=build,
                       validated=True,
                       time=1.0,
                       datatype="float64"))
        session.commit()

    monkeypatch.setattr(plotting.recording, "ensure_aggregated", lambda p: p)
    data = plotting.load_results(path, preset="S")
    assert isinstance(data, pd.DataFrame)
    assert sorted(data["framework"]) == [
        "dace_cpu",
        "dace_cpu/canonicalize/extended",
        "dace_cpu/parallel/extended",
        "dace_cpu/parallel/main",
        "numpy",
    ]
