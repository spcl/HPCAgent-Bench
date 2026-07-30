# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Typed SQLModel schema for the framework-benchmark ``results`` table: the single Result model derives
both the DDL (``create_all``) and row inserts, replacing the old hand-written CREATE TABLE/INSERT pair."""
from typing import Optional

from sqlmodel import Field, SQLModel, create_engine


class Result(SQLModel, table=True):
    """One (framework, flavor, build, benchmark, preset, datatype, variant) runtime sample."""

    __tablename__ = "results"

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: int  # epoch seconds; groups the rows of one run
    benchmark: str  # kernel short_name
    domain: Optional[str] = None  # taxonomy label; used as a heatmap grouping key
    preset: str  # S | M | L | XL
    framework: str  # numpy | dace_cpu | jax | ... -- the backend, WITHOUT its flavor suffix
    # Which optimizer inside the framework produced this row: dace_cpu's `parallel` / `autoopt` /
    # `canonicalize`. NULL == the framework's own default path, which for a searching column means
    # "the fastest of its pipelines" -- so `WHERE framework='dace_cpu'` gathers every DaCe CPU row
    # and `flavor` is what tells them apart. One flat name on the CLI (`--framework
    # dace_cpu_parallel`), two columns here; framework.split_flavor is the one place that maps
    # between them.
    flavor: Optional[str] = None
    agent: Optional[str] = None  # who produced the optimization (None == direct framework run)
    validated: bool  # output matched the NumPy oracle
    time: float  # host-measured runtime, milliseconds
    native_time: Optional[float] = None  # framework-internal runtime, ms (None if no native timer)
    datatype: Optional[str] = None  # float32 | float64 | ... (None == legacy float64)
    variant: Optional[str] = None  # sparse storage/distribution axis (None == dense)
    # WHICH BUILD ran it -- upstream `main` vs the fork's `extended`, a different BLAS, a different
    # image. A separate axis from `flavor` because you cannot ASK for it: the flavor is a column you
    # name on the command line, the build is whatever PYTHONPATH resolved to, so it is stamped by
    # the launcher (HPCAGENT_BENCH_RECORD_BUILD) exactly like `execution`. NULL == single-build run.
    # Without it, the same pipeline measured on two DaCe trees is two indistinguishable rows.
    build: Optional[str] = None
    prompt_hash: Optional[str] = None  # -> the content-addressed prompt store (None if no prompt)
    execution: str = "native"  # native (no container) | container -- where the runtime was measured


def add_missing_columns(engine) -> None:
    """Add to an EXISTING ``results`` table any nullable column :class:`Result` has grown since.

    ``create_all`` is CREATE TABLE IF NOT EXISTS: it builds the table when absent and does nothing
    whatsoever when it is present. A results DB is a persistent artifact -- every machine and every
    rank of every past run has one -- so on all of them a newly declared column would exist in the
    model and not in the table, and the very next INSERT would fail with "table results has no
    column named X". Reconciling here rather than at each call site is what keeps the model the
    single source of the schema.

    ADD COLUMN only: additive, no table rewrite, cannot lose a row, and a legacy row reads back with
    NULL for the new column -- which is exactly what "this run predates the axis" means. A missing
    NOT NULL column is NOT invented: there is no honest value to backfill, so it is raised."""
    table = Result.__table__
    with engine.connect() as conn:
        present = {row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table.name})")}
        if not present:
            return  # create_all just built it from the model; nothing to reconcile
        for name, column in table.columns.items():
            if name in present:
                continue
            if not column.nullable:
                raise RuntimeError(f"results table lacks the NOT NULL column {name!r} and no value can be "
                                   f"backfilled for existing rows; migrate {engine.url.database} by hand")
            sql_type = column.type.compile(engine.dialect)
            conn.exec_driver_sql(f"ALTER TABLE {table.name} ADD COLUMN {name} {sql_type}")
        conn.commit()


def results_engine(db_path: str):
    """A SQLModel engine for the results DB at ``db_path``, with the schema ensured: the table is
    created from :class:`Result` when absent, and reconciled to it when present
    (:func:`add_missing_columns`)."""
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    add_missing_columns(engine)
    return engine
