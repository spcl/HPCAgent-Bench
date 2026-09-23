# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Typed SQLModel schema for the framework-benchmark ``results`` table: the Result model derives both
the DDL (``create_all``) and the row inserts."""

import re
from typing import ClassVar

from sqlalchemy import Table
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlmodel import Field, SQLModel, create_engine

#: The table name; SQLModel's default would be the class name lowercased, which is not it.
RESULTS_TABLE = "results"

#: SQLite's message when another connection ran the same check-then-act DDL first (CREATE TABLE in
#: ``create_all``, ALTER TABLE in :func:`add_missing_columns`). Matched by message: SQLite raises a
#: plain ``OperationalError`` for real schema errors too.
CONCURRENT_SCHEMA_RACE = re.compile(r"table \S+ already exists|duplicate column name")


def is_concurrent_schema_race(exc: OperationalError) -> bool:
    """True when ``exc`` is the loser of a same-DDL race against another writer to this shard file,
    not a genuine schema problem -- the table/column it wanted now exists either way."""
    message = str(exc.orig) if exc.orig is not None else str(exc)
    return CONCURRENT_SCHEMA_RACE.search(message) is not None


class Result(SQLModel, table=True):
    """One (framework, flavor, build, benchmark, preset, datatype, variant) runtime sample."""

    __tablename__: ClassVar[str] = RESULTS_TABLE

    id: int | None = Field(default=None, primary_key=True)
    timestamp: int  # epoch seconds; groups the rows of one run
    benchmark: str  # kernel short_name
    domain: str | None = None  # taxonomy label; used as a heatmap grouping key
    preset: str  # S | M | L | XL
    framework: str  # numpy | dace_cpu | jax | ... -- the backend, WITHOUT its flavor suffix
    # Optimizer inside the framework (dace_cpu's parallel / autoopt / canonicalize); NULL == the
    # framework's default path. framework.split_flavor maps the flat CLI name to this pair.
    flavor: str | None = None
    agent: str | None = None  # who produced the optimization (None == direct framework run)
    validated: bool  # output matched the NumPy oracle
    time: float  # host-measured runtime, milliseconds
    native_time: float | None = None  # framework-internal runtime, ms (None if no native timer)
    datatype: str | None = None  # float32 | float64 | ... (None == legacy float64)
    variant: str | None = None  # sparse storage/distribution axis (None == dense)
    # Which build ran it (dace main vs extended, another BLAS or image), stamped by the launcher via
    # HPCAGENT_BENCH_RECORD_BUILD. NULL == single-build run.
    build: str | None = None
    prompt_hash: str | None = None  # -> the content-addressed prompt store (None if no prompt)
    execution: str = "native"  # native (no container) | container -- where the runtime was measured
    # CPU model that measured it (osinfo.cpu_model); required, so every row names its hardware.
    cpu: str
    # Device the measurement ran on; NULL for a CPU-only column even when the node has a GPU.
    gpu: str | None = None
    # Node name (osinfo.node_name); ``cpu`` cannot separate two nodes of one cluster. NULL on
    # rows recorded before the column existed.
    node: str | None = None


#: The per-kernel static metrics table; one row per (kernel, column, implementation, metric).
KERNEL_METRICS_TABLE = "kernel_metrics"


class KernelMetric(SQLModel, table=True):
    """One named count about a column's compiled kernel (``autovec.loops_vectorized``,
    ``parallelism.map``, ...). Long format: a new metric is a new ``metric`` value, so existing DBs
    need no migration. Counts only; rates belong to the report."""

    __tablename__: ClassVar[str] = KERNEL_METRICS_TABLE

    id: int | None = Field(default=None, primary_key=True)
    timestamp: int  # epoch seconds; groups the rows of one run
    benchmark: str  # kernel short_name
    framework: str  # the column WITHOUT its flavor suffix, as in results
    flavor: str | None = None
    impl: str  # the implementation name the report hooks key on
    datatype: str | None = None
    metric: str  # <family>.<count>
    value: float
    # Conditions of the count as "key=value" pairs (compiler family, cost model).
    detail: str | None = None
    build: str | None = None
    cpu: str
    node: str | None = None


def add_missing_columns(engine: Engine) -> None:
    """Add to an existing ``results`` table every nullable column :class:`Result` declares and the
    table lacks (``create_all`` never alters a present table). Old rows read NULL for it; a missing
    NOT NULL column raises, since no value can be backfilled. Each ADD COLUMN commits on its own, so
    losing a same-column race to another rank (:data:`CONCURRENT_SCHEMA_RACE`) keeps the columns
    this call already added."""
    # The metadata, not ``Result.__table__``: the same Table object, and the one spelling typed.
    table: Table = SQLModel.metadata.tables[RESULTS_TABLE]
    with engine.connect() as conn:
        present: set[str] = {row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table.name})")}
        if not present:
            return  # create_all just built it from the model; nothing to reconcile
        for name, column in table.columns.items():
            if name in present:
                continue
            if not column.nullable:
                raise RuntimeError(
                    f"results table lacks the NOT NULL column {name!r} and no value can be "
                    f"backfilled for existing rows; migrate {engine.url.database} by hand"
                )
            sql_type = column.type.compile(engine.dialect)
            try:
                conn.exec_driver_sql(f"ALTER TABLE {table.name} ADD COLUMN {name} {sql_type}")
            except OperationalError as exc:
                if not is_concurrent_schema_race(exc):
                    raise
                conn.rollback()  # another rank's ALTER for this column won the race; keep going
                continue
            conn.commit()


def results_engine(db_path: str) -> Engine:
    """A SQLModel engine for the results DB at ``db_path``, with the table created from
    :class:`Result` when absent and reconciled to it when present (:func:`add_missing_columns`).
    A lost CREATE TABLE race (:data:`CONCURRENT_SCHEMA_RACE`) is ignored: the winner built the
    same table."""
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    try:
        SQLModel.metadata.create_all(engine)
    except OperationalError as exc:
        if not is_concurrent_schema_race(exc):
            raise
    add_missing_columns(engine)
    return engine
