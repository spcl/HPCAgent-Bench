# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Typed SQLModel schema for the framework-benchmark ``results`` table: the single Result model derives
both the DDL (``create_all``) and row inserts, replacing the old hand-written CREATE TABLE/INSERT pair."""

from typing import ClassVar

from sqlalchemy import Table
from sqlalchemy.engine import Engine
from sqlmodel import Field, SQLModel, create_engine

#: The table name; SQLModel's default would be the class name lowercased, which is not it.
RESULTS_TABLE = "results"


class Result(SQLModel, table=True):
    """One (framework, flavor, build, benchmark, preset, datatype, variant) runtime sample."""

    __tablename__: ClassVar[str] = RESULTS_TABLE

    id: int | None = Field(default=None, primary_key=True)
    timestamp: int  # epoch seconds; groups the rows of one run
    benchmark: str  # kernel short_name
    domain: str | None = None  # taxonomy label; used as a heatmap grouping key
    preset: str  # S | M | L | XL
    framework: str  # numpy | dace_cpu | jax | ... -- the backend, WITHOUT its flavor suffix
    # Which optimizer inside the framework produced this row: dace_cpu's `parallel` / `autoopt` /
    # `canonicalize`. NULL == the framework's own default path, which for a searching column means
    # "the fastest of its pipelines" -- so `WHERE framework='dace_cpu'` gathers every DaCe CPU row
    # and `flavor` is what tells them apart. One flat name on the CLI (`--framework
    # dace_cpu_parallel`), two columns here; framework.split_flavor is the one place that maps
    # between them.
    flavor: str | None = None
    agent: str | None = None  # who produced the optimization (None == direct framework run)
    validated: bool  # output matched the NumPy oracle
    time: float  # host-measured runtime, milliseconds
    native_time: float | None = None  # framework-internal runtime, ms (None if no native timer)
    datatype: str | None = None  # float32 | float64 | ... (None == legacy float64)
    variant: str | None = None  # sparse storage/distribution axis (None == dense)
    # WHICH BUILD ran it -- upstream `main` vs the fork's `extended`, a different BLAS, a different
    # image. A separate axis from `flavor` because you cannot ASK for it: the flavor is a column you
    # name on the command line, the build is whatever PYTHONPATH resolved to, so it is stamped by
    # the launcher (HPCAGENT_BENCH_RECORD_BUILD) exactly like `execution`. NULL == single-build run.
    # Without it, the same pipeline measured on two DaCe trees is two indistinguishable rows.
    build: str | None = None
    prompt_hash: str | None = None  # -> the content-addressed prompt store (None if no prompt)
    execution: str = "native"  # native (no container) | container -- where the runtime was measured
    # WHICH MACHINE measured it. Two nodes are two experiments: a baseline timed on one CPU against
    # a candidate timed on another is a hardware comparison wearing a software label, and nothing
    # downstream can tell, because both rows look perfectly normal. REQUIRED, unlike every other
    # axis here -- a row that cannot name its host cannot be checked for that, so it must not be
    # expressible. Stamped from the machine (osinfo.cpu_model) rather than an env knob like `build`,
    # because the one thing that must never be forgotten is the one nobody has to remember.
    cpu: str
    # The DEVICE the measurement ran on; NULL for a CPU-only column. Not "the GPU in this box": a
    # device that took no part in the run must not split the figure for it, or the same CPU
    # measurement lands in two plots because someone swapped a card that was never used.
    gpu: str | None = None
    # The NODE (osinfo.node_name). ``cpu`` cannot separate two nodes of one cluster, and a ratio
    # across two nodes is a hardware comparison. NULL == recorded before the column existed.
    node: str | None = None


#: The per-kernel static metrics table; one row per (kernel, column, implementation, metric).
KERNEL_METRICS_TABLE = "kernel_metrics"


class KernelMetric(SQLModel, table=True):
    """One named count about a column's compiled kernel (``autovec.loops_vectorized``,
    ``parallelism.map``, ...). Long rather than wide: a new metric is a new ``metric`` value, not a
    column, so a metric family added later lands in every existing DB without a migration. Counts
    only; a rate is a reading of counts and belongs to the report."""

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
    # What the count was taken under, as "key=value" pairs (compiler family, cost model): two rows
    # with different details are two measurements, never one averaged number.
    detail: str | None = None
    build: str | None = None
    cpu: str
    node: str | None = None


def add_missing_columns(engine: Engine) -> None:
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
            conn.exec_driver_sql(f"ALTER TABLE {table.name} ADD COLUMN {name} {sql_type}")
        conn.commit()


def results_engine(db_path: str) -> Engine:
    """A SQLModel engine for the results DB at ``db_path``, with the schema ensured: the table is
    created from :class:`Result` when absent, and reconciled to it when present
    (:func:`add_missing_columns`)."""
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    add_missing_columns(engine)
    return engine
