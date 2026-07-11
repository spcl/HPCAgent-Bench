# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Typed SQLModel schema for the framework-benchmark ``results`` table.

The model IS the schema: one ``class Result(SQLModel, table=True)`` with typed,
``Optional``-nullable fields replaces the parallel hand-written ``CREATE TABLE`` /
``INSERT`` string pair that used to live in :mod:`optarena.infrastructure.utilities`
and could drift out of sync. ``SQLModel.metadata.create_all`` derives the DDL and a
``Session`` writes rows, both from this single definition.

The perf record is deliberately lean -- who/what produced a runtime and the runtime
itself, nothing more: benchmark, framework, agent, an optional content-addressed
prompt hash, preset, the two runtimes (host ``time`` and framework-internal
``native_time``, both milliseconds), the validation verdict, datatype, variant, and
the ``execution`` provenance (``native`` vs ``container`` -- so a containerized number
is never compared against a native one unknowingly).
``timestamp`` groups the rows of one run; ``id`` is the rowid. The old
provenance-only columns (kind / dwarf / version / details / mode / cpu) were pruned:
no reader consumed them (kind/dwarf/version are dropped verbatim by the heatmap plot,
mode was always the constant ``"main"``, cpu had no reader).
"""
from typing import Optional

from sqlmodel import Field, SQLModel, create_engine


class Result(SQLModel, table=True):
    """One (framework, benchmark, preset, datatype, variant) runtime sample."""

    __tablename__ = "results"

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: int  # epoch seconds; groups the rows of one run
    benchmark: str  # kernel short_name
    domain: Optional[str] = None  # taxonomy label; used as a heatmap grouping key
    preset: str  # S | M | L | XL
    framework: str  # numpy | dace | jax | ...
    agent: Optional[str] = None  # who produced the optimization (None == direct framework run)
    validated: bool  # output matched the NumPy oracle
    time: float  # host-measured runtime, milliseconds
    native_time: Optional[float] = None  # framework-internal runtime, ms (None if no native timer)
    datatype: Optional[str] = None  # float32 | float64 | ... (None == legacy float64)
    variant: Optional[str] = None  # sparse storage/distribution axis (None == dense)
    prompt_hash: Optional[str] = None  # -> the content-addressed prompt store (None if no prompt)
    execution: str = "native"  # native (no container) | container -- where the runtime was measured


def results_engine(db_path: str):
    """A SQLModel engine for the results DB at ``db_path``, with the schema ensured.

    ``create_all`` is idempotent (``CREATE TABLE IF NOT EXISTS``) so this is safe to
    call on every run. ``check_same_thread=False`` matches the historic reuse of a
    single :func:`sqlite3.connect` handle across the run. Note: ``create_all`` only
    CREATEs a missing table -- it does not ALTER an existing legacy ``results`` table,
    so a DB written under the old (wider) schema is not auto-migrated to the pruned one.
    """
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine
