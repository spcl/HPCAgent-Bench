# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared by the ledger sync tools (recv, send, merge, install): where things live and what a valid snapshot is.

Standard library only: the tools run from ``~/dphpc26_ledger/bin`` without the project environment, where
``install.py`` copies this file as ``common.py`` beside them."""

import contextlib
import pathlib
import shutil
import sqlite3

#: Everything the tools keep lives under one visible directory of the user's home.
ROOT = pathlib.Path.home() / "dphpc26_ledger"
BIN = ROOT / "bin"
INBOX = ROOT / "inbox"  # received snapshots waiting to be merged
REJECTED = ROOT / "rejected"  # snapshots that failed a check, each beside a .reason file
TEAM_DB = ROOT / "team.db"  # the merge of every snapshot received
TEAM_FILE = ROOT / "team"  # one user@host per line: where send delivers
KEY = pathlib.Path.home() / ".ssh" / "dphpc26_ledger_key"
#: The last field of the authorized_keys line the installer writes, so it can find its own line again.
MARKER = "dphpc26_ledger"

#: The tables a snapshot holds, exactly (no others, no triggers, no views).
TABLES = {"source_files", "measurements", "timing_samples"}
#: The ledger's tables, a copy of ``SCHEMA`` in hpcagent_bench/support/collect/ledger.py (this file cannot import the
#: package): keep the two in step. Only ``init_team_db`` uses it; a merge copies the columns both sides have.
SCHEMA = """
CREATE TABLE IF NOT EXISTS source_files (
    hash    TEXT PRIMARY KEY,               -- sha256 of content, UTF-8
    content TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS measurements (
    measurement_id TEXT PRIMARY KEY,        -- uuid4, made when the row is built, so a write retry keeps it
    invocation_id  TEXT NOT NULL,           -- uuid4 of one `run-framework` command
    started_at     TEXT NOT NULL,           -- the command's start, UTC ISO 8601
    user           TEXT NOT NULL DEFAULT '',-- the OS user who ran the command (getpass.getuser)
    command        TEXT NOT NULL,           -- JSON list: sys.argv (not the shell line that launched it)
    metadata       TEXT NOT NULL,           -- JSON, layout `format` 1 (capture)
    sources        TEXT NOT NULL DEFAULT '{}',
                                            -- JSON {path relative to the checkout: source_files.hash | null}
    run            TEXT NOT NULL DEFAULT '',-- the --canon-run label, as in canon.run
    "column"       TEXT NOT NULL,           -- the framework column
    kernel         TEXT NOT NULL,
    preset         TEXT NOT NULL,
    datatype       TEXT NOT NULL,
    impl           TEXT NOT NULL DEFAULT '',
    median_ms      REAL,                    -- NULL = no time
    median_source  TEXT NOT NULL DEFAULT '',-- 'native' (device) or 'python' (host): the series median_ms is of
    validated      INTEGER CHECK (validated IN (0, 1)),
    status         TEXT NOT NULL CHECK (status IN ('ok', 'crash', 'timeout')),
    failure        TEXT NOT NULL DEFAULT '',
    error          TEXT NOT NULL DEFAULT '',
    node           TEXT NOT NULL DEFAULT '',
    gpu            TEXT NOT NULL DEFAULT '',
    ts_ms          INTEGER NOT NULL,
    UNIQUE (invocation_id, "column", kernel, preset, datatype, impl)
);
CREATE TABLE IF NOT EXISTS timing_samples (
    measurement_id TEXT NOT NULL REFERENCES measurements (measurement_id) ON DELETE CASCADE,
    sample_index   INTEGER NOT NULL,        -- 0-based, execution order; warmups are not retained
    host_ms        REAL,                    -- NULL = not measured, never 0
    device_ms      REAL,                    -- CUDA events; NULL on a column without them
    PRIMARY KEY (measurement_id, sample_index)
);
"""

#: Limits recv enforces.
MAX_BYTES = 64 * 1024 * 1024  # one snapshot
MAX_PENDING = 20  # snapshots waiting in the inbox from one sender
MAX_INBOX_BYTES = 512 * 1024 * 1024  # the whole inbox


def init_team_db() -> None:
    """Create ``team.db`` with empty tables unless it exists, so it can be read before the first merge."""
    ROOT.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(sqlite3.connect(TEAM_DB)) as conn:
        conn.executescript(SCHEMA)


def check_snapshot(path: pathlib.Path) -> str | None:
    """Why ``path`` is not a valid ledger snapshot, ``None`` when it is.

    Opened immutable, so the check leaves no -wal or -shm file beside it."""
    try:
        with contextlib.closing(sqlite3.connect(f"{path.as_uri()}?immutable=1", uri=True)) as conn:
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                return "integrity_check failed"
            objects = conn.execute("SELECT type, name FROM sqlite_master").fetchall()
    except sqlite3.DatabaseError as error:
        return f"not a database: {error}"
    if extra := sorted(name for kind, name in objects if kind in ("trigger", "view")):
        return f"holds triggers or views: {extra}"
    if {name for kind, name in objects if kind == "table"} != TABLES:
        return f"tables are not {sorted(TABLES)}"
    return None


def reject(path: pathlib.Path, reason: str) -> None:
    """Move ``path`` to the rejected directory and write the reason beside it."""
    REJECTED.mkdir(parents=True, exist_ok=True)
    target = REJECTED / path.name.removesuffix(".part")
    shutil.move(path, target)
    target.with_name(target.name + ".reason").write_text(reason + "\n", encoding="utf-8")
