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
#: Limits recv enforces.
MAX_BYTES = 64 * 1024 * 1024  # one snapshot
MAX_PENDING = 20  # snapshots waiting in the inbox from one sender
MAX_INBOX_BYTES = 512 * 1024 * 1024  # the whole inbox


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
