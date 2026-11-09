#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Send a snapshot of your ledger to every account in the team file. Installed as ``dphpc26_ledger_send``.

    dphpc26_ledger_send [--db results/ledger.db]

The snapshot is taken with SQLite's backup API (a plain copy of a WAL database can miss committed data), then sent over
ssh to each ``user@host`` in ``~/dphpc26_ledger/team`` -- yourself included, so every member ends up with the same
data. Each target's forced command (``recv.py``) answers ``OK`` or ``ERR``; a failed target does not stop the others."""

import argparse
import contextlib
import getpass
import hashlib
import pathlib
import re
import sqlite3
import subprocess
import sys
import tempfile

import common


def snapshot(db: pathlib.Path) -> bytes:
    """A consistent copy of ``db`` as bytes, in rollback-journal mode so it is one self-contained file."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = pathlib.Path(tmp) / "snapshot.db"
        with contextlib.closing(sqlite3.connect(db)) as source, contextlib.closing(sqlite3.connect(copy)) as target:
            source.backup(target)
            target.execute("PRAGMA journal_mode = DELETE")
        return copy.read_bytes()


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--db", type=pathlib.Path, default=pathlib.Path("results/ledger.db"))
    args = parser.parse_args()
    if not args.db.is_file():
        print(f"no ledger at {args.db}")
        return 1
    if not common.TEAM_FILE.is_file():
        print(f"no team file at {common.TEAM_FILE}; run install.py")
        return 1
    targets = [
        line.strip()
        for line in common.TEAM_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    try:
        data = snapshot(args.db)
    except sqlite3.DatabaseError as error:
        print(f"{args.db} is not a readable ledger: {error}")
        return 1
    label = re.sub(r"[^A-Za-z0-9]", "", getpass.getuser())[:32]
    payload = f"LEDGER1 {label} {len(data)} {hashlib.sha256(data).hexdigest()}\n".encode() + data
    failed = 0
    for target in targets:
        # No remote command: the receiving account's forced command is what runs.
        command = [
            "ssh", "-T", "-i", str(common.KEY),
            "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ControlPath=none",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20",
            target,
        ]  # fmt: skip
        try:
            done = subprocess.run(command, input=payload, capture_output=True, timeout=600, check=False)
        except subprocess.TimeoutExpired:
            print(f"{target}: timed out")
            failed += 1
            continue
        replies = done.stdout.decode(errors="replace").strip().splitlines()
        reply = replies[-1] if replies else done.stderr.decode(errors="replace").strip()
        print(f"{target}: {reply}")
        failed += not reply.startswith("OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
