#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rename the data names of a results file to the setup/study vocabulary, in place and loss-free.

A file is one of three kinds, each recognised by its content and refused when it matches none:

* a results database at schema v1: table ``arms`` becomes ``setups``, ``arms.arm`` and ``runs.arm``
  become ``setup``, ``arms.experiment`` (it holds a study tag such as ``llr40``) becomes ``study``,
  the view ``grades_flat`` is recreated over the new names, and ``PRAGMA user_version`` goes 1 -> 2;
* an extracted observations database: the ``arm`` column of every table that has one becomes ``setup``;
* an extracted or frozen observations CSV: the ``arm`` header cell becomes ``setup``.

Each file is migrated in one transaction (a CSV in one atomic replace). Before the commit the migrated
content is compared with the original: the table and column lists under the old -> new mapping, the row
count of every table and a SHA-256 over every table's rows in rowid order, with the columns mapped. Any
difference rolls the file back untouched. A file that already carries the new names is reported and
left alone, so a second run changes nothing.

    migrate_names_to_v2.py PATH [PATH ...]            migrate files; a directory is walked for *.db, *.sqlite, *.csv
    migrate_names_to_v2.py --dry-run PATH ...         do everything but commit (a SQLite file is rolled back)
    migrate_names_to_v2.py --verify ORIGINAL NEW      compare a migrated copy against the original it came from

Exit status: 0 when every file migrated, was already migrated or verified equal; 2 when any file was
refused or differed.
"""

import argparse
import csv
import dataclasses
import hashlib
import os
import pathlib
import sqlite3
import sys
import tempfile
from collections.abc import Iterator, Sequence

SCHEMA_OLD = 1
SCHEMA_NEW = 2
LEGACY_TABLES = frozenset({"calls", "submissions", "attempts", "submission_cells", "regrade_tasks", "regrades"})
RESULTS_TABLES = frozenset({"arms", "runs", "sources", "grades"})
TABLE_RENAMES: dict[str, str] = {"arms": "setups"}
#: (table before the rename, old column) -> new column.
COLUMN_RENAMES: dict[tuple[str, str], str] = {
    ("arms", "arm"): "setup",
    ("arms", "experiment"): "study",
    ("runs", "arm"): "setup",
}
OBSERVATION_COLUMNS = frozenset({"run_id", "benchmark", "arm"})
GRADES_FLAT = """CREATE VIEW grades_flat AS
SELECT a.study, a.model, a.language, a.device, a.packet, a.harness, r.setup, r.job, r.label, r.rep, g.*
FROM grades AS g
JOIN runs AS r ON r.id = g.run_id
JOIN setups AS a ON a.setup = r.setup"""
CHUNK = 1 << 20
SUFFIXES = (".db", ".sqlite", ".csv")


class Refused(Exception):
    """The file is not one this script recognises, or migrating it would change its content."""


@dataclasses.dataclass(frozen=True, slots=True)
class Snapshot:
    """What the verification compares: the shape of every table and view, row counts and row hashes."""

    columns: dict[str, tuple[str, ...]]
    counts: dict[str, int]
    digests: dict[str, str]


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_columns(conn: sqlite3.Connection) -> dict[str, tuple[str, ...]]:
    """Every table and view with its columns, in declaration order."""
    names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name")]
    return {n: tuple(r[1] for r in conn.execute(f"PRAGMA table_info({quote(n)})")) for n in names}


def digest(conn: sqlite3.Connection, table: str, columns: Sequence[str]) -> str:
    """SHA-256 over the rows of ``table`` in rowid order, ``columns`` selected by name."""
    sha = hashlib.sha256()
    select = ", ".join(quote(c) for c in columns)
    for row in conn.execute(f"SELECT {select} FROM {quote(table)} ORDER BY rowid"):
        sha.update(repr(row).encode("utf-8", "surrogatepass"))
        sha.update(b"\n")
    return sha.hexdigest()


def snapshot(conn: sqlite3.Connection, skip: frozenset[str] = frozenset()) -> Snapshot:
    """Shape, counts and row hashes of every real table; views are compared by shape only."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    columns = {name: cols for name, cols in table_columns(conn).items() if name in tables}
    counts: dict[str, int] = {}
    digests: dict[str, str] = {}
    for name in sorted(tables - skip):
        counts[name] = int(conn.execute(f"SELECT COUNT(*) FROM {quote(name)}").fetchone()[0])
        digests[name] = digest(conn, name, columns[name])
    return Snapshot(columns, counts, digests)


def mapped(before: Snapshot, table_map: dict[str, str], column_map: dict[tuple[str, str], str]) -> Snapshot:
    """``before`` with every table and column renamed the way the migration renames them."""
    columns = {
        table_map.get(t, t): tuple(column_map.get((t, c), c) for c in cols) for t, cols in before.columns.items()
    }
    return Snapshot(
        columns,
        {table_map.get(t, t): n for t, n in before.counts.items()},
        {table_map.get(t, t): d for t, d in before.digests.items()},
    )


def describe(conn: sqlite3.Connection) -> list[str]:
    """One line per table and view: name, columns, row count."""
    lines = []
    kinds = dict(conn.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"))
    for name, columns in table_columns(conn).items():
        rows = int(conn.execute(f"SELECT COUNT(*) FROM {quote(name)}").fetchone()[0]) if kinds[name] == "table" else -1
        count = f"{rows} rows" if rows >= 0 else "view"
        lines.append(f"    {name} ({count}): {', '.join(columns)}")
    return lines


def classify(conn: sqlite3.Connection) -> str:
    """``results-v1``, ``results-v2``, ``observations`` or ``observations-done``; raises Refused otherwise."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if tables & LEGACY_TABLES:
        raise Refused("a legacy (pre-v1) results database: this script renames v1 files only")
    if RESULTS_TABLES <= tables:
        if version == SCHEMA_OLD:
            return "results-v1"
        raise Refused(f"results tables with user_version {version} (expected {SCHEMA_OLD})")
    if {"setups", "runs", "sources", "grades"} <= tables:
        if version == SCHEMA_NEW:
            return "results-v2"
        raise Refused(f"setups table with user_version {version} (expected {SCHEMA_NEW})")
    if "observations" in tables:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(observations)")}
        if OBSERVATION_COLUMNS <= columns:
            return "observations"
        if "setup" in columns and {"run_id", "benchmark"} <= columns:
            return "observations-done"
    raise Refused("neither a v1 results database nor an observations database")


def migrate_results(conn: sqlite3.Connection) -> None:
    """The v1 -> v2 renames, inside the caller's transaction."""
    arms = [r[1] for r in conn.execute("PRAGMA table_info(arms)")]
    if not {"arm", "experiment"} <= set(arms):
        raise Refused(f"arms table without arm/experiment columns: {arms}")
    conn.execute("DROP VIEW IF EXISTS grades_flat")
    conn.execute("ALTER TABLE arms RENAME TO setups")
    conn.execute("ALTER TABLE setups RENAME COLUMN arm TO setup")
    conn.execute("ALTER TABLE setups RENAME COLUMN experiment TO study")
    conn.execute("ALTER TABLE runs RENAME COLUMN arm TO setup")
    conn.execute(GRADES_FLAT)
    conn.execute(f"PRAGMA user_version = {SCHEMA_NEW}")


def migrate_observations(conn: sqlite3.Connection) -> dict[tuple[str, str], str]:
    """Rename ``arm`` to ``setup`` in every table that has it; returns the column map applied."""
    column_map: dict[tuple[str, str], str] = {}
    for table, columns in table_columns(conn).items():
        kind = conn.execute("SELECT type FROM sqlite_master WHERE name = ?", (table,)).fetchone()[0]
        if kind == "table" and "arm" in columns:
            if "setup" in columns:
                raise Refused(f"table {table} has both arm and setup")
            conn.execute(f"ALTER TABLE {quote(table)} RENAME COLUMN arm TO setup")
            column_map[(table, "arm")] = "setup"
    return column_map


def check_equal(
    before: Snapshot, after: Snapshot, table_map: dict[str, str], column_map: dict[tuple[str, str], str]
) -> None:
    expected = mapped(before, table_map, column_map)
    for label, want, got in (
        ("tables and columns", expected.columns, after.columns),
        ("row counts", expected.counts, after.counts),
        ("row checksums", expected.digests, after.digests),
    ):
        if want != got:
            bad = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
            raise Refused(f"{label} differ after the migration in: {bad}")


def migrate_database(path: pathlib.Path, dry_run: bool) -> str:
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        kind = classify(conn)
        if kind in ("results-v2", "observations-done"):
            return "already migrated"
        print(f"  before ({kind}):")
        print("\n".join(describe(conn)))
        conn.execute("BEGIN IMMEDIATE")
        try:
            before = snapshot(conn)
            if kind == "results-v1":
                migrate_results(conn)
                table_map, column_map = TABLE_RENAMES, COLUMN_RENAMES
            else:
                table_map, column_map = {}, migrate_observations(conn)
            after = snapshot(conn)
            check_equal(before, after, table_map, column_map)
            if kind == "results-v1":
                broken = conn.execute("PRAGMA foreign_key_check").fetchall()
                if broken:
                    raise Refused(f"foreign_key_check reports {len(broken)} violations after the migration")
                flat = int(conn.execute("SELECT COUNT(*) FROM grades_flat").fetchone()[0])
                if flat != after.counts["grades"]:
                    raise Refused(f"grades_flat holds {flat} rows for {after.counts['grades']} grades")
            print("  after:")
            print("\n".join(describe(conn)))
            conn.execute("ROLLBACK" if dry_run else "COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        return "would migrate (rolled back)" if dry_run else "migrated"
    except sqlite3.DatabaseError as exc:
        raise Refused(f"not a readable SQLite database: {exc}") from exc
    finally:
        conn.close()


def split_header(raw: bytes) -> tuple[bytes, bytes, bytes]:
    """The first line without its newline, the newline bytes and the rest of the file."""
    end = raw.find(b"\n")
    if end < 0:
        return raw, b"", b""
    cut = end - 1 if end > 0 and raw[end - 1 : end] == b"\r" else end
    return raw[:cut], raw[cut : end + 1], raw[end + 1 :]


def csv_cells(header: bytes) -> list[str]:
    return next(csv.reader([header.decode("utf-8")]), [])


def render_cells(cells: Sequence[str]) -> bytes:
    out = tempfile.SpooledTemporaryFile(mode="w+", newline="")
    csv.writer(out, lineterminator="").writerow(cells)
    out.seek(0)
    return out.read().encode("utf-8")


def migrate_csv(path: pathlib.Path, dry_run: bool) -> str:
    raw = path.read_bytes()
    header, newline, rest = split_header(raw)
    try:
        cells = csv_cells(header)
    except UnicodeDecodeError as exc:
        raise Refused(f"not a UTF-8 CSV: {exc}") from exc
    if "setup" in cells and "arm" not in cells and {"run_id", "benchmark"} <= set(cells):
        return "already migrated"
    if not OBSERVATION_COLUMNS <= set(cells):
        raise Refused(f"CSV header lacks {sorted(OBSERVATION_COLUMNS - set(cells))}: not an observations table")
    if "setup" in cells:
        raise Refused("CSV has both arm and setup columns")
    print(f"  before (observations csv): {len(cells)} columns, {len(rest)} body bytes")
    renamed = ["setup" if c == "arm" else c for c in cells]
    out = render_cells(renamed) + newline + rest
    new_header, new_newline, new_rest = split_header(out)
    if new_rest != rest or new_newline != newline or csv_cells(new_header) != renamed:
        raise Refused("the rewritten CSV differs from the original beyond the header cell")
    if hashlib.sha256(new_rest).digest() != hashlib.sha256(rest).digest():
        raise Refused("body checksum differs after the rewrite")
    print(
        f"  after: arm -> setup at column {cells.index('arm')}, body checksum {hashlib.sha256(rest).hexdigest()[:16]}"
    )
    if dry_run:
        return "would migrate (nothing written)"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(out)
        os.chmod(tmp, path.stat().st_mode & 0o7777)
        os.replace(tmp, path)
    except BaseException:
        pathlib.Path(tmp).unlink(missing_ok=True)
        raise
    return "migrated"


def migrate(path: pathlib.Path, dry_run: bool) -> str:
    if path.suffix == ".csv":
        return migrate_csv(path, dry_run)
    return migrate_database(path, dry_run)


def walk(paths: Sequence[pathlib.Path]) -> Iterator[pathlib.Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in SUFFIXES)
        else:
            yield path


def read_only(path: pathlib.Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def verify(original: pathlib.Path, migrated: pathlib.Path) -> None:
    """Raise Refused unless ``migrated`` is ``original`` under the renames, table by table."""
    if original.suffix == ".csv":
        old_header, _, old_rest = split_header(original.read_bytes())
        new_header, _, new_rest = split_header(migrated.read_bytes())
        want = ["setup" if c == "arm" else c for c in csv_cells(old_header)]
        if want != csv_cells(new_header) or old_rest != new_rest:
            raise Refused("CSV differs beyond the arm -> setup header cell")
        print(f"  equal: {len(want)} columns, body {hashlib.sha256(old_rest).hexdigest()[:16]}")
        return
    with read_only(original) as old, read_only(migrated) as new:
        kind = classify(old)
        before = snapshot(old)
        after = snapshot(new)
        if kind == "results-v1":
            check_equal(before, after, TABLE_RENAMES, COLUMN_RENAMES)
            flat = int(new.execute("SELECT COUNT(*) FROM grades_flat").fetchone()[0])
            if flat != int(old.execute("SELECT COUNT(*) FROM grades_flat").fetchone()[0]):
                raise Refused("grades_flat holds a different number of rows")
        elif kind == "observations":
            columns = {
                (t, "arm"): "setup" for t, cols in before.columns.items() if "arm" in cols and t in before.counts
            }
            check_equal(before, after, {}, columns)
        else:
            raise Refused(f"original is already {kind}: nothing to compare")
        total = sum(before.counts.values())
        print(f"  equal: {len(before.counts)} tables, {total} rows, {len(before.digests)} checksums")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=pathlib.Path, help="files or directories")
    parser.add_argument("--dry-run", action="store_true", help="check and report, change nothing")
    parser.add_argument("--verify", action="store_true", help="PATHS is ORIGINAL NEW: compare, change nothing")
    args = parser.parse_args(argv)
    failed = 0
    if args.verify:
        if len(args.paths) != 2:
            parser.error("--verify takes exactly ORIGINAL and NEW")
        print(f"{args.paths[1]} against {args.paths[0]}")
        try:
            verify(*args.paths)
        except (Refused, sqlite3.DatabaseError) as exc:
            print(f"  DIFFERS: {exc}")
            return 2
        return 0
    for path in walk(args.paths):
        print(f"{path}")
        try:
            print(f"  -> {migrate(path, args.dry_run)}")
        except Refused as exc:
            failed += 1
            print(f"  -> REFUSED: {exc}")
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
