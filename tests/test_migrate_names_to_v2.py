# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``scripts/migrate_names_to_v2.py`` renames a results file loss-free, once, and refuses what it does not know.

The database fixtures are built here with the v1 names (``arms``, ``arm``, ``arms.experiment``) the script
exists to replace, so they stay valid after the repository's own schema moves on.
"""

import csv
import pathlib
import subprocess
import sys

from tests.sqlite_closing import connect

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "migrate_names_to_v2.py"

V1_SCHEMA = """
PRAGMA user_version = 1;
CREATE TABLE arms (arm TEXT PRIMARY KEY, experiment TEXT, model TEXT, language TEXT NOT NULL,
                   device TEXT NOT NULL, packet TEXT NOT NULL DEFAULT '', harness TEXT NOT NULL) STRICT;
CREATE TABLE runs (id INTEGER PRIMARY KEY, arm TEXT NOT NULL REFERENCES arms (arm), job INTEGER,
                   label TEXT NOT NULL, rep INTEGER NOT NULL DEFAULT 1) STRICT;
CREATE TABLE sources (hash TEXT PRIMARY KEY, text TEXT NOT NULL) STRICT;
CREATE TABLE grades (id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL REFERENCES runs (id),
                     benchmark TEXT NOT NULL, speedup REAL) STRICT;
CREATE VIEW grades_flat AS
SELECT a.experiment, a.model, a.language, a.device, a.packet, a.harness, r.arm, r.job, r.label, r.rep, g.*
FROM grades AS g JOIN runs AS r ON r.id = g.run_id JOIN arms AS a ON a.arm = r.arm;
"""
OBSERVATION_HEADER = ["run_root", "job", "run_id", "arm", "benchmark", "speedup"]


def v1_database(path: pathlib.Path) -> pathlib.Path:
    with connect(path) as conn:
        conn.executescript(V1_SCHEMA)
        conn.executemany(
            "INSERT INTO arms VALUES (?, ?, ?, 'c', 'cpu', '', 'claude')",
            [("llr40-qwen38-c", "llr40", "qwen38"), ("mlscale20-oss120b-hip", "mlscale20", None)],
        )
        conn.executemany(
            "INSERT INTO runs (arm, job, label, rep) VALUES (?, ?, ?, 1)",
            [("llr40-qwen38-c", 100 + i, f"llr40-qwen38-c.n0.p{i}.w{i}") for i in range(4)],
        )
        conn.executemany("INSERT INTO grades (run_id, benchmark, speedup) VALUES (?, 'gemm', ?)", [(1, 2.5), (2, None)])
        conn.execute("INSERT INTO sources VALUES ('abc', 'void gemm(void) {}')")
    return path


def migrate(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True, check=False)


def tables(path: pathlib.Path) -> dict[str, list[str]]:
    with connect(path) as conn:
        names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        return {n: [r[1] for r in conn.execute(f"PRAGMA table_info({n})")] for n in names}


def test_a_v1_database_migrates_with_every_row_kept(tmp_path: pathlib.Path) -> None:
    original = v1_database(tmp_path / "original.db")
    copy = v1_database(tmp_path / "copy.db")
    done = migrate(copy)
    assert done.returncode == 0 and "-> migrated" in done.stdout, done.stdout + done.stderr
    shape = tables(copy)
    assert "arms" not in shape and shape["setups"][:2] == ["setup", "study"] and shape["runs"][1] == "setup"
    with connect(copy) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        flat = conn.execute("SELECT study, setup, label, speedup FROM grades_flat ORDER BY id").fetchall()
    assert flat == [
        ("llr40", "llr40-qwen38-c", "llr40-qwen38-c.n0.p0.w0", 2.5),
        ("llr40", "llr40-qwen38-c", "llr40-qwen38-c.n0.p1.w1", None),
    ]
    check = migrate("--verify", original, copy)
    assert check.returncode == 0 and "equal: 4 tables" in check.stdout, check.stdout


def test_a_second_run_changes_nothing_and_verify_catches_a_lost_row(tmp_path: pathlib.Path) -> None:
    original = v1_database(tmp_path / "original.db")
    copy = v1_database(tmp_path / "copy.db")
    assert migrate(copy).returncode == 0
    before = copy.read_bytes()
    again = migrate(copy)
    assert again.returncode == 0 and "already migrated" in again.stdout and copy.read_bytes() == before
    with connect(copy) as conn:
        conn.execute("DELETE FROM grades WHERE id = 2")
    broken = migrate("--verify", original, copy)
    assert broken.returncode == 2 and "DIFFERS" in broken.stdout and "grades" in broken.stdout


def test_a_failure_inside_the_migration_leaves_the_file_exactly_as_it_was(tmp_path: pathlib.Path) -> None:
    path = v1_database(tmp_path / "no_runs_arm.db")
    with connect(path) as conn:
        conn.executescript(
            "DROP VIEW grades_flat; DROP TABLE grades; DROP TABLE runs;"
            "CREATE TABLE runs (id INTEGER PRIMARY KEY, label TEXT NOT NULL) STRICT;"
            "CREATE TABLE grades (id INTEGER PRIMARY KEY, run_id INTEGER, benchmark TEXT NOT NULL) STRICT;"
        )
    before = path.read_bytes()
    done = migrate(path)
    assert done.returncode == 2 and "REFUSED" in done.stdout
    assert path.read_bytes() == before and "arms" in tables(path)


def test_files_that_are_not_v1_results_are_refused_untouched(tmp_path: pathlib.Path) -> None:
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"not sqlite" * 40)
    legacy = tmp_path / "legacy.db"
    with connect(legacy) as conn:
        conn.executescript(
            "CREATE TABLE calls (id); CREATE TABLE arms (arm); CREATE TABLE runs (id); CREATE TABLE grades (id);"
        )
    other = tmp_path / "other.csv"
    other.write_text("a,b\n1,2\n")
    before = {p: p.read_bytes() for p in (junk, legacy, other)}
    done = migrate(junk, legacy, other)
    assert done.returncode == 2 and done.stdout.count("REFUSED") == 3, done.stdout
    assert {p: p.read_bytes() for p in before} == before


def test_an_observations_database_and_csv_rename_arm_and_keep_every_row(tmp_path: pathlib.Path) -> None:
    rows = [["rr", "1", "r1", "a-c", "gemm", "1.5"], ["rr", "2", "r2", "a-hip", "atax", ""]]
    csv_path, csv_original = tmp_path / "obs.csv", tmp_path / "obs.original.csv"
    for path in (csv_path, csv_original):
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(OBSERVATION_HEADER)
            writer.writerows(rows)
    db, db_original = tmp_path / "obs.db", tmp_path / "obs.original.db"
    for path in (db, db_original):
        with connect(path) as conn:
            conn.execute(
                "CREATE TABLE observations (run_root TEXT, job INTEGER, run_id TEXT, arm TEXT, benchmark TEXT, speedup REAL)"
            )
            conn.executemany("INSERT INTO observations VALUES (?, ?, ?, ?, ?, NULLIF(?, ''))", rows)
    done = migrate(csv_path, db)
    assert done.returncode == 0 and done.stdout.count("-> migrated") == 2, done.stdout
    assert csv_path.read_text().splitlines()[0] == "run_root,job,run_id,setup,benchmark,speedup"
    assert tables(db)["observations"][3] == "setup"
    assert migrate("--verify", csv_original, csv_path).returncode == 0
    assert migrate("--verify", db_original, db).returncode == 0
    assert migrate(csv_path, db).stdout.count("already migrated") == 2


if __name__ == "__main__":
    import tempfile

    for test in (
        test_a_v1_database_migrates_with_every_row_kept,
        test_a_second_run_changes_nothing_and_verify_catches_a_lost_row,
        test_a_failure_inside_the_migration_leaves_the_file_exactly_as_it_was,
        test_files_that_are_not_v1_results_are_refused_untouched,
        test_an_observations_database_and_csv_rename_arm_and_keep_every_row,
    ):
        with tempfile.TemporaryDirectory() as scratch:
            test(pathlib.Path(scratch))
            print("ok", test.__name__)
