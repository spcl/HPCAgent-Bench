# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``scripts/migrate_names_to_v3.py`` moves a v2 results database to v3 loss-free, once, and refuses what it does not know.

The v2 fixtures are built from ``tests/data/results_schema_v2.sql``, the schema this repository shipped before v3, so they
stay valid after the repository's own schema moves on.
"""

import pathlib
import subprocess
import sys

from tests.sqlite_closing import connect

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "migrate_names_to_v3.py"
V2_SCHEMA = ROOT / "tests" / "data" / "results_schema_v2.sql"


def v2_database(path: pathlib.Path, setups: tuple[tuple[str, str, str], ...]) -> pathlib.Path:
    """A v2 database with one episode and one grade per ``(setup, study, label prefix)``."""
    with connect(path) as conn:
        conn.executescript(V2_SCHEMA.read_text(encoding="utf-8"))
        for index, (setup, study, prefix) in enumerate(setups, start=1):
            conn.execute(
                "INSERT INTO setups (setup, study, model, language, device, packet, harness) "
                "VALUES (?, ?, 'qwen38', 'c', 'cpu', '', 'claude')",
                (setup, study),
            )
            conn.execute(
                "INSERT INTO runs (id, setup, job, label, rep, benchmark) VALUES (?, ?, ?, ?, 1, 'gemm')",
                (index, setup, 100 + index, f"{prefix}.n0.p0.w0"),
            )
            conn.execute(
                "INSERT INTO grades (run_id, benchmark, ts_ms, kind, status) VALUES (?, 'gemm', ?, 'score', 'ok')",
                (index, 1000 + index),
            )
        conn.execute(
            "INSERT INTO reference_scaling_points (source, benchmark, mode, ranks, ts_ms) VALUES ('t', 'gemm', 'weak', 1, 5)"
        )
        conn.execute("INSERT INTO sources VALUES ('abc', 'void gemm(void) {}')")
    return path


def migrate(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True, check=False)


def rows(path: pathlib.Path, sql: str) -> list[tuple[object, ...]]:
    with connect(path) as conn:
        return [tuple(row) for row in conn.execute(sql)]


def test_a_v2_database_migrates_with_every_row_kept(tmp_path: pathlib.Path) -> None:
    db = v2_database(tmp_path / "r.db", (("llr40-qwen38-c", "llr40", "llr40-qwen38-c"),))
    done = migrate(db)
    assert done.returncode == 0 and "-> migrated" in done.stdout, done.stdout
    assert rows(db, "PRAGMA user_version") == [(3,)]
    assert rows(db, "SELECT count(*) FROM episodes") == [(1,)]
    assert rows(db, "SELECT episode_id, kernel FROM grades") == [(1, "gemm")]
    assert rows(db, "SELECT kernel FROM reference_scaling_points") == [("gemm",)]
    assert rows(db, "SELECT setup, study, kernel, label FROM grades_flat") == [
        ("llr40-qwen38-c", "llr40", "gemm", "llr40-qwen38-c.n0.p0.w0")
    ]
    assert not rows(db, "SELECT name FROM sqlite_master WHERE name IN ('runs', 'runs_key', 'grades_run')")


def test_a_dry_run_prints_every_change_and_leaves_the_file_alone(tmp_path: pathlib.Path) -> None:
    db = v2_database(
        tmp_path / "r.db", (("gpu-llr-focus40-qwen38-c-openmp", "gpu-llr-focus40", "gpu-llr-focus40-qwen38-c-openmp"),)
    )
    before = db.read_bytes()
    done = migrate("--dry-run", db)
    assert done.returncode == 0 and "would migrate" in done.stdout
    assert "table runs -> episodes" in done.stdout
    assert "column grades.run_id -> episode_id" in done.stdout
    assert "setup gpu-llr-focus40-qwen38-c-openmp -> llr40-qwen38-c-openmp" in done.stdout
    assert "study gpu-llr-focus40 -> llr40" in done.stdout
    assert db.read_bytes() == before


def test_an_old_setup_spelling_is_rewritten_in_setups_episodes_and_labels(tmp_path: pathlib.Path) -> None:
    db = v2_database(tmp_path / "r.db", (("llrblind-qwen38-c", "llrblind", "llrblind-qwen38-c"),))
    assert migrate(db).returncode == 0
    assert rows(db, "SELECT setup, study FROM setups") == [("llr40-qwen38-c-blind", "llr-focus40-blind")]
    assert rows(db, "SELECT setup, label FROM episodes") == [("llr40-qwen38-c-blind", "llr40-qwen38-c-blind.n0.p0.w0")]


def test_two_spellings_of_one_setup_merge_into_one_row(tmp_path: pathlib.Path) -> None:
    db = v2_database(
        tmp_path / "r.db",
        (
            ("llrblind-qwen38-c", "llrblind", "llrblind-qwen38-c"),
            ("llrblind-cmp-qwen38-c", "llr-focus40-blind", "llrblind-cmp-qwen38-c"),
        ),
    )
    done = migrate(db)
    assert done.returncode == 0, done.stdout
    assert rows(db, "SELECT setup FROM setups") == [("llr40-qwen38-c-blind",)]
    assert rows(db, "SELECT count(*) FROM episodes WHERE setup = 'llr40-qwen38-c-blind'") == [(2,)]


def test_a_merge_of_setups_that_differ_in_identity_is_refused_and_leaves_the_file_alone(tmp_path: pathlib.Path) -> None:
    db = v2_database(
        tmp_path / "r.db",
        (
            ("llrblind-qwen38-c", "llrblind", "llrblind-qwen38-c"),
            ("llrblind-cmp-qwen38-c", "llr-focus40-blind", "llrblind-cmp-qwen38-c"),
        ),
    )
    with connect(db) as conn:
        conn.execute("UPDATE setups SET language = 'fortran' WHERE setup = 'llrblind-cmp-qwen38-c'")
    before = db.read_bytes()
    done = migrate(db)
    assert done.returncode == 2 and "REFUSED" in done.stdout and "differ" in done.stdout
    assert db.read_bytes() == before


def test_the_clean_suffix_folds_and_a_kernel_graded_both_ways_is_listed(tmp_path: pathlib.Path) -> None:
    db = v2_database(
        tmp_path / "r.db",
        (("llr40-qwen38-c", "llr40", "llr40-qwen38-c"), ("llr40-qwen38-c-clean", "llr40", "llr40-qwen38-c-clean")),
    )
    dry = migrate("--dry-run", db)
    assert "collision llr40-qwen38-c: 1 kernels graded under a clean and a plain spelling: gemm" in dry.stdout
    assert migrate(db).returncode == 0
    assert rows(db, "SELECT setup FROM setups") == [("llr40-qwen38-c",)]
    assert rows(db, "SELECT count(*) FROM episodes WHERE setup = 'llr40-qwen38-c'") == [(2,)]


def test_the_cpf_archive_setups_move_to_the_llr40_prefix(tmp_path: pathlib.Path) -> None:
    db = v2_database(
        tmp_path / "r.db", (("cpf-llr-focus40-qwen38-c-cpf-clean", "llr40", "cpf-llr-focus40-qwen38-c-cpf-clean"),)
    )
    assert migrate(db).returncode == 0
    assert rows(db, "SELECT setup FROM setups") == [("llr40-qwen38-c-cpf",)]
    assert rows(db, "SELECT label FROM episodes") == [("llr40-qwen38-c-cpf.n0.p0.w0",)]


def test_the_retired_final_stamp_and_the_mlscale_study_are_rewritten(tmp_path: pathlib.Path) -> None:
    db = v2_database(tmp_path / "r.db", (("mlscale20-qwen38-c", "mlscale", "mlscale20-qwen38-c"),))
    with connect(db) as conn:
        conn.execute("UPDATE grades SET timing_reduction = 'mw4x5-final-v2'")
    done = migrate("--dry-run", db)
    assert "grades.timing_reduction mw4x5-final-v2 -> mw4x5 (1 grades)" in done.stdout
    assert migrate(db).returncode == 0
    assert rows(db, "SELECT timing_reduction FROM grades") == [("mw4x5",)]
    assert rows(db, "SELECT study FROM setups") == [("mlscale20",)]


def test_a_corrupted_language_is_repaired_and_its_packet_kept(tmp_path: pathlib.Path) -> None:
    db = v2_database(tmp_path / "r.db", (("llr40-qwen38-c", "llr40", "llr40-qwen38-c"),))
    with connect(db) as conn:
        conn.execute("UPDATE setups SET language = 'triton-skills-clean', packet = ''")
    assert migrate(db).returncode == 0
    assert rows(db, "SELECT language, packet FROM setups") == [("triton", "lang-skills")]


def test_a_second_run_changes_nothing(tmp_path: pathlib.Path) -> None:
    db = v2_database(tmp_path / "r.db", (("llr40-qwen38-c", "llr40", "llr40-qwen38-c"),))
    assert migrate(db).returncode == 0
    migrated = db.read_bytes()
    again = migrate(db)
    assert again.returncode == 0 and "already migrated" in again.stdout
    assert db.read_bytes() == migrated


def test_verify_compares_a_migrated_copy_with_its_original(tmp_path: pathlib.Path) -> None:
    original = v2_database(tmp_path / "orig.db", (("llr40-qwen38-c", "llr40", "llr40-qwen38-c"),))
    copy = tmp_path / "copy.db"
    copy.write_bytes(original.read_bytes())
    assert migrate(copy).returncode == 0
    assert migrate("--verify", original, copy).returncode == 0
    with connect(copy) as conn:
        conn.execute("UPDATE grades SET status = 'incorrect'")
    assert migrate("--verify", original, copy).returncode == 2


def test_a_file_that_is_not_a_v2_results_database_is_refused(tmp_path: pathlib.Path) -> None:
    other = tmp_path / "other.db"
    with connect(other) as conn:
        conn.execute("CREATE TABLE t (x)")
    done = migrate(other)
    assert done.returncode == 2 and "REFUSED" in done.stdout


if __name__ == "__main__":
    import tempfile

    for test in (
        test_a_v2_database_migrates_with_every_row_kept,
        test_a_dry_run_prints_every_change_and_leaves_the_file_alone,
        test_an_old_setup_spelling_is_rewritten_in_setups_episodes_and_labels,
        test_two_spellings_of_one_setup_merge_into_one_row,
        test_a_merge_of_setups_that_differ_in_identity_is_refused_and_leaves_the_file_alone,
        test_the_clean_suffix_folds_and_a_kernel_graded_both_ways_is_listed,
        test_the_cpf_archive_setups_move_to_the_llr40_prefix,
        test_the_retired_final_stamp_and_the_mlscale_study_are_rewritten,
        test_a_corrupted_language_is_repaired_and_its_packet_kept,
        test_a_second_run_changes_nothing,
        test_verify_compares_a_migrated_copy_with_its_original,
        test_a_file_that_is_not_a_v2_results_database_is_refused,
    ):
        with tempfile.TemporaryDirectory() as directory:
            test(pathlib.Path(directory))
