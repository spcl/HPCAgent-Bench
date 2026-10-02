# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Several results databases read as one (``hpcagent_bench/stats/databases.py``): the core database
and, when asked for, the CPF archive, whose setups the core database leaves out."""

import contextlib
import pathlib
import shutil
import sqlite3

import pytest

from hpcagent_bench import dataset, observations_extract, paths
from hpcagent_bench.harness import results_db
from hpcagent_bench.stats import databases
from tests import results_seed

CORE_SETUP = "llr-focus40-qwen38-c"
CPF_SETUP = "cpf-llr-focus40-qwen38-c-cpfsrc-v2-clean"
KERNEL = "argmax_with_index"
#: When the seeded grades were recorded (after every reader's cut-off).
TS_MS = 1_790_000_000_000


def seed(db: pathlib.Path, setup: str, speedups: tuple[float, ...]) -> None:
    """One episode of ``setup`` submitting ``KERNEL`` once per speedup, and a final grade of the last."""
    last = 0
    for step, speedup in enumerate(speedups, start=1):
        last = results_seed.submission(
            db, f"{setup}.n0.p0.w0", KERNEL, TS_MS + step, speedup, job=7, timing_reduction="mw4x5"
        )
    with contextlib.closing(results_db.open_db(db)) as conn:
        run = conn.execute("SELECT episode_id FROM grades WHERE id = ?", (last,)).fetchone()[0]
        results_db.add_grade(
            conn, run, KERNEL, "final", ts_ms=TS_MS + 100, values={"of_grade_id": last, "speedup": 1.5}
        )
        conn.commit()


def counts(db: pathlib.Path) -> dict[str, int]:
    with contextlib.closing(sqlite3.connect(db)) as conn:
        return {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("setups", "episodes", "grades")
        }


def test_one_database_is_read_as_it_is(tmp_path: pathlib.Path) -> None:
    core = tmp_path / "core.db"
    seed(core, CORE_SETUP, (2.0,))
    with databases.union([core]) as db:
        assert db == core


def test_two_databases_are_unioned_with_their_ids_remapped(tmp_path: pathlib.Path) -> None:
    """Both files number their runs and grades from 1: merged by natural key, every grade keeps its
    run and a final grade keeps the grade it re-timed."""
    core, archive = tmp_path / "core.db", tmp_path / "cpf.db"
    seed(core, CORE_SETUP, (2.0, 3.0))
    seed(archive, CPF_SETUP, (4.0,))
    with databases.union([core, archive]) as db:
        assert counts(db) == {"setups": 2, "episodes": 2, "grades": 5}
        with contextlib.closing(sqlite3.connect(db)) as conn:
            finals = conn.execute(
                "SELECT r.setup, o.speedup FROM grades g JOIN grades o ON o.id = g.of_grade_id "
                "JOIN episodes r ON r.id = o.episode_id WHERE g.kind = 'final' ORDER BY r.setup"
            ).fetchall()
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert finals == [(CPF_SETUP, 4.0), (CORE_SETUP, 3.0)]
    assert not db.exists(), "the merge is temporary"


def test_a_setup_held_twice_with_different_rows_is_refused(tmp_path: pathlib.Path) -> None:
    core, other = tmp_path / "core.db", tmp_path / "other.db"
    seed(core, CORE_SETUP, (2.0,))
    seed(other, CORE_SETUP, (5.0,))
    with pytest.raises(databases.SetupConflict, match=CORE_SETUP), databases.union([core, other]):
        pass


def test_a_setup_held_twice_with_identical_rows_is_read_once(tmp_path: pathlib.Path) -> None:
    core = tmp_path / "core.db"
    seed(core, CORE_SETUP, (2.0,))
    copy = shutil.copy(core, tmp_path / "copy.db")
    with databases.union([core, pathlib.Path(copy)]) as db:
        assert counts(db) == counts(core)


def test_the_loader_reads_the_cpf_setups_only_when_the_archive_is_passed(tmp_path: pathlib.Path) -> None:
    core, archive = tmp_path / "core.db", tmp_path / "cpf.db"
    seed(core, CORE_SETUP, (2.0,))
    seed(archive, CPF_SETUP, (4.0,))
    alone, _ = dataset.build("llr40", tmp_path / "alone.db", frozen=None, root=tmp_path, dbs=(core,))
    both, _ = dataset.build("llr40", tmp_path / "both.db", frozen=None, root=tmp_path, dbs=(core, archive))
    assert set(alone["setup"]) == {CORE_SETUP}
    assert set(both["setup"]) == {CORE_SETUP, CPF_SETUP}


def test_the_extractor_refuses_two_named_databases_that_disagree_on_a_setup(tmp_path: pathlib.Path) -> None:
    core, other = tmp_path / "core.db", tmp_path / "other.db"
    seed(core, CORE_SETUP, (2.0,))
    seed(other, CORE_SETUP, (5.0,))
    options = observations_extract.Options(runs=(str(core), str(other)), benchmarks=paths.BENCHMARKS)
    with pytest.raises(databases.SetupConflict, match=CORE_SETUP):
        observations_extract.extract(options)
