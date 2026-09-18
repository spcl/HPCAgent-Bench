# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""scripts/migrate_canon_scratch_dirs.py -- the one-shot tool for the pre-existing
$SCRATCH/canon-*/smoke-* directories predating the cache-rooted work-dir convention.

Real `sacct` cannot run in a test sandbox, so every test monkeypatches ``sacct_lookup`` with a
canned (states, names) pair instead of shelling out -- the property under test is what this script
DOES with a given Slurm state, not whether `sacct`'s own output parses (that part is a two-line
`--parsable2` split with no logic of its own).
"""

import contextlib
import csv
import importlib.util
import pathlib
import sqlite3
import sys

import pytest

from hpcagent_bench import paths

SPEC = importlib.util.spec_from_file_location(
    "migrate_canon_scratch_dirs", paths.ROOT / "scripts" / "migrate_canon_scratch_dirs.py"
)
migrate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = migrate
SPEC.loader.exec_module(migrate)

ROW = {"kernel": "wf_triangular", "preset": "fuzzed", "datatype": "float64", "median_ms": "1.5", "validated": "True"}


def write_column(run_dir: pathlib.Path, column: str, jobid: str | None = None) -> None:
    """One CSV shard for ``column``, its DaCe build tree, and (if ``jobid`` is given) the
    ``<name>-<column>-<jobid>.out``/``.err`` pair a real canon job would have left, named exactly
    as ``submit-canon-llr40.sh``'s ``--job-name`` convention does."""
    shard = run_dir / f"{column}.rank0.csv"
    with shard.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["kernel", "preset", "datatype", "median_ms", "validated"])
        writer.writeheader()
        writer.writerow(ROW)
    (run_dir / f"dacecache-{column}").mkdir()
    if jobid is not None:
        (run_dir / f"canon-llr-{column}-{jobid}.out").write_text("")
        (run_dir / f"canon-llr-{column}-{jobid}.err").write_text("")


def read_canon_rows(db_path: pathlib.Path) -> list[tuple]:
    with contextlib.closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        return conn.execute("SELECT column, kernel FROM canon ORDER BY column").fetchall()


def test_a_completed_columns_job_is_merged_and_its_build_tree_cleared(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "canon-llr-cpu-old"
    run_dir.mkdir()
    write_column(run_dir, "numba", jobid="1")
    monkeypatch.setattr(migrate, "sacct_lookup", lambda ids: ({"1": "COMPLETED"}, {"1": "canon-llr-numba"}))
    db = tmp_path / "canon.db"
    archive = tmp_path / "archive"

    migrate.migrate_one(run_dir, db, archive, apply=True)

    assert not (run_dir / "dacecache-numba").exists()
    assert read_canon_rows(db) == [("numba", "wf_triangular")]


def test_a_column_whose_job_is_not_completed_keeps_every_one_of_its_files(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CANCELLED, FAILED, RUNNING, PENDING -- anything but exactly COMPLETED -- must not be merged
    or cleaned; this script makes no judgment call about a non-completed run's partial data."""
    run_dir = tmp_path / "canon-llr-cpu-old"
    run_dir.mkdir()
    write_column(run_dir, "cc", jobid="2")
    monkeypatch.setattr(migrate, "sacct_lookup", lambda ids: ({"2": "CANCELLED by 1000"}, {"2": "canon-llr-cc"}))
    db = tmp_path / "canon.db"
    archive = tmp_path / "archive"

    migrate.migrate_one(run_dir, db, archive, apply=True)

    assert (run_dir / "dacecache-cc").exists()
    assert (run_dir / "cc.rank0.csv").exists()
    assert not db.exists()


def test_a_directory_with_a_mix_of_states_merges_only_the_completed_columns_and_survives(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact shape observed in canon-llr-cpu-20260917-1314: some columns COMPLETED, others
    CANCELLED and resubmitted elsewhere. The directory must not be removed while any column in it
    is unresolved, even though its completed columns are still merged and cleared."""
    run_dir = tmp_path / "canon-llr-cpu-old"
    run_dir.mkdir()
    write_column(run_dir, "numba", jobid="1")
    write_column(run_dir, "cc", jobid="2")
    monkeypatch.setattr(
        migrate,
        "sacct_lookup",
        lambda ids: (
            {"1": "COMPLETED", "2": "CANCELLED by 1000"},
            {"1": "canon-llr-numba", "2": "canon-llr-cc"},
        ),
    )
    db = tmp_path / "canon.db"
    archive = tmp_path / "archive"

    migrate.migrate_one(run_dir, db, archive, apply=True)

    assert not (run_dir / "dacecache-numba").exists(), "the completed column must still be cleared"
    assert (run_dir / "dacecache-cc").exists(), "the cancelled column must be kept"
    assert run_dir.exists(), "the directory itself must survive while a column remains unresolved"
    assert read_canon_rows(db) == [("numba", "wf_triangular")]


def test_a_fully_completed_directory_is_archived_then_removed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once every column in a directory is COMPLETED and merged, the directory itself goes -- but
    its CSVs (the external reproducibility repos' own hand-off) must survive in the archive."""
    run_dir = tmp_path / "canon-llr-cpu-old"
    run_dir.mkdir()
    write_column(run_dir, "numba", jobid="1")
    write_column(run_dir, "cc", jobid="2")
    monkeypatch.setattr(
        migrate,
        "sacct_lookup",
        lambda ids: (
            {"1": "COMPLETED", "2": "COMPLETED"},
            {"1": "canon-llr-numba", "2": "canon-llr-cc"},
        ),
    )
    db = tmp_path / "canon.db"
    archive = tmp_path / "archive"

    migrate.migrate_one(run_dir, db, archive, apply=True)

    assert not run_dir.exists()
    assert (archive / "canon-llr-cpu-old" / "numba.rank0.csv").exists()
    assert (archive / "canon-llr-cpu-old" / "cc.rank0.csv").exists()
    assert sorted(read_canon_rows(db)) == [("cc", "wf_triangular"), ("numba", "wf_triangular")]


def test_a_column_with_no_matching_job_name_is_left_alone(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manual smoke invocation (smoke-canon-parallel*, smoke-optreports*) does not follow
    submit-canon-llr40.sh's `<prefix>-<column>` job-name convention; this script must recognize
    that it cannot identify the owning job rather than guess one."""
    run_dir = tmp_path / "smoke-canon-parallel-old"
    run_dir.mkdir()
    write_column(run_dir, "cc", jobid="9")
    (run_dir / "smoke-parallel-9.out").write_text("")  # a job file exists, but its NAME is not "*-cc"
    monkeypatch.setattr(migrate, "sacct_lookup", lambda ids: ({"9": "COMPLETED"}, {"9": "smoke-parallel"}))
    db = tmp_path / "canon.db"
    archive = tmp_path / "archive"

    migrate.migrate_one(run_dir, db, archive, apply=True)

    assert (run_dir / "dacecache-cc").exists()
    assert run_dir.exists()
    assert not db.exists()


def test_dry_run_performs_no_mutation_at_all(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The default (no --apply) is the safety net for a script meant to be run by hand once: it
    must be provably a no-op on disk regardless of what it decides the plan would be."""
    run_dir = tmp_path / "canon-llr-cpu-old"
    run_dir.mkdir()
    write_column(run_dir, "numba", jobid="1")
    before = (run_dir / "numba.rank0.csv").read_bytes()
    monkeypatch.setattr(migrate, "sacct_lookup", lambda ids: ({"1": "COMPLETED"}, {"1": "canon-llr-numba"}))
    db = tmp_path / "canon.db"
    archive = tmp_path / "archive"

    migrate.migrate_one(run_dir, db, archive, apply=False)

    assert (run_dir / "dacecache-numba").exists()
    assert (run_dir / "numba.rank0.csv").read_bytes() == before
    assert not db.exists()
    assert not archive.exists()
    assert run_dir.exists()
