# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A run folder's RUN.json names its uid, every row the run writes carries it, and the folder is found by that uid
after it moved; a row from before results schema 40 reads as uid '0' until ``runs adopt`` gives its run one."""

import contextlib
import json
import pathlib
import shutil
import sqlite3
import tarfile

import pytest

from hpcagent_bench import run_layout
from hpcagent_bench.harness import results_db
from hpcagent_bench.results import run_identity
from hpcagent_bench.support.collect import canon_db
from tests import results_seed


def run_folder(root: pathlib.Path, experiment: str, run_id: str) -> pathlib.Path:
    """An experiment run root's job folder with a launch stamp, as ``run_cluster.sh`` leaves it."""
    folder = root / experiment / run_id
    folder.mkdir(parents=True)
    stamp = {"slurm_job_id": run_id, "run_id": run_id, "nodes": 2, "started_s": 1795700000}
    (folder / run_layout.LAUNCH_STAMP_NAME).write_text(json.dumps(stamp), encoding="utf-8")
    return folder


def uids(db: pathlib.Path, table: str) -> list[str]:
    """``run_uid`` of every row of ``table`` in ``db``."""
    with contextlib.closing(sqlite3.connect(db)) as conn:
        return [str(row[0]) for row in conn.execute(f"SELECT run_uid FROM {table}")]


def test_ensure_writes_one_record_and_a_second_job_in_the_folder_keeps_it(tmp_path: pathlib.Path) -> None:
    folder = run_folder(tmp_path, "cpf-llr40-20261127", "765700")
    first = run_identity.ensure(folder, {})
    again = run_identity.ensure(folder, {"SLURM_JOB_ID": "765800"})
    assert again == first
    assert (first.experiment, first.run_id, first.slurm_job_id, first.created_s) == (
        "cpf-llr40-20261127",
        "765700",
        "765700",
        1795700000,
    )
    assert run_identity.read(folder) == first
    assert first.run_uid not in ("", run_identity.UNKNOWN_UID)


def test_a_record_with_a_field_this_release_does_not_know_still_reads(tmp_path: pathlib.Path) -> None:
    """A later release adds fields with defaults; this one ignores what it does not know."""
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / run_layout.RUN_RECORD_NAME).write_text(json.dumps({"run_uid": "abc", "version": 2, "future": 1}))
    record = run_identity.read(folder)
    assert record is not None and record.run_uid == "abc" and record.version == 2


def test_rows_carry_the_uid_the_job_exports_and_0_outside_a_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "results.db"
    monkeypatch.delenv(run_identity.UID_ENV, raising=False)
    results_seed.score(db, "setup.n0.p1.w1", "gemm", 1, 2.0)
    monkeypatch.setenv(run_identity.UID_ENV, "f00d")
    results_seed.score(db, "setup.n0.p2.w1", "gemm", 2, 3.0)
    canon_db.record(
        db,
        [
            {
                "run": "e1",
                "column": "numba",
                "kernel": "gemm",
                "preset": "S",
                "datatype": "float64",
                "median_ms": 1.0,
                "status": "ok",
            }
        ],
    )
    assert sorted(uids(db, "episodes")) == ["0", "f00d"]
    assert sorted(uids(db, "grades")) == ["0", "f00d"]
    assert uids(db, "canon") == ["f00d"]


def test_a_moved_run_is_found_by_its_uid_and_a_stale_hint_is_not_trusted(tmp_path: pathlib.Path) -> None:
    folder = run_folder(tmp_path / "old", "rev-llr40-20261114", "750001")
    record = run_identity.ensure(folder, {})
    other = run_identity.ensure(run_folder(tmp_path / "old", "rev-llr40-20261114", "750002"), {})
    moved = tmp_path / "new" / "archive" / "rev-llr40-20261114" / "750001"
    moved.parent.mkdir(parents=True)
    folder.rename(moved)
    roots = [tmp_path / "new", tmp_path / "old"]
    assert run_identity.locate(record.run_uid, hint=folder, roots=roots) == moved
    stale = tmp_path / "old" / "rev-llr40-20261114" / "750002"
    assert run_identity.locate(record.run_uid, hint=stale, roots=roots) == moved
    assert run_identity.locate(other.run_uid, hint=stale, roots=roots) == stale
    assert run_identity.locate("no-such-uid", roots=roots) is None


def test_an_unknown_uid_is_found_only_through_its_hint(tmp_path: pathlib.Path) -> None:
    folder = run_folder(tmp_path, "llr40-20261114", "749000")
    assert run_identity.locate(run_identity.UNKNOWN_UID, hint=folder, roots=[tmp_path]) == folder
    assert run_identity.locate(run_identity.UNKNOWN_UID, hint=tmp_path / "gone", roots=[tmp_path]) is None


def test_the_search_stops_at_its_depth_and_skips_environments(tmp_path: pathlib.Path) -> None:
    deep = tmp_path / "a" / "b" / "c" / "d" / "e"
    deep.mkdir(parents=True)
    hidden = tmp_path / ".venv" / "run"
    hidden.mkdir(parents=True)
    for folder, uid in ((deep, "deep"), (hidden, "venv")):
        run_identity.write(folder, run_identity.RunRecord(run_uid=uid))
    assert run_identity.locate("deep", roots=[tmp_path], depth=4) is None
    assert run_identity.locate("deep", roots=[tmp_path], depth=5) == deep
    assert run_identity.locate("venv", roots=[tmp_path], depth=5) is None


def test_adopt_gives_old_runs_a_uid_and_fills_their_rows_and_a_pooled_db(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "rev-llr40-20261114"
    folder = run_folder(tmp_path, root.name, "750001")
    own = folder / run_layout.MERGED_DB_NAME
    results_seed.score(own, "setup.n0.p1.w1", "gemm", 1, 2.0)
    pooled = tmp_path / "canon.db"
    canon_db.record(
        pooled,
        [
            {
                "run": root.name,
                "column": "numba",
                "kernel": "gemm",
                "preset": "S",
                "datatype": "float64",
                "median_ms": 1.0,
                "status": "ok",
            },
            {
                "run": "another",
                "column": "numba",
                "kernel": "gemm",
                "preset": "S",
                "datatype": "float64",
                "median_ms": 1.0,
                "status": "ok",
            },
        ],
    )
    results_seed.as_version(own, 39)
    [(found, record, filled)] = run_identity.adopt([root], [pooled])
    assert found == folder and record.adopted and filled == 3
    assert uids(own, "episodes") == [record.run_uid] and uids(own, "grades") == [record.run_uid]
    assert sorted(uids(pooled, "canon")) == sorted([record.run_uid, run_identity.UNKNOWN_UID])
    with contextlib.closing(sqlite3.connect(own)) as conn:
        assert results_db.schema_version(conn) == results_db.SCHEMA_VERSION
    assert run_identity.adopt([folder]) == [(folder, record, 0)]


def test_the_cli_prints_the_folder_of_a_uid_and_refuses_one_it_cannot_find(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = run_folder(tmp_path, "cpf-scicomp40-20261123", "760000")
    assert run_identity.main(["ensure", str(folder)]) == 0
    uid = capsys.readouterr().out.strip()
    assert run_identity.main(["locate", uid, "--root", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == str(folder)
    assert run_identity.main(["locate", "missing", "--root", str(tmp_path)]) == 1


def test_a_packed_run_is_found_through_the_index_beside_its_tar(tmp_path: pathlib.Path) -> None:
    """A run packed with its record keeps that uid; one packed before the record existed gets a uid from the index,
    and indexing the tar again keeps it."""
    folder = run_folder(tmp_path / "runs", "llr40-blind-20261114", "748000")
    record = run_identity.ensure(folder, {})
    older = folder.parent / "747000"
    results_seed.score(older / run_layout.MERGED_DB_NAME, "setup.n0.p1.w1", "gemm", 1, 2.0)
    packed = tmp_path / "runs" / "llr40-blind-20261114.tar.xz"
    with tarfile.open(packed, "w:xz") as tar:
        tar.add(folder.parent, arcname=folder.parent.name)
    shutil.rmtree(folder.parent)
    assert run_identity.locate(record.run_uid, roots=[tmp_path]) is None
    assert run_identity.locate_archived(record.run_uid, roots=[tmp_path]) is None
    entries = run_identity.index_archive(packed)
    assert [(entry.run_uid, entry.member) for entry in entries][0] == (record.run_uid, "llr40-blind-20261114/748000")
    [adopted] = [entry for entry in entries if entry.member == "llr40-blind-20261114/747000"]
    assert adopted.run_uid not in (record.run_uid, run_identity.UNKNOWN_UID) and len(entries) == 2
    assert run_identity.locate_archived(record.run_uid, roots=[tmp_path]) == (packed, entries[0].member)
    assert run_identity.locate_archived(adopted.run_uid, roots=[tmp_path]) == (packed, adopted.member)
    assert run_identity.index_archive(packed) == entries
    assert len(run_identity.read_index(packed.parent)) == 2
