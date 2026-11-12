# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""hpcagent_bench.results.collect: copy-only collection of run roots and DBs, then verify and archive."""

import contextlib
import hashlib
import pathlib
import sqlite3
import tarfile
from collections.abc import Iterator

import pytest

from hpcagent_bench.harness import results_db, sandbox
from hpcagent_bench.results import collect
from hpcagent_bench.results.collect import DataSource
from tests.sqlite_closing import connect


def tree_digest(root: pathlib.Path) -> dict[str, str]:
    """Every file under ``root`` with its bytes' hash: what 'the source is untouched' compares.

    A WAL DB's ``-shm`` is skipped: it is SQLite's shared lock index, which every reader updates."""
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.name.endswith("-shm")
    }


@pytest.fixture
def sources(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[pathlib.Path]:
    """A runs root with one job (a WAL judge DB, metadata, an agent home) and a regrade dir."""
    monkeypatch.setenv("SCRATCH", str(tmp_path / "scratch"))
    src = tmp_path / "src"
    job = src / "runs" / "camp-1" / "900"
    (job / "judge" / "rank-0").mkdir(parents=True)
    conn = sqlite3.connect(job / "judge" / "rank-0" / "hpcagent_bench0.db")
    conn.execute("pragma journal_mode=wal")
    conn.execute("create table submissions (x)")
    conn.execute("insert into submissions values (42)")
    conn.commit()  # left open: the rows live in the -wal only, as in a running job
    (job / ".env").write_text("SETUP=a\n")
    (job / "agents" / "p0").mkdir(parents=True)
    (job / "agents" / "p0" / "tokens.json").write_text("{}")
    (job / "agents" / "p0" / "claude.log").write_text("transcript")
    (job / "home" / ".cache").mkdir(parents=True)
    (job / "home" / "x.json").write_text("{}")
    (job / "observations").mkdir()
    (job / "observations" / "curve.bin").write_bytes(b"\1")
    (src / "regrades" / "w1").mkdir(parents=True)
    with connect(src / "regrades" / "w1" / "regrade-cells-0.db") as c:
        c.execute("create table regrade_cells (x)")
    (src / "regrades" / "w1" / "worklist.jsonl").write_text("{}\n")
    yield src
    conn.close()


def roots(src: pathlib.Path) -> list[collect.Root]:
    return [
        collect.Root(kind=DataSource.RUNS, path=src / "runs"),
        collect.Root(kind=DataSource.DB, path=src / "regrades"),
    ]


@pytest.mark.parametrize(
    ("kind", "rel", "kept"),
    [
        (DataSource.RUNS, "900/judge/rank-0/hpcagent_bench0.db", True),
        (DataSource.RUNS, "900/.env", True),
        (DataSource.RUNS, "900/agents/p0/tokens.json", True),
        (DataSource.RUNS, "900/MERGE_FAILED", True),
        (DataSource.RUNS, "900/observations/curve.bin", True),
        (DataSource.RUNS, "900/agents/p0/claude.log", False),
        (DataSource.RUNS, "900/home/x.json", False),
        (DataSource.RUNS, "900/dacecache-cc/x.csv", False),
        (DataSource.DB, "w1/regrade-cells-0.db", True),
        (DataSource.DB, "w1/worklist.jsonl", False),
    ],
)
def test_wanted(kind: DataSource, rel: str, kept: bool) -> None:
    """Each source kind keeps its data files and DBs and skips transcripts, homes and build output."""
    assert collect.wanted(kind, pathlib.PurePosixPath(rel)) is kept


def test_copy_collects_every_kind_and_leaves_sources_untouched(sources: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """The copy holds the selected files under <kind>/<root name>, a WAL DB's committed rows, a
    checksum list that verifies, and the sources are byte-identical afterwards."""
    before = tree_digest(sources)
    out = tmp_path / "data"
    n = collect.copy(roots(sources), out, threads=2)
    assert tree_digest(sources) == before
    copied = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert copied >= {
        "runs/runs/camp-1/900/judge/rank-0/hpcagent_bench0.db",
        "runs/runs/camp-1/900/.env",
        "runs/runs/camp-1/900/agents/p0/tokens.json",
        "runs/runs/camp-1/900/observations/curve.bin",
        "db/regrades/w1/regrade-cells-0.db",
        "SHA256SUMS",
        "SOURCES.tsv",
        "env.sh",
        "COMMIT",
    }
    assert not any("home/" in c or c.endswith(("claude.log", "worklist.jsonl")) for c in copied)
    assert n == 5
    with connect(out / "runs/runs/camp-1/900/judge/rank-0/hpcagent_bench0.db") as conn:
        assert conn.execute("select x from submissions").fetchall() == [(42,)]
    assert collect.verify(out) == []
    listed = [line.split("\t") for line in (out / "SOURCES.tsv").read_text().splitlines()]
    assert [(DataSource.parse(row[0]), row[1]) for row in listed] == [(r.kind, str(r.dest)) for r in roots(sources)]
    env = (out / "env.sh").read_text()
    assert 'RUNS="$DATA/runs/runs"' in env


def test_verify_reports_tampering(sources: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """A changed, a missing and an unlisted file are each reported."""
    out = tmp_path / "data"
    collect.copy(roots(sources), out)
    (out / "runs/runs/camp-1/900/agents/p0/tokens.json").write_text("changed\n")
    (out / "runs/runs/camp-1/900/.env").unlink()
    (out / "extra.txt").write_text("x")
    assert sorted(collect.verify(out)) == [
        "checksum runs/runs/camp-1/900/agents/p0/tokens.json",
        "missing runs/runs/camp-1/900/.env",
        "not in SHA256SUMS: extra.txt",
    ]


def test_verify_reports_a_malformed_sums_line(sources: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """A cut checksum line is a problem to report: verify is what a damaged copy is checked with."""
    out = tmp_path / "data"
    collect.copy(roots(sources), out)
    sums = out / collect.SUMS
    sums.write_text(sums.read_text() + "truncated\n")
    assert collect.verify(out) == ["malformed SHA256SUMS line: truncated"]


@pytest.mark.parametrize("inside", ["runs/data", "regrades/data", "."])
def test_copy_refuses_an_output_overlapping_a_source(sources: pathlib.Path, inside: str) -> None:
    """An output inside, or wrapping, a source is refused before anything is written."""
    before = tree_digest(sources)
    with pytest.raises(SystemExit, match="overlaps source"):
        collect.copy(roots(sources), sources / inside)
    assert tree_digest(sources) == before


def test_copy_refuses_a_non_empty_output(sources: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Collecting into an old copy would mix two collections."""
    out = tmp_path / "data"
    out.mkdir()
    (out / "old").write_text("x")
    with pytest.raises(SystemExit, match="not empty"):
        collect.copy(roots(sources), out)


def test_copy_refuses_two_roots_with_one_destination(sources: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Two roots named alike under one kind would overwrite each other's files."""
    (tmp_path / "other" / "runs").mkdir(parents=True)
    twins = [
        collect.Root(kind=DataSource.RUNS, path=sources / "runs"),
        collect.Root(kind=DataSource.RUNS, path=tmp_path / "other" / "runs"),
    ]
    with pytest.raises(SystemExit, match="both land on"):
        collect.copy(twins, tmp_path / "data")


def test_archive_cli_verifies_then_tars_and_keeps_the_copy(sources: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """``collect copy`` then ``collect archive``: the tar holds the copy, and the copy stays."""
    out = tmp_path / "data"
    assert (
        collect.main(
            ["copy", "--out", str(out), "--runs", str(sources / "runs"), "--db-root", str(sources / "regrades")]
        )
        == 0
    )
    assert collect.main(["archive", str(out)]) == 0
    target = next(tmp_path.glob("data.tar.*"))
    if target.suffix == ".gz":
        with tarfile.open(target) as tar:
            assert "data/SHA256SUMS" in tar.getnames()
    assert collect.verify(out) == []


def test_archive_refuses_a_copy_that_does_not_verify(sources: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """A copy that fails verification is never archived."""
    out = tmp_path / "data"
    collect.copy(roots(sources), out)
    (out / "COMMIT").write_text("tampered\n")
    assert collect.main(["archive", str(out)]) == 1
    assert not list(tmp_path.glob("data.tar.*"))


#: The graded answer of :func:`graded_run`: a source and a prebuilt library, as the judge recorded them.
SOURCE = "void gemm(double *c) { c[0] = 1.0; }\n"
LIBRARY = "/shared/agent-0/libgemm.so"
#: The stamps every grade row carries.
STAMP = {"preset": "S", "datatype": "float64", "source_mode": "restricted", "task_hash": "test-task", "correct": 1}


def graded_run(runs: pathlib.Path) -> tuple[str, str]:
    """A job under ``runs`` whose judge shard holds one graded source and one graded prebuilt library (the library
    file in the job's shared folder, its only copy); returns their sha256."""
    job = runs / "camp-1" / "901"
    library = job / "shared" / "agent-0" / "libgemm.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"\x7fELF graded")
    library_hash = hashlib.sha256(library.read_bytes()).hexdigest()
    with contextlib.closing(results_db.open_db(job / "judge" / "rank-0" / "hpcagent_bench0.db")) as conn:
        results_db.ensure_setup(conn, results_db.Setup("s", "c", "cpu"))
        episode = results_db.ensure_episode(conn, "s", "s.n0.p0.w0", 901)
        grade, _ = results_db.add_grade(conn, episode, "gemm", "submit", ts_ms=1, values=STAMP)
        source_hash = results_db.store_source(conn, grade, "host", "c", SOURCE)
        values = STAMP | {"source_mode": "any", "library": LIBRARY, "library_hash": library_hash}
        results_db.add_grade(conn, episode, "gemm", "submit", ts_ms=2, values=values)
        conn.commit()
    return source_hash, library_hash


def copied_db(out: pathlib.Path) -> pathlib.Path:
    return out / "runs/runs/camp-1/901/judge/rank-0/hpcagent_bench0.db"


def copied_library(out: pathlib.Path) -> pathlib.Path:
    """The graded library's place in the copy: the job's shared folder, as in the run root."""
    return out / "runs/runs/camp-1/901/shared/agent-0/libgemm.so"


def test_copy_keeps_every_graded_source_and_library(sources: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """The collected copy alone gives back a graded answer by its hash: the source text from its DB, the prebuilt
    library (recorded at the judge's ``/shared`` path) from the job's shared folder in the copy, where a re-grade
    looks for it. A copy that lacked either would let the run root's removal lose the answer."""
    source_hash, library_hash = graded_run(sources / "runs")
    out = tmp_path / "data"
    collect.copy(roots(sources), out)
    with results_db.reading(copied_db(out), finished=True) as conn:
        text = conn.execute("SELECT text FROM sources WHERE hash = ?", (source_hash,)).fetchone()
    assert text is not None
    assert results_db.source_hash(text[0]) == source_hash
    assert hashlib.sha256(copied_library(out).read_bytes()).hexdigest() == library_hash
    assert sandbox.run_file(LIBRARY, copied_db(out)) == copied_library(out).resolve()
    assert collect.verify(out) == []


@pytest.mark.parametrize("damage", ["library", "source"])
def test_verify_refuses_a_copy_without_a_graded_answer(
    sources: pathlib.Path, tmp_path: pathlib.Path, damage: str
) -> None:
    """A copy whose checksum list is consistent but which lacks a graded library, or holds a source whose text is not
    the one its hash names, fails verify, so ``collect archive`` refuses it."""
    source_hash, library_hash = graded_run(sources / "runs")
    out = tmp_path / "data"
    collect.copy(roots(sources), out)
    if damage == "library":
        copied_library(out).unlink()
        expected = f"library {LIBRARY} ({library_hash}) of {copied_db(out).relative_to(out).as_posix()}"
    else:
        with contextlib.closing(sqlite3.connect(copied_db(out))) as conn, conn:
            conn.execute("UPDATE sources SET text = 'other' WHERE hash = ?", (source_hash,))
        expected = f"source {source_hash} of {copied_db(out).relative_to(out).as_posix()}"
    collect.write_sums(out)
    problems = collect.verify(out)
    assert len(problems) == 1
    assert problems[0].startswith(expected)
    assert collect.main(["archive", str(out)]) == 1
