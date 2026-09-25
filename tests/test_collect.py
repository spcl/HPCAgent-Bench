# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""hpcagent_bench.collect: copy-only collection of run roots, DBs and frozen CSVs, then verify and archive."""

import hashlib
import pathlib
import sqlite3
import tarfile
from collections.abc import Iterator

import pytest

from hpcagent_bench import collect
from hpcagent_bench.collect import DataSource


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
    """A runs root with one job (a WAL judge DB, metadata, an agent home), a regrade dir and a frozen dir."""
    monkeypatch.setenv("SCRATCH", str(tmp_path / "scratch"))
    src = tmp_path / "src"
    job = src / "runs" / "camp-1" / "900"
    (job / "judge" / "rank-0").mkdir(parents=True)
    conn = sqlite3.connect(job / "judge" / "rank-0" / "hpcagent_bench0.db")
    conn.execute("pragma journal_mode=wal")
    conn.execute("create table submissions (x)")
    conn.execute("insert into submissions values (42)")
    conn.commit()  # left open: the rows live in the -wal only, as in a running job
    (job / ".env").write_text("ARM=a\n")
    (job / "agents" / "p0").mkdir(parents=True)
    (job / "agents" / "p0" / "tokens.json").write_text("{}")
    (job / "agents" / "p0" / "claude.log").write_text("transcript")
    (job / "home" / ".cache").mkdir(parents=True)
    (job / "home" / "x.json").write_text("{}")
    (job / "observations").mkdir()
    (job / "observations" / "curve.bin").write_bytes(b"\1")
    (src / "regrades" / "w1").mkdir(parents=True)
    with sqlite3.connect(src / "regrades" / "w1" / "regrade-cells-0.db") as c:
        c.execute("create table regrade_cells (x)")
    (src / "regrades" / "w1" / "worklist.jsonl").write_text("{}\n")
    (src / "frozen" / "llr").mkdir(parents=True)
    (src / "frozen" / "llr" / "llr40_observations.csv").write_text("a,b\n1,2\n")
    (src / "frozen" / "llr" / "build.so").write_bytes(b"\0")
    yield src
    conn.close()


def roots(src: pathlib.Path) -> list[collect.Root]:
    return [
        collect.Root(kind=DataSource.RUNS, path=src / "runs"),
        collect.Root(kind=DataSource.DB, path=src / "regrades"),
        collect.Root(kind=DataSource.FROZEN_CSV, path=src / "frozen"),
    ]


@pytest.mark.parametrize(
    ("kind", "rel", "kept"),
    [
        (DataSource.RUNS, "900/judge/rank-0/hpcagent_bench0.db", True),
        (DataSource.RUNS, "900/.env", True),
        (DataSource.RUNS, "900/agents/p0/tokens.json", True),
        (DataSource.RUNS, "900/observations/curve.bin", True),
        (DataSource.RUNS, "900/agents/p0/claude.log", False),
        (DataSource.RUNS, "900/home/x.json", False),
        (DataSource.RUNS, "900/dacecache-cc/x.csv", False),
        (DataSource.DB, "w1/regrade-cells-0.db", True),
        (DataSource.DB, "w1/worklist.jsonl", False),
        (DataSource.FROZEN_CSV, "llr/llr40_observations.csv", True),
        (DataSource.FROZEN_CSV, "llr/build.so", False),
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
        "frozen-csv/frozen/llr/llr40_observations.csv",
        "SHA256SUMS",
        "SOURCES.tsv",
        "env.sh",
        "COMMIT",
    }
    assert not any("home/" in c or c.endswith(("claude.log", "build.so", "worklist.jsonl")) for c in copied)
    assert n == 6
    with sqlite3.connect(out / "runs/runs/camp-1/900/judge/rank-0/hpcagent_bench0.db") as conn:
        assert conn.execute("select x from submissions").fetchall() == [(42,)]
    assert collect.verify(out) == []
    env = (out / "env.sh").read_text()
    assert 'RUNS="$DATA/runs/runs"' in env
    assert 'HPCAGENT_BENCH_FROZEN_OBSERVATIONS="$DATA/frozen-csv/frozen"' in env


def test_verify_reports_tampering(sources: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """A changed, a missing and an unlisted file are each reported."""
    out = tmp_path / "data"
    collect.copy(roots(sources), out)
    (out / "frozen-csv/frozen/llr/llr40_observations.csv").write_text("changed\n")
    (out / "runs/runs/camp-1/900/.env").unlink()
    (out / "extra.txt").write_text("x")
    assert sorted(collect.verify(out)) == [
        "checksum frozen-csv/frozen/llr/llr40_observations.csv",
        "missing runs/runs/camp-1/900/.env",
        "not in SHA256SUMS: extra.txt",
    ]


@pytest.mark.parametrize("inside", ["runs/data", "frozen/data", "."])
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
            + ["--frozen-observations", str(sources / "frozen")]
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
