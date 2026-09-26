# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""data_guard: no output may overlap a source, and no removal may reach a protected root or a DB."""

import importlib.util
import pathlib
import sqlite3
import sys

import pytest

from hpcagent_bench import data_guard, observations_extract

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def scratch(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """A scratch whose runs root is ``<tmp>/hpcagent-bench-runs`` and whose frozen dir is ``<tmp>/frozen``."""
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    monkeypatch.setenv("HPCAGENT_BENCH_FROZEN_OBSERVATIONS", str(tmp_path / "frozen"))
    monkeypatch.delenv(data_guard.ENV, raising=False)
    (tmp_path / "hpcagent-bench-runs" / "camp" / "123" / "judge").mkdir(parents=True)
    (tmp_path / "frozen").mkdir()
    return tmp_path


def make_db(path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("create table t (x)")
    return path


@pytest.mark.parametrize(
    ("dest", "refused"),
    [
        ("src", True),  # the source itself
        ("src/out", True),  # inside the source
        (".", True),  # an ancestor of the source
        ("out", False),  # a sibling
        ("src-out", False),  # a name sharing the prefix is not inside
    ],
)
def test_check_output_refuses_every_overlap(tmp_path: pathlib.Path, dest: str, refused: bool) -> None:
    """An output equal to, inside, or wrapping a source is refused; a sibling is returned resolved."""
    (tmp_path / "src").mkdir()
    if refused:
        with pytest.raises(data_guard.ProtectedPathError):
            data_guard.check_output(tmp_path / dest, [tmp_path / "src"])
    else:
        assert data_guard.check_output(tmp_path / dest, [tmp_path / "src"]) == (tmp_path / dest).resolve()


def test_check_output_refuses_a_source_file(tmp_path: pathlib.Path) -> None:
    """Rewriting an output that IS an input DB would delete the input."""
    db = make_db(tmp_path / "shard.db")
    with pytest.raises(data_guard.ProtectedPathError):
        data_guard.check_output(db, [db])


def test_protected_roots_are_runs_frozen_and_env(scratch: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The runs root, the frozen dir and every $HPCAGENT_BENCH_PROTECTED_ROOTS entry are protected."""
    monkeypatch.setenv(data_guard.ENV, f"{scratch / 'a'}:{scratch / 'b'}")
    assert data_guard.protected_roots() == tuple(
        (scratch / n).resolve() for n in ("hpcagent-bench-runs", "frozen", "a", "b")
    )


@pytest.mark.parametrize("victim", ["hpcagent-bench-runs", "hpcagent-bench-runs/camp", "frozen", "."])
def test_removal_refused_at_or_under_or_above_a_protected_root(scratch: pathlib.Path, victim: str) -> None:
    """A protected root, anything inside it, and any ancestor of it are never removed."""
    with pytest.raises(data_guard.ProtectedPathError):
        data_guard.safe_rmtree(scratch / victim)
    assert (scratch / "hpcagent-bench-runs" / "camp" / "123").is_dir()


def test_removal_allowed_under_an_allowlisted_scratch_dir(scratch: pathlib.Path) -> None:
    """A code snapshot under the runs root goes when its parent is allowlisted, and only then."""
    frozen_tree = scratch / "hpcagent-bench-runs" / ".frozen" / "job-1"
    (frozen_tree / "pkg").mkdir(parents=True)
    (frozen_tree / "pkg" / "a.py").write_text("x = 1\n")
    with pytest.raises(data_guard.ProtectedPathError):
        data_guard.safe_rmtree(frozen_tree)
    data_guard.safe_rmtree(frozen_tree, allow=[scratch / "hpcagent-bench-runs" / ".frozen"])
    assert not frozen_tree.exists()


def test_removal_refused_for_a_tree_holding_a_database(scratch: pathlib.Path) -> None:
    """Outside every protected root, a tree still keeps its DBs: it is refused whole."""
    tree = scratch / "canon-sweep"
    make_db(tree / "db" / "cc" / "hpcagent_bench0.db")
    with pytest.raises(data_guard.ProtectedPathError, match="SQLite"):
        data_guard.safe_rmtree(tree)
    assert (tree / "db" / "cc" / "hpcagent_bench0.db").is_file()


def test_removal_of_a_plain_tree(scratch: pathlib.Path) -> None:
    """A tree outside every root with no DB is removed."""
    tree = scratch / "build"
    (tree / "sub").mkdir(parents=True)
    (tree / "sub" / "x.o").write_bytes(b"\0")
    data_guard.safe_rmtree(tree)
    assert not tree.exists()


@pytest.mark.parametrize("out", ["camp/123/judge", "camp/123", "camp"])
def test_extract_refuses_an_unscanned_output_that_is_a_source_or_holds_a_judge_database(
    scratch: pathlib.Path, capsys: pytest.CaptureFixture, out: str
) -> None:
    """Skipping --out is no excuse to write into a judge database's directory (the skip would hide
    it), into a job directory the scan reads, or over the run root itself."""
    runs = scratch / "hpcagent-bench-runs"
    make_db(runs / "camp" / "123" / "judge" / "hpcagent_bench0.db")
    rc = observations_extract.main(
        ["--runs", str(runs / "camp"), "--benchmarks", str(scratch / "frozen"), "--out", str(runs / out)]
    )
    assert rc == 1
    assert "overlaps source" in capsys.readouterr().err


def test_extract_writes_a_jobs_own_record_inside_the_run_root_it_reads(scratch: pathlib.Path) -> None:
    """run_cluster.sh freezes each job's record into <job>/observations, inside the --runs it reads:
    allowed, because the scan skips its own --out."""
    runs = scratch / "hpcagent-bench-runs"
    job = runs / "camp" / "123"
    out = job / "observations"
    argv = ["--runs", str(job), "--benchmarks", str(scratch / "frozen"), "--out", str(out)]
    assert observations_extract.main([*argv, "--db", str(out / "observations.sqlite")]) == 0
    assert (out / "llr40_observations.csv").is_file() and (out / "observations.sqlite").is_file()
    # Again over its own earlier record, as a re-run of the extraction does.
    assert observations_extract.main([*argv, "--db", str(out / "observations.sqlite")]) == 0


def test_the_run_root_scan_skips_the_extractions_own_output(tmp_path: pathlib.Path) -> None:
    """A database under the output directory is never read back as a judge record."""
    judge = make_db(tmp_path / "camp" / "123" / "judge" / "hpcagent_bench0.db")
    make_db(tmp_path / "camp" / "123" / "observations" / "sources" / "copied.db")
    found = observations_extract.discover_databases(
        [str(tmp_path / "camp")], [tmp_path / "camp" / "123" / "observations"]
    )
    assert [db.path for db in found] == [judge.resolve()]


def test_merge_results_refuses_to_overwrite_a_shard(tmp_path: pathlib.Path) -> None:
    """experiments/merge_results.py rebuilds --out from scratch, so --out naming a shard is refused."""
    spec = importlib.util.spec_from_file_location("merge_results", REPO / "experiments" / "merge_results.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    shard = make_db(tmp_path / "judge" / "rank-0" / "hpcagent_bench0.db")
    with pytest.raises(SystemExit, match="one of the shards"):
        module.merge(tmp_path, shard)
    assert shard.is_file()
