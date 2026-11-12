# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""hpcagent_bench.results.collect: copy-only collection of run roots and DBs, then verify and archive."""

import contextlib
import hashlib
import pathlib
import shutil
import sqlite3
import tarfile
from collections.abc import Iterator

import pytest

from hpcagent_bench.categories import SourceMode
from hpcagent_bench.harness import grade_under, results_db, sandbox
from hpcagent_bench.results import collect, convert_v0, data_guard
from hpcagent_bench.results.collect import DataSource
from tests.sqlite_closing import connect
from tests.test_convert_v0 import v0_shard


def tree_digest(root: pathlib.Path) -> dict[str, str]:
    """Every file under ``root`` with its bytes' hash: what 'the source is untouched' compares.

    A WAL DB's ``-shm`` is skipped: it is SQLite's shared lock index, which every reader updates; so is an empty
    ``-wal``, which a read-only reader of a closed WAL DB leaves behind."""
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.name.endswith("-shm") and not (p.name.endswith("-wal") and not p.stat().st_size)
    }


@pytest.fixture
def sources(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[pathlib.Path]:
    """A runs root with one job (a WAL judge DB, metadata, an episode's worker directory with its transcripts, logs,
    agent home and secrets, a job-level home) and a regrade dir."""
    monkeypatch.setenv("SCRATCH", str(tmp_path / "scratch"))
    src = tmp_path / "src"
    job = src / "runs" / "camp-1" / "900"
    (job / "judge" / "rank-0").mkdir(parents=True)
    conn = sqlite3.connect(job / "judge" / "rank-0" / "hpcagent_bench0.db")
    conn.execute("pragma journal_mode=wal")
    conn.execute("create table submissions (x)")
    conn.execute("insert into submissions values (42)")
    conn.commit()  # left open: the rows live in the -wal only, as in a running job
    for rel, text in (*TRANSCRIPTS.items(), *SECRETS.items(), *UNCOLLECTED.items()):
        (job / rel).parent.mkdir(parents=True, exist_ok=True)
        (job / rel).write_text(text)
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


#: An episode's worker directory, relative to its job.
WORKER = "agents/node-0/problem-0-worker-0"
#: What the fixture's episode recorded: its record, its harness transcript and session (a Claude Code subagent's
#: under the agent's home), a workflow's experiment log and file: every file of a transcript or log, whatever its type.
TRANSCRIPTS = {
    f"{WORKER}/tokens.json": "{}",
    f"{WORKER}/claude.log": '{"type": "result"}\n',
    f"{WORKER}/home/.claude/projects/p/session.jsonl": '{"role": "assistant"}\n',
    f"{WORKER}/.experiments/exp-1.c": "void k(void) {}\n",
    f"{WORKER}/results.tsv": "exp\tspeedup\n",
}
#: Files holding a key or a token, by name: never copied.
SECRETS = {
    ".env": "VLLM_API_KEY=planted\n",
    "episode-secret": "planted",
    "agents/node-0/litellm.yaml": "api_key: planted\n",
    f"{WORKER}/mcp.json": '{"env": {"HPCAGENT_BENCH_EPISODE_TOKEN": "planted"}}',
    f"{WORKER}/home/.claude.json": '{"oauthAccount": "planted"}',
    f"{WORKER}/home/.claude/.credentials.json": '{"token": "planted"}',
    f"{WORKER}/OPENAI_API_KEY.txt": "planted",
}
#: What a worker directory holds that is no transcript: the agent's home caches, an environment.
UNCOLLECTED = {
    f"{WORKER}/home/.debug/usr/lib.so.debug": "symbols",
    f"{WORKER}/home/.bash_history": "ls\n",
    f"{WORKER}/.venv/lib/site.py": "",
}


def roots(src: pathlib.Path) -> list[collect.Root]:
    return [
        collect.Root(kind=DataSource.RUNS, path=src / "runs"),
        collect.Root(kind=DataSource.DB, path=src / "regrades"),
    ]


@pytest.mark.parametrize(
    ("kind", "rel", "kept"),
    [
        (DataSource.RUNS, "900/judge/rank-0/hpcagent_bench0.db", True),
        (DataSource.RUNS, "900/MERGE_FAILED", True),
        (DataSource.RUNS, "900/observations/curve.bin", True),
        *((DataSource.RUNS, f"900/{rel}", True) for rel in TRANSCRIPTS),
        *((DataSource.RUNS, f"900/{rel}", False) for rel in (*SECRETS, *UNCOLLECTED)),
        (DataSource.RUNS, "900/agents/node-0/notes.txt", False),
        (DataSource.RUNS, "900/home/x.json", False),
        (DataSource.RUNS, "900/dacecache-cc/x.csv", False),
        (DataSource.DB, "w1/regrade-cells-0.db", True),
        (DataSource.DB, "w1/worklist.jsonl", False),
    ],
)
def test_wanted(kind: DataSource, rel: str, kept: bool) -> None:
    """Each source kind keeps its data files, DBs and every transcript and log of an episode, and skips secrets,
    homes, environments and build output."""
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
        *(f"runs/runs/camp-1/900/{rel}" for rel in TRANSCRIPTS),
        "runs/runs/camp-1/900/observations/curve.bin",
        "db/regrades/w1/regrade-cells-0.db",
        "SHA256SUMS",
        "SOURCES.tsv",
        "env.sh",
        "COMMIT",
    }
    assert not any("900/home/" in c or c.endswith("worklist.jsonl") for c in copied)
    assert n == 3 + len(TRANSCRIPTS)
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
    (out / f"runs/runs/camp-1/900/{WORKER}/tokens.json").write_text("changed\n")
    (out / "runs/runs/camp-1/900/observations/curve.bin").unlink()
    (out / "extra.txt").write_text("x")
    assert sorted(collect.verify(out)) == [
        f"checksum runs/runs/camp-1/900/{WORKER}/tokens.json",
        "missing runs/runs/camp-1/900/observations/curve.bin",
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


#: The graded answers of :func:`graded_run`: a source, a prebuilt library, and a source linking the agent's own
#: libraries, as the judge recorded them.
SOURCE = "void gemm(double *c) { c[0] = 1.0; }\n"
LIBRARY = "/shared/agent-0/libgemm.so"
AGENT_ROOT = "/shared/agent-0"
#: The stamps every grade row carries.
STAMP = {"preset": "S", "datatype": "float64", "source_mode": "restricted", "task_hash": "test-task", "correct": 1}


def graded_run(runs: pathlib.Path) -> tuple[str, str, str]:
    """A job under ``runs`` whose judge shard holds a graded source, a graded prebuilt library and a graded source
    that linked the agent's own ``lib``/``include`` (the files in the job's shared folder, their only copy), all of
    one episode with its transcript in its worker directory; returns the source's and the library's sha256 and the
    agent folder's digest."""
    job = runs / "camp-1" / "901"
    (job / WORKER).mkdir(parents=True)
    (job / WORKER / "claude.log").write_text('{"type": "result"}\n')
    agent = job / "shared" / "agent-0"
    for name, data in (
        ("libgemm.so", b"\x7fELF graded"),
        ("lib/libtuned.so", b"\x7fELF agent"),
        ("include/tuned.h", b""),
    ):
        (agent / name).parent.mkdir(parents=True, exist_ok=True)
        (agent / name).write_bytes(data)
    library_hash = hashlib.sha256((agent / "libgemm.so").read_bytes()).hexdigest()
    agent_libs = sandbox.library_digest(agent)
    with contextlib.closing(results_db.open_db(job / "judge" / "rank-0" / "hpcagent_bench0.db")) as conn:
        results_db.ensure_setup(conn, results_db.Setup("s", "c", "cpu"))
        episode = results_db.ensure_episode(conn, "s", "s.n0.p0.w0", 901)
        grade, _ = results_db.add_grade(conn, episode, "gemm", "submit", ts_ms=1, values=STAMP)
        source_hash = results_db.store_source(conn, grade, "host", "c", SOURCE)
        values = STAMP | {"source_mode": "any", "library": LIBRARY, "library_hash": library_hash}
        results_db.add_grade(conn, episode, "gemm", "submit", ts_ms=2, values=values)
        linked = STAMP | {"agent_root": AGENT_ROOT, "agent_libs": agent_libs}
        grade, _ = results_db.add_grade(conn, episode, "gemm", "submit", ts_ms=3, values=linked)
        results_db.store_source(conn, grade, "host", "c", SOURCE)
        conn.commit()
    return source_hash, library_hash, agent_libs


def copied_db(out: pathlib.Path) -> pathlib.Path:
    return out / "runs/runs/camp-1/901/judge/rank-0/hpcagent_bench0.db"


def copied_library(out: pathlib.Path) -> pathlib.Path:
    """The graded library's place in the copy: the job's shared folder, as in the run root."""
    return out / "runs/runs/camp-1/901/shared/agent-0/libgemm.so"


def copied_agent(out: pathlib.Path) -> pathlib.Path:
    """The agent folder's place in the copy: the job's shared folder, as in the run root."""
    return out / "runs/runs/camp-1/901/shared/agent-0"


def test_copy_keeps_every_graded_source_and_library(sources: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """The collected copy alone gives back a graded answer by its hash: the source text from its DB, the prebuilt
    library (recorded at the judge's ``/shared`` path) from the job's shared folder in the copy, where a re-grade
    looks for it. A copy that lacked either would let the run root's removal lose the answer."""
    source_hash, library_hash, _ = graded_run(sources / "runs")
    out = tmp_path / "data"
    collect.copy(roots(sources), out)
    with results_db.reading(copied_db(out), finished=True) as conn:
        text = conn.execute("SELECT text FROM sources WHERE hash = ?", (source_hash,)).fetchone()
    assert text is not None
    assert results_db.source_hash(text[0]) == source_hash
    assert hashlib.sha256(copied_library(out).read_bytes()).hexdigest() == library_hash
    assert sandbox.run_path(LIBRARY, copied_db(out)) == copied_library(out).resolve()
    assert collect.verify(out) == []


def test_a_regrade_of_the_copy_links_the_agent_libraries_the_copy_holds(
    sources: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A graded source that linked the agent's own ``lib``/``include`` is re-graded from the copy alone: the copy holds
    the agent folder at its place in the job's shared folder, unchanged (the digest the grade recorded), and the
    re-grade's envelope links from there, not from the judge's ``/shared`` path or the run root."""
    _, _, agent_libs = graded_run(sources / "runs")
    out = tmp_path / "data"
    collect.copy(roots(sources), out)
    held = sorted(p.relative_to(copied_agent(out)).as_posix() for p in copied_agent(out).rglob("*") if p.is_file())
    assert held == ["include/tuned.h", "lib/libtuned.so", "libgemm.so"]
    with results_db.reading(copied_db(out), finished=True) as conn:
        (grade_id,) = conn.execute("SELECT id FROM grades WHERE agent_libs != ''").fetchone()
    item = grade_under.Item(
        db=str(copied_db(out)),
        grade_id=grade_id,
        episode_id="s.n0.p0.w0",
        kernel="gemm",
        ts_ms=3,
        setup="s",
        language="c",
        source_mode=SourceMode.RESTRICTED,
        final=True,
        env={},
        agent_root=AGENT_ROOT,
        agent_libs=agent_libs,
    )
    assert grade_under.submission_of(item).agent_root == str(copied_agent(out).resolve())
    assert collect.verify(out) == []


@pytest.mark.parametrize("damage", ["library", "source", "agent", "transcript"])
def test_verify_refuses_a_copy_without_a_graded_answer(
    sources: pathlib.Path, tmp_path: pathlib.Path, damage: str
) -> None:
    """A copy whose checksum list is consistent but which lacks a graded library, the agent libraries a graded source
    linked or the transcript of a recorded episode, or holds a source whose text is not the one its hash names, fails
    verify, so ``collect archive`` refuses it."""
    source_hash, library_hash, _ = graded_run(sources / "runs")
    out = tmp_path / "data"
    collect.copy(roots(sources), out)
    rel = copied_db(out).relative_to(out).as_posix()
    if damage == "library":
        copied_library(out).unlink()
        expected = f"library {LIBRARY} of {rel}: not in the copy with sha256 {library_hash}"
    elif damage == "agent":
        for sub in ("lib", "include"):
            shutil.rmtree(copied_agent(out) / sub)
        expected = f"agent libraries {AGENT_ROOT} of {rel}: not in the copy"
    elif damage == "transcript":
        shutil.rmtree(out / "runs/runs/camp-1/901" / WORKER)
        expected = f"transcript of an episode of {rel}: runs/runs/camp-1/901/{WORKER} not in the copy"
    else:
        with contextlib.closing(sqlite3.connect(copied_db(out))) as conn, conn:
            conn.execute("UPDATE sources SET text = 'other' WHERE hash = ?", (source_hash,))
        expected = f"source {source_hash} of {rel}"
    collect.write_sums(out)
    problems = collect.verify(out)
    assert len(problems) == 1
    assert problems[0].startswith(expected)
    assert collect.main(["archive", str(out)]) == 1


def shard_rel(shard: pathlib.Path, src: pathlib.Path) -> str:
    """Where the copy keeps ``shard`` of the runs root under ``src``."""
    return f"runs/{shard.relative_to(src).as_posix()}"


def test_a_pre_release_shard_verifies_only_with_every_stored_source(
    sources: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A pre-release shard keeps its sources outside the DB (``<stem>_prompts/``): the copy carries them, verifies
    with all of them, and refuses (so ``collect archive`` does) once one is missing from it."""
    shard = v0_shard(sources)
    with contextlib.closing(sqlite3.connect(shard)) as conn, conn:
        conn.execute("DELETE FROM sources WHERE path = '00/lost.txt'")  # no text hashes to this key
    out = tmp_path / "data"
    collect.copy(roots(sources), out)
    assert collect.verify(out) == []
    with contextlib.closing(sqlite3.connect(shard)) as conn:
        digest, path = conn.execute("SELECT hash, path FROM sources ORDER BY id").fetchone()
    (convert_v0.store_dir(out / shard_rel(shard, sources)) / path).unlink()
    collect.write_sums(out)
    assert collect.verify(out) == [
        f"pre-release source {digest} of {shard_rel(shard, sources)}: not in the copy with sha256 {digest}"
    ]
    assert collect.main(["archive", str(out)]) == 1


def test_a_pre_release_shard_whose_sources_are_elsewhere_verifies_once_converted(
    sources: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A pre-release shard whose store lost a text is refused, and the same collection with the shard converted
    (``convert_v0`` finding the text by sha256 where it survives) into a collected DB verifies: the copy holds every
    graded source by its hash."""
    shard = v0_shard(sources)
    elsewhere = tmp_path / "kept"
    with contextlib.closing(sqlite3.connect(shard)) as conn, conn:
        conn.execute("DELETE FROM sources WHERE path = '00/lost.txt'")
        digest, path = conn.execute("SELECT hash, path FROM sources ORDER BY id").fetchone()
    elsewhere.mkdir()
    shutil.move(convert_v0.store_dir(shard) / path, elsewhere / "host.txt")
    collect.copy(roots(sources), tmp_path / "refused")
    assert collect.verify(tmp_path / "refused") == [
        f"pre-release source {digest} of {shard_rel(shard, sources)}: not in the copy with sha256 {digest}"
    ]

    convert_v0.convert(shard, sources / "regrades" / "converted" / "640146.db", convert_v0.index_texts([elsewhere]))
    out = tmp_path / "data"
    collect.copy(roots(sources), out)
    assert collect.verify(out) == []


def test_archive_removes_the_sources_only_through_remove_sources_after_every_check(
    sources: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """``collect archive --remove-sources`` is the one removal of run data: it lifts a finished job's seal and removes
    the collected roots once the copy verifies, the tar matches it and every root is in it as it is now; the copy
    and the tar then still give back every graded answer."""
    source_hash, _, _ = graded_run(sources / "runs")
    data_guard.seal(sources / "runs" / "camp-1" / "901", writable=False)
    out = tmp_path / "data"
    collect.copy(roots(sources), out)
    assert collect.main(["archive", "--remove-sources", str(out)]) == 0
    assert not (sources / "runs").exists()
    assert not (sources / "regrades").exists()
    assert collect.verify(out) == []
    with results_db.reading(copied_db(out), finished=True) as conn:
        assert conn.execute("SELECT 1 FROM sources WHERE hash = ?", (source_hash,)).fetchone() is not None
    assert next(tmp_path.glob("data.tar.*")).is_file()


def test_archive_removes_nothing_when_a_root_holds_a_grade_the_copy_lacks(
    sources: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A grade recorded after the copy (a judge DB that went on writing) is only in the root: removal is refused and
    every root stays, though the copy itself verifies."""
    graded_run(sources / "runs")
    out = tmp_path / "data"
    collect.copy(roots(sources), out)
    with contextlib.closing(results_db.open_db(sources / "runs/camp-1/901/judge/rank-0/hpcagent_bench0.db")) as conn:
        episode = results_db.ensure_episode(conn, "s", "s.n0.p0.w0", 901)
        grade, _ = results_db.add_grade(conn, episode, "gemm", "submit", ts_ms=4, values=STAMP)
        results_db.store_source(conn, grade, "host", "c", "a later answer")
        conn.commit()
    before = tree_digest(sources)
    with pytest.raises(SystemExit, match="nothing removed"):
        collect.main(["archive", "--remove-sources", str(out)])
    assert tree_digest(sources) == before


def test_archive_never_removes_a_protected_root_itself(
    sources: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A protected root as a whole (the experiments' runs root) is refused even with a passing copy."""
    monkeypatch.setenv(data_guard.ENV, str(sources / "runs"))
    out = tmp_path / "data"
    collect.copy(roots(sources), out)
    before = tree_digest(sources)
    with pytest.raises(SystemExit, match="protected root"):
        collect.main(["archive", "--remove-sources", str(out)])
    assert tree_digest(sources) == before


def test_copy_keeps_every_transcript_and_log_and_no_secret(sources: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Every file of an episode's worker directory is in the copy with its bytes, whatever its type (a harness log, a
    Claude Code session under the agent's home, a workflow's experiment log and file); no file named as a secret is,
    nor the agent's home caches or an environment; the copy verifies."""
    out = tmp_path / "data"
    collect.copy(roots(sources), out)
    job, kept = sources / "runs/camp-1/900", out / "runs/runs/camp-1/900"
    assert {rel: (kept / rel).read_bytes() for rel in TRANSCRIPTS} == {
        rel: (job / rel).read_bytes() for rel in TRANSCRIPTS
    }
    assert [rel for rel in (*SECRETS, *UNCOLLECTED) if (kept / rel).exists()] == []
    assert not any(b"planted" in p.read_bytes() for p in out.rglob("*") if p.is_file())
    assert collect.verify(out) == []


def test_archive_removes_nothing_while_a_transcript_in_a_root_is_not_in_the_copy(
    sources: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A transcript that went on growing after the copy (an agent still writing) is only whole in the root: removal is
    refused and every root stays, though the copy itself verifies."""
    graded_run(sources / "runs")
    out = tmp_path / "data"
    collect.copy(roots(sources), out)
    with (sources / "runs/camp-1/901" / WORKER / "claude.log").open("a") as log:
        log.write('{"type": "assistant"}\n')
    before = tree_digest(sources)
    with pytest.raises(SystemExit, match="nothing removed"):
        collect.main(["archive", "--remove-sources", str(out)])
    assert tree_digest(sources) == before
