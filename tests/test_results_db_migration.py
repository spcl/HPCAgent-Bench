# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The results-DB schema and :func:`recording.migrate` across the schema's history.

``tests/data/results_db_vintages.json`` holds the DDL each vintage of ``recording.py`` created,
rebuilt from git (comments stripped): a DB from before the ``runs`` table, one before ``packets``
(with the legacy ``host`` column), one before the per-cell tables, one whose ``scaling_curves``
carry no law, release-v0.1's own and the one after the first schema cleanup. Every fixture is
synthetic rows on that DDL, written the way the writers of the time wrote them.
"""

import contextlib
import hashlib
import json
import pathlib
import sqlite3
from typing import Any

import pytest

from hpcagent_bench import experiments, observations_extract
from hpcagent_bench.harness import recording

VINTAGES: dict[str, list[str]] = json.loads(
    (pathlib.Path(__file__).parent / "data" / "results_db_vintages.json").read_text()
)
PRE_RUNS = next(name for name in VINTAGES if name.startswith("pre-runs"))
MIGRATABLE = [name for name in VINTAGES if name != PRE_RUNS]
#: The richest vintage: every column any retirement has a condition on.
RELEASE = next(name for name in VINTAGES if name.startswith("release-v0.1"))
RUN_ID = "llr-focus40-qwen38-c.n0.p1.w2"
TS = 1_790_000_000_000
#: Columns no campaign writer filled: ``seed_nonce`` / ``request_id`` on ``calls`` and
#: ``scaling_efficiency`` never (git history of the writers), ``prompt_hash`` only under
#: ``hpcagent-bench --record``.
NEVER_WRITTEN = frozenset(
    {("calls", "seed_nonce"), ("calls", "request_id"), ("submissions", "scaling_efficiency")}
    | {(table, "prompt_hash") for table in ("submissions", "attempts", "calls")}
)
#: Tables no campaign writer filled (``completions`` had no writer; ``prompts`` only ``--record``).
EMPTY_TABLES = frozenset({"prompts", "completions"})
#: Synthetic values for columns whose type alone does not give a valid one. Row ``i`` is grade
#: ``TS + i``, whose curve is one law: ``LAWS[i]``, measured at P = 1 and P = 2.
LAWS = ("strong", "weak")
SPECIAL: dict[str, Any] = {"run_id": RUN_ID, "benchmark": "gemm", "ranks": 2, "mpi_ranks": 2}


def value(table: str, column: str, kind: str, row: int) -> object:
    """A deterministic synthetic value; NULL for a column no writer ever filled."""
    if (table, column) in NEVER_WRITTEN:
        return None
    if column == "ts":
        return TS + row
    if column in ("scaling_mode", "mpi_mode"):
        return LAWS[row]
    if (table, column) == ("submission_cells", "baseline_winner"):
        return value(table, "baseline", kind, row)
    if column in SPECIAL:
        return SPECIAL[column]
    match kind.upper():
        case "INTEGER":
            return row % 2
        case "REAL":
            return 1.5 + row
        case _:
            return f"{column}-{row}"


def build(path: pathlib.Path, vintage: str) -> None:
    """``path`` created by ``vintage``'s DDL, holding two synthetic rows per table."""
    with contextlib.closing(sqlite3.connect(path)) as conn:
        for ddl in VINTAGES[vintage]:
            conn.execute(ddl)
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        for table in (t for t in tables if t not in EMPTY_TABLES):
            columns = [(r[1], r[2]) for r in conn.execute(f"PRAGMA table_info({table})") if r[1] != "id"]
            for row in range(2):
                values = [value(table, name, kind, row) for name, kind in columns]
                if table in ("runs", "benchmarks", "prompts", "packets"):  # one row per natural key
                    values = [f"{v}-{row}" if isinstance(v, str) else v for v in values]
                conn.execute(
                    f"INSERT INTO {table} ({', '.join(n for n, _ in columns)}) VALUES ({', '.join('?' * len(columns))})",
                    values,
                )
        conn.commit()


def extracted(path: pathlib.Path, out: pathlib.Path) -> str:
    """The observations CSV the extractor writes for ``path`` (its own path blanked), as text.

    The CSV is the artifact every figure reads; it spells a NULL and a column the vintage lacks
    alike, as an empty cell."""
    db = observations_extract.Database(path, "root", path.parent, "job")
    rows = [{**row, "judge_db": ""} for row in observations_extract.read_db(db, "", frozenset(), 0).observations]
    assert rows, "the fixture yielded no observations"
    observations_extract.write_csv(out, observations_extract.OBSERVATION_FIELDS, rows)
    return out.read_text()


def identity_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    """What :func:`experiments.read_database` yields for ``path``."""
    return list(experiments.read_database(experiments.Database(path, "root", "job"), {}))


def table_info(path: pathlib.Path, table: str) -> list[tuple[Any, ...]]:
    with contextlib.closing(sqlite3.connect(path)) as conn:
        return list(conn.execute(f"PRAGMA table_info({table})"))


def names(path: pathlib.Path, kind: str) -> set[str]:
    with contextlib.closing(sqlite3.connect(path)) as conn:
        query = "SELECT name FROM sqlite_master WHERE type = ? AND name NOT LIKE 'sqlite_%'"
        return {row[0] for row in conn.execute(query, (kind,))}


@pytest.fixture(name="fresh")
def fresh_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "fresh.db"
    recording.connect(str(path)).close()
    return path


def test_a_fresh_db_holds_exactly_the_schema(fresh: pathlib.Path) -> None:
    assert names(fresh, "table") == set(recording.TABLES)
    assert names(fresh, "index") == set(recording.INDEXES)


def test_no_retired_column_or_table_is_created_fresh(fresh: pathlib.Path) -> None:
    for table, column in recording.RETIRED_COLUMNS:
        assert column not in {row[1] for row in table_info(fresh, table)}, (table, column)
    assert not names(fresh, "table") & set(recording.RETIRED_TABLES)


@pytest.mark.parametrize("vintage", MIGRATABLE)
def test_a_migrated_copy_has_a_fresh_dbs_columns_in_a_fresh_dbs_order(
    tmp_path: pathlib.Path, fresh: pathlib.Path, vintage: str
) -> None:
    """Columns the schema never named (``host``) may trail; nothing else differs."""
    old, out = tmp_path / "old.db", tmp_path / "out.db"
    build(old, vintage)
    recording.migrate(str(old), str(out))
    for table in recording.TABLES:
        want = table_info(fresh, table)
        assert table_info(out, table)[: len(want)] == want, table
        extra = {row[1] for row in table_info(out, table)[len(want) :]}
        assert extra <= {"host"}, (table, extra)
    assert names(out, "table") == set(recording.TABLES)
    assert names(out, "index") == set(recording.INDEXES)


@pytest.mark.parametrize("vintage", MIGRATABLE)
def test_migrating_never_changes_the_source(tmp_path: pathlib.Path, vintage: str) -> None:
    old = tmp_path / "old.db"
    build(old, vintage)
    before = hashlib.sha256(old.read_bytes()).hexdigest()
    recording.migrate(str(old), str(tmp_path / "out.db"))
    assert hashlib.sha256(old.read_bytes()).hexdigest() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["old.db", "out.db"]  # no sidecar left beside it


@pytest.mark.parametrize("vintage", MIGRATABLE)
def test_the_extractor_writes_the_same_csv_off_a_migrated_copy(tmp_path: pathlib.Path, vintage: str) -> None:
    """The paper's numbers come out of the extractor; a migration that moved one of them would
    silently change a figure."""
    old, out = tmp_path / "old.db", tmp_path / "out.db"
    build(old, vintage)
    recording.migrate(str(old), str(out))
    assert extracted(out, tmp_path / "after.csv") == extracted(old, tmp_path / "before.csv")


@pytest.mark.parametrize("vintage", MIGRATABLE)
def test_the_identity_reader_loses_only_retired_columns(tmp_path: pathlib.Path, vintage: str) -> None:
    """Every value :func:`experiments.read_database` returned survives, or is derivable
    (:data:`recording.SCALING_SUMMARY`); a column only the copy has reads NULL."""
    old, out = tmp_path / "old.db", tmp_path / "out.db"
    build(old, vintage)
    recording.migrate(str(old), str(out))
    before, after = identity_rows(old), identity_rows(out)
    assert len(after) == len(before) > 0
    retired = {column for _table, column in recording.RETIRED_COLUMNS}
    with contextlib.closing(sqlite3.connect(out)) as conn:
        summary = f"SELECT {', '.join(recording.SCALING_SUMMARY.values())} FROM submissions WHERE id = ?"
        for old_row, new_row in zip(before, after, strict=True):
            dropped = set(old_row) - set(new_row)
            assert dropped <= retired, dropped
            derived = dict(zip(recording.SCALING_SUMMARY, conn.execute(summary, (new_row["id"],)).fetchone()))
            lost = {c: old_row[c] for c in dropped if old_row[c] is not None}
            assert not lost or (new_row["record"] == "submissions" and lost == {c: derived[c] for c in lost}), lost
            assert {k: new_row[k] for k in old_row if k in new_row} == {k: old_row[k] for k in old_row if k in new_row}
            assert all(new_row[k] is None for k in set(new_row) - set(old_row))


def test_the_legacy_host_column_is_kept(tmp_path: pathlib.Path) -> None:
    """``host`` is the pre-``node`` spelling of the machine name; a reader may still look it up."""
    vintage = next(name for name in MIGRATABLE if "host" in " ".join(VINTAGES[name]))
    old, out = tmp_path / "old.db", tmp_path / "out.db"
    build(old, vintage)
    recording.migrate(str(old), str(out))
    with contextlib.closing(sqlite3.connect(out)) as conn:
        for table in ("submissions", "attempts", "calls"):
            assert {row[0] for row in conn.execute(f"SELECT host FROM {table}")} == {"host-0", "host-1"}, table


def test_a_db_from_before_the_runs_table_is_refused_and_nothing_is_written(tmp_path: pathlib.Path) -> None:
    """Its identity lives in the arm name; ``scripts/migrate_db.py`` derives it, this does not."""
    old, out = tmp_path / "old.db", tmp_path / "out.db"
    build(old, PRE_RUNS)
    with pytest.raises(ValueError, match="migrate_db.py"):
        recording.migrate(str(old), str(out))
    assert not out.exists()


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        ("UPDATE calls SET seed_nonce = 7", r"calls\.seed_nonce"),
        ("UPDATE calls SET prompt_hash = 'abc'", r"calls\.prompt_hash"),
        ("UPDATE submission_cells SET baseline_winner = 'numba'", r"submission_cells\.baseline_winner"),
        ("UPDATE submissions SET mpi_mode = 'weak'", r"submissions\.mpi_mode"),
        ("UPDATE submissions SET mpi_ranks = 64", r"submissions\.mpi_ranks"),
        ("UPDATE submissions SET scaling_efficiency = 0.9", r"submissions\.scaling_efficiency"),
        (
            "INSERT INTO prompts (hash, n_bytes, path, first_seen) VALUES ('h', 1, 'h.txt', 1)",
            r"prompts holds data",
        ),
        (
            "INSERT INTO completions (hash, run_id, ts, benchmark, round, n_bytes, path) VALUES ('h','r',1,'k',1,1,'p')",
            r"completions holds data",
        ),
    ],
)
def test_a_retired_value_that_is_not_recoverable_is_refused_and_nothing_is_written(
    tmp_path: pathlib.Path, change: str, refusal: str
) -> None:
    """Retiring a column is only lossless while it holds nothing, or only what the schema derives."""
    old, out = tmp_path / "old.db", tmp_path / "out.db"
    build(old, RELEASE)
    with contextlib.closing(sqlite3.connect(old)) as conn:
        conn.execute(change)
        conn.commit()
    with pytest.raises(ValueError, match=refusal):
        recording.migrate(str(old), str(out))
    assert not out.exists()


def test_a_curve_without_a_law_takes_its_grades_one_law(tmp_path: pathlib.Path) -> None:
    vintage = next(name for name in MIGRATABLE if name.startswith("law-less"))
    old, out = tmp_path / "old.db", tmp_path / "out.db"
    build(old, vintage)
    recording.migrate(str(old), str(out))
    with contextlib.closing(sqlite3.connect(out)) as conn:
        assert conn.execute("SELECT ts - ?, scaling_mode FROM scaling_curves ORDER BY ts", (TS,)).fetchall() == [
            (0, "strong"),
            (1, "weak"),
        ]


def test_migrate_never_overwrites_its_destination(tmp_path: pathlib.Path) -> None:
    old, out = tmp_path / "old.db", tmp_path / "out.db"
    build(old, RELEASE)
    out.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        recording.migrate(str(old), str(out))
    assert out.read_bytes() == b"keep"


def test_migrating_a_migrated_db_changes_nothing(tmp_path: pathlib.Path) -> None:
    old, once, twice = tmp_path / "old.db", tmp_path / "once.db", tmp_path / "twice.db"
    build(old, RELEASE)
    recording.migrate(str(old), str(once))
    recording.migrate(str(once), str(twice))
    assert extracted(twice, tmp_path / "twice.csv") == extracted(once, tmp_path / "once.csv")
    for table in recording.TABLES:
        assert table_info(twice, table) == table_info(once, table), table


@pytest.mark.parametrize("vintage", MIGRATABLE)
def test_a_judge_resuming_an_old_shard_still_records(tmp_path: pathlib.Path, vintage: str) -> None:
    """connect() only adds: the retired tables and columns stay, and an old ``submissions`` that
    foreign-keys to ``benchmarks`` takes a row for a kernel ``benchmarks`` never saw."""
    old = tmp_path / "old.db"
    build(old, vintage)
    tables = names(old, "table")
    with contextlib.closing(recording.connect(str(old))) as conn:
        conn.execute(
            "INSERT INTO submissions (run_id, ts, benchmark, preset, datatype, source_mode, baseline, speedup)"
            " VALUES ('new.n0.p0.w0', 1, 'kernel-not-in-benchmarks', 'XL', 'float64', 'restricted', 'c', 2.0)"
        )
        conn.commit()
    assert tables <= names(old, "table")
    assert {c for c, _ in recording.canonical_columns()["calls"]} <= {r[1] for r in table_info(old, "calls")}


def test_aggregating_an_old_shard_leaves_the_retired_table_behind(tmp_path: pathlib.Path) -> None:
    base = tmp_path / "hpcagent_bench.db"
    build(pathlib.Path(recording.shard_db_path(0, str(base))), RELEASE)
    recording.aggregate(str(base))
    assert names(base, "table") == set(recording.TABLES)
    with contextlib.closing(sqlite3.connect(base)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM submissions").fetchone() == (2,)


def test_aggregating_keeps_a_retired_table_that_holds_rows(tmp_path: pathlib.Path) -> None:
    """The aggregate is rebuilt from the shards; a ``--record`` shard's prompts must not vanish from it."""
    base = tmp_path / "hpcagent_bench.db"
    shard = pathlib.Path(recording.shard_db_path(0, str(base)))
    build(shard, RELEASE)
    with contextlib.closing(sqlite3.connect(shard)) as conn:
        conn.execute("INSERT INTO prompts (hash, n_bytes, path, first_seen) VALUES ('h', 1, 'h.txt', 1)")
        conn.commit()
    recording.aggregate(str(base))
    assert names(base, "table") == set(recording.TABLES) | {"prompts"}
