# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A run's identity: the uid every results row of the run carries, and the record its folder holds.

    hpcagent-bench runs ensure RUN_DIR             print the run's uid, writing its RUN.json first if it has none
    hpcagent-bench runs locate UID [--hint PATH]   print the folder of the run with that uid
    hpcagent-bench runs adopt ROOT... [--db DB]    give runs from before user_version 40 a uid and fill their rows

A run is one job directory, ``<runs root>/<experiment>-<stamp>/<run id>/`` (``run_cluster.sh``'s ``RUN_DIR``). Its
:data:`~hpcagent_bench.run_layout.RUN_RECORD_NAME` is a :class:`RunRecord`, written once by the first job that runs
there and never changed after. The job exports the uid as :data:`UID_ENV`, and every ``episodes``, ``grades`` and
``canon`` row it writes stores it in ``run_uid`` (results schema 40); :data:`UNKNOWN_UID` is the column's default,
so a row from before 40, or one written outside a run, reads as not known.

A database row names its run by uid, never by path, so moving or renaming a folder loses nothing: :func:`locate`
tries the path a caller still has (a row's ``run_root``, an old note), then searches the runs root of the
configuration, :data:`ROOTS_ENV`, ``$SCRATCH`` and ``$HOME`` for the record whose uid matches.
"""

import argparse
import contextlib
import os
import pathlib
import sqlite3
import sys
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence

import pydantic

from hpcagent_bench import experiments, run_layout
from hpcagent_bench.harness import results_db
from hpcagent_bench.results import data_guard, inventory

__all__ = [
    "ROOTS_ENV",
    "SEARCH_DEPTH",
    "SKIPPED_DIRS",
    "TABLES",
    "UID_ENV",
    "UNKNOWN_UID",
    "RunRecord",
    "adopt",
    "current_uid",
    "ensure",
    "fill_rows",
    "locate",
    "main",
    "read",
    "run_dbs",
    "search_roots",
]

#: ``run_uid`` of a row whose run is not known: the column's default, and what a row from before schema 40 holds.
UNKNOWN_UID = results_db.UNKNOWN_RUN_UID
#: The uid of the run a process writes for (``run_cluster.sh`` exports it; the container steps inherit it).
UID_ENV = results_db.RUN_UID_ENV
#: More folders :func:`locate` searches, ``os.pathsep`` separated, before ``$SCRATCH`` and ``$HOME``.
ROOTS_ENV = "HPCAGENT_BENCH_RUN_ROOTS"
#: The tables whose rows carry ``run_uid``.
TABLES = ("episodes", "grades", "canon")
#: How many folder levels below a search root :func:`locate` descends: ``$SCRATCH/x86_64/hpcagent-bench-runs/
#: <experiment>/<run id>`` is four.
SEARCH_DEPTH = 4
#: Folders a search never enters: environments, caches and version control, which hold no run and many files.
SKIPPED_DIRS = frozenset({
    ".git", ".venv", "venv", "node_modules", "site-packages", "__pycache__", ".cache", "spack", "spack-buildcache",
    ".dacecache", "agents", "shared", "judge", "episodes",
})


class RunRecord(pydantic.BaseModel):
    """What a run folder says about itself (:data:`~hpcagent_bench.run_layout.RUN_RECORD_NAME`)."""

    model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

    #: This record's own layout; a later field is added with a default, so an older record still reads.
    version: int = 1
    run_uid: str = pydantic.Field(min_length=1)
    #: The experiment run root the folder was created in (``<experiment>-<stamp>``) and its own name (the run id).
    experiment: str = ""
    run_id: str = ""
    slurm_job_id: str = ""
    created_s: int = 0
    #: True for a record ``runs adopt`` wrote into a folder from before the record existed.
    adopted: bool = False


def read(run_dir: pathlib.Path) -> RunRecord | None:
    """The record of ``run_dir``; None when it has none, or one that does not validate."""
    try:
        return RunRecord.model_validate_json((run_dir / run_layout.RUN_RECORD_NAME).read_bytes())
    except (OSError, pydantic.ValidationError):
        return None


def write(run_dir: pathlib.Path, record: RunRecord) -> None:
    """Write ``record`` into ``run_dir`` whole: to a temporary name, then renamed over."""
    target = run_dir / run_layout.RUN_RECORD_NAME
    staged = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    staged.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    staged.replace(target)


def new_record(run_dir: pathlib.Path, environ: Mapping[str, str], *, adopted: bool = False) -> RunRecord:
    """A record with a fresh uid for ``run_dir``: the Slurm job and run id from its launch stamp, else ``environ``."""
    stamp = inventory.launch_stamp(run_dir)
    job = stamp.slurm_job_id if stamp else environ.get("SLURM_JOB_ID", "")
    run_id = stamp.run_id if stamp else environ.get("HPCAGENT_BENCH_RUN_ID", "") or job
    created = stamp.started_s if stamp else int(time.time())
    return RunRecord(
        run_uid=uuid.uuid4().hex,
        experiment=run_dir.resolve().parent.name,
        run_id=run_id or run_dir.name,
        slurm_job_id=job,
        created_s=created,
        adopted=adopted,
    )


def ensure(run_dir: pathlib.Path, environ: Mapping[str, str] = os.environ, *, adopted: bool = False) -> RunRecord:
    """The record of ``run_dir``, written first when it has none (a job that runs in the folder again keeps it)."""
    record = read(run_dir)
    if record is None:
        record = new_record(run_dir, environ, adopted=adopted)
        write(run_dir, record)
    return record


def current_uid(environ: Mapping[str, str] = os.environ) -> str:
    """The uid of the run this process writes for; :data:`UNKNOWN_UID` outside one."""
    return results_db.run_uid(environ)


def search_roots(environ: Mapping[str, str] = os.environ) -> list[pathlib.Path]:
    """Where :func:`locate` looks, in order and once each: the configured runs root, :data:`ROOTS_ENV`, ``$SCRATCH``
    and ``$HOME``."""
    named = [experiments.runs_root(), *(pathlib.Path(one) for one in environ.get(ROOTS_ENV, "").split(os.pathsep) if one)]
    named += [pathlib.Path(environ[name]) for name in ("SCRATCH", "HOME") if environ.get(name)]
    roots: list[pathlib.Path] = []
    for root in named:
        if root.is_dir() and root.resolve() not in {seen.resolve() for seen in roots}:
            roots.append(root)
    return roots


def folders(root: pathlib.Path, depth: int) -> Iterator[pathlib.Path]:
    """``root`` and the folders below it, ``depth`` levels deep, breadth first, never into :data:`SKIPPED_DIRS`,
    a symlink or a folder that is a run (a run holds no other run)."""
    level = [root]
    for remaining in range(depth, -1, -1):
        below: list[pathlib.Path] = []
        for folder in level:
            yield folder
            if remaining == 0 or (folder / run_layout.RUN_RECORD_NAME).is_file():
                continue
            with contextlib.suppress(OSError):
                below += sorted(
                    child for child in folder.iterdir()
                    if child.name not in SKIPPED_DIRS and not child.is_symlink() and child.is_dir()
                )
        level = below


def locate(
    run_uid: str,
    hint: pathlib.Path | str | None = None,
    roots: Sequence[pathlib.Path] | None = None,
    depth: int = SEARCH_DEPTH,
) -> pathlib.Path | None:
    """The folder of the run ``run_uid``: ``hint`` when its record carries that uid, else the first folder of
    :func:`search_roots` (or ``roots``) whose record does; None when there is none. A run of :data:`UNKNOWN_UID`
    has no record to match: its ``hint`` when that folder exists, else None."""
    if hint is not None:
        hinted = pathlib.Path(hint)
        if run_uid == UNKNOWN_UID:
            return hinted if hinted.is_dir() else None
        record = read(hinted)
        if record is not None and record.run_uid == run_uid:
            return hinted
    if run_uid == UNKNOWN_UID:
        return None
    for root in search_roots() if roots is None else roots:
        for folder in folders(root, depth):
            record = read(folder)
            if record is not None and record.run_uid == run_uid:
                return folder
    return None


def run_dbs(run_dir: pathlib.Path) -> list[pathlib.Path]:
    """The results databases a run folder holds: its merged ``results.db`` and every judge rank's shards."""
    found = [run_dir / run_layout.MERGED_DB_NAME]
    found += sorted((run_dir / run_layout.JUDGE_DIRNAME).glob(f"{run_layout.RANK_DIR_PREFIX}*/*.db"))
    return [db for db in found if db.is_file()]


def fill_rows(db: pathlib.Path, run_uid: str, where: Mapping[str, str] | None = None) -> int:
    """Set ``run_uid`` on the rows of ``db`` that hold :data:`UNKNOWN_UID` (and, per table of ``where``, match its SQL
    condition, ``?`` bound to nothing); the file is upgraded to the current schema first. The number of rows set."""
    changed = 0
    with contextlib.closing(results_db.open_db(db, journal="DELETE")) as conn, conn:
        present = results_db.table_names(conn)
        for table in TABLES:
            if table not in present or (where is not None and table not in where):
                continue
            condition = f" AND ({where[table]})" if where is not None else ""
            cursor = conn.execute(f"UPDATE {table} SET run_uid = ? WHERE run_uid = ?{condition}", (run_uid, UNKNOWN_UID))
            changed += cursor.rowcount
    return changed


def is_run(folder: pathlib.Path) -> bool:
    """Whether ``folder`` is one job's run directory: it has a launch stamp, a merged database or judge shards."""
    return any((folder / name).exists() for name in (run_layout.LAUNCH_STAMP_NAME, run_layout.MERGED_DB_NAME,
                                                      run_layout.JUDGE_DIRNAME))


def adopt(roots: Sequence[pathlib.Path], dbs: Sequence[pathlib.Path] = ()) -> list[tuple[pathlib.Path, RunRecord, int]]:
    """Give every run under ``roots`` (a run folder, or an experiment root of them) a record, and fill the uid into the
    rows of its own databases that have none. Each of ``dbs`` (a database outside the runs, a pooled or a canon one)
    gets the uid of a run on the rows that name it: ``episodes``/``grades`` by its Slurm job, ``canon`` by its
    experiment when that experiment has this one run. Per run: its folder, its record and the rows it filled."""
    runs = [root for root in roots if is_run(root)]
    runs += [child for root in roots if not is_run(root) for child in sorted(root.iterdir())
             if child.is_dir() and is_run(child)]
    adopted: list[tuple[pathlib.Path, RunRecord, int]] = []
    for run_dir in runs:
        with data_guard.unsealed(run_dir):
            record = ensure(run_dir, {}, adopted=True)
            filled = sum(fill_rows(db, record.run_uid) for db in run_dbs(run_dir))
        adopted.append((run_dir, record, filled))
    per_experiment: dict[str, list[RunRecord]] = {}
    for _, record, _ in adopted:
        per_experiment.setdefault(record.experiment, []).append(record)
    for db in dbs:
        for index, (run_dir, record, filled) in enumerate(adopted):
            where: dict[str, str] = {}
            if record.run_id.isdigit():
                job = int(record.run_id)
                where["episodes"] = f"job = {job}"
                where["grades"] = f"episode_id IN (SELECT id FROM episodes WHERE job = {job})"
            if len(per_experiment[record.experiment]) == 1:
                where["canon"] = "run = " + sqlite_text(record.experiment)
            if where:
                adopted[index] = (run_dir, record, filled + fill_rows(db, record.run_uid, where))
    return adopted


def sqlite_text(value: str) -> str:
    """``value`` as an SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def build_parser() -> argparse.ArgumentParser:
    """The parser of the run-identity verbs of ``hpcagent-bench runs``."""
    parser = argparse.ArgumentParser(prog="hpcagent-bench runs", description=(__doc__ or "").split("\n\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    ensure_ = sub.add_parser("ensure", help="print a run folder's uid, writing its RUN.json first if it has none")
    ensure_.add_argument("run_dir", type=pathlib.Path)
    locate_ = sub.add_parser("locate", help="print the folder of the run with this uid")
    locate_.add_argument("run_uid")
    locate_.add_argument("--hint", type=pathlib.Path, help="a path the run may still be at (a row's run_root)")
    locate_.add_argument("--root", type=pathlib.Path, action="append", help="search only these folders; repeatable")
    adopt_ = sub.add_parser("adopt", help="give runs from before results schema 40 a uid and fill their rows")
    adopt_.add_argument("roots", type=pathlib.Path, nargs="+", help="run folders or experiment roots of them")
    adopt_.add_argument("--db", type=pathlib.Path, action="append", default=[],
                        help="a database outside the runs whose rows name them (pooled, canon); repeatable")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command line of the run-identity verbs: ensure, locate, adopt."""
    args = build_parser().parse_args(sys.argv[1:] if argv is None else list(argv))
    match args.command:
        case "ensure":
            print(ensure(args.run_dir).run_uid)
            return 0
        case "locate":
            found = locate(args.run_uid, args.hint, args.root)
            if found is None:
                print(f"runs locate: no run folder holds uid {args.run_uid}", file=sys.stderr)
                return 1
            print(found)
            return 0
        case _:
            try:
                adopted = adopt(args.roots, args.db)
            except (OSError, sqlite3.Error, results_db.SchemaVersionError) as failed:
                print(f"runs adopt: {failed}", file=sys.stderr)
                return 1
            for run_dir, record, filled in adopted:
                print(f"{record.run_uid} {run_dir} rows={filled}")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
