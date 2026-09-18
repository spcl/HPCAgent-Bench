#!/usr/bin/env python3
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""ONE-SHOT migration of pre-existing canon/smoke work dirs into the cache-rooted convention.

Before ``experiments/canon_column.sh``/``submit-canon-llr40.sh`` learned to derive their work dir
under ``${HPCAGENT_BENCH_RUNS_ROOT}`` (see ``.cache/README.md``'s "Job work dirs" section), a
campaign's ``OUT_ROOT`` was a bare ``${SCRATCH}/<name>`` directory nothing ever swept: it still
holds one DaCe build tree (``dacecache-<column>[_rank<N>]``) per column, forever. This script folds
each such directory's per-column CSV rows into the persistent ``canon.db`` and reclaims that build
tree -- the same verified-merge-then-delete contract ``canon_column.sh``'s own in-job finalize step
now uses -- for directories THIS repo already stopped writing to.

**Granularity is per COLUMN, not per directory.** Several columns of one campaign share one
``out_root`` (``submit-canon-llr40.sh`` submits one Slurm job per column into the same
``OUT_ROOT``), and columns finish, get cancelled, or get resubmitted independently -- observed on
2026-09-17 in ``canon-llr-cpu-20260917-1314``: ``numba`` (640106) and the three ``dace_cpu*``
columns (640115-640117) COMPLETED while ``cc``/``cc_autopar``/``cpp``/``fortran``/
``fortran_autopar``/``pluto`` (640107-640110, 640112, 640114) show CANCELLED, having been
resubmitted into a sibling directory. A column is identified as ``run_dir``'s glob of
``<column>.rank*.csv`` (the same discovery ``scripts/collect_canon.py`` uses); the Slurm job that
wrote it is found by NAME (``sacct``'s ``JobName`` ending in ``-<column>``, matching
``submit-canon-llr40.sh``'s ``--job-name="${JOB_PREFIX}-${JOB_TAG:-${col%%,*}}"``), and that job's
OWN state gates that column alone.

**"COMPLETED" only, literally** -- not CANCELLED, FAILED, TIMEOUT, or anything else terminal. A
column whose job is not exactly COMPLETED is left completely untouched, files and all: this script
makes no judgment call about whether a cancelled or failed run's partial CSV is worth keeping.
Broadening this is a decision for whoever runs it, not something to guess here.

**A directory is removed WHOLESALE only once every column found in it has been merged** (no
un-COMPLETED column remains) -- the literal ask ("removes the dir"). Its CSVs and
``reports/<column>/`` are archived to ``<archive-dir>/<directory name>/`` FIRST, because the
external reproducibility repos' own ``collect_canon.py`` pass (``experiments/README.md``'s canon
section) reads those CSVs from a live ``out_root`` and this script must not be the reason that
hand-off becomes impossible after the fact. A directory with at least one un-COMPLETED (or
unmatched) column keeps its own CSVs/reports where they are and is NOT removed, even though the
COMPLETED columns in it still get merged and cleared.

Read-only unless ``--apply`` is given: without it, every action below is PRINTED, nothing is
written, merged, moved, or deleted. This script is not wired into any submitter or CI job and is
meant to be run by hand, once, after checking the printed plan.

    . scripts/cache_env.sh                     # HPCAGENT_BENCH_RESULTS_DIR
    python3 scripts/migrate_canon_scratch_dirs.py [DIR...]            # dry run (default)
    python3 scripts/migrate_canon_scratch_dirs.py [DIR...] --apply    # actually merge + clean up

With no DIR given, every ``${SCRATCH}/canon-*``, ``${SCRATCH}/smoke-canon-*`` and
``${SCRATCH}/smoke-optreports*`` sibling of this repo checkout is considered.
"""

import argparse
import csv
import os
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
JOBFILE_RE = re.compile(r"-(\d+)\.(?:out|err)$")


def discover_columns(run_dir: pathlib.Path) -> list[str]:
    """Every column with at least one CSV shard in ``run_dir`` (mirrors collect_canon.py)."""
    return sorted({shard.name.split(".rank", 1)[0] for shard in run_dir.glob("*.rank*.csv")})


def job_ids_in_dir(run_dir: pathlib.Path) -> set[str]:
    """Every Slurm job id named by a ``*-<jobid>.out``/``.err`` file in ``run_dir``."""
    ids: set[str] = set()
    for entry in run_dir.iterdir():
        match = JOBFILE_RE.search(entry.name)
        if match:
            ids.add(match.group(1))
    return ids


def sacct_lookup(job_ids: set[str]) -> tuple[dict[str, str], dict[str, str]]:
    """``(states, names)`` for every id in ``job_ids``, both keyed by job id string.

    ``-X`` so a job's own row is used, not its ``.batch``/``.extern`` step rows, which can show a
    different (usually COMPLETED-regardless) state than the job itself."""
    if not job_ids:
        return {}, {}
    out = subprocess.run(
        ["sacct", "-j", ",".join(sorted(job_ids)), "--format=JobID,JobName,State", "--noheader", "--parsable2", "-X"],
        capture_output=True,
        text=True,
        check=True,
    )
    states: dict[str, str] = {}
    names: dict[str, str] = {}
    for line in out.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        jobid, jobname, state = parts[0].strip(), parts[1].strip(), parts[2].strip()
        states[jobid] = state
        names[jobid] = jobname
    return states, names


def column_job_id(column: str, job_ids: set[str], names: dict[str, str]) -> str | None:
    """The job id whose ``JobName`` ends in ``-<column>`` -- ``submit-canon-llr40.sh``'s own
    ``--job-name="${JOB_PREFIX}-${JOB_TAG:-${col%%,*}}"``. ``None`` when no job in this directory
    was named for this column (e.g. a comma-joined ``ONE_JOB=1`` job only names its FIRST column;
    every OTHER column that job also processed cannot be matched by name here and is left alone --
    a limitation, not a guess)."""
    suffix = "-" + column
    for jid in job_ids:
        if names.get(jid, "").endswith(suffix):
            return jid
    return None


def csv_row_count(run_dir: pathlib.Path, column: str) -> int:
    """Independent count of ``column``'s CSV data rows -- the same number
    ``canon_column.sh``'s finalize step passes as ``--expected``."""
    total = 0
    for shard in sorted(run_dir.glob(f"{column}.rank*.csv")):
        with shard.open(newline="") as fh:
            total += sum(1 for _ in csv.DictReader(fh))
    return total


def merge_column(run_dir: pathlib.Path, column: str, db: pathlib.Path, *, apply: bool) -> bool:
    expected = csv_row_count(run_dir, column)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "merge_canon_results.py"),
        "--run-dir",
        str(run_dir),
        "--column",
        column,
        "--run",
        run_dir.name,
        "--db",
        str(db),
        "--expected",
        str(expected),
    ]
    if not apply:
        print(f"  [dry run] would merge {column}: {' '.join(cmd)}")
        return True
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"  {result.stdout.strip()}" or f"  merge_canon_results.py produced no output for {column}")
    if result.returncode != 0:
        print(f"  {column}: merge NOT verified -- {result.stderr.strip()}", file=sys.stderr)
        return False
    for build_dir in [run_dir / f"dacecache-{column}", *run_dir.glob(f"dacecache-{column}_rank*")]:
        if build_dir.exists():
            shutil.rmtree(build_dir)
    shard_dir = run_dir / "db" / column
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    print(f"  {column}: merged and cleared its build tree + shard DB")
    return True


def archive_and_remove(run_dir: pathlib.Path, archive_dir: pathlib.Path, *, apply: bool) -> None:
    dest = archive_dir / run_dir.name
    if not apply:
        print(f"  [dry run] would archive CSVs + reports/ to {dest} and remove {run_dir}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    for shard in run_dir.glob("*.rank*.csv"):
        shutil.copy2(shard, dest / shard.name)
    reports = run_dir / "reports"
    if reports.is_dir():
        shutil.copytree(reports, dest / "reports", dirs_exist_ok=True)
    shutil.rmtree(run_dir)
    print(f"  archived CSVs + reports/ to {dest} and removed {run_dir}")


def default_dirs() -> list[pathlib.Path]:
    scratch = pathlib.Path(os.environ.get("SCRATCH", str(ROOT.parent)))
    patterns = ("canon-*", "smoke-canon-*", "smoke-optreports*")
    found: list[pathlib.Path] = []
    for pattern in patterns:
        found.extend(sorted(p for p in scratch.glob(pattern) if p.is_dir()))
    return found


def migrate_one(run_dir: pathlib.Path, db: pathlib.Path, archive_dir: pathlib.Path, *, apply: bool) -> None:
    print(f"=== {run_dir} ===")
    columns = discover_columns(run_dir)
    if not columns:
        print("  no <column>.rank*.csv shards here yet -- nothing to do")
        return
    job_ids = job_ids_in_dir(run_dir)
    states, names = sacct_lookup(job_ids)

    all_clear = True
    for column in columns:
        jid = column_job_id(column, job_ids, names)
        if jid is None:
            print(f"  {column}: no Slurm job in this directory is named for it -- leaving untouched")
            all_clear = False
            continue
        state = states.get(jid, "UNKNOWN")
        if state != "COMPLETED":
            print(f"  {column}: job {jid} is {state}, not COMPLETED -- leaving untouched")
            all_clear = False
            continue
        if not merge_column(run_dir, column, db, apply=apply):
            all_clear = False

    if all_clear:
        archive_and_remove(run_dir, archive_dir, apply=apply)
    else:
        print(f"  {run_dir} keeps its remaining files (not every column is COMPLETED and merged)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", type=pathlib.Path, help="work dirs to migrate (default: discovered)")
    ap.add_argument(
        "--db",
        type=pathlib.Path,
        default=None,
        help="persistent canon DB (default: $HPCAGENT_BENCH_RESULTS_DIR/canon.db)",
    )
    ap.add_argument(
        "--archive-dir",
        type=pathlib.Path,
        default=None,
        help="where a fully-migrated directory's CSVs/reports are archived before removal "
        "(default: $HPCAGENT_BENCH_RESULTS_DIR/canon-archive)",
    )
    ap.add_argument(
        "--apply", action="store_true", help="actually merge, archive and delete (default: print the plan only)"
    )
    args = ap.parse_args(argv)

    results_dir = os.environ.get("HPCAGENT_BENCH_RESULTS_DIR")
    db = args.db or (pathlib.Path(results_dir) / "canon.db" if results_dir else None)
    archive_dir = args.archive_dir or (pathlib.Path(results_dir) / "canon-archive" if results_dir else None)
    if db is None or archive_dir is None:
        print(
            "no --db/--archive-dir given and $HPCAGENT_BENCH_RESULTS_DIR is unset; "
            "source scripts/cache_env.sh first or pass both explicitly",
            file=sys.stderr,
        )
        return 2

    dirs = args.dirs or default_dirs()
    if not dirs:
        print("no candidate directories found", file=sys.stderr)
        return 1

    if not args.apply:
        print("DRY RUN -- pass --apply to actually merge/archive/delete anything\n")
    for run_dir in dirs:
        migrate_one(run_dir, db, archive_dir, apply=args.apply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
