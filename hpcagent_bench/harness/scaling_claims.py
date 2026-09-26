# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The claim table that lets any number of ML-scaling grade jobs, and every gang of each, grade one
out directory concurrently without two of them replaying the same submission.

One small sqlite file in the out dir (:data:`CLAIM_DB`) maps a submission key (``regrade.KEY``) to
the claimer grading it (``<job>-<gang>``), its state (``claimed`` / ``done``) and a heartbeat. A
claim is taken under ``BEGIN IMMEDIATE`` -- the write lock is held from the first read, so two
claimers can never both see a key free. A ``claimed`` row whose heartbeat is older than the stale
limit belongs to a dead job and is taken over. A heartbeat runs in its own process
(``python -m hpcagent_bench.harness.scaling_claims beat``): the grade forks, and a forked child must
inherit no live sqlite connection or thread (``scaling_grade.run_shard``). It stops when its parent
dies, so a killed job's claims go stale by themselves.

The grade rows themselves (``scaling_grade.graded_keys``) stay THE record of what is graded; a
``done`` claim is the fast path and the duration history :func:`item_estimate` reads.
"""

import argparse
import contextlib
import dataclasses
import os
import pathlib
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence

__all__ = [
    "BASELINE_DB",
    "CLAIM_DB",
    "CLAIM_DDL",
    "HEARTBEAT_S",
    "MIN_HISTORY",
    "STALE_S",
    "Claimer",
    "Key",
    "beat",
    "beat_loop",
    "claim",
    "claimed_by_job",
    "connection",
    "finish",
    "heartbeat",
    "held_keys",
    "item_estimate",
    "main",
    "release",
]

#: The claim DB's file name in the out dir. Not ``scaling-grade-*.db``: graded_keys globs those.
CLAIM_DB: str = "scaling-claims.db"
CLAIM_DDL: str = (
    "CREATE TABLE IF NOT EXISTS claims (db TEXT, run_id TEXT, benchmark TEXT, ts_ms INTEGER, "
    "claimer TEXT, job TEXT, gang INTEGER, state TEXT, claimed_at REAL, heartbeat REAL, done_at REAL, "
    "PRIMARY KEY (db, run_id, benchmark, ts_ms))"
)
#: How often a claimer's heartbeat is written, and after how long without one a claim is stale.
HEARTBEAT_S: float = 60.0
STALE_S: float = 600.0
#: Past grades needed before their durations replace the default per-item estimate.
MIN_HISTORY: int = 3
#: ``db`` of a torch.distributed baseline point's claim (``torch_dist_curve.claim_key``): a work
#: item that is not a submission, so it neither counts against MAX_ITEMS nor enters the per-item
#: estimate (a point takes minutes, a submission's whole sweep up to an hour).
BASELINE_DB: str = "torch_dist"

Key = tuple[str, str, str, int]


@dataclasses.dataclass(frozen=True, slots=True)
class Claimer:
    """One gang of one job: ``job`` bounds MAX_ITEMS, ``name`` owns claims and the heartbeat."""

    path: pathlib.Path
    job: str
    gang: int
    stale_s: float = STALE_S

    @property
    def name(self) -> str:
        return f"{self.job}-{self.gang}"


@contextlib.contextmanager
def connection(path: pathlib.Path) -> Iterator[sqlite3.Connection]:
    """An autocommit connection (transactions are explicit), the table created if new; closed on
    exit -- never held across a grade's fork."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=120, isolation_level=None)
    try:
        conn.execute(CLAIM_DDL)
        yield conn
    finally:
        conn.close()


def claimed_by_job(path: pathlib.Path, job: str) -> int:
    """How many submissions ``job`` has claimed, done ones included (the MAX_ITEMS count)."""
    with connection(path) as conn:
        return int(
            conn.execute("SELECT COUNT(*) FROM claims WHERE job = ? AND db != ?", (job, BASELINE_DB)).fetchone()[0]
        )


def held_keys(path: pathlib.Path, stale_s: float = STALE_S, now: float | None = None) -> set[Key]:
    """Every submission a live claimer holds or has finished (what no new claimer can take)."""
    now = time.time() if now is None else now
    with connection(path) as conn:
        rows = conn.execute(
            "SELECT db, run_id, benchmark, ts_ms FROM claims WHERE state = 'done' OR heartbeat >= ?", (now - stale_s,)
        ).fetchall()
    return {(str(db), str(run_id), str(benchmark), int(ts)) for db, run_id, benchmark, ts in rows}


def claim(claimer: Claimer, keys: Sequence[Key], batch: int, max_items: int = 0, now: float | None = None) -> list[Key]:
    """Atomically take up to ``batch`` of ``keys``, in order: a key never claimed, or claimed by a
    claimer whose heartbeat is stale. ``max_items`` > 0 caps the claims of ``claimer.job`` over its
    whole life, across its gangs. Returns the keys taken."""
    now = time.time() if now is None else now
    taken: list[Key] = []
    with connection(claimer.path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            room = batch
            if max_items > 0:
                used = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM claims WHERE job = ? AND db != ?", (claimer.job, BASELINE_DB)
                    ).fetchone()[0]
                )
                room = min(batch, max_items - used)
            for key in keys:
                if len(taken) >= room:
                    break
                row = conn.execute(
                    "SELECT state, heartbeat FROM claims WHERE db = ? AND run_id = ? AND benchmark = ? AND ts_ms = ?",
                    key,
                ).fetchone()
                if row is not None and (row[0] != "claimed" or float(row[1]) >= now - claimer.stale_s):
                    continue
                conn.execute(
                    "INSERT OR REPLACE INTO claims VALUES (?, ?, ?, ?, ?, ?, ?, 'claimed', ?, ?, NULL)",
                    (*key, claimer.name, claimer.job, claimer.gang, now, now),
                )
                taken.append(key)
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    return taken


def beat(path: pathlib.Path, name: str, now: float | None = None) -> None:
    """Refresh the heartbeat of every submission ``name`` holds."""
    with connection(path) as conn:
        conn.execute(
            "UPDATE claims SET heartbeat = ? WHERE claimer = ? AND state = 'claimed'",
            (time.time() if now is None else now, name),
        )


def finish(claimer: Claimer, key: Key) -> None:
    """Mark one of ``claimer``'s submissions graded."""
    with connection(claimer.path) as conn:
        conn.execute(
            "UPDATE claims SET state = 'done', done_at = ? "
            "WHERE db = ? AND run_id = ? AND benchmark = ? AND ts_ms = ? AND claimer = ?",
            (time.time(), *key, claimer.name),
        )


def release(claimer: Claimer) -> None:
    """Hand back every submission ``claimer`` holds and has not graded."""
    with connection(claimer.path) as conn:
        conn.execute("DELETE FROM claims WHERE claimer = ? AND state = 'claimed'", (claimer.name,))


def item_estimate(path: pathlib.Path, default_s: float) -> float:
    """Seconds one more grade is expected to take: the 90th percentile of the claim-to-done
    durations recorded so far (any job), or ``default_s`` before :data:`MIN_HISTORY` of them --
    never less than the default's half, so one lucky early grade cannot open a too-short window."""
    with connection(path) as conn:
        durations = sorted(
            float(row[0])
            for row in conn.execute(
                "SELECT done_at - claimed_at FROM claims WHERE state = 'done' AND done_at IS NOT NULL AND db != ?",
                (BASELINE_DB,),
            )
        )
    if len(durations) < MIN_HISTORY:
        return default_s
    return max(durations[int(0.9 * (len(durations) - 1))], default_s / 2)


@contextlib.contextmanager
def heartbeat(claimer: Claimer, interval_s: float = HEARTBEAT_S) -> Iterator[None]:
    """Keep ``claimer``'s claims fresh from a separate process while the body runs."""
    beater = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hpcagent_bench.harness.scaling_claims",
            "beat",
            "--claims",
            str(claimer.path),
            "--claimer",
            claimer.name,
            "--interval",
            str(interval_s),
            "--parent",
            str(os.getpid()),
        ],
        stdin=subprocess.DEVNULL,
    )
    try:
        yield
    finally:
        beater.terminate()
        beater.wait()


def beat_loop(path: pathlib.Path, name: str, interval_s: float, parent: int) -> int:
    """Beat until the parent is gone (re-parented: it died, so its claims must go stale)."""
    while os.getppid() == parent:
        with contextlib.suppress(sqlite3.OperationalError):  # a busy lock: the next beat retries
            beat(path, name)
        time.sleep(interval_s)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    beating = sub.add_parser("beat", help="refresh one claimer's heartbeat until its parent exits")
    beating.add_argument("--claims", required=True, type=pathlib.Path)
    beating.add_argument("--claimer", required=True)
    beating.add_argument("--interval", type=float, default=HEARTBEAT_S)
    beating.add_argument("--parent", required=True, type=int)
    args = ap.parse_args(argv)
    return beat_loop(args.claims, args.claimer, args.interval, args.parent)


if __name__ == "__main__":
    sys.exit(main())
