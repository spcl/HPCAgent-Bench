# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""A worklist every shard of a job takes items from: each shard claims the next unclaimed item through an ``O_EXCL``
marker file on the shared filesystem, so no shard idles behind a static split of items 100x apart in cost.

A held claim is touched every :data:`HEARTBEAT_S`; one untouched for :data:`STALE_S` belonged to a shard that died
and is taken over through the item's next claim file, so two shards never both win one claim."""

import contextlib
import dataclasses
import hashlib
import os
import pathlib
import socket
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence

__all__ = [
    "HEARTBEAT_S",
    "MAX_CLAIMS",
    "POLL_S",
    "STALE_S",
    "ClaimQueue",
    "Held",
    "claim_each",
    "create_exclusive",
    "heartbeat",
    "item_key",
    "job_tag",
    "new_job",
    "stale",
]

#: Seconds between the touches that keep a held claim alive.
HEARTBEAT_S = 30.0
#: Seconds after which an untouched claim is taken over: its shard died.
STALE_S = 300.0
#: Claims one job makes on one item; an item whose every holder died is given up on (it likely killed them).
MAX_CLAIMS = 3
#: Seconds a shard with nothing left to claim waits before it looks again for a claim gone stale.
POLL_S = 30.0


def new_job() -> str:
    """A claim namespace of its own: a job no other shard is part of."""
    return f"local-{uuid.uuid4().hex[:12]}"


def job_tag() -> str:
    """This job's claim namespace, shared by its shards: ``SLURM_JOB_ID``, else :func:`new_job`. A later job retries
    what an earlier one gave up on."""
    return os.environ.get("SLURM_JOB_ID") or new_job()


def item_key(*parts: object) -> str:
    """A file-name-safe key of one item, from the ``parts`` that identify it."""
    return hashlib.sha256("\0".join(str(part) for part in parts).encode()).hexdigest()[:32]


def create_exclusive(path: pathlib.Path) -> bool:
    """Create ``path`` naming this host and process; False when it already exists (atomic on a shared filesystem)."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as fh:
        fh.write(f"{socket.gethostname()} {os.getpid()}\n")
    return True


def stale(path: pathlib.Path) -> bool:
    """Whether claim ``path`` went untouched for :data:`STALE_S`; a vanished claim is not stale."""
    try:
        return time.time() - path.stat().st_mtime > STALE_S
    except FileNotFoundError:
        return False


@contextlib.contextmanager
def heartbeat(path: pathlib.Path) -> Iterator[None]:
    """Touch ``path`` every :data:`HEARTBEAT_S` while the block runs, so other shards see its holder alive."""
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(HEARTBEAT_S):
            with contextlib.suppress(OSError):
                os.utime(path)

    thread = threading.Thread(target=beat, name="claim-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


@dataclasses.dataclass(frozen=True, slots=True)
class ClaimQueue:
    """The claims on one worklist under ``root``: ``<key>.done`` once an item's grade is written (every job skips
    it); ``<job>/<key>.<n>`` the job's n-th claim on it; ``<job>/<key>.closed`` an item the job gave up on."""

    root: pathlib.Path
    job: str

    def __post_init__(self) -> None:
        (self.root / self.job).mkdir(parents=True, exist_ok=True)

    def done(self, key: str) -> pathlib.Path:
        return self.root / f"{key}.done"

    def claim(self, key: str, attempt: int) -> pathlib.Path:
        return self.root / self.job / f"{key}.{attempt}"

    def closed(self, key: str) -> pathlib.Path:
        return self.root / self.job / f"{key}.closed"

    def settled(self, key: str) -> bool:
        """Whether ``key`` needs no shard of this job: graded, or given up on."""
        return self.done(key).exists() or self.closed(key).exists()

    def live_elsewhere(self, key: str) -> list[str]:
        """The other jobs holding a live claim on ``key`` (two jobs may grade one ``--out-dir`` at once)."""
        return sorted(
            path.parent.name
            for path in self.root.glob(f"*/{key}.*")
            if path.parent.name != self.job and path.suffix != ".closed" and not stale(path)
        )

    def take(self, key: str) -> pathlib.Path | None:
        """The claim on ``key`` this shard just won; None while a live shard of any job holds it. An item whose every
        claim in this job went stale is closed instead. Two jobs that claim at once both see it: the later job name
        backs off."""
        if self.live_elsewhere(key):
            return None
        for attempt in range(MAX_CLAIMS):
            path = self.claim(key, attempt)
            if create_exclusive(path):
                rivals = self.live_elsewhere(key)
                # Its holder finished, renaming this claim away, since the caller looked; or another job won the race.
                if self.done(key).exists() or any(rival < self.job for rival in rivals):
                    path.unlink()
                    return None
                return path
            if not stale(path):
                return None
        create_exclusive(self.closed(key))
        return None

    def finish(self, key: str, claim: pathlib.Path | None = None) -> None:
        """Record ``key`` as graded; ``claim`` (this shard's) becomes the marker, so a finished item costs one file."""
        if claim is not None:
            with contextlib.suppress(FileNotFoundError):
                claim.rename(self.done(key))
                return
        create_exclusive(self.done(key))


@dataclasses.dataclass(slots=True)
class Held[ItemT]:
    """An item this shard holds the claim on; :meth:`finish` once its grade is written."""

    item: ItemT
    finished: bool = False

    def finish(self) -> None:
        self.finished = True


def claim_each[ItemT](
    items: Sequence[ItemT], queue: ClaimQueue, key: Callable[[ItemT], str], *, wait: bool = True
) -> Iterator[Held[ItemT]]:
    """Every item of ``items`` this shard wins, in worklist order, held (heartbeat) while the caller grades it.

    An item the caller did not :meth:`Held.finish` is closed for this job. With ``wait``, a shard left with nothing to
    claim polls every :data:`POLL_S` until each item is settled, taking over any claim that went stale."""
    pending = list(items)
    while pending:
        held_elsewhere: list[ItemT] = []
        for item in pending:
            name = key(item)
            if queue.settled(name):
                continue
            claim = queue.take(name)
            if claim is None:
                if not queue.settled(name):
                    held_elsewhere.append(item)
                continue
            held = Held(item)
            with heartbeat(claim):
                yield held
            if held.finished:
                queue.finish(name, claim)
            else:
                create_exclusive(queue.closed(name))
        if not wait or not held_elsewhere:
            return
        pending = held_elsewhere
        time.sleep(POLL_S)
