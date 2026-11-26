# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The per-push rotation of a slow set: a push sweeps :data:`PER_PUSH` members picked by its commit, so every member
runs within ``len(set)`` pushes; a dispatched run (``HPCAGENT_BENCH_E2E_SUBSET=0``) and a local run sweep them all."""

import functools
import os
import subprocess
from collections.abc import Callable, Iterable, Sequence

from hpcagent_bench import paths

#: Members of a rotated set one push sweeps.
PER_PUSH = 1


def per_push() -> bool:
    """Whether this run is a push's slice (the workflow sets ``HPCAGENT_BENCH_E2E_SUBSET=1`` except when dispatched)."""
    return os.environ.get("HPCAGENT_BENCH_E2E_SUBSET") == "1"


def commit() -> str:
    """The commit under test: ``GITHUB_SHA`` in CI, else the checkout's HEAD; every xdist worker reads the same."""
    return os.environ.get("GITHUB_SHA") or paths.git_head()


def rotation(members: Iterable[str], sha: str, count: int = PER_PUSH) -> list[str]:
    """``count`` members of ``members`` in sorted order from the one ``sha`` picks, wrapping around."""
    ordered = sorted(set(members))
    if not ordered:
        return []
    start = int(sha[:8], 16) % len(ordered) if sha else 0
    return [ordered[(start + step) % len(ordered)] for step in range(min(count, len(ordered)))]


def rotated(members: Iterable[str]) -> list[str]:
    """``members`` for this run: every one, or on a push's slice its :func:`rotation`."""
    return rotation(members, commit()) if per_push() else sorted(set(members))


def kept[Member](members: Sequence[Member], subset: Iterable[str], key: Callable[[Member], str] = str) -> list[Member]:
    """``members``, or on a push's slice the fixed ``subset`` of them (by ``key``). A subset is chosen from the CI
    history (the cases that failed in any run from 2026-10-14 to 11-26) and, among cases that drive the same code,
    keeps the hardest: one that passes leaves little an easier one could catch. A dispatched run sweeps every member."""
    if not per_push():
        return list(members)
    names = frozenset(subset)
    return [member for member in members if key(member) in names]


@functools.lru_cache(maxsize=1, typed=True)
def changed_paths() -> frozenset[str]:
    """Repository paths the commit under test changes against its merge base with ``origin/main``; empty when the
    checkout has no such history (a shallow clone) or git cannot run."""
    argv = ["git", "-C", str(paths.ROOT), "diff", "--name-only", "origin/main...HEAD"]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    return frozenset(done.stdout.split()) if done.returncode == 0 else frozenset()
