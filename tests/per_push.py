# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The per-push rotation of a slow set: a push sweeps :data:`PER_PUSH` members picked by its commit, so every member
runs within ``len(set)`` pushes; a dispatched run (``HPCAGENT_BENCH_E2E_SUBSET=0``) and a local run sweep them all."""

import os
from collections.abc import Iterable

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
