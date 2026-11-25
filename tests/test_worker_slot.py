# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""An xdist worker keeps every CPU when its share would be too narrow for the multi-thread tests."""

import os

from tests.worker_slot import MIN_SHARE, PINNED_ENV, pin_worker_cpus


def test_a_share_under_min_share_cpus_is_not_pinned() -> None:
    """GitHub's 4-vCPU runner with two workers: a 2-CPU share would leave the 4-thread tests one core."""
    cpus = frozenset(os.sched_getaffinity(0))
    count = max(2, len(cpus) // (MIN_SHARE - 1))
    env = {"PYTEST_XDIST_WORKER": "gw1", "PYTEST_XDIST_WORKER_COUNT": str(count)}
    assert len(cpus) // count < MIN_SHARE
    assert pin_worker_cpus(env) == cpus
    assert PINNED_ENV not in env
    assert frozenset(os.sched_getaffinity(0)) == cpus
