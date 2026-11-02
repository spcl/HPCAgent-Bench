# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A pytest plugin failing the run on any skip it does not allow by exact reason, for a CI step that installed
what its tests need (``-p tests.no_silent_skips``). Test and collection skips count alike, so a ``pytest.skip``,
a ``skipif`` and an ``importorskip`` fall under one rule, and a reworded reason goes red."""

import pytest

#: What a step without a GPU legitimately cannot run.
ALLOWED_REASONS = ("no CUDA device / cupy", "nvcc + an MPI wrapper are required")


def pytest_sessionfinish(session: pytest.Session) -> None:
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:  # an xdist worker: the controller sees every report
        return
    reasons = [report.longrepr[2].removeprefix("Skipped: ") for report in reporter.stats.get("skipped", ())]
    unallowed = [reason for reason in reasons if reason not in ALLOWED_REASONS]
    if unallowed:
        reporter.write_line(f"skips this step does not allow: {unallowed}", red=True)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
