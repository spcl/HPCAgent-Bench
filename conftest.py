# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Repo-root conftest: what has to be true before ANY test module in ANY suite is imported.

Both test trees (``tests/`` and ``hpcagent_bench/numpy_translators/tests/``) are collected from
this directory in CI, so a root conftest is the one place a process-wide pin can live without
being written twice -- the translator suite deliberately imports nothing from ``hpcagent_bench``,
so it cannot share a helper module with the other one.

The pin itself lives in :mod:`dace_build_isolation`, not here: each tree has a ``conftest`` of its
own, so a test importing ``conftest`` by name gets whichever was imported first.
"""

import os

import pytest

from dace_build_isolation import pin_per_worker_dace_build_folder

pin_per_worker_dace_build_folder()


#: Error annotations one pytest process emits per test before the end-of-run list takes over.
#: GitHub keeps 10 error annotations per step, so the list at the end needs the tenth.
FAILURE_ANNOTATION_LIMIT = 9

failures_annotated: list[str] = []


def workflow_escape(text: str, *, prop: bool = False) -> str:
    """``text`` escaped for a GitHub Actions workflow command (message, or a ``prop`` value)."""
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return text.replace(":", "%3A").replace(",", "%2C") if prop else text


def failure_line(report: pytest.TestReport) -> str:
    """The one line that says why ``report`` failed: pytest's crash message, else its last line."""
    crash = getattr(report.longrepr, "reprcrash", None)
    if crash is not None:
        return str(crash.message)
    lines = str(report.longrepr).strip().splitlines()
    return lines[-1] if lines else "(no failure text)"


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """On GitHub Actions, turn a failure into an error ANNOTATION as it happens.

    A public repository's job logs need admin rights to read (``/actions/jobs/<id>/logs`` answers
    403), but a check run's annotations do not (``/check-runs/<id>/annotations``). Without this a
    red job told anyone outside the admin list only "Process completed with exit code 1"."""
    if os.environ.get("GITHUB_ACTIONS") != "true" or not report.failed:
        return
    failures_annotated.append(f"{report.nodeid} ({report.when}): {failure_line(report)[:300]}".replace("\n", " | "))
    if len(failures_annotated) <= FAILURE_ANNOTATION_LIMIT:
        title = workflow_escape(f"FAILED {report.nodeid}", prop=True)
        print(f"\n::error title={title}::{workflow_escape(failure_line(report)[:2000])}", flush=True)


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Every failure of the run in one annotation, past the per-step cap the ones above hit."""
    if os.environ.get("GITHUB_ACTIONS") != "true" or not failures_annotated:
        return
    title = workflow_escape(f"{len(failures_annotated)} failed", prop=True)
    body = "\n".join(failures_annotated)[:60000]
    terminalreporter.write_line(f"::error title={title}::{workflow_escape(body)}")
