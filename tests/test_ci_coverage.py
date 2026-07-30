# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every test file runs somewhere in CI.

This exists because the opposite was true and nothing said so: 144 test files, 44 named anywhere
in the workflow, 94 that never executed -- including guards written for regressions they were
meant to catch. A hand-written file list drifts in one direction only, because a new test is inert
by default and inertness is silent.
"""
import pathlib
import re
from typing import Set

REPO = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "tests.yml"
DEDICATED = REPO / ".github" / "dedicated_tests.txt"


def dedicated_files() -> Set[str]:
    """Paths the exclusion file claims, comments and blanks dropped -- the same parse tests.yml does."""
    out: Set[str] = set()
    for line in DEDICATED.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line)
    return out


def all_test_files() -> Set[str]:
    return {f"tests/{p.name}" for p in sorted((REPO / "tests").glob("test_*.py"))}


def test_every_test_file_runs_somewhere() -> None:
    """The invariant: a file is swept, or claimed by a dedicated phase. There is no third state."""
    claimed = dedicated_files()
    swept = all_test_files() - claimed
    assert swept, "the unit sweep would select nothing"
    orphaned = claimed - all_test_files()
    assert not orphaned, (f"dedicated_tests.txt names files that do not exist: {sorted(orphaned)}. "
                          "A stale entry silently shrinks the sweep.")


def test_a_dedicated_file_is_actually_run_by_some_phase() -> None:
    """Excluding a file from the sweep is only legitimate when another phase runs it.

    Without this, dedicated_tests.txt becomes the new silent-inertness mechanism -- the exact
    failure it was introduced to end, one indirection later."""
    workflow = WORKFLOW.read_text()
    missing = [name for name in sorted(dedicated_files()) if name not in workflow]
    assert not missing, (f"excluded from the sweep but named by no phase, so they run NOWHERE: {missing}. "
                         "Either give the file a phase, or quarantine it with a written reason.")


def test_the_sweep_is_discovered_not_enumerated() -> None:
    """The workflow must derive its file list from the filesystem, not carry one."""
    workflow = WORKFLOW.read_text()
    assert "dedicated_tests.txt" in workflow, "the sweep no longer reads the exclusion file"
    assert "ls tests/test_*.py" in workflow, "the sweep no longer discovers files with ls"


def test_ci_never_asks_for_a_billed_runner() -> None:
    """Standard GitHub-hosted runners are free on a public repo; LARGER runners bill per minute
    even here. Self-hosted is our own hardware and bills nothing.

    A guard rather than a review habit: `runs-on: ubuntu-latest-8-cores` is one plausible edit away
    from `ubuntu-latest`, reads as a harmless speedup, and the cost of getting it wrong arrives on
    an invoice rather than in a test run."""
    standard = {"ubuntu-latest", "ubuntu-24.04", "ubuntu-22.04", "windows-latest", "macos-latest"}
    offenders = []
    for workflow in sorted((REPO / ".github" / "workflows").glob("*.y*ml")):
        for line in workflow.read_text().splitlines():
            match = re.match(r"\s*runs-on:\s*(.+?)\s*$", line)
            if not match:
                continue
            value = match.group(1)
            if value.startswith("["):  # a label list -- self-hosted, i.e. our own machine
                if "self-hosted" not in value:
                    offenders.append(f"{workflow.name}: {value}")
            elif value not in standard:
                offenders.append(f"{workflow.name}: {value}")
    assert not offenders, (f"non-standard, billed-per-minute runners requested: {offenders}. "
                           f"Free on a public repo are {sorted(standard)}, plus self-hosted labels.")
