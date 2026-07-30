# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every test file runs somewhere in CI, and the shards stay balanced.

This exists because the opposite was true and nothing said so: 144 test files, 44 named anywhere
in the workflow, 94 that never executed -- including guards written for regressions they were
meant to catch. A hand-written file list drifts in one direction only, because a new test is inert
by default and inertness is silent.
"""
import pathlib
import subprocess
import sys
from typing import List, Set

import pytest

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


def shard(index: int, total: int, files: List[str]) -> List[str]:
    """One shard, as the workflow computes it."""
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "ci_shard.py"), "--shard", f"{index}/{total}", "--files", " ".join(files)
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        check=True,
    )
    return [line for line in proc.stdout.split() if line]


@pytest.fixture(scope="module")
def translator_files() -> List[str]:
    root = REPO / "hpcagent_bench" / "numpy_translators" / "tests"
    return [f"hpcagent_bench/numpy_translators/tests/{p.name}" for p in sorted(root.glob("test_*.py"))]


def test_shards_partition_the_files_exactly(translator_files: List[str]) -> None:
    """No file run twice (wasted runner) and none dropped (a silent hole in coverage)."""
    shards = [shard(i, 3, translator_files) for i in range(3)]
    flat = [f for s in shards for f in s]
    assert sorted(flat) == sorted(translator_files)
    assert len(flat) == len(set(flat)), "a file landed in more than one shard"


def test_the_split_is_deterministic(translator_files: List[str]) -> None:
    """Every runner computes its own shard with no coordination, so twice must give the same answer."""
    assert shard(0, 3, translator_files) == shard(0, 3, translator_files)


def test_heavy_files_do_not_pile_into_one_shard(translator_files: List[str]) -> None:
    """The point of weighting: a round-robin deal balances the FILE COUNT and nothing else.

    Costs here span orders of magnitude, so the check is on spread, not on equality -- a perfect
    balance is not achievable and not required. What must not happen is one shard holding a
    multiple of another's work."""
    from scripts.ci_shard import pack, weight_of  # noqa: PLC0415 -- the module under test

    paths = [REPO / f for f in translator_files]
    packed = pack(paths, 3, {})
    loads = [sum(weight_of(p, {}) for p in group) for group in packed]
    assert min(loads) > 0, "a shard drew no work"
    assert max(loads) / min(loads) < 1.5, f"shard loads are lopsided: {loads}"
