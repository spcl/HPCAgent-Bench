# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Ratchet for the curated ruff lint gate ([tool.ruff.lint] select in pyproject.toml).

A bare `ruff check` under that select reports 21k+ findings across hpcagent_bench/experiments/
tests/scripts/tools today -- fixing all of it before the gate can be enforced is not this task.
Like tests/test_annotation_ratchet.py, this measures the DIRECTION instead: a file may not gain
findings, and a file that loses them must say so by regenerating the baseline. Runs BOTH WAYS,
same reasoning as the annotation ratchet -- an entry that overstates the debt hides a regression
in its own slack.

Regenerate after a real cleanup (or when the config's select/per-file-ignores change):

    python tests/test_ruff_ratchet.py --write

Scope is every ruff-selected finding under ROOTS, INCLUDING hpcagent_bench/benchmarks: the
per-file-ignores in pyproject.toml already silence the numpy-kernel naming/signature noise there,
so what is left (F841 dead code, B-series bugs, PERF, ...) is real debt this gate should see.
"""

import collections
import functools
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
BASELINE = pathlib.Path(__file__).with_name("ruff_ratchet_baseline.json")

ROOTS = ("hpcagent_bench", "tests", "scripts", "experiments", "tools")


@functools.lru_cache(maxsize=None, typed=True)
def tracked() -> frozenset[str]:
    """Repo-relative paths git TRACKS -- see test_annotation_ratchet.tracked for why."""
    proc = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True)
    return frozenset(proc.stdout.split("\0"))


@functools.lru_cache(maxsize=None, typed=True)
def violations() -> collections.Counter[str]:
    """``{repo-relative path: finding count}`` from ruff itself, under pyproject.toml's curated
    select and per-file-ignores (no --select override here -- that IS the gate)."""
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--output-format", "json", "--no-cache", *ROOTS],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,  # ruff exits 1 on findings, not on failure -- returncode is inspected below
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"ruff could not run (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    if not proc.stdout.strip():
        raise RuntimeError(f"ruff produced no output (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    found: collections.Counter[str] = collections.Counter()
    known = tracked()
    for item in json.loads(proc.stdout):
        path = str(pathlib.Path(item["filename"]).resolve().relative_to(REPO))
        if path in known:
            found[path] += 1
    return found


def test_no_file_gains_a_lint_finding() -> None:
    """A new violation, or a new file full of them, fails here."""
    baseline = json.loads(BASELINE.read_text())
    found = violations()
    grown = {f: (baseline.get(f, 0), n) for f, n in found.items() if n > baseline.get(f, 0)}
    assert not grown, (
        "these files gained ruff findings (was, now):\n  "
        + "\n  ".join(f"{f}: {was} -> {now}" for f, (was, now) in sorted(grown.items()))
        + "\nFix what you added: ruff check --fix (safe fixes), then a manual look at the rest."
    )


def test_the_baseline_does_not_overstate_the_debt() -> None:
    """The other direction. A stale entry is slack a regression can hide in."""
    baseline = json.loads(BASELINE.read_text())
    found = violations()
    shrunk = {f: (was, found.get(f, 0)) for f, was in baseline.items() if found.get(f, 0) < was}
    assert not shrunk, (
        "these files have FEWER ruff findings than the baseline claims (was, now):\n  "
        + "\n  ".join(f"{f}: {was} -> {now}" for f, (was, now) in sorted(shrunk.items()))
        + "\nRegenerate: python tests/test_ruff_ratchet.py --write"
    )


def test_the_baseline_names_files_the_checkout_has() -> None:
    """Tracked, not merely present -- see test_annotation_ratchet's twin for why."""
    missing = sorted(set(json.loads(BASELINE.read_text())) - tracked())
    assert not missing, f"the baseline names files no checkout has: {missing[:10]}"


if __name__ == "__main__":
    if "--write" not in sys.argv:
        raise SystemExit("usage: python tests/test_ruff_ratchet.py --write")
    counts = dict(sorted(violations().items()))
    BASELINE.write_text(json.dumps(counts, indent=1, sort_keys=True) + "\n")
    print(f"ruff ratchet baseline: {sum(counts.values())} findings across {len(counts)} files")
