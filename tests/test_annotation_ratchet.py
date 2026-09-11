# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every function gets type hints -- ratcheted, because 12,759 of them do not yet.

USER RULE: write Python as if it were statically typed. Every function annotated, no name rebound
to a different type, no dynamic instance attributes. Turning ruff's ANN rules on repo-wide would
report ~12.7k violations at once and block every other change, so this measures the direction of
travel instead: a file may not gain violations, and a file that loses them must say so.

The ratchet runs BOTH WAYS, like the DaCe refusal list. An entry whose count DROPPED is a failure
too: the baseline is what makes the gate mean something, and one that quietly overstates the debt
lets a regression hide inside the slack it left. Fix either direction by regenerating:

    python tests/test_annotation_ratchet.py --write

The kernel corpus is excluded. ``hpcagent_bench/benchmarks/`` is reference numpy that a reader is
meant to compare against a paper, and annotating it is a separate decision about what those files
are for -- not debt this gate should book.
"""

import collections
import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
BASELINE = pathlib.Path(__file__).with_name("annotation_baseline.json")

#: Where the rule applies. The benchmark corpus is deliberately absent -- see the module docstring.
ROOTS = ("hpcagent_bench", "tests", "scripts", "experiments", "tools")
EXCLUDE = "hpcagent_bench/benchmarks"


def violations() -> collections.Counter:
    """``{repo-relative path: ANN violation count}`` from ruff itself, not a reimplementation."""
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "ANN", "--output-format", "json",
         "--no-cache", "--exclude", EXCLUDE, *ROOTS],
        cwd=REPO, capture_output=True, text=True,
    )
    if proc.returncode not in (0, 1):  # 0 = clean, 1 = findings; anything else is ruff failing
        raise RuntimeError(f"ruff could not run (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    found = collections.Counter()
    for item in json.loads(proc.stdout or "[]"):
        found[str(pathlib.Path(item["filename"]).resolve().relative_to(REPO))] += 1
    return found


def test_no_file_gains_an_unannotated_function() -> None:
    """A new function without hints, or a new file full of them, fails here."""
    baseline = json.loads(BASELINE.read_text())
    found = violations()
    grown = {f: (baseline.get(f, 0), n) for f, n in found.items() if n > baseline.get(f, 0)}
    assert not grown, (
        "these files gained unannotated functions (was, now):\n  "
        + "\n  ".join(f"{f}: {was} -> {now}" for f, (was, now) in sorted(grown.items()))
        + "\nAnnotate what you added: every parameter and the return."
    )


def test_the_baseline_does_not_overstate_the_debt() -> None:
    """The other direction. A stale entry is slack a regression can hide in."""
    baseline = json.loads(BASELINE.read_text())
    found = violations()
    shrunk = {f: (was, found.get(f, 0)) for f, was in baseline.items() if found.get(f, 0) < was}
    assert not shrunk, (
        "these files have FEWER unannotated functions than the baseline claims (was, now):\n  "
        + "\n  ".join(f"{f}: {was} -> {now}" for f, (was, now) in sorted(shrunk.items()))
        + "\nRegenerate: python tests/test_annotation_ratchet.py --write"
    )


def test_the_baseline_names_files_that_exist() -> None:
    missing = sorted(f for f in json.loads(BASELINE.read_text()) if not (REPO / f).exists())
    assert not missing, f"the baseline names files that are gone: {missing[:10]}"


if __name__ == "__main__":
    if "--write" not in sys.argv:
        raise SystemExit("usage: python tests/test_annotation_ratchet.py --write")
    counts = dict(sorted(violations().items()))
    BASELINE.write_text(json.dumps(counts, indent=1, sort_keys=True) + "\n")
    print(f"annotation baseline: {sum(counts.values())} violations across {len(counts)} files")
