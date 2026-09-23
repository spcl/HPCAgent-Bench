# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Ratchet for the type gate ([tool.pyright] in pyproject.toml, standard mode, repo-wide).

Same shape and same reasoning as tests/test_ruff_ratchet.py: standard-mode pyright over the
whole non-kernel tree reports real debt today, so this measures the DIRECTION -- a file may not
gain diagnostics, and a file that loses them must say so by regenerating:

    python tests/test_pyright_ratchet.py --write

This is DELIBERATELY separate from pyrightconfig.strict.json's growing allowlist: strict mode is
an opt-in target for a file being cleaned up all the way; this ratchet is the floor every other
file already clears (or is tracked against) today. A file promoted into the strict list keeps its
entry here too until someone removes it -- strict is a stronger promise, not a replacement scope.

pyright needs an interpreter with the project's dependencies AND the repo + translator src on the
import path. This reads them from the environment the test itself was launched in (PYTHONPATH,
and sys.executable as --pythonpath) rather than hardcoding a venv -- see CONTRIBUTING.md "Lint
gate" for the exact invocation this ratchet expects to be run under.
"""

import collections
import functools
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
BASELINE = pathlib.Path(__file__).with_name("pyright_ratchet_baseline.json")

#: Mirrors [tool.pyright] include/exclude in pyproject.toml -- kept here as plain data (not read
#: back out of the toml) so a baseline entry and the config's own scope cannot silently drift
#: apart without a diff showing both.
ROOTS = ("hpcagent_bench", "tests", "scripts", "experiments", "tools")
EXCLUDE = "hpcagent_bench/benchmarks"


@functools.lru_cache(maxsize=None, typed=True)
def tracked() -> frozenset[str]:
    proc = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True)
    return frozenset(proc.stdout.split("\0"))


@functools.lru_cache(maxsize=None, typed=True)
def diagnostics() -> collections.Counter[str]:
    """``{repo-relative path: diagnostic count}`` from pyright itself (errors + warnings)."""
    extra_path = [str(REPO), str(REPO / "hpcagent_bench/numpy_translators/src")]
    existing_path = os.environ.get("PYTHONPATH", "")
    full_path = os.pathsep.join(extra_path + ([existing_path] if existing_path else []))
    proc = subprocess.run(
        [shutil.which("pyright"), "--pythonpath", sys.executable, "--outputjson"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,  # pyright exits non-zero on diagnostics, not on failure
        env=os.environ | {"PYTHONPATH": full_path},
    )
    if not proc.stdout.strip():
        raise RuntimeError(f"pyright produced no output (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    payload = json.loads(proc.stdout)
    found: collections.Counter[str] = collections.Counter()
    known = tracked()
    for item in payload["generalDiagnostics"]:
        path = str(pathlib.Path(item["file"]).resolve().relative_to(REPO))
        if path.startswith(f"{EXCLUDE}/") or not any(path.startswith(f"{r}/") for r in ROOTS):
            continue
        if path in known:
            found[path] += 1
    return found


@pytest.mark.integration
def test_no_file_gains_a_type_diagnostic() -> None:
    if shutil.which("pyright") is None:
        pytest.skip("pyright not on PATH")
    baseline = json.loads(BASELINE.read_text())
    found = diagnostics()
    grown = {f: (baseline.get(f, 0), n) for f, n in found.items() if n > baseline.get(f, 0)}
    assert not grown, (
        "these files gained pyright diagnostics (was, now):\n  "
        + "\n  ".join(f"{f}: {was} -> {now}" for f, (was, now) in sorted(grown.items()))
        + "\nFix the type error at the source -- never a blanket # type: ignore."
    )


@pytest.mark.integration
def test_the_baseline_does_not_overstate_the_debt() -> None:
    if shutil.which("pyright") is None:
        pytest.skip("pyright not on PATH")
    baseline = json.loads(BASELINE.read_text())
    found = diagnostics()
    shrunk = {f: (was, found.get(f, 0)) for f, was in baseline.items() if found.get(f, 0) < was}
    assert not shrunk, (
        "these files have FEWER pyright diagnostics than the baseline claims (was, now):\n  "
        + "\n  ".join(f"{f}: {was} -> {now}" for f, (was, now) in sorted(shrunk.items()))
        + "\nRegenerate: python tests/test_pyright_ratchet.py --write"
    )


def test_the_baseline_names_files_the_checkout_has() -> None:
    missing = sorted(set(json.loads(BASELINE.read_text())) - tracked())
    assert not missing, f"the baseline names files no checkout has: {missing[:10]}"


if __name__ == "__main__":
    if "--write" not in sys.argv:
        raise SystemExit("usage: python tests/test_pyright_ratchet.py --write")
    counts = dict(sorted(diagnostics().items()))
    BASELINE.write_text(json.dumps(counts, indent=1, sort_keys=True) + "\n")
    print(f"pyright ratchet baseline: {sum(counts.values())} diagnostics across {len(counts)} files")
