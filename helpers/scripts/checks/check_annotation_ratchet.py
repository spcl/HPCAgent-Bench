# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every function gets type hints -- ratcheted per file, because thousands do not yet.

Rule: write Python as if it were statically typed. Turning ruff's ANN rules on repo-wide would
report every violation at once and block every other change, so this holds the direction of
travel instead: a file may not gain violations, and a file that loses them must lower its entry in
``annotation_baseline.json`` beside this script. The ratchet runs both ways: an entry above the
real count is slack a regression can hide in. Entries name tracked files only (an untracked file
exists on one machine and cannot be cleared anywhere else).

The kernel corpus (``hpcagent_bench/benchmarks/``) is reference numpy a reader compares against a
paper, so it is out of scope.

    python helpers/scripts/checks/check_annotation_ratchet.py [FILE ...]          # check
    python helpers/scripts/checks/check_annotation_ratchet.py --write [FILE ...]  # update the baseline
"""

import collections
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
BASELINE = pathlib.Path(__file__).with_name("annotation_baseline.json")
ROOTS = ("hpcagent_bench", "tests", "scripts", "experiments", "tools")
EXCLUDE = "hpcagent_bench/benchmarks"


def tracked() -> frozenset[str]:
    proc = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"], capture_output=True, text=True, check=True)
    return frozenset(proc.stdout.split("\0"))


def in_scope(rel: str) -> bool:
    return rel.endswith(".py") and rel.split("/", 1)[0] in ROOTS and not rel.startswith(EXCLUDE + "/")


def violations(paths: list[str]) -> collections.Counter[str]:
    """``{repo-relative path: ANN violation count}`` from ruff itself, over ``paths``."""
    command = [sys.executable, "-m", "ruff", "check", "--select", "ANN", "--output-format", "json", "--no-cache"]
    proc = subprocess.run([*command, "--exclude", EXCLUDE, *paths], cwd=REPO, capture_output=True, text=True)
    if proc.returncode not in (0, 1) or not proc.stdout.strip():  # 1 = findings; "No module named ruff" is 1 too
        raise RuntimeError(f"ruff could not run (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    return collections.Counter(
        str(pathlib.Path(item["filename"]).resolve().relative_to(REPO)) for item in json.loads(proc.stdout)
    )


def main(argv: list[str]) -> int:
    write = "--write" in argv
    known = tracked()
    named = [rel for rel in argv if rel != "--write"]
    files = [rel for rel in named if in_scope(rel) and rel in known]
    if named and not files:
        return 0
    found = violations(files or list(ROOTS))
    baseline: dict[str, int] = json.loads(BASELINE.read_text())
    scope = set(files) if files else {rel for rel in known if in_scope(rel)} | set(baseline)
    if write:
        for rel in scope:
            if found.get(rel, 0) and rel in known:
                baseline[rel] = found[rel]
            else:
                baseline.pop(rel, None)
        BASELINE.write_text(json.dumps(dict(sorted(baseline.items())), indent=1) + "\n")
        return 0
    faults = [
        f"{rel}: {baseline.get(rel, 0)} -> {found.get(rel, 0)} unannotated functions"
        + (
            " (annotate what you added)"
            if found.get(rel, 0) > baseline.get(rel, 0)
            else " (lower the baseline: --write)"
        )
        for rel in sorted(scope)
        if found.get(rel, 0) != baseline.get(rel, 0)
    ]
    faults += [f"{rel}: in the baseline but untracked" for rel in sorted(set(baseline) - known)]
    print("\n".join(faults))
    return 1 if faults else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
