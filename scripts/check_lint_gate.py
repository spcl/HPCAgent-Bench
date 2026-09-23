#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pre-commit hook: curated ruff check + the ratchet, scoped to the files being committed.

Two failure modes, both fast (only the changed files are linted, never the whole tree):

1. A changed file has ANY ruff finding under [tool.ruff.lint]'s curated select -- new or touched
   code is held to the gate directly, not just "no worse than before".
2. A changed file's finding count exceeds tests/ruff_ratchet_baseline.json's entry for it -- this
   only fires for a file NOT in the diff's own file list picking up findings from unrelated
   pre-existing code, which case 1 already would not allow; kept as a second check because the
   ratchet is the source of truth the full-repo test (tests/test_ruff_ratchet.py) also reads, and
   a baseline entry that quietly drops out of sync here would surface as ONLY a CI failure later.

This intentionally does not run pyright or pylint: both are minutes-scale over the whole repo
(pyright ~1-2 files/sec including import resolution; pylint slower again in its own venv) and a
commit hook must stay sub-second to seconds. The full type ratchet (tests/test_pyright_ratchet.py)
and pylint run in CI / on demand -- see CONTRIBUTING.md "Lint gate".
"""

import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
BASELINE = REPO / "tests" / "ruff_ratchet_baseline.json"


def changed_py_files(file_args: list[str]) -> list[str]:
    return [f for f in file_args if f.endswith(".py") and (REPO / f).is_file()]


def ruff_findings(files: list[str]) -> dict[str, list[dict]]:
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--output-format", "json", "--no-cache", *files],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,  # ruff exits 1 on findings, not on failure -- returncode is inspected below
    )
    if proc.returncode not in (0, 1):
        print(f"error: ruff could not run (rc={proc.returncode}):\n{proc.stderr[-2000:]}", file=sys.stderr)
        raise SystemExit(2)
    by_file: dict[str, list[dict]] = {}
    for item in json.loads(proc.stdout or "[]"):
        path = str(pathlib.Path(item["filename"]).resolve().relative_to(REPO))
        by_file.setdefault(path, []).append(item)
    return by_file


def main(argv: list[str] | None = None) -> int:
    files = changed_py_files(sys.argv[1:] if argv is None else argv)
    if not files:
        return 0

    baseline: dict[str, int] = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    found = ruff_findings(files)

    offenders: list[str] = []
    for path in files:
        hits = found.get(path, [])
        if not hits:
            continue
        was = baseline.get(path, 0)
        if len(hits) > was:
            offenders.append(path)
            print(f"\n{path}: {len(hits)} finding(s) (ratchet baseline: {was})")
            for item in hits:
                loc = item["location"]
                print(f"  {loc['row']}:{loc['column']}  {item['code']}  {item['message']}")

    if not offenders:
        print(f"lint-gate: {len(files)} changed file(s) OK")
        return 0

    print(
        f"\nlint-gate: {len(offenders)} of {len(files)} changed file(s) gained ruff findings.\n"
        "Fix with:  ruff check --fix <file>   (safe fixes), then review the rest by hand.\n"
        "If this is a deliberate, reviewed cleanup that LOWERED a count instead, regenerate:\n"
        "  python tests/test_ruff_ratchet.py --write"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
