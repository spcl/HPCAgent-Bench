#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pre-commit guard: the python embedded in an image Dockerfile must not reference an unbound name.

The image Dockerfiles end with a gate written as ``python <<'PY' ... PY``: it imports the built
stack and asserts the versions, backends and parser flags the campaign depends on. That block runs
LAST, so a name typed once and never bound is not a lint -- it is the whole build. 620855 spent 43
minutes assembling vLLM 0.27.1 with aiter and died on the final line with ``NameError: name
'triton_version' is not defined``, after every real check in the gate had already passed.

Nothing else looks at these blocks: they are heredoc text to the Dockerfile, and the repo's python
hooks match ``*.py``. Extracting them and asking ruff for its undefined-name rules costs
milliseconds and catches the one failure mode whose feedback loop is measured in hours.

F821 undefined-name is the rule that matters; F811 redefinition and F822 undefined ``__all__`` come
along because they are the same class of typo and cost nothing to include.

Exit status: 0 when every embedded block is clean, 1 otherwise.
"""

import argparse
import pathlib
import re
import subprocess
import sys
import tempfile

#: Every embedded block in the tree uses this one marker; a new spelling should be added here
#: rather than silently escaping the check.
HEREDOC_OPEN = re.compile(r"<<'PY'\s*$")
HEREDOC_CLOSE = re.compile(r"^PY$")
RULES = "F821,F822,F811"


def blocks(path: pathlib.Path) -> list[tuple[int, str]]:
    """Every embedded python block in one Dockerfile as ``(first line number, source)``."""
    found: list[tuple[int, str]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    start: int | None = None
    for index, line in enumerate(lines):
        if start is None:
            if HEREDOC_OPEN.search(line):
                start = index + 1
        elif HEREDOC_CLOSE.match(line):
            found.append((start + 1, "\n".join(lines[start:index]) + "\n"))
            start = None
    return found


def offenders(paths: list[pathlib.Path]) -> list[str]:
    """ruff's undefined-name findings, each already rewritten to name the Dockerfile and its line."""
    reports: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for path in paths:
            for offset, source in blocks(path):
                # The block is written out alone, so ruff sees exactly what the interpreter will:
                # the Dockerfile's shell context binds nothing a python name could resolve to.
                extracted = pathlib.Path(tmp) / f"{path.parent.name}-{offset}.py"
                extracted.write_text(source, encoding="utf-8")
                run = subprocess.run(
                    ["ruff", "check", "--isolated", "--select", RULES, "--output-format", "concise", str(extracted)],
                    capture_output=True,
                    text=True,
                )
                if run.returncode == 0:
                    # A clean run still prints "All checks passed!" on stdout, so the exit status is
                    # what says whether there is anything to read.
                    continue
                for line in run.stdout.splitlines():
                    # concise output is `<file>:<line>:<col>: <code> <message>`; the line number is
                    # the BLOCK's, so shift it back onto the Dockerfile the reader has open.
                    parts = line.split(":", 3)
                    if len(parts) == 4 and parts[1].isdigit():
                        reports.append(f"  {path}:{offset + int(parts[1]) - 1}:{parts[2]}:{parts[3]}")
                    elif line.strip() and not line.startswith("Found "):
                        reports.append(f"  {path}: {line}")
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="Dockerfiles to check; default is every image Dockerfile")
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    paths = [pathlib.Path(f) for f in args.files] or sorted(root.glob("containers/cluster/ce-images/*/Dockerfile"))
    paths = [p for p in paths if p.name == "Dockerfile"]
    if not paths:
        return 0

    try:
        found = offenders(paths)
    except FileNotFoundError:
        print("error: ruff is not on PATH (pip install ruff)", file=sys.stderr)
        return 1

    if not found:
        return 0
    print(f"error: {len(found)} unbound name(s) in embedded Dockerfile python:\n", file=sys.stderr)
    for report in found:
        print(report, file=sys.stderr)
    print(
        "\nThese blocks run at the END of a multi-hour image build, so a NameError here costs the "
        "whole build (620855). Bind the name where it is computed.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
