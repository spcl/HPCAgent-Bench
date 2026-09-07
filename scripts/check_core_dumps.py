# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every batch script must disable core dumps before it runs anything.

Beverin's ``core_pattern`` is the machine-global ``core_%h_%p`` and a dump lands in the CRASHING
PROCESS'S CWD -- which for a campaign arm is the repository checkout. A segfaulting agent kernel,
a wedged engine, an OOM-killed rank: each leaves a ``core_nid<node>_<pid>`` file behind, and the
inode cost is paid on a filesystem whose quota is inodes rather than bytes.

``ulimit -c 0`` is the whole fix and it must be in the SCRIPT BODY: slurm propagates the
submitting shell's rlimits into the job by default, so a login shell with an unlimited core limit
hands that limit to every step. Setting it inside the script overrides that for the step and
everything it spawns. There is no ``#SBATCH`` flag that does this -- ``--propagate`` selects which
of the SUBMITTER'S limits to carry, so it can only pass a bad limit along, never impose a good one.

A script that EMITS a batch script counts too. ``scripts/preset_sweep.py --emit-sbatch`` writes a
submittable header from an f-string, so the guard has to be inside the emitted text -- and a check
keyed on the ``.sbatch`` suffix never sees it. Those are reported, never auto-fixed: the insertion
point sits inside a quoted template, where a blind splice would land in the wrong string.

    python scripts/check_core_dumps.py [--fix] [paths...]

With no paths it walks every ``*.sbatch`` and every tracked file that emits an SBATCH header.
"""

import argparse
import pathlib
import subprocess
import sys

#: The line every batch script must carry, and the comment that says why it is there.
GUARD = "ulimit -c 0"
BLOCK = """
# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_nid<node>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
"""


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def display(path: pathlib.Path) -> str:
    """Repo-relative when it can be, absolute otherwise -- a path argument may sit outside the tree."""
    try:
        return str(path.relative_to(repo_root()))
    except ValueError:
        return str(path)


#: A real submission header always names the job; `#SBATCH` alone also matches prose about it.
EMITTED_HEADER = "#SBATCH --job-name"


def tracked(root: pathlib.Path, *globs: str) -> list[pathlib.Path]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", *globs], capture_output=True, text=True, check=False
    ).stdout.split()
    return [root / name for name in out]


def batch_scripts(paths: list[str]) -> list[pathlib.Path]:
    if paths:
        return [pathlib.Path(p) for p in paths if p.endswith(".sbatch")]
    return tracked(repo_root(), "*.sbatch")


def emitters(paths: list[str]) -> list[pathlib.Path]:
    """Tracked non-``.sbatch`` files that write an SBATCH header into a script they generate."""
    candidates = [pathlib.Path(p) for p in paths] if paths else tracked(repo_root(), "*.py", "*.sh")
    found = []
    for path in candidates:
        if path.suffix == ".sbatch" or not path.is_file() or path.name == pathlib.Path(__file__).name:
            continue
        if EMITTED_HEADER in path.read_text(encoding="utf-8", errors="ignore"):
            found.append(path)
    return found


def insertion_point(lines: list[str]) -> int:
    """After the shebang, the ``#SBATCH`` block and any ``set -e`` line -- before real work.

    Placed after ``set -euo pipefail`` rather than before it so a script that has one keeps its
    failure semantics on the very first command, and so the guard reads as setup rather than as
    part of the SBATCH header a reader scans for resources.
    """
    last_header = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if index == 0 and stripped.startswith("#!"):
            last_header = 1
        elif stripped.startswith("#SBATCH") or stripped.startswith("set -"):
            last_header = index + 1
    return last_header


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="insert the guard instead of only reporting")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    offenders = []
    for path in batch_scripts(args.paths):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if GUARD in text:
            continue
        if not args.fix:
            offenders.append(path)
            continue
        lines = text.splitlines(keepends=True)
        at = insertion_point(lines)
        lines.insert(at, BLOCK)
        path.write_text("".join(lines), encoding="utf-8")
        print(f"core-dumps: added the guard to {path}")

    emitted = [p for p in emitters(args.paths) if GUARD not in p.read_text(encoding="utf-8", errors="ignore")]

    if offenders:
        names = "\n  ".join(display(p) for p in offenders)
        print(
            f"core-dumps: {len(offenders)} batch script(s) do not disable core dumps:\n  {names}\n"
            "Run: python scripts/check_core_dumps.py --fix",
            file=sys.stderr,
        )
    if emitted:
        names = "\n  ".join(display(p) for p in emitted)
        print(
            f"core-dumps: {len(emitted)} file(s) emit an SBATCH header without the guard:\n  {names}\n"
            f"Add `{GUARD}` to the EMITTED script body (not the emitting file's own header).",
            file=sys.stderr,
        )
    return 1 if offenders or emitted else 0


if __name__ == "__main__":
    raise SystemExit(main())
