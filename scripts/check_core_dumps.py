# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every shell entry point in this repo must disable core dumps before it runs anything.

A core dump lands in the CRASHING PROCESS'S CWD -- which for a campaign arm is the repository checkout. A segfaulting agent kernel,
a wedged engine, an OOM-killed rank: each leaves a ``core_<host>_<pid>`` file behind, and the inode
cost is paid on a filesystem whose quota is inodes rather than bytes.

``ulimit -c 0`` is the whole fix and it must be in the SCRIPT BODY: slurm propagates the
submitting shell's rlimits into the job by default, so a login shell with an unlimited core limit
hands that limit to every step. Setting it inside the script overrides that for the step and
everything it spawns. There is no ``#SBATCH`` flag that does this -- ``--propagate`` selects which
of the SUBMITTER'S limits to carry, so it can only pass a bad limit along, never impose a good one.
With neither ``-S`` nor ``-H``, bash sets BOTH limits (measured), so nothing the script spawns can
raise it back -- which is the point of a floor, and why the marker below exists for the one script
that genuinely wants its own dump.

Three rules, because each one alone has been escaped:

1. EVERY shell entry point carries the guard -- ``*.sbatch``, ``*.sh``, and any tracked file whose
   shebang names a shell (``containers/agent/bin/hpcagent-bench-tool`` and the OpenHands shell
   ``bash-norc`` carry no suffix and sit closest to the compiler that crashes). A sourced library
   counts too: setting the limit there is what carries the floor into the caller's shell.
2. No script RE-ENABLES them: "the text contains ``ulimit -c 0``" would accept a later
   ``ulimit -c unlimited`` that still dumps, so a non-zero ``ulimit -c`` needs
   a same-line ``# core-dumps-ok: <reason>`` marker, so a deliberate one (a probe that gdbs its own
   core in a container's /tmp and deletes it) is reviewed rather than silent.
3. A script that EMITS a batch script counts as one. ``scripts/preset_sweep.py --emit-sbatch``
   writes a submittable header from an f-string, so the guard has to be inside the emitted text --
   and a check keyed on the suffix never sees it. Those are reported, never auto-fixed: the
   insertion point sits inside a quoted template, where a blind splice would land in the wrong
   string.

Python processes are covered from the other side by :mod:`hpcagent_bench.core_dumps`, which drops
the limit at package import -- that is what catches an ad-hoc script outside the repo.

    python scripts/check_core_dumps.py [--fix] [paths...]
"""

import argparse
import pathlib
import re
import subprocess
import sys

#: The line every shell entry point must carry, and the comment that says why it is there.
GUARD = "ulimit -c 0"
BLOCK = """# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
"""

#: Suffixes that are a shell entry point: something a human or a job runs, or sources into one.
SHELL_SUFFIXES = (".sbatch", ".sh")

#: ...and a shebang naming a shell, for the entry points that carry no suffix at all. The agent's
#: own `containers/agent/bin/hpcagent-bench-tool` and the OpenHands shell `bash-norc` are both
#: extensionless `#!/bin/sh` wrappers, so a suffix rule never sees the two scripts closest to the
#: compiler that crashes.
SHELL_SHEBANG = re.compile(rb"^#!.*\b(?:ba|da|k|z|a)?sh\b")

#: A real submission header always names the job; `#SBATCH` alone also matches prose about it.
EMITTED_HEADER = "#SBATCH --job-name"

#: A ``ulimit -c`` whose operand is anything but 0 -- ``unlimited``, or a block count.
REENABLE = re.compile(r"\bulimit\s+(?:-[A-Za-z]*\s+)*-c\s+(?!0\b)(\S+)")

#: Same-line opt-out for a deliberate core dump, e.g. a probe that gdbs its own core under /tmp.
MARKER = "# core-dumps-ok:"


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def display(path: pathlib.Path) -> str:
    """Repo-relative when it can be, absolute otherwise -- a path argument may sit outside the tree."""
    try:
        return str(path.relative_to(repo_root()))
    except ValueError:
        return str(path)


def tracked(root: pathlib.Path, *globs: str) -> list[pathlib.Path]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", *globs], capture_output=True, text=True, check=False
    ).stdout.split()
    return [root / name for name in out]


def is_shell(path: pathlib.Path) -> bool:
    """A shell entry point by suffix, or by a shebang naming a shell."""
    if path.suffix in SHELL_SUFFIXES:
        return True
    with path.open("rb") as handle:
        return SHELL_SHEBANG.match(handle.readline(256)) is not None


def shell_scripts(paths: list[str]) -> list[pathlib.Path]:
    """Every shell entry point: the given paths filtered, or the whole tracked tree.

    With no paths this reads the first line of every tracked file, which is what finds the
    extensionless ones; the pre-commit run gets staged paths and never pays for the walk.
    """
    candidates = [pathlib.Path(p) for p in paths] if paths else tracked(repo_root())
    return [p for p in candidates if p.is_file() and is_shell(p)]


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


def reenabled(path: pathlib.Path) -> list[tuple[int, str]]:
    """Unmarked ``ulimit -c <non-zero>`` lines -- the guard string alone does not prove cores are off.

    Shell files only: this script and its test both SPELL the offending command, and a suffix rule
    that cannot match them beats an exclusion list that a rename would silently invalidate.
    """
    if path.suffix not in SHELL_SUFFIXES:
        return []
    hits = []
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        if REENABLE.search(line) and MARKER not in line:
            hits.append((number, line.strip()))
    return hits


def insertion_point(lines: list[str]) -> int:
    """After the shebang, the leading comment block, the ``#SBATCH`` header and a leading ``set -``.

    Scanning stops at the first line of real work, so a ``set -x`` inside a quoted inner shell
    later in the file never receives the guard. Placed after ``set -euo pipefail`` rather than before so
    a script that has one keeps its failure semantics on the very first command.
    """
    at = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if index == 0 and stripped.startswith("#!"):
            at = 1
        elif not stripped or stripped.startswith(("#", "set -")):
            at = index + 1
        else:
            break
    return at


def report(label: str, offenders: list[pathlib.Path], hint: str) -> None:
    names = "\n  ".join(display(p) for p in offenders)
    print(f"core-dumps: {len(offenders)} {label}:\n  {names}\n{hint}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="insert the guard instead of only reporting")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    offenders, rearmed = [], []
    for path in shell_scripts(args.paths):
        text = path.read_text(encoding="utf-8")
        hits = reenabled(path)
        if hits:
            rearmed.append(path)
            print(f"core-dumps: {display(path)} re-enables core dumps:", file=sys.stderr)
            for number, line in hits:
                print(f"  {number}: {line}", file=sys.stderr)
        if GUARD in text:
            continue
        if not args.fix:
            offenders.append(path)
            continue
        lines = text.splitlines(keepends=True)
        at = insertion_point(lines)
        pad = "" if at and not lines[at - 1].strip() else "\n"
        lines.insert(at, pad + BLOCK)
        path.write_text("".join(lines), encoding="utf-8")
        print(f"core-dumps: added the guard to {display(path)}")

    emitted = [p for p in emitters(args.paths) if GUARD not in p.read_text(encoding="utf-8", errors="ignore")]

    if offenders:
        report("shell script(s) do not disable core dumps", offenders, "Run: python scripts/check_core_dumps.py --fix")
    if rearmed:
        report(
            "script(s) re-enable core dumps",
            rearmed,
            f"Set `{GUARD}`, or keep the dump and say why with a same-line `{MARKER} <reason>`.",
        )
    if emitted:
        report(
            "file(s) emit an SBATCH header without the guard",
            emitted,
            f"Add `{GUARD}` to the EMITTED script body (not the emitting file's own header).",
        )
    return 1 if offenders or rearmed or emitted else 0


if __name__ == "__main__":
    raise SystemExit(main())
