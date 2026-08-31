#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pre-commit guard: a kernel may not write through a numpy ``out=`` keyword.

``np.add(a, b, out=c)`` is a store the CALL performs, not one the statement performs, and every
consumer of these kernels has to reconstruct that. numba rejects the keyword outright; pythran
accepts it and silently does nothing, so ``c`` keeps whatever ``np.empty`` handed back and the
column reports uninitialized memory as a numerical failure; the native translators have to
recognise the keyword as an aliased store before they can lower it at all. The slice-assign form
``c[:] = np.add(a, b)`` says the same thing to numpy -- the right-hand side is evaluated in full
and then stored, aliasing included -- and says it in the one spelling every backend already
lowers.

So the rule is the spelling, not the semantics: write the store as a store.

Scope: Python sources under ``hpcagent_bench/benchmarks``. Auto-generated siblings are skipped --
they are a function of the reference beside them, so an ``out=`` there is the reference's to fix
and would be reported twice.

Exit status: 0 when no source writes through ``out=``, 1 otherwise (each offender is printed with
the slice-assign it should be).
"""
import argparse
import ast
import subprocess
import sys
from pathlib import Path

#: Where kernels live.
BENCH_ROOT = "hpcagent_bench/benchmarks"

#: Module aliases a kernel may spell numpy as.
NUMPY_MODULES = frozenset({"np", "numpy"})


def tracked_sources():
    """Every tracked Python file under the benchmarks tree (standalone-scan fallback)."""
    out = subprocess.run(["git", "ls-files", f"{BENCH_ROOT}/**/*.py"], capture_output=True, text=True)
    if out.returncode != 0:
        return []
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def is_generated(path: Path) -> bool:
    """A generated sibling carries the autogen marker on its first line."""
    try:
        with path.open(encoding="utf-8") as handle:
            return "hpcagent_bench-autogen" in handle.readline()
    except OSError:
        return False


def offenders(paths):
    """Yield ``(path, lineno, source_line)`` for each ``np.<f>(..., out=X)`` call."""
    for rel in paths:
        path = Path(rel)
        if BENCH_ROOT not in path.as_posix() or path.suffix != ".py" or is_generated(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (OSError, SyntaxError):
            continue
        lines = text.splitlines()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            root = node.func.value
            if not (isinstance(root, ast.Name) and root.id in NUMPY_MODULES):
                continue
            if any(kw.arg == "out" for kw in node.keywords):
                yield rel, node.lineno, lines[node.lineno - 1].strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="files to check (default: the tracked benchmark sources)")
    args = ap.parse_args(argv)

    bad = sorted(set(offenders(args.files if args.files else tracked_sources())))
    if not bad:
        return 0

    print(f"error: {len(bad)} numpy call(s) write through out=:\n", file=sys.stderr)
    for rel, lineno, text in bad:
        print(f"  {rel}:{lineno}\n    {text}", file=sys.stderr)
    print(
        "\nWrite the store as a store: np.f(a, b, out=X) -> X[:] = np.f(a, b) (or X[<slice>] = ... "
        "when the target is already a slice). numba rejects out=, pythran silently ignores it.",
        file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
