#!/usr/bin/env python3
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Migrate the flat foundation/ benchmark tree into per-kernel subfolders.

    foundation/adist.yaml
    foundation/adist_numpy.py         ->   foundation/adist/adist.yaml
    foundation/adist_reference.cpp            foundation/adist/adist_numpy.py
    ...                                      foundation/adist/adist_reference.cpp

WHY THIS IS SAFE WITHOUT CODE CHANGES
The kernel registry discovers manifests with ``base.rglob("*.yaml")`` (spec.py), so a
manifest one directory deeper is still found. ``relative_path`` is derived from the
manifest's parent, and every downstream path -- the numpy reference
(``numpy_reference_path``), the emit output, and its ``cpp_backend/`` -- is derived from
``relative_path`` (autogen.py: ``kdir = BENCHMARKS / relative_path``; ``cppdir = kdir /
"cpp_backend"``). So moving a kernel's files together re-homes all of them at once, with no
loader or emitter edit. ``cpp_backend/`` itself is generated + gitignored and is NOT moved;
it regenerates per-kernel in the new location on the next build.

TRACKED FILES ONLY
Only ``git``-tracked files are moved. A kernel's C/C++/Fortran wrapper (``<stem>_cpp.py``),
its numba/tvm siblings (``<stem>_numba_*.py``, ``<stem>_tvm.py``), and everything under
``cpp_backend/`` are GENERATED and gitignored -- ``git mv`` refuses them, and they do not
need moving: the emitter re-derives their path from ``relative_path`` and regenerates them
in the new subfolder on the next build (any stale flat copy left behind is a gitignored
orphan). So the move set is the tracked artifacts: the manifest, the numpy reference, the
``_reference.*`` sources, an optional ``<stem>.py`` initializer, and any tracked ``_mpi.*``.

KERNEL IDENTITY AND THE ONE COLLISION
A kernel is a ``<stem>.yaml`` manifest. A tracked sibling belongs to it when its name is
that stem followed by a suffix (empty, or beginning with ``_`` / ``.``). Assignment is
LONGEST-STEM-FIRST: ``ext_floordiv_offset_m_numpy.py`` is claimed by
``ext_floordiv_offset_m`` (its files start the same way as ``ext_floordiv_offset``'s),
never by the shorter ``ext_floordiv_offset``. That pair is the only prefix collision in the
corpus; longest-first makes the rule general anyway.

Files that match no kernel stem (e.g. ``__init__.py``) are left in place and reported.

Usage:
    python scripts/restructure_foundation.py            # dry-run: print the plan
    python scripts/restructure_foundation.py --apply    # execute the moves via ``git mv``

``git mv`` preserves history and stages the moves; nothing is committed here.
"""
import argparse
import collections
import pathlib
import subprocess
import sys

FOUNDATION = pathlib.Path("hpcagent_bench/benchmarks/foundation")

#: Files that are not per-kernel artifacts and must stay at the foundation root.
KEEP_AT_ROOT = {"__init__.py", "README.md"}


def kernel_stems(root: pathlib.Path):
    """Manifest stems, LONGEST first, so a longer kernel name claims its files before a
    shorter kernel whose name is a prefix of it."""
    return sorted((p.stem for p in root.glob("*.yaml")), key=len, reverse=True)


def owning_kernel(filename: str, stems):
    """The kernel that owns ``filename``: the longest stem it starts with, where what
    follows the stem is empty or begins with ``_`` / ``.`` (a suffix, not a longer name)."""
    for stem in stems:
        if filename.startswith(stem):
            rest = filename[len(stem):]
            if rest == "" or rest[0] in "_.":
                return stem
    return None


def tracked_filenames(root: pathlib.Path):
    """The git-tracked file BASENAMES directly under ``root`` (not recursive).

    ``git ls-files`` lists tracked paths; generated files (``_cpp.py`` wrappers, ``_numba_*``,
    ``_tvm.py``, ``cpp_backend/``) are gitignored and so are absent -- exactly the ones
    ``git mv`` cannot move and that regenerate from ``relative_path`` anyway.
    """
    out = subprocess.run(["git", "ls-files", str(root)], capture_output=True, text=True, check=True).stdout
    names = set()
    for line in out.splitlines():
        rel = pathlib.PurePosixPath(line).relative_to(root.as_posix())
        if len(rel.parts) == 1:  # directly under root, not already in a subfolder
            names.add(rel.parts[0])
    return names


def build_plan(root: pathlib.Path):
    """Return ``(moves, unclaimed)``: kernel -> [tracked filenames], and tracked files
    matching no kernel stem."""
    stems = kernel_stems(root)
    tracked = tracked_filenames(root)
    moves = collections.defaultdict(list)
    unclaimed = []
    for name in sorted(tracked):
        if name in KEEP_AT_ROOT:
            continue
        owner = owning_kernel(name, stems)
        if owner is None:
            unclaimed.append(name)
        else:
            moves[owner].append(name)
    return moves, unclaimed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="execute the moves (default: dry-run)")
    parser.add_argument("--root", default=str(FOUNDATION), help="foundation directory")
    args = parser.parse_args(argv)
    root = pathlib.Path(args.root)
    if not root.is_dir():
        parser.error(f"{root} is not a directory")

    moves, unclaimed = build_plan(root)
    total = sum(len(v) for v in moves.values())
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] {len(moves)} kernels, {total} files to move (root={root})")

    if unclaimed:
        # A file matching no kernel is either a new artifact type or a stray. Do not guess a
        # home for it -- report and skip, so the migration never silently misfiles anything.
        print(f"  {len(unclaimed)} file(s) match no kernel stem, left in place:", file=sys.stderr)
        for name in unclaimed:
            print(f"    {name}", file=sys.stderr)

    moved = 0
    for kernel in sorted(moves):
        dest = root / kernel
        if dest.exists():
            print(f"  SKIP {kernel}: {dest} already exists (already migrated?)")
            continue
        for name in moves[kernel]:
            src = root / name
            dst = dest / name
            if args.apply:
                dest.mkdir(exist_ok=True)
                subprocess.run(["git", "mv", str(src), str(dst)], check=True)
                moved += 1
            else:
                print(f"  {src}  ->  {dst}")
    if args.apply:
        print(f"[APPLY] moved {moved} files into {len(moves)} kernel folders (staged, not committed).")
        print("Verify: python -c \"from hpcagent_bench.spec import registry; print(len(registry()))\" "
              "and run a foundation e2e before committing.")


if __name__ == "__main__":
    main()
