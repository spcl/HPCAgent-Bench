# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Free inodes by removing regenerable tool caches under given roots, and nothing else.

    hpcagent-bench clean caches --root DIR [--root DIR ...]            # a dry run: what would go, by class, with df -i
    hpcagent-bench clean caches --root DIR [--root DIR ...] --apply    # remove it

An allowlist, not a blocklist: only directories named :data:`TOOL_CACHES` (``__pycache__``, ``.ruff_cache``, ...) and
``dacecache*`` build folders are candidates. Checked first, whatever the name: never a virtual environment (``.venv``,
``venv``), the ``uv`` cache, a spack build cache, an image-build or container-image tree, the protected run roots and
archives (:func:`hpcagent_bench.results.data_guard.protected_roots`), nor anything beneath a ``dacecache*`` or a directory a
running job of the user works in (a build folder in use). A symlink is never followed or removed. There is no ``uv cache
clean`` here, on purpose: the ``uv`` cache is mounted into image builds, and cleaning it broke one (2026-11-17); ``clean uv``
refuses.
"""

import argparse
import dataclasses
import fnmatch
import os
import pathlib
import shutil
import subprocess
import sys
from collections.abc import Iterator, Sequence

from hpcagent_bench.results import data_guard

__all__ = ["BUILD_FOLDERS", "NEVER_NAMES", "NEVER_PARTS", "TOOL_CACHES", "Candidate", "candidates", "main"]

#: Directories every tool regenerates on its next run, safe to remove at any time.
TOOL_CACHES = frozenset({"__pycache__", ".ruff_cache", ".pytest_cache", ".mypy_cache", ".hypothesis"})
#: Build folders (DaCe's): regenerable, but in use while a job works in the tree.
BUILD_FOLDERS = "dacecache*"
#: Names that are never touched, nor entered: environments and the caches other work is built from.
NEVER_NAMES = frozenset({".venv", "venv", "uv-cache", "buildcache", "site-packages", ".git"})
#: A path part containing one of these (image builds, container images, run data, archives) is never touched.
NEVER_PARTS = ("image-build", "ce-images", "base-images", "hpcagent-bench-runs", "archives", "aarch64")


@dataclasses.dataclass(frozen=True, slots=True)
class Candidate:
    path: pathlib.Path
    kind: str
    inodes: int


def inodes_of(path: pathlib.Path) -> int:
    """Files and directories under ``path`` (itself counted), without following a link."""
    return 1 + sum(len(dirs) + len(files) for _, dirs, files in os.walk(path, followlinks=False))


def never(path: pathlib.Path, protected: Sequence[pathlib.Path]) -> bool:
    """Whether ``path`` is one nothing may touch or enter."""
    return (
        path.name in NEVER_NAMES
        or any(part in path.name for part in NEVER_PARTS)
        or any(data_guard.within(path, root) for root in protected)
    )


def running_workdirs() -> list[pathlib.Path]:
    """The directories the user's jobs in Slurm work in (``squeue %Z``); none without Slurm."""
    try:
        shown = subprocess.run(
            ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%Z"], capture_output=True, text=True, check=False
        ).stdout
    except FileNotFoundError:
        return []
    return [data_guard.real(line) for line in shown.split() if line]


def candidates(
    roots: Sequence[pathlib.Path],
    protected: Sequence[pathlib.Path] | None = None,
    busy: Sequence[pathlib.Path] | None = None,
) -> Iterator[Candidate]:
    """The removable directories under ``roots``: the allowlisted names, none of them in :func:`never`'s reach and no
    build folder under a ``busy`` (running job's) directory; their trees are not entered."""
    guarded = data_guard.protected_roots() if protected is None else tuple(protected)
    working = running_workdirs() if busy is None else list(busy)
    for root in roots:
        if never(root, guarded):
            continue
        for folder, dirs, _ in os.walk(root, followlinks=False):
            here = pathlib.Path(folder)
            keep: list[str] = []
            for name in sorted(dirs):
                path = here / name
                if path.is_symlink() or never(path, guarded):
                    continue
                if name in TOOL_CACHES:
                    yield Candidate(path, name, inodes_of(path))
                elif fnmatch.fnmatch(name, BUILD_FOLDERS):
                    if not any(data_guard.within(path, work) for work in working):
                        yield Candidate(path, BUILD_FOLDERS, inodes_of(path))
                else:
                    keep.append(name)
            dirs[:] = keep


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hpcagent-bench clean", description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    caches = sub.add_parser("caches", help="remove regenerable tool caches under roots (dry run without --apply)")
    caches.add_argument("--root", type=pathlib.Path, action="append", required=True)
    caches.add_argument("--apply", action="store_true", help="remove them")
    sub.add_parser("uv", help="refused: the uv cache is mounted into image builds")
    args = parser.parse_args(argv)
    if args.command == "uv":
        raise SystemExit(
            "clean: the uv cache is never cleaned (it is mounted into image builds; `uv cache clean` deleted the "
            "mounted directory and broke build 747153). Free inodes with `clean caches`."
        )
    found = list(candidates([data_guard.real(root) for root in args.root]))
    by_kind: dict[str, list[Candidate]] = {}
    for one in found:
        by_kind.setdefault(one.kind, []).append(one)
    for kind, items in sorted(by_kind.items()):
        print(f"{kind}: {len(items)} folders, {sum(item.inodes for item in items)} inodes")
    if args.apply:
        for one in found:
            shutil.rmtree(one.path, ignore_errors=True)
    else:
        print("dry run: nothing removed (--apply removes)")
    for root in args.root:
        print(
            subprocess.run(["df", "-i", str(root)], capture_output=True, text=True, check=False)
            .stdout.strip()
            .splitlines()[-1]
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
