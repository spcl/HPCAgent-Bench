# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Guards that keep collection, extraction and cleanup code from deleting source data.

Run roots and judge databases are the only copy of a campaign. Every tool that rewrites an output
file or removes a tree calls one of these first:

* :func:`check_output` -- the output must not be, contain, or sit inside any source it reads.
* :func:`safe_rmtree` -- a tree is removed only when it holds no SQLite database, is not a
  protected root (or an ancestor of one) and, inside a protected root, only under an allowlisted
  scratch subdirectory.

Protected roots: the campaign run root (:func:`hpcagent_bench.campaigns.runs_root`), the frozen
observations directory, and every entry of ``$HPCAGENT_BENCH_PROTECTED_ROOTS`` (``os.pathsep``
separated).
"""

import os
import pathlib
import shutil
from collections.abc import Iterable

from hpcagent_bench import frozen_observations

#: Extra protected roots, ``os.pathsep`` separated.
ENV = "HPCAGENT_BENCH_PROTECTED_ROOTS"

#: File suffixes that mark a tree as holding recorded data.
DB_SUFFIXES = frozenset({".db", ".sqlite", ".db-wal", ".sqlite-wal"})

type PathLike = str | os.PathLike[str]


class ProtectedPathError(ValueError):
    """A write or delete would touch source data."""


def protected_roots() -> tuple[pathlib.Path, ...]:
    """Every root whose contents no collection or cleanup tool may delete or overwrite."""
    from hpcagent_bench import campaigns  # lazy: keeps check_output free of the registry imports

    roots = [campaigns.runs_root()]
    frozen = frozen_observations.default_dir()
    if frozen is not None:
        roots.append(frozen)
    roots.extend(pathlib.Path(p) for p in os.environ.get(ENV, "").split(os.pathsep) if p)
    return tuple(_real(r) for r in roots)


def _real(path: PathLike) -> pathlib.Path:
    return pathlib.Path(path).expanduser().resolve()


def _within(path: pathlib.Path, root: pathlib.Path) -> bool:
    return path == root or root in path.parents


def check_output(dest: PathLike, sources: Iterable[PathLike]) -> pathlib.Path:
    """``dest`` resolved; raises :class:`ProtectedPathError` when it overlaps any of ``sources``.

    Overlap is either direction: writing inside a source mixes outputs into inputs (and a rewrite
    deletes the input it replaces), and writing to an ancestor of a source wraps the source in the
    output.
    """
    out = _real(dest)
    for source in sources:
        src = _real(source)
        if _within(out, src) or _within(src, out):
            raise ProtectedPathError(f"output {out} overlaps source {src}; write it outside the sources")
    return out


def holds_database(path: pathlib.Path) -> bool:
    """Whether ``path`` is, or (as a directory) contains, a SQLite database file."""
    if path.is_file() or path.is_symlink():
        return path.suffix in DB_SUFFIXES
    return any(p.suffix in DB_SUFFIXES for p in path.rglob("*") if p.is_file())


def check_removable(path: PathLike, *, allow: Iterable[PathLike] = ()) -> pathlib.Path:
    """``path`` resolved when :func:`safe_rmtree` may remove it; raises :class:`ProtectedPathError`."""
    target = _real(path)
    allowed = [_real(a) for a in allow]
    for root in protected_roots():
        if _within(root, target):
            raise ProtectedPathError(f"refusing to remove {target}: it is (or holds) protected root {root}")
        if _within(target, root) and not any(_within(target, a) for a in allowed):
            raise ProtectedPathError(f"refusing to remove {target}: it lies under protected root {root}")
    if target.exists() and holds_database(target):
        raise ProtectedPathError(f"refusing to remove {target}: it holds a SQLite database")
    return target


def safe_rmtree(path: PathLike, *, allow: Iterable[PathLike] = ()) -> None:
    """``shutil.rmtree`` behind :func:`check_removable`; ``allow`` names scratch dirs inside protected roots."""
    shutil.rmtree(check_removable(path, allow=allow))
