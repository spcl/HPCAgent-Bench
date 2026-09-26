# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Collect a campaign's recorded data into one self-describing directory, verify it, archive it.

    hpcagent-bench collect copy --out $DATA          # run roots + their DBs + frozen observations
    hpcagent-bench collect verify $DATA              # re-hash every file, quick_check every DB
    hpcagent-bench collect archive $DATA             # verify, then $DATA.tar.zst (or .tar.gz)

Copy-only: sources are opened read-only, nothing is moved or deleted, and an output that overlaps
a source is refused (:func:`hpcagent_bench.data_guard.check_output`). Deleting the sources after a
verified archive is a separate, manual step.

Three kinds of source (:class:`DataSource`), each copied under ``<out>/<kind>/<root name>/``:

* ``runs``        campaign run roots: run metadata (arm env, prompts, token files, JSON/JSONL/CSV
                  records, mlscale observations) and every SQLite DB, agent homes and caches skipped;
* ``db``          directories whose SQLite DBs are wanted alone (regrade shards, mlscale grades);
* ``frozen-csv``  frozen extracted observations and sweep CSVs (``hpcagent_bench.frozen_observations``).

A DB is copied with SQLite's online backup API, so a DB a job is still writing arrives as one
consistent snapshot. ``<out>/env.sh`` points the extractor at the copy.
"""

import argparse
import concurrent.futures
import contextlib
import dataclasses
import enum
import fnmatch
import hashlib
import pathlib
import shutil
import sqlite3
import subprocess
import sys
import tarfile
from collections.abc import Iterable, Iterator, Sequence

from hpcagent_bench import campaigns, data_guard, frozen_observations, paths

__all__ = [
    "DB_SUFFIXES",
    "FROZEN_SUFFIXES",
    "RUN_FILE_GLOBS",
    "SKIP_DIRS",
    "SKIP_DIR_GLOBS",
    "SOURCES",
    "SUMS",
    "DataSource",
    "Root",
    "archive",
    "build_parser",
    "check_roots",
    "copy",
    "copy_db",
    "copy_file",
    "env_script",
    "files_of",
    "is_db",
    "main",
    "quick_check",
    "roots_of",
    "sha256",
    "skipped_dir",
    "verify",
    "walk",
    "wanted",
    "write_sums",
]


class DataSource(enum.Enum):
    """Which kind of source a collected root is; the value is its directory under ``<out>``."""

    RUNS = "runs"
    DB = "db"
    FROZEN_CSV = "frozen-csv"


#: Directory names never descended into: agent homes, model and build caches.
SKIP_DIRS = frozenset({"home", ".cache", "vllm", "dacecache", "__pycache__", ".git", ".frozen", ".frozen-store"})
#: Directory-name patterns never descended into.
SKIP_DIR_GLOBS = ("rocprof_out*", "dacecache-*", "dbg*")
#: SQLite database files (their -wal/-shm sidecars are folded in by the backup).
DB_SUFFIXES = frozenset({".db", ".sqlite"})
#: Run-root files kept besides the DBs.
RUN_FILE_GLOBS = (".env", ".env.*", "*.resolved", "*.json", "*.jsonl", "*.csv", "prompt.txt")
#: Frozen-root files kept besides the DBs.
FROZEN_SUFFIXES = frozenset({".csv", ".tsv", ".json", ".jsonl", ".txt", ".md"})

SUMS = "SHA256SUMS"
SOURCES = "SOURCES.tsv"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Root:
    """One source root and where its copy lands."""

    kind: DataSource
    path: pathlib.Path

    @property
    def dest(self) -> pathlib.PurePosixPath:
        return pathlib.PurePosixPath(self.kind.value, self.path.name)


def is_db(rel: pathlib.PurePath) -> bool:
    return rel.suffix in DB_SUFFIXES


def skipped_dir(name: str) -> bool:
    return name in SKIP_DIRS or any(fnmatch.fnmatch(name, g) for g in SKIP_DIR_GLOBS)


def wanted(kind: DataSource, rel: pathlib.PurePath) -> bool:
    """Whether the file at ``rel`` (relative to its root) is collected for a root of ``kind``."""
    if any(skipped_dir(part) for part in rel.parts[:-1]):
        return False
    if is_db(rel):
        return True
    match kind:
        case DataSource.RUNS:
            return "observations" in rel.parts[:-1] or any(fnmatch.fnmatch(rel.name, g) for g in RUN_FILE_GLOBS)
        case DataSource.DB:
            return False
        case DataSource.FROZEN_CSV:
            return rel.suffix in FROZEN_SUFFIXES


def walk(root: Root) -> Iterator[pathlib.PurePosixPath]:
    """Every collected file of ``root``, relative to it, pruning skipped directories."""
    for directory, dirs, files in root.path.walk():
        dirs[:] = sorted(d for d in dirs if not skipped_dir(d))
        for name in sorted(files):
            rel = pathlib.PurePosixPath((directory / name).relative_to(root.path).as_posix())
            if wanted(root.kind, rel) and not (directory / name).is_symlink():
                yield rel


def copy_db(src: pathlib.Path, dst: pathlib.Path) -> None:
    """A consistent snapshot of ``src`` at ``dst``, read-only on the source.

    Falls back to copying the file with its -wal/-shm sidecars when SQLite cannot open the source
    read-only (a WAL DB in a directory the reader cannot write)."""
    try:
        with (
            contextlib.closing(sqlite3.connect(f"{src.as_uri()}?mode=ro", uri=True, timeout=60)) as source,
            contextlib.closing(sqlite3.connect(dst)) as target,
        ):
            source.backup(target)
            target.execute("PRAGMA journal_mode=DELETE")  # one self-contained file, no -wal/-shm
    except sqlite3.Error:
        dst.unlink(missing_ok=True)
        for suffix in ("", "-wal", "-shm"):
            side = src.with_name(src.name + suffix)
            if side.is_file():
                shutil.copy2(side, dst.with_name(dst.name + suffix))


def copy_file(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if is_db(dst):
        copy_db(src, dst)
    else:
        shutil.copy2(src, dst)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def roots_of(args: argparse.Namespace) -> list[Root]:
    roots = [Root(kind=DataSource.RUNS, path=pathlib.Path(p)) for p in args.runs or [campaigns.runs_root()]]
    roots += [Root(kind=DataSource.DB, path=pathlib.Path(p)) for p in args.db_root]
    frozen = frozen_observations.resolve(args.frozen_observations)
    roots += [Root(kind=DataSource.FROZEN_CSV, path=p) for p in ([frozen] if frozen else [])]
    roots += [Root(kind=DataSource.FROZEN_CSV, path=pathlib.Path(p)) for p in args.csv_root]
    return roots


def check_roots(roots: Sequence[Root], out: pathlib.Path) -> None:
    """Refuse missing roots, two roots landing on one destination, and an output overlapping a root."""
    seen: dict[pathlib.PurePosixPath, pathlib.Path] = {}
    for root in roots:
        if not root.path.is_dir():
            raise SystemExit(f"{root.kind.value} root {root.path} is not a directory")
        if root.dest in seen:
            raise SystemExit(f"{root.path} and {seen[root.dest]} both land on {root.dest}; collect them separately")
        seen[root.dest] = root.path
    try:
        data_guard.check_output(out, [r.path for r in roots])
    except data_guard.ProtectedPathError as exc:
        raise SystemExit(str(exc)) from exc
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; collect into a fresh directory")


def env_script(roots: Sequence[Root]) -> str:
    """``env.sh`` for the unpacked copy: the extractor's run roots and frozen directory."""
    lines = ['DATA=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)', "export DATA"]
    runs = [r for r in roots if r.kind is DataSource.RUNS]
    if runs:
        lines.append(f'export RUNS="$DATA/{runs[0].dest}"')
    frozen = [r for r in roots if r.kind is DataSource.FROZEN_CSV]
    lines.append(f'export {frozen_observations.ENV}="{f"$DATA/{frozen[0].dest}" if frozen else ""}"')
    return "\n".join(lines) + "\n"


def copy(roots: Sequence[Root], out: pathlib.Path, *, threads: int = 8) -> int:
    """Copy every collected file of ``roots`` under ``out``; write SOURCES.tsv, env.sh, SHA256SUMS."""
    check_roots(roots, out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(r.path.absolute() / rel, out / r.dest / rel) for r in roots for rel in walk(r)]
    with concurrent.futures.ThreadPoolExecutor(threads) as pool:
        list(pool.map(lambda job: copy_file(*job), jobs))
    (out / SOURCES).write_text("".join(f"{r.kind.value}\t{r.dest}\t{r.path.resolve()}\n" for r in roots))
    (out / "env.sh").write_text(env_script(roots))
    commit = subprocess.run(
        ["git", "-C", str(paths.repo_root()), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    (out / "COMMIT").write_text(f"{commit or 'unknown'}\n")
    write_sums(out, threads=threads)
    return len(jobs)


def files_of(out: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in out.rglob("*") if p.is_file() and p.name != SUMS)


def write_sums(out: pathlib.Path, *, threads: int = 8) -> None:
    files = files_of(out)
    with concurrent.futures.ThreadPoolExecutor(threads) as pool:
        digests = list(pool.map(sha256, files))
    (out / SUMS).write_text(
        "".join(f"{d}  {p.relative_to(out).as_posix()}\n" for d, p in zip(digests, files, strict=True))
    )


def verify(out: pathlib.Path, *, threads: int = 8) -> list[str]:
    """Problems with a collected directory: checksum mismatches, missing/extra files, corrupt DBs."""
    sums = out / SUMS
    if not sums.is_file():
        return [f"{sums} missing"]
    expected = {rel: digest for digest, rel in (line.split("  ", 1) for line in sums.read_text().splitlines() if line)}
    present = {p.relative_to(out).as_posix(): p for p in files_of(out)}
    problems = [f"missing {rel}" for rel in sorted(expected.keys() - present.keys())]
    problems += [f"not in {SUMS}: {rel}" for rel in sorted(present.keys() - expected.keys())]
    common = sorted(expected.keys() & present.keys())
    with concurrent.futures.ThreadPoolExecutor(threads) as pool:
        digests = list(pool.map(lambda rel: sha256(present[rel]), common))
    problems += [f"checksum {rel}" for rel, got in zip(common, digests, strict=True) if got != expected[rel]]
    for rel in common:
        if is_db(pathlib.PurePath(rel)):
            problems += [f"sqlite {rel}: {msg}" for msg in quick_check(present[rel])]
    return problems


def quick_check(db: pathlib.Path) -> list[str]:
    try:
        with contextlib.closing(sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True)) as conn:
            result = [row[0] for row in conn.execute("PRAGMA quick_check")]
    except sqlite3.Error as exc:
        return [str(exc)]
    return [] if result == ["ok"] else result


def archive(out: pathlib.Path) -> pathlib.Path:
    """``out`` as a tar next to it (zstd when installed, else gzip). ``out`` itself stays."""
    if shutil.which("zstd"):
        target = out.with_name(out.name + ".tar.zst")
        subprocess.run(["tar", "-C", str(out.parent), "-I", "zstd -T0 -10", "-cf", str(target), out.name], check=True)
    else:
        target = out.with_name(out.name + ".tar.gz")
        with tarfile.open(target, "w:gz") as tar:
            tar.add(out, arcname=out.name)
    return target


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hpcagent-bench collect", description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="action", required=True)
    cp = sub.add_parser("copy", help="copy the sources into a fresh directory and checksum it")
    cp.add_argument("--out", required=True, type=pathlib.Path, help="fresh output directory")
    cp.add_argument(
        "--runs",
        action="append",
        default=[],
        metavar="DIR",
        help="run root; repeatable (default the campaign runs root)",
    )
    cp.add_argument("--db-root", action="append", default=[], metavar="DIR", help="directory of SQLite DBs; repeatable")
    cp.add_argument(
        "--frozen-observations",
        default=None,
        metavar="DIR",
        help=f"frozen observations (default ${frozen_observations.ENV}; '' collects none)",
    )
    cp.add_argument("--csv-root", action="append", default=[], metavar="DIR", help="frozen CSV root; repeatable")
    for p in (
        cp,
        *(sub.add_parser(name, help=h) for name, h in (("verify", "re-check a copy"), ("archive", "verify, then tar"))),
    ):
        p.add_argument("--threads", type=int, default=8)
        if p is not cp:
            p.add_argument("out", type=pathlib.Path)
    return ap


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(None if argv is None else list(argv))
    match args.action:
        case "copy":
            n = copy(roots_of(args), args.out, threads=args.threads)
            print(f"collected {n} files into {args.out}")
            return 0
        case "verify" | "archive":
            problems = verify(args.out, threads=args.threads)
            for problem in problems:
                print(problem, file=sys.stderr)
            if problems:
                return 1
            print(f"verified {args.out}")
            if args.action == "archive":
                print(f"archived {archive(args.out)}; the sources are untouched")
            return 0
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
