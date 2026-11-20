# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The file primitives every on-disk cache shares: a sha256 digest, and replacing a file atomically.

A cache is read by concurrent jobs and written by jobs that may be killed, so an entry is written beside
its final name and renamed over it: a reader sees the old file or the new one, never half of one.
"""

import contextlib
import hashlib
import json
import os
import pathlib
import uuid
from collections.abc import Iterator

__all__ = ["append_line", "file_sha256", "json_digest", "replacing", "sha256_hex", "write_atomic"]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: pathlib.Path) -> str:
    """The sha256 of ``path``'s bytes, read in blocks."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_digest(value: object) -> str:
    """The sha256 of ``value`` as canonical JSON, so dict insertion order never moves a key."""
    return sha256_hex(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


@contextlib.contextmanager
def replacing(path: pathlib.Path, mode: int | None = None, parent_mode: int = 0o777) -> Iterator[pathlib.Path]:
    """A temporary path beside ``path`` to write into: on a clean exit it replaces ``path`` (with
    permission bits ``mode`` when given), on an exception it is removed. The parent directory is created
    with ``parent_mode`` when missing."""
    path.parent.mkdir(mode=parent_mode, parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        yield tmp
        if mode is not None:
            tmp.chmod(mode)
        pathlib.Path(tmp).replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def write_atomic(path: pathlib.Path, data: bytes, mode: int | None = None) -> None:
    """Replace ``path`` with ``data`` (:func:`replacing`)."""
    with replacing(path, mode) as tmp:
        tmp.write_bytes(data)


def append_line(path: pathlib.Path, line: str) -> None:
    """Append ``line`` and a newline to ``path``, flushed to disk; a torn last line (a process killed mid-write) is
    ended first, so the new line stays readable. The caller serializes writers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        size = handle.seek(0, os.SEEK_END)
        torn = size > 0 and os.pread(handle.fileno(), 1, size - 1) != b"\n"
        handle.write((b"\n" if torn else b"") + line.encode("utf-8") + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
