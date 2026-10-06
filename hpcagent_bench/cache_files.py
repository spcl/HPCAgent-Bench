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

__all__ = ["file_sha256", "json_digest", "replacing", "sha256_hex", "write_atomic"]


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
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def write_atomic(path: pathlib.Path, data: bytes, mode: int | None = None) -> None:
    """Replace ``path`` with ``data`` (:func:`replacing`)."""
    with replacing(path, mode) as tmp:
        tmp.write_bytes(data)
