# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``sqlite3.connect`` as a block that commits AND closes.

A connection's own context manager only commits; the handle it leaves open is garbage-collected later as
a ResourceWarning, which fails a ``-W error`` run and drowns a zero-warning one.
"""

import contextlib
import pathlib
import sqlite3
from collections.abc import Iterator

__all__ = ["connect"]


@contextlib.contextmanager
def connect(path: str | pathlib.Path) -> Iterator[sqlite3.Connection]:
    with contextlib.closing(sqlite3.connect(path)) as conn, conn:
        yield conn
