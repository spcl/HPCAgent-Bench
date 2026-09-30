# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""An agent episode's record, ``tokens.json`` (written by ``hpcagent_bench/cluster/agent_driver.py`` beside the
worker's transcript), as the episode columns of its ``runs`` row.

:func:`ingest` folds a finished job's records into the job's results DB; ``hpcagent_bench/cluster/migrate_db.py``
reads archived records through :func:`episode_values` too, so both apply one rule: token counts are
taken only from a record folded by the current token rule (:data:`MIN_TOKEN_FOLD`), since an older
fold double-counted reasoning.
"""

import contextlib
import json
import pathlib
import sqlite3
from collections.abc import Mapping

from hpcagent_bench import config, fused
from hpcagent_bench.harness import recording, results_db

__all__ = [
    "MIN_TOKEN_FOLD",
    "OUTCOME_COLUMNS",
    "RECORD_GLOB",
    "TOKEN_COLUMNS",
    "episode_values",
    "ingest",
    "read_record",
    "trusted_fold",
]

#: Every worker's record under a job directory: ``agents/node-<n>/problem-<p>-worker-<w>/tokens.json``.
RECORD_GLOB = "agents/*/*/tokens.json"
#: The oldest ``token_fold`` whose counts are trusted (a lower one double-counted reasoning).
MIN_TOKEN_FOLD = 2
#: record key -> ``runs`` column, for how the episode ended (besides its ``result`` text).
OUTCOME_COLUMNS: dict[str, str] = {
    "returncode": "returncode",
    "turns": "turns",
    "wall_ms": "wall_ms",
    "api_ms": "api_ms",
}
#: record key -> ``runs`` column, for what it cost.
TOKEN_COLUMNS: dict[str, str] = {
    "fresh_input": "fresh_input_tokens",
    "cached_input": "cached_input_tokens",
    "output": "output_tokens",
    "thinking_estimate": "thinking_tokens",
    "tokens_billed": "billed_tokens",
    "tokens_effective": "effective_tokens",
    "tokens_billed_crashed": "crashed_billed_tokens",
    "tokens_effective_crashed": "crashed_effective_tokens",
}


def read_record(path: pathlib.Path) -> dict[str, object] | None:
    """The record at ``path``; None when unreadable or not a JSON object."""
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return {str(key): value for key, value in parsed.items()} if isinstance(parsed, dict) else None


def trusted_fold(record: Mapping[str, object]) -> bool:
    """Whether the record's token counts were folded by the current rule."""
    fold = record.get("token_fold")
    return isinstance(fold, int) and not isinstance(fold, bool) and fold >= MIN_TOKEN_FOLD


def whole(value: object) -> int | None:
    """``value`` as an integer column; None for anything that is no number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def episode_values(record: Mapping[str, object]) -> dict[str, results_db.Value]:
    """The ``runs`` columns a record fills: how the episode ended, the kernel it was assigned, how
    often it was relaunched and when its final attempt began, and -- from a trusted fold only -- its
    token counts."""
    result = record.get("result")
    values: dict[str, results_db.Value] = {"result": None if result is None else str(result)}
    values |= {column: whole(record.get(key)) for key, column in OUTCOME_COLUMNS.items()}
    if trusted_fold(record):
        values |= {column: whole(record.get(key)) for key, column in TOKEN_COLUMNS.items()}
    kernel = record.get("kernel")
    values["benchmark"] = str(kernel).rsplit("/", 1)[-1] if kernel else None
    values["relaunches"] = max((whole(record.get("attempts")) or 1) - 1, 0)
    start = whole(record.get("final_attempt_start_ms"))
    values["final_attempt_start_ms"] = start if start else None
    return values


def fill(conn: sqlite3.Connection, run_id: int, values: Mapping[str, results_db.Value]) -> None:
    """Set the episode columns of run ``run_id`` (the record is their one source)."""
    columns = [name for name, value in values.items() if value is not None]
    if columns:
        assignments = ", ".join(f"{name} = ?" for name in columns)
        conn.execute(f"UPDATE runs SET {assignments} WHERE id = ?", (*(values[name] for name in columns), run_id))


def ingest(conn: sqlite3.Connection, job_dir: pathlib.Path) -> tuple[int, int]:
    """Fold every worker record under ``job_dir`` into ``conn``'s runs of its job
    (:func:`recording.job_of_dir`; created, with the
    arm's identity, for an episode that never reached the judge); returns ``(filled, unattributed)``.
    A record naming no ``run_id`` (written before the driver named it) is unattributed. A fused
    wave's record names its setup, whose identity the arm takes."""
    filled = unattributed = 0
    job = recording.job_of_dir(job_dir)
    for path in sorted(job_dir.glob(RECORD_GLOB)):
        record = read_record(path)
        label = str(record.get("run_id") or "") if record is not None else ""
        if record is None or not label:
            unattributed += 1
            continue
        setup = str(record.get("setup") or "")
        scope = config.scoped_environment(fused.judge_overlay(setup)) if setup else contextlib.nullcontext()
        with scope:
            run_id = recording.open_episode(conn, label, job)
        fill(conn, run_id, episode_values(record))
        filled += 1
    conn.commit()
    return filled, unattributed
