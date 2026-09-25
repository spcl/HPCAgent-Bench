# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The per-cell re-timing report refuses to pool rows that are not measurements of the same thing.

A ratio is only comparable to another ratio when the arithmetic that reduced it, the protocol that
took it and the policy that chose its denominator all agree. Pooling across any of them makes the
difference between the POLICIES read as a property of the submissions -- the exact error the
stamps exist to prevent -- so the pooled line is refused rather than drawn.
"""

import contextlib
import gc
import importlib.util
import pathlib
import sqlite3
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("percell_report", REPO / "statistics" / "percell_regrade_report.py")
report = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = report
SPEC.loader.exec_module(report)

COLUMNS = (
    "db, run_id, benchmark, ts_ms, n_cells, n_credited, g_i, gsd_i, s_i, gated, score_rule, "
    "original_speedup, original_reduction, timing_reduction, grading_protocol, baseline_policy, "
    "residency, final, status, reason, job, arm, source_hash, node, commit_sha, regrade_ts"
)


def row(**changes: object) -> dict[str, object]:
    base: dict[str, object] = {
        "db": "d",
        "run_id": "r",
        "benchmark": "k",
        "ts_ms": 1,
        "n_cells": 3,
        "n_credited": 3,
        "g_i": 2.0,
        "gsd_i": 1.2,
        "s_i": 2.0,
        "gated": 0,
        "score_rule": "s-v3",
        "original_speedup": 2.0,
        "original_reduction": "mwd-v2",
        "timing_reduction": "mwd-v2",
        "grading_protocol": "sealed-nonce-v1",
        "baseline_policy": "single-v1",
        "residency": "host",
        "final": 1,
        "status": "graded",
        "reason": "",
        "job": "1",
        "arm": "a",
        "source_hash": "h",
        "node": "nid001",
        "commit_sha": "abc",
        "regrade_ts": 1,
    }
    base.update(changes)
    return base


def write(path: pathlib.Path, rows: list[dict[str, object]]) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    db = path / "regrade-cells-0.db"
    with contextlib.closing(sqlite3.connect(db)) as conn, conn:
        conn.execute(f"CREATE TABLE regrade_tasks ({COLUMNS})")
        for item in rows:
            names = list(item)
            conn.execute(
                f"INSERT INTO regrade_tasks ({', '.join(names)}) VALUES ({','.join('?' * len(names))})",
                [item[name] for name in names],
            )
    return db


@pytest.mark.parametrize("changed", ["timing_reduction", "grading_protocol", "baseline_policy"])
def test_the_pooled_shift_is_refused_when_a_measurement_stamp_disagrees(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], changed: str
) -> None:
    write(tmp_path / "out", [row(), row(ts_ms=2, **{changed: "other"})])
    report.main([str(tmp_path / "out")])
    printed = capsys.readouterr().out
    assert "REFUSED: 2 measurement stamps" in printed, printed


def test_one_stamp_pools_and_reports_the_shift(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The refusal must not fire on rows that ARE comparable, or the report says nothing at all."""
    write(tmp_path / "out", [row(), row(ts_ms=2, g_i=4.0, original_speedup=2.0)])
    report.main([str(tmp_path / "out")])
    printed = capsys.readouterr().out
    assert "REFUSED" not in printed, printed
    assert "n=2" in printed, printed


def test_a_shard_without_the_policy_column_reads_as_the_legacy_policy(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The chunks timed before the policy stamp existed ran under the one policy there was. Reading
    their absence as a blank would pool them with rows whose policy is genuinely unknown, and would
    refuse a comparison that is in fact valid."""
    rows = [row(), row(ts_ms=2)]
    for item in rows:
        del item["baseline_policy"]
    write(tmp_path / "out", rows)
    report.main([str(tmp_path / "out")])
    printed = capsys.readouterr().out
    assert "REFUSED" not in printed, printed
    assert f"baseline_policy={report.LEGACY_BASELINE_POLICY}" in printed, printed


def test_reading_the_rows_closes_every_connection(tmp_path: pathlib.Path) -> None:
    """A connection left to the collector warns at whatever LATER test the collection lands in
    (``-W error`` then fails that test): the reader closes what it opens."""
    write(tmp_path / "out", [row(), row(ts_ms=2)])
    assert len(report.task_rows([tmp_path / "out"])) == 2
    gc.collect()


def test_best_of_v2_and_v3_rows_carry_one_measurement_stamp() -> None:
    """USER 2026-09-24: v2 and v3 are one baseline family, so their rows pool in the shift report;
    USER 2026-09-25: so do best-of-v1 over c-autopar+c+numba and single-v1:vendored. Any other rule
    stays apart."""
    v2 = report.measurement_stamp(row(baseline_policy="best-of-v2:c+numba"))
    assert report.measurement_stamp(row(baseline_policy="best-of-v3:numba+c")) == v2
    assert report.measurement_stamp(row(baseline_policy="best-of-v1:c-autopar+c+numba")) == v2
    assert report.measurement_stamp(row(baseline_policy="single-v1:vendored")) == v2
    assert report.measurement_stamp(row(baseline_policy="best-of-v1:c-autopar+c")) != v2
