# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A weak/strong scaling curve persists to the results DB (``scaling_points`` + ``scaling_curves``)
and comes back out of the extractor as ``record = "scaling"`` rows, so every scaling figure can be
rebuilt from stored rows instead of from a grade that lived only in memory."""

import contextlib
import json
import math
import pathlib
import sqlite3

import pytest

from hpcagent_bench import observations_extract
from hpcagent_bench.harness import metric, recording
from hpcagent_bench.harness.envelope import Submission
from hpcagent_bench.harness.scoring import Score
from hpcagent_bench.harness.task import Task

KERNEL = "tsvc_2_s212"  # any real, fast-loading kernel: record() loads its spec
RUN_ID = "mlscale-strong-qwen38-hip.n0.p1.w2"
TS = 1_790_000_000_000


def strong_curve() -> metric.ScalingScore:
    """P = 1, 4, 16 measured and P = 8 dropped at build, as metric.scaling_score assembles it from a sweep."""
    curve = metric.scaling_score(
        KERNEL,
        "strong",
        8000,
        {1: 8000, 4: 2500, 16: 1000},
        nodes={1: 1, 4: 1, 8: 2, 16: 4},
        shapes={1: {"N": 64}, 4: {"N": 64}, 8: {"N": 64}, 16: {"N": 64}},
        rank_notes={8: "mpi build failed"},
    )
    assert curve is not None
    return curve


def weak_curve() -> metric.ScalingScore:
    """P = 2 is not a perfect square: rounded, measured, and its note rides on the point."""
    curve = metric.scaling_score(
        KERNEL,
        "weak",
        4000,
        {1: 4000, 2: 4400, 4: 5000},
        work_exponent=2,
        work_ratio={1: 1.0, 2: 2.0164, 4: 4.0},
        shapes={1: {"N": 64}, 2: {"N": 91}, 4: {"N": 128}},
        rank_notes={2: "k=2, m=1.414 -> sizes {'N': 91}, work ratio 2.02 (not a perfect k-th power; rounded)"},
    )
    assert curve is not None
    return curve


def rows(db: pathlib.Path, sql: str) -> list[dict]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql)]
    finally:
        conn.close()


def record(db: pathlib.Path, curve: metric.ScalingScore | None, mode: str, **kw: object) -> int:
    conn = recording.connect(str(db))
    try:
        return recording.record_scaling(conn, run_id=RUN_ID, ts_ms=TS, benchmark=KERNEL, scaling=curve, mode=mode, **kw)
    finally:
        conn.close()


def test_every_requested_rank_count_is_one_row_holes_included(tmp_path: pathlib.Path) -> None:
    """Three measured P and one dropped P are four rows: a curve with a hole must read as a hole."""
    db = tmp_path / "r.db"
    assert record(db, strong_curve(), "strong") == 4
    got = [r["ranks"] for r in rows(db, "SELECT ranks FROM scaling_points ORDER BY ranks")]
    assert got == [1, 4, 8, 16], got


def test_a_dropped_point_has_no_time_and_no_efficiency_only_its_reason(tmp_path: pathlib.Path) -> None:
    db = tmp_path / "r.db"
    record(db, strong_curve(), "strong")
    (hole,) = rows(db, "SELECT * FROM scaling_points WHERE ranks = 8")
    assert (hole["ranked_ns"], hole["achieved_speedup"], hole["ideal_speedup"], hole["efficiency"]) == (
        None,
        None,
        None,
        None,
    ), hole
    assert hole["note"] == "mpi build failed"
    assert hole["nodes"] == 2  # the launch was placed before the build failed
    assert hole["single_rank_ns"] == 8000  # the curve's anchor, so the row still says what it would divide


def test_a_measured_point_stores_the_graders_own_numbers(tmp_path: pathlib.Path) -> None:
    db = tmp_path / "r.db"
    record(db, strong_curve(), "strong")
    (row,) = rows(db, "SELECT * FROM scaling_points WHERE ranks = 4")
    want = metric.scaling_point("strong", 4, 8000, 2500)
    got = (row["single_rank_ns"], row["ranked_ns"], row["achieved_speedup"], row["ideal_speedup"], row["efficiency"])
    assert got == (8000, 2500, want.achieved_speedup, want.ideal_speedup, want.efficiency), row
    assert row["scaling_mode"] == "strong" and row["work_ratio"] is None and row["note"] is None
    assert json.loads(row["shape"]) == {"N": 64}


@pytest.mark.parametrize(("ranks", "nodes"), [(1, 1), (4, 1), (8, 2), (16, 4)])
def test_nodes_is_the_placement_the_sweep_recorded(tmp_path: pathlib.Path, ranks: int, nodes: int) -> None:
    db = tmp_path / "r.db"
    record(db, strong_curve(), "strong")
    (row,) = rows(db, f"SELECT nodes FROM scaling_points WHERE ranks = {ranks}")
    assert row["nodes"] == nodes, row


def test_an_unplaced_point_records_null_nodes_not_a_derived_count(tmp_path: pathlib.Path) -> None:
    """A launcher that places ranks itself reports nothing, and P / ranks-per-node is not a measurement."""
    db = tmp_path / "r.db"
    record(db, weak_curve(), "weak")
    got = [r["nodes"] for r in rows(db, "SELECT nodes FROM scaling_points ORDER BY ranks")]
    assert got == [None, None, None], got


def test_a_weak_point_keeps_its_work_ratio_and_its_rounding_note(tmp_path: pathlib.Path) -> None:
    db = tmp_path / "r.db"
    record(db, weak_curve(), "weak")
    (row,) = rows(db, "SELECT * FROM scaling_points WHERE ranks = 2")
    assert row["scaling_mode"] == "weak" and row["work_ratio"] == 2.0164
    assert "rounded" in row["note"], row
    assert row["efficiency"] == metric.scaling_point("weak", 2, 4000, 4400, work_ratio=2.0164).efficiency


def test_mean_efficiency_and_work_exponent_live_once_per_curve(tmp_path: pathlib.Path) -> None:
    db = tmp_path / "r.db"
    curve = weak_curve()
    record(db, curve, "weak")
    got = rows(db, "SELECT * FROM scaling_curves")
    assert got == [
        {
            "run_id": RUN_ID,
            "ts": TS,
            "benchmark": KERNEL,
            "scaling_mode": "weak",
            "work_exponent": 2,
            "mean_efficiency": curve.mean_efficiency,
        }
    ], got


def test_both_laws_of_one_grade_are_two_curves_side_by_side(tmp_path: pathlib.Path) -> None:
    """An ML grade records BOTH laws under its one stamp: the law is part of the key, so recording
    the weak curve neither replaces nor collides with the strong one, and re-recording one law
    leaves the other alone."""
    db = tmp_path / "r.db"
    record(db, strong_curve(), "strong")
    record(db, weak_curve(), "weak")
    record(db, strong_curve(), "strong")
    points = rows(db, "SELECT scaling_mode, COUNT(*) AS n FROM scaling_points GROUP BY scaling_mode ORDER BY 1")
    assert [(r["scaling_mode"], r["n"]) for r in points] == [("strong", 4), ("weak", len(weak_curve().points))]
    curves = rows(db, "SELECT scaling_mode FROM scaling_curves ORDER BY scaling_mode")
    assert [r["scaling_mode"] for r in curves] == ["strong", "weak"]


def test_re_recording_a_grade_replaces_it_instead_of_duplicating(tmp_path: pathlib.Path) -> None:
    db = tmp_path / "r.db"
    record(db, strong_curve(), "strong")
    assert record(db, strong_curve(), "strong") == 4
    assert rows(db, "SELECT COUNT(*) AS n FROM scaling_points") == [{"n": 4}]
    assert rows(db, "SELECT COUNT(*) AS n FROM scaling_curves") == [{"n": 1}]


def test_a_curve_whose_every_point_dropped_is_still_on_record(tmp_path: pathlib.Path) -> None:
    """No curve survives (scaling None), yet each requested P gets its hole row and no curve row."""
    db = tmp_path / "r.db"
    holes = metric.scaling_drops({}, {2: "unsizable (strong-only)", 4: "unsizable (strong-only)"})
    assert record(db, None, "weak", dropped=holes) == 2
    got = [(r["ranks"], r["note"], r["efficiency"]) for r in rows(db, "SELECT * FROM scaling_points ORDER BY ranks")]
    assert got == [(2, "unsizable (strong-only)", None), (4, "unsizable (strong-only)", None)], got
    assert rows(db, "SELECT COUNT(*) AS n FROM scaling_curves") == [{"n": 0}]


@pytest.mark.parametrize("mode", ["", "Strong", "amdahl"])
def test_a_mode_that_is_not_a_scaling_law_is_refused(tmp_path: pathlib.Path, mode: str) -> None:
    with pytest.raises(ValueError, match="'weak' or 'strong'"):
        record(tmp_path / "r.db", strong_curve(), mode)


def test_a_mode_contradicting_the_curve_is_refused(tmp_path: pathlib.Path) -> None:
    """mode is a real column: a strong curve filed as weak would be scored against the wrong ideal."""
    with pytest.raises(ValueError, match="graded 'strong'"):
        record(tmp_path / "r.db", strong_curve(), "weak")


def test_record_scaling_is_exported_from_recording() -> None:
    from hpcagent_bench.harness.recording import record_scaling

    assert record_scaling is recording.record_scaling


def test_a_db_from_before_the_scaling_tables_gains_them_on_connect(tmp_path: pathlib.Path) -> None:
    """An old results DB opened by new code must be migrated, not broken, and keep its rows."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(recording._BENCHMARKS_DDL + recording._SUBMISSIONS_DDL + recording._ATTEMPTS_DDL)
    conn.execute("INSERT INTO benchmarks(name) VALUES (?)", (KERNEL,))
    conn.commit()
    conn.close()
    assert record(db, strong_curve(), "strong") == 4
    assert rows(db, "SELECT name FROM benchmarks") == [{"name": KERNEL}]


def law_of(curve: metric.ScalingScore) -> metric.LawCurve:
    return metric.LawCurve(curve.mode, curve, (), curve.dropped, {})


def test_record_writes_the_curve_under_the_graded_rows_own_stamp(tmp_path: pathlib.Path) -> None:
    """The curve joins its submission on (run_id, benchmark, ts), so both must carry the same ts."""
    db = tmp_path / "r.db"
    score = Score(True, 0.0, 1000, True, "", baseline_ns=2000, speedup=2.0, public_correct=True, hidden_correct=True)
    table = recording.record(
        score,
        Submission(language="c", source="/* x */", build=[]),
        Task(KERNEL, "restricted", "c"),
        run_id=RUN_ID,
        path=str(db),
        curves=(law_of(strong_curve()),),
    )[0]
    assert table == "submission"
    (sub,) = rows(db, "SELECT ts FROM submissions")
    stamps = {r["ts"] for r in rows(db, "SELECT ts FROM scaling_points")}
    assert stamps == {sub["ts"]}, (stamps, sub)


def test_record_keeps_the_holes_of_a_grade_whose_every_point_dropped(tmp_path: pathlib.Path) -> None:
    """No curve survived for this law; its holes still land under the law's name."""
    db = tmp_path / "r.db"
    score = Score(True, 0.0, 1000, True, "", baseline_ns=2000, speedup=2.0, scaling_mode="strong,weak")
    holes = metric.scaling_drops({}, {2: "unsizable", 4: "unsizable"})
    recording.record(
        score,
        Submission(language="c", source="/* x */", build=[]),
        Task(KERNEL, "restricted", "c"),
        run_id=RUN_ID,
        path=str(db),
        curves=(metric.LawCurve("weak", None, (), holes, {}),),
    )
    got = [(r["ranks"], r["scaling_mode"], r["note"]) for r in rows(db, "SELECT * FROM scaling_points ORDER BY ranks")]
    assert got == [(2, "weak", "unsizable"), (4, "weak", "unsizable")], got


def test_a_grade_without_a_sweep_writes_no_scaling_rows(tmp_path: pathlib.Path) -> None:
    db = tmp_path / "r.db"
    score = Score(True, 0.0, 1000, True, "", baseline_ns=2000, speedup=2.0)
    recording.record(
        score, Submission(language="c", source="/* x */", build=[]), Task(KERNEL, "restricted", "c"), path=str(db)
    )
    assert rows(db, "SELECT COUNT(*) AS n FROM scaling_points") == [{"n": 0}]


def test_shards_merge_their_curves_without_duplicating_a_grade(tmp_path: pathlib.Path) -> None:
    a, b, dest = tmp_path / "a.db", tmp_path / "b.db", tmp_path / "all.db"
    record(a, strong_curve(), "strong")
    record(b, strong_curve(), "strong")  # the same grade seen by two shards
    recording.aggregate(str(dest), [str(a), str(b)])
    assert rows(dest, "SELECT COUNT(*) AS n FROM scaling_points") == [{"n": 4}]
    assert rows(dest, "SELECT COUNT(*) AS n FROM scaling_curves") == [{"n": 1}]


def extracted(db: pathlib.Path) -> list[dict]:
    database = observations_extract.Database(db, "mlscale", db.parent, "700001")
    result = observations_extract.read_db(database, frozenset(), "mlscale-", frozenset(), 0)
    return [r for r in result.observations if r["record"] == "scaling"]


def test_the_extractor_emits_one_scaling_row_per_point_and_hole(tmp_path: pathlib.Path) -> None:
    db = tmp_path / "r.db"
    curve = strong_curve()
    record(db, curve, "strong")
    got = [
        (r["ranks"], r["nodes"], r["scaling_mode"], r["ranked_ns"], r["efficiency"], r["scaling_note"])
        for r in extracted(db)
    ]
    eff = {p.ranks: p.efficiency for p in curve.points}
    assert got == [
        (1, 1, "strong", 8000, eff[1], ""),
        (4, 1, "strong", 2500, eff[4], ""),
        (8, 2, "strong", "", "", "mpi build failed"),
        (16, 4, "strong", 1000, eff[16], ""),
    ], got


def test_every_extracted_scaling_row_carries_its_curves_mean_efficiency(tmp_path: pathlib.Path) -> None:
    db = tmp_path / "r.db"
    curve = strong_curve()
    record(db, curve, "strong")
    assert {r["mean_efficiency"] for r in extracted(db)} == {curve.mean_efficiency}


def test_an_extracted_scaling_row_has_every_column_the_table_declares(tmp_path: pathlib.Path) -> None:
    """A key outside OBSERVATION_FIELDS is dropped silently by the CSV writer."""
    db = tmp_path / "r.db"
    record(db, weak_curve(), "weak")
    row = extracted(db)[0]
    assert set(row) <= set(observations_extract.OBSERVATION_FIELDS), set(row) - set(
        observations_extract.OBSERVATION_FIELDS
    )
    assert json.loads(row["scaling_shape"]) == {"N": 64} and row["ts_ms"] == TS and row["arm"]


def test_the_extracted_efficiency_is_the_one_recomputed_from_the_times(tmp_path: pathlib.Path) -> None:
    """The plotting side REFUSES a recorded eta that disagrees with metric.scaling_point on the same row."""
    db = tmp_path / "r.db"
    record(db, weak_curve(), "weak")
    for row in extracted(db):
        again = metric.scaling_point(
            "weak", row["ranks"], row["single_rank_ns"], row["ranked_ns"], work_ratio=row["work_ratio"]
        )
        assert math.isclose(row["efficiency"], again.efficiency, rel_tol=1e-12), row


@pytest.mark.parametrize("suffix", [".csv", ".db"])
def test_the_scaling_figures_rebuild_the_recorded_curve_from_the_extracted_table(
    tmp_path: pathlib.Path, suffix: str
) -> None:
    """The consumer the table exists for: the figures module reads the extracted file back into the
    same curve the grade measured, holes included, and finds no efficiency it disagrees with."""
    from hpcagent_bench import experiments
    from hpcagent_bench.stats.figures import scaling

    db = tmp_path / "judge.db"
    curve = strong_curve()
    record(db, curve, "strong")
    out = tmp_path / f"observations{suffix}"
    if suffix == ".csv":
        observations_extract.write_csv(out, observations_extract.OBSERVATION_FIELDS, extracted(db))
    else:
        observations_extract.write_db(out, observations_extract.OBSERVATION_FIELDS, extracted(db))
    frame = experiments.read_observations(out)
    (rebuilt,) = scaling.curves(frame)
    assert rebuilt.mode == "strong" and rebuilt.ranks == (1, 4, 16), rebuilt
    assert [p.nodes for p in rebuilt.points] == [1, 1, 4]
    assert [p.efficiency for p in rebuilt.points] == [p.efficiency for p in curve.points]
    assert rebuilt.dropped == ((8, "mpi build failed"),)
    assert scaling.disagreements(frame) == []


def test_the_extractor_reads_a_db_written_before_the_law_joined_the_curve_key(tmp_path: pathlib.Path) -> None:
    """Every results DB since the scaling tables landed holds them in the old shape (no law in
    scaling_curves); the extractor must still read those rows, and a new DB's curves per law."""
    from hpcagent_bench import observations_extract as ox

    old = tmp_path / "old.db"
    with contextlib.closing(sqlite3.connect(old)) as conn:
        conn.execute(recording.SCALING_POINTS_DDL.replace("scaling_mode, ranks)", "ranks)"))
        conn.execute(
            "CREATE TABLE scaling_curves (run_id TEXT, ts INTEGER, benchmark TEXT, work_exponent INTEGER,"
            " mean_efficiency REAL NOT NULL, PRIMARY KEY (run_id, ts, benchmark))"
        )
        conn.execute(
            "INSERT INTO scaling_points(run_id, ts, benchmark, ranks, scaling_mode) VALUES (?,?,?,?,?)",
            (RUN_ID, TS, KERNEL, 1, "strong"),
        )
        conn.execute("INSERT INTO scaling_curves VALUES (?,?,?,?,?)", (RUN_ID, TS, KERNEL, None, 0.5))
        conn.commit()
    new = tmp_path / "new.db"
    record(new, strong_curve(), "strong")
    record(new, weak_curve(), "weak")
    for path, want in ((old, {("strong", "0.5")}), (new, None)):
        with contextlib.closing(sqlite3.connect(path)) as conn:
            conn.row_factory = sqlite3.Row
            db = ox.Database(path, "root", tmp_path, "job")
            got = ox.scaling_rows(conn, db, frozenset(), ("", frozenset()), ({}, {}))
        effs = {(r["scaling_mode"], str(r["mean_efficiency"])) for r in got}
        if want is not None:
            assert effs == want, effs
        else:
            assert {mode for mode, _ in effs} == {"strong", "weak"}
            assert all(eff not in ("", "None") for _, eff in effs), effs
