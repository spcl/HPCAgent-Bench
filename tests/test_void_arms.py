# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``results_db.delete_arms``: an arm declared void leaves no row behind, and nothing of another arm goes."""

import contextlib
import pathlib

from hpcagent_bench.harness import results_db
from tests import results_seed

VOID = "void-arm"
KEPT = "kept-arm"


def test_a_void_arm_leaves_no_row_and_the_other_arm_keeps_every_one(tmp_path: pathlib.Path) -> None:
    db = tmp_path / "r.db"
    shared = "void gemm(void) { /* both */ }"
    void = results_seed.submission(db, f"{VOID}.n0.p0.w0", "gemm", 10, source="void gemm(void) { /* void */ }")
    results_seed.submission(db, f"{KEPT}.n0.p0.w0", "gemm", 11, source=shared)
    results_seed.submission(db, f"{VOID}.n0.p1.w1", "gemm", 12, source=shared)
    with contextlib.closing(results_db.open_db(db)) as conn:
        run = conn.execute("SELECT run_id FROM grades WHERE id = ?", (void,)).fetchone()[0]
        final, _ts = results_db.add_grade(conn, run, "gemm", "final", ts_ms=20, values={"of_grade_id": void})
        results_db.add_cells(conn, final, [{"cell": 0, "ratio": 2.0}])
        conn.commit()
        removed = results_db.delete_arms(conn, [VOID])
        conn.commit()
        left = {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in results_db.TABLES}
        arms = [row[0] for row in conn.execute("SELECT arm FROM arms")]
        texts = [row[0] for row in conn.execute("SELECT text FROM sources")]
    assert removed["grades"] == 3 and removed["runs"] == 2 and removed["arms"] == 1 and removed["grade_cells"] == 1
    assert arms == [KEPT] and texts == [shared]
    assert (left["runs"], left["grades"], left["grade_sources"], left["grade_cells"]) == (1, 1, 1, 0)
