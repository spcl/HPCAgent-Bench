# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``studies``: which files are judge records, and the rules ``read_observations`` applies to the rows it reads
(foreign kernels, relaunches, cancellations, ad-hoc rows). A row's identity is its setup's recorded row, read as
it is."""

import contextlib
import pathlib
import sqlite3
import warnings

import pandas as pd
import pytest

from hpcagent_bench import run_layout
from hpcagent_bench.results import studies


@pytest.mark.parametrize("merge_failed", [False, True])
def test_a_job_marked_merge_failed_is_read_from_its_shards_not_its_results_db(
    tmp_path: pathlib.Path, merge_failed: bool
) -> None:
    """``MERGE_FAILED`` stays until a merge succeeds, so a ``results.db`` beside it may hold only some
    shards: reading it and skipping the shards silently dropped every other rank's grades."""
    merged = tmp_path / run_layout.MERGED_DB_NAME
    shard = tmp_path / run_layout.JUDGE_DIRNAME / f"{run_layout.RANK_DIR_PREFIX}0" / "hpcagent_bench0.db"
    shard.parent.mkdir(parents=True)
    merged.touch()
    shard.touch()
    if merge_failed:
        (tmp_path / "MERGE_FAILED").touch()  # the marker run_cluster.sh writes

    assert studies.judge_database(merged) is not merge_failed
    assert studies.judge_database(shard) is merge_failed


def foreign_kernel_frame() -> pd.DataFrame:
    """Task w38 was given ``wf_diff_skew``; its agent also scored ``wf_triangular``, the kernel task
    w39 was given. Run w40 has judge rows and no task row (extracted before task records)."""
    common = {"run_root": "r", "job": "636537", "setup": "a"}
    return pd.DataFrame(
        [
            {**common, "episode_id": "a.n0.p38.w38", "row_kind": "episode", "kernel": "wf_diff_skew"},
            {**common, "episode_id": "a.n0.p38.w38", "row_kind": "call", "kernel": "wf_diff_skew"},
            {**common, "episode_id": "a.n0.p38.w38", "row_kind": "call", "kernel": "wf_triangular"},
            {**common, "episode_id": "a.n0.p39.w39", "row_kind": "episode", "kernel": "wf_triangular"},
            {**common, "episode_id": "a.n0.p39.w39", "row_kind": "submission", "kernel": "wf_triangular"},
            {**common, "episode_id": "a.n0.p40.w40", "row_kind": "call", "kernel": "tsvc_2_s115"},
        ]
    )


def test_a_judge_row_naming_another_tasks_kernel_is_dropped_with_a_warning() -> None:
    """Spec X6: the agent sent another kernel's name, so the row is no row of any task on that
    kernel -- kept, it would be the latest task on wf_triangular and hide w39's answer."""
    with pytest.warns(UserWarning, match="dropped 1 judge row"):
        kept = studies.drop_foreign_kernel_rows(foreign_kernel_frame())
    assert ("a.n0.p38.w38", "wf_triangular") not in set(zip(kept.episode_id, kept.kernel, strict=True))
    assert len(kept) == 5


def test_a_run_without_a_task_row_keeps_its_judge_rows() -> None:
    """A run extracted before task records has no kernel of record to compare against, so its rows
    are left as they are rather than guessed foreign."""
    frame = foreign_kernel_frame()
    frame = frame[frame.row_kind != "episode"]
    kept = studies.drop_foreign_kernel_rows(frame)
    assert len(kept) == len(frame)


def relaunched_frame() -> pd.DataFrame:
    """Task w38 crashed at 500 and its relaunch started at 1000; the grades at 200 and 700 were
    scored on the workspace that relaunch deleted. Task w39 never relaunched (no stamp)."""
    common = {"run_root": "r", "job": "636537", "setup": "a", "kernel": "gemm"}
    return pd.DataFrame(
        [
            {
                **common,
                "episode_id": "a.n0.p38.w38",
                "row_kind": "episode",
                "ts_ms": 100,
                "episode_final_attempt_start_ms": 1000,
            },
            {
                **common,
                "episode_id": "a.n0.p38.w38",
                "row_kind": "call",
                "ts_ms": 200,
                "episode_final_attempt_start_ms": "",
            },
            {
                **common,
                "episode_id": "a.n0.p38.w38",
                "row_kind": "submission",
                "ts_ms": 700,
                "episode_final_attempt_start_ms": "",
            },
            {
                **common,
                "episode_id": "a.n0.p38.w38",
                "row_kind": "submission",
                "ts_ms": 1200,
                "episode_final_attempt_start_ms": "",
            },
            {
                **common,
                "episode_id": "a.n0.p39.w39",
                "row_kind": "episode",
                "ts_ms": 100,
                "episode_final_attempt_start_ms": 0,
            },
            {
                **common,
                "episode_id": "a.n0.p39.w39",
                "row_kind": "call",
                "ts_ms": 200,
                "episode_final_attempt_start_ms": "",
            },
        ]
    )


def test_a_judge_row_from_before_the_tasks_final_attempt_is_dropped_with_a_warning() -> None:
    """Spec X7: a fresh relaunch deleted what those grades were given, so they are no answer of the
    task that finished -- kept, the 700 submission would be its answer (R1-R2) and the 200 call
    would date its start (R3)."""
    with pytest.warns(UserWarning, match="dropped 2 judge row"):
        kept = studies.drop_pre_relaunch_rows(relaunched_frame())
    assert kept.ts_ms.tolist() == [100, 1200, 100, 200]


def scored_relaunch(early: float, late: float | None) -> pd.DataFrame:
    """One task relaunched at ts 1000: an earlier attempt answering ``early`` at ts 700, and the final
    attempt answering ``late`` at ts 1200 (none when None)."""
    common = {"run_root": "r", "job": "636537", "setup": "a", "kernel": "gemm", "episode_id": "a.n0.p38.w38"}
    rows = [
        {
            **common,
            "row_kind": "episode",
            "ts_ms": 100,
            "episode_final_attempt_start_ms": 1000,
            "speedup": None,
            "timing_suspect": 0,
        },
        {
            **common,
            "row_kind": "submission",
            "ts_ms": 700,
            "episode_final_attempt_start_ms": "",
            "speedup": early,
            "timing_suspect": 0,
        },
    ]
    if late is not None:
        rows.append(
            {
                **common,
                "row_kind": "submission",
                "ts_ms": 1200,
                "episode_final_attempt_start_ms": "",
                "speedup": late,
                "timing_suspect": 0,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(("early", "late", "kept"), [
    (8.0, 2.0, [100, 700]),  # the earlier attempt answered better: it stands
    (2.0, 8.0, [100, 1200]),  # the final attempt answered better: the earlier reading
    (8.0, None, [100, 700]),  # the final attempt answered nothing: the earlier answer is not lost
])  # fmt: skip
def test_a_relaunched_task_keeps_its_best_attempts_answer(early: float, late: float | None, kept: list[int]) -> None:
    """A task's answer is its best verified answer over its attempts, so a crash
    after a good answer does not turn the kernel unsolved."""
    frame = scored_relaunch(early, late)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = studies.drop_pre_relaunch_rows(frame)
    assert rows.ts_ms.tolist() == kept
    # A drop is always warned about, and only a drop.
    assert bool(caught) == (len(rows) < len(frame)), [str(w.message) for w in caught]


def test_a_suspect_earlier_answer_does_not_beat_the_final_attempt() -> None:
    """The judge's plausibility flag holds across attempts too: a flagged answer is no answer."""
    frame = scored_relaunch(900.0, 2.0)
    frame.loc[frame.ts_ms == 700, "timing_suspect"] = 1
    with pytest.warns(UserWarning, match="spec X7"):
        assert studies.drop_pre_relaunch_rows(frame).ts_ms.tolist() == [100, 1200]


def test_a_task_that_never_relaunched_keeps_every_row() -> None:
    """No stamp, nothing wiped, nothing to cut -- the same reading as before X7 existed."""
    frame = relaunched_frame()
    frame = frame[frame.episode_id == "a.n0.p39.w39"]
    assert len(studies.drop_pre_relaunch_rows(frame)) == len(frame)


def test_the_task_start_is_taken_over_the_rows_x7_kept() -> None:
    """R3 reads ts_ms off the frame ``read_observations`` returns, so a task's start is the start of
    its FINAL attempt and a rerun cannot be dated by an attempt that was thrown away."""
    with pytest.warns(UserWarning, match="spec X7"):
        kept = studies.drop_pre_relaunch_rows(relaunched_frame())
    relaunched = kept[kept.episode_id == "a.n0.p38.w38"]
    assert relaunched.ts_ms.min() == 100  # the task row's own stamp, the only pre-cut row kept


def cancelled_frame() -> pd.DataFrame:
    """Task w38's agent was still working when the job went down; task w39's finished."""
    common = {"run_root": "r", "job": "636537", "setup": "a", "kernel": "gemm"}
    return pd.DataFrame(
        [
            {**common, "episode_id": "a.n0.p38.w38", "row_kind": "episode", "episode_cancelled": "1", "tokens": 900},
            {**common, "episode_id": "a.n0.p38.w38", "row_kind": "call", "episode_cancelled": "", "tokens": 400},
            {**common, "episode_id": "a.n0.p39.w39", "row_kind": "episode", "episode_cancelled": "0", "tokens": 800},
            {**common, "episode_id": "a.n0.p39.w39", "row_kind": "submission", "episode_cancelled": "", "tokens": 700},
        ]
    )


def test_every_row_of_a_cancelled_task_is_dropped_with_a_warning() -> None:
    """Spec X8: the job ended the agent mid-task, so its rows report part of an episode and its
    token total prices part of one. The task row goes too -- a partial cost reported as a cheap setup
    is exactly what X8 exists to keep out."""
    with pytest.warns(UserWarning, match="dropped 2 row"):
        kept = studies.drop_cancelled_episode_rows(cancelled_frame())
    assert kept.episode_id.unique().tolist() == ["a.n0.p39.w39"]


def test_a_frame_without_a_cancelled_column_is_left_alone() -> None:
    """Extractions predating the flag say nothing about cancellation, and a guess is not a record."""
    frame = cancelled_frame().drop(columns=["episode_cancelled"])
    assert len(studies.drop_cancelled_episode_rows(frame)) == len(frame)


def adhoc_frame() -> pd.DataFrame:
    """The production shape: worker w4 graded under its own episode id, and a curl without one filed two
    ``tsvc_2_s323`` grades under the judge's ``adhoc`` default."""
    common = {"run_root": "r", "job": "640078", "row_kind": "submission"}
    return pd.DataFrame(
        [
            {**common, "episode_id": "a.n0.p4.w4", "setup": "a", "kernel": "tsvc_2_s1113"},
            {**common, "episode_id": "adhoc", "setup": "adhoc", "kernel": "tsvc_2_s323"},
            {**common, "episode_id": "adhoc", "setup": "adhoc", "kernel": "tsvc_2_s323"},
        ]
    )


def test_every_row_stored_under_adhoc_is_dropped_with_a_warning() -> None:
    """A grade filed with no episode id has no episode identity, so it answers no setup's kernel, and the
    kernel is owed a rerun instead."""
    with pytest.warns(UserWarning, match="dropped 2 row"):
        kept = studies.drop_adhoc_rows(adhoc_frame())
    assert kept.episode_id.tolist() == ["a.n0.p4.w4"]


def test_read_observations_never_returns_an_adhoc_row(tmp_path: pathlib.Path) -> None:
    """Every figure reads through here."""
    path = tmp_path / "obs.csv"
    adhoc_frame().to_csv(path, index=False)
    with pytest.warns(UserWarning, match="stored under episode id 'adhoc'"):
        frame = studies.read_observations(path)
    assert frame.episode_id.tolist() == ["a.n0.p4.w4"]


def graded_episode(kernel: str, graded: list[tuple[str, str]]) -> pd.DataFrame:
    """One episode's task row, a call, and its graded ``/submit`` rows as ``(record, reason)`` in the
    order the agent sent them (ts 200, 300, ...)."""
    common = {"run_root": "r", "job": "648827", "episode_id": "a.n0.p2.w2", "setup": "a", "kernel": kernel}
    rows = [{**common, "row_kind": "episode", "ts_ms": 100, "reason": ""}, {**common, "row_kind": "call", "ts_ms": 150}]
    rows += [
        {**common, "row_kind": record, "ts_ms": 200 + 100 * index, "attempt_index": index + 1, "reason": reason}
        for index, (record, reason) in enumerate(graded)
    ]
    return pd.DataFrame(rows)


def graded_stamps(frame: pd.DataFrame) -> list[int]:
    """The ``ts_ms`` of every graded row left, in order."""
    return frame[frame.row_kind.isin(("submission", "attempt"))].ts_ms.tolist()


@pytest.mark.parametrize("kernel", ["xsbench", "tsvc_2_s252", "argmax_over_a_dimension"])
def test_read_observations_keeps_every_graded_submit_of_an_episode(tmp_path: pathlib.Path, kernel: str) -> None:
    """Every track answers with the last verified submission, as the multi-submission prompt tells the
    agent (population.last_per_episode): no later /submit is dropped, scientific computing included."""
    path = tmp_path / "obs.csv"
    graded_episode(kernel, [("attempt", "incorrect"), ("submission", ""), ("submission", "")]).to_csv(path, index=False)
    assert graded_stamps(studies.read_observations(path)) == [200, 300, 400]


def test_a_task_whose_job_was_never_recorded_is_labelled_not_refused() -> None:
    """A migrated episode with no Slurm job reads back with a missing ``job``; its task still has one
    label, joined with an empty job rather than raising on the missing value."""
    rows = pd.DataFrame({"run_root": ["r", "r"], "job": [pd.NA, "7"], "episode_id": ["w0", "w0"]}, dtype="string")
    assert studies.episode_labels(rows).tolist() == ["r\x1f\x1fw0", "r\x1f7\x1fw0"]


@pytest.mark.parametrize("directory", ["run#2", "50%25", "a?b"])
def test_read_table_opens_a_db_under_a_directory_sqlite_uris_would_misread(
    tmp_path: pathlib.Path, directory: str
) -> None:
    """A run directory named ``run#2`` or ``50%25`` is a fragment or an escape in a raw ``file:`` URI;
    the read-only open must still read that very file."""
    path = tmp_path / directory / "obs.db"
    path.parent.mkdir()
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.execute("INSERT INTO t VALUES (7)")
        conn.commit()
    assert studies.read_table(path, "t").x.tolist() == [7]
