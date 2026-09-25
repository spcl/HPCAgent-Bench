# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The extract -> fuse -> write -> load pipeline one experiment's figures read.

Every rule here was a way the old ad-hoc merging produced a plausible wrong number rather than an
error: a frozen row shadowing a live one, a column silently filled with NaN, a retired arm counted."""

import pathlib

import pandas as pd
import pytest

from hpcagent_bench import campaigns, dataset, frozen_observations

ARM = "git-scicomp-qwen38-repo"
RETIRED = "cpf-llr-focus40-qwen38-c-cpfsrc"
FOREIGN = "cpf-llr-focus40-qwen38-c"


def row(job: str, benchmark: str, arm: str = ARM, frozen: str = "0", **extra: object) -> dict[str, object]:
    return {
        "run_root": "git-scicomp-20260917",
        "job": job,
        "record": "submission",
        "arm": arm,
        "benchmark": benchmark,
        "speedup": 2.0,
        frozen_observations.COLUMN: frozen,
        **extra,
    }


@pytest.fixture
def selection(tmp_path: pathlib.Path) -> campaigns.Selection:
    return campaigns.resolve("git-scicomp", root=tmp_path)


def test_a_fused_frame_keeps_the_live_row_and_drops_the_frozen_one_for_the_same_job(
    selection: campaigns.Selection,
) -> None:
    """A job read from both sides would double every one of its kernels, which reads as twice the
    coverage rather than as a merge fault."""
    live = pd.DataFrame([row("100", "dfa")])
    frozen = pd.DataFrame([row("100", "dfa", frozen="1")])
    frame, provenance = dataset.fuse(selection, live, frozen.iloc[0:0])
    assert len(frame) == 1
    assert provenance.live_rows == 1
    assert provenance.frozen_rows == 0


def test_a_frozen_row_keeps_its_flag_through_the_fuse(selection: campaigns.Selection) -> None:
    """Without the flag a reader cannot tell a measurement that still has its judge DB from one
    whose only surviving record is the 2026-09-19 extract."""
    frame, provenance = dataset.fuse(
        selection, pd.DataFrame([row("100", "dfa")]), pd.DataFrame([row("200", "kmp", frozen="1")])
    )
    assert set(frame[frozen_observations.COLUMN]) == {"0", "1"}
    assert provenance.frozen_rows == 1
    assert provenance.frozen_jobs == ("200",)


def test_frozen_rows_missing_a_live_column_raise_instead_of_filling_nan(
    selection: campaigns.Selection,
) -> None:
    """pandas fills an absent column with NaN, and a NaN speedup reads downstream as a kernel
    nobody ran rather than as a column that was never extracted."""
    live = pd.DataFrame([row("100", "dfa", tokens=10)])
    frozen = pd.DataFrame([row("200", "kmp", frozen="1")])
    with pytest.raises(ValueError, match="frozen rows lack 1 live column"):
        dataset.fuse(selection, live, frozen)


def test_a_retired_arm_is_dropped_and_counted_apart_from_a_foreign_one(
    selection: campaigns.Selection,
) -> None:
    """Retired means the user took a real arm out; foreign means another experiment shares the run
    root. Reporting them as one number hides which of the two shrank a population."""
    live = pd.DataFrame([row("100", "dfa"), row("101", "dfa", arm=RETIRED), row("102", "dfa", arm=FOREIGN)])
    frame, provenance = dataset.fuse(selection, live, pd.DataFrame())
    assert list(frame["arm"]) == [ARM]
    assert (provenance.dropped_retired, provenance.dropped_foreign) == (0, 2)


def test_every_row_carries_the_time_it_was_extracted(selection: campaigns.Selection) -> None:
    """Two extractions of one experiment were previously told apart only by file mtime, which a
    copy destroys."""
    frame, provenance = dataset.fuse(selection, pd.DataFrame([row("100", "dfa")]), pd.DataFrame())
    assert set(frame[dataset.EXTRACTED_AT]) == {provenance.extracted_at}
    assert frame[dataset.EXPERIMENT_COLUMN].eq("git-scicomp").all()


def test_a_frame_written_as_a_db_and_as_a_csv_reads_back_the_same(
    selection: campaigns.Selection, tmp_path: pathlib.Path
) -> None:
    """A figure takes either file and must not be able to tell which it was given."""
    frame, provenance = dataset.fuse(selection, pd.DataFrame([row("100", "dfa"), row("100", "kmp")]), pd.DataFrame())
    assert provenance.live_rows == 2
    dataset.write_db(frame, tmp_path / "x.db")
    dataset.write_csv(frame, tmp_path / "x.csv")
    from_db, from_csv = dataset.load(tmp_path / "x.db"), dataset.load(tmp_path / "x.csv")
    assert list(from_db["benchmark"]) == list(from_csv["benchmark"]) == ["dfa", "kmp"]
    assert list(from_db["arm"]) == list(from_csv["arm"])


def test_writing_a_db_twice_replaces_it_rather_than_appending(
    selection: campaigns.Selection, tmp_path: pathlib.Path
) -> None:
    """An appending write doubled a re-extracted experiment, and the duplicate rows are identical,
    so nothing downstream could flag them."""
    frame, provenance = dataset.fuse(selection, pd.DataFrame([row("100", "dfa")]), pd.DataFrame())
    assert provenance.live_rows == 1
    dataset.write_db(frame, tmp_path / "x.db")
    dataset.write_db(frame, tmp_path / "x.db")
    assert len(dataset.load(tmp_path / "x.db")) == 1


def test_a_row_on_a_kernel_outside_the_roster_is_dropped_and_counted(tmp_path: pathlib.Path) -> None:
    """The SciComp waves served scicomp40 plus the 09-13 kernels; the campaigns name scicomp35, and a
    figure counts an arm over every kernel its rows touch, so an atax row must not reach it."""
    selection = campaigns.resolve("scicomp-focus40", root=tmp_path)
    arm = "scicomp-perf-playbook-qwen38-plain"
    live = pd.DataFrame([row("100", "gemm", arm=arm), row("101", "atax", arm=arm)])
    frame, provenance = dataset.fuse(selection, live, pd.DataFrame())
    assert list(frame["benchmark"]) == ["gemm"]
    assert provenance.dropped_off_roster == 1
