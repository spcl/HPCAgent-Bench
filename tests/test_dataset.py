# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The extract -> select -> write -> load pipeline one study's figures read.

Every rule here was a way the old ad-hoc merging produced a plausible wrong number rather than an
error: a retired setup counted, a duplicated write, a kernel outside the tag."""

import pathlib

import pandas as pd
import pytest

from hpcagent_bench import experiments, dataset

SETUP = "gitscicomp10-qwen38-c-repo"
RETIRED = "llr40-qwen38-c-unionalpha"
FOREIGN = "llr40-qwen38-c"


def row(job: str, kernel: str, setup: str = SETUP, **extra: object) -> dict[str, object]:
    return {
        "run_root": "gitscicomp10-20260917",
        "job": job,
        "row_kind": "submission",
        "setup": setup,
        "kernel": kernel,
        "speedup": 2.0,
        **extra,
    }


@pytest.fixture
def selection(tmp_path: pathlib.Path) -> experiments.Selection:
    return experiments.resolve("gitscicomp10", root=tmp_path)


def test_a_retired_setup_is_dropped_and_counted_apart_from_a_foreign_one(
    selection: experiments.Selection,
) -> None:
    """Retired means the user took a real setup out; foreign means another study shares the run
    root. Reporting them as one number hides which of the two shrank a population."""
    live = pd.DataFrame([row("100", "dfa"), row("101", "dfa", setup=RETIRED), row("102", "dfa", setup=FOREIGN)])
    frame, provenance = dataset.select(selection, live)
    assert list(frame["setup"]) == [SETUP]
    assert (provenance.dropped_retired, provenance.dropped_foreign) == (0, 2)


def test_every_row_carries_the_time_it_was_extracted(selection: experiments.Selection) -> None:
    """Two extractions of one study were previously told apart only by file mtime, which a
    copy destroys."""
    frame, provenance = dataset.select(selection, pd.DataFrame([row("100", "dfa")]))
    assert set(frame[dataset.EXTRACTED_AT]) == {provenance.extracted_at}
    assert frame[dataset.STUDY_COLUMN].eq("gitscicomp10").all()


def test_a_frame_written_as_a_db_and_as_a_csv_reads_back_the_same(
    selection: experiments.Selection, tmp_path: pathlib.Path
) -> None:
    """A figure takes either file and must not be able to tell which it was given."""
    frame, provenance = dataset.select(selection, pd.DataFrame([row("100", "dfa"), row("100", "kmp")]))
    assert provenance.rows == 2
    dataset.write_db(frame, tmp_path / "x.db")
    dataset.write_csv(frame, tmp_path / "x.csv")
    from_db, from_csv = dataset.load(tmp_path / "x.db"), dataset.load(tmp_path / "x.csv")
    assert list(from_db["kernel"]) == list(from_csv["kernel"]) == ["dfa", "kmp"]
    assert list(from_db["setup"]) == list(from_csv["setup"])


def test_writing_a_db_twice_replaces_it_rather_than_appending(
    selection: experiments.Selection, tmp_path: pathlib.Path
) -> None:
    """An appending write doubled a re-extracted study, and the duplicate rows are identical,
    so nothing downstream could flag them."""
    frame, provenance = dataset.select(selection, pd.DataFrame([row("100", "dfa")]))
    assert provenance.rows == 1
    dataset.write_db(frame, tmp_path / "x.db")
    dataset.write_db(frame, tmp_path / "x.db")
    assert len(dataset.load(tmp_path / "x.db")) == 1


def test_a_row_on_a_kernel_outside_the_tag_is_dropped_and_counted(tmp_path: pathlib.Path) -> None:
    """The SciComp waves served more kernels than the tag; the experiments name scicomp40, and a
    figure counts a setup over every kernel its rows touch, so an atax row must not reach it."""
    selection = experiments.resolve("scicomp40", root=tmp_path)
    setup = "scicomp40-qwen38-c"
    live = pd.DataFrame([row("100", "gemm", setup=setup), row("101", "atax", setup=setup)])
    frame, provenance = dataset.select(selection, live)
    assert list(frame["kernel"]) == ["gemm"]
    assert provenance.dropped_off_tag == 1
