# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""hpcagent_bench.harness.kept_archive: reading a `sources` row back out of a reduced job dir's
`kept-<jobid>.tar` (audit-20260918/reduce_finished_jobs.py), post-cleanup."""

import pathlib
import tarfile

import pytest

from hpcagent_bench.harness import kept_archive


def _make_kept_tar(job_dir: pathlib.Path, rank_name: str, source_path: str, body: bytes) -> None:
    """A minimal stand-in for what the reducer writes: one source member under
    `sources/<rank_name>/<source_path>`, named after the job dir itself."""
    payload = job_dir / "_payload"
    payload.write_bytes(body)
    with tarfile.open(kept_archive.kept_tar_path(job_dir), "w") as tar:
        tar.add(payload, arcname=f"sources/{rank_name}/{source_path}")
    payload.unlink()


def test_read_kept_source_returns_the_exact_bytes_the_reducer_packed(tmp_path: pathlib.Path) -> None:
    job_dir = tmp_path / "campaign" / "635346"
    job_dir.mkdir(parents=True)
    body = b"// device kernel body\n"
    _make_kept_tar(job_dir, "rank-0", "fd/fd5589.txt", body)

    got = kept_archive.read_kept_source(job_dir, job_dir / "judge" / "rank-0", "fd/fd5589.txt")

    assert got == body


def test_source_tar_member_matches_the_reducers_own_layout(tmp_path: pathlib.Path) -> None:
    """The reducer names members `sources/<rank dir name>/<sources.path>`; a reader that computes
    this path differently silently misses every row once the live store is gone."""
    rank_dir = tmp_path / "judge" / "rank-3"
    assert kept_archive.source_tar_member(rank_dir, "ab/abcdef.txt") == "sources/rank-3/ab/abcdef.txt"


def test_read_kept_source_names_the_archive_when_the_job_was_never_reduced(tmp_path: pathlib.Path) -> None:
    job_dir = tmp_path / "campaign" / "999999"
    job_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="kept-999999.tar"):
        kept_archive.read_kept_source(job_dir, job_dir / "judge" / "rank-0", "fd/fd5589.txt")


def test_read_kept_source_names_the_member_for_a_path_not_in_this_jobs_sources_table(tmp_path: pathlib.Path) -> None:
    job_dir = tmp_path / "campaign" / "635346"
    job_dir.mkdir(parents=True)
    _make_kept_tar(job_dir, "rank-0", "fd/fd5589.txt", b"present")

    with pytest.raises(FileNotFoundError, match="sources/rank-0/ab/absent.txt"):
        kept_archive.read_kept_source(job_dir, job_dir / "judge" / "rank-0", "ab/absent.txt")
