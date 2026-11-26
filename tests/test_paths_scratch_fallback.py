# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""hpcagent_bench.paths: the one Python-side fallback for "no $SCRATCH" (repo_root,
scratch_or_repo, scratch_root's own fallback branch).

When $SCRATCH is unset the answer is $HPCAGENT_BENCH_REPO if a caller resolved one, else this
checkout's own root (paths.ROOT).
"""

import pathlib

import pytest

from hpcagent_bench import paths

ENV_VARS = ("SCRATCH", "HPCAGENT_BENCH_REPO")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        pytest.param(
            {"SCRATCH": "/scratch/user", "HPCAGENT_BENCH_REPO": "/some/checkout"},
            pathlib.Path("/scratch/user"),
            id="prefers-scratch",
        ),
        pytest.param(
            {"HPCAGENT_BENCH_REPO": "/some/checkout"}, pathlib.Path("/some/checkout"), id="repo-without-scratch"
        ),
    ],
)
def test_scratch_or_repo_prefers_scratch(
    monkeypatch: pytest.MonkeyPatch, environ: dict[str, str], expected: pathlib.Path
) -> None:
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    assert paths.scratch_or_repo() == expected


def test_scratch_root_lands_under_scratch_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRATCH", "/scratch/user")
    assert paths.scratch_root("hpcagent_sizing") == pathlib.Path("/scratch/user/hpcagent_sizing")


@pytest.mark.parametrize(
    ("repo", "expected"),
    [
        # A container or CI box with no $SCRATCH lands under the checkout it is actually running from, not a guess at
        # the invoking user's home (~/.cache/<name>).
        pytest.param("/some/checkout", pathlib.Path("/some/checkout/.cache/hpcagent_sizing"), id="a-resolved-checkout"),
        # No env var configured at all (a bare `pytest tests/` with no cluster session sourced) is still the
        # ROOT-relative default, not an error and not a guess at $HOME.
        pytest.param(None, paths.ROOT / ".cache" / "hpcagent_sizing", id="neither-variable-set"),
    ],
)
def test_scratch_root_falls_back_to_repo_cache_without_scratch(
    monkeypatch: pytest.MonkeyPatch, repo: str | None, expected: pathlib.Path
) -> None:
    if repo is not None:
        monkeypatch.setenv("HPCAGENT_BENCH_REPO", repo)
    assert paths.scratch_root("hpcagent_sizing") == expected
