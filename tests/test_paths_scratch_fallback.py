# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""hpcagent_bench.paths: the one Python-side fallback for "no $SCRATCH" (repo_root,
scratch_or_repo, scratch_root's own fallback branch).

Before this, three scripts (experiments/campaign_status.py, scripts/canon_sdfg_prerender.py,
scripts/migrate_canon_scratch_dirs.py) each guessed their own answer to "what if SCRATCH is
unset" -- a repo-parent, a bare __file__ walk, and a plain ~/.cache -- three different answers to
the same question. These tests pin the one answer now shared: $HPCAGENT_BENCH_REPO if a caller
resolved one, else this checkout's own root (paths.ROOT).
"""

import pathlib

import pytest

from hpcagent_bench import paths

ENV_VARS = ("SCRATCH", "HPCAGENT_BENCH_REPO")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_repo_root_uses_hpcagent_bench_repo_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HPCAGENT_BENCH_REPO", "/some/checkout")
    assert paths.repo_root() == pathlib.Path("/some/checkout")


def test_repo_root_falls_back_to_this_checkout_when_unset() -> None:
    assert paths.repo_root() == paths.ROOT


def test_scratch_or_repo_prefers_scratch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRATCH", "/scratch/user")
    monkeypatch.setenv("HPCAGENT_BENCH_REPO", "/some/checkout")
    assert paths.scratch_or_repo() == pathlib.Path("/scratch/user")


def test_scratch_or_repo_falls_back_to_repo_root_without_scratch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HPCAGENT_BENCH_REPO", "/some/checkout")
    assert paths.scratch_or_repo() == pathlib.Path("/some/checkout")


def test_scratch_root_lands_under_scratch_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRATCH", "/scratch/user")
    assert paths.scratch_root("hpcagent_sizing") == pathlib.Path("/scratch/user/hpcagent_sizing")


def test_scratch_root_falls_back_to_repo_cache_without_scratch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The old default (~/.cache/<name>) is gone: a container or CI box with no $SCRATCH now lands
    under the checkout it is actually running from, not a guess at the invoking user's home."""
    monkeypatch.setenv("HPCAGENT_BENCH_REPO", "/some/checkout")
    assert paths.scratch_root("hpcagent_sizing") == pathlib.Path("/some/checkout/.cache/hpcagent_sizing")


def test_scratch_root_with_neither_var_set_matches_repo_root_directly() -> None:
    """No env var configured at all (a bare `pytest tests/` with no cluster session sourced) is
    still the ROOT-relative default, not an error and not a guess at $HOME."""
    assert paths.scratch_root("hpcagent_sizing") == paths.ROOT / ".cache" / "hpcagent_sizing"
