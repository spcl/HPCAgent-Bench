# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``numerical_oracle.dace_build_root`` must land under the unified JIT cache, not a second,
bare ``$SCRATCH`` root.

Regression for the 2026-09-19 inode-quota incident: before this, the DaCe probe children's build
tree defaulted to ``paths.scratch_root("hpcagent_bench") / "dace_numeric"`` unconditionally --
``$SCRATCH/hpcagent_bench/dace_numeric``, outside ``$SCRATCH/.hpcagentbench-cache`` and swept by
nothing that cleans that tree -- and accounted for 35k of the 1.67M inodes that blew the quota.
"""

import pathlib

import pytest

from tests import numerical_oracle as no

#: Every env var dace_build_root reads, cleared before each test so one test's setenv cannot leak
#: into the next and so a developer's real shell (SCRATCH set on a login node) cannot change the
#: verdict.
ENV_VARS = ("HPCAGENT_BENCH_DACE_BUILD_ROOT", "JIT_CACHE_ROOT", "HPCAGENT_BENCH_CACHE", "SCRATCH")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_prefers_the_unified_jit_cache_root_over_a_bare_scratch_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The unified cache root (scripts/cache_env.sh's JIT_CACHE_ROOT) wins, so the build tree lives
    beside every other small/many/written tree that root already covers."""
    monkeypatch.setenv("JIT_CACHE_ROOT", "/scratch/.hpcagentbench-cache")
    monkeypatch.setenv("SCRATCH", "/scratch")  # would win under the old, buggy default
    assert no.dace_build_root() == pathlib.Path("/scratch/.hpcagentbench-cache/dace_numeric")


def test_falls_back_to_hpcagent_bench_cache_when_jit_cache_root_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two names scripts/cache_env.sh exports both resolve to the same unified tree; either is
    an acceptable source, and JIT_CACHE_ROOT (checked first) is not required to be the one set."""
    monkeypatch.setenv("HPCAGENT_BENCH_CACHE", "/scratch/.hpcagentbench-cache")
    assert no.dace_build_root() == pathlib.Path("/scratch/.hpcagentbench-cache/dace_numeric")


def test_an_explicit_override_wins_over_the_unified_cache_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """HPCAGENT_BENCH_DACE_BUILD_ROOT is the escape hatch a caller reaches for on purpose; it must
    not be shadowed by the cache-root defaults this fix adds."""
    monkeypatch.setenv("HPCAGENT_BENCH_DACE_BUILD_ROOT", "/somewhere/else")
    monkeypatch.setenv("JIT_CACHE_ROOT", "/scratch/.hpcagentbench-cache")
    assert no.dace_build_root() == pathlib.Path("/somewhere/else")


def test_never_lands_under_a_bare_scratch_root_when_the_unified_cache_root_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact regression: SCRATCH alone used to be enough to pick
    $SCRATCH/hpcagent_bench/dace_numeric; with JIT_CACHE_ROOT also set (the normal case on a
    cluster that has sourced scripts/cache_env.sh) that stray path must never be chosen."""
    monkeypatch.setenv("SCRATCH", "/scratch")
    monkeypatch.setenv("JIT_CACHE_ROOT", "/scratch/.hpcagentbench-cache")
    got = no.dace_build_root()
    assert got == pathlib.Path("/scratch/.hpcagentbench-cache/dace_numeric")
    assert got != pathlib.Path("/scratch/hpcagent_bench/dace_numeric")


def test_falls_back_to_the_bare_scratch_root_when_no_unified_cache_root_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare local invocation with cache_env.sh unsourced (no JIT_CACHE_ROOT, no
    HPCAGENT_BENCH_CACHE) keeps the old paths.scratch_root behaviour rather than erroring."""
    monkeypatch.setenv("SCRATCH", "/scratch")
    assert no.dace_build_root() == pathlib.Path("/scratch/hpcagent_bench/dace_numeric")
