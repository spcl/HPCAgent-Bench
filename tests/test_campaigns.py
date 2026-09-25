# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Which rows belong to an experiment, resolved from the registry alone.

The mapping used to live in three unsynchronised copies (wave_board, migrate_db, kernel_comparison),
so a prefix added to one was absent from the others with nothing to catch it."""

import pytest

from hpcagent_bench import campaigns
from hpcagent_bench.experiment_tags import registry


@pytest.mark.parametrize(
    ("arm", "prefix"),
    [
        ("git-scicomp-qwen38-repo", "git-scicomp"),
        ("cpf-llr-focus40-qwen38-c", "cpf-llr-focus40"),
        ("gpu-llr-focus40-oss120b-hip", "gpu-llr-focus40"),
        # The trap this rule exists for: the stem also matches, and the longer key must win or the
        # GPU arm resolves to the CPU campaign with "gpu" read as its model.
        ("scicomp-dc-gpu-qwen38-hip-plain", "scicomp-dc-gpu"),
        ("scicomp-dc-qwen38-plain", "scicomp-dc"),
        ("scicomp-perf-playbook-gpu-oss120b-hip", "scicomp-perf-playbook-gpu"),
        ("adhoc", ""),
        ("", ""),
    ],
)
def test_the_longest_campaign_prefix_wins(arm: str, prefix: str) -> None:
    assert campaigns.prefix_of(arm) == prefix


def test_a_campaign_prefix_only_matches_on_a_hyphen_boundary() -> None:
    """``harness20`` and ``harness-focus20`` are different campaigns; a bare startswith would let
    one swallow arms of the other."""
    assert campaigns.prefix_of("harness20-oss120b-claude") == "harness20"
    assert campaigns.prefix_of("harness20x-oss120b-claude") == ""


@pytest.mark.parametrize(
    ("arm", "retired"),
    [
        ("cpf-llr-focus40-qwen38-c-cpfsrc", True),
        # Only cpfsrc-v2 counts (user 2026-09-19), so the v2 arm must survive the same regex.
        ("cpf-llr-focus40-qwen38-c-cpfsrc-v2", False),
        ("cpf-llr-focus40-qwen38-fortran", True),
        ("cpf-llr-focus40-qwen38-c", False),
        ("git-scicomp-qwen38-repo", False),
    ],
)
def test_a_retired_arm_is_recognised_from_the_registry_regex(arm: str, retired: bool) -> None:
    assert campaigns.dropped(arm) is retired


def test_an_unknown_experiment_raises_and_names_the_ones_that_exist() -> None:
    """An empty selection would read downstream as a campaign that produced no rows, which is
    exactly what a real coverage gap looks like."""
    with pytest.raises(KeyError, match="no campaign feeds experiment 'llr-focus41'"):
        campaigns.resolve("llr-focus41")


def test_one_experiment_collects_every_campaign_that_feeds_it() -> None:
    """llr-focus40 ran under two launchers, CPU and GPU; a selection that took only one would
    silently halve the experiment."""
    selection = campaigns.resolve("llr-focus40")
    assert set(selection.prefixes) == {"cpf-llr-focus40", "gpu-llr-focus40"}
    assert set(selection.devices) == {"CPU", "GPU"}


def test_a_run_glob_is_the_prefix_under_the_runs_root(tmp_path) -> None:
    """A launcher names its run root ``<prefix>-<date>``, and dated and lettered suffixes
    (``git-scicomp-20260917b``) both have to match or a wave goes missing."""
    selection = campaigns.resolve("git-scicomp", root=tmp_path)
    assert selection.run_globs() == (str(tmp_path / "git-scicomp-*"),)


def test_the_selection_carries_the_roster_its_campaigns_served() -> None:
    """numba and pluto were swept over the whole 248-kernel loop-level-reasoning track; a baseline
    reduced over that instead of the 40 kernels the agents saw is a different number."""
    selection = campaigns.resolve("llr-focus40")
    assert selection.tag == "llr-focus40"
    assert len(selection.roster) == 40


def test_the_baseline_names_canon_columns_not_another_campaign() -> None:
    """The canon sweep is not a campaign and has no job-name prefix, so a baseline declared as an
    experiment name would resolve to no run root at all."""
    selection = campaigns.resolve("llr-focus40")
    assert selection.baseline.denominator == "numba"
    assert "pluto" in selection.baseline.comparators
    assert selection.canon_columns()[0] == "numba"


def test_every_campaign_names_an_experiment_the_registry_lists() -> None:
    """A campaign pointing at an unlisted experiment draws under a raw tag instead of its name."""
    known = set(registry().experiments)
    unlisted = sorted({entry.experiment for entry in campaigns.campaigns().values()} - known)
    assert not unlisted, unlisted


def test_every_declared_baseline_belongs_to_an_experiment_a_campaign_feeds() -> None:
    """A baseline declared for an experiment nothing runs is a typo that never surfaces."""
    fed = set(campaigns.experiments_available())
    orphans = sorted(set(registry().experiment_baselines) - fed)
    assert not orphans, orphans
