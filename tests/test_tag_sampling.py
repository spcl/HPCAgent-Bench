# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""hpcagent_bench.tags.sample: a seeded, reproducible kernel draw from (selector, count) rules, its
tags.yaml ``sample:`` entry, and the ``tags sample --save`` CLI that freezes a draw into tags.yaml.

Every test that touches tags.yaml points ``tags.REGISTRY`` at its own temp file and clears the
registry cache, so none reads or writes the real experiments/tags.yaml.
"""

import argparse
import pathlib
import sys
from collections.abc import Iterator

import pytest

from hpcagent_bench import config, tags

#: Real kernels with known levels (shared with tests/test_tags.py): kmp=2, dfa=2, heat_3d=2.
THREE = "explicit:kmp,dfa,heat_3d"
ML_RULES = (("machine_learning@lvl1", 5), ("machine_learning@lvl2", 5))


@pytest.fixture(autouse=True)
def fresh_registry_cache() -> Iterator[None]:
    """monkeypatch restores ``tags.REGISTRY`` after a test but not the lru_cache built from the
    temp file, so a later test would read a registry that no longer exists."""
    tags.registry.cache_clear()
    yield
    tags.registry.cache_clear()


def write_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, text: str) -> pathlib.Path:
    path = tmp_path / "tags.yaml"
    path.write_text(text)
    monkeypatch.setattr(tags, "REGISTRY", path)
    tags.registry.cache_clear()
    tags.RESOLVING.clear()
    return path


def run_cli(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["tags", *argv])
    return tags.main()


def test_the_same_seed_draws_the_same_list() -> None:
    assert tags.sample(ML_RULES, 0) == tags.sample(ML_RULES, 0)


def test_a_different_seed_draws_a_different_list() -> None:
    assert tags.sample(ML_RULES, 0) != tags.sample(ML_RULES, 1)


def test_each_rule_contributes_exactly_its_count_in_rule_order() -> None:
    picked = tags.sample(ML_RULES, 0)
    lvl1 = set(tags.rule_candidates("machine_learning@lvl1"))
    lvl2 = set(tags.rule_candidates("machine_learning@lvl2"))
    assert len(picked) == 10
    assert set(picked[:5]) <= lvl1
    assert set(picked[5:]) <= lvl2


def test_overlapping_rules_never_pick_the_same_kernel_twice() -> None:
    picked = tags.sample((("machine_learning@lvl1", 5), ("machine_learning", 20)), 0)
    assert len(picked) == len(set(picked)) == 25


def test_each_rule_is_sorted_within_itself() -> None:
    picked = tags.sample(ML_RULES, 0)
    assert picked[:5] == sorted(picked[:5])
    assert picked[5:] == sorted(picked[5:])


def test_appending_a_rule_leaves_the_earlier_picks_unchanged() -> None:
    before = tags.sample(ML_RULES, 3)
    after = tags.sample((*ML_RULES, ("machine_learning", 4)), 3)
    assert after[: len(before)] == before


def test_a_count_above_the_candidates_raises_naming_the_selector_count_and_size() -> None:
    with pytest.raises(ValueError, match=r"'explicit:kmp,dfa,heat_3d' asks for 4 kernels but only 3"):
        tags.sample(((THREE, 4),), 0)


def test_a_pool_restricts_every_rules_candidates() -> None:
    pool = tags.operand_keys("explicit:kmp,dfa")
    assert set(tags.sample(((THREE, 2),), 0, sorted(pool))) == pool


def test_a_count_above_the_pool_intersection_raises() -> None:
    pool = sorted(tags.operand_keys("explicit:kmp"))
    with pytest.raises(ValueError, match="only 1 are available"):
        tags.sample(((THREE, 2),), 0, pool)


def test_a_registered_tag_is_a_valid_rule_selector(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    write_registry(monkeypatch, tmp_path, "tags:\n  three:\n    list:\n      - explicit:kmp,dfa,heat_3d\n")
    assert set(tags.sample((("three", 3),), 0)) == tags.operand_keys(THREE)


def test_a_sample_entry_resolves_to_the_same_draw_as_the_library(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    write_registry(
        monkeypatch,
        tmp_path,
        "tags:\n  mlq:\n    sample:\n      seed: 5\n      rules:\n"
        "        - {select: machine_learning@lvl1, count: 5}\n"
        "        - {select: machine_learning@lvl2, count: 5}\n",
    )
    assert tags.resolve("mlq") == sorted(tags.sample(ML_RULES, 5))


def test_a_sample_entry_without_a_seed_uses_the_configured_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    write_registry(
        monkeypatch,
        tmp_path,
        "tags:\n  mlq:\n    sample:\n      rules:\n        - {select: machine_learning, count: 6}\n",
    )
    with config.overridden("seeds.kernel_sample", 11):
        assert tags.resolve("mlq") == sorted(tags.sample((("machine_learning", 6),), 11))


def test_a_sample_entry_from_file_restricts_the_pool(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    listing = tmp_path / "kernels-pool.txt"
    listing.write_text("# pool\nkmp\ndfa  # trailing comment\n")
    write_registry(
        monkeypatch,
        tmp_path,
        f"tags:\n  pooled:\n    sample:\n      from_file: {listing}\n      rules:\n"
        "        - {select: 'explicit:kmp,dfa,heat_3d', count: 2}\n",
    )
    assert set(tags.resolve("pooled")) == tags.operand_keys("explicit:kmp,dfa")


@pytest.mark.parametrize(
    "block",
    [
        "sample: {rules: []}",
        "sample: {rules: [{select: kmp}]}",
        "sample: {rules: [{select: kmp, count: 0}]}",
        "sample: {seed: x, rules: [{select: kmp, count: 1}]}",
    ],
)
def test_a_malformed_sample_block_fails_when_tags_yaml_is_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, block: str
) -> None:
    write_registry(monkeypatch, tmp_path, f"tags:\n  bad:\n    {block}\n")
    with pytest.raises(ValueError, match="'bad'"):
        tags.registry()


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("machine_learning@lvl1:5", ("machine_learning@lvl1", 5)),
        ("explicit:kmp,dfa:2", ("explicit:kmp,dfa", 2)),
    ],
)
def test_a_cli_rule_splits_on_its_last_colon(text: str, rule: tuple[str, int]) -> None:
    assert tags.parse_rule(text) == rule


@pytest.mark.parametrize("text", ["kmp", "kmp:0", "kmp:x", ":3"])
def test_a_malformed_cli_rule_is_rejected(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        tags.parse_rule(text)


def test_the_cli_prints_the_draw_one_kernel_per_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_cli(monkeypatch, "sample", "machine_learning@lvl1:5", "machine_learning@lvl2:5", "--seed", "0") == 0
    assert capsys.readouterr().out.splitlines() == tags.sample(ML_RULES, 0)


def test_save_writes_an_explicit_entry_that_resolves_to_the_same_draw(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_registry(monkeypatch, tmp_path, "# header kept\ntags: {}\naliases: {}\n")
    assert run_cli(monkeypatch, "sample", "machine_learning:6", "--seed", "2", "--save", "frozen") == 0
    printed = capsys.readouterr().out.splitlines()
    assert tags.registry().tags["frozen"].op == "list"
    assert tags.resolve("frozen") == sorted(printed)


def test_save_records_the_rules_and_seed_and_keeps_existing_comments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    path = write_registry(monkeypatch, tmp_path, "# header kept\ntags: {}\naliases: {}\n")
    assert run_cli(monkeypatch, "sample", "machine_learning:6", "--seed", "2", "--save", "frozen") == 0
    text = path.read_text()
    assert "# header kept" in text
    assert "tags sample machine_learning:6 seed=2 on " in text


def test_save_refuses_an_existing_tag_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    original = "tags:\n  taken:\n    list:\n      - kmp\n"
    path = write_registry(monkeypatch, tmp_path, original)
    assert run_cli(monkeypatch, "sample", "machine_learning:3", "--save", "taken") == 2
    assert "already exists" in capsys.readouterr().err
    assert path.read_text() == original
