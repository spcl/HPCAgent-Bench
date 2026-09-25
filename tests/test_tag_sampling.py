# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""hpcagent_bench.tags.sample: a seeded, reproducible kernel draw from (selector, count) rules, and
the ``tags sample --save`` CLI that saves a draw as a tag file.

Every test points ``tags.TAGS_DIR`` at its own temp folder holding one tag, ``three``, so none reads
or writes the real tag folder.
"""

import argparse
import pathlib
import sys
from collections.abc import Iterator

import pytest

from hpcagent_bench import tags
from hpcagent_bench.spec import KERNELS

#: A temp tag of three real kernels.
THREE = "three"
ML_RULES = (("machine_learning@lvl1", 5), ("machine_learning@lvl2", 5))


@pytest.fixture(autouse=True)
def temp_tags(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> Iterator[pathlib.Path]:
    """A temp tag folder holding ``three`` (kmp, dfa, heat_3d); the index cache cleared around it."""
    (tmp_path / f"{THREE}.txt").write_text("kmp\ndfa\nheat_3d\n")
    monkeypatch.setattr(tags, "TAGS_DIR", tmp_path)
    tags.index.cache_clear()
    yield tmp_path
    tags.index.cache_clear()


def keys(*names: str) -> set[str]:
    return set(tags.kernel_keys(names, "test"))


def run_cli(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["tags", *argv])
    return tags.main()


def test_the_same_seed_draws_the_same_list() -> None:
    assert tags.sample(ML_RULES, 0) == tags.sample(ML_RULES, 0)


def test_a_different_seed_draws_a_different_list() -> None:
    assert tags.sample(ML_RULES, 0) != tags.sample(ML_RULES, 1)


def test_each_rule_contributes_exactly_its_count_in_rule_order() -> None:
    picked = tags.sample(ML_RULES, 0)
    lvl1 = set(KERNELS.select_keys("machine_learning@lvl1"))
    lvl2 = set(KERNELS.select_keys("machine_learning@lvl2"))
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
    with pytest.raises(ValueError, match=r"'three' asks for 4 kernels but only 3"):
        tags.sample(((THREE, 4),), 0)


def test_a_pool_restricts_every_rules_candidates() -> None:
    pool = keys("kmp", "dfa")
    assert set(tags.sample(((THREE, 2),), 0, sorted(pool))) == pool


def test_a_count_above_the_pool_intersection_raises() -> None:
    pool = sorted(keys("kmp"))
    with pytest.raises(ValueError, match="only 1 are available"):
        tags.sample(((THREE, 2),), 0, pool)


def test_a_tag_and_a_selector_are_both_valid_rule_selectors() -> None:
    assert set(tags.sample(((THREE, 3),), 0)) == keys("kmp", "dfa", "heat_3d")
    assert set(tags.sample((("kmp", 1),), 0)) == keys("kmp")


def test_a_cli_rule_is_a_selector_and_a_count() -> None:
    assert tags.parse_rule("machine_learning@lvl1:5") == ("machine_learning@lvl1", 5)


@pytest.mark.parametrize("text", ["kmp", "kmp:0", "kmp:x", ":3"])
def test_a_malformed_cli_rule_is_rejected(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        tags.parse_rule(text)


def test_the_cli_prints_the_draw_one_kernel_per_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_cli(monkeypatch, "sample", "machine_learning@lvl1:5", "machine_learning@lvl2:5", "--seed", "0") == 0
    assert capsys.readouterr().out.splitlines() == tags.sample(ML_RULES, 0)


def test_save_writes_a_tag_file_that_resolves_to_the_same_draw(
    monkeypatch: pytest.MonkeyPatch, temp_tags: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_cli(monkeypatch, "sample", "machine_learning:6", "--seed", "2", "--save", "frozen") == 0
    printed = capsys.readouterr().out.splitlines()
    assert tags.resolve("frozen") == sorted(printed)
    assert "# tags sample machine_learning:6 seed=2 on " in (temp_tags / "frozen.txt").read_text()


def test_save_refuses_an_existing_tag_name(
    monkeypatch: pytest.MonkeyPatch, temp_tags: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    original = (temp_tags / f"{THREE}.txt").read_text()
    assert run_cli(monkeypatch, "sample", "machine_learning:3", "--save", THREE) == 2
    assert "already exists" in capsys.readouterr().err
    assert (temp_tags / f"{THREE}.txt").read_text() == original


def test_from_file_restricts_the_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    listing = tmp_path / "pool.list"
    listing.write_text("# pool\nkmp\ndfa  # trailing comment\n")
    assert run_cli(monkeypatch, "sample", f"{THREE}:2", "--from-file", str(listing)) == 0
    assert set(capsys.readouterr().out.splitlines()) == keys("kmp", "dfa")
