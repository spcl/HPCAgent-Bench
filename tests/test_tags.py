# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""hpcagent_bench.tags: dynamic kernel-set tags composed from existing selectors, so a new roster
(``mixed``: bfcd7766 touched 20 manifests) needs one experiments/tags.yaml entry instead of a
manifest edit per kernel.

Every test points ``tags.REGISTRY`` at its own temp file and clears ``tags.registry``'s cache, so
none of them read or write the real experiments/tags.yaml (shipped empty) and none of them race a
concurrently running test over shared state.
"""

import pathlib
from collections.abc import Iterator

import pytest

from hpcagent_bench import tags

#: Real scientific_computing kernels (shared with other submit-*.sh tests), with known levels:
#: kmp=2, dfa=2, heat_3d=2, eigh_test=3.
LEVEL2 = ("kmp", "dfa", "heat_3d")


@pytest.fixture(autouse=True)
def forget_the_temp_registry() -> Iterator[None]:
    """Clear ``tags.registry``'s cache AFTER each test too: monkeypatch restores ``tags.REGISTRY``,
    but the lru_cache still held this test's temp registry, so the next test in the same process
    (tests/test_mixed_roster.py's `mixed` alias) resolved against it and failed by run order."""
    yield
    tags.registry.cache_clear()
    tags.RESOLVING.clear()


def write_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, text: str) -> None:
    path = tmp_path / "tags.yaml"
    path.write_text(text)
    monkeypatch.setattr(tags, "REGISTRY", path)
    tags.registry.cache_clear()
    tags.RESOLVING.clear()


def test_a_missing_registry_file_is_an_empty_registry_not_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setattr(tags, "REGISTRY", tmp_path / "does-not-exist.yaml")
    tags.registry.cache_clear()
    assert tags.registry().tags == {}
    assert tags.registry().aliases == {}


def test_union_is_the_set_union_of_every_operand(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    write_registry(
        monkeypatch,
        tmp_path,
        "tags:\n  mytag:\n    union:\n      - explicit:kmp,dfa\n      - explicit:heat_3d\n",
    )
    stems = {k.rsplit("/", 1)[-1] for k in tags.resolve_registered("mytag")}
    assert stems == {"kmp", "dfa", "heat_3d"}


def test_list_is_the_same_as_union(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    write_registry(monkeypatch, tmp_path, "tags:\n  mytag:\n    list:\n      - explicit:kmp,dfa\n")
    stems = {k.rsplit("/", 1)[-1] for k in tags.resolve_registered("mytag")}
    assert stems == {"kmp", "dfa"}


def test_intersect_is_the_set_intersection_of_every_operand(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    write_registry(
        monkeypatch,
        tmp_path,
        "tags:\n  mytag:\n    intersect:\n      - explicit:kmp,dfa,heat_3d\n      - explicit:dfa,heat_3d,eigh_test\n",
    )
    stems = {k.rsplit("/", 1)[-1] for k in tags.resolve_registered("mytag")}
    assert stems == {"dfa", "heat_3d"}


def test_diff_is_the_first_operand_minus_every_other(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    write_registry(
        monkeypatch,
        tmp_path,
        "tags:\n  mytag:\n    diff:\n      - explicit:kmp,dfa,heat_3d\n      - explicit:dfa\n",
    )
    stems = {k.rsplit("/", 1)[-1] for k in tags.resolve_registered("mytag")}
    assert stems == {"kmp", "heat_3d"}


def test_a_level_clause_filters_the_operand_by_resolved_level(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """eigh_test is level 3; kmp/dfa/heat_3d are level 2 -- [level<=2] must drop only eigh_test."""
    write_registry(
        monkeypatch, tmp_path, "tags:\n  mytag:\n    list:\n      - 'explicit:kmp,dfa,heat_3d,eigh_test[level<=2]'\n"
    )
    stems = {k.rsplit("/", 1)[-1] for k in tags.resolve_registered("mytag")}
    assert stems == set(LEVEL2)


@pytest.mark.parametrize(
    ("clause", "expect"),
    [
        ("[level<=2]", {"kmp", "dfa", "heat_3d"}),
        ("[level>=3]", {"eigh_test"}),
        ("[level==3]", {"eigh_test"}),
        ("[level<3]", {"kmp", "dfa", "heat_3d"}),
        ("[level>2]", {"eigh_test"}),
    ],
)
def test_every_level_operator_filters_correctly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, clause: str, expect: set[str]
) -> None:
    write_registry(
        monkeypatch, tmp_path, f"tags:\n  mytag:\n    list:\n      - 'explicit:kmp,dfa,heat_3d,eigh_test{clause}'\n"
    )
    stems = {k.rsplit("/", 1)[-1] for k in tags.resolve_registered("mytag")}
    assert stems == expect


def test_an_alias_resolves_to_its_targets_own_definition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    write_registry(
        monkeypatch,
        tmp_path,
        "tags:\n  mytag:\n    list:\n      - explicit:kmp,dfa\naliases:\n  shortcut: mytag\n",
    )
    assert tags.canonical("shortcut") == "mytag"
    assert tags.resolve_registered("shortcut") == tags.resolve_registered("mytag")


def test_an_unregistered_alias_target_still_resolves_registered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """An unknown tag PASSES THROUGH canonical() unchanged (experiment_tags.py's own convention),
    so a typo'd alias target is a plain 'no such entry' KeyError, not a crash."""
    write_registry(monkeypatch, tmp_path, "aliases:\n  shortcut: nosuchtag\n")
    assert tags.canonical("shortcut") == "nosuchtag"
    with pytest.raises(KeyError):
        tags.resolve_registered("shortcut")


def test_an_unknown_tag_raises_key_error(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    write_registry(monkeypatch, tmp_path, "tags: {}\n")
    with pytest.raises(KeyError):
        tags.resolve_registered("no-such-tag")
    assert not tags.is_registered("no-such-tag")


def test_an_unknown_kernel_in_an_operand_is_refused_not_silently_dropped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    write_registry(monkeypatch, tmp_path, "tags:\n  mytag:\n    list:\n      - explicit:kmp,nosuchkernel123\n")
    with pytest.raises(KeyError, match="nosuchkernel123"):
        tags.resolve_registered("mytag")


def test_an_intersection_that_resolves_to_nothing_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    write_registry(
        monkeypatch,
        tmp_path,
        "tags:\n  mytag:\n    intersect:\n      - explicit:kmp\n      - explicit:dfa\n",
    )
    with pytest.raises(ValueError, match="resolved to nothing"):
        tags.resolve_registered("mytag")


def test_a_circular_reference_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    write_registry(
        monkeypatch,
        tmp_path,
        "tags:\n  a:\n    union:\n      - 'all@b'\n  b:\n    union:\n      - 'all@a'\n",
    )
    with pytest.raises(ValueError, match="circular"):
        tags.resolve_registered("a")


def test_an_entry_must_name_exactly_one_operator(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    write_registry(monkeypatch, tmp_path, "tags:\n  mytag:\n    union: [explicit:kmp]\n    diff: [explicit:dfa]\n")
    with pytest.raises(ValueError, match="exactly one"):
        tags.registry()


def test_resolution_is_deterministic_regardless_of_operand_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    write_registry(monkeypatch, tmp_path, "tags:\n  mytag:\n    union:\n      - explicit:heat_3d,dfa,kmp\n")
    result = tags.resolve_registered("mytag")
    assert result == sorted(result)


def test_a_kernels_file_always_wins_over_a_tags_yaml_entry_of_the_same_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Migration is opt-in: an existing flat-file roster never gets silently shadowed by a
    tags.yaml entry sharing its name."""
    write_registry(monkeypatch, tmp_path, "tags:\n  git-scicomp:\n    list:\n      - explicit:kmp\n")
    # kernels-git-scicomp.txt is the real, committed roster (10 kernels); resolve() must use IT,
    # never the one-kernel tags.yaml entry of the same name shadowed above.
    resolved = {k.rsplit("/", 1)[-1] for k in tags.resolve("git-scicomp")}
    assert resolved == {
        "fv3_dycore", "lda_xc_potential", "edge_laplacian", "addusxx_g", "warpx_boris_push",
        "kmp", "dfa", "fdtd_2d", "heat_3d", "jacobi_2d",
    }  # fmt: skip


def test_version_is_stable_for_the_same_definition_and_changes_when_it_moves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    write_registry(monkeypatch, tmp_path, "tags:\n  mytag:\n    list:\n      - explicit:kmp,dfa\n")
    first = tags.version("mytag")
    second = tags.version("mytag")
    assert first == second

    write_registry(monkeypatch, tmp_path, "tags:\n  mytag:\n    list:\n      - explicit:kmp,dfa,heat_3d\n")
    assert tags.version("mytag") != first


def test_cli_resolve_prints_sorted_stems_and_refuses_with_a_clear_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_registry(monkeypatch, tmp_path, "tags:\n  mytag:\n    list:\n      - explicit:kmp,dfa\n")
    monkeypatch.setattr("sys.argv", ["tags.py", "resolve", "mytag"])
    assert tags.main() == 0
    assert capsys.readouterr().out.strip() == "dfa,kmp"

    monkeypatch.setattr("sys.argv", ["tags.py", "resolve", "no-such-tag"])
    assert tags.main() == 2
    assert "no-such-tag" in capsys.readouterr().err
