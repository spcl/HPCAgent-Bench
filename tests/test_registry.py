# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The generic registry: keys, aliases, and slots taken in registration order."""

from collections.abc import Callable

import pytest

from hpcagent_bench.registry import Kind, RegistryError


def test_add_returns_the_entry_and_lookup_resolves_keys_and_aliases() -> None:
    kind: Kind[str] = Kind("widgets")
    assert kind.add("a", "Alpha", aliases=("alpha",)) == "Alpha"
    assert kind["a"] == kind["alpha"] == "Alpha"
    assert "alpha" in kind
    assert "zzz" not in kind
    with pytest.raises(KeyError, match="widgets: 'zzz' is not registered"):
        kind["zzz"]


def test_a_taken_key_or_alias_is_refused_and_leaves_the_registry_as_it_was() -> None:
    kind: Kind[str] = Kind("widgets")
    kind.add("a", "A", aliases=("alpha",))
    for key, aliases, message in (
        ("a", (), "registered twice"),
        ("alpha", (), "registered twice"),
        ("b", ("alpha",), "alias 'alpha'"),
        ("b", ("x", "x"), "alias 'x'"),
    ):
        with pytest.raises(RegistryError, match=message):
            kind.add(key, "N", aliases=aliases)
    assert list(kind.entries) == ["a"]
    assert kind.aliases == {"alpha": "a"}


def test_the_slot_is_the_registration_position() -> None:
    kind: Kind[str] = Kind("widgets")
    for key in ("c", "a", "b"):
        kind.add(key, key.upper())
    assert kind.keys() == tuple(kind) == ("c", "a", "b")
    assert [kind.slot(key) for key in ("c", "a", "b")] == [0, 1, 2]


def test_an_entry_without_a_slot_comes_first_and_takes_none() -> None:
    kind: Kind[str] = Kind("widgets")
    kind.add("x", "X")
    kind.add("", "control", slotted=False)
    kind.add("y", "Y")
    assert kind.keys() == ("", "x", "y")
    assert (kind.slot(""), kind.slot("y")) == (None, 1)


def test_canonical_resolves_an_alias_and_passes_an_unregistered_tag_through() -> None:
    kind: Kind[str] = Kind("widgets")
    kind.add("a", "A", aliases=("alpha",))
    assert kind.canonical("alpha") == "a"
    assert kind.canonical("zzz") == "zzz"


def test_register_adds_the_decorated_object_unchanged() -> None:
    kind: Kind[Callable[[int], int]] = Kind("functions")

    @kind.register("double")
    def double(value: int) -> int:
        return 2 * value

    assert kind["double"] is double
    assert double(3) == 6
    assert kind.slot("double") == 0
