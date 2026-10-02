# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The generic registry: what a decorated class must provide, and the rules every kind shares."""

from typing import Any

import pytest

from hpcagent_bench.registry import Field, Kind, RegistryError


def widget_kind() -> Kind[tuple[str, int]]:
    """A throwaway kind: a required ``label`` and an optional ``size``, built into a tuple."""
    return Kind(
        "widgets",
        {"label": Field(str, doc="what is printed"), "size": Field(int, 1)},
        lambda key, attrs: (attrs["label"], attrs["size"]),
    )


def test_a_decorated_class_becomes_an_entry_with_its_defaults_filled_in() -> None:
    kind = widget_kind()

    @kind.register("a", order=0)
    class A:
        label = "Alpha"

    assert kind.entries == {"a": ("Alpha", 1)} and A.label == "Alpha"


def test_a_missing_required_attribute_a_wrong_type_and_a_typo_are_refused_at_registration() -> None:
    kind = widget_kind()
    with pytest.raises(RegistryError, match="required attribute 'label'"):

        @kind.register("a", order=0)
        class Missing:
            size = 2

    with pytest.raises(RegistryError, match="size must be"):

        @kind.register("b", order=1)
        class Wrong:
            label = "B"
            size = "big"

    with pytest.raises(RegistryError, match=r"unknown attribute\(s\) \['lable'\]"):

        @kind.register("c", order=2)
        class Typo:
            label = "C"
            lable = "oops"

    assert kind.entries == {}


def test_a_taken_key_alias_or_order_is_refused_and_leaves_the_registry_as_it_was() -> None:
    kind = widget_kind()

    @kind.register("a", order=0, aliases=("alpha",))
    class A:
        label = "A"

    def attempt(key: str, order: int | None, aliases: tuple[str, ...] = ()) -> None:
        @kind.register(key, order=order, aliases=aliases)
        class New:
            label = "N"

    with pytest.raises(RegistryError, match="registered twice"):
        attempt("a", 5)
    with pytest.raises(RegistryError, match="registered twice"):
        attempt("alpha", 5)
    with pytest.raises(RegistryError, match="already 'a'.s slot"):
        attempt("b", 0)
    with pytest.raises(RegistryError, match="alias 'alpha'"):
        attempt("b", 1, ("alpha",))
    with pytest.raises(RegistryError, match="non-negative integer"):
        attempt("b", -1)
    assert list(kind.entries) == ["a"] and kind.aliases == {"alpha": "a"}


def test_import_order_never_decides_the_slot_order() -> None:
    kind = widget_kind()
    for key, order in (("c", 2), ("a", 0), ("b", 1)):
        kind.add(key, (key.upper(), 1), order=order)
    assert kind.keys() == ("a", "b", "c") and kind.next_order() == 3 and kind.slot("b") == 1


def test_an_entry_without_a_slot_comes_first_and_does_not_count_toward_the_next_order() -> None:
    kind = widget_kind()
    kind.add("x", ("X", 1), order=0)
    kind.add("", ("control", 1), order=None)
    assert kind.keys() == ("", "x") and kind.next_order() == 1 and kind.slot("") is None


def test_canonical_resolves_an_alias_and_passes_an_unregistered_tag_through() -> None:
    kind: Kind[Any] = widget_kind()
    kind.add("a", ("A", 1), order=0, aliases=("alpha",))
    assert kind.canonical("alpha") == "a" and kind.canonical("zzz") == "zzz"


if __name__ == "__main__":
    for test in (
        test_a_decorated_class_becomes_an_entry_with_its_defaults_filled_in,
        test_a_missing_required_attribute_a_wrong_type_and_a_typo_are_refused_at_registration,
        test_a_taken_key_alias_or_order_is_refused_and_leaves_the_registry_as_it_was,
        test_import_order_never_decides_the_slot_order,
        test_an_entry_without_a_slot_comes_first_and_does_not_count_toward_the_next_order,
        test_canonical_resolves_an_alias_and_passes_an_unregistered_tag_through,
    ):
        test()
        print("ok", test.__name__)
