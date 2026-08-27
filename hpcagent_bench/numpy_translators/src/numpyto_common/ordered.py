# Copyright 2025 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Insertion-ordered set -- the ONE place the translators get set semantics from.

A plain ``set`` iterates in hash order. For ``str`` elements that order is fixed only
within one interpreter (``PYTHONHASHSEED`` randomises it per process); for anything that
inherits ``object.__hash__`` -- an ``ast`` node -- it follows ``id()``, so it changes with
the allocator even at a pinned seed. Either way, the moment such an order reaches the
emitted text the translator stops being reproducible: the same kernel emits two different
sources, and every downstream diff, cache key, and A/B measurement picks up the noise.

``OrderedSet`` is a thin wrapper over ``dict``, whose insertion order is guaranteed by the
language. Use it for EVERY set in the translators, membership-only ones included. A set
that is only probed today becomes an iterated one after one edit, and that edit is where
the non-reproducibility lands -- far from the line that introduced it. The membership path
costs a dict lookup either way, so there is nothing to trade.

Where the emitted order should not depend on collection order at all (a declaration block,
a symbol list), ``sorted()`` at the point of emission is the stronger answer and is used
directly there; this type is for the cases that must keep source order.
"""
from typing import Any, Dict, Iterable, Iterator


class OrderedSet:
    """A ``set`` that iterates in insertion order. Only the operations the translators
    actually use are implemented; add the one you need here rather than reaching for a
    plain ``set`` at the call site."""

    __slots__ = ("items", )

    def __init__(self, iterable: Iterable[Any] = ()) -> None:
        self.items: Dict[Any, None] = dict.fromkeys(iterable)

    def add(self, item: Any) -> None:
        self.items[item] = None

    def discard(self, item: Any) -> None:
        self.items.pop(item, None)

    def __contains__(self, item: Any) -> bool:
        return item in self.items

    def __iter__(self) -> Iterator[Any]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def update(self, iterable: Iterable[Any]) -> None:
        for item in iterable:
            self.items[item] = None

    def __or__(self, other: Iterable[Any]) -> "OrderedSet":
        merged = OrderedSet(self.items)
        merged.update(other)
        return merged

    def __ior__(self, other: Iterable[Any]) -> "OrderedSet":
        self.update(other)
        return self

    def __eq__(self, other: Any) -> bool:
        # Compares equal to a plain set of the same members: callers and their tests treat
        # this as a set, and order is a property of iteration, not of set identity.
        if isinstance(other, OrderedSet):
            return self.items.keys() == other.items.keys()
        if isinstance(other, (set, frozenset)):
            return self.items.keys() == other
        return NotImplemented

    def __repr__(self) -> str:
        return f"OrderedSet({list(self.items)!r})"
