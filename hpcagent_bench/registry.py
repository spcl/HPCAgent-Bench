# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One ordered registry (:class:`Kind`) per kind of thing the repository names: models, packets, columns, gates.
An entry's SLOT (colour, marker, listing position) is its registration position, so a new entry is added last and
never repaints another; a key or alias taken twice is refused. See ``docs/extending/registry.md``."""

import dataclasses
from collections.abc import Callable, Iterable, Iterator

__all__ = ["Kind", "RegistryError"]


class RegistryError(ValueError):
    """An entry or its registration breaks the contract of its kind."""


@dataclasses.dataclass(slots=True)
class Kind[Entry]:
    """The ordered registry of one kind."""

    name: str
    entries: dict[str, Entry] = dataclasses.field(default_factory=dict[str, Entry])
    orders: dict[str, int | None] = dataclasses.field(default_factory=dict[str, int | None])
    aliases: dict[str, str] = dataclasses.field(default_factory=dict[str, str])

    def add(self, key: str, entry: Entry, *, slotted: bool = True, aliases: Iterable[str] = ()) -> Entry:
        """Register ``entry`` under ``key`` at the next slot (none unless ``slotted``); a taken key or alias raises."""
        if key in self.entries or key in self.aliases:
            raise RegistryError(f"{self.name}: {key!r} is registered twice")
        names = tuple(aliases)
        for alias in names:
            if alias in self.entries or self.aliases.get(alias, key) != key or alias == key or names.count(alias) > 1:
                raise RegistryError(f"{self.name} {key!r}: alias {alias!r} is already a name or an alias")
        self.entries[key] = entry
        self.orders[key] = sum(order is not None for order in self.orders.values()) if slotted else None
        self.aliases.update(dict.fromkeys(names, key))
        return entry

    def register(self, key: str, *, aliases: Iterable[str] = ()) -> Callable[[Entry], Entry]:
        """A decorator adding the decorated object itself as the entry: for kinds whose entries are code."""

        def apply(entry: Entry) -> Entry:
            self.add(key, entry, aliases=aliases)
            return entry

        return apply

    def keys(self) -> tuple[str, ...]:
        """The registered keys in slot order; an entry with no slot comes first."""
        return tuple(sorted(self.entries, key=lambda key: -1 if (order := self.orders[key]) is None else order))

    def __iter__(self) -> Iterator[str]:
        """The keys, in :meth:`keys` order."""
        return iter(self.keys())

    def __getitem__(self, key: str) -> Entry:
        """The entry of ``key`` or of the alias ``key``; KeyError naming the kind when neither is registered."""
        try:
            return self.entries[self.canonical(key)]
        except KeyError:
            raise KeyError(f"{self.name}: {key!r} is not registered") from None

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.canonical(key) in self.entries

    def canonical(self, tag: str) -> str:
        """``tag`` with an alias resolved to the key it names; an unregistered tag passes through."""
        return self.aliases.get(tag, tag)

    def slot(self, key: str) -> int | None:
        """The order of a registered key (``None`` for a key without a slot); ``KeyError`` when unregistered."""
        return self.orders[key]
