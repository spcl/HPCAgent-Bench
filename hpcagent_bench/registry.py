# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One ordered registry per kind of thing the repository names, filled by a class decorator.

A kind (``models``, ``packets``, ``frameworks``, ...) is a :class:`Kind`: its fields, how a decorated
class becomes an entry, and the entries registered so far. Its decorator is the one way to add to it::

    @llm("qwen38", order=0, aliases=("qwen3.8",))
    class Qwen38:
        name = "Qwen3.8-27B"
        serves = "Qwen/Qwen3.8-27B-FP8"

What a decorated class must provide is its kind's ``fields``: every field without a default is
required, every other attribute is optional with its default, and an attribute the kind does not
declare is refused, so a typo is an error at import and never a silently ignored setting. Each value
is checked against its declared type. The checks run when the decorator is applied; nothing is
deferred to first use.

``order`` is an explicit integer, unique within the kind. It is the entry's SLOT (the colour and
marker an entity takes, the position a column is listed at): import order never matters, an existing
entry's order never changes, and a new entry takes :meth:`Kind.next_order`. ``order=None`` registers
an entry with no slot (the no-packet control). Names and aliases share one namespace per kind.

Every registered kind is documented, with the fields it requires, in ``docs/extending/registry.md``.
"""

import dataclasses
from collections.abc import Callable, Iterable, Mapping
from typing import Any

__all__ = ["Field", "Kind", "RegistryError", "fields_of"]


class RegistryError(ValueError):
    """A decorated class or its registration breaks the contract of its kind."""


#: Marks a field as required (it has no default).
REQUIRED: Any = object()


@dataclasses.dataclass(frozen=True, slots=True)
class Field:
    """One attribute a decorated class of a kind may carry: its type and, unless required, its default."""

    type: type | tuple[type, ...]
    default: Any = REQUIRED
    #: A one-line reason, for the error a wrong value raises and for the docs.
    doc: str = ""


def fields_of(cls: type) -> dict[str, Any]:
    """The public attributes ``cls`` itself defines (no inherited ones, no dunders, no methods)."""
    return {key: value for key, value in vars(cls).items() if not key.startswith("_") and not callable(value)}


class Kind[Entry]:
    """The ordered registry of one kind, and the decorator that fills it."""

    __slots__ = ("aliases", "build", "entries", "fields", "name", "orders")

    def __init__(self, name: str, fields: Mapping[str, Field], build: Callable[[str, dict[str, Any]], Entry]) -> None:
        self.name = name
        self.fields = dict(fields)
        self.build = build
        self.entries: dict[str, Entry] = {}
        self.orders: dict[str, int | None] = {}
        self.aliases: dict[str, str] = {}

    def register(self, key: str, *, order: int | None, aliases: Iterable[str] = ()) -> Callable[[type], type]:
        """The class decorator of this kind: validate ``cls``, build its entry, register it under ``key``."""

        def apply(cls: type) -> type:
            self.add(key, self.build(key, self.read(key, cls)), order=order, aliases=tuple(aliases))
            return cls

        return apply

    def read(self, key: str, cls: type) -> dict[str, Any]:
        """``cls``'s attributes checked against the kind's fields, defaults filled in."""
        given = fields_of(cls)
        unknown = sorted(set(given) - set(self.fields))
        if unknown:
            raise RegistryError(
                f"{self.name} {key!r}: unknown attribute(s) {unknown}; the fields are {sorted(self.fields)}"
            )
        values: dict[str, Any] = {}
        for attribute, field in self.fields.items():
            if attribute in given:
                value = given[attribute]
                if not isinstance(value, field.type):
                    raise RegistryError(
                        f"{self.name} {key!r}: {attribute} must be {field.type}, got {type(value).__name__}"
                        + (f" ({field.doc})" if field.doc else "")
                    )
                values[attribute] = value
            elif field.default is REQUIRED:
                raise RegistryError(f"{self.name} {key!r}: required attribute {attribute!r} is missing")
            else:
                values[attribute] = field.default
        return values

    def add(self, key: str, entry: Entry, *, order: int | None, aliases: tuple[str, ...] = ()) -> None:
        """Register ``entry``; raises when the key, an alias or the order is taken."""
        if not isinstance(key, str):
            raise RegistryError(f"{self.name}: key {key!r} must be a string")
        if order is not None and (not isinstance(order, int) or isinstance(order, bool) or order < 0):
            raise RegistryError(f"{self.name} {key!r}: order must be a non-negative integer or None, got {order!r}")
        if key in self.entries or key in self.aliases:
            raise RegistryError(f"{self.name}: {key!r} is registered twice")
        if order is not None and order in self.orders.values():
            holder = next(name for name, taken in self.orders.items() if taken == order)
            raise RegistryError(f"{self.name} {key!r}: order {order} is already {holder!r}'s slot")
        for alias in aliases:
            if alias in self.entries or alias in self.aliases or alias == key:
                raise RegistryError(f"{self.name} {key!r}: alias {alias!r} is already a name or an alias")
        self.entries[key] = entry
        self.orders[key] = order
        self.aliases.update(dict.fromkeys(aliases, key))

    def next_order(self) -> int:
        """The next free slot: one past the highest order registered."""
        return max((order for order in self.orders.values() if order is not None), default=-1) + 1

    def keys(self) -> tuple[str, ...]:
        """The registered keys in slot order; an entry with no slot comes first."""
        return tuple(sorted(self.entries, key=lambda key: -1 if self.orders[key] is None else self.orders[key]))

    def canonical(self, tag: str) -> str:
        """``tag`` with an alias resolved to the key it names; an unregistered tag passes through."""
        return self.aliases.get(tag, tag)

    def slot(self, key: str) -> int | None:
        """The order of a registered key (``None`` for a key without a slot); ``KeyError`` when unregistered."""
        return self.orders[key]
