# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The rules of ``helpers/scripts/checks/check_interpreter_floor.py`` on synthetic text: each is shown the
exact source a py3.12 judge failed to import, and the fallback or ordering that makes it safe."""

from collections.abc import Callable

import pytest

from hpcagent_bench import paths
from tests.fresh_module import module_at

floor = module_at(
    paths.ROOT / "helpers" / "scripts" / "checks" / "check_interpreter_floor.py", "check_interpreter_floor"
)

#: Defects that reached a py3.12 judge while the venv imported them clean, each with its verdict.
FLOOR_DEFECTS = (
    ("from typing import Any, Callable, TypeGuard, TypeIs, cast\n", floor.too_new_typing_names, True),
    (
        "try:\n    from typing import TypeIs\nexcept ImportError:\n    from typing_extensions import TypeIs\n",
        floor.too_new_typing_names,
        False,
    ),
    (
        "_OVERRIDES: dict[str, ConfigValue] = {}\nConfigValue = bool | int | str | None\n",
        floor.eager_annotation_faults,
        True,
    ),
    ("def lookup(key: str) -> 'Entry' | None:\n    return None\n", floor.eager_annotation_faults, True),
    # a PEP 695 alias is lazy and unions at runtime; a PEP 695 parameter is bound in its own scope
    (
        "type Entry = dict[str, 'Later']\ndef lookup(key: str) -> Entry | None:\n    return None\n",
        floor.eager_annotation_faults,
        False,
    ),
    ("class Box[T]:\n    def get(self) -> T | None:\n        return None\n", floor.eager_annotation_faults, False),
    ("def first[T](items: list[T]) -> T | None:\n    return None\n", floor.eager_annotation_faults, False),
    (
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import pandas as pd\ndef rows(f: pd.DataFrame) -> None: ...\n",
        floor.eager_annotation_faults,
        True,
    ),
    (
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import pandas as pd\ndef rows(f: 'pd.DataFrame') -> None: ...\n",
        floor.eager_annotation_faults,
        False,
    ),
    (
        'from typing import TypeAlias\nValue: TypeAlias = "int | list[Value]"\ndef first(v: Value) -> Value | None: ...\n',
        floor.eager_annotation_faults,
        True,
    ),
    (
        'from typing import TypeAlias, TypeVar\nT = TypeVar("T")\nBox: TypeAlias = "list[T]"\ndef first(b: Box[T]) -> None: ...\n',
        floor.eager_annotation_faults,
        True,
    ),
    ("ConfigValue = int\n_OVERRIDES: dict[str, ConfigValue] = {}\n", floor.eager_annotation_faults, False),
)


def test_the_too_new_lists_name_only_what_the_floor_cannot_run() -> None:
    """A construct at or below FLOOR is allowed (PEP 695 at the 3.12 floor)."""

    def at_or_below(version: str) -> bool:
        return tuple(map(int, version.split("."))) <= floor.FLOOR

    at_floor = [entry[2] for entry in floor.TOO_NEW if at_or_below(entry[1])]
    at_floor += [name for name, version in floor.TYPING_TOO_NEW.items() if at_or_below(version)]
    assert not at_floor, f"listed as newer than the {floor.FLOOR} floor but not: {at_floor}"


@pytest.mark.parametrize(
    ("source", "rule", "flagged"),
    FLOOR_DEFECTS,
    ids=[
        "typeis",
        "typeis-guarded",
        "forward-name",
        "string-union",
        "pep695-alias-union",
        "pep695-generic-class",
        "pep695-generic-def",
        "type-checking-only",
        "type-checking-quoted",
        "string-alias-union",
        "string-alias-subscript",
        "bound-first",
    ],
)
def test_the_floor_rules_flag_the_defects_that_reached_a_judge(
    source: str, rule: Callable[[str], list[str]], flagged: bool
) -> None:
    """Each rule is shown the exact text a py3.12 judge failed to import, and the fallback or
    ordering that makes the same line safe, so a rule that stops firing fails here."""
    assert bool(rule(source)) is flagged, rule(source)


if __name__ == "__main__":
    test_the_too_new_lists_name_only_what_the_floor_cannot_run()
    for source, rule, flagged in FLOOR_DEFECTS:
        test_the_floor_rules_flag_the_defects_that_reached_a_judge(source, rule, flagged)
