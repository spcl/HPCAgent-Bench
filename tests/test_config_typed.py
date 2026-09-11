# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A config value arrives with a type, or it raises.

``config.get`` returns a union, so every caller of it inherits an unchecked value: a key read as a
string and compared against one still type-checks when it holds an int, and the mistake surfaces far
from the key that caused it. The typed accessors convert once and refuse a value that cannot be what
was asked for.
"""

import pytest

from hpcagent_bench import config


#: The typed accessors, by name, so a table-driven case names one without reaching for getattr.
ACCESSORS = {
    "get_str": config.get_str,
    "get_bool": config.get_bool,
    "get_int": config.get_int,
    "get_float": config.get_float,
}


@pytest.fixture
def pinned():
    """Set keys the way an env var or a runtime override does, and clear them after."""
    keys: list[str] = []

    def pin(dotted: str, value: object) -> str:
        config.set_override(dotted, value)
        keys.append(dotted)
        return dotted

    yield pin
    for key in keys:
        config.clear_override(key)


@pytest.mark.parametrize(
    "stored, want",
    [("hello", "hello"), (7, "7"), (1.5, "1.5"), (True, "True"), ("", "")],
)
def test_a_value_read_as_text_arrives_as_text(pinned, stored, want):
    pinned("t.value", stored)
    assert config.get_str("t.value") == want


@pytest.mark.parametrize(
    "stored, want",
    [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("", False),
        (True, True),
        (False, False),
        (1, True),
        (0, False),
    ],
)
def test_a_flag_reads_the_spellings_an_env_var_can_carry(pinned, stored, want):
    """An env var can only carry text, so `HPCAGENT_BENCH_X=true` and a YAML `true` must agree."""
    pinned("t.flag", stored)
    assert config.get_bool("t.flag") is want


@pytest.mark.parametrize("stored, want", [(7, 7), ("7", 7), (7.0, 7), (True, 1), (" 8 ", 8)])
def test_an_integer_reads_through_the_spellings(pinned, stored, want):
    pinned("t.n", stored)
    assert config.get_int("t.n") == want


@pytest.mark.parametrize(
    "accessor, default", [("get_str", "d"), ("get_bool", True), ("get_int", 5), ("get_float", 2.5)]
)
def test_a_missing_key_returns_the_default(accessor, default):
    assert ACCESSORS[accessor]("t.absent.entirely", default) == default


@pytest.mark.parametrize(
    "accessor, default", [("get_str", "d"), ("get_bool", True), ("get_int", 5), ("get_float", 2.5)]
)
def test_a_key_holding_none_returns_the_default(pinned, accessor, default):
    """A key present but null is absent as far as a caller is concerned, and must not become the
    string "None" or the integer 0."""
    pinned("t.null", None)
    assert ACCESSORS[accessor]("t.null", default) == default


@pytest.mark.parametrize(
    "accessor, stored",
    [
        ("get_str", ["a", "b"]),
        ("get_str", {"a": 1}),
        ("get_int", "not-a-number"),
        ("get_int", 1.5),
        ("get_float", "not-a-number"),
        ("get_bool", "maybe"),
        ("get_bool", ["a"]),
    ],
)
def test_a_value_that_cannot_be_the_type_asked_for_raises(pinned, accessor, stored):
    """It raises AT THE KEY. Coercing it silently is how mpi.launcher became a 34-character string
    that launched one argument per character."""
    pinned("t.wrong", stored)
    with pytest.raises((TypeError, ValueError)):
        ACCESSORS[accessor]("t.wrong")


def test_a_non_whole_float_is_not_an_integer(pinned):
    """Truncating 1.5 to 1 would silently halve a count nobody set to 1."""
    pinned("t.n", 1.5)
    with pytest.raises(TypeError, match="integer"):
        config.get_int("t.n")


def test_an_env_var_reaches_the_typed_accessor(monkeypatch):
    """The env is how a campaign sets every one of these, so it is the path that has to work."""
    monkeypatch.setenv("HPCAGENT_BENCH_T_FROM_ENV", "12")
    assert config.get_int("t.from_env") == 12
    assert config.get_str("t.from_env") == "12"
