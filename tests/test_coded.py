# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
""":class:`hpcagent_agent.coded.Coded`: members are ``enum.auto()``, the spelling exists only at a boundary."""

import enum

import pydantic
import pytest
from hpcagent_agent.coded import Coded


class Mode(Coded):
    SEPARATOR = enum.nonmember("-")

    PY_BINDING = enum.auto()
    SOURCE = enum.auto()


class Dialect(Coded):
    SPELLINGS = enum.nonmember({"CPP": "c++"})

    C = enum.auto()
    CPP = enum.auto()
    LOOP_LEVEL = enum.auto()


class Clash(Coded):
    SPELLINGS = enum.nonmember({"B": "a"})

    A = enum.auto()
    B = enum.auto()


class Wire(pydantic.BaseModel):
    mode: Mode
    dialect: Dialect | None = None


def test_a_member_is_spelled_by_its_name_separator_and_overrides() -> None:
    assert Mode.PY_BINDING.text == "py-binding"
    assert Dialect.CPP.text == "c++"
    assert Dialect.LOOP_LEVEL.text == "loop_level"
    assert Mode.texts() == ("py-binding", "source")


def test_parse_inverts_text_and_names_the_choices() -> None:
    for member in (*Mode, *Dialect):
        assert type(member).parse(member.text) is member
    with pytest.raises(ValueError, match="one of py-binding, source"):
        Mode.parse("py_binding")


def test_a_member_is_not_its_spelling() -> None:
    assert Mode.SOURCE != "source"
    assert not isinstance(Mode.SOURCE.value, str)


def test_two_members_spelled_alike_are_refused() -> None:
    with pytest.raises(TypeError, match="both spelled 'a'"):
        Clash.parse("a")


def test_a_boundary_model_reads_and_writes_the_spelling() -> None:
    wire = Wire.model_validate({"mode": "py-binding", "dialect": "c++"})
    assert wire.mode is Mode.PY_BINDING
    assert wire.dialect is Dialect.CPP
    assert wire.model_dump(mode="json") == {"mode": "py-binding", "dialect": "c++"}
    assert Wire.model_validate_json(wire.model_dump_json()) == wire
    assert Wire(mode=Mode.SOURCE).mode is Mode.SOURCE
    assert Wire.model_json_schema()["properties"]["mode"]["enum"] == ["py-binding", "source"]
    with pytest.raises(pydantic.ValidationError, match="not a Mode"):
        Wire.model_validate({"mode": "nope"})
