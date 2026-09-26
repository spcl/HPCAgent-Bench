# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every version bound in pyproject.toml says why it is there, and the lists agree with each other.

An upper bound or an exact pin (``<``, ``<=``, ``==``, ``~=``) carries a comment naming the break or
the recorded run it reproduces, after it on its line or on the line above. Exact pins live only in the
dependency groups, whose job is to reproduce the recorded arms. A package named in several lists
(dependencies, extras, groups, build requirements) never gets a pin another list excludes. Static:
the file is read, nothing is resolved.
"""

import functools
import pathlib
import tomllib
from collections.abc import Iterator
from typing import NamedTuple

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"

#: Operators that cap a version from above or fix it outright.
CAPPING = frozenset({"<", "<=", "==", "~=", "==="})


class Entry(NamedTuple):
    """One requirement string as written, the list it is in, and its parse."""

    where: str
    text: str
    req: Requirement


@functools.cache
def source() -> str:
    return PYPROJECT.read_text(encoding="utf-8")


def requirement_lists() -> Iterator[tuple[str, list[str]]]:
    """``(list name, requirement strings)`` for every list of requirements in the file."""
    data = tomllib.loads(source())
    yield "build-system.requires", data["build-system"]["requires"]
    yield "project.dependencies", data["project"]["dependencies"]
    for name, entries in data["project"]["optional-dependencies"].items():
        yield f"optional-dependencies.{name}", entries
    for name, entries in data["dependency-groups"].items():
        yield f"dependency-groups.{name}", [entry for entry in entries if isinstance(entry, str)]


def entries() -> list[Entry]:
    """Every third-party requirement; the project's own extras (``hpcagent_bench[dev]``) are skipped."""
    project = canonicalize_name(tomllib.loads(source())["project"]["name"])
    return [
        Entry(where, text, Requirement(text))
        for where, texts in requirement_lists()
        for text in texts
        if canonicalize_name(Requirement(text).name) != project
    ]


def commented(text: str, document: str) -> bool:
    """Whether every line of ``document`` quoting ``text`` has a comment after it or right above it."""
    lines = document.splitlines()
    quoted = f'"{text}"'
    hits = [index for index, line in enumerate(lines) if quoted in line]
    assert hits, f"{quoted} is not written verbatim in the document"
    return all(
        "#" in lines[index].split(quoted, 1)[1] or (index > 0 and lines[index - 1].lstrip().startswith("#"))
        for index in hits
    )


def capped() -> list[Entry]:
    return [entry for entry in entries() if {spec.operator for spec in entry.req.specifier} & CAPPING]


@pytest.mark.parametrize("entry", capped(), ids=lambda entry: f"{entry.where}:{entry.text}")
def test_every_upper_bound_or_pin_says_why(entry: Entry) -> None:
    assert commented(entry.text, source()), f"{entry.where}: {entry.text} caps a version without a comment"


def test_exact_pins_live_only_in_the_dependency_groups() -> None:
    pinned = [
        f"{entry.where}: {entry.text}"
        for entry in entries()
        if any(spec.operator in ("==", "===") for spec in entry.req.specifier)
        and not entry.where.startswith("dependency-groups.")
    ]
    assert pinned == [], pinned


def test_a_package_named_twice_is_never_pinned_where_another_list_excludes_it() -> None:
    by_name: dict[str, list[Entry]] = {}
    for entry in entries():
        by_name.setdefault(canonicalize_name(entry.req.name), []).append(entry)
    clashes = [
        f"{pinning.where} pins {pinning.text}, {other.where} allows {other.text}"
        for named in by_name.values()
        for pinning in named
        for spec in pinning.req.specifier
        if spec.operator == "=="
        for other in named
        if not other.req.specifier.contains(spec.version, prereleases=True)
    ]
    assert clashes == [], clashes


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        ('    "pkg<2",  # 2.0 breaks x', True),
        ('    # 2.0 breaks x\n    "pkg<2",', True),
        ('    "pkg<2",', False),
        ('    "pkg<2",  # 2.0 breaks x\n    "pkg<2",', False),
    ],
)
def test_the_comment_rule_reads_trailing_and_preceding_comments(lines: str, expected: bool) -> None:
    assert commented("pkg<2", f'x = [\n    "other",\n{lines}\n]\n') is expected
