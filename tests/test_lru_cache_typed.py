# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every functools cache in production code is ``typed=True``.

An untyped cache keys ``1``, ``1.0`` and ``True`` (and ``numpy.float32(x)`` against a Python float)
onto one entry, so a dtype- or precision-keyed lookup silently returns the value computed for the
other type.
"""

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PRODUCTION_ROOTS: tuple[str, ...] = ("hpcagent_bench", "experiments")
CACHE_DECORATORS = frozenset({"lru_cache", "cache"})


def functools_aliases(tree: ast.Module) -> frozenset[str]:
    """Local names bound to ``functools.lru_cache`` / ``functools.cache`` by a from-import."""
    return frozenset(
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "functools"
        for alias in node.names
        if alias.name in CACHE_DECORATORS
    )


def is_cache_decorator(decorator: ast.expr, aliases: frozenset[str]) -> bool:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
        return target.value.id == "functools" and target.attr in CACHE_DECORATORS
    return isinstance(target, ast.Name) and target.id in aliases


def is_typed(decorator: ast.expr) -> bool:
    if not isinstance(decorator, ast.Call):
        return False
    return any(
        kw.arg == "typed" and isinstance(kw.value, ast.Constant) and kw.value.value is True for kw in decorator.keywords
    )


def untyped_cache_lines(source: str) -> list[int]:
    """Line of every cache decorator in ``source`` that is not ``typed=True``."""
    tree = ast.parse(source)
    aliases = functools_aliases(tree)
    return [
        decorator.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for decorator in node.decorator_list
        if is_cache_decorator(decorator, aliases) and not is_typed(decorator)
    ]


@pytest.mark.parametrize(
    ("source", "want"),
    [
        ("import functools\n@functools.lru_cache\ndef f(): pass\n", [2]),
        ("import functools\n@functools.lru_cache()\ndef f(): pass\n", [2]),
        ("import functools\n@functools.lru_cache(maxsize=1)\ndef f(): pass\n", [2]),
        ("import functools\n@functools.lru_cache(maxsize=1, typed=False)\ndef f(): pass\n", [2]),
        ("import functools\n@functools.cache\ndef f(): pass\n", [2]),
        ("from functools import lru_cache\n@lru_cache(maxsize=None)\ndef f(): pass\n", [2]),
        ("from functools import lru_cache as memo\n@memo(maxsize=8)\ndef f(): pass\n", [2]),
        ("from functools import cache\nclass A:\n    @cache\n    def f(self): pass\n", [3]),
        ("import functools\n@functools.lru_cache(maxsize=None, typed=True)\ndef f(): pass\n", []),
        ("from functools import lru_cache\n@lru_cache(maxsize=256, typed=True)\ndef f(): pass\n", []),
        ("def cache(f): return f\n@cache\ndef f(): pass\n", []),
    ],
    ids=[
        "bare",
        "empty-call",
        "maxsize-only",
        "typed-false",
        "functools-cache",
        "from-import",
        "aliased",
        "method",
        "typed",
        "from-import-typed",
        "unrelated-cache-name",
    ],
)
def test_the_scan_flags_exactly_the_untyped_caches(source: str, want: list[int]) -> None:
    assert untyped_cache_lines(source) == want


def test_every_production_cache_decorator_is_typed() -> None:
    offenders = [
        f"{path.relative_to(REPO)}:{line}"
        for root in PRODUCTION_ROOTS
        for path in sorted((REPO / root).rglob("*.py"))
        if "cache" in (text := path.read_text(encoding="utf-8"))
        for line in untyped_cache_lines(text)
    ]
    assert offenders == [], offenders
