# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Package conventions: every module names its public surface (``__all__``), every class has static
fields (``__slots__`` / ``@dataclass(slots=True)``), and every memoised function is typed
(``lru_cache(..., typed=True)``: ``f(1)`` and ``f(1.0)`` are different calls)."""

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
MODULES = sorted(p for p in (REPO / "hpcagent_bench").rglob("*.py") if "benchmarks" not in p.parts)

#: Classes that need a per-instance ``__dict__``: each names why.
DICT_CLASSES: dict[str, str] = {
    "BaseEmitter": "functools.cached_property stores into the instance __dict__",
}


def parsed(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def rel(path: pathlib.Path) -> str:
    return str(path.relative_to(REPO))


def defines_all(tree: ast.Module) -> bool:
    return any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            getattr(t, "id", "") == "__all__" for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
        for node in tree.body
    )


def public_names(tree: ast.Module) -> list[str]:
    names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    return [name for name in names if not name.startswith("_")]


def test_every_module_with_a_public_name_declares_all() -> None:
    missing = [rel(p) for p in MODULES if public_names(tree := parsed(p)) and not defines_all(tree)]
    assert not missing, "modules without __all__:\n  " + "\n  ".join(missing)


def test_every_dataclass_is_slotted() -> None:
    offenders = [
        f"{rel(p)}:{node.lineno} {node.name}"
        for p in MODULES
        for node in ast.walk(parsed(p))
        if isinstance(node, ast.ClassDef)
        for deco in node.decorator_list
        if "dataclass" in ast.unparse(deco) and "slots=True" not in ast.unparse(deco)
    ]
    assert not offenders, "dataclasses without slots=True:\n  " + "\n  ".join(offenders)


def test_every_base_less_class_declares_slots() -> None:
    offenders = []
    for p in MODULES:
        for node in ast.walk(parsed(p)):
            if not isinstance(node, ast.ClassDef) or node.bases or node.decorator_list or node.keywords:
                continue
            slotted = any(
                isinstance(s, ast.Assign) and any(getattr(t, "id", "") == "__slots__" for t in s.targets)
                for s in node.body
            )
            if not slotted and node.name not in DICT_CLASSES:
                offenders.append(f"{rel(p)}:{node.lineno} {node.name}")
    assert not offenders, "classes without __slots__:\n  " + "\n  ".join(offenders)


@pytest.mark.parametrize("path", [pytest.param(p, id=rel(p)) for p in MODULES])
def test_every_cache_is_typed(path: pathlib.Path) -> None:
    untyped = [
        f"{node.lineno} {node.name}: {ast.unparse(deco)}"
        for node in ast.walk(parsed(path))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for deco in node.decorator_list
        if (text := ast.unparse(deco)).split("(")[0].split(".")[-1] in ("lru_cache", "cache")
        and "typed=True" not in text
    ]
    assert not untyped, untyped
