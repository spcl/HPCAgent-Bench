# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Members are static: production code never reaches an attribute through a name held in a variable
(``getattr``/``hasattr``/``setattr``/``delattr``/``vars``), except the audited functions in :data:`AUDITED`."""

import ast
import pathlib

from hpcagent_bench.paths import ROOT as REPO

SOURCES = ("hpcagent_bench", "agent/hpcagent_agent", "helpers")
BUILTINS = frozenset({"getattr", "hasattr", "setattr", "delattr", "vars"})

#: ``(path, enclosing function)`` of every audited by-name access.
AUDITED = frozenset(
    {
        ("hpcagent_bench/modules.py", "members"),
        ("hpcagent_bench/modules.py", "defines"),
        ("hpcagent_bench/modules.py", "optional"),
        ("hpcagent_bench/modules.py", "optional_function"),
        ("hpcagent_bench/translators/numpyto_common/ast_build.py", "set_field"),
        ("hpcagent_bench/translators/numpyto_common/ast_build.py", "attach"),
        ("hpcagent_bench/translators/numpyto_common/ast_build.py", "attached"),
        ("hpcagent_bench/translators/numpyto_common/ast_build.py", "walk_nodes"),
        ("hpcagent_bench/cluster/systems.py", "script_header"),
        ("hpcagent_bench/cluster/systems.py", "main"),
        ("hpcagent_bench/cluster/systems.py", "options_main"),
    }
)


def name_accesses(path: pathlib.Path) -> list[tuple[str, int, str]]:
    """``(enclosing function, line, builtin)`` of every by-name attribute builtin called in ``path``."""
    found: list[tuple[str, int, str]] = []

    def walk(node: ast.AST, owner: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                walk(child, child.name)
                continue
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id in BUILTINS:
                found.append((owner, child.lineno, child.func.id))
            walk(child, owner)

    walk(ast.parse(path.read_text(encoding="utf-8")), "")
    return found


def test_production_code_reads_members_statically() -> None:
    offenders = [
        f"{rel}:{line} {builtin} in {owner or '<module>'}"
        for root in SOURCES
        for path in sorted((REPO / root).rglob("*.py"))
        for rel in [path.relative_to(REPO).as_posix()]
        for owner, line, builtin in name_accesses(path)
        if (rel, owner) not in AUDITED
    ]
    assert not offenders, offenders


def test_every_audited_access_still_exists() -> None:
    """A listed access that was removed leaves a hole a new one could hide in."""
    for rel, owner in AUDITED:
        assert any(found == owner for found, _line, _builtin in name_accesses(REPO / rel)), (rel, owner)


if __name__ == "__main__":
    test_production_code_reads_members_statically()
    test_every_audited_access_still_exists()
