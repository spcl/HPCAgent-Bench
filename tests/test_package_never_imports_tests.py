# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The installed package must not import this repository's ``tests`` package: a wheel ships
``hpcagent_bench/`` without ``tests/``, so such an import fails the moment the module loads."""

import ast
import pathlib

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "hpcagent_bench"

#: Modules that import DaCe's own ``tests.corpus`` (not this repository's), swapping ``tests`` in
#: ``sys.modules`` for the span of that import (:func:`hpcagent_bench.metrics.parallelism.import_dace_tests_corpus`).
DACE_TESTS_IMPORTERS = frozenset({"metrics/parallelism.py"})


def is_test_file(path: pathlib.Path) -> bool:
    """Test modules that live inside the package tree; pytest collects them, the package never imports them."""
    rel = path.relative_to(PACKAGE)
    return path.name.startswith("test_") or path.name == "conftest.py" or "tests" in rel.parts[:-1]


def imported_modules(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


def test_no_package_module_imports_the_repository_tests_package() -> None:
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if is_test_file(path) or path.relative_to(PACKAGE).as_posix() in DACE_TESTS_IMPORTERS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        bad = [name for name in imported_modules(tree) if name == "tests" or name.startswith("tests.")]
        if bad:
            offenders.append(f"{path.relative_to(PACKAGE)}: {', '.join(bad)}")
    assert not offenders, "package modules import the repository's tests package:\n" + "\n".join(offenders)
