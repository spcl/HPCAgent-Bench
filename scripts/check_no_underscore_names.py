#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Refuse underscore-prefixed module and variable names, and self-named or underscore import aliases.

Rules (bare ``_``, dunder names such as ``__all__`` / ``__init__.py``, and :data:`ALLOWED` are exempt):
  U001  module file name starts with an underscore           (``_helpers.py``)
  U002  package directory name starts with an underscore      (``_vendor/``)
  U003  variable name starts with an underscore               (assignment, loop, comprehension, with,
                                                               walrus, except, parameter, match capture)
  U004  import alias repeats the imported name                (``import X as X``, ``from m import X as X``)
  U005  import alias starts with an underscore                (``import X as _x``, ``import X as _X``)

Scans the ``git ls-files`` Python sources of each repository given; exits 1 if anything is found.
"""

import argparse
import ast
import collections
import dataclasses
import pathlib
import subprocess
import sys
from collections.abc import Iterator


@dataclasses.dataclass(frozen=True, slots=True)
class Finding:
    path: str
    line: int
    col: int
    rule: str
    message: str


#: Names an outside protocol fixes: ``ast.AST`` subclasses must spell ``_fields`` / ``_attributes``, and a
#: DaCe ``Pass.apply_pass`` receives ``_pipeline_results`` by that name.
ALLOWED = frozenset({"_fields", "_attributes", "_pipeline_results"})


def is_underscored(name: str) -> bool:
    """True for ``_x`` / ``__x``; False for the throwaway ``_``, for dunders and for :data:`ALLOWED`."""
    if name in ALLOWED or name == "_":
        return False
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def target_names(node: ast.AST) -> Iterator[ast.Name]:
    if isinstance(node, ast.Name):
        yield node
    elif isinstance(node, (ast.Tuple, ast.List)):
        for element in node.elts:
            yield from target_names(element)
    elif isinstance(node, ast.Starred):
        yield from target_names(node.value)


def bound_names(node: ast.AST) -> Iterator[tuple[str, int, int, str]]:
    """``(name, line, col, kind)`` for every name ``node`` binds as a variable."""
    targets: list[tuple[ast.AST, str]] = []
    if isinstance(node, ast.Assign):
        targets = [(target, "assignment") for target in node.targets]
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [(node.target, "assignment")]
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        targets = [(node.target, "loop")]
    elif isinstance(node, ast.comprehension):
        targets = [(node.target, "comprehension")]
    elif isinstance(node, ast.withitem) and node.optional_vars is not None:
        targets = [(node.optional_vars, "with")]
    elif isinstance(node, ast.NamedExpr):
        targets = [(node.target, "walrus")]
    for target, kind in targets:
        for name in target_names(target):
            yield name.id, name.lineno, name.col_offset, kind
    if isinstance(node, ast.ExceptHandler) and node.name is not None:
        yield node.name, node.lineno, node.col_offset, "except"
    elif isinstance(node, ast.arg):
        yield node.arg, node.lineno, node.col_offset, "parameter"
    elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name is not None:
        yield node.name, node.lineno, node.col_offset, "match"
    elif isinstance(node, ast.MatchMapping) and node.rest is not None:
        yield node.rest, node.lineno, node.col_offset, "match"


def import_findings(path: str, node: ast.Import | ast.ImportFrom) -> Iterator[Finding]:
    for alias in node.names:
        if alias.asname is None:
            continue
        spelled = f"import {alias.name} as {alias.asname}"
        if isinstance(node, ast.ImportFrom):
            spelled = f"from {'.' * node.level}{node.module or ''} {spelled}"
        if alias.asname == alias.name:
            yield Finding(path, alias.lineno, alias.col_offset, "U004", f"alias repeats the name: `{spelled}`")
        if is_underscored(alias.asname):
            yield Finding(path, alias.lineno, alias.col_offset, "U005", f"underscore alias: `{spelled}`")


def source_findings(path: str, source: str) -> Iterator[Finding]:
    tree = ast.parse(source, filename=path)
    class_body_statements = {
        id(statement) for node in ast.walk(tree) if isinstance(node, ast.ClassDef) for statement in node.body
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield from import_findings(path, node)
            continue
        for name, line, col, kind in bound_names(node):
            if is_underscored(name):
                where = "class attribute" if id(node) in class_body_statements else f"{kind} variable"
                yield Finding(path, line, col, "U003", f"{where} `{name}`")


def tracked_python_files(repo: pathlib.Path) -> list[str]:
    listing = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z", "--", "*.py"], capture_output=True, check=True
    ).stdout
    return sorted(entry.decode() for entry in listing.split(b"\0") if entry)


def repo_findings(repo: pathlib.Path, only: list[str]) -> tuple[list[Finding], list[str]]:
    findings: list[Finding] = []
    unparsable: list[str] = []
    flagged_dirs: set[str] = set()
    for relative in tracked_python_files(repo):
        if only and not any(relative.startswith(prefix) for prefix in only):
            continue
        shown = f"{repo.name}/{relative}"
        parts = pathlib.PurePosixPath(relative).parts
        for depth, directory in enumerate(parts[:-1]):
            prefix = "/".join(parts[: depth + 1])
            if is_underscored(directory) and prefix not in flagged_dirs:
                flagged_dirs.add(prefix)
                findings.append(Finding(f"{repo.name}/{prefix}", 0, 0, "U002", f"package directory `{directory}`"))
        if is_underscored(pathlib.PurePosixPath(relative).stem):
            findings.append(Finding(shown, 0, 0, "U001", f"module file `{parts[-1]}`"))
        file = repo / relative
        if not file.is_file():  # tracked but deleted in the worktree, or a broken symlink
            continue
        try:
            findings.extend(source_findings(shown, file.read_text(encoding="utf-8", errors="replace")))
        except (SyntaxError, ValueError):
            unparsable.append(shown)
    return sorted(findings, key=lambda finding: (finding.path, finding.line, finding.col)), unparsable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repos", nargs="+", type=pathlib.Path, help="git repositories to scan")
    parser.add_argument(
        "--only", action="append", default=[], help="report only paths under this repo-relative prefix (repeatable)"
    )
    arguments = parser.parse_args()
    total = 0
    for repo in arguments.repos:
        findings, unparsable = repo_findings(repo.resolve(), arguments.only)
        for finding in findings:
            print(f"{finding.path}:{finding.line}:{finding.col}: {finding.rule} {finding.message}")
        counts = collections.Counter(finding.rule for finding in findings)
        summary = ", ".join(f"{rule}={counts[rule]}" for rule in sorted(counts)) or "clean"
        print(
            f"summary {repo.resolve().name}: {len(findings)} findings ({summary}); unparsable={len(unparsable)}",
            file=sys.stderr,
        )
        for path in unparsable:
            print(f"  unparsable: {path}", file=sys.stderr)
        total += len(findings)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
