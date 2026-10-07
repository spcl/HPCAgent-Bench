# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The package parses, compiles and imports on the OLDEST interpreter that has to import it.

The venv and CI run 3.14, but an experiment does not: the job's container interpreter can be older.
Syntax, a typing name or an eagerly evaluated annotation newer than the floor raises at IMPORT, which
takes the whole setup down with no graded row while the suite, on 3.14, stays green. So the floor is
checked by reading, not by running:

* the package (``hpcagent_bench/`` minus generated files) parses under the floor grammar, compiles
  (``ast.parse`` does not enforce that a ``from __future__`` import comes first; ``compile`` does),
  uses no construct and no ``typing`` name newer than the floor;
* no tracked module evaluates an annotation at import that the floor cannot (a name bound later, a
  string joined with ``|`` or subscripted), and none postpones annotations: on the floor a future
  import only adds a second annotation semantics that hides exactly those faults.

    python helpers/scripts/checks/check_interpreter_floor.py [FILE ...]
"""

import ast
import builtins
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]

#: The oldest interpreter an experiment may hand the package to. Raise it only when every container
#: that runs a setup has been rebuilt past it.
FLOOR = (3, 12)


#: Constructs newer than FLOOR, each with the version that introduced it. A plain grep, because the
#: point is to catch them on an interpreter that CANNOT parse them.
TOO_NEW = ((re.compile(r"TypeVar\([^)]*\bdefault="), "3.13", "PEP 696 TypeVar default"),)


#: Names the typing module gained after FLOOR. Importing one parses and compiles everywhere and is
#: an ImportError only on the older interpreter.
TYPING_TOO_NEW = {
    "NoDefault": "3.13",
    "ReadOnly": "3.13",
    "TypeIs": "3.13",
    "get_protocol_members": "3.13",
    "is_protocol": "3.13",
    "evaluate_forward_ref": "3.14",
}


def too_new_typing_names(source: str) -> list[str]:
    """`from typing import X` or `typing.X` for an X above the floor, outside a `try` that falls back."""
    found: list[str] = []
    stack: list[ast.AST] = [ast.parse(source)]
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Try, ast.TryStar)) and node.handlers:
            stack.extend([*node.handlers, *node.orelse, *node.finalbody])
            continue
        names: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "typing":
            names = [node.attr]
        found.extend(
            f"{node.lineno}: typing.{name} needs {TYPING_TOO_NEW[name]}" for name in names if name in TYPING_TOO_NEW
        )
        stack.extend(ast.iter_child_nodes(node))
    return found


def is_type_checking(test: ast.expr) -> bool:
    """`if TYPE_CHECKING:` -- a block that binds nothing when the module runs."""
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def bindings(stmt: ast.stmt) -> set[str]:
    """Names a statement binds in the scope it runs in; a def or class body binds in its own."""
    names: set[str] = set()
    stack: list[ast.AST] = [stmt]
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            continue
        if isinstance(node, ast.If) and is_type_checking(node.test):
            stack.extend(node.orelse)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        stack.extend(ast.iter_child_nodes(node))
    return names


def blocks(stmt: ast.stmt) -> list[list[ast.stmt]]:
    """The statement lists that run in the same scope as `stmt`."""
    if isinstance(stmt, ast.If) and is_type_checking(stmt.test):
        return [stmt.orelse]
    if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While)):
        return [stmt.body, stmt.orelse]
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return [stmt.body]
    if isinstance(stmt, (ast.Try, ast.TryStar)):
        return [stmt.body, *(handler.body for handler in stmt.handlers), stmt.orelse, stmt.finalbody]
    return []


def annotations_of(stmt: ast.stmt) -> list[ast.expr]:
    """The annotations the floor evaluates when `stmt` runs."""
    if isinstance(stmt, ast.AnnAssign):
        return [stmt.annotation]
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = stmt.args
        every = [*args.posonlyargs, *args.args, *args.kwonlyargs, *(a for a in (args.vararg, args.kwarg) if a)]
        return [a.annotation for a in every if a.annotation is not None] + ([stmt.returns] if stmt.returns else [])
    return []


def string_aliases(source: str) -> frozenset[str]:
    """Names bound as `X: TypeAlias = "..."`, which are a str at runtime."""
    return frozenset(
        stmt.target.id
        for stmt in ast.parse(source).body
        if isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.target, ast.Name)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
        and "TypeAlias" in ast.unparse(stmt.annotation)
    )


def is_runtime_str(node: ast.expr, aliases: frozenset[str]) -> bool:
    """A string literal, or a name spelled as a string TypeAlias."""
    return (isinstance(node, ast.Constant) and isinstance(node.value, str)) or (
        isinstance(node, ast.Name) and node.id in aliases
    )


def annotation_faults(annotation: ast.expr, bound: set[str], starred: bool, aliases: frozenset[str]) -> list[str]:
    """What evaluating `annotation` raises when only `bound` names exist yet."""
    faults: list[str] = []
    for node in ast.walk(annotation):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            if any(is_runtime_str(side, aliases) for side in (node.left, node.right)):
                faults.append(f"{node.lineno}: a string joined with | is a TypeError")
        elif isinstance(node, ast.Subscript) and is_runtime_str(node.value, aliases):
            faults.append(f"{node.lineno}: a string subscripted is a TypeError")
        elif isinstance(node, ast.Name) and not starred and node.id not in bound:
            faults.append(f"{node.lineno}: {node.id} is not bound yet, a NameError")
    return faults


def scope_faults(body: list[ast.stmt], bound: set[str], starred: bool, aliases: frozenset[str]) -> list[str]:
    """Walk one scope in execution order, checking each eager annotation against what is bound."""
    faults: list[str] = []
    for stmt in body:
        # PEP 695: a generic def/class binds its type parameters for its own annotations and body
        params = {param.name for param in getattr(stmt, "type_params", ())}
        for annotation in annotations_of(stmt):
            faults.extend(annotation_faults(annotation, bound | params, starred, aliases))
        if isinstance(stmt, ast.ClassDef):
            faults.extend(scope_faults(stmt.body, bound | params, starred, aliases))
        for block in blocks(stmt):
            faults.extend(scope_faults(block, set(bound), starred, aliases))
        bound.update(bindings(stmt))
    return faults


def eager_annotation_faults(source: str, imported: frozenset[str] = frozenset()) -> list[str]:
    """Import-time annotation failures on the floor, which evaluates annotations when defined."""
    tree = ast.parse(source)
    starred = any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in ast.walk(tree))
    return scope_faults(tree.body, set(dir(builtins)), starred, imported | string_aliases(source))


def package_sources() -> list[pathlib.Path]:
    """Every module an experiment can import: the package minus its generated files."""
    return [p for p in sorted((REPO / "hpcagent_bench").rglob("*.py")) if not p.name.endswith("_generated.py")]


def tracked_sources() -> list[pathlib.Path]:
    """Every tracked hand-written module: the package plus the drivers, tools and tests beside it."""
    listed = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z", "*.py"], capture_output=True, text=True, check=True
    )
    return [REPO / p for p in listed.stdout.split("\0") if p and not p.endswith("_generated.py")]


def package_faults(path: pathlib.Path, text: str) -> list[str]:
    """Floor faults of one package module: grammar, compile, too-new constructs and typing names."""
    try:
        ast.parse(text, feature_version=FLOOR)
        compile(text, str(path), "exec", dont_inherit=True)
    except SyntaxError as exc:
        return [f"{exc.lineno}: {exc.msg} on the {FLOOR[0]}.{FLOOR[1]} floor"]
    faults = [
        f"{text.count(chr(10), 0, m.start()) + 1}: {what} needs {version}"
        for pattern, version, what in TOO_NEW
        for m in [pattern.search(text)]
        if m
    ]
    return faults + too_new_typing_names(text)


def main(argv: list[str]) -> int:
    """Check the modules named in ``argv``; with none, every tracked module and every package module,
    tracked or not -- the numba and other references the harness generates beside a kernel are
    untracked and imported all the same."""
    wanted = {(REPO / rel).resolve() for rel in argv}
    tracked = [path for path in tracked_sources() if path.is_file()]
    candidates = tracked if wanted else sorted({*tracked, *package_sources()})
    texts = {path: path.read_text() for path in candidates if not wanted or path.resolve() in wanted}
    # A string TypeAlias is visible to every module that imports it, so the alias set is the tree's.
    aliases = frozenset().union(
        *(string_aliases(path.read_text()) for path in tracked if "TypeAlias" in path.read_text())
    )
    package = set(package_sources())
    faults = []
    for path, text in texts.items():
        rel = path.relative_to(REPO)
        if path in package:
            faults += [f"{rel}:{fault}" for fault in package_faults(path, text)]
        faults += [f"{rel}:{fault}" for fault in eager_annotation_faults(text, aliases)]
        if any(isinstance(n, ast.ImportFrom) and n.module == "__future__" for n in ast.parse(text).body):
            faults.append(f"{rel}: postpones annotations (from __future__ import annotations)")
    print("\n".join(faults))
    return 1 if faults else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
