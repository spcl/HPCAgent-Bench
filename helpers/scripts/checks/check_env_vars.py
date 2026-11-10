#!/usr/bin/env python
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Inventory every environment name the tree reads, as a TSV table (F10 alpha).

The env cleanup needs a measurable baseline: the same kind of scan that substantiated it, re-runnable after
each migration step, so the remaining surface only shrinks. Two sides:

* Python (AST, not grep): every literal name read through ``os.environ[...]`` / ``.get`` / ``.setdefault`` /
  ``.pop``, ``os.getenv``, and the ``environ``-style mappings the tree passes around (a bare receiver named
  ``environ``, ``environment`` or ``env``). Reads through a constant or a computed name are a known lower
  bound, as is every name assembled at runtime.
* Shell: every all-caps ``${NAME}`` in ``*.sh`` / ``*.bash`` / ``*.sbatch`` / ``*.env`` and in extensionless
  scripts with a shell shebang. Includes locals and launcher wiring, so it is an upper bound.

A name is ``project`` unless a vendor/tool prefix (``SLURM_``, ``OMP_``, ``CUDA_``, ``UV_``, ...) or a generic
name (``PATH``, ``HOME``, ...) claims it; that split is the cleanup's moving part, not a fact about the names.

Output: ``name, py, sh, first_py, first_sh, group`` sorted by total reads then name; ``first_*`` is
``<path>:<line>`` of the first read, root-relative, the column the cleanup's table is built from. Totals go
to stderr.

Run standalone: ``python helpers/scripts/checks/check_env_vars.py --out results/env_vars.tsv``
"""

import argparse
import ast
import os
import pathlib
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterator

from hpcagent_bench import paths

#: Directories no scan enters (vendored code, output trees, docs: the cleanup scope is code).
SKIP_DIRS = frozenset(
    {
        ".git",
        ".cache",
        ".ruff_cache",
        ".scratch",
        ".venv",
        "__pycache__",
        "docs",
        "node_modules",
        "results",
        "third_party",
    }
)

#: Suffixes the shell side scans, besides extensionless files whose first line is a shell shebang.
SHELL_SUFFIXES = frozenset({".sh", ".bash", ".sbatch", ".env"})
SHELL_SHEBANG = re.compile(r"^#!\s*\S*/(ba)?sh\b")

#: Bare receiver names whose ``.get`` / ``.setdefault`` / ``.pop`` / ``[...]`` reads count as environment reads
#: (the environ-style mappings the tree passes around); ``os.environ`` and ``os.getenv`` are handled besides.
ENV_RECEIVERS = frozenset({"env", "environ", "environment"})

#: Read methods of an environment mapping, besides subscripting.
ENV_METHODS = frozenset({"get", "pop", "setdefault"})

#: Read methods of the deprecated-name alias layer (agent/hpcagent_agent/env_aliases.py): a name that reaches the
#: environment only through it would otherwise drop out of the inventory as the cleanup migrates its readers.
ALIAS_METHODS = frozenset({"resolve", "value"})

#: Prefixes of the vendor and tool stacks whose names the project does not own; the env cleanup never moves one.
VENDOR_PREFIXES = (
    "ANTHROPIC_",
    "BLIS_",
    "CCACHE_",
    "CMAKE_",
    "CUDA_",
    "DACE_",
    "GCC_",
    "HCC_",
    "HF_",
    "HIP_",
    "KMP_",
    "LD_",
    "LITELLM_",
    "MKL_",
    "MPI_",
    "MPI4PY_",
    "MCP_",
    "MSWEA_",
    "NUMEXPR_",
    "NVIDIA_",
    "NV_",
    "OMP_",
    "OPENBLAS_",
    "PIP_",
    "PYTHON",
    "PYTORCH_",
    "ROCR_",
    "ROCM_",
    "SLURM",
    "UV_",
)

#: Generic shell and toolchain names no side owns.
GENERIC_NAMES = frozenset(
    {
        "CC",
        "CI",
        "CPATH",
        "CXX",
        "C_INCLUDE_PATH",
        "HOME",
        "IS_SANDBOX",
        "LANG",
        "LIBRARY_PATH",
        "PATH",
        "PKG_CONFIG_PATH",
        "PWD",
        "SHELL",
        "TMPDIR",
        "USER",
        "VIRTUAL_ENV",
    }
)

#: An all-caps ``${NAME}``, with the usual parameter expansions (``${VAR:-x}``) allowed after the name.
SHELL_VAR = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?![A-Za-z0-9_])")

TSV_HEADER = "# name\tpy\tsh\tfirst_py\tfirst_sh\tgroup"


def is_project(name: str) -> bool:
    """Whether the project owns ``name``: no vendor prefix, no generic shell name."""
    return name not in GENERIC_NAMES and not name.startswith(VENDOR_PREFIXES)


def files(root: pathlib.Path, wanted: Callable[[pathlib.Path], bool]) -> Iterator[pathlib.Path]:
    """Every file under ``root`` ``wanted`` accepts, pruned at the directories no scan enters."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in SKIP_DIRS)
        base = pathlib.Path(dirpath)
        for name in sorted(filenames):
            path = base / name
            if wanted(path):
                yield path


def is_python(path: pathlib.Path) -> bool:
    """Whether ``path`` is Python the scan reads."""
    return path.suffix == ".py"


def is_shell(path: pathlib.Path) -> bool:
    """Whether ``path`` is shell the scan reads: a known suffix, or a shell-shebang script with no suffix."""
    if path.suffix in SHELL_SUFFIXES:
        return True
    if path.suffix:
        return False
    try:
        with path.open("rb") as handle:
            return SHELL_SHEBANG.match(handle.readline(160).decode("utf-8", "replace")) is not None
    except OSError:
        return False


def record(counts: Counter, first: dict, name: str, where: str) -> None:
    """Count one read of ``name`` at ``where`` (``<path>:<line>``), keeping only the first."""
    counts[name] += 1
    first.setdefault(name, where)


def py_names(path: pathlib.Path, root: pathlib.Path, counts: Counter, first: dict) -> None:
    """Add ``path``'s literal environment reads to ``counts``, recording the first read of each name."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return
    where = f"{path.relative_to(root)}:{{line}}"
    for node in ast.walk(tree):
        name = read_name(node)
        if name is not None:
            record(counts, first, name, where.format(line=node.lineno))


def read_name(node: ast.AST) -> str | None:
    """The literal environment name ``node`` reads: a subscript or a read-method call on ``os.environ``,
    ``os.getenv``, or a bare environ-style receiver; None otherwise."""
    if isinstance(node, ast.Subscript):
        if isinstance(node.ctx, ast.Load) and isinstance(node.slice, ast.Constant) and env_receiver(node.value):
            return literal(node.slice)
        return None
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or not node.args:
        return None
    func, receiver = node.func, node.func.value
    if func.attr == "getenv":
        return literal(node.args[0]) if isinstance(receiver, ast.Name) and receiver.id == "os" else None
    if func.attr in ENV_METHODS and env_receiver(receiver):
        return literal(node.args[0])
    if func.attr in ALIAS_METHODS and isinstance(receiver, ast.Name) and receiver.id == "env_aliases":
        return literal(node.args[0])
    return None


def env_receiver(node: ast.AST) -> bool:
    """Whether ``node`` is the environment: ``os.environ`` or a bare environ-style receiver name."""
    if isinstance(node, ast.Name):
        return node.id in ENV_RECEIVERS
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def literal(node: ast.AST) -> str | None:
    """A Constant node's string value; None for anything else."""
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def sh_names(path: pathlib.Path, root: pathlib.Path, counts: Counter, first: dict) -> None:
    """Add ``path``'s ``${NAME}`` mentions to ``counts``, recording the first mention of each name."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    where = f"{path.relative_to(root)}:{{line}}"
    for match in SHELL_VAR.finditer(text):
        record(counts, first, match.group(1), where.format(line=text.count("\n", 0, match.start()) + 1))


def inventory(roots: list[pathlib.Path]) -> tuple[Counter, Counter, dict, dict]:
    """``(py counts, sh counts, py first reads, sh first reads)`` of every tree in ``roots``."""
    py_counts, sh_counts, py_first, sh_first = Counter(), Counter(), {}, {}
    for root in roots:
        for path in files(root, is_python):
            py_names(path, root, py_counts, py_first)
        for path in files(root, is_shell):
            sh_names(path, root, sh_counts, sh_first)
    return py_counts, sh_counts, py_first, sh_first


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=pathlib.Path, default=paths.ROOT, help="tree to scan")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="TSV destination (default: stdout)")
    args = parser.parse_args(argv)
    root = args.root.resolve()

    py_counts, sh_counts, py_first, sh_first = inventory([root])
    names = sorted(set(py_counts) | set(sh_counts), key=lambda name: (-(py_counts[name] + sh_counts[name]), name))
    rows = [
        "\t".join(
            str(cell)
            for cell in (
                name,
                py_counts[name],
                sh_counts[name],
                py_first.get(name, ""),
                sh_first.get(name, ""),
                "project" if is_project(name) else "vendor",
            )
        )
        for name in names
    ]
    table = "\n".join([TSV_HEADER, *rows])

    project_py = sum(1 for name in py_counts if is_project(name))
    shell_project = sum(1 for name in sh_counts if is_project(name))
    prefix = "HPCAGENT_BENCH_"
    scoped_py, _, _, _ = inventory([root / "hpcagent_bench", root / "agent"])
    scoped_project = sum(1 for name in scoped_py if is_project(name))
    print(f"env inventory of {root}", file=sys.stderr)
    print(
        f"python, whole tree: {len(py_counts)} distinct names, {project_py} project-specific",
        file=sys.stderr,
    )
    print(
        f"python, hpcagent_bench/ + agent/ (the F10 scope): {len(scoped_py)} distinct, {scoped_project} project"
        " (the F10 scan said 164 distinct / 127 project)",
        file=sys.stderr,
    )
    print(
        f"shell, whole tree: {len(sh_counts)} distinct ${{NAME}}, {shell_project} project-shaped"
        " (the F10 scan said 462, an upper bound over locals)",
        file=sys.stderr,
    )
    print(
        f"python reads of {prefix}* names: {sum(1 for name in py_counts if name.startswith(prefix))}", file=sys.stderr
    )
    print(
        f"shell mentions of {prefix}* names: {sum(1 for name in sh_counts if name.startswith(prefix))}", file=sys.stderr
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n", encoding="utf-8")
        print(f"wrote {len(rows)} names to {args.out}", file=sys.stderr)
    else:
        print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
