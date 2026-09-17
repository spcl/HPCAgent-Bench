# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""CI lint: no script may hardcode a path or a Slurm setting specific to one person's account.

This repo runs from ``/ritom/scratch/cscs/<user>/<arch>`` and submits to a Slurm account that
differs per person. A literal ``/users/ybudanaz/...``, ``/iopsstor/scratch/cscs/ybudanaz``, or
``-A a-g34`` works once, for the person who wrote it, and then silently reads or writes the WRONG
user's data (or bills the wrong project) for everyone else who checks the repo out. The central
resolvers exist precisely so nothing has to guess: ``scripts/cache_env.sh`` (FAST_SCRATCH,
HPCAGENT_BENCH_CACHE, HF_HOME, JIT_CACHE_ROOT), ``scripts/cscs/account_env.sh`` (the Slurm
account), ``experiments/env.sh`` (sources both) and ``EDF_PATH`` / ``${HOME}/.edf`` (EDFs).

The scan looks at LIVE code only, the same way ``test_no_literal_flags.py`` does: for Python it
walks the AST and inspects only non-docstring string literals (comments never reach the AST, and
a docstring recording migration history -- e.g. by naming the pre-rename project name in passing
-- is documentation, not a hardcoded setting); for shell/toml/env/yaml files it strips ``#`` line
comments first (markdown and JSON have no ``#``-comment syntax -- markdown's ``#`` is a heading
marker -- so those are scanned verbatim). A reference to ``${USER}``, ``$USER``, ``$(id -un)`` or
``$(uname -m)`` is a resolver, not a hardcode, and is never flagged.

Allowlisted files are legitimate: this file and its sibling test below embed the forbidden
strings on purpose (they test FOR their absence); ``scripts/cscs/account_env.sh`` is the one place
an account name may appear behind a check ruling it out; test fixtures use placeholder identities
(``/users/someone``, ``nid001234``) that are not real accounts. Adding a file here requires a
justification in this list.
"""

import ast
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]

_SKIP_DIRS = {
    ".git",
    "third_party",
    "__pycache__",
    ".cache",
    "hpcagent_bench.egg-info",
    "node_modules",
}

_PY_EXT = ".py"
_OTHER_EXTS = (".sh", ".sbatch", ".toml", ".yaml", ".yml", ".md", ".json")
_OTHER_SUFFIXES = (".toml.example",)
#: Formats with no ``#``-comment syntax -- markdown's ``#`` is a heading marker and JSON has no
#: comments at all -- so stripping after the first ``#`` would silently blind the scan on those.
_NO_HASH_COMMENT_EXTS = (".md", ".json")

#: The project's pre-rename name, split so this file (which must scan FOR it) does not itself
#: read as a leftover occurrence to a plain text search.
_LEGACY_NAME = "opt" + "arena"


def _is_env_or_makefile(name: str) -> bool:
    return name.startswith(".env") or name in ("Makefile", "makefile")


# name -> (compiled pattern, human explanation used in the failure message)
_PATTERNS = {
    "literal /users/<name> path": (
        re.compile(
            r"/users/(?!\$\{USER\}|\$USER\b|\$\(id -un\)|someone\b)[A-Za-z][A-Za-z0-9_.-]{1,31}"
        ),
        "use ${HOME} (or EDF_PATH) instead of a literal /users/<name> path -- see EDF_PATH / "
        "${HOME}/.edf in experiments/run_cluster.sh and experiments/preflight_gpu.sh",
    ),
    "literal /iopsstor/scratch/cscs/<name> path": (
        re.compile(r"/iopsstor/scratch/cscs/(?!\$\{USER\}|\$USER\b|\$\(id -un\))[A-Za-z]"),
        "use ${USER} (FAST_SCRATCH in scripts/cache_env.sh already does) instead of a literal user "
        "segment",
    ),
    "literal /ritom/scratch/cscs/<name> path": (
        re.compile(r"/ritom/scratch/cscs/(?!\$\{USER\}|\$USER\b|\$\(id -un\))[A-Za-z]"),
        "use ${USER} (SCRATCH is /ritom/scratch/cscs/$USER/$(uname -m)) instead of a literal user "
        "segment",
    ),
    "legacy project-name leftover": (
        re.compile(_LEGACY_NAME, re.IGNORECASE),
        f"the project was renamed from its pre-rename name ({_LEGACY_NAME}) to hpcagent-bench; "
        "this string should not appear in live code or docs (a historical docstring recording the "
        "rename without spelling out the pre-rename name is fine and is excluded)",
    ),
    "hardcoded Slurm account": (
        re.compile(r"(?<![\w-])(?:a-g34|a-g200|g34|g200)(?![\w-])"),
        "the Slurm account is resolved once by scripts/cscs/account_env.sh and exported; no "
        "submitter or #SBATCH directive should name one (see the HPCAgent-Bench launch contract)",
    ),
}

_ALLOW = {
    "tests/test_no_hardcoded_user_paths.py",  # this file: embeds the patterns' own text
    "tests/test_materialize_shared.py",  # asserts a-g34/a-g200 are ABSENT from account_env.sh
    "tests/test_agent_driver_sealed.py",  # HOST_HOME = "/users/someone" is a placeholder fixture
    "scripts/cscs/account_env.sh",  # the one file allowed to rule account names in/out by name
}


def _candidate_files():
    for p in REPO.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(REPO)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        if rel.as_posix() in _ALLOW:
            continue
        name = p.name
        if (
            p.suffix == _PY_EXT
            or p.suffix in _OTHER_EXTS
            or name.endswith(_OTHER_SUFFIXES)
            or _is_env_or_makefile(name)
        ):
            yield p, rel.as_posix()


#: AST nodes that carry a leading docstring (module / class / def / async def).
_DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _docstring_constant_ids(tree):
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, _DOCSTRING_OWNERS):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def _check(text: str, rel: str) -> list[str]:
    offenders = []
    for label, (pattern, _why) in _PATTERNS.items():
        for m in pattern.finditer(text):
            lineno = text.count("\n", 0, m.start()) + 1
            offenders.append(f"{rel}:{lineno}: {label}: {m.group(0)!r}")
    return offenders


def _py_offenders(text: str, rel: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _raw_offenders(text, rel)
    skip = _docstring_constant_ids(tree)
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            for label, (pattern, _why) in _PATTERNS.items():
                if pattern.search(node.value):
                    offenders.append(f"{rel}:{node.lineno}: {label}: {node.value.strip()[:100]!r}")
    return offenders


def _raw_offenders(text: str, rel: str) -> list[str]:
    """Line scan with ``#`` comments stripped -- shell, sbatch, toml, .env and yaml files. Markdown
    and JSON have no ``#``-comment syntax, so those are scanned verbatim (see _NO_HASH_COMMENT_EXTS)."""
    if rel.endswith(_NO_HASH_COMMENT_EXTS):
        stripped = text
    else:
        stripped = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    return _check(stripped, rel)


def _scan(root: pathlib.Path) -> list[str]:
    """Scan ``root`` the same way the real test scans REPO. Exposed so a synthetic tree can
    prove the detection actually fires (see test_lint_catches_a_reintroduced_hit below)."""
    offenders = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        name = p.name
        if not (
            p.suffix == _PY_EXT
            or p.suffix in _OTHER_EXTS
            or name.endswith(_OTHER_SUFFIXES)
            or _is_env_or_makefile(name)
        ):
            continue
        text = p.read_text(errors="ignore")
        relp = rel.as_posix()
        offenders += _py_offenders(text, relp) if p.suffix == _PY_EXT else _raw_offenders(text, relp)
    return offenders


def test_no_hardcoded_user_paths_or_accounts() -> None:
    offenders = []
    for p, rel in _candidate_files():
        text = p.read_text(errors="ignore")
        offenders += _py_offenders(text, rel) if p.suffix == _PY_EXT else _raw_offenders(text, rel)
    assert not offenders, (
        "Hardcoded user-specific paths or Slurm settings found -- route them through the central "
        "resolvers (scripts/cache_env.sh, scripts/cscs/account_env.sh, experiments/env.sh) or "
        "allowlist with a justification in this file:\n  " + "\n  ".join(sorted(offenders))
    )


def test_lint_catches_a_reintroduced_hit(tmp_path: pathlib.Path) -> None:
    """Proves the scan is not vacuously green: a synthetic tree with each offense must be caught."""
    (tmp_path / "bad.sh").write_text(
        "#!/usr/bin/env bash\n"
        "# this comment mentions /users/ybudanaz and must NOT be flagged\n"
        'CE_EDF="/users/ybudanaz/x86_64/.edf/agent.toml"\n'
        'FAST_SCRATCH="/iopsstor/scratch/cscs/ybudanaz"\n'
        'SCRATCH="/ritom/scratch/cscs/ybudanaz/$(uname -m)"\n'
        "sbatch --account=a-g34 bad.sh\n"
    )
    (tmp_path / "bad.py").write_text(
        f'"""Historical note: the {_LEGACY_NAME} rename moved these files. Not a violation."""\n'
        'REPO_DEFAULT = "/users/ybudanaz/x86_64/hpcagent-bench"\n'
        'ACCOUNT = "a-g200"\n'
    )
    (tmp_path / "bad.md").write_text(
        f"# Notes\n\nDo not reintroduce {_LEGACY_NAME} anywhere in the docs.\n"
    )
    (tmp_path / "good.sh").write_text(
        "#!/usr/bin/env bash\n"
        '. "${HPCAGENT_BENCH_REPO}/scripts/cscs/account_env.sh"\n'
        'FAST_SCRATCH="${FAST_SCRATCH:-/iopsstor/scratch/cscs/${USER}}"\n'
        'SCRATCH="/ritom/scratch/cscs/${USER}/$(uname -m)"\n'
    )

    offenders = _scan(tmp_path)

    assert any("bad.sh" in o and "/users/ybudanaz" in o for o in offenders), offenders
    assert any("bad.sh" in o and "iopsstor" in o for o in offenders), offenders
    assert any("bad.sh" in o and "ritom" in o for o in offenders), offenders
    assert any("bad.sh" in o and "a-g34" in o for o in offenders), offenders
    assert any("bad.py" in o and "REPO_DEFAULT" not in o and "/users/ybudanaz" in o for o in offenders), offenders
    assert any("bad.py" in o and "a-g200" in o for o in offenders), offenders
    # A reintroduction inside a markdown doc must be caught too -- '#' there is a heading marker,
    # not a comment, so this also proves the scan does not blind itself on the heading line.
    assert any("bad.md" in o and _LEGACY_NAME in o.lower() for o in offenders), offenders
    # bad.py's docstring records history without spelling out a live setting: the AST carve-out
    # must exclude it, so no bad.py offender may name the legacy project name.
    assert not any("bad.py" in o and _LEGACY_NAME in o.lower() for o in offenders), offenders
    assert not any("good.sh" in o for o in offenders), offenders
