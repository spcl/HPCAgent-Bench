# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Repo-wide guard: no file hardcodes a value that belongs to one site or one person.

Site values -- storage mounts, the Slurm account and partition, node names, user names -- come from
the environment, with ONE default place each (docs/configuration.md):

* ``scripts/site_env.sh`` loads the site layer (``experiments/layers/site-<name>.env``): fast
  storage (``FAST_SCRATCH``), ``SBATCH_PARTITION``, node exclusions, vendor artefact paths;
* ``scripts/cache_env.sh`` derives every cache path from ``SCRATCH`` / ``FAST_SCRATCH``;
* ``scripts/cscs/account_env.sh`` resolves the account from the user's own Slurm associations;
* ``hpcagent_bench/paths.py`` is the Python side of the same roots.

A literal works once, for the person who wrote it, and then silently reads or writes the wrong
user's data (or bills the wrong project) for everyone else.

The scan looks at LIVE text only: for Python, non-docstring string literals (a docstring or comment
may name a site to explain it); for shell/toml/env/yaml, lines with ``#`` comments stripped, except
``#SBATCH`` directives, which Slurm executes; markdown and JSON verbatim, since they have no
``#``-comment syntax. ``${USER}``, ``$USER``, ``$(id -un)`` and the placeholder ``/users/someone``
are resolvers or fixtures, never flagged.
"""

import ast
import pathlib
import re
from collections.abc import Iterator
from collections.abc import Set as AbstractSet

REPO = pathlib.Path(__file__).resolve().parents[1]

_SKIP_DIRS = {".git", "third_party", "__pycache__", ".cache", "hpcagent_bench.egg-info", "node_modules", "results"}

_PY_EXT = ".py"
_OTHER_EXTS = (".sh", ".sbatch", ".toml", ".yaml", ".yml", ".md", ".json", ".env", ".def", ".cfg", ".ini", ".html")
_OTHER_SUFFIXES = (".toml.example",)
_OTHER_NAMES = ("Dockerfile", "Makefile", "makefile")
#: Formats with no ``#``-comment syntax: stripping after ``#`` would blind the scan on those.
_NO_HASH_COMMENT_EXTS = (".md", ".json", ".html")

#: The project's pre-rename name, split so this file does not itself read as a leftover.
_LEGACY_NAME = "opt" + "arena"

_USER_RESOLVER = r"(?!\$\{USER\}|\$USER\b|\$\(id -un\))"

# name -> (pattern, fix); applied to every scanned file.
_PATTERNS = {
    "literal /users/<name> path": (
        re.compile(r"/users/(?!\$\{USER\}|\$USER\b|\$\(id -un\)|someone\b)[A-Za-z][A-Za-z0-9_.-]{1,31}"),
        "use ${HOME} (or EDF_PATH) instead of a literal home directory",
    ),
    "literal storage mount": (
        re.compile(r"(?<![\w$])/(?:ritom|iopsstor)(?:/|\b)|/capstor/scratch(?:/|\b)"),
        "route through ${SCRATCH} / ${FAST_SCRATCH}; the site layer names the mount",
    ),
    "legacy project-name leftover": (
        re.compile(_LEGACY_NAME, re.IGNORECASE),
        f"the project was renamed from {_LEGACY_NAME} to hpcagent-bench",
    ),
}

#: Never anywhere, comments and docstrings included: a user name, a site email or an account in
#: prose is copied into code, and Slurm executes ``#SBATCH`` lines although they read as comments.
RAW_PATTERNS = {
    "user name": (re.compile(r"\bybudanaz\b"), "use ${USER} / $(id -un)"),
    "site email": (
        re.compile(r"[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)*(?:ethz|cscs)\.ch\b"),
        "no personal or site email in the tree",
    ),
    "hardcoded Slurm account": (
        re.compile(r"(?<![\w-])(?:a-g34|a-g200|g34|g200)(?![\w-])"),
        "the account comes from SBATCH_ACCOUNT (scripts/cscs/account_env.sh)",
    ),
    "site value in an #SBATCH directive": (
        re.compile(
            r"(?m)^[ \t]*#SBATCH[ \t]+(?:--(?:partition|account|reservation|nodelist|exclude)=\S+|-[Apw][ \t]+\S+)"
        ),
        "the partition and account come from SBATCH_PARTITION / SBATCH_ACCOUNT (site layer)",
    ),
}

#: Node and partition literals in executable code (tests may fixture node names).
CODE_PATTERNS = {
    "node name": (re.compile(r"\bnid(?:\d{6}|\[[^\]\s]*\]?)"), "node lists belong in the site layer"),
    "literal Slurm partition": (
        re.compile(r"--partition[= ](?![\"'$<{])[A-Za-z]\S*"),
        "the default partition is SBATCH_PARTITION (site layer); pass --partition only from a variable",
    ),
}

#: Files that legitimately carry a flagged string. Adding one requires a reason here.
_ALLOW = {
    "tests/test_no_hardcoded_user_paths.py",  # this file: embeds the patterns' own text
    "experiments/layers/site-cscs.env",  # THE site layer for one real site: its values live here
    "docs/configuration.md",  # shows that site layer's values next to the generic ones
    "pyproject.toml",  # package author contact (PyPI metadata), not a runtime value
    # Legacy MCP server/env keys READ from already-recorded rows and worker dirs, and fixtures of them.
    "hpcagent_bench/observations_extract.py",
    "experiments/owed_wave.py",
    "tests/test_fused_owed_wave.py",
    "tests/test_extract_llr40_task_rows.py",
    "tests/test_ablation_stats.py",
}

#: Areas another change set is cleaning; each must leave this tuple once clean.
PENDING = (
    "containers/images/",
    "containers/inference/",
    "containers/lib/git_mirror.sh",
    "containers/lib/device_arch_gate.sh",
)


def is_candidate(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return (
        name.endswith((_PY_EXT, *_OTHER_EXTS, *_OTHER_SUFFIXES, ".Dockerfile"))
        or name.startswith(".env")
        or name in _OTHER_NAMES
    )


def candidate_files(root: pathlib.Path) -> Iterator[tuple[pathlib.Path, str]]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if any(part in _SKIP_DIRS for part in rel.split("/")):
            continue
        if is_candidate(rel):
            yield p, rel


#: AST nodes that carry a leading docstring (module / class / def / async def).
_DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _docstring_constant_ids(tree: ast.AST) -> set[int]:
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


def _match(text: str, rel: str, patterns: dict, line_of: int | None = None) -> list[str]:
    offenders = []
    for label, (pattern, _fix) in patterns.items():
        for m in pattern.finditer(text):
            lineno = line_of if line_of is not None else text.count("\n", 0, m.start()) + 1
            offenders.append(f"{rel}:{lineno}: {label}: {m.group(0).strip()[:100]!r}")
    return offenders


def _strip_comments(text: str, rel: str) -> str:
    if rel.endswith(_NO_HASH_COMMENT_EXTS):
        return text
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def file_offenders(text: str, rel: str) -> list[str]:
    """Every hit in one file's live text."""
    code_rules = not rel.startswith("tests/") and not rel.endswith(_NO_HASH_COMMENT_EXTS)
    if rel.endswith(_PY_EXT):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = None
        if tree is not None:
            skip = _docstring_constant_ids(tree)
            offenders = _match(text, rel, RAW_PATTERNS)
            patterns = _PATTERNS | (CODE_PATTERNS if code_rules else {})
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
                    offenders += _match(node.value, rel, patterns, line_of=node.lineno)
            return offenders
    offenders = _match(text, rel, RAW_PATTERNS)
    live = _strip_comments(text, rel)
    offenders += _match(live, rel, _PATTERNS | (CODE_PATTERNS if code_rules else {}))
    return offenders


def scan(root: pathlib.Path, allow: AbstractSet[str] = frozenset(), pending: tuple[str, ...] = ()) -> list[str]:
    offenders = []
    for p, rel in candidate_files(root):
        if rel in allow or rel.startswith(pending):
            continue
        offenders += file_offenders(p.read_text(errors="ignore"), rel)
    return offenders


def test_no_site_or_user_values_are_hardcoded() -> None:
    offenders = scan(REPO, _ALLOW, PENDING)
    assert not offenders, (
        "Hardcoded site or user values found -- read them from the environment (docs/configuration.md: "
        "scripts/site_env.sh, scripts/cache_env.sh, scripts/cscs/account_env.sh) or allowlist with a "
        "reason in this file:\n  " + "\n  ".join(sorted(offenders))
    )


def test_every_allowlisted_or_pending_path_exists() -> None:
    """A stale entry would silently exempt whatever later reuses the name."""
    missing = [rel for rel in _ALLOW if not (REPO / rel).exists()]
    missing += [prefix for prefix in PENDING if not any(REPO.glob(prefix.rstrip("/") + "*"))]
    assert not missing, missing


def test_the_scan_catches_every_kind_of_hit(tmp_path: pathlib.Path) -> None:
    """Proves the scan is not vacuously green: a synthetic tree with each offense must be caught."""
    (tmp_path / "bad.sh").write_text(
        "#!/usr/bin/env bash\n"
        "#SBATCH --partition=gpu1\n"
        "#SBATCH -A proj\n"
        "# this comment mentions /users/ybudanaz and must NOT be flagged\n"
        'CE_EDF="/users/ybudanaz/x86_64/.edf/agent.toml"\n'
        'FAST_SCRATCH="/iopsstor/scratch/cscs/${USER}"\n'
        'SCRATCH="/ritom/scratch/cscs/$(id -un)"\n'
        "sbatch --account=a-g34 --partition=gpu1 --exclude=nid[001,002] bad.sh\n"
    )
    (tmp_path / "bad.py").write_text(
        f'"""Historical note: the {_LEGACY_NAME} rename moved these files. Not a violation."""\n'
        'REPO_DEFAULT = "/capstor/scratch/cscs/someone/hpcagent-bench"\n'
        'ACCOUNT = "a-g200"\n'
        'CONTACT = "someone@inf.ethz.ch"\n'
    )
    (tmp_path / "bad.md").write_text(f"# Notes\n\nDo not reintroduce {_LEGACY_NAME} anywhere in the docs.\n")
    (tmp_path / "site.env").write_text('FAST_SCRATCH="${FAST_SCRATCH:-/iopsstor/scratch/cscs/${USER}}"\n')
    (tmp_path / "good.sh").write_text(
        "#!/usr/bin/env bash\n"
        "#SBATCH --nodes=1\n"
        '. "${HPCAGENT_BENCH_REPO}/scripts/cscs/account_env.sh"\n'
        'FAST_SCRATCH="${FAST_SCRATCH:-${SCRATCH}}"\n'
        'sbatch ${part:+--partition="${part}"} --partition="${PARTITION}" job.sbatch\n'
        'HOST_HOME="/users/someone"\n'
    )

    offenders = scan(tmp_path)

    def hit(file: str, text: str) -> bool:
        return any(o.startswith(file) and text in o for o in offenders)

    for text in ("/users/ybudanaz", "iopsstor", "ritom", "a-g34", "--partition=gpu1", "-A proj", "nid["):
        assert hit("bad.sh", text), (text, offenders)
    for text in ("/capstor/scratch", "a-g200", "ethz.ch"):
        assert hit("bad.py", text), (text, offenders)
    assert hit("bad.md", _LEGACY_NAME), offenders
    assert hit("site.env", "iopsstor"), offenders
    # the docstring records history; the AST carve-out must exclude it
    assert not hit("bad.py", _LEGACY_NAME), offenders
    assert not hit("good.sh", ""), offenders


def test_the_storage_pattern_matches_only_real_mounts() -> None:
    """Fires on the mounts even behind a $USER/${VAR} suffix; quiet on lookalikes and on the vendor
    tree under /capstor/store, which only the site layer names."""
    pattern = _PATTERNS["literal storage mount"][0]
    positive = [
        'SCRATCH="/ritom/scratch/cscs/someone/$(uname -m)"',
        'FAST_SCRATCH="${FAST_SCRATCH:-/iopsstor/scratch/cscs/${USER}}"',
        '    echo "/capstor/scratch/cscs" >&2',
        '"/ritom:/ritom"',
    ]
    negative = [
        'FAST_SCRATCH="${FAST_SCRATCH:-${SCRATCH}}"',
        'ALT="/iopsstorbackup/old"',
        'ALT2="/capstor/scratchpad/tmp"',
        'NESTED="something/ritom/x"',
        'FIXTURE="/scratchfs/runs/1"',
    ]
    for text in positive:
        assert pattern.search(text), text
    for text in negative:
        assert not pattern.search(text), text
