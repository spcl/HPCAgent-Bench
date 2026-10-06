# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Repo-wide rules over tracked files, as one pre-commit hook.

Each rule maps the files to check (the ones pre-commit passes; every tracked file with no
arguments or under ``--all-files``, which is what CI runs) to offender lines. Every allowlist
entry carries its reason and must still name a tracked file:

* ``hash-seed``: PYTHONHASHSEED=0 is set in exactly two places, the job environment
  (hpcagent_bench/cluster/env.sh) and CI's workflow env, because dace hashes iteration order into
  generated code.
* ``import-path``: no file edits ``sys.path`` or ``PYTHONPATH``; the packages are installed.
* ``enroot``: no file names enroot; the one container seam is the CE mounting a podman-built
  squashfs.
* ``no-requeue``: every submittable batch script carries ``#SBATCH --no-requeue``: a NODE_FAIL
  requeue reruns the job id into the same run directory and stacks duplicate rows.
* ``scratch``: ``.scratch/`` tracks only its ``.gitkeep``.
* ``gitignore``: no tracked file matches an ignore rule (a hand override at a generated name gets a
  ``!`` line), and every such ``!`` line names a tracked file.
* ``gfx-literal``: AMD archs come from containers/images/gpu_arch.env, never spelled in builds,
  launchers, scripts or the harness.
* ``opt-flags``: ``-O3`` / ``-march=native`` / ``-ffast-math`` come from hpcagent_bench/flags.py.
* ``site-values``: no site or user value (home directory, storage mount, account, partition,
  node, registry, run directory) is hardcoded; they come from the environment
  (docs/configuration.md).

Python files are read as live text: string literals that are not docstrings, never comments.

    python scripts/checks/check_repo_rules.py [FILE ...]
"""

import ast
import functools
import pathlib
import re
import subprocess
import sys
from collections.abc import Callable, Iterable

REPO = pathlib.Path(__file__).resolve().parents[2]
SELF = "scripts/checks/check_repo_rules.py"


@functools.cache
def tracked() -> tuple[str, ...]:
    """Every tracked path; a git failure raises, so a rule never passes on an empty listing."""
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"], capture_output=True, check=True).stdout
    return tuple(rel for rel in out.decode().split("\0") if rel)


@functools.cache
def text_of(rel: str) -> str | None:
    """A regular tracked file's text, or ``None`` for a symlink, a directory or binary content."""
    path = REPO / rel
    if path.is_symlink() or not path.is_file():
        return None
    data = path.read_bytes()
    return None if b"\0" in data[:8192] else data.decode("utf-8", errors="replace")


#: AST nodes that carry a leading docstring.
DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def docstring_constant_ids(tree: ast.AST) -> set[int]:
    """``id()`` of the string constants that are docstrings."""
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, DOCSTRING_OWNERS) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                ids.add(id(first.value))
    return ids


def live_strings(text: str) -> list[ast.Constant] | None:
    """The non-docstring string constants of Python ``text``, or ``None`` when it does not parse."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    skip = docstring_constant_ids(tree)
    return [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip]


def line_hits(rel: str, text: str, pattern: re.Pattern[str], skip_comments: bool = False) -> list[str]:
    return [
        f"{rel}:{number}"
        for number, line in enumerate(text.splitlines(), 1)
        if not (skip_comments and line.lstrip().startswith("#")) and pattern.search(line)
    ]


def stale(allow: Iterable[str], rule: str) -> list[str]:
    return [f"{rel}: allowlisted for {rule} but untracked" for rel in allow if rel not in set(tracked())]


# hash-seed

HASH_SEED = re.compile(r"PYTHONHASHSEED\s*[=:]\s*['\"]?0|[\"']PYTHONHASHSEED[\"']\s*:")
HASH_SEED_HOMES = {"hpcagent_bench/cluster/env.sh", ".github/workflows/tests.yml"}


def hash_seed(files: list[str]) -> list[str]:
    out = []
    for rel in files:
        if rel in HASH_SEED_HOMES or rel == SELF or rel.startswith("tests/") or rel.endswith(".md"):
            continue
        text = text_of(rel)
        if text is not None and HASH_SEED.search(text):
            out.append(f"{rel}: sets PYTHONHASHSEED; only {sorted(HASH_SEED_HOMES)} may")
    missing = [rel for rel in HASH_SEED_HOMES if not HASH_SEED.search(text_of(rel) or "")]
    return out + [f"{rel}: no longer sets PYTHONHASHSEED=0" for rel in missing]


# import-path

#: A ``sys.path`` insert/append/extend or assignment, pytest's ``syspath_prepend``,
#: ``site.addsitedir``, or ``PYTHONPATH`` set in a shell, an env dict or a keyword. Reading it is not an edit.
IMPORT_PATH_EDIT = re.compile(
    r"""sys\.path\.(insert|append|extend)\("""
    r"""|sys\.path(\[[^\]]*\])?\s*=(?!=)"""
    r"""|syspath_prepend\("""
    r"""|addsitedir\("""
    r"""|\bPYTHONPATH\s*=(?!=)"""
    r"""|\[\s*["']PYTHONPATH["']\s*\]\s*=(?!=)"""
    r"""|["']PYTHONPATH["']\s*:"""
    r"""|export\s+PYTHONPATH\b"""
)


IMPORT_PATH_ALLOW = {"tests/test_check_repo_rules.py": "spells the edits the pattern must catch"}


def import_path(files: list[str]) -> list[str]:
    out = stale(IMPORT_PATH_ALLOW, "import-path")
    for rel in files:
        text = text_of(rel)
        if rel != SELF and rel not in IMPORT_PATH_ALLOW and text is not None:
            out += [
                f"{hit}: edits the import path" for hit in line_hits(rel, text, IMPORT_PATH_EDIT, skip_comments=True)
            ]
    return out


# enroot

ENROOT = re.compile(r"enroot", re.IGNORECASE)
ENROOT_ALLOW = {"tests/test_container_factory.py": "asserts that enroot is not a backend the factory resolves"}


def enroot(files: list[str]) -> list[str]:
    out = stale(ENROOT_ALLOW, "enroot")
    for rel in files:
        text = text_of(rel)
        if rel != SELF and rel not in ENROOT_ALLOW and text is not None:
            out += [f"{hit}: names enroot" for hit in line_hits(rel, text, ENROOT)]
    return out + [
        f"{rel}: allowlisted but names no enroot" for rel in ENROOT_ALLOW if not ENROOT.search(text_of(rel) or "")
    ]


# no-requeue

REQUEUE_GUARD = "#SBATCH --no-requeue"
#: A real directive line (a flag right after the keyword), not prose about one.
SBATCH_DIRECTIVE = re.compile(r"^\s*#SBATCH\s+--", re.M)
SHELL_SHEBANG = re.compile(r"^#!.*\b(?:ba|da|k|z|a)?sh\b")


def no_requeue(files: list[str]) -> list[str]:
    out = []
    for rel in files:
        text = text_of(rel)
        if text is None:
            continue
        entry = rel.endswith(".sbatch") or (SHELL_SHEBANG.match(text) and SBATCH_DIRECTIVE.search(text))
        if entry and REQUEUE_GUARD not in text:
            out.append(f"{rel}: batch script without `{REQUEUE_GUARD}`")
    return out


# scratch, gitignore


def scratch(files: list[str]) -> list[str]:
    extra = [
        f"{rel}: tracked under .scratch/" for rel in files if rel.startswith(".scratch/") and rel != ".scratch/.gitkeep"
    ]
    return extra + ([] if ".scratch/.gitkeep" in tracked() else [".scratch/.gitkeep: untracked"])


def git_lines(*args: str) -> list[str]:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.split()


def gitignore(files: list[str]) -> list[str]:
    out = [
        f"{rel}: tracked but matches an ignore rule" for rel in git_lines("ls-files", "-i", "-c", "--exclude-standard")
    ]
    lines = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    overrides = [line.removeprefix("!/") for line in lines if line.startswith("!/hpcagent_bench/benchmarks/")]
    tracked_set = set(tracked())
    return out + [f".gitignore: `!/{rel}` re-admits an untracked name" for rel in overrides if rel not in tracked_set]


# gfx-literal

GFX_LITERAL = re.compile(r"\bgfx[0-9a-f]{3,4}\b")
REASON_SETUP_ROCM = "verbatim upstream setup_rocm.py text the sglang recipe's asserted multi-arch edit matches"
#: Non-comment gfx literals that must stay, keyed by (file, stripped line), with the reason.
GFX_EXCEPTIONS: dict[tuple[str, str], str] = {
    (
        "containers/images/sglang/Dockerfile",
        '(\'if amdgpu_target not in ["gfx942", "gfx950", "gfx1250"]:\\n\',',
    ): REASON_SETUP_ROCM,
    (
        "containers/images/sglang/Dockerfile",
        '\'if not set(amdgpu_target.split(";")) <= {"gfx90a", "gfx942", "gfx950", "gfx1250"}:\\n\'),',
    ): REASON_SETUP_ROCM,
    (
        "containers/images/sglang/Dockerfile",
        '(\'if amdgpu_target == "gfx942" else "-DHIP_FP8_TYPE_E4M3"\',',
    ): REASON_SETUP_ROCM,
    (
        "containers/images/sglang/Dockerfile",
        '\'if "gfx942" in amdgpu_target.split(";") else "-DHIP_FP8_TYPE_E4M3"\'),',
    ): REASON_SETUP_ROCM,
    (
        "containers/images/sglang/Dockerfile",
        '(\'48 * 1024 if amdgpu_target == "gfx942" else\', \'48 * 1024 if "gfx942" in amdgpu_target.split(";") else\'),',
    ): REASON_SETUP_ROCM,
}


def in_gfx_scope(rel: str) -> bool:
    """The files where a spelled-out arch would be a second source of truth."""
    name = rel.rsplit("/", 1)[-1]
    parts = rel.split("/")
    if rel == "containers/images/gpu_arch.env" or name.endswith(".md") or {"skills", "tests"} & set(parts):
        return False
    if (
        name in ("Dockerfile", "build.sh", "image.sh")
        or name.endswith(".sbatch")
        or re.fullmatch(r"edf.*\.toml\.example", name)
    ):
        return True
    if parts[0] == "hpcagent_bench":
        return name.endswith(".py")
    return parts[0] in ("experiments", "scripts") and rel != SELF


def gfx_literal(files: list[str]) -> list[str]:
    out, scanned = [], 0
    for rel in files:
        text = text_of(rel) if in_gfx_scope(rel) else None
        if text is None:
            continue
        scanned += 1
        if not GFX_LITERAL.search(text):
            continue
        if rel.endswith(".py"):
            pieces = [node.value for node in live_strings(text) or []]
        else:
            pieces = [re.sub(r"(^|\s)#.*$", "", line).strip() for line in text.splitlines()]
        out += [
            f"{rel}: {piece}" for piece in pieces if GFX_LITERAL.search(piece) and (rel, piece) not in GFX_EXCEPTIONS
        ]
    vacuous = len(files) == len(tracked()) and scanned <= 100
    return out + ([f"only {scanned} files in scope: the scan checks nothing"] if vacuous else [])


# opt-flags

OPT_FLAG = re.compile(r"-O3|-march=native|-ffast-math")
OPT_FLAG_ALLOW = {
    "hpcagent_bench/flags.py": "the matrix itself",
    "hpcagent_bench/harbor.py": "agent-facing prose naming the flags the harness applies, not a build command",
    "hpcagent_bench/benchmarks/scientific_computing/n_body_methods/gromacs/nbnxm/tests/test_gromacs_nbnxm.py": (
        "builds a reference-C correctness oracle, not the graded matrix"
    ),
    "hpcagent_bench/benchmarks/scientific_computing/n_body_methods/lavamd/tests/test_lavamd.py": (
        "builds a reference-C correctness oracle, not the graded matrix"
    ),
    "hpcagent_bench/benchmarks/scientific_computing/map_reduce/xsbench/tests/test_xsbench.py": (
        "builds a reference-C correctness oracle, not the graded matrix"
    ),
}


def opt_flags(files: list[str]) -> list[str]:
    out = stale(OPT_FLAG_ALLOW, "opt-flags")
    for rel in files:
        in_scope = rel.endswith(".py") and rel.split("/")[0] in ("hpcagent_bench", "scripts")
        if not in_scope or rel == SELF or rel in OPT_FLAG_ALLOW:
            continue
        text = text_of(rel)
        if text is None:
            continue
        strings = live_strings(text)
        if strings is None:
            out += [f"{hit}: literal optimization flag" for hit in line_hits(rel, text.split("#")[0], OPT_FLAG)]
            continue
        out += [f"{rel}:{node.lineno}: literal optimization flag" for node in strings if OPT_FLAG.search(node.value)]
    return out


# site-values

HASH_COMMENT_EXTS = (
    ".sh", ".sbatch", ".bash", ".toml", ".yaml", ".yml", ".env", ".def", ".cfg", ".ini", ".conf", ".txt",
    ".example", ".gitignore", ".dockerignore",
)  # fmt: skip
HASH_COMMENT_NAMES = ("Dockerfile", "Makefile", "makefile")
#: The project's pre-rename name, split so this file does not itself read as one.
OLD_NAME = "opt" + "arena"
#: Resolvers, the placeholder fixture, and the images' own fixed agent home.
USER_RESOLVER = r"(?!\$\{USER\}|\$USER\b|\$\(id -un\)|someone\b|agent\b)"
#: The partitions a site names; a partition CONTEXT (flag, variable, key) holding one is flagged.
PARTITION_WORDS = r"(?:mi300a?|mi200|mi250x?|gh200|a100|normal|debug|amdgpu|gpu|cpu)"

Patterns = dict[str, re.Pattern[str]]

#: Applied to the live text of every file.
SITE_PATTERNS: Patterns = {
    "literal home directory (use ${HOME} or EDF_PATH)": re.compile(
        rf"(?<![\w.-])/(?:users|home)/{USER_RESOLVER}[A-Za-z][A-Za-z0-9_.-]{{1,31}}|~[a-z][a-z0-9_-]{{2,31}}/"
    ),
    "literal storage mount (route through ${SCRATCH} / ${FAST_SCRATCH})": re.compile(
        r"(?<![\w$])/(?:capstor|iopsstor|ritom)(?:/|\b)"
    ),
    f"the pre-rename project name ({OLD_NAME} is hpcagent-bench)": re.compile(OLD_NAME, re.IGNORECASE),
    "site image registry (REGISTRY_REPO in containers/images/images.env)": re.compile(
        r"\b[\w-]+\.svc\.cscs\.ch\b|jfrog\.[\w.-]+"
    ),
}
#: Applied everywhere, comments and docstrings included: Slurm executes ``#SBATCH`` lines.
RAW_PATTERNS: Patterns = {
    "user name (use ${USER} / $(id -un))": re.compile(r"\bybudanaz\b"),
    "site email": re.compile(r"[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)*(?:ethz|cscs)\.ch\b"),
    "hardcoded Slurm account (SBATCH_ACCOUNT)": re.compile(r"(?<![\w-])(?:a-g34|a-g200|g34|g200)(?![\w-])"),
    "site value in an #SBATCH directive (SBATCH_PARTITION / SBATCH_ACCOUNT)": re.compile(
        r"(?m)^[ \t]*#SBATCH[ \t]+(?:--(?:partition|account|reservation|nodelist|exclude)=\S+|-[Apw][ \t]+\S+)"
    ),
}
#: Node, host and partition literals in executable code outside tests.
CODE_PATTERNS: Patterns = {
    "node name (the site layer)": re.compile(r"\bnid(?:\d{4,6}|\[[^\]\s]*\]?)|(?<![\w-])beverin(?![\w.-])"),
    "literal Slurm partition (SBATCH_PARTITION)": re.compile(
        r"--partition[= ](?![\"'$<{])[A-Za-z]\S*"
        rf"|(?<![\w-])-p[ \t]+{PARTITION_WORDS}\b"
        rf"|\b\w*PARTITION\b[\"']?[ \t]*[:=][ \t]*[\"']?(?:\$\{{\w+:-)?{PARTITION_WORDS}\b"
        rf"|\bpartition[\"']?[ \t]*[:=][ \t]*[\"']{PARTITION_WORDS}\b"
    ),
    "one experiment's run directory (${HPCAGENT_BENCH_RUNS_ROOT}/<kind>/<name>-<stamp>)": re.compile(
        r"hpcagent-bench-runs/(?![$<{*])[\w.-]*\d{6,}"
        r"|/[\w.-]*[-_]20[2-3]\d[01]\d[0-3]\d[a-z]?(?![\w-])"
        r"|/\d{6,7}(?=/)"
        r"|(?<![\w.-])(?:canon|smoke|wave)-\d{6,}(?![\w-])"
    ),
}
#: Files (or ``path::matched text``) that legitimately carry a flagged string, one reason each.
SITE_ALLOW = {
    "experiments/layers/site-cscs.env": "THE site layer for one real site: its values live here",
    "experiments/layers/hardware-mi200.env": "names the MI250X hardware (docs/configuration.md)",
    "hpcagent_bench/cluster/systems.yaml": "the job shape of each named system (docs/configuration.md)",
    "docs/configuration.md": "shows the CSCS site layer's values next to the generic ones",
    "pyproject.toml": "package author contact (PyPI metadata), not a runtime value",
    "agent/pyproject.toml": "package author contact (metadata), not a runtime value",
    "hpcagent_bench/observations_extract.py": "reads the recorded MCP server/env keys of existing rows",
    "tests/test_extract_llr40_episode_rows.py": "fixtures of recorded keys",
    "tests/test_iteration_counts.py": "fixtures of recorded keys",
    "tests/test_harbor_images.py": "asserts a generated Harbor task names no storage mount",
    "tests/test_check_repo_rules.py": "synthetic offenders for this check",
    "containers/inference/serve-private.sbatch::PRESET_PARTITION=mi300": (
        "MI300A serving recipe: the preset is the hardware, checked against its partition"
    ),
    "containers/inference/serve-private.sbatch::PRESET_PARTITION=mi200": (
        "MI200 serving recipe: the preset is the hardware, checked against its partition"
    ),
}


def comment_style(rel: str, text: str) -> str:
    """``py``, ``hash`` or ``verbatim``: how ``rel``'s comments are told from its live text."""
    name = rel.rsplit("/", 1)[-1]
    first = text.split("\n", 1)[0]
    if name.endswith(".py") or (first.startswith("#!") and "python" in first):
        return "py"
    if name.endswith(HASH_COMMENT_EXTS) or name.startswith(HASH_COMMENT_NAMES) or first.startswith("#!"):
        return "hash"
    return "verbatim"


def matches(text: str, rel: str, patterns: Patterns, line_of: int | None = None) -> list[tuple[str, str]]:
    """``(location: label, matched text)`` for every hit of ``patterns`` in ``text``."""
    out = []
    for label, pattern in patterns.items():
        for m in pattern.finditer(text):
            line = line_of if line_of is not None else text.count("\n", 0, m.start()) + 1
            out.append((f"{rel}:{line}: {label}", m.group(0).strip()[:100]))
    return out


def site_hits(rel: str, text: str) -> list[tuple[str, str]]:
    """Every site-value hit in one file's live text."""
    style = comment_style(rel, text)
    patterns = SITE_PATTERNS | (CODE_PATTERNS if not rel.startswith("tests/") and style != "verbatim" else {})
    hits = matches(text, rel, RAW_PATTERNS)
    if style == "py":
        strings = live_strings(text)
        if strings is not None:
            return hits + [hit for node in strings for hit in matches(node.value, rel, patterns, node.lineno)]
    if style != "verbatim":
        text = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    return hits + matches(text, rel, patterns)


def site_values(files: list[str], allow: dict[str, str] = SITE_ALLOW) -> list[str]:
    paths = {key.split("::", 1)[0] for key in allow}
    out = [f"{rel}: allowlisted for site-values but untracked" for rel in paths if rel not in set(tracked())]
    used: set[str] = set()
    for rel in files:
        text = None if rel == SELF or rel in allow else text_of(rel)
        for where, matched in site_hits(rel, text) if text is not None else []:
            if f"{rel}::{matched}" in allow:
                used.add(f"{rel}::{matched}")
            else:
                out.append(f"{where}: {matched!r}")
    scanned = set(files)
    unused = [key for key in allow if "::" in key and key.split("::", 1)[0] in scanned and key not in used]
    return out + [f"{key}: allowlisted but matches nothing" for key in unused]


RULES: dict[str, Callable[[list[str]], list[str]]] = {
    "hash-seed": hash_seed,
    "import-path": import_path,
    "enroot": enroot,
    "no-requeue": no_requeue,
    "scratch": scratch,
    "gitignore": gitignore,
    "gfx-literal": gfx_literal,
    "opt-flags": opt_flags,
    "site-values": site_values,
}


def main(argv: list[str]) -> int:
    files = [rel for rel in tracked() if not argv or rel in set(argv)]
    failed = 0
    for name, rule in RULES.items():
        for offender in rule(files):
            print(f"[{name}] {offender}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
