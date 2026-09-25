# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Repo-wide guard: no tracked file hardcodes a value that belongs to one site or one person.

Site values -- storage mounts, home directories, the Slurm account and partition, node and host
names, user names, the image registry, one campaign's run directories -- come from the environment,
with ONE default place each (docs/configuration.md):

* ``scripts/site_env.sh`` loads the site layer (``experiments/layers/site-<name>.env``): fast
  storage (``FAST_SCRATCH``), ``SBATCH_PARTITION``, node exclusions, vendor artefact paths;
* ``scripts/cache_env.sh`` derives every cache and work path from ``SCRATCH`` / ``FAST_SCRATCH``
  (``JIT_CACHE_ROOT``, ``HPCAGENT_BENCH_CACHE``, ``HPCAGENT_BENCH_RUNS_ROOT``);
* ``scripts/cscs/account_env.sh`` resolves the account from the user's own Slurm associations;
* ``containers/images/images.env`` names the image registry and every image;
* ``hpcagent_bench/paths.py`` is the Python side of the same roots.

A literal works once, for the person who wrote it, and then silently reads or writes the wrong
user's data (or bills the wrong project) for everyone else.

The scan covers every file ``git ls-files`` lists (tracked files only: a local scratch file is not
the release). It looks at LIVE text: for Python, non-docstring string literals (a docstring or
comment may name a site to explain it); for shell-like files (shell, sbatch, env, toml, yaml,
Dockerfile, rosters), lines with ``#`` comments stripped; everything else verbatim. ``RAW_PATTERNS``
apply to comments too: a user name, an account or an ``#SBATCH`` site directive is never right.
``${USER}``, ``$USER``, ``$(id -un)`` and the placeholder ``/users/someone`` are resolvers or
fixtures, never flagged.
"""

import ast
import pathlib
import re
import subprocess
from collections.abc import Iterable
from collections.abc import Set as AbstractSet

REPO = pathlib.Path(__file__).resolve().parents[1]

PY_EXT = ".py"
#: Formats whose comments start with ``#``: the scan strips them.
HASH_COMMENT_EXTS = (
    ".sh",
    ".sbatch",
    ".bash",
    ".toml",
    ".yaml",
    ".yml",
    ".env",
    ".def",
    ".cfg",
    ".ini",
    ".conf",
    ".txt",
    ".example",
    ".gitignore",
    ".dockerignore",
)
HASH_COMMENT_NAMES = ("Dockerfile", "Makefile", "makefile")

#: The project's pre-rename name, split so this file does not itself read as a leftover.
LEGACY_NAME = "opt" + "arena"

#: Resolvers, the placeholder fixture, and the images' own fixed agent home.
USER_RESOLVER = r"(?!\$\{USER\}|\$USER\b|\$\(id -un\)|someone\b|agent\b)"

#: The partitions a site names; a partition CONTEXT (flag, variable, key) holding one is flagged.
PARTITION_WORDS = r"(?:mi300a?|mi200|mi250x?|gh200|a100|normal|debug|amdgpu|gpu|cpu)"

# name -> (pattern, fix); applied to the live text of every scanned file.
PATTERNS = {
    "literal home directory": (
        re.compile(
            rf"(?<![\w.-])/(?:users|home)/{USER_RESOLVER}[A-Za-z][A-Za-z0-9_.-]{{1,31}}|~[a-z][a-z0-9_-]{{2,31}}/"
        ),
        "use ${HOME} (or EDF_PATH) instead of a literal home directory",
    ),
    "literal storage mount": (
        re.compile(r"(?<![\w$])/(?:capstor|iopsstor|ritom)(?:/|\b)"),
        "route through ${SCRATCH} / ${FAST_SCRATCH}; the site layer names the mount",
    ),
    "legacy project-name leftover": (
        re.compile(LEGACY_NAME, re.IGNORECASE),
        f"the project was renamed from {LEGACY_NAME} to hpcagent-bench",
    ),
    "site image registry": (
        re.compile(r"\b[\w-]+\.svc\.cscs\.ch\b|jfrog\.[\w.-]+"),
        "the registry is REGISTRY_REPO (containers/images/images.env)",
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

#: Node, host and partition literals in executable code (tests may fixture them).
CODE_PATTERNS = {
    "node name": (
        re.compile(r"\bnid(?:\d{4,6}|\[[^\]\s]*\]?)|(?<![\w-])beverin(?![\w.-])"),
        "node lists and login hosts belong in the site layer",
    ),
    "literal Slurm partition": (
        re.compile(
            r"--partition[= ](?![\"'$<{])[A-Za-z]\S*"
            rf"|(?<![\w-])-p[ \t]+{PARTITION_WORDS}\b"
            rf"|\b\w*PARTITION\b[\"']?[ \t]*[:=][ \t]*[\"']?(?:\$\{{\w+:-)?{PARTITION_WORDS}\b"
            rf"|\bpartition[\"']?[ \t]*[:=][ \t]*[\"']{PARTITION_WORDS}\b"
        ),
        "the partition is SBATCH_PARTITION (site layer); pass --partition only from a variable",
    ),
    "one campaign's run directory": (
        re.compile(
            r"hpcagent-bench-runs/(?![$<{*])[\w.-]*\d{6,}"
            r"|/[\w.-]*[-_]20[2-3]\d[01]\d[0-3]\d[a-z]?(?![\w-])"
            r"|/\d{6,7}(?=/)"
            r"|(?<![\w.-])(?:canon|smoke|wave)-\d{6,}(?![\w-])"
        ),
        "a run directory is ${HPCAGENT_BENCH_RUNS_ROOT}/<kind>/<name>-<stamp>, derived by the job",
    ),
}

#: Tracked files that legitimately carry a flagged string, one reason each. A key ``path::text``
#: exempts only the hits in ``path`` whose matched text is ``text``.
ALLOW = {
    "tests/test_no_hardcoded_user_paths.py": "this file: embeds the patterns' own text",
    "experiments/layers/site-cscs.env": "THE site layer for one real site: its values live here",
    "experiments/layers/partition-mi200.env": "names the MI250X hardware profile (docs/configuration.md)",
    "docs/configuration.md": "shows the CSCS site layer's values next to the generic ones",
    "pyproject.toml": "package author contact (PyPI metadata), not a runtime value",
    "hpcagent_bench/observations_extract.py": "reads legacy MCP server/env keys of already-recorded rows",
    "experiments/owed_wave.py": "reads legacy MCP server/env keys of already-recorded worker dirs",
    "tests/test_fused_owed_wave.py": "fixtures of legacy recorded keys",
    "tests/test_extract_llr40_task_rows.py": "fixtures of legacy recorded keys",
    "tests/test_ablation_stats.py": "fixtures of legacy recorded keys",
    "containers/inference/serve-private.sbatch::PRESET_PARTITION=mi300": (
        "MI300A serving recipe: the preset is the hardware profile, checked against its partition"
    ),
    "containers/inference/serve-private.sbatch::PRESET_PARTITION=mi200": (
        "MI200 serving recipe: the preset is the hardware profile, checked against its partition"
    ),
}


def tracked_files(root: pathlib.Path) -> list[str]:
    """Every path ``git ls-files`` lists under ``root``: the release is what git tracks."""
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True).stdout
    return [rel for rel in out.decode().split("\0") if rel]


def comment_style(rel: str, text: str) -> str:
    """``py``, ``hash`` or ``verbatim``: how ``rel``'s comments are told from its live text."""
    name = rel.rsplit("/", 1)[-1]
    first = text.split("\n", 1)[0]
    if name.endswith(PY_EXT) or (first.startswith("#!") and "python" in first):
        return "py"
    if name.endswith(HASH_COMMENT_EXTS) or name.startswith(HASH_COMMENT_NAMES) or first.startswith("#!"):
        return "hash"
    return "verbatim"


#: AST nodes that carry a leading docstring (module / class / def / async def).
DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def docstring_constant_ids(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, DOCSTRING_OWNERS):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def match(text: str, rel: str, patterns: dict, line_of: int | None = None) -> list[str]:
    offenders = []
    for label, (pattern, _fix) in patterns.items():
        for m in pattern.finditer(text):
            lineno = line_of if line_of is not None else text.count("\n", 0, m.start()) + 1
            offenders.append(f"{rel}:{lineno}: {label}: {m.group(0).strip()[:100]!r}")
    return offenders


def python_offenders(text: str, rel: str, patterns: dict) -> list[str] | None:
    """Hits in the non-docstring string literals of ``text``, or None when it does not parse."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    skip = docstring_constant_ids(tree)
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            offenders += match(node.value, rel, patterns, line_of=node.lineno)
    return offenders


def file_offenders(text: str, rel: str) -> list[str]:
    """Every hit in one file's live text."""
    style = comment_style(rel, text)
    code_rules = not rel.startswith("tests/") and style != "verbatim"
    patterns = PATTERNS | (CODE_PATTERNS if code_rules else {})
    offenders = match(text, rel, RAW_PATTERNS)
    if style == "py":
        live = python_offenders(text, rel, patterns)
        if live is not None:
            return offenders + live
    if style != "verbatim":
        text = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    return offenders + match(text, rel, patterns)


def allowed(offender: str, allow: AbstractSet[str]) -> bool:
    """Whether ``allow`` exempts ``offender`` (``rel:line: label: 'text'``) by path or by path::text."""
    rel, rest = offender.split(":", 1)
    text = ast.literal_eval(rest.split(": ", 2)[2])
    return rel in allow or f"{rel}::{text}" in allow


def scan(root: pathlib.Path, rels: Iterable[str], allow: AbstractSet[str] = frozenset()) -> list[str]:
    offenders = []
    for rel in rels:
        path = root / rel
        if rel in allow or not path.is_file() or path.is_symlink():
            continue
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue
        offenders += file_offenders(data.decode(errors="ignore"), rel)
    return [offender for offender in offenders if not allowed(offender, allow)]


def test_no_site_or_user_values_are_hardcoded() -> None:
    offenders = scan(REPO, tracked_files(REPO), ALLOW.keys())
    assert not offenders, (
        "Hardcoded site or user values found -- read them from the environment (docs/configuration.md: "
        "scripts/site_env.sh, scripts/cache_env.sh, scripts/cscs/account_env.sh, containers/images/images.env) "
        "or allowlist with a reason in this file:\n  " + "\n  ".join(sorted(offenders))
    )


def test_every_allowlisted_path_is_tracked_and_has_a_reason() -> None:
    """A stale entry would silently exempt whatever later reuses the name."""
    tracked = set(tracked_files(REPO))
    assert not [key for key in ALLOW if key.split("::", 1)[0] not in tracked], ALLOW
    assert all(reason.strip() for reason in ALLOW.values()), ALLOW
    for key in (key for key in ALLOW if "::" in key):
        rel, text = key.split("::", 1)
        assert any(allowed(hit, {key}) for hit in scan(REPO, [rel])), f"{key} no longer matches anything"


def test_only_tracked_files_are_scanned(tmp_path: pathlib.Path) -> None:
    """An untracked local file (a scratch note, a copied site.env) is not the release."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "tracked.sh").write_text('SCRATCH="/capstor/scratch/cscs/x"\n')
    (tmp_path / "untracked.sh").write_text('SCRATCH="/capstor/scratch/cscs/x"\n')
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.sh"], check=True)

    offenders = scan(tmp_path, tracked_files(tmp_path))

    assert [o.split(":", 1)[0] for o in offenders] == ["tracked.sh"], offenders


def test_the_scan_catches_every_kind_of_hit(tmp_path: pathlib.Path) -> None:
    """Proves the scan is not vacuously green: a synthetic tree with each offense must be caught."""
    files = {
        "bad.sh": (
            "#!/usr/bin/env bash\n"
            "#SBATCH --partition=gpu1\n"
            "#SBATCH -A proj\n"
            "# this comment mentions /users/ybudanaz and must NOT be flagged as a home directory\n"
            'CE_EDF="/users/ybudanaz/x86_64/.edf/agent.toml"\n'
            'FAST_SCRATCH="/iopsstor/scratch/cscs/${USER}"\n'
            'SCRATCH="/ritom/scratch/cscs/$(id -un)"\n'
            'STORE="/capstor/store/cscs/project"\n'
            'OLD_HOME="/home/alice/runs"\n'
            "sbatch --account=a-g34 --partition=gpu1 --exclude=nid[001,002] bad.sh\n"
            "srun -p debug -N 1 true\n"
            'SBATCH_PARTITION="${SBATCH_PARTITION:-normal}"\n'
            "ssh -J beverin nid002664\n"
            'RUNS="${SCRATCH}/hpcagent-bench-runs/cpf-llr-focus40-20260916/639344"\n'
            'OUT="${SCRATCH}/canon-648131"\n'
            "IMAGE=jfrog.svc.cscs.ch/hpcagent/judge:latest\n"
        ),
        "bad.py": (
            f'"""Historical note: the {LEGACY_NAME} rename moved these files. Not a violation."""\n'
            'REPO_DEFAULT = "/capstor/scratch/cscs/someone/hpcagent-bench"\n'
            'ACCOUNT = "a-g200"\n'
            'CONTACT = "someone@inf.ethz.ch"\n'
            'CMD = "srun -p mi300 true"\n'
        ),
        "bad.md": f"# Notes\n\nDo not reintroduce {LEGACY_NAME} anywhere in the docs.\n",
        "bad.tsv": "job\tdb\n1\t/capstor/scratch/cscs/x/hpcagent-bench-runs/wave/1/rank-0.db\n",
        "site.env": 'FAST_SCRATCH="${FAST_SCRATCH:-/iopsstor/scratch/cscs/${USER}}"\n',
        "good.sh": (
            "#!/usr/bin/env bash\n"
            "#SBATCH --nodes=1\n"
            '. "${HPCAGENT_BENCH_REPO}/scripts/cscs/account_env.sh"\n'
            'FAST_SCRATCH="${FAST_SCRATCH:-${SCRATCH}}"\n'
            'sbatch ${part:+--partition="${part}"} --partition="${PARTITION}" job.sbatch\n'
            'HOST_HOME="/users/someone"\n'
            'RUNS="${HPCAGENT_BENCH_RUNS_ROOT}/canon/${tag}-${stamp}"\n'
            "EDF=hpcagent-bench-agent-mi300-latest\n"
            "sbatch beverin.sbatch\n"
        ),
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text)

    offenders = scan(tmp_path, files)

    def hit(file: str, text: str) -> bool:
        return any(o.startswith(file) and text in o for o in offenders)

    for text in (
        "/users/ybudanaz",
        "iopsstor",
        "ritom",
        "/capstor/",
        "/home/alice",
        "a-g34",
        "--partition=gpu1",
        "-A proj",
        "nid[",
        "-p debug",
        "PARTITION:-normal",
        "beverin",
        "nid002664",
        "hpcagent-bench-runs/cpf-llr",
        "canon-648131",
        "jfrog.svc.cscs.ch",
    ):
        assert hit("bad.sh", text), (text, offenders)
    for text in ("/capstor/", "a-g200", "ethz.ch", "-p mi300"):
        assert hit("bad.py", text), (text, offenders)
    assert hit("bad.md", LEGACY_NAME), offenders
    assert hit("bad.tsv", "/capstor/"), offenders
    assert hit("site.env", "iopsstor"), offenders
    # the docstring records history; the AST carve-out must exclude it
    assert not hit("bad.py", LEGACY_NAME), offenders
    assert not hit("good.sh", ""), offenders


def test_the_storage_pattern_matches_only_real_mounts() -> None:
    """Fires on the mounts even behind a $USER/${VAR} suffix; quiet on lookalikes."""
    pattern = PATTERNS["literal storage mount"][0]
    positive = [
        'SCRATCH="/ritom/scratch/cscs/someone/$(uname -m)"',
        'FAST_SCRATCH="${FAST_SCRATCH:-/iopsstor/scratch/cscs/${USER}}"',
        '    echo "/capstor/scratch/cscs" >&2',
        '"/ritom:/ritom"',
        "BASE=/capstor/store/cscs/cscs/public",
    ]
    negative = [
        'FAST_SCRATCH="${FAST_SCRATCH:-${SCRATCH}}"',
        'ALT="/iopsstorbackup/old"',
        'NESTED="something/ritom/x"',
        'FIXTURE="/scratchfs/runs/1"',
    ]
    for text in positive:
        assert pattern.search(text), text
    for text in negative:
        assert not pattern.search(text), text
