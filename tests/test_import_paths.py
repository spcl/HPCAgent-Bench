# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""No tracked file edits ``sys.path`` or ``PYTHONPATH``: the package is installed (``pip install -e .``
on a host, baked into the images), and the suite takes its path from pyproject's pytest
``pythonpath``. Every other edit is either on :data:`ALLOWED` below, with the reason it has to
exist, or a failure here. Markdown is scanned too, so an instruction to export PYTHONPATH cannot come
back into the docs.
"""

import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]

#: An edit of the import path: a ``sys.path`` insert/append/extend or assignment, pytest's
#: ``syspath_prepend``, ``site.addsitedir``, or ``PYTHONPATH`` set in a shell, an env dict or a
#: keyword argument. Reading ``PYTHONPATH`` or naming it in a comment is not an edit.
EDIT = re.compile(
    r"""sys\.path\.(insert|append|extend)\("""
    r"""|sys\.path(\[[^\]]*\])?\s*=(?!=)"""
    r"""|syspath_prepend\("""
    r"""|addsitedir\("""
    r"""|\bPYTHONPATH\s*=(?!=)"""
    r"""|\[\s*["']PYTHONPATH["']\s*\]\s*=(?!=)"""
    r"""|["']PYTHONPATH["']\s*:"""
    r"""|export\s+PYTHONPATH\b"""
)

#: Repo-relative path -> why that file may edit the import path.
ALLOWED: dict[str, str] = {
    # Tests: a child process or a temp module, given its own path.
    "tests/test_dace_helper_programs.py": "temp module written under tmp_path",
    "tests/test_packaging.py": "child imports the installed wheel and nothing else",
    "tests/test_import_paths.py": "this file spells the patterns it searches for",
}


def tracked_code() -> list[pathlib.Path]:
    """Every tracked regular file (symlinks are scanned at their target)."""
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True)
    names = [name for name in listed.stdout.split("\0") if name]
    return [REPO / name for name in names if (REPO / name).is_file() and not (REPO / name).is_symlink()]


def edits(path: pathlib.Path) -> list[int]:
    """Line numbers of ``path`` that edit the import path; comment lines do not count."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    return [
        number
        for number, line in enumerate(text.splitlines(), 1)
        if not line.lstrip().startswith("#") and EDIT.search(line)
    ]


def test_no_file_edits_the_import_path_outside_the_allowlist() -> None:
    offenders = []
    for path in tracked_code():
        rel = path.relative_to(REPO).as_posix()
        if rel not in ALLOWED:
            offenders.extend(f"{rel}:{number}" for number in edits(path))
    assert not offenders, (
        "import-path edits outside tests/test_import_paths.py's ALLOWED (install the package, or rely "
        "on pytest's pythonpath instead):\n" + "\n".join(offenders)
    )


def test_every_allowlisted_file_still_edits_the_import_path() -> None:
    """An entry whose file no longer edits the path (or no longer exists) is removed, not kept."""
    stale = [rel for rel in ALLOWED if not (REPO / rel).is_file() or not edits(REPO / rel)]
    assert not stale, f"ALLOWED entries with no import-path edit left: {stale}"


def test_the_pattern_tells_an_edit_from_a_read() -> None:
    for edit in (
        'sys.path.insert(0, "x")',
        "sys.path[:] = saved",
        'export PYTHONPATH="${x}"',
        'env["PYTHONPATH"] = x',
        'env = {"PYTHONPATH": x}',
        'env.update(PYTHONPATH="x")',
        "monkeypatch.syspath_prepend(d)",
    ):
        assert EDIT.search(edit), edit
    for read in ('env.get("PYTHONPATH", "")', 'value = env["PYTHONPATH"]', "if x == sys.path:", '"PYTHONPATH",'):
        assert not EDIT.search(read), read
