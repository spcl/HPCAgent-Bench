# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The copyright/SPDX header must sit at the top of every core ``.py`` AND survive yapf untouched.

Two pre-commit fixers touch the top of a file: ``check_headers.py --fix`` inserts the two-line
copyright header (after an optional shebang / coding line, so it is the first line of *content*),
and ``check_format.py --fix`` runs yapf over the body. This test locks both guarantees so the two
fixers cannot fight -- restamping or shifting the notice on every commit:

* ``--fix`` inserts the header at the top, is idempotent (never a second copy), and honours a shebang.
* a yapf pass leaves those header lines BYTE-FOR-BYTE identical while it reformats the code below.
"""
import importlib.util
import shutil
import subprocess
from pathlib import Path
from typing import List

import pytest

REPO = Path(__file__).resolve().parent.parent
STYLE = REPO / ".style.yapf"
HEADER: tuple = (
    "# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.",
    "# SPDX-License-Identifier: GPL-3.0-or-later",
)


def _load_check_headers():
    """Import ``scripts/check_headers.py`` as a module (it is not an installed package)."""
    spec = importlib.util.spec_from_file_location("check_headers", REPO / "scripts" / "check_headers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fix_inserts_header_at_content_top(tmp_path: Path) -> None:
    """``insert_header`` prepends the two-line header as the first lines of a headerless file."""
    ch = _load_check_headers()
    f = tmp_path / "mod.py"
    f.write_text("import os\n\n\ndef g():\n    return os.getpid()\n", encoding="utf-8")
    assert ch.insert_header(f) is True
    lines = f.read_text(encoding="utf-8").splitlines()
    assert lines[0] == HEADER[0] and lines[1] == HEADER[1]


def test_fix_is_idempotent_and_never_double_stamps(tmp_path: Path) -> None:
    """A second ``--fix`` is a no-op: an already-headered file is left untouched (no stacked notice)."""
    ch = _load_check_headers()
    f = tmp_path / "mod.py"
    f.write_text("x = 1\n", encoding="utf-8")
    assert ch.insert_header(f) is True
    before = f.read_text(encoding="utf-8")
    assert ch.insert_header(f) is False  # already headered
    assert f.read_text(encoding="utf-8") == before
    assert f.read_text(encoding="utf-8").count(HEADER[0]) == 1


def test_fix_places_header_after_a_shebang(tmp_path: Path) -> None:
    """A ``#!`` shebang stays on line 1; the header follows it (so an executable script still runs)."""
    ch = _load_check_headers()
    f = tmp_path / "cli.py"
    f.write_text("#!/usr/bin/env python\nimport sys\n", encoding="utf-8")
    assert ch.insert_header(f) is True
    lines = f.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "#!/usr/bin/env python"
    assert lines[1] == HEADER[0] and lines[2] == HEADER[1]


def test_yapf_leaves_the_header_byte_for_byte(tmp_path: Path) -> None:
    """The OTHER pre-commit fixer must not disturb the header: yapf reformats the mangled body but
    the shebang + header lines come out identical to what went in."""
    yapf = shutil.which("yapf")
    assert yapf, "yapf must be installed (the format pre-commit hook + CI format-check both require it)"
    f = tmp_path / "mod.py"
    top = "#!/usr/bin/env python\n" + HEADER[0] + "\n" + HEADER[1] + "\n"
    f.write_text(top + "import   os,sys\ndef  f( x ):\n        return x+1\n", encoding="utf-8")
    subprocess.run([yapf, "--style", str(STYLE), "-i", str(f)], check=True)
    out: List[str] = f.read_text(encoding="utf-8").splitlines()
    assert out[0] == "#!/usr/bin/env python"
    assert out[1] == HEADER[0] and out[2] == HEADER[1]
    assert "def f(x):" in f.read_text(encoding="utf-8")  # body WAS reformatted (test is meaningful)


def test_header_hook_and_yapf_compose(tmp_path: Path) -> None:
    """End to end: insert the header into a headerless, badly-formatted file, then run yapf -- the
    header is present at the top and untouched, proving the two fixers do not fight."""
    yapf = shutil.which("yapf")
    assert yapf, "yapf must be installed"
    ch = _load_check_headers()
    f = tmp_path / "mod.py"
    f.write_text("import   os\ndef  h( ):\n    return  os.getpid( )\n", encoding="utf-8")
    assert ch.insert_header(f) is True
    subprocess.run([yapf, "--style", str(STYLE), "-i", str(f)], check=True)
    lines = f.read_text(encoding="utf-8").splitlines()
    assert lines[0] == HEADER[0] and lines[1] == HEADER[1]
