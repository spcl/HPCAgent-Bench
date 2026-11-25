# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The copyright/SPDX header must sit at the top of every core ``.py`` AND survive the formatter.

Two pre-commit fixers touch the top of a file: ``check_headers.py --fix`` inserts the two-line
copyright header (after an optional shebang / coding line, so it is the first line of *content*),
and ``check_format.py --fix`` runs ``ruff format`` over the body. This test locks both guarantees so
the two fixers cannot fight -- restamping or shifting the notice on every commit:

* ``--fix`` inserts the header at the top, is idempotent (never a second copy), and honours a shebang.
* a format pass leaves those header lines BYTE-FOR-BYTE identical while it reformats the code below.
"""

import shutil
import subprocess
import types
from pathlib import Path

import pytest

from hpcagent_bench import paths
from tests.fresh_module import module_at

REPO = paths.ROOT
#: Passed explicitly rather than discovered, so the test formats the way check_format.py does.
LINE_LENGTH = "120"
HEADER: tuple = (
    "# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.",
    "# SPDX-License-Identifier: GPL-3.0-or-later",
)


def _load_check_headers() -> types.ModuleType:
    """Import ``helpers/scripts/checks/check_headers.py`` as a module (it is not an installed package)."""
    return module_at(REPO / "helpers" / "scripts" / "checks" / "check_headers.py")


@pytest.mark.parametrize(
    ("body", "above"),
    [
        pytest.param("import os\n\n\ndef g():\n    return os.getpid()\n", [], id="headerless"),
        pytest.param("#!/usr/bin/env python\nimport sys\n", ["#!/usr/bin/env python"], id="shebang"),
    ],
)
def test_fix_inserts_header_at_content_top_after_a_shebang(tmp_path: Path, body: str, above: list[str]) -> None:
    """``insert_header`` prepends the two-line header as the first lines of a headerless file; a ``#!`` shebang
    stays on line 1 and the header follows it (so an executable script still runs)."""
    ch = _load_check_headers()
    f = tmp_path / "mod.py"
    f.write_text(body, encoding="utf-8")
    assert ch.insert_header(f) is True
    assert f.read_text(encoding="utf-8").splitlines()[: len(above) + 2] == [*above, *HEADER]


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


def test_the_formatter_leaves_the_header_byte_for_byte(tmp_path: Path) -> None:
    """The OTHER pre-commit fixer must not disturb the header: ruff reformats the mangled body but
    the shebang + header lines come out identical to what went in."""
    ruff = shutil.which("ruff")
    assert ruff, "ruff must be installed (the format pre-commit hook + CI format-check both require it)"
    f = tmp_path / "mod.py"
    top = "#!/usr/bin/env python\n" + HEADER[0] + "\n" + HEADER[1] + "\n"
    f.write_text(top + "import   os,sys\ndef  f( x ):\n        return x+1\n", encoding="utf-8")
    subprocess.run([ruff, "format", "--line-length", LINE_LENGTH, str(f)], check=True)
    out: list[str] = f.read_text(encoding="utf-8").splitlines()
    assert out[0] == "#!/usr/bin/env python"
    assert out[1] == HEADER[0]
    assert out[2] == HEADER[1]
    assert "def f(x):" in f.read_text(encoding="utf-8")  # body WAS reformatted (test is meaningful)
