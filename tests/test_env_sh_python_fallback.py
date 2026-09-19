# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""experiments/env.sh must not hand a dead interpreter to its callers.

Inside the judge container the host venv's python links into /users, which the EDF does not
mount; every test job that sourced env.sh then died with rc 127 before running a single test.
"""

import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
ENV_SH = REPO / "experiments" / "env.sh"


def resolved_py(venv: pathlib.Path) -> str:
    script = f'VENV="{venv}"; unset PY; . "{ENV_SH}"; printf "%s" "$PY"'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    return out.stdout


def test_a_dead_venv_interpreter_falls_back_to_python3(tmp_path: pathlib.Path) -> None:
    """A venv whose python is a dangling link resolves PY to the python3 on PATH."""
    bindir = tmp_path / "venv" / "bin"
    bindir.mkdir(parents=True)
    (bindir / "python").symlink_to(tmp_path / "missing" / "python3")
    python3 = subprocess.run(["bash", "-c", "command -v python3"], capture_output=True, text=True, check=True)
    assert resolved_py(tmp_path / "venv") == python3.stdout.strip()


def test_a_live_venv_interpreter_is_kept(tmp_path: pathlib.Path) -> None:
    """A venv whose python exists stays the interpreter, byte for byte."""
    bindir = tmp_path / "venv" / "bin"
    bindir.mkdir(parents=True)
    python = bindir / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    assert resolved_py(tmp_path / "venv") == str(python)
