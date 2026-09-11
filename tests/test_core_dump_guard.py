# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Nothing this repo submits may leave core dumps in the checkout.

Beverin's ``core_pattern`` is the machine-global ``core_%h_%p``, so a crashing rank writes a
multi-GB dump into its CWD -- the checkout -- on a filesystem whose quota is inodes. One campaign
left 131 of them, 43 GB. The guard is ``ulimit -c 0`` in the script body; these tests keep it
there, including in scripts that are GENERATED rather than checked in.
"""

import importlib.util
import subprocess
import sys

from hpcagent_bench import paths

SPEC = importlib.util.spec_from_file_location("check_core_dumps", paths.ROOT / "scripts" / "check_core_dumps.py")
check_core_dumps = importlib.util.module_from_spec(SPEC)
sys.modules["check_core_dumps"] = check_core_dumps
SPEC.loader.exec_module(check_core_dumps)


def test_every_tracked_batch_script_disables_core_dumps() -> None:
    missing = [p for p in check_core_dumps.batch_scripts([]) if check_core_dumps.GUARD not in p.read_text()]
    assert not missing, f"batch scripts without `{check_core_dumps.GUARD}`: {[str(p) for p in missing]}"


def test_every_sbatch_emitter_disables_core_dumps() -> None:
    """A .py/.sh that writes an SBATCH header submits a job too, and the suffix check misses it."""
    missing = [p for p in check_core_dumps.emitters([]) if check_core_dumps.GUARD not in p.read_text()]
    assert not missing, f"sbatch emitters without `{check_core_dumps.GUARD}`: {[str(p) for p in missing]}"


def test_emitter_without_the_guard_is_reported(tmp_path) -> None:
    """The check must FAIL on a regression -- a clean repo alone does not prove it looks."""
    bad = tmp_path / "emitter.py"
    bad.write_text('TEMPLATE = """#!/bin/bash\n#SBATCH --job-name=x\nsrun true\n"""\n')
    assert check_core_dumps.emitters([str(bad)]) == [bad]

    # Through the SCRIPT, because the offender rule -- "emits a header AND lacks the guard" -- is an
    # inline comprehension in main() and is not callable on its own. ``emitters()`` answers only the
    # first half, so it still returns the guarded file; the previous assertion here sidestepped that
    # by checking ``GUARD in bad.read_text()``, which is the string the line above had just written.
    guarded = tmp_path / "guarded.py"
    guarded.write_text('TEMPLATE = """#!/bin/bash\n#SBATCH --job-name=x\nulimit -c 0\nsrun true\n"""\n')
    assert check_core_dumps.emitters([str(guarded)]) == [guarded], "it still emits a header"
    script = paths.ROOT / "scripts" / "check_core_dumps.py"
    for target, expected in ((bad, 1), (guarded, 0)):
        proc = subprocess.run([sys.executable, str(script), str(target)], capture_output=True, text=True)
        assert proc.returncode == expected, f"{target.name}: rc={proc.returncode}\n{proc.stderr}"


def test_checker_passes_over_the_whole_repo() -> None:
    script = paths.ROOT / "scripts" / "check_core_dumps.py"
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=paths.ROOT)
    assert done.returncode == 0, done.stderr
