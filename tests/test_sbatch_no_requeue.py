# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""CI lint: every Slurm batch script this repo submits carries ``#SBATCH --no-requeue``.

The standing rule is jobs never self-resubmit or requeue: a NODE_FAIL auto-requeue reran job
637040 into the SAME run directory under the same id and stacked duplicate rows on top of the
partial ones the failed attempt had already written. ``#SBATCH --no-requeue`` is the one-line fix,
and commit 32b0e3d6f swept it into every ``*.sbatch`` file that lacked it. That sweep keyed on the
``.sbatch`` suffix alone and missed two more surfaces the same policy has to cover:

1. A tracked file that is itself a submittable batch script but does not carry the ``.sbatch``
   suffix -- ``experiments/smoke_library_requests.sh``, whose own header comment documents
   submitting it directly with ``sbatch experiments/smoke_library_requests.sh``. Any tracked file
   that embeds a literal ``#SBATCH`` directive line is this kind of entry point, suffix aside.
2. A script that EMITS an sbatch header from a template rather than shipping one as a file --
   ``scripts/preset_sweep.py --emit-sbatch`` builds a submittable script from an f-string
   (``render_sbatch``). The guard has to be inside the EMITTED text, so a check that only greps
   the generator's own source (which never contains the literal line, just the code that writes
   it) would not see a regression here; this loads the module and calls ``render_sbatch`` the same
   way ``tests/test_preset_sweep.py`` does, and inspects what it actually returns.

Scope: ``git ls-files`` (the same enumeration the review that raised this asked for) never
descends into ``third_party/KernelBench`` -- a git submodule recorded as a single gitlink entry,
not individual files -- so the vendored tree is excluded without a special case.
"""

import importlib.util
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

#: The exact directive line the policy requires, verbatim as commit 32b0e3d6f wrote it everywhere.
GUARD = "#SBATCH --no-requeue"

#: A real directive line, not prose ABOUT one: ``#SBATCH directives without an account`` (a comment
#: in tests/test_materialize_shared.py) and a doc's inline ``#SBATCH --partition=...`` code snippet
#: (docs/serving/README.md, never itself submitted) both contain the substring "#SBATCH" but are
#: not entry points. Requiring "--" right after the keyword is what a real flag always has and
#: prose almost never does.
DIRECTIVE = re.compile(r"^#SBATCH\s+--")

#: A shebang naming a shell -- narrows the whole-tree scan to files that are actually RUN, the same
#: signal ``scripts/check_core_dumps.py`` (``SHELL_SHEBANG``) uses for the identical question.
SHEBANG = re.compile(rb"^#!.*\b(?:ba|da|k|z|a)?sh\b")

PRESET_SWEEP = REPO / "scripts" / "preset_sweep.py"


def tracked(*globs: str) -> list[pathlib.Path]:
    """Tracked repo paths matching ``globs`` (all tracked files with none given), as absolute
    paths -- mirrors ``scripts/check_core_dumps.py``'s helper of the same name and purpose."""
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", *globs], capture_output=True, text=True, check=True
    ).stdout.split()
    return [REPO / name for name in out]


def sbatch_entry_points() -> list[pathlib.Path]:
    """Every tracked file that is itself a submittable batch script: any ``*.sbatch`` file, plus
    any other tracked, shell-shebanged file that embeds a real ``#SBATCH`` directive line."""
    found: dict[pathlib.Path, None] = {}
    for path in tracked("*.sbatch"):
        found[path] = None
    for path in tracked():
        if path in found or not path.is_file():
            continue
        with path.open("rb") as handle:
            first_line = handle.readline(256)
        if not SHEBANG.match(first_line):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(DIRECTIVE.match(line.lstrip()) for line in text.splitlines()):
            found[path] = None
    return list(found)


def load_preset_sweep():
    """Load ``scripts/preset_sweep.py`` as a module (it is a script, not a package member) --
    same loader ``tests/test_preset_sweep.py`` uses, kept local so this file does not depend on
    another test module's internals."""
    spec = importlib.util.spec_from_file_location("preset_sweep_no_requeue_check", PRESET_SWEEP)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_sbatch_entry_point_never_requeues() -> None:
    """A tracked ``.sbatch`` file, or any other tracked file carrying an ``#SBATCH`` header,
    always disables Slurm's auto-requeue."""
    missing = [p for p in sbatch_entry_points() if GUARD not in p.read_text(encoding="utf-8", errors="ignore")]
    assert not missing, (
        f"sbatch entry point(s) without `{GUARD}` (a NODE_FAIL requeue reruns the job id into the "
        f"same RUN_DIR and stacks rows -- see job 637040): "
        f"{sorted(str(p.relative_to(REPO)) for p in missing)}"
    )


def test_preset_sweep_emitted_header_never_requeues() -> None:
    """``scripts/preset_sweep.py --emit-sbatch`` writes a submittable header from an f-string; the
    guard has to be inside that EMITTED text, which a source-file grep would never see."""
    module = load_preset_sweep()
    emitted = module.render_sbatch("a", framework="numpy", presets=["L"], single_core_presets=(), repeat=1)
    assert GUARD in emitted, f"scripts/preset_sweep.py render_sbatch() omits `{GUARD}` from its emitted header"
