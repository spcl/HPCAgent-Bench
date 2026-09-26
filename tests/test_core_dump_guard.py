# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Nothing this repo runs may leave core dumps in the checkout.

Beverin's ``core_pattern`` is the machine-global ``core_%h_%p``, so a crashing process writes a
multi-GB dump into its CWD -- the checkout -- on a filesystem whose quota is inodes. One campaign
left 131 of them, 43 GB; a login-node CPF repro left 21.6 GB in two files on 2026-09-20. The guard
is ``ulimit -c 0`` in every shell entry point, plus :func:`hpcagent_bench.core_dumps.disable` for a
python process started by a script outside the repo. These tests keep both there, including in
scripts that are GENERATED rather than checked in, and prove the check FAILS on a regression.
"""

import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

from hpcagent_bench import core_dumps, paths

SPEC = importlib.util.spec_from_file_location(
    "check_core_dumps", paths.ROOT / "scripts" / "checks" / "check_core_dumps.py"
)
check_core_dumps = importlib.util.module_from_spec(SPEC)
sys.modules["check_core_dumps"] = check_core_dumps
SPEC.loader.exec_module(check_core_dumps)

CHECKER = paths.ROOT / "scripts" / "checks" / "check_core_dumps.py"


def checker_rc(target: pathlib.Path) -> int:
    """The check's verdict on one file, through the SCRIPT -- main() owns the offender rule."""
    done = subprocess.run([sys.executable, str(CHECKER), str(target)], capture_output=True, text=True, check=False)
    return done.returncode


def test_every_tracked_shell_script_disables_core_dumps() -> None:
    """Every .sbatch, .sh and shell shebang: wrappers, login-node helpers and the agent's own tool."""
    missing = [p for p in check_core_dumps.shell_scripts([]) if not check_core_dumps.guarded(p.read_text())]
    assert not missing, f"shell scripts without `{check_core_dumps.GUARD}`: {[str(p) for p in missing]}"


def test_every_sbatch_emitter_disables_core_dumps() -> None:
    """A .py/.sh that writes an SBATCH header submits a job too, and the suffix check misses it."""
    missing = [p for p in check_core_dumps.emitters([]) if not check_core_dumps.guarded(p.read_text())]
    assert not missing, f"sbatch emitters without `{check_core_dumps.GUARD}`: {[str(p) for p in missing]}"


def test_unguarded_shell_script_is_reported(tmp_path: pathlib.Path) -> None:
    """A deliberately unguarded .sh fails the check, and the same file passes once guarded.

    The repo being clean proves only that nothing is wrong today, not that the check looks -- and
    the scope it looks at is exactly what was widened, so a .sh (not a .sbatch) is the case to pin.
    """
    bad = tmp_path / "launcher.sh"
    bad.write_text("#!/usr/bin/env bash\nset -euo pipefail\nsrun ./kernel\n")
    assert checker_rc(bad) == 1, "an unguarded .sh must fail the check"

    good = tmp_path / "guarded.sh"
    good.write_text("#!/usr/bin/env bash\nset -euo pipefail\nulimit -c 0\nsrun ./kernel\n")
    assert checker_rc(good) == 0

    # No suffix at all: hpcagent-bench-tool and bash-norc are both `#!/bin/sh` one-liners, and the
    # suffix rule that replaced the .sbatch rule still walked straight past them.
    extensionless = tmp_path / "agent-tool"
    extensionless.write_text('#!/bin/sh\nexec python3 "$0.py" "$@"\n')
    assert checker_rc(extensionless) == 1, "a shell shebang is an entry point whatever it is called"


def test_script_that_re_enables_core_dumps_is_reported(tmp_path: pathlib.Path) -> None:
    """The guard STRING is not proof: a later `ulimit -c unlimited` turns dumps back on."""
    header = "#!/bin/bash\n#SBATCH --job-name=x\nulimit -c 0\n"
    rearmed = tmp_path / "rearmed.sbatch"
    rearmed.write_text(header + "srun bash -c 'ulimit -c unlimited; ./kernel'\n")
    assert checker_rc(rearmed) == 1, "a re-enabling script must fail even with the guard present"

    marked = tmp_path / "marked.sbatch"
    marked.write_text(header + "srun bash -c '\n  ulimit -c unlimited  # core-dumps-ok: gdb reads it\n  ./kernel\n'\n")
    assert checker_rc(marked) == 0, "a re-enable that says why passes"


def test_fix_does_not_splice_into_a_quoted_inner_shell(tmp_path: pathlib.Path) -> None:
    """--fix inserts after the header, not at the LAST `set -` line -- which sat inside a string."""
    target = tmp_path / "inner.sbatch"
    target.write_text("#!/bin/bash\n#SBATCH --job-name=x\nset -uo pipefail\nsrun bash -c '\n  set -x\n  ./kernel\n'\n")
    subprocess.run([sys.executable, str(CHECKER), "--fix", str(target)], capture_output=True, text=True, check=False)
    lines = target.read_text().splitlines()
    assert lines[lines.index("ulimit -c 0") - 1].startswith("#"), "guard landed outside the leading block"
    assert lines.index("ulimit -c 0") < lines.index("srun bash -c '"), "guard must precede the work"
    assert checker_rc(target) == 0


def test_checker_passes_over_the_whole_repo() -> None:
    done = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True, cwd=paths.ROOT, check=False)
    assert done.returncode == 0, done.stderr


def test_nothing_samples_stacks_with_the_faulthandler_watchdog() -> None:
    """``faulthandler.dump_traceback_later`` is the one API that reliably CAUSES the core dump.

    Its watchdog is a C thread that walks every other thread's ``_PyInterpreterFrame`` chain with
    no GIL and no synchronisation. Against an interpreter churning frames -- a dace parse, a sympy
    rewrite -- it dereferences a frame the main thread has already popped and the process dies in
    ``dump_frame``. Measured 2026-09-20: a 20-line recursion loop plus a 10 ms sampler segfaults in
    seconds on 3.12.3 and 3.14.7, and the same sampler killed a ``warpx_field_gather``
    canonicalize twice for 21.6 GB of core files. The SAFE sampler is
    ``faulthandler.register(signal.SIGUSR1)`` plus an external ``kill -USR1``: that dumps
    synchronously on the signalled thread. The same run, sampled that way, never crashed.
    """
    needle = "dump_traceback_later"
    here = pathlib.Path(__file__).resolve()
    hits = [
        path
        for path in check_core_dumps.tracked(paths.ROOT, "*.py")
        if path.resolve() != here and needle in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert not hits, f"{needle} samples frames without the GIL and segfaults: {[str(p) for p in hits]}"


def core_limit_after_import(extra: dict[str, str]) -> tuple[int, int]:
    """(soft, hard) RLIMIT_CORE in a child that raised soft to hard and THEN imported the package."""
    probe = (
        "import resource\n"
        "soft, hard = resource.getrlimit(resource.RLIMIT_CORE)\n"
        "resource.setrlimit(resource.RLIMIT_CORE, (hard, hard))\n"
        "import hpcagent_bench\n"
        "print(*resource.getrlimit(resource.RLIMIT_CORE))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=dict(os.environ, **extra), check=True
    )
    soft, hard = (int(field) for field in done.stdout.split())
    return soft, hard


def test_importing_the_package_drops_the_core_limit() -> None:
    """The python-side half: a python process started outside the repo still cannot write a core."""
    soft, _ = core_limit_after_import({})
    assert soft == 0, "import hpcagent_bench must drop RLIMIT_CORE even from a shell that raised it"


def test_the_opt_out_keeps_the_core_limit() -> None:
    """A debugger session that asks for the dump gets it -- the guard is a floor, not a wall."""
    soft, hard = core_limit_after_import({core_dumps.ALLOW: "1"})
    assert soft == hard, f"{core_dumps.ALLOW}=1 must leave the caller's limit alone"


JUDGE_PROBE = """
import resource, subprocess, sys
from hpcagent_bench import core_dumps
core_dumps.keep_for_judge()
child = "import hpcagent_bench, resource; print(resource.getrlimit(resource.RLIMIT_CORE)[0])"
grading_child = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, check=True)
soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
print(soft, hard, grading_child.stdout.strip())
"""


def judge_core_limits(extra: dict[str, str]) -> tuple[int, int, int]:
    """(judge soft, judge hard, grading child soft) for a judge that imported the package (soft 0)
    and then called keep_for_judge, with one child started the way it starts a grading child."""
    done = subprocess.run(
        [sys.executable, "-c", JUDGE_PROBE], capture_output=True, text=True, env=dict(os.environ, **extra), check=True
    )
    soft, hard, child = (int(field) for field in done.stdout.split())
    return soft, hard, child


def test_the_judge_keeps_its_own_core_only_when_the_arm_asks() -> None:
    """A crash-diagnosis arm wants the judge's core; every other arm keeps the floor."""
    soft, hard = judge_core_limits({core_dumps.JUDGE: "1"})[:2]
    assert hard != 0, "precondition: this shell's hard limit forbids any core, nothing to test"
    assert soft == hard
    assert judge_core_limits({})[0] == 0


def test_a_grading_child_of_a_core_keeping_judge_still_dumps_nothing() -> None:
    """The judge's children run submissions, which crash by the dozen: they re-import the package
    and drop back to 0 although they inherit the judge's raised limit."""
    assert judge_core_limits({core_dumps.JUDGE: "1"})[2] == 0


FLOORED = ["experiments/run_cluster.sh"]


@pytest.mark.parametrize("script", FLOORED)
@pytest.mark.parametrize(("flag", "hard_is_zero"), [("", True), ("1", False)], ids=["floor", "judge-arm"])
def test_the_shell_floor_leaves_the_hard_limit_only_for_a_judge_core_arm(
    script: str, flag: str, hard_is_zero: bool
) -> None:
    """``ulimit -c 0`` sets BOTH limits, after which nothing below can raise its own. Every script on
    the judge's launch path must floor the soft limit alone on a judge-core arm."""
    text = (paths.ROOT / script).read_text()
    (guard,) = [line for line in text.splitlines() if line.startswith("if [[") and "ulimit" in line]
    done = subprocess.run(
        ["bash", "-c", f"{guard}; ulimit -H -c"],
        capture_output=True,
        text=True,
        env=dict(os.environ, **{core_dumps.JUDGE: flag}),
        check=True,
    )
    assert (done.stdout.strip() == "0") == hard_is_zero, done.stdout
