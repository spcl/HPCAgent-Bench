# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional compiler-report + lowered-code dumps, written under ``perf_reports/``.

Two INDEPENDENT capabilities, BOTH OFF BY DEFAULT:

* ``opt_report``   -- what the compiler's vectorizer DID (and at what width) and
  what it REFUSED (and why).
* ``lowered_code`` -- the machine code actually emitted, disassembled.

Neither may perturb a TIMED run, which is enforced structurally rather than by
promise:

* Each knob is off unless explicitly turned on, so a default sweep never runs
  either path.
* Even when on, a report is produced OUTSIDE the timed bracket -- the harness asks
  only after :meth:`Framework.measure` has returned (``frameworks/test.py``).
* The optimization report comes from a SEPARATE compile-only run into a scratch
  directory; the report flags never reach the build whose ``.so`` gets timed. (They
  are in fact codegen-neutral -- ``objdump``'s ``.text`` is byte-identical with and
  without them -- but not relying on that is free, and it also dodges the reverse
  hazard: a cached ``.so`` built without the flags would otherwise have to be
  rebuilt to report on, silently re-timing a different artifact.)
* The disassembly reads the timed ``.so`` that already exists; it never rebuilds.

This module is the MECHANISM only -- where a report goes, and how to disassemble a
library. WHAT a report says is the framework's own answer, via the
:meth:`Framework.opt_report` / :meth:`Framework.lowered_code` hooks; WHEN to ask is
the harness's. It therefore imports nothing from :mod:`hpcagent_bench.frameworks` (which
imports the harness that calls this), and takes the two path components it needs --
``relative_path`` / ``module_name`` -- as plain strings.
"""
import pathlib
import shutil
import subprocess
from typing import Optional

from hpcagent_bench import config, paths

#: Root of the report tree. MIRRORS the benchmark folder structure, so a kernel's
#: reports sit at the same relative path its sources do (``perf_reports/hpc/
#: map_reduce/arc_distance/``). Gitignored + gitkeep'd: the per-kernel directories
#: are created on demand by :func:`write`, never committed -- there are 349 kernels
#: and materialising that tree up front would commit 349 empty directories to hold
#: output that only an opted-in run produces.
REPORTS: pathlib.Path = paths.ROOT / "perf_reports"

#: Report kind -> the filename suffix it lands under. The kind is also the config
#: key (``perf_reports.<kind>``) and the env knob (``$HPCAGENT_BENCH_PERF_REPORTS_<KIND>``),
#: so the two capabilities stay independently switchable with no third name to keep
#: in sync.
KINDS = {
    "opt_report": "opt-report.txt",
    "lowered_code": "asm.txt",
}


def enabled(kind: str) -> bool:
    """Whether report ``kind`` is switched on (default: NO).

    Reads ``perf_reports.<kind>``, so ``$HPCAGENT_BENCH_PERF_REPORTS_OPT_REPORT=1`` turns
    one on for a run -- the repo's env/config idiom rather than a CLI flag, which
    also means :mod:`hpcagent_bench.containers` forwards it into a container for free.
    """
    if kind not in KINDS:
        raise KeyError(f"unknown report kind {kind!r}; known: {sorted(KINDS)}")
    return bool(config.get(f"perf_reports.{kind}", False))


def report_path(relative_path: str, module_name: str, framework: str, impl_name: str, kind: str) -> pathlib.Path:
    """Where report ``kind`` for one (kernel, framework, implementation) lands.

    ``<REPORTS>/<relative_path>/<module_name>.<framework>.<impl_name>.<suffix>``.

    Framework and implementation are in the FILENAME, not directory levels: the
    variants of one kernel are read side by side (why did clang vectorize this loop
    and gcc not; did numba's parallel track vectorize where its serial track did
    not), which per-variant subdirectories would scatter. ``impl_name`` is included
    even when a framework has only one implementation -- an artifact that was timed
    separately gets a report of its own, and a name that is uniform is one fewer rule
    to remember.
    """
    if kind not in KINDS:
        raise KeyError(f"unknown report kind {kind!r}; known: {sorted(KINDS)}")
    return REPORTS / relative_path / f"{module_name}.{framework}.{impl_name}.{KINDS[kind]}"


def write(relative_path: str, module_name: str, framework: str, impl_name: str, kind: str,
          text: Optional[str]) -> Optional[pathlib.Path]:
    """Write ``text`` as report ``kind``, creating the directory on demand.

    ``text=None`` means the framework does not support this report (a GPU flavor has
    no ``.so`` to disassemble; a compiler has no report channel wired; numba's serial
    track has no parallel diagnostics). That is a normal answer, not an error:
    nothing is written and ``None`` comes back, so a caller can enable a knob across
    a mixed sweep and get reports from the frameworks that have one without
    special-casing the ones that do not.
    """
    if text is None:
        return None
    path = report_path(relative_path, module_name, framework, impl_name, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def objdump(lib: pathlib.Path) -> Optional[str]:
    """Disassemble ``lib`` (a built ``.so``) with ``objdump -d -C``, or ``None``.

    ``None`` when objdump is absent or the file is not there / not an object it can
    read -- a dump is a diagnostic, so a missing one must degrade to "no report",
    never take down the run that produced the number.

    ``-C`` demangles: the C++ flavors (llvm/polly, both clang++) otherwise name every
    symbol in its mangled form. Default AT&T syntax is kept deliberately -- the
    register names it prints (``%zmm``/``%ymm``) are what an ISA census greps for.
    """
    exe = shutil.which("objdump")
    if exe is None or not pathlib.Path(lib).is_file():
        return None
    proc = subprocess.run([exe, "-d", "-C", str(lib)], capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    return proc.stdout
