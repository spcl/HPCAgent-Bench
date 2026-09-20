# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The per-kernel memory cap must bound the HEAP (RLIMIT_DATA), not the whole address space
(RLIMIT_AS), or a GPU column crashes under it.

2026-09-20: canon_column.sh gained a per-kernel memory cap (job 640519: pluto rank 2 OOM-killed at
~465 GB RSS under --mem=0's no-per-rank-reservation, taking every sibling rank's in-flight kernel
down with it) via ``ulimit -v`` (RLIMIT_AS), applied unconditionally to every column including the
GPU ones. A HIP process reserves its own VRAM aperture as address space at ``hipInit`` -- measured
with a probe job (644414, gfx942): ~97 GiB with no limit, shrunk to ~73 GiB to fit inside a 96 GiB
RLIMIT_AS cap -- leaving too little of the same 96 GiB budget for the kernel's own device+host
buffers. Job 644343 (ppcg_hip, the first GPU canon column run under the cap) crashed 7 of 40
kernels: ``hipMalloc`` "out of memory" and, on the host side, a numpy ``MemoryError`` on a 2.84 GiB
array a 513 GB node should never fail to give.

RLIMIT_DATA does not count that aperture (VmData held flat across every case the probe measured,
whether hipMalloc'd or not) while still rejecting a real over-cap heap allocation (confirmed
separately, outside the probe job: a RLIMIT_DATA cap does reject an over-cap allocation the same
way RLIMIT_AS does -- it is not a no-op on this kernel), so it is the knob that protects against
job 640519's failure mode without breaking a GPU column.

This drives the real script's ``inner`` entry point against a stub ``hpcagent_bench.cli`` that
reports its own rlimits instead of running a kernel -- the same way
``test_canon_column_kernel_timeout.py`` drives it against a stub that never returns -- so a
regression in the actual ulimit line is caught, not a reimplementation of it.
"""

import os
import pathlib
import re
import resource
import subprocess

from hpcagent_bench import paths

CANON_COLUMN = paths.ROOT / "experiments" / "canon_column.sh"


def stub_opt_reporting_rlimits(tmp_path: pathlib.Path) -> pathlib.Path:
    """An ``opt`` tree with just enough to satisfy ``inner`` up to the run-framework call: a no-op
    ``scripts/cache_env.sh`` and a fake ``hpcagent_bench.cli`` that reports the rlimits it was
    actually started with instead of running a kernel."""
    opt_dir = tmp_path / "opt"
    (opt_dir / "scripts").mkdir(parents=True)
    (opt_dir / "scripts" / "cache_env.sh").write_text("# stub cache_env.sh for this test, no-op\n")
    pkg = opt_dir / "hpcagent_bench"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "cli.py").write_text(
        "# stub run-framework that reports its own rlimits instead of running a kernel, so the\n"
        "# test can see exactly what canon_column.sh's ulimit line left this process with.\n"
        "import resource\n\n"
        "if __name__ == '__main__':\n"
        "    data_soft, data_hard = resource.getrlimit(resource.RLIMIT_DATA)\n"
        "    as_soft, as_hard = resource.getrlimit(resource.RLIMIT_AS)\n"
        "    print(f'RLIMIT_DATA={data_soft},{data_hard}')\n"
        "    print(f'RLIMIT_AS={as_soft},{as_hard}')\n"
    )
    return opt_dir


def run_inner(tmp_path: pathlib.Path, extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    out_root = tmp_path / "out"
    out_root.mkdir()
    opt_dir = stub_opt_reporting_rlimits(tmp_path)
    env = dict(os.environ, SLURM_PROCID="0", SLURM_NTASKS="1", **extra_env)
    return subprocess.run(
        ["bash", str(CANON_COLUMN), "inner", "stubcol", str(out_root), "onlykernel", "fuzzed", str(opt_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def rlimits_from(stdout: str) -> dict[str, tuple[int, int]]:
    """The stub's rlimits, from its LAST print of each name.

    ``inner`` invokes the stub cli TWICE before a kernel is even reached: once for its own
    ``preflight --tools-only`` gate (unconstrained -- that call sits outside the per-kernel
    subshell) and once per kernel inside the ``ulimit`` subshell this test actually cares about.
    The last occurrence is always the kernel-run one."""
    limits = {}
    for name in ("RLIMIT_DATA", "RLIMIT_AS"):
        matches = re.findall(rf"^{name}=(-?\d+),(-?\d+)$", stdout, re.MULTILINE)
        assert matches, f"stub did not report {name}: {stdout!r}"
        soft, hard = matches[-1]
        limits[name] = (int(soft), int(hard))
    return limits


def test_default_cap_bounds_rlimit_data_not_rlimit_as(tmp_path: pathlib.Path) -> None:
    result = run_inner(tmp_path, {})
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"

    limits = rlimits_from(result.stdout)
    expected_bytes = 100663296 * 1024  # 96 GiB, canon_column.sh's own default (KB, ulimit -d unit)
    assert limits["RLIMIT_DATA"] == (expected_bytes, expected_bytes)
    # The bug this guards: RLIMIT_AS must be left untouched, or a GPU column's hipInit aperture
    # (~97 GiB measured, probe job 644414) competes with the kernel's own buffers for one budget.
    unlimited = (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    assert limits["RLIMIT_AS"] == unlimited, (
        "canon_column.sh must not constrain RLIMIT_AS -- a GPU column's HIP aperture alone can "
        f"reserve tens of GB of address space, got {limits['RLIMIT_AS']}"
    )


def test_canon_kernel_mem_kb_overrides_the_default(tmp_path: pathlib.Path) -> None:
    result = run_inner(tmp_path, {"CANON_KERNEL_MEM_KB": "2097152"})  # 2 GiB
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"

    limits = rlimits_from(result.stdout)
    expected_bytes = 2097152 * 1024
    assert limits["RLIMIT_DATA"] == (expected_bytes, expected_bytes)
