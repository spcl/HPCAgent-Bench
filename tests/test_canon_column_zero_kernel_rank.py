# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A rank canon_column.sh's inner mode hands zero kernels must not crash the whole step.

``mine`` (the kernels a given rank owns) is built by ``i % nranks == rank`` over the comma-list.
With fewer kernels than ranks -- exactly what a small smoke run does, e.g. 3 kernels over the
usual 4-ranks-per-node split -- the highest-numbered rank(s) get an empty ``mine``, the
``run-framework`` loop never runs for them, and their per-rank CSV is never created. The trailing
awk summary then tried to open that nonexistent file and died with "cannot open file", which is a
FATAL awk exit -- not caught by ``set -e`` (this script does not use it) but still the last
command's exit status, which is what srun sees as the task's own failure and uses to tear down
every sibling task in the step (smoke job 640088: 3 kernels, 4 ranks, rank 3's awk killed ranks
0-2 mid-run even though they had already produced valid rows).

This drives the real script's ``inner`` entry point directly with SLURM_PROCID/SLURM_NTASKS set to
reproduce that exact split -- not a reimplementation of its kernel-assignment math -- so a
regression in either the split or the summary guard is caught the same way.
"""

import os
import pathlib
import subprocess

from hpcagent_bench import paths

CANON_COLUMN = paths.ROOT / "experiments" / "canon_column.sh"


def test_a_rank_with_no_assigned_kernels_does_not_crash_the_summary_step(tmp_path: pathlib.Path) -> None:
    out_root = tmp_path / "out"
    out_root.mkdir()
    opt_dir = tmp_path / "opt"
    opt_dir.mkdir()

    # 3 kernels, 4 ranks: indices 0,1,2 go to ranks 0,1,2; rank 3 gets none (i % 4 never equals 3).
    env = dict(os.environ, SLURM_PROCID="3", SLURM_NTASKS="4")
    # A rank with an empty share never calls run-framework, so it needs no working PYTHONPATH/dace
    # tree -- the whole point is that this path must succeed without reaching that code at all.
    result = subprocess.run(
        ["bash", str(CANON_COLUMN), "inner", "stubcol", str(out_root), "a,b,c", "fuzzed", str(opt_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"rank 3 of 4 (0 of 3 kernels) should be a no-op, not a failure: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert not (out_root / "stubcol.rank3.csv").exists(), (
        "this test is only meaningful if rank 3 truly got no kernels and wrote no CSV; "
        "if it exists, the kernel-assignment math changed and this test no longer covers the empty-mine case"
    )
    assert "0 rows" in result.stdout, f"expected a zero-row summary line, got: {result.stdout!r}"
    assert "cannot open file" not in result.stderr, f"awk still choked on the missing CSV: {result.stderr!r}"
