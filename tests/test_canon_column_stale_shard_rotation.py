# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""canon_column.sh's ``outer`` mode must rotate a column's PRE-EXISTING shard CSVs aside before a
fresh run ever opens a CSV path, or a re-run into the same out_root can resurrect a stale row.

The bug this guards against needed more than plain staleness to show up: run-framework's own CSV
writer APPENDS to an existing rank CSV (sweep.write_csv_rows), so once a roster or rank-count change
moves a kernel to a DIFFERENT rank between two runs into the SAME out_root (a smoke then the full
sweep, an owed resubmit), the file that kept the kernel's OLD row can go on to receive a FRESH
append for some other kernel this run -- which bumps that file's mtime to the end of the run, past
the file holding the genuinely fresh row for the moved kernel. Ordering shards by mtime (or by
filename) then applies the stale row LAST, and ``INSERT OR REPLACE`` keeps whichever row it applied
last.

Exact repro: an OLD run's 3-rank roster put kernel ``d`` on rank 0 (``i % 3``), which also carries
``a``; a NEW run's 4-rank roster moves ``d`` to rank 3 (``i % 4``) and leaves ``a`` on rank 0. Rank 3
finishes first, rank 0 (which re-touches its file for ``a``) finishes last. Without rotation, rank
0's file mtime ends up newer than rank 3's, and the merge would keep rank 0's stale ``d`` row.

This drives ``outer`` end to end (fake ``srun``/``run-framework``, real ``scripts/merge_canon_results.py``),
the same harness shape as tests/test_canon_column_work_dir.py, extended with a multi-rank ``srun``
stub that runs ranks in a controlled order so the mtimes land exactly as the repro needs.
"""

import contextlib
import os
import pathlib
import shutil
import sqlite3
import stat
import subprocess

from hpcagent_bench import paths
from tests.dace_checkout import pinned_dace

CANON_COLUMN = paths.ROOT / "experiments" / "canon_column.sh"
REAL_PYTHON3 = shutil.which("python3")


def _write_executable(path: pathlib.Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _fake_bin_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """A stub ``srun`` that honours ``--ntasks=N`` (every other `--flag` token is dropped, matching
    canon_column.sh's own launch line) and runs ranks HIGHEST-NUMBERED FIRST -- rank 3 before rank
    0 -- to reproduce the exact finish order the repro needs without depending on real timing. A
    stub ``python3`` fakes only ``-m hpcagent_bench.cli run-framework`` (one fixed-value CSV row per
    call, proving nothing beyond "this run wrote it"); everything else, including the real
    ``scripts/merge_canon_results.py``, runs under the real interpreter."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _write_executable(
        bindir / "srun",
        "#!/usr/bin/env bash\n"
        "ntasks=1\n"
        "args=()\n"
        'for a in "$@"; do\n'
        '    case "$a" in\n'
        '        --ntasks=*) ntasks="${a#--ntasks=}" ;;\n'
        "        --*) : ;;\n"
        '        *) args+=("$a") ;;\n'
        "    esac\n"
        "done\n"
        "rc=0\n"
        "for ((i = ntasks - 1; i >= 0; i--)); do\n"
        '    SLURM_PROCID="$i" SLURM_NTASKS="$ntasks" "${args[@]}" || rc=1\n'
        "done\n"
        'exit "$rc"\n',
    )
    _write_executable(
        bindir / "python3",
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "-m" && "$2" == "hpcagent_bench.cli" && "$3" == "run-framework" ]]; then\n'
        "    shift 3\n"
        '    kernel="" preset="" csv=""\n'
        "    while [[ $# -gt 0 ]]; do\n"
        '        case "$1" in\n'
        "            -b) kernel=$2; shift 2 ;;\n"
        "            -p) preset=$2; shift 2 ;;\n"
        "            --csv) csv=$2; shift 2 ;;\n"
        "            *) shift ;;\n"
        "        esac\n"
        "    done\n"
        '    [[ -s "$csv" ]] || printf \'kernel,preset,datatype,median_ms,validated\\n\' > "$csv"\n'
        '    printf \'%s,%s,float64,1.5,True\\n\' "$kernel" "$preset" >> "$csv"\n'
        "    exit 0\n"
        "fi\n"
        f'exec "{REAL_PYTHON3}" "$@"\n',
    )
    return bindir


def _base_env(tmp_path: pathlib.Path, bindir: pathlib.Path, ranks: str) -> dict:
    cache_root = tmp_path / "jitcache"
    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env["JIT_CACHE_ROOT"] = str(cache_root)
    env.pop("HPCAGENT_BENCH_CACHE", None)
    env.pop("HPCAGENT_BENCH_RUNS_ROOT", None)
    env.pop("HPCAGENT_BENCH_RESULTS_DIR", None)
    env.pop("HPCAGENT_BENCH_RECORD_DB_PATH", None)
    env["HPCAGENT_BENCH_IMAGE_PYTHON"] = str(bindir / "python3")
    env["HPCAGENT_BENCH_HOST_PYTHON"] = REAL_PYTHON3
    env.update(pinned_dace(tmp_path))
    env["CANON_RANKS"] = ranks
    env["CANON_OPT_REPORTS"] = "0"
    return env


def _write_stale_shard(out_root: pathlib.Path, column: str, rank: int, kernel: str, median_ms: str) -> pathlib.Path:
    shard = out_root / f"{column}.rank{rank}.csv"
    shard.write_text(f"kernel,preset,datatype,median_ms,validated\n{kernel},fuzzed,float64,{median_ms},True\n")
    old_t = 1_700_000_000
    os.utime(shard, (old_t, old_t))
    return shard


def test_a_stale_shard_from_an_old_roster_never_resurrects_through_the_real_outer_path(
    tmp_path: pathlib.Path,
) -> None:
    bindir = _fake_bin_dir(tmp_path)
    env = _base_env(tmp_path, bindir, ranks="4")
    runs_root = pathlib.Path(env["JIT_CACHE_ROOT"]) / "runs"
    out_root = runs_root / "canon" / "unit-test"
    out_root.mkdir(parents=True)

    # The OLD run's 3-rank roster put "d" (i=3, 3%3=0) on rank 0, alongside "a" (i=0, 0%3=0).
    # Only rank 0's shard matters for the repro: it is the one the NEW run's rank 0 re-touches.
    _write_stale_shard(out_root, "fakecol", rank=0, kernel="d", median_ms="999.0")

    # The NEW run's 4-rank roster: a->rank0, b->rank1, c->rank2, d->rank3 (i % 4). Without
    # rotation, rank 0 (processed LAST by the stub srun above) appends "a" to the SAME file that
    # still holds the stale "d,999.0" row, giving that file the newest mtime of the run and letting
    # its stale "d" row win the merge over rank 3's fresh one.
    result = subprocess.run(
        ["bash", str(CANON_COLUMN), "outer", "fakecol", str(out_root), "a,b,c,d", "fuzzed", str(paths.ROOT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "merged 4 row(s)" in result.stdout, result.stdout

    db_path = pathlib.Path(env["JIT_CACHE_ROOT"]) / "results" / "canon.db"
    with contextlib.closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        rows = {r["kernel"]: dict(r) for r in conn.execute("SELECT * FROM canon WHERE column = 'fakecol'")}

    assert rows["d"]["median_ms"] == 1.5, (
        "the stale 999.0 row from the old 3-rank roster must not win over this run's fresh row"
    )
    assert rows["a"]["median_ms"] == 1.5

    # Rotated aside, not lost: the old row must still be recoverable somewhere under out_root.
    stale_root = out_root / ".stale-shards"
    assert stale_root.is_dir(), "the pre-existing shard must have been moved aside, not left in place"
    stale_contents = "\n".join(p.read_text() for p in stale_root.rglob("*.csv"))
    assert "999.0" in stale_contents, "the old row must survive the rotation for inspection"

    # And the CURRENT run's own shards must be clean of the old row: rotation happened before
    # inner ever opened a CSV path for this run.
    assert "999.0" not in (out_root / "fakecol.rank0.csv").read_text()
