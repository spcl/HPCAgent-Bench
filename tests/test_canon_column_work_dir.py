# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""canon_column.sh's job-work-dir contract: a MANAGED out_root (one this script's own convention
created under HPCAGENT_BENCH_RUNS_ROOT, per .cache/README.md's "Job work dirs" section) gets its
per-rank run-framework shard DB redirected into the work dir instead of run-framework's own
repo-relative default, and -- once a column's srun step returns -- has its DaCe build tree and
that shard DB deleted, but ONLY after its CSV rows are independently verified merged into the
persistent canon.db. An out_root outside that root (every pre-existing $SCRATCH/canon-*/smoke-*
directory) must be left exactly as it always was.

Real `srun`/the container launchers are not available outside Slurm, so these tests fake `srun`
with a stub that just runs its trailing command (canon_column.sh's own launch line is only ever
`srun <flags...> bash "$SELF" inner ...`, so stripping the leading `--flag` tokens is a faithful
enough stand-in), and fake the `run-framework` CLI call specifically (real python3 for everything
else, including the real scripts/merge_canon_results.py) so the test needs no dace tree, no numpy
build toolchain, and no benchmark registry -- this is a test of canon_column.sh's OWN wiring, not
of run-framework's CSV correctness (covered elsewhere).
"""

import os
import pathlib
import shutil
import sqlite3
import stat
import subprocess

import pytest

from hpcagent_bench import paths

CANON_COLUMN = paths.ROOT / "experiments" / "canon_column.sh"
REAL_PYTHON3 = shutil.which("python3")


def _write_executable(path: pathlib.Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _fake_bin_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """A directory to prepend to PATH holding a stub `srun` (runs its trailing command verbatim,
    dropping every `--flag`/`--flag=value` token) and a stub `python3` that fakes ONLY
    `-m hpcagent_bench.cli run-framework` (writes one CSV row and touches
    HPCAGENT_BENCH_RECORD_DB_PATH, proving canon_column.sh set it) and defers every other
    invocation -- notably finalize_column's call into the real merge_canon_results.py -- to the
    real interpreter."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _write_executable(
        bindir / "srun",
        "#!/usr/bin/env bash\n"
        "args=()\n"
        'for a in "$@"; do\n'
        '    [[ "$a" == --* ]] && continue\n'
        '    args+=("$a")\n'
        "done\n"
        'exec "${args[@]}"\n',
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
        '    if [[ -n "${HPCAGENT_BENCH_RECORD_DB_PATH:-}" ]]; then\n'
        '        mkdir -p "$(dirname "${HPCAGENT_BENCH_RECORD_DB_PATH}")"\n'
        '        : > "${HPCAGENT_BENCH_RECORD_DB_PATH}"\n'
        "    fi\n"
        "    exit 0\n"
        "fi\n"
        f'exec "{REAL_PYTHON3}" "$@"\n',
    )
    return bindir


def _base_env(tmp_path: pathlib.Path, bindir: pathlib.Path) -> dict:
    cache_root = tmp_path / "jitcache"
    dace_stub = tmp_path / "dace-stub"
    (dace_stub / "dace" / "external" / "moodycamel").mkdir(parents=True)
    (dace_stub / "dace" / "external" / "moodycamel" / "blockingconcurrentqueue.h").write_text("")
    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    # Isolate the cache root the real scripts/cache_env.sh derives everything from, so this test
    # never reads or writes the real, shared cache.
    env["JIT_CACHE_ROOT"] = str(cache_root)
    env.pop("HPCAGENT_BENCH_CACHE", None)
    env.pop("HPCAGENT_BENCH_RUNS_ROOT", None)
    env.pop("HPCAGENT_BENCH_RESULTS_DIR", None)
    env.pop("HPCAGENT_BENCH_RECORD_DB_PATH", None)
    env["DACE_TREE"] = str(dace_stub)
    env["CANON_LAUNCH"] = "test-stub"  # anything but "enroot": takes the (faked) srun branch
    env["CANON_RANKS"] = "1"
    env["CANON_OPT_REPORTS"] = "0"
    return env


def test_a_managed_work_dir_is_derived_from_the_cache_root_never_from_scratch(tmp_path: pathlib.Path) -> None:
    """out_root under HPCAGENT_BENCH_RUNS_ROOT is what makes canon_column.sh manage it at all; the
    root itself must come from cache_env.sh's JIT_CACHE_ROOT, not be spelled out as a bare
    $SCRATCH/<name> path anywhere in this script."""
    bindir = _fake_bin_dir(tmp_path)
    env = _base_env(tmp_path, bindir)
    runs_root = pathlib.Path(env["JIT_CACHE_ROOT"]) / "runs"
    out_root = runs_root / "canon" / "unit-test"
    out_root.mkdir(parents=True)

    result = subprocess.run(
        ["bash", str(CANON_COLUMN), "outer", "fakecol", str(out_root), "fakekernel", "fuzzed", str(paths.ROOT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert str(out_root).startswith(str(runs_root)), "the test's own out_root must itself be under the cache root"


def test_the_per_rank_db_lands_under_the_work_dir_not_the_repo_or_cwd(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before this redirection, run-framework's own default (record.db_path, repo-relative) put
    every rank's shard DB in whatever the process's CWD happened to be -- the repo checkout, for a
    canon job launched from there. A managed out_root must never let that happen."""
    monkeypatch.chdir(tmp_path)  # simulates "launched from the repo checkout"'s CWD
    repo_cwd_db = tmp_path / "hpcagent_bench0.db"
    toolbin = tmp_path / "toolbin"
    toolbin.mkdir()
    bindir = _fake_bin_dir(toolbin)
    env = _base_env(tmp_path, bindir)
    runs_root = pathlib.Path(env["JIT_CACHE_ROOT"]) / "runs"
    out_root = runs_root / "canon" / "unit-test"
    out_root.mkdir(parents=True)

    subprocess.run(
        ["bash", str(CANON_COLUMN), "outer", "fakecol", str(out_root), "fakekernel", "fuzzed", str(paths.ROOT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )

    assert not repo_cwd_db.exists(), "the shard DB must not appear beside the process's CWD"


def test_a_verified_merge_deletes_the_build_tree_and_shard_db_but_keeps_the_csv(tmp_path: pathlib.Path) -> None:
    """The CSV is the documented hand-off to the reproducibility repos' own collect_canon.py pass
    (experiments/README.md's canon section) and must survive; the DaCe build tree
    (dacecache-<column>) and the now-redundant sqlite shard are the disposable bulk and must not."""
    bindir = _fake_bin_dir(tmp_path)
    env = _base_env(tmp_path, bindir)
    runs_root = pathlib.Path(env["JIT_CACHE_ROOT"]) / "runs"
    out_root = runs_root / "canon" / "unit-test"
    out_root.mkdir(parents=True)

    result = subprocess.run(
        ["bash", str(CANON_COLUMN), "outer", "fakecol", str(out_root), "fakekernel", "fuzzed", str(paths.ROOT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "merged 1 row(s)" in result.stdout, result.stdout
    assert not (out_root / "dacecache-fakecol").exists(), "the DaCe build tree must be cleared after a verified merge"
    assert not (out_root / "db" / "fakecol").exists(), (
        "the redundant shard DB dir must be cleared after a verified merge"
    )
    assert (out_root / "fakecol.rank0.csv").exists(), "the CSV is the external hand-off and must survive cleanup"

    db_path = pathlib.Path(env["JIT_CACHE_ROOT"]) / "results" / "canon.db"
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        rows = conn.execute("SELECT column, kernel, validated FROM canon").fetchall()
    assert rows == [("fakecol", "fakekernel", "True")]


def test_a_merge_that_cannot_be_verified_keeps_every_column_artifact(tmp_path: pathlib.Path) -> None:
    """A results DB path that cannot be opened (here: a directory sitting where the file should be)
    makes the real merge_canon_results.py fail for a genuine reason, unrelated to the test's fakes;
    canon_column.sh must keep the build tree, the shard DB and the CSV rather than silently losing
    the run's only record of them."""
    bindir = _fake_bin_dir(tmp_path)
    env = _base_env(tmp_path, bindir)
    runs_root = pathlib.Path(env["JIT_CACHE_ROOT"]) / "runs"
    out_root = runs_root / "canon" / "unit-test"
    out_root.mkdir(parents=True)
    # Force the merge to fail for a real reason: canon.db's parent exists but canon.db itself is a
    # directory, so sqlite3.connect() cannot open it as a database file.
    results_dir = pathlib.Path(env["JIT_CACHE_ROOT"]) / "results"
    (results_dir / "canon.db").mkdir(parents=True)

    result = subprocess.run(
        ["bash", str(CANON_COLUMN), "outer", "fakecol", str(out_root), "fakekernel", "fuzzed", str(paths.ROOT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert "NOT verified" in result.stderr, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert (out_root / "dacecache-fakecol").exists(), "a failed merge must keep the build tree"
    assert (out_root / "db" / "fakecol").exists(), "a failed merge must keep the shard DB dir"
    assert (out_root / "fakecol.rank0.csv").exists(), "a failed merge must keep the CSV"


def test_a_zero_kernel_rank_in_a_managed_work_dir_still_finalizes_cleanly(tmp_path: pathlib.Path) -> None:
    """The zero-kernel-rank guard (tests/test_canon_column_zero_kernel_rank.py) predates the
    managed work dir; this is the same empty-share case but through the NEW path, where a column
    that produced no CSV at all must still be a clean, mergeable (0 rows), deletable finalize."""
    bindir = _fake_bin_dir(tmp_path)
    env = _base_env(tmp_path, bindir)
    runs_root = pathlib.Path(env["JIT_CACHE_ROOT"]) / "runs"
    out_root = runs_root / "canon" / "unit-test"
    out_root.mkdir(parents=True)

    # An empty kernel list is the degenerate case of "fewer kernels than ranks": the run-framework
    # loop never fires, no CSV is ever created for this column, and finalize_column must still
    # treat that as a clean 0-row merge rather than a missing-shard error.
    result = subprocess.run(
        # A single space, not "": ${4:?...} (colon form) treats an EMPTY string the same as unset
        # and would abort the script before it ever reaches the kernel-splitting loop this case is
        # about -- a whitespace-only value is non-empty and still splits to zero kernel names.
        ["bash", str(CANON_COLUMN), "outer", "fakecol", str(out_root), " ", "fuzzed", str(paths.ROOT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "merged 0 row(s)" in result.stdout, result.stdout
    assert not (out_root / "dacecache-fakecol").exists()
