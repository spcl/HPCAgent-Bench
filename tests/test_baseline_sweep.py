# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``hpcagent-bench job baseline``: one compiler column over a roster, split over the tasks of a step.

The kernels run through a STUB image interpreter whose ``-m hpcagent_bench.cli`` is a small script (a
``run-framework`` that writes one CSV row, one that hangs, one that reports its rlimits), so no dace tree, build
toolchain or benchmark registry is needed: what is tested is the sweep's own wiring -- the rank split, the wall
and heap caps, the timeout row, the managed work dir (shard DB redirection, rotation of an old run's shards, the
verified merge into ``canon.db``) and the roster resolution -- not ``run-framework``'s CSV correctness.
"""

import contextlib
import dataclasses
import os
import pathlib
import re
import resource
import shutil
import sqlite3
import subprocess
import sys

import pytest

from hpcagent_bench import paths
from hpcagent_bench.cluster import baseline, jobs
from tests.dace_checkout import pinned_dace

#: A ``run-framework`` that writes one row per call in ``merge_canon_results.py``'s shape and marks what it saw.
WRITING_CLI = """
import os
import pathlib
import sys

if sys.argv[1] == "preflight":
    raise SystemExit(0)
args = sys.argv[2:]
kernel, preset, csv = args[args.index("-b") + 1], args[args.index("-p") + 1], pathlib.Path(args[args.index("--csv") + 1])
if not csv.is_file():
    csv.write_text("kernel,preset,datatype,median_ms,validated\\n")
with csv.open("a") as handle:
    handle.write(f"{kernel},{preset},float64,1.5,True\\n")
db = os.environ.get("HPCAGENT_BENCH_RECORD_DB_PATH")
if db:
    pathlib.Path(db).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(db).touch()
pathlib.Path(str(csv) + ".record_build").write_text(os.environ.get("HPCAGENT_BENCH_RECORD_BUILD", "") + "\\n")
"""

#: ``preflight --tools-only`` has to return, or the hang under test happens there, outside the per-kernel cap.
HANGING_CLI = (
    "import sys\nimport time\n\nif sys.argv[1:2] == ['preflight']:\n    raise SystemExit(0)\ntime.sleep(9999)\n"
)

RLIMIT_CLI = (
    "import resource\n\n"
    "data_soft, data_hard = resource.getrlimit(resource.RLIMIT_DATA)\n"
    "as_soft, as_hard = resource.getrlimit(resource.RLIMIT_AS)\n"
    "print(f'RLIMIT_DATA={data_soft},{data_hard}')\n"
    "print(f'RLIMIT_AS={as_soft},{as_hard}')\n"
)


def stub_checkout(root: pathlib.Path, cli: str) -> pathlib.Path:
    """A checkout whose ``scripts/`` are the real ones (the cache roots derive from ``JIT_CACHE_ROOT``) and whose
    image interpreter answers ``-m hpcagent_bench.cli`` from ``cli``."""
    opt = root / "opt"
    shutil.copytree(paths.ROOT / "scripts", opt / "scripts", ignore=shutil.ignore_patterns("checks", "*.py"))
    shutil.copy2(paths.ROOT / "scripts" / "merge_canon_results.py", opt / "scripts" / "merge_canon_results.py")
    (opt / "cli.py").write_text(cli)
    python = root / "image-python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        f'[[ "$1 $2" == "-m hpcagent_bench.cli" ]] && {{ shift 2; exec "{sys.executable}" "{opt / "cli.py"}" "$@"; }}\n'
        f'exec "{sys.executable}" "$@"\n'
    )
    python.chmod(0o755)
    return opt


def environment(root: pathlib.Path, **extra: str) -> dict[str, str]:
    """The step's environment: a private cache root, the stub interpreter and a pinned stand-in dace checkout."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"HPCAGENT_BENCH_CACHE", "HPCAGENT_BENCH_RUNS_ROOT", "HPCAGENT_BENCH_RESULTS_DIR"}
        and key not in {"HPCAGENT_BENCH_RECORD_DB_PATH", "HPCAGENT_BENCH_RECORD_BUILD", "SLURM_PROCID", "SLURM_NTASKS"}
    }
    env.update(JIT_CACHE_ROOT=str(root / "jitcache"), HPCAGENT_BENCH_IMAGE_PYTHON=str(root / "image-python"))
    env.update(pinned_dace(root))
    env.update(CANON_OPT_REPORTS="0", SLURM_CPUS_PER_TASK="2", **extra)
    return env


def sweep_of(root: pathlib.Path, opt: pathlib.Path, kernels: tuple[str, ...], managed: bool = True, **extra: str):
    """A sweep of column ``fakecol``; ``managed`` puts its out_root under the cache's runs root."""
    env = baseline.cache_environment(opt, environment(root, **extra))
    out_root = (pathlib.Path(env["HPCAGENT_BENCH_RUNS_ROOT"]) / "canon" / "unit") if managed else root / "out"
    out_root.mkdir(parents=True)
    return baseline.Sweep("fakecol", out_root, kernels, "fuzzed", opt, env)


def canon_rows(sweep: baseline.Sweep) -> dict[str, dict]:
    database = pathlib.Path(sweep.environ["HPCAGENT_BENCH_RESULTS_DIR"]) / "canon.db"
    with contextlib.closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        return {row["kernel"]: dict(row) for row in conn.execute("SELECT * FROM canon WHERE column = 'fakecol'")}


# ------------------------------------------------------------------------------------------- the rank split


@pytest.mark.parametrize("size", [1, 2, 3, 4, 7])
def test_the_ranks_split_a_roster_disjointly_and_completely(size: int) -> None:
    roster = [f"k{index}" for index in range(10)]
    shares = [jobs.share(roster, jobs.Rank(index, size)) for index in range(size)]
    assert sorted(name for share in shares for name in share) == sorted(roster)
    assert all(not set(a) & set(b) for i, a in enumerate(shares) for b in shares[i + 1 :])


def test_a_rank_with_no_kernels_is_a_no_op_that_needs_no_dace_tree(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """3 kernels over 4 ranks leave rank 3 empty. Its CSV is never created, and it must neither crash nor need a
    working checkout: a nonzero task exit makes ``srun`` tear down the sibling ranks that already have rows."""
    out_root = tmp_path / "out"
    out_root.mkdir()
    empty = tmp_path / "opt"
    empty.mkdir()
    sweep = baseline.Sweep("stubcol", out_root, ("a", "b", "c"), "fuzzed", empty, {})
    assert baseline.run(sweep, jobs.Rank(3, 4)) == 0
    assert not (out_root / "stubcol.rank3.csv").exists()
    out = capsys.readouterr().out
    assert "no kernels assigned" in out and "0 rows" in out


# -------------------------------------------------------------------------------------------- the per-kernel caps


def test_a_hung_kernel_is_killed_and_recorded_as_a_timeout_row_not_a_silent_gap(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stalled ``run-framework`` used to hold a rank for ten hours, until the job's own limit killed it, and every
    kernel after it got no row on any rank."""
    opt = stub_checkout(tmp_path, HANGING_CLI)
    sweep = sweep_of(tmp_path, opt, ("onlykernel",), managed=False, CANON_KERNEL_TIMEOUT_SEC="2")
    assert baseline.run(sweep, jobs.Rank(0, 1)) == 0
    out = capsys.readouterr().out
    assert "FAILED onlykernel (wall timeout after 2s" in out
    rows = sweep.csv(0).read_text().splitlines()
    assert rows[0] == baseline.CSV_HEADER
    assert len(rows) == 2, rows
    fields = rows[1].split(",")
    assert (fields[0], fields[3], fields[5], fields[8]) == ("fakecol", "onlykernel", "timeout", "timeout")
    # status != ok, so the row is tallied as crashed, and the nonzero-exit counter agrees with finish's row count.
    assert "1 rows -- 0 ok, 0 unsupported, 0 tool-missing, 1 crashed, 0 failed-in-column, 1 nonzero-exit" in out


def rlimits_of(out: str) -> dict[str, tuple[int, int]]:
    """The stub's rlimits, from its LAST print of each name: the preflight gate prints them too, unconstrained."""
    limits = {}
    for name in ("RLIMIT_DATA", "RLIMIT_AS"):
        matches = re.findall(rf"^{name}=(-?\d+),(-?\d+)$", out, re.MULTILINE)
        assert matches, f"stub did not report {name}: {out!r}"
        limits[name] = (int(matches[-1][0]), int(matches[-1][1]))
    return limits


def test_the_default_cap_bounds_the_heap_not_the_address_space(
    tmp_path: pathlib.Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """RLIMIT_AS counts the ~97 GiB VRAM aperture hipInit reserves, so under it a GPU column crashed 7 of 40
    kernels; RLIMIT_DATA does not count it and still rejects a real over-cap heap allocation."""
    opt = stub_checkout(tmp_path, RLIMIT_CLI)
    sweep = sweep_of(tmp_path, opt, ("onlykernel",), managed=False)
    assert baseline.run(sweep, jobs.Rank(0, 1)) == 0
    limits = rlimits_of(capfd.readouterr().out)
    heap = 100663296 * 1024  # 96 GiB, CANON_KERNEL_MEM_KB's default
    assert limits["RLIMIT_DATA"] == (heap, heap)
    assert limits["RLIMIT_AS"] == (resource.RLIM_INFINITY, resource.RLIM_INFINITY)


def test_canon_kernel_mem_kb_overrides_the_default(tmp_path: pathlib.Path, capfd: pytest.CaptureFixture[str]) -> None:
    opt = stub_checkout(tmp_path, RLIMIT_CLI)
    sweep = sweep_of(tmp_path, opt, ("onlykernel",), managed=False, CANON_KERNEL_MEM_KB="2097152")
    assert baseline.run(sweep, jobs.Rank(0, 1)) == 0
    assert rlimits_of(capfd.readouterr().out)["RLIMIT_DATA"] == (2097152 * 1024, 2097152 * 1024)


def test_each_rank_is_masked_to_one_hip_device_of_the_jobs_list() -> None:
    """``srun`` hands every task the job's whole gres and nothing downstream picks a device by rank."""
    env = {"ROCR_VISIBLE_DEVICES": "0,1,2,3"}
    assert [baseline.hip_device(env, rank) for rank in range(6)] == ["0", "1", "2", "3", "0", "1"]
    assert baseline.hip_device({}, 0) is None, "a CPU column inherits no list and keeps the step's binding"


# ------------------------------------------------------------------------------------------ the CSV summary


def test_the_summary_counts_what_the_csv_says_not_the_exit_codes(tmp_path: pathlib.Path) -> None:
    """``run-framework`` exits 0 for a kernel a column merely does not support: ok needs status ok AND no failure."""
    shard = tmp_path / "c.rank0.csv"
    shard.write_text(
        f"{baseline.CSV_HEADER}\n"
        "c,S,float64,k1,c,ok,True,1.5,,\n"
        "c,S,float64,k2,c,ok,True,,unsupported,no such op\n"
        "c,S,float64,k3,c,ok,True,,tool_missing,ppcg\n"
        "c,S,float64,k4,c,crash,False,,,segfault, with a comma\n"
        "c,S,float64,k5,c,ok,True,,other,\n"
        "k6,S,float64,1.5,True\n"
    )
    assert baseline.summary_line("c", 0, shard, 2) == (
        "canon c rank 0: 6 rows -- 1 ok, 1 unsupported, 1 tool-missing, 2 crashed, 1 failed-in-column, 2 nonzero-exit"
    )


# -------------------------------------------------------------------------------------- the managed work dir


def test_a_managed_work_dir_is_derived_from_the_cache_root_never_from_scratch(tmp_path: pathlib.Path) -> None:
    """``out_root`` under ``HPCAGENT_BENCH_RUNS_ROOT`` is what makes a sweep manage it at all, and that root comes
    from cache_env.sh's ``JIT_CACHE_ROOT``, not from a bare ``$SCRATCH/<name>`` spelled out here."""
    opt = stub_checkout(tmp_path, WRITING_CLI)
    sweep = sweep_of(tmp_path, opt, ("fakekernel",))
    assert sweep.environ["HPCAGENT_BENCH_RUNS_ROOT"] == str(tmp_path / "jitcache" / "runs")
    assert sweep.managed
    assert not baseline.Sweep("c", tmp_path / "elsewhere", (), "S", opt, sweep.environ).managed


def test_the_per_rank_db_lands_under_the_work_dir_not_the_cwd(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run-framework``'s own default is repo-relative, so every rank's shard DB used to land in the CWD of a job
    launched from the checkout. A managed out_root never lets that happen."""
    monkeypatch.chdir(tmp_path)
    opt = stub_checkout(tmp_path, WRITING_CLI)
    sweep = sweep_of(tmp_path, opt, ("fakekernel",))
    assert baseline.run(sweep, jobs.Rank(0, 1)) == 0
    assert (sweep.out_root / "db" / "fakecol" / "hpcagent_bench.db").is_file()
    assert not (tmp_path / "hpcagent_bench0.db").exists()


def test_an_unmanaged_out_root_keeps_its_default_db_and_is_never_finished(tmp_path: pathlib.Path) -> None:
    """Every pre-existing ``$SCRATCH/canon-*`` directory is the documented hand-off to collect_canon.py: no DB
    redirection, no merge, nothing deleted."""
    opt = stub_checkout(tmp_path, WRITING_CLI)
    sweep = sweep_of(tmp_path, opt, ("fakekernel",), managed=False)
    assert baseline.run(sweep, jobs.Rank(0, 1)) == 0
    assert not (sweep.out_root / "db").exists()
    assert baseline.finish(sweep) == 0
    assert (sweep.out_root / "dacecache-fakecol").is_dir() and sweep.csv(0).is_file()


def test_a_verified_merge_deletes_the_build_tree_and_shard_db_but_keeps_the_csv(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CSV is the hand-off to collect_canon.py; the DaCe build tree and the now-redundant shard DB are the
    disposable bulk."""
    opt = stub_checkout(tmp_path, WRITING_CLI)
    sweep = sweep_of(tmp_path, opt, ("fakekernel",))
    baseline.run(sweep, jobs.Rank(0, 1))
    assert baseline.finish(sweep) == 0
    assert f"merged 1 row(s) into {sweep.environ['HPCAGENT_BENCH_RESULTS_DIR']}/canon.db" in capsys.readouterr().out
    assert not (sweep.out_root / "dacecache-fakecol").exists()
    assert not (sweep.out_root / "db" / "fakecol").exists()
    assert sweep.csv(0).is_file()
    assert {name: row["validated"] for name, row in canon_rows(sweep).items()} == {"fakekernel": "True"}


def test_a_merge_that_cannot_be_verified_keeps_every_column_artifact(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A directory where canon.db should be makes the real merge fail for a genuine reason: the sweep keeps the
    build tree, the shard DB and the CSV rather than losing the run's only record of them."""
    opt = stub_checkout(tmp_path, WRITING_CLI)
    sweep = sweep_of(tmp_path, opt, ("fakekernel",))
    (pathlib.Path(sweep.environ["HPCAGENT_BENCH_RESULTS_DIR"]) / "canon.db").mkdir(parents=True)
    baseline.run(sweep, jobs.Rank(0, 1))
    baseline.finish(sweep)
    assert "NOT verified" in capsys.readouterr().err
    assert (sweep.out_root / "dacecache-fakecol").is_dir()
    assert (sweep.out_root / "db" / "fakecol").is_dir()
    assert sweep.csv(0).is_file()


def test_a_column_that_produced_no_csv_still_finishes_as_a_clean_zero_row_merge(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    opt = stub_checkout(tmp_path, WRITING_CLI)
    sweep = sweep_of(tmp_path, opt, ())
    assert baseline.run(sweep, jobs.Rank(0, 1)) == 0
    assert baseline.finish(sweep) == 0
    assert "merged 0 row(s)" in capsys.readouterr().out


def test_a_column_stamps_the_dace_commit_into_its_build_record(tmp_path: pathlib.Path) -> None:
    """``record.build`` is NULL on a canon row unless the sweep stamps it, from the same commit that keys the PCH
    cache, through the knob ``run-framework`` already reads."""
    opt = stub_checkout(tmp_path, WRITING_CLI)
    sweep = sweep_of(tmp_path, opt, ("fakekernel",))
    baseline.run(sweep, jobs.Rank(0, 1))
    commit = subprocess.run(
        ["git", "-C", sweep.environ["DACE_DIR"], "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert pathlib.Path(f"{sweep.csv(0)}.record_build").read_text().strip() == f"dace {commit}"
    assert (sweep.out_root / "fakecol.rank0.dace").read_text().strip() == f"dace {commit}"


def test_a_caller_supplied_build_label_is_left_alone(tmp_path: pathlib.Path) -> None:
    opt = stub_checkout(tmp_path, WRITING_CLI)
    sweep = sweep_of(tmp_path, opt, ("fakekernel",), HPCAGENT_BENCH_RECORD_BUILD="extended-fork")
    baseline.run(sweep, jobs.Rank(0, 1))
    assert pathlib.Path(f"{sweep.csv(0)}.record_build").read_text().strip() == "extended-fork"


def test_a_stale_shard_from_an_old_roster_never_resurrects(tmp_path: pathlib.Path) -> None:
    """``run-framework`` APPENDS to an existing rank CSV, so once a roster or rank-count change moves a kernel to a
    different rank between two runs into ONE out_root, the file that kept its OLD row can receive a fresh append for
    another kernel and end up newer than the file holding the genuinely fresh row; the merge would keep the stale one.

    Exact repro: the old run's 3-rank roster put ``d`` on rank 0 (with ``a``); the new 4-rank roster moves it to
    rank 3 and leaves ``a`` on rank 0, which finishes last. Rotation in ``begin`` keeps the old row out of the file."""
    opt = stub_checkout(tmp_path, WRITING_CLI)
    sweep = sweep_of(tmp_path, opt, ("a", "b", "c", "d"))
    stale = sweep.csv(0)
    stale.write_text("kernel,preset,datatype,median_ms,validated\nd,fuzzed,float64,999.0,True\n")
    os.utime(stale, (1_700_000_000, 1_700_000_000))
    baseline.begin(sweep)
    for index in (3, 2, 1, 0):  # rank 0 last: its file gets the newest mtime
        assert baseline.run(sweep, jobs.Rank(index, 4)) == 0
    baseline.finish(sweep)
    rows = canon_rows(sweep)
    assert rows["d"]["median_ms"] == 1.5 and rows["a"]["median_ms"] == 1.5
    kept = "\n".join(path.read_text() for path in (sweep.out_root / ".stale-shards").rglob("*.csv"))
    assert "999.0" in kept, "rotated aside, not lost"
    assert "999.0" not in sweep.csv(0).read_text()


def test_begin_forgets_the_previous_runs_dace_labels(tmp_path: pathlib.Path) -> None:
    opt = stub_checkout(tmp_path, WRITING_CLI)
    sweep = sweep_of(tmp_path, opt, ("k",))
    label = sweep.out_root / "fakecol.rank0.dace"
    label.write_text("dace old\n")
    baseline.begin(sweep)
    assert not label.exists()


# --------------------------------------------------------------------------------------- the roster and phases


def test_a_kernels_file_narrows_the_roster_and_is_read_sorted(tmp_path: pathlib.Path) -> None:
    """One name per line, comments and blanks dropped, replacing the tag's roster; the loop order must not depend on
    how the file was written."""
    names = tmp_path / "owed.txt"
    names.write_text("kmp\n# a comment line\ndfa  # rerun\n\n")
    assert baseline.resolve_kernels("llr-focus40", "", names) == ("dfa", "kmp")


def test_an_unknown_kernel_name_is_refused_not_silently_dropped(tmp_path: pathlib.Path) -> None:
    names = tmp_path / "owed.txt"
    names.write_text("kmp\nnosuchkernel123\n")
    with pytest.raises(SystemExit, match="nosuchkernel123"):
        baseline.resolve_kernels("", "", names)


def test_a_missing_or_empty_kernels_file_is_refused(tmp_path: pathlib.Path) -> None:
    with pytest.raises(SystemExit, match="missing or empty"):
        baseline.resolve_kernels("", "", tmp_path / "does-not-exist.txt")
    blank = tmp_path / "blank.txt"
    blank.write_text("# nothing\n")
    with pytest.raises(SystemExit, match="names no kernels"):
        baseline.resolve_kernels("", "", blank)


def test_the_whole_tag_is_the_roster_when_nothing_narrows_it() -> None:
    assert len(baseline.resolve_kernels("llr-focus40", "", None)) == 40


def test_an_unknown_column_is_refused_before_a_node_is_held() -> None:
    with pytest.raises(SystemExit, match="unknown column"):
        baseline.check_column("nosuchcolumn")


def test_one_phase_of_a_multi_task_step_must_be_named(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``all`` runs begin, run and finish in one task; with several there is no barrier between them."""
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_PROCID", "1")
    code = subprocess.run(
        [sys.executable, "-m", "hpcagent_bench", "job", "baseline", "--column", "numba", "--out-root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "SLURM_NTASKS": "4", "SLURM_PROCID": "1"},
    )
    assert code.returncode != 0 and "run --phase begin" in code.stderr, code.stderr


def test_a_device_column_with_no_gpu_behind_it_says_so(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``ppcg_hip`` was once submitted with no GPU at all and reported a decline for every kernel."""
    opt = stub_checkout(tmp_path, WRITING_CLI)
    plain = sweep_of(tmp_path, opt, ("fakekernel",), managed=False)
    no_gpu = {k: v for k, v in plain.environ.items() if not k.endswith("_VISIBLE_DEVICES")}
    sweep = dataclasses.replace(plain, environ=no_gpu)
    device = dataclasses.replace(sweep, column="ppcg_hip")
    assert baseline.run(device, jobs.Rank(0, 1)) == 0
    assert "ppcg_hip builds for a device but the step holds no GPU" in capsys.readouterr().err
    assert baseline.run(sweep, jobs.Rank(0, 1)) == 0
    assert "holds no GPU" not in capsys.readouterr().err, "a CPU column is not told to hold GPUs"
