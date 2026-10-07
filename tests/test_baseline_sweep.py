# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``hpcagent-bench job baseline``: one compiler column over a tag, split over the tasks of a step.

The kernels run through a STUB image interpreter whose ``-m hpcagent_bench.cli`` is a small script (a
``run-framework`` that records one canon row, one that hangs, one that reports its rlimits), so no dace tree, build
toolchain or benchmark registry is needed: what is tested is the sweep's own wiring -- the rank split, the wall
and heap caps, the timeout row, the managed work dir (shard DB redirection, an earlier run's rows dropped, the
build tree cleared) and the tag resolution -- not ``run-framework``'s row correctness.
"""

import dataclasses
import os
import pathlib
import re
import resource
import shutil
import subprocess
import sys

import pytest

from hpcagent_bench import cpf_canonical, paths
from hpcagent_bench.cluster import baseline, jobs
from hpcagent_bench.support.collect import canon_db

#: A ``run-framework`` that records one canon row per call, as the real one does, and touches its shard DB.
WRITING_CLI = """
import os
import pathlib
import sys

from hpcagent_bench.support.collect import canon_db

if sys.argv[1] == "preflight":
    raise SystemExit(0)
args = sys.argv[2:]
value = lambda flag: args[args.index(flag) + 1]
row = {
    "run": value("--canon-run"), "column": value("-f"), "kernel": value("-b"), "preset": value("-p"),
    "datatype": "float64", "median_ms": 1.5, "validated": "True", "status": "ok", "failure": "",
    "build": os.environ.get("HPCAGENT_BENCH_RECORD_BUILD") or None,
}
canon_db.record(pathlib.Path(value("--canon-db")), [row])
db = os.environ.get("HPCAGENT_BENCH_RECORD_DB_PATH")
if db:
    pathlib.Path(db).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(db).touch()
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
    """A checkout whose ``helpers/scripts/`` are the real ones (the cache roots derive from ``JIT_CACHE_ROOT``) and whose
    image interpreter answers ``-m hpcagent_bench.cli`` from ``cli``."""
    opt = root / "opt"
    shutil.copytree(
        paths.ROOT / "helpers" / "scripts", opt / "helpers" / "scripts", ignore=shutil.ignore_patterns("checks", "*.py")
    )
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
    """The step's environment: a private cache root, the stub interpreter."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"HPCAGENT_BENCH_CACHE", "HPCAGENT_BENCH_RUNS_ROOT", "HPCAGENT_BENCH_RESULTS_DIR"}
        and key not in {"HPCAGENT_BENCH_RECORD_DB_PATH", "HPCAGENT_BENCH_RECORD_BUILD", "SLURM_PROCID", "SLURM_NTASKS"}
    }
    env.update(JIT_CACHE_ROOT=str(root / "jitcache"), HPCAGENT_BENCH_IMAGE_PYTHON=str(root / "image-python"))
    env.update(CANON_OPT_REPORTS="0", SLURM_CPUS_PER_TASK="2", **extra)
    return env


def sweep_of(root: pathlib.Path, opt: pathlib.Path, kernels: tuple[str, ...], managed: bool = True, **extra: str):
    """A sweep of column ``fakecol``; ``managed`` puts its out_root under the cache's runs root."""
    env = baseline.cache_environment(opt, environment(root, **extra))
    out_root = (pathlib.Path(env["HPCAGENT_BENCH_RUNS_ROOT"]) / "canon" / "unit") if managed else root / "out"
    out_root.mkdir(parents=True)
    return baseline.Sweep("fakecol", out_root, kernels, "fuzzed", opt, env)


def canon_rows(sweep: baseline.Sweep) -> dict[str, dict]:
    return {str(row["kernel"]): row for row in canon_db.read(sweep.canon(), column=sweep.column)}


# the rank split


@pytest.mark.parametrize("size", [1, 2, 3, 4, 7])
def test_the_ranks_split_a_tag_disjointly_and_completely(size: int) -> None:
    tag_kernels = [f"k{index}" for index in range(10)]
    shares = [jobs.share(tag_kernels, jobs.Rank(index, size)) for index in range(size)]
    assert sorted(name for share in shares for name in share) == sorted(tag_kernels)
    assert all(not set(a) & set(b) for i, a in enumerate(shares) for b in shares[i + 1 :])


def test_a_rank_with_no_kernels_is_a_no_op_that_needs_no_dace_tree(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """3 kernels over 4 ranks leave rank 3 empty. It must neither crash nor need a working checkout or a results
    DB: a nonzero task exit makes ``srun`` tear down the sibling ranks that already have rows."""
    out_root = tmp_path / "out"
    out_root.mkdir()
    empty = tmp_path / "opt"
    empty.mkdir()
    sweep = baseline.Sweep("stubcol", out_root, ("a", "b", "c"), "fuzzed", empty, {})
    assert baseline.run(sweep, jobs.Rank(3, 4)) == 0
    out = capsys.readouterr().out
    assert "no kernels assigned" in out and "0 rows" in out


# the per-kernel caps


def test_a_hung_kernel_is_killed_and_recorded_as_a_timeout_row_not_a_silent_gap(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stalled ``run-framework`` would hold a rank until the job's own limit killed it, and every kernel after it
    would get no row on any rank."""
    opt = stub_checkout(tmp_path, HANGING_CLI)
    sweep = sweep_of(tmp_path, opt, ("onlykernel",), managed=False, CANON_KERNEL_TIMEOUT_SEC="2")
    assert baseline.run(sweep, jobs.Rank(0, 1)) == 0
    out = capsys.readouterr().out
    assert "FAILED onlykernel (wall timeout after 2s" in out
    row = canon_rows(sweep)["onlykernel"]
    assert (row["run"], row["status"], row["failure"]) == (sweep.run_label, "timeout", "timeout")
    # status != ok, so the row is tallied as crashed, and the nonzero-exit counter agrees with it.
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
    heap = baseline.DEFAULT_KERNEL_MEM_KB * 1024  # CANON_KERNEL_MEM_KB unset: the default cap
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


def test_the_summary_counts_what_the_rows_say_not_the_exit_codes() -> None:
    """``run-framework`` exits 0 for a kernel a column merely does not support: ok needs status ok AND no failure."""
    rows: list[dict[str, object]] = [
        {"status": "ok", "failure": ""},
        {"status": "ok", "failure": "unsupported"},
        {"status": "ok", "failure": "tool_missing"},
        {"status": "crash", "failure": ""},
        {"status": "ok", "failure": "other"},
        {"status": "timeout", "failure": "timeout"},
    ]
    assert baseline.summary_line("c", 0, rows, 2) == (
        "canon c rank 0: 6 rows -- 1 ok, 1 unsupported, 1 tool-missing, 2 crashed, 1 failed-in-column, 2 nonzero-exit"
    )


# the managed work dir


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
    """``run-framework``'s own default is repo-relative, so every rank's shard DB would land in the CWD of a job
    launched from the checkout. A managed out_root never lets that happen."""
    monkeypatch.chdir(tmp_path)
    opt = stub_checkout(tmp_path, WRITING_CLI)
    sweep = sweep_of(tmp_path, opt, ("fakekernel",))
    assert baseline.run(sweep, jobs.Rank(0, 1)) == 0
    assert (sweep.out_root / "db" / "fakecol" / "hpcagent_bench.db").is_file()
    assert not (tmp_path / "hpcagent_bench0.db").exists()


def test_an_unmanaged_out_root_keeps_its_default_db_and_is_never_finished(tmp_path: pathlib.Path) -> None:
    """Any directory outside the runs root accumulates: no DB redirection, its rows recorded, nothing deleted."""
    opt = stub_checkout(tmp_path, WRITING_CLI)
    sweep = sweep_of(tmp_path, opt, ("fakekernel",), managed=False)
    assert baseline.run(sweep, jobs.Rank(0, 1)) == 0
    assert not (sweep.out_root / "db").exists()
    assert baseline.finish(sweep) == 0
    assert (sweep.out_root / "dacecache-fakecol").is_dir()
    assert set(canon_rows(sweep)) == {"fakekernel"}


def test_finish_clears_the_build_tree_and_shard_db_and_keeps_the_rows(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The DaCe build tree and the shard DB are the disposable bulk; the canon row is the result."""
    opt = stub_checkout(tmp_path, WRITING_CLI)
    sweep = sweep_of(tmp_path, opt, ("fakekernel",))
    baseline.run(sweep, jobs.Rank(0, 1))
    assert baseline.finish(sweep) == 0
    assert f"1 row(s) in {sweep.canon()}" in capsys.readouterr().out
    assert not (sweep.out_root / "dacecache-fakecol").exists()
    assert not (sweep.out_root / "db" / "fakecol").exists()
    assert {name: row["validated"] for name, row in canon_rows(sweep).items()} == {"fakekernel": "True"}


def test_a_column_that_recorded_no_row_still_finishes(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    opt = stub_checkout(tmp_path, WRITING_CLI)
    sweep = sweep_of(tmp_path, opt, ())
    assert baseline.run(sweep, jobs.Rank(0, 1)) == 0
    assert baseline.finish(sweep) == 0
    assert "0 row(s)" in capsys.readouterr().out


def test_a_column_stamps_the_dace_commit_into_its_rows(tmp_path: pathlib.Path) -> None:
    """A canon row's ``build`` is the commit that keys the PCH cache, through the knob ``run-framework`` reads."""
    opt = stub_checkout(tmp_path, WRITING_CLI)
    sweep = sweep_of(tmp_path, opt, ("fakekernel",))
    baseline.run(sweep, jobs.Rank(0, 1))
    assert canon_rows(sweep)["fakekernel"]["build"] == f"dace {cpf_canonical.dace_commit()[:9]}"


def test_a_caller_supplied_build_label_is_left_alone(tmp_path: pathlib.Path) -> None:
    opt = stub_checkout(tmp_path, WRITING_CLI)
    sweep = sweep_of(tmp_path, opt, ("fakekernel",), HPCAGENT_BENCH_RECORD_BUILD="extended-fork")
    baseline.run(sweep, jobs.Rank(0, 1))
    assert canon_rows(sweep)["fakekernel"]["build"] == "extended-fork"


def test_a_kernel_an_earlier_run_reached_never_resurrects(tmp_path: pathlib.Path) -> None:
    """A re-run into one work dir with a narrower tag must not keep the earlier run's row for a kernel it never
    reaches, and a kernel it does reach carries only this run's row: ``begin`` drops the earlier rows."""
    opt = stub_checkout(tmp_path, WRITING_CLI)
    sweep = sweep_of(tmp_path, opt, ("a", "b"))
    stale = {"run": sweep.run_label, "column": sweep.column, "preset": "fuzzed", "datatype": "float64"}
    canon_db.record(sweep.canon(), [{**stale, "kernel": k, "median_ms": 999.0, "validated": "True"} for k in "abz"])
    baseline.begin(sweep)
    for index in (1, 0):
        assert baseline.run(sweep, jobs.Rank(index, 2)) == 0
    rows = canon_rows(sweep)
    assert set(rows) == {"a", "b"}
    assert rows["a"]["median_ms"] == 1.5 and rows["b"]["median_ms"] == 1.5


# the tag and phases


def test_a_kernels_file_narrows_the_tag_and_is_read_sorted(tmp_path: pathlib.Path) -> None:
    """One name per line, comments and blanks dropped, replacing the tag's tag; the loop order must not depend on
    how the file was written."""
    names = tmp_path / "owed.txt"
    names.write_text("kmp\n# a comment line\ndfa  # rerun\n\n")
    assert baseline.resolve_kernels("llr40", "", names) == ("dfa", "kmp")


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


def test_the_whole_tag_is_the_tag_when_nothing_narrows_it() -> None:
    assert len(baseline.resolve_kernels("llr40", "", None)) == 40


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
    """A device column on a step with no GPU says so instead of reporting a decline for every kernel."""
    opt = stub_checkout(tmp_path, WRITING_CLI)
    plain = sweep_of(tmp_path, opt, ("fakekernel",), managed=False)
    no_gpu = {k: v for k, v in plain.environ.items() if not k.endswith("_VISIBLE_DEVICES")}
    sweep = dataclasses.replace(plain, environ=no_gpu)
    device = dataclasses.replace(sweep, column="ppcg_hip")
    assert baseline.run(device, jobs.Rank(0, 1)) == 0
    assert "ppcg_hip builds for a device but the step holds no GPU" in capsys.readouterr().err
    assert baseline.run(sweep, jobs.Rank(0, 1)) == 0
    assert "holds no GPU" not in capsys.readouterr().err, "a CPU column is not told to hold GPUs"
