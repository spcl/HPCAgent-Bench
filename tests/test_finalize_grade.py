# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Finalize grading: the final grade (mw4x5) of one fast-graded agent job, planned at the
start of the finalize job chained on it (experiments/finalize_grade.sbatch -> finalize_grade_owed.py --job).

The fixtures are real judge shards (recording.connect's schema, sources in the content store) under
a run root laid out as ``<campaign>/<job>/judge/rank-0/``, real ``regrade_tasks`` shards, and
squeue / sacct stand-ins on PATH answering in the formats finalize_grade_owed.py asks for.
"""

import json
import os
import pathlib
import shutil
import sqlite3
import subprocess
import sys

import pytest

from hpcagent_bench.harness import recording, timing

REPO = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"
ARM = "scicomp-focus40-qwen38-c"
JOB = "700100"


def judge_shard(runs: pathlib.Path, job: str, rows: list[tuple[str, str, int, float]]) -> pathlib.Path:
    """One judge shard of ``job`` holding (run id, kernel, ts, speedup) submissions, each source stored."""
    db = runs / "scicomp-focus40-20260924" / job / "judge" / "rank-0" / "hpcagent_bench0.db"
    conn = recording.connect(str(db))
    conn.execute("PRAGMA foreign_keys = OFF")
    for run_id, kernel, ts, speedup in rows:
        benchmark = f"scientific_computing/{kernel}"
        recording.store_source(
            conn,
            f"/* {run_id} {ts} */",
            benchmark,
            run_id=run_id,
            ts=ts,
            language="c",
            store_dir=str(db.parent / "hpcagent_bench0_prompts"),
        )
        conn.execute(
            "INSERT INTO submissions (run_id, ts, benchmark, preset, datatype, source_mode, baseline, speedup, timing_reduction)"
            " VALUES (?, ?, ?, 'XL', 'float64', 'restricted', 'c', ?, 'mwd-final')",
            (run_id, ts, benchmark, speedup),
        )
    conn.commit()
    conn.close()
    return db


def final_grade(regrades: pathlib.Path, db: pathlib.Path, run_id: str, kernel: str, ts: int) -> None:
    """A regrade shard that GRADED this submission under the final rule."""
    path = regrades / "mwd-final-regrades-vtest" / "out" / "regrade-cells-0.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS regrade_tasks (db TEXT, run_id TEXT, benchmark TEXT, ts_ms INTEGER, status TEXT,"
            " timing_reduction TEXT, score_rule TEXT, regrade_ts INTEGER, node TEXT, commit_sha TEXT)"
        )
        conn.execute(
            "INSERT INTO regrade_tasks VALUES (?, ?, ?, ?, 'graded', ?, 'mw4x5', 1, 'nid1', 'abc')",
            (str(db), run_id, f"scientific_computing/{kernel}", ts, timing.FINAL_GRADE_REDUCTION),
        )


def stub(bin_dir: pathlib.Path, name: str, body: str) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    path = bin_dir / name
    path.write_text(f"#!/usr/bin/env bash\n{body}\n")
    path.chmod(0o755)


def queue(tmp_path: pathlib.Path, squeue: str = "", sacct: dict[str, str] | None = None) -> pathlib.Path:
    """squeue answering ``squeue`` (``%i|%j|%T|%M|%l`` lines) and sacct answering each job's
    ``WorkDir|SubmitLine``."""
    bin_dir = tmp_path / "bin"
    (tmp_path / "squeue.txt").write_text(squeue)
    stub(bin_dir, "squeue", f'cat "{tmp_path / "squeue.txt"}"')
    for job, line in (sacct or {}).items():
        (tmp_path / f"sacct-{job}.txt").write_text(line + "\n")
    stub(
        bin_dir,
        "sacct",
        f'j=""; while [ $# -gt 0 ]; do [ "$1" = -j ] && j=$2; shift; done; cat "{tmp_path}/sacct-$j.txt" 2>/dev/null || true',
    )
    return bin_dir


def plan(
    tmp_path: pathlib.Path, job: str = JOB, own_job: str = "", exempt: pathlib.Path | None = None
) -> list[tuple[str, str, int]]:
    """(run id, kernel, ts) of every item ``finalize_grade_owed.py --job`` plans, in worklist order."""
    bin_dir = tmp_path / "bin"
    if not bin_dir.exists():
        queue(tmp_path)
    out = tmp_path / "plan" / "worklist.jsonl"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "SLURM_JOB_ID": own_job,
    }
    argv = [sys.executable, str(EXPERIMENTS / "finalize_grade_owed.py"), "--job", job]
    argv += ["--worklist-out", str(out)]
    argv += ["--runs", str(tmp_path / "runs"), "--regrades", str(tmp_path / "regrades" / "mwd-final-regrades-*")]
    argv += ["--sbatch-dir", str(tmp_path), "--exempt", str(exempt or tmp_path / "no-exempt.tsv")]
    done = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 0, done.stderr
    items = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(item["run_id"], item["benchmark"].rsplit("/", 1)[-1], item["ts_ms"]) for item in items]


def run_id(problem: int, arm: str = ARM) -> str:
    return f"{arm}.n0.p{problem}.w{problem}"


def test_the_plan_is_the_jobs_own_latest_credited_answers_with_no_final_grade(tmp_path: pathlib.Path) -> None:
    """Graded kernels, uncredited ones and another job's answers are not this job's finalize work."""
    runs = tmp_path / "runs"
    db = judge_shard(
        runs,
        JOB,
        [(run_id(1), "lulesh", 100, 2.0), (run_id(2), "hpccg", 110, 3.0), (run_id(3), "minife", 120, 0.0)],
    )
    judge_shard(runs, "700200", [(run_id(4, "scicomp-focus40-qwen38-hip"), "lulesh", 130, 2.0)])
    final_grade(tmp_path / "regrades", db, run_id(2), "hpccg", 110)
    assert plan(tmp_path) == [(run_id(1), "lulesh", 100)]


def test_a_superseded_submission_is_not_planned(tmp_path: pathlib.Path) -> None:
    """The final grade belongs to the arm's NEWEST credited answer: an older one in the same episode,
    or one a later job of the same arm (its -clean rerun included) answered again, needs none."""
    runs = tmp_path / "runs"
    judge_shard(
        runs, JOB, [(run_id(1), "lulesh", 100, 2.0), (run_id(1), "lulesh", 150, 2.5), (run_id(2), "hpccg", 100, 3.0)]
    )
    judge_shard(runs, "700300", [(run_id(7, f"{ARM}-clean"), "hpccg", 200, 1.5)])
    assert plan(tmp_path) == [(run_id(1), "lulesh", 150)]


def test_an_item_a_live_regrade_job_holds_is_not_planned(tmp_path: pathlib.Path) -> None:
    """A queued regrade job whose worklist holds the answer grades it; finalizing it too would grade twice."""
    runs = tmp_path / "runs"
    db = judge_shard(runs, JOB, [(run_id(1), "lulesh", 100, 2.0), (run_id(2), "hpccg", 100, 3.0)])
    held = {"db": str(db), "run_id": run_id(1), "benchmark": "scientific_computing/lulesh", "ts_ms": 100, "arm": ARM}
    held |= {
        "language": "c",
        "source_mode": "restricted",
        "source": "s.c",
        "device_source": "",
        "final": True,
        "env": {},
    }
    (tmp_path / "held.jsonl").write_text(json.dumps(held) + "\n")
    queue(
        tmp_path,
        "800001|regrade-v9-09251100-00|PENDING|0:00|3:00:00\n",
        {"800001": f"{tmp_path}|sbatch --nodes=1 regrade.sbatch held.jsonl out-held finalize"},
    )
    assert plan(tmp_path) == [(run_id(2), "hpccg", 100)]


def test_a_waiting_finalize_job_of_the_same_agent_job_holds_it_whole(tmp_path: pathlib.Path) -> None:
    """Of two finalize jobs of one agent job, the one with the lower id plans it; the other leaves it."""
    judge_shard(tmp_path / "runs", JOB, [(run_id(1), "lulesh", 100, 2.0)])
    queue(
        tmp_path,
        f"800001|regrade-finalize-{JOB}|PENDING|0:00|3:00:00\n",
        {"800001": f"{tmp_path}|sbatch --dependency=afterany:{JOB} finalize_grade.sbatch {JOB}"},
    )
    assert plan(tmp_path, own_job="800002") == []
    assert plan(tmp_path, own_job="800000") == [(run_id(1), "lulesh", 100)]


def test_the_planning_finalize_job_does_not_hold_its_own_agent_job(tmp_path: pathlib.Path) -> None:
    """At its start the finalize job is itself a running regrade job with no worklist yet."""
    judge_shard(tmp_path / "runs", JOB, [(run_id(1), "lulesh", 100, 2.0)])
    queue(
        tmp_path,
        f"800001|regrade-finalize-{JOB}|RUNNING|0:05|3:00:00\n",
        {"800001": f"{tmp_path}|sbatch --dependency=afterany:{JOB} finalize_grade.sbatch {JOB}"},
    )
    assert plan(tmp_path, own_job="800001") == [(run_id(1), "lulesh", 100)]


def test_a_live_exempt_answer_is_not_planned(tmp_path: pathlib.Path) -> None:
    """A submission on the exemption list keeps its live grade as the final one (2026-09-25 USER)."""
    db = judge_shard(tmp_path / "runs", JOB, [(run_id(1), "lulesh", 100, 2.0), (run_id(2), "hpccg", 100, 3.0)])
    exempt = tmp_path / "final-grade-exempt.tsv"
    exempt.write_text(
        "# test\njob\trun_id\tbenchmark\tts_ms\tarm\tdb\treason\n"
        f"{JOB}\t{run_id(1)}\tscientific_computing/lulesh\t100\t{ARM}\t{db}\tsource deleted\n"
    )
    assert plan(tmp_path, exempt=exempt) == [(run_id(2), "hpccg", 100)]


@pytest.mark.parametrize("arm", ["mlscale-qwen38-hip-gemmhint", f"{ARM}-smoke", "scicomp-focus40-glm53-c"])
def test_arms_with_another_final_grade_or_none_are_not_planned(tmp_path: pathlib.Path, arm: str) -> None:
    """ML scaling answers are finalized by mlscale-grade.sbatch; smoke and off-board arms are no paper data."""
    judge_shard(tmp_path / "runs", JOB, [(run_id(1, arm), "lulesh", 100, 2.0)])
    assert plan(tmp_path) == []


def finalize_tree(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """A checkout whose experiments/ is a copy (regrade.sbatch replaced by a recorder) and whose
    package is this one: finalize_grade.sbatch run as Slurm runs it, from experiments/."""
    repo = tmp_path / "repo"
    shutil.copytree(
        EXPERIMENTS,
        repo / "experiments",
        ignore=shutil.ignore_patterns("mwd-final-*", "*.out", "*.err", "owed", ".rendered", "__pycache__"),
    )
    (repo / "hpcagent_bench").symlink_to(REPO / "hpcagent_bench")
    (repo / "scripts").symlink_to(REPO / "scripts")
    (repo / "experiments" / "regrade.sbatch").write_text('printf "%s\\n" "$@" > "${STUB_MARKERS}/regrade-argv.txt"\n')
    return repo, repo / "experiments"


def run_finalize(tmp_path: pathlib.Path) -> subprocess.CompletedProcess[str]:
    _, experiments = finalize_tree(tmp_path)
    bin_dir = queue(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "SLURM_SUBMIT_DIR": str(experiments),
        "SLURM_JOB_ID": "800009",
        "SCRATCH": str(tmp_path),
        "HPCAGENT_BENCH_HOST_PYTHON": sys.executable,
        "STUB_MARKERS": str(tmp_path),
    }
    env.pop("HPCAGENT_BENCH_REPO", None)
    return subprocess.run(
        ["bash", str(EXPERIMENTS / "finalize_grade.sbatch"), JOB],
        cwd=experiments,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_an_empty_plan_ends_the_finalize_job_at_once(tmp_path: pathlib.Path) -> None:
    (tmp_path / "hpcagent-bench-runs").mkdir()
    done = run_finalize(tmp_path)
    assert done.returncode == 0, done.stderr
    assert "owes no final grade" in done.stdout
    assert not (tmp_path / "regrade-argv.txt").exists(), "nothing to grade, no grading step"


def test_the_finalize_job_grades_its_plan_per_cell_under_the_final_rule(tmp_path: pathlib.Path) -> None:
    """Its own worklist, graded as regrade.sbatch `finalize`, into a directory every
    mwd-final-regrades-* glob reads."""
    judge_shard(tmp_path / "hpcagent-bench-runs", JOB, [(run_id(1), "lulesh", 100, 2.0)])
    done = run_finalize(tmp_path)
    assert done.returncode == 0, done.stderr
    where = tmp_path / "repo" / "experiments" / "mwd-final-regrades-finalize" / f"{JOB}-800009"
    argv = (tmp_path / "regrade-argv.txt").read_text().splitlines()
    assert argv == [str(where / "worklist.jsonl"), str(where / "cells"), "finalize"]
    (item,) = [json.loads(line) for line in (where / "worklist.jsonl").read_text().splitlines()]
    assert (item["run_id"], item["ts_ms"], item["job"]) == (run_id(1), 100, JOB)
