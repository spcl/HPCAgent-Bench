# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``experiments/recover_adhoc.py`` re-attributes an ``adhoc`` judge submission to its worker only on
proof, and ``extract_llr40 --retags`` extracts the row under that worker's run.

The shape reproduces 640097 (2026-09-17): gpt-oss-120b's MCP calls failed on the server-name
mismatch, it curled ``/submit`` without a run id, and the judge filed fuse_diamond's 37.53x under
``adhoc``. The driver stripes problem P onto judge rank ``P % n_judges``, and the run id -- not the
worker directory -- carries P.
"""

import csv
import importlib.util
import json
import pathlib
import sys
from types import ModuleType

import pytest

from hpcagent_bench.harness import recording

REPO = pathlib.Path(__file__).resolve().parents[1]

#: The judge's own response digits for fuse_diamond in 640097, echoed into the agent's tool result.
ECHOED = 37.52937768218807


def load(name: str, path: pathlib.Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="recover", scope="module")
def recover_fixture() -> ModuleType:
    return load("recover_adhoc", REPO / "experiments" / "recover_adhoc.py")


@pytest.fixture(name="extract", scope="module")
def extract_fixture() -> ModuleType:
    return load("extract_llr40_retags", REPO / "reproducibility" / "llr40" / "extract_llr40.py")


def judge_db(job: pathlib.Path, rank: int, rows: list[tuple[str, str, float]]) -> pathlib.Path:
    """``judge/rank-<rank>`` of ``job`` holding one submission per ``(run_id, benchmark, speedup)``."""
    path = job / "judge" / f"rank-{rank}" / f"hpcagent_bench{rank}.db"
    path.parent.mkdir(parents=True)
    conn = recording.connect(str(path))
    for ts, (run_id, benchmark, speedup) in enumerate(rows, start=1):
        conn.execute("INSERT OR IGNORE INTO benchmarks (name) VALUES (?)", (benchmark,))
        conn.execute(
            "INSERT INTO submissions (run_id, ts, benchmark, preset, datatype, source_mode, baseline, speedup) "
            "VALUES (?, ?, ?, 'fuzzed', 'float64', 'restricted', 'numba', ?)",
            (run_id, ts, benchmark, speedup),
        )
    conn.commit()
    conn.close()
    return path


def worker(job: pathlib.Path, problem_id: int, index: int, benchmark: str, echo: str = "") -> str:
    """A worker directory as the driver leaves it; ``index`` is the problem's place in the job's list."""
    run_id = f"llr-oss120b-c.n0.p{index}.w{index}"
    path = job / "agents" / "node-0" / f"problem-{problem_id}-worker-{index}"
    path.mkdir(parents=True)
    env = {"HPCAGENT_BENCH_RUN_ID": run_id, "HPCAGENT_BENCH_OPTIMIZER": "openai/gpt-oss-120b"}
    (path / "mcp.json").write_text(json.dumps({"mcpServers": {"hpcagent-bench": {"env": env}}}), encoding="utf-8")
    kernel = f"loop_level_reasoning/{benchmark}/{benchmark}"
    (path / "prompt.txt").write_text(f"Optimize benchmark kernel {kernel}. Target language: c.\n", encoding="utf-8")
    (path / "claude.log").write_text(echo, encoding="utf-8")
    return run_id


def tool_result(speedup: float) -> str:
    """The judge's /submit body as a transcript line carries it: JSON inside a JSON string."""
    body = json.dumps({"correct": True, "speedup": speedup})
    return json.dumps({"type": "user", "message": {"content": [{"type": "tool_result", "content": body}]}}) + "\n"


def test_an_echoed_speedup_proves_the_worker(recover: ModuleType, tmp_path: pathlib.Path) -> None:
    job = tmp_path / "640097"
    run_id = worker(job, 4, 4, "fuse_diamond", echo=tool_result(ECHOED))
    worker(job, 5, 5, "tsvc_2_s115")
    db = judge_db(job, 0, [("adhoc", "fuse_diamond", ECHOED)])
    judge_db(job, 1, [])
    retags, owed = recover.recover([db])
    assert owed == []
    assert [(r["run_id"], r["evidence"], r["worker"]) for r in retags] == [
        (run_id, "transcript", "agents/node-0/problem-4-worker-4")
    ]


def test_the_judge_rank_separates_repeats_of_one_kernel(recover: ModuleType, tmp_path: pathlib.Path) -> None:
    """REPEAT=3 puts three workers on one kernel; only the one striped onto the row's rank ran it there.
    The index comes from the run id: an owed rerun's worker directory names the problem ID (11)."""
    job = tmp_path / "641678"
    worker(job, 10, 2, "tsvc_2_s115")
    wanted = worker(job, 11, 3, "tsvc_2_s115")
    worker(job, 12, 4, "tsvc_2_s115")
    judge_db(job, 0, [])
    db = judge_db(job, 1, [("adhoc", "tsvc_2_s115", 2.9662943113415983)])
    retags, owed = recover.recover([db])
    assert owed == []
    assert [(r["run_id"], r["evidence"]) for r in retags] == [(wanted, "kernel_unique")]


def test_two_candidates_without_an_echo_stay_owed(recover: ModuleType, tmp_path: pathlib.Path) -> None:
    job = tmp_path / "640085"
    worker(job, 0, 0, "warpx_boris_push")
    worker(job, 2, 2, "warpx_boris_push")
    db = judge_db(job, 0, [("adhoc", "warpx_boris_push", 10.554666120949358)])
    judge_db(job, 1, [])
    retags, owed = recover.recover([db])
    assert retags == []
    assert [(r["job"], r["benchmark"]) for r in owed] == [("640085", "warpx_boris_push")]


def test_a_round_speedup_is_no_proof(recover: ModuleType, tmp_path: pathlib.Path) -> None:
    """A promoted 1.0 is in every transcript that ever printed a baseline-speed grade; two workers
    echo it here, so neither is proven and the two candidates leave the row owed."""
    job = tmp_path / "632999"
    worker(job, 0, 0, "tsvc_2_s235", echo=tool_result(1.0))
    worker(job, 1, 1, "tsvc_2_s235", echo=tool_result(1.0))
    db = judge_db(job, 0, [("adhoc", "tsvc_2_s235", 1.0)])
    retags, owed = recover.recover([db])
    assert (retags, len(owed)) == ([], 1)


def test_a_row_with_its_own_run_id_is_left_alone(recover: ModuleType, tmp_path: pathlib.Path) -> None:
    job = tmp_path / "639211"
    run_id = worker(job, 0, 0, "s000", echo=tool_result(ECHOED))
    db = judge_db(job, 0, [(run_id, "s000", ECHOED)])
    assert recover.recover([db]) == ([], [])


def test_extract_reads_a_retagged_row_under_its_run(
    recover: ModuleType, extract: ModuleType, tmp_path: pathlib.Path
) -> None:
    """End to end: the recovered CSV makes the extractor file the adhoc row under the worker's arm,
    so an arm filter keeps it, and leaves the evidence on the row."""
    job = tmp_path / "640097"
    run_id = worker(job, 4, 4, "fuse_diamond", echo=tool_result(ECHOED))
    db = judge_db(job, 0, [("adhoc", "fuse_diamond", ECHOED)])
    out = tmp_path / "retags.csv"
    assert recover.main([str(tmp_path), f"--out={out}"]) == 0
    with out.open(encoding="utf-8", newline="") as handle:
        assert [row["run_id"] for row in csv.DictReader(handle)] == [run_id]

    database = extract.Database(db.resolve(), "640097", job, "640097")
    plain = extract.read_db(database, frozenset(), "llr-oss120b", frozenset(), 0)
    assert [row for row in plain.observations if row["record"] == "submission"] == []

    result = extract.read_db(database, frozenset(), "llr-oss120b", frozenset(), 0, extract.load_retags([str(out)]))
    rows = [row for row in result.observations if row["record"] == "submission"]
    assert [(r["run_id"], r["arm"], r["optimizer"], r["retagged"]) for r in rows] == [
        (run_id, "llr-oss120b-c", "openai/gpt-oss-120b", "transcript")
    ]
    assert (rows[0]["problem_index"], rows[0]["speedup"]) == ("4", ECHOED)
