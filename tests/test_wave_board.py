# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The wave board's arm identity, status rule and data embedding: the persistent experiment-status page
reports these, and a wrong one shows a finished experiment as owed or an owed one as finished."""

import importlib.util
import json
import os
import pathlib
import sqlite3
import sys
import types

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "wave_board.py"
MODELS = ("kimi27sglang", "oss120b", "qwen38", "glm53")


@pytest.fixture(scope="module")
def board() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("wave_board", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        (
            "gpu-llr-focus40-kimi27sglang-c-openmp-skills",
            ("gpu-llr-focus40", "kimi27sglang", "c-openmp-skills", False),
        ),
        ("llrblind-kimi27sglang-fortran-skills", ("llrblind", "kimi27sglang", "fortran-skills", False)),
        ("scicomp-dc-qwen38-cpfsrc", ("scicomp-dc", "qwen38", "cpfsrc", False)),
        ("cpf-llr-focus40-qwen38-c-cpf-clean", ("cpf-llr-focus40", "qwen38", "c-cpf", True)),
        ("cpf-llr-focus40-oss120b-c-clean", ("cpf-llr-focus40", "oss120b", "c", True)),
        (
            "scicomp-perf-playbook-gpu-oss120b-hip-perf-playbook-amd",
            ("scicomp-perf-playbook-gpu", "oss120b", "hip-perf-playbook-amd", False),
        ),
        (
            "scicomp-perf-playbook-qwen38-perf-playbook-cpu",
            ("scicomp-perf-playbook", "qwen38", "perf-playbook-cpu", False),
        ),
    ],
)
def test_an_arm_name_splits_into_its_campaign_model_variant_and_clean_flag(
    board: types.ModuleType, arm: str, expected: tuple[str, str, str, bool]
) -> None:
    """A clean re-run reads as the SAME variant as the arm it re-ran: the suffix is a flag, not a
    condition, and a variant that carried it would split one condition into two board rows."""
    assert board.split_arm(arm, MODELS) == expected


def test_the_harness_smoke_is_back_on_the_board(board: types.ModuleType) -> None:
    """Agent Harness Comparison@20's smoke (CAMPAIGNS 2026-09-18) is reported again: it disappeared
    only because no CAMPAIGNS entry matched harness-focus20-smoke-<model>-<harness>, not because it
    was meant to stay off the board."""
    assert board.campaign_of("harness-focus20-smoke-qwen38-claude-autokernel") == "harness-focus20-smoke"
    assert board.split_arm("harness-focus20-smoke-oss120b-optimas", MODELS) == (
        "harness-focus20-smoke",
        "oss120b",
        "optimas",
        False,
    )


def test_scicomp_perf_playbook_gpu_wins_over_its_cpu_prefix(board: types.ModuleType) -> None:
    """campaign_of takes the LONGEST matching prefix: "scicomp-perf-playbook-gpu" must win over
    "scicomp-perf-playbook", or a GPU arm's rest-of-name would start "gpu-<model>-..." and the model
    would never match (submit-scicomp-perf-playbook.sh's DEVICE=gpu knob, mirroring submit-scicomp-dc.sh)."""
    assert board.campaign_of("scicomp-perf-playbook-gpu-oss120b-hip-perf-playbook-amd") == "scicomp-perf-playbook-gpu"
    row_campaign, model, variant, clean = board.split_arm(
        "scicomp-perf-playbook-gpu-qwen38-hip-perf-playbook-amd", MODELS
    )
    assert (row_campaign, model, variant, clean) == (
        "scicomp-perf-playbook-gpu",
        "qwen38",
        "hip-perf-playbook-amd",
        False,
    )


def test_a_job_outside_every_campaign_has_no_campaign(board: types.ModuleType) -> None:
    """Agent gate jobs share the queue; the board must not invent an experiment for them."""
    assert board.campaign_of("tc-suite-ab") == ""


@pytest.mark.parametrize(
    ("done", "roster", "states", "expected"),
    [
        (40, 40, ["COMPLETED", "PENDING"], "running"),
        (0, 40, ["COMPLETED", "RUNNING"], "running"),
        (40, 40, ["FAILED", "COMPLETED"], "complete"),
        (39, 40, ["COMPLETED"], "incomplete"),
        (0, 0, ["COMPLETED"], "incomplete"),
    ],
)
def test_an_arm_is_running_before_its_coverage_decides(
    board: types.ModuleType, done: int, roster: int, states: list[str], expected: str
) -> None:
    assert board.arm_status(done, roster, states) == expected


def test_the_embedded_data_survives_a_value_that_closes_a_script_element(board: types.ModuleType) -> None:
    """A value carrying ``</script>`` would end the data block early and the page would parse nothing."""
    data = {"generated": "now", "cluster": "beverin", "arms": [{"arm": "</script><b>x</b>"}]}
    page = board.render(data)
    body = page.split('<script type="application/json" id="data">', 1)[1].split("</script>", 1)[0]
    assert json.loads(body) == data


@pytest.mark.parametrize(
    ("campaign", "variant", "experiment"),
    [
        ("cpf-llr-focus40", "c-cpf", "cpf-llr"),
        ("cpf-llr-focus40", "c-cpfsrc", "cpf-llr"),
        ("cpf-llr-focus40", "c-skills", "llr-focus40"),
        ("scicomp-dc", "cpf", "cpf-scicomp"),
        ("scicomp-dc", "dc-cpfsrc", "cpf-scicomp"),
        ("scicomp-dc", "plain", "scicomp-focus40"),
        ("llrblind", "c-cpf", "llr-focus40-blind"),
    ],
)
def test_a_cpf_arm_is_its_own_experiment_on_the_board(
    board: types.ModuleType, campaign: str, variant: str, experiment: str
) -> None:
    """CPF-LLR and CPF-SciComp are reported apart from the campaigns their arms ran in."""
    assert board.board_campaign(campaign, variant).experiment == experiment


def test_the_dropped_gpu_smoke_is_not_on_the_board(board: types.ModuleType) -> None:
    """GPU Smoke@5 was dropped from reporting; its arms must not come back as an experiment window."""
    assert board.campaign_of("gpusmoke5-hip-cpf") == ""


def job_dir_with_rows(root: pathlib.Path, job_id: str, benchmarks: list[str]) -> pathlib.Path:
    """A run directory whose one judge shard holds a submissions row per name in ``benchmarks``."""
    shard = root / job_id / "judge" / "rank-0"
    shard.mkdir(parents=True)
    conn = sqlite3.connect(shard / "hpcagent_bench.db")
    with conn:
        conn.execute("create table submissions (benchmark text)")
        conn.executemany("insert into submissions values (?)", [(name,) for name in benchmarks])
    conn.close()
    return root / job_id


@pytest.mark.parametrize(
    ("rows", "done", "status"),
    [
        ({"100": ["a", "b"]}, 2, "incomplete"),
        ({"100": ["a"], "200": ["b", "c"]}, 3, "complete"),
        ({"100": ["a", "b"], "200": ["b"]}, 2, "incomplete"),
    ],
)
def test_an_arms_coverage_is_the_union_of_every_jobs_rows(
    board: types.ModuleType, tmp_path: pathlib.Path, rows: dict[str, list[str]], done: int, status: str
) -> None:
    """A complement wave grades only what the first wave left, so reading one job reports finished kernels owed."""
    arm = "cpf-llr-focus40-oss120b-c-cpfsrc"
    dirs = {job_id: job_dir_with_rows(tmp_path, job_id, names) for job_id, names in rows.items()}
    jobs = [board.Job(job_id, arm, "COMPLETED", 3, "", "") for job_id in rows]
    row = board.arm_row(arm, jobs, dirs, ["a", "b", "c"], MODELS)
    assert (row["done"], row["status"]) == (done, status), row


def test_a_clean_reruns_row_counts_only_its_own_jobs(board: types.ModuleType, tmp_path: pathlib.Path) -> None:
    """The superseded arm's tasks are dropped at read (spec X9), so counting them here would report a
    coverage no table will ever use -- and an arm that owes half its roster would read as complete."""
    arm = "cpf-llr-focus40-oss120b-c-cpfsrc"
    dirs = {
        "100": job_dir_with_rows(tmp_path, "100", ["a", "b", "c"]),
        "200": job_dir_with_rows(tmp_path, "200", ["a"]),
    }
    jobs = [board.Job("100", arm, "COMPLETED", 3, "", ""), board.Job("200", arm + "-clean", "COMPLETED", 3, "", "")]
    row = board.arm_row(arm + "-clean", jobs, dirs, ["a", "b", "c"], MODELS)
    assert (row["clean"], row["done"], row["status"]) == (True, 1, "incomplete"), row
    assert [job["id"] for job in row["jobs"]] == ["100", "200"], row


def canon_csv(path: pathlib.Path, col: str, rank: int, rows: list[tuple[str, str]]) -> None:
    """A ``<col>.rank<N>.csv`` shard with a header and one ``kernel,status`` row per entry in ``rows``."""
    path.mkdir(parents=True, exist_ok=True)
    lines = ["framework,preset,datatype,kernel,impl,status,validated,median_ms,failure,error"]
    for kernel, status in rows:
        lines.append(f"{col},fuzzed,float64,{kernel},default,{status},True,1.0,,")
    (path / f"{col}.rank{rank}.csv").write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize(
    ("col", "device"), [("cc", "CPU"), ("dace_cpu", "CPU"), ("dace_gpu", "GPU"), ("pluto", "CPU"), ("ppcg", "GPU")]
)
def test_a_canon_columns_device_reads_the_framework_registrys_arch_field(
    board: types.ModuleType, col: str, device: str
) -> None:
    """``ppcg``'s name has no "gpu" in it; reading FRAMEWORK_META["arch"] instead of guessing from
    the name is what gets it (and every future column) right."""
    assert board.canon_device(col) == device


@pytest.mark.parametrize(
    ("name", "col", "expected"),
    [
        ("canon40-cc", "cc", True),
        ("canon40-cc_autopar", "cc", False),
        ("canon40-cc-b", "cc", True),
        ("canon40-dace_cpu", "dace_cpu", True),
        ("canon40-dace_cpu_canonicalize", "dace_cpu", False),
        ("canon40-dace_cpu_canonicalize-b", "dace_cpu_canonicalize", True),
    ],
)
def test_a_canon_job_name_does_not_fold_into_a_column_that_prefixes_its_own(
    board: types.ModuleType, name: str, col: str, expected: bool
) -> None:
    """``cc`` prefixes ``cc_autopar`` and ``dace_cpu`` prefixes ``dace_cpu_canonicalize``: a bare
    startswith would count one column's job as the other's."""
    assert board.canon_job_name_matches(name, "canon40", col) == expected


@pytest.mark.parametrize(
    ("name", "prefix", "col", "expected"),
    [
        ("canon-llr-cc", "canon-llr", "cc", True),
        ("canon-llr-cc_autopar", "canon-llr", "cc", False),
        ("canon-loop_level_reasoning-dace_gpu", "canon-loop_level_reasoning", "dace_gpu", True),
        ("canon-loop_level_reasoning-dace_gpu_canonicalize", "canon-loop_level_reasoning", "dace_gpu", False),
        ("canon-scicomp37-numba", "canon-scicomp37", "numba", True),
    ],
)
def test_a_canon_job_name_matches_its_tags_own_prefix(
    board: types.ModuleType, name: str, prefix: str, col: str, expected: bool
) -> None:
    """The sweeps write canon-<tag>-<col> job names now, not just the historical canon40-<col>."""
    assert board.canon_job_name_matches(name, prefix, col) == expected


def test_a_canon_columns_done_and_failed_kernels_read_the_latest_dir(
    board: types.ModuleType, tmp_path: pathlib.Path
) -> None:
    """A later canon directory (a fresh stamp, or a ``-b`` re-run) supersedes an earlier one's status
    for the same kernel -- the row must read the latest one, not the union of every wave's rows."""
    base = tmp_path / "canon-llr-focus40-20260915"
    rerun = tmp_path / "canon-llr-focus40-20260915-b"
    canon_csv(base, "cc", 0, [("a", "ok"), ("b", "crash"), ("c", "crash")])
    canon_csv(rerun, "cc", 0, [("b", "ok")])  # the re-run fixed b; c is still owed and still failed
    dirs = [base, rerun]

    row = board.canon_column_row("llr-focus40", "cc", dirs, ["a", "b", "c"], [])

    assert (row["done"], row["failed"], row["roster"]) == (2, ["c"], 3), row
    assert row["device"] == "CPU"
    assert row["experiment_name"] == "Compiler baselines: Loop Level Reasoning Focus@40"


def test_canon_dirs_finds_only_this_tags_directories_oldest_first(
    board: types.ModuleType, tmp_path: pathlib.Path
) -> None:
    older = tmp_path / "canon-llr-focus40-20260915"
    newer = tmp_path / "canon-llr-focus40-20260915-b"
    unrelated = tmp_path / "canon-scicomp40-20260915"
    for path in (older, unrelated):
        path.mkdir()
    os.utime(older, (1000, 1000))
    newer.mkdir()
    os.utime(newer, (2000, 2000))

    assert board.canon_dirs(tmp_path, "llr-focus40") == [older, newer]


def test_canon_rows_join_the_arms_list_as_their_own_experiment_group(
    board: types.ModuleType, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The board shows compiler baselines the same way it shows every other experiment: strip, rows,
    jobs, grouped by ``experiment`` -- so a canon row must carry that same shape. canon_rows is fixed
    to CANON_TAGS now, not CAMPAIGNS: the three canon experiments are their own thing."""
    root = tmp_path / "canon-llr-focus40-20260915"
    canon_csv(root, "numba", 0, [("a", "ok")])
    canon_csv(root, "dace_gpu", 0, [("a", "crash")])
    monkeypatch.setattr(board.remaining_kernels, "roster", lambda tag, opt: ["a"])
    monkeypatch.setattr(board, "canon_jobs", lambda since, prefixes: [])

    rows = board.canon_rows(tmp_path, "/opt")

    by_col = {row["variant"]: row for row in rows if row["experiment"] == "canon40-llr-focus40"}
    assert by_col["numba"]["done"] == 1 and by_col["numba"]["device"] == "CPU"
    assert by_col["dace_gpu"]["done"] == 0 and by_col["dace_gpu"]["device"] == "GPU"
    assert len(rows) == len(board.CANON_COLUMNS)  # only llr-focus40 has a directory here


def test_a_clean_rerun_replaces_the_arm_it_supersedes(
    board: types.ModuleType, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PROPERTY CHANGED on purpose (user, 2026-09-18): where an arm has a clean re-run, only the clean
    arm is on the board and the analysis continues on it; the 2026-09-15 rule showed both rows."""
    arm = "cpf-llr-focus40-oss120b-c-cpfsrc"
    runs = tmp_path / "runs" / "cpf-llr-focus40-20260915"
    for job_id, names in (("100", ["a", "b"]), ("200", ["a"])):
        job_dir_with_rows(runs, job_id, names)
    jobs = [board.Job("100", arm, "COMPLETED", 3, "", ""), board.Job("200", arm + "-clean", "COMPLETED", 3, "", "")]
    monkeypatch.setattr(board, "slurm_jobs", lambda ids: jobs)
    monkeypatch.setattr(board, "queued_ids", list)
    monkeypatch.setattr(board.remaining_kernels, "roster", lambda tag, opt: ["a", "b"])

    rows = board.arm_rows(tmp_path / "runs", "/opt", MODELS)

    assert [(row["arm"], row["clean"], row["done"]) for row in rows] == [(arm + "-clean", True, 1)], rows


@pytest.mark.parametrize(
    "arm",
    [
        "cpf-llr-focus40-s1of8-unionalpha-c",
        "scicomp-dc-cpp-qwen38-plain",
        "scicomp-dc-gpu-oss120b-c-openmp-plain",
        "cpf-llr-focus40-qwen38-fortran",
        "cpf-llr-focus40-oss120b-fortran-skills",
    ],
)
def test_an_arm_the_user_dropped_is_not_on_the_board(
    board: types.ModuleType, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    """User 2026-09-18: union-alpha, scicomp C++, scicomp GPU c-openmp and LLR CPU Fortran arms are out
    of the experiments; a kept sibling (scicomp Fortran, GPU hip) must stay."""
    runs = tmp_path / "runs" / "x-20260918"
    job_dir_with_rows(runs, "100", ["a"])
    job_dir_with_rows(runs, "101", ["a"])
    kept = "scicomp-dc-fortran-qwen38-plain"
    jobs = [board.Job("100", arm, "COMPLETED", 3, "", ""), board.Job("101", kept, "COMPLETED", 3, "", "")]
    monkeypatch.setattr(board, "slurm_jobs", lambda ids: jobs)
    monkeypatch.setattr(board, "queued_ids", list)
    monkeypatch.setattr(board.remaining_kernels, "roster", lambda tag, opt: ["a"])

    rows = board.arm_rows(tmp_path / "runs", "/opt", MODELS)

    assert [row["arm"] for row in rows] == [kept], rows
