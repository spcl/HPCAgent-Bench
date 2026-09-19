# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The wave board's arm identity, status rule and data embedding: the persistent experiment-status page
reports these, and a wrong one shows a finished experiment as owed or an owed one as finished."""

import importlib.util
import json
import os
import pathlib
import sqlite3
import subprocess
import sys
import types

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "wave_board.py"
MODELS = ("kimi27sglang", "oss120b", "qwen38", "glm53")

#: A row's ``ts``, arbitrary-but-after-any-real-commit: these tests use fake kernel names ("a", "b",
#: "c", "d", "retired_kernel") that resolve no real manifest, so remaining_kernels.comparable_since_ms
#: always returns 0 for them regardless of the ``opt`` these tests pass -- it exists only because the
#: real schema requires the column (2026-09-18 manifest-epoch fix).
FAR_FUTURE_TS_MS = 10**13


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
            ("gpu-llr-focus40", "kimi27sglang", "c-openmp-skills"),
        ),
        ("llrblind-kimi27sglang-fortran-skills", ("llrblind", "kimi27sglang", "fortran-skills")),
        ("scicomp-dc-qwen38-cpfsrc", ("scicomp-dc", "qwen38", "cpfsrc")),
        ("cpf-llr-focus40-qwen38-c-cpf", ("cpf-llr-focus40", "qwen38", "c-cpf")),
        ("cpf-llr-focus40-oss120b-c", ("cpf-llr-focus40", "oss120b", "c")),
        (
            "scicomp-perf-playbook-gpu-oss120b-hip-perf-playbook-amd",
            ("scicomp-perf-playbook-gpu", "oss120b", "hip-perf-playbook-amd"),
        ),
        (
            "scicomp-perf-playbook-qwen38-perf-playbook-cpu",
            ("scicomp-perf-playbook", "qwen38", "perf-playbook-cpu"),
        ),
        # scicomp-dc-fortran-<model>-plain: EXPERIMENT is overridden to put the language BEFORE the
        # model, unlike a GPU arm's <model>-<lang>-plain -- split_arm must still find the model and
        # keep "fortran-" on the variant so a Fortran baseline row reads apart from the C one.
        ("scicomp-dc-fortran-oss120b-plain", ("scicomp-dc", "oss120b", "fortran-plain")),
        ("scicomp-dc-fortran-qwen38-plain", ("scicomp-dc", "qwen38", "fortran-plain")),
        # scicomp-dc-gpu-<model>-<lang>-plain: same shape as scicomp-perf-playbook-gpu's arms.
        ("scicomp-dc-gpu-oss120b-hip-plain", ("scicomp-dc-gpu", "oss120b", "hip-plain")),
        ("scicomp-dc-gpu-qwen38-triton-plain", ("scicomp-dc-gpu", "qwen38", "triton-plain")),
    ],
)
def test_an_arm_name_splits_into_its_campaign_model_and_variant(
    board: types.ModuleType, arm: str, expected: tuple[str, str, str]
) -> None:
    """split_arm never sees a ``-clean`` suffix: arm_rows folds a clean re-run into the identity it
    re-runs (2026-09-18, remaining_kernels.base_arm) before split_arm is ever called on it."""
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
    )


def test_scicomp_perf_playbook_gpu_wins_over_its_cpu_prefix(board: types.ModuleType) -> None:
    """campaign_of takes the LONGEST matching prefix: "scicomp-perf-playbook-gpu" must win over
    "scicomp-perf-playbook", or a GPU arm's rest-of-name would start "gpu-<model>-..." and the model
    would never match (submit-scicomp-perf-playbook.sh's DEVICE=gpu knob, mirroring submit-scicomp-dc.sh)."""
    assert board.campaign_of("scicomp-perf-playbook-gpu-oss120b-hip-perf-playbook-amd") == "scicomp-perf-playbook-gpu"
    row_campaign, model, variant = board.split_arm("scicomp-perf-playbook-gpu-qwen38-hip-perf-playbook-amd", MODELS)
    assert (row_campaign, model, variant) == (
        "scicomp-perf-playbook-gpu",
        "qwen38",
        "hip-perf-playbook-amd",
    )


def test_scicomp_dc_gpu_wins_over_its_cpu_prefix(board: types.ModuleType) -> None:
    """Same trap as scicomp-perf-playbook-gpu: without its own CAMPAIGNS key, a
    scicomp-dc-gpu-<model>-<lang>-plain arm falls through to "scicomp-dc" (CPU) and its "rest" starts
    "gpu-<model>-...", so the model never matches and both device and model come back wrong."""
    assert board.campaign_of("scicomp-dc-gpu-oss120b-hip-plain") == "scicomp-dc-gpu"
    assert board.CAMPAIGNS["scicomp-dc-gpu"].device == "GPU"
    row_campaign, model, variant = board.split_arm("scicomp-dc-gpu-oss120b-hip-plain", MODELS)
    assert (row_campaign, model, variant) == ("scicomp-dc-gpu", "oss120b", "hip-plain")


def test_scicomp_baseline_and_perf_playbook_are_one_board_experiment(board: types.ModuleType) -> None:
    """User 2026-09-19 (corrected same day): scicomp-dc (the plain baseline), scicomp-dc-gpu,
    scicomp-perf-playbook and scicomp-perf-playbook-gpu report under ONE name -- "..., Perf Playbook",
    no "Divide and Conquer" and no ", GPU" suffix, CPU and GPU alike -- split into a CPU and a GPU
    section only by device (the board groups rows by (experiment, device))."""
    keys = ("scicomp-dc", "scicomp-dc-gpu", "scicomp-perf-playbook", "scicomp-perf-playbook-gpu")
    for key in keys:
        assert board.CAMPAIGNS[key].experiment == "scicomp-focus40"
    names = {board.CAMPAIGNS[key].name for key in keys}
    assert names == {"Scientific Computing Focus@40, Perf Playbook"}
    assert "GPU" not in next(iter(names))
    assert "Divide and Conquer" not in next(iter(names))
    assert board.CAMPAIGNS["scicomp-dc"].device == board.CAMPAIGNS["scicomp-perf-playbook"].device == "CPU"
    assert board.CAMPAIGNS["scicomp-dc-gpu"].device == board.CAMPAIGNS["scicomp-perf-playbook-gpu"].device == "GPU"


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
        conn.execute("create table submissions (benchmark text, ts integer)")
        conn.executemany("insert into submissions values (?, ?)", [(name, FAR_FUTURE_TS_MS) for name in benchmarks])
    conn.close()
    return root / job_id


def write_episode(
    job_dir: pathlib.Path, index: int, kernel: str, returncode: int, *, cancelled: bool = False, start_ms: int = 1000
) -> None:
    """One worker's ``tokens.json`` (agent_driver.write_cost_record's shape) plus, if ``cancelled``,
    its sibling ``agent_driver.CANCELLED_MARKER`` file -- the two files owed_exit_classes reads."""
    workdir = job_dir / "agents" / "node-0" / f"problem-{index}-worker-{index}"
    workdir.mkdir(parents=True, exist_ok=True)
    tokens = {
        "kernel": f"loop_level_reasoning/{kernel}/{kernel}",
        "returncode": returncode,
        "final_attempt_start_ms": start_ms,
    }
    (workdir / "tokens.json").write_text(json.dumps(tokens), encoding="utf-8")
    if cancelled:
        (workdir / "cancelled").write_text("rc\n", encoding="utf-8")


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
    row = board.arm_row(arm, jobs, dirs, ["a", "b", "c"], MODELS, str(tmp_path))
    assert (row["done"], row["status"]) == (done, status), row


def test_a_touched_kernel_outside_the_roster_does_not_inflate_done(
    board: types.ModuleType, tmp_path: pathlib.Path
) -> None:
    """A submissions row for a kernel a retired tag or rename dropped from the CURRENT roster must
    not push `done` past `roster` -- kernel_status bounds `done` to `full`, the same bound
    remaining_kernels.py's own report_arm keeps by summing over the roster rather than counting
    every touched name."""
    arm = "cpf-llr-focus40-oss120b-c-cpfsrc"
    dirs = {"100": job_dir_with_rows(tmp_path, "100", ["a", "b", "retired_kernel"])}
    jobs = [board.Job("100", arm, "COMPLETED", 3, "", "")]
    row = board.arm_row(arm, jobs, dirs, ["a", "b"], MODELS, str(tmp_path))
    assert (row["done"], row["roster"], row["status"]) == (2, 2, "complete"), row


def test_a_clean_reruns_row_folds_into_the_arm_it_supersedes(board: types.ModuleType, tmp_path: pathlib.Path) -> None:
    """The clean job's coverage ADDS to the plain arm's (2026-09-18 fold), so an arm the clean re-run
    only partly repeated still reads its plain jobs' rows too, not just the clean one's."""
    arm = "cpf-llr-focus40-oss120b-c-cpfsrc"
    dirs = {
        "100": job_dir_with_rows(tmp_path, "100", ["a", "b"]),
        "200": job_dir_with_rows(tmp_path, "200", ["c"]),
    }
    jobs = [board.Job("100", arm, "COMPLETED", 3, "", ""), board.Job("200", arm + "-clean", "COMPLETED", 3, "", "")]
    row = board.arm_row(arm, jobs, dirs, ["a", "b", "c"], MODELS, str(tmp_path))
    assert (row["clean"], row["done"], row["status"]) == (True, 3, "complete"), row
    assert [job["id"] for job in row["jobs"]] == ["100", "200"], row


def test_owed_kernels_split_into_done_by_rule_budget_and_infra(board: types.ModuleType, tmp_path: pathlib.Path) -> None:
    """A kernel with no ``submissions`` row is still DONE when its latest episode ended on its own
    (context overflow, rc 126); one that hit its own timeout (rc 124) is owed at BUDGET; one the job
    cancelled mid-episode is owed as INFRA. All three must be told apart in one arm's row."""
    arm = "cpf-llr-focus40-oss120b-c-cpfsrc"
    job_dir = job_dir_with_rows(tmp_path, "100", ["a"])  # a: a real submissions row
    write_episode(job_dir, 1, "b", 126)  # b: context overflow -> done by rule
    write_episode(job_dir, 2, "c", 124)  # c: hit AGENT_TIMEOUT_SECONDS -> owed, budget
    write_episode(job_dir, 3, "d", 124, cancelled=True)  # d: the job took it down -> owed, infra

    dirs = {"100": job_dir}
    jobs = [board.Job("100", arm, "COMPLETED", 3, "", "")]
    row = board.arm_row(arm, jobs, dirs, ["a", "b", "c", "d"], MODELS, str(tmp_path))

    assert (row["done"], row["owed_budget"], row["owed_infra"], row["status"]) == (2, 1, 1, "incomplete"), row


def make_git_repo_with_manifest(tmp_path: pathlib.Path, kernel: str = "probe_kernel") -> tuple:
    """A real git checkout: ``kernel``'s manifest committed once, then resized at a LATER commit --
    mirrors test_remaining_kernels.py's fixture of the same name. Returns ``(repo dir, kernel name,
    the resize commit's ts in epoch ms)``."""
    repo = tmp_path / "opt"
    manifest_dir = repo / "hpcagent_bench" / "benchmarks" / "track" / kernel
    manifest_dir.mkdir(parents=True)
    manifest = manifest_dir / f"{kernel}.yaml"
    manifest.write_text("preset: {XL: {n: 100}}\n", encoding="utf-8")
    git = ["git", "-C", str(repo)]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["config", "user.email", "t@t"], check=True)
    subprocess.run(git + ["config", "user.name", "t"], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "add manifest"], check=True)
    manifest.write_text("preset: {XL: {n: 200}}\n", encoding="utf-8")  # sizing changed
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "resize XL"], check=True)
    out = subprocess.run(git + ["log", "-1", "--format=%ct"], capture_output=True, text=True, check=True)
    return repo, kernel, int(out.stdout.strip()) * 1000


def test_arm_row_forwards_opt_so_a_stale_pre_resize_row_stays_owed(
    board: types.ModuleType, tmp_path: pathlib.Path
) -> None:
    """arm_row's ``opt`` argument must actually reach remaining_kernels.touched, not get dropped on
    the way down through kernel_status -- a submissions row graded before the kernel's own manifest
    last changed (2026-09-18 manifest-epoch fix, job 641739) must leave the board reporting it owed,
    not done, exactly like remaining_kernels.py's own report would."""
    repo, kernel, changed_ts_ms = make_git_repo_with_manifest(tmp_path)
    arm = "cpf-llr-focus40-oss120b-c-cpfsrc"
    shard = tmp_path / "runs" / "100" / "judge" / "rank-0"
    shard.mkdir(parents=True)
    conn = sqlite3.connect(shard / "hpcagent_bench.db")
    with conn:
        conn.execute("create table submissions (benchmark text, ts integer)")
        conn.execute("insert into submissions values (?, ?)", (kernel, changed_ts_ms - 1000))  # stale
    conn.close()

    dirs = {"100": tmp_path / "runs" / "100"}
    jobs = [board.Job("100", arm, "COMPLETED", 3, "", "")]
    row = board.arm_row(arm, jobs, dirs, [kernel], MODELS, str(repo))
    assert row["done"] == 0, row


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


def test_a_clean_rerun_folds_into_one_board_row(
    board: types.ModuleType, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PROPERTY CHANGED on purpose (user, 2026-09-18): an arm and its clean re-run are ONE identity,
    ONE board row, union coverage over both -- not two rows and not the clean one replacing the
    other (the 2026-09-15 rule showed both; a 2026-09-18 rule before this one showed only the clean)."""
    arm = "cpf-llr-focus40-oss120b-c-cpfsrc"
    runs = tmp_path / "runs" / "cpf-llr-focus40-20260915"
    for job_id, names in (("100", ["a"]), ("200", ["b"])):
        job_dir_with_rows(runs, job_id, names)
    jobs = [board.Job("100", arm, "COMPLETED", 3, "", ""), board.Job("200", arm + "-clean", "COMPLETED", 3, "", "")]
    monkeypatch.setattr(board, "slurm_jobs", lambda ids: jobs)
    monkeypatch.setattr(board, "queued_ids", list)
    monkeypatch.setattr(board.remaining_kernels, "roster", lambda tag, opt: ["a", "b"])

    rows = board.arm_rows(tmp_path / "runs", "/opt", MODELS)

    assert [(row["arm"], row["clean"], row["done"], row["status"]) for row in rows] == [(arm, True, 2, "complete")]
    assert [job["id"] for job in rows[0]["jobs"]] == ["100", "200"], rows


def test_a_smoke_job_that_reused_a_real_arms_name_is_excluded(
    board: types.ModuleType, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Job 641175 (2026-09-18): a smoke sanity check submitted under a REAL arm's name, with nothing
    in ``runs.arm`` or the job name telling it apart. Its rows must not count as that arm's coverage,
    or an arm the smoke run never really covered reads as further along than its real jobs show."""
    arm = "harness20-qwen38-claude"
    runs = tmp_path / "runs" / "harness20-20260918"
    job_dir_with_rows(runs, "100", ["a"])
    smoke_job_id = next(iter(board.remaining_kernels.SMOKE_JOBS))
    job_dir_with_rows(runs, smoke_job_id, ["a", "b"])  # the smoke job's own kernel row must not count
    jobs = [
        board.Job("100", arm, "COMPLETED", 3, "", ""),
        board.Job(smoke_job_id, arm, "COMPLETED", 1, "", ""),
    ]
    monkeypatch.setattr(board, "slurm_jobs", lambda ids: jobs)
    monkeypatch.setattr(board, "queued_ids", list)
    monkeypatch.setattr(board.remaining_kernels, "roster", lambda tag, opt: ["a", "b"])

    rows = board.arm_rows(tmp_path / "runs", "/opt", MODELS)

    assert [(row["arm"], row["done"]) for row in rows] == [(arm, 1)], rows
    assert [job["id"] for job in rows[0]["jobs"]] == ["100"], rows


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
