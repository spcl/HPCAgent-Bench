# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``CLEAN=1`` and ``DEADLINE=`` in experiments/submit-gpu-llr40.sh, driven with SUBMIT=0.

Ported from experiments/submit-scicomp-dc.sh and submit-cpf-llr40.sh (commit e51947af8): a clean
re-run has to be recognisable from the arm name alone, and a wave launched against a deadline has to
END before it rather than be killed mid-episode. Both are decided before a single node is allocated,
so both are checked without touching the queue.
"""

import datetime
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

from tests.env_render import SPEC_INPUTS, set_base
from tests.test_submit_scicomp_dc_cpfsrc import env_dict, stub

REPO = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"

SUBMIT_INPUTS = (
    *SPEC_INPUTS,
    "submit-gpu-llr40.sh",
    "arm_nodes.sh",
    "record_identity.sh",
    "submit_common.sh",
    "pin_env_kv.sh",
    "make_problems.py",
    "packet_env.py",
)

#: Real llr-focus40 kernels (shared with tests/test_submit_cpf_llr40.py), so make_problems.py's
#: kernel selection resolves them without a fabricated manifest.
ROSTER_KERNELS = ("fuse_diamond", "tsvc_2_s115")

#: arm_nodes.sh: image pull, engine start and the readiness probe, before any agent runs.
STAGING_SECONDS = 3 * 3600
#: submit-gpu-llr40.sh: the slack between the job's own end and the deadline.
MARGIN_SECONDS = 300
#: base-qwen38's own AGENT_TIMEOUT_SECONDS: the episode every hip arm runs, deadline or no.
CONFIGURED_AGENT_SECONDS = 21600
#: How far ahead the deadline is placed. Far enough that the job limit alone would allow a LONGER
#: episode than the base env's, which is the case the cap exists for.
HOURS_AHEAD = 30
#: Close enough that the deadline, not the base env, decides the episode.
HOURS_TIGHT = 5


def deadline(hours: float) -> str:
    """An ISO timestamp ``hours`` from now, spelled the way an operator types DEADLINE."""
    return (datetime.datetime.now() + datetime.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")


def seconds_of(walltime: str) -> int:
    hours, minutes, seconds = (int(part) for part in walltime.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def prepared(result: subprocess.CompletedProcess[str]) -> dict[str, str]:
    """``{arm: walltime}`` off the launcher's SUBMIT=0 report."""
    pattern = re.compile(r"^prepared (\S+) \(\d+ nodes, (\d\d:\d\d:\d\d), .*\) -- not submitted")
    return {match.group(1): match.group(2) for line in result.stdout.splitlines() if (match := pattern.match(line))}


def built_lines(result: subprocess.CompletedProcess[str]) -> list[str]:
    return [line for line in result.stdout.splitlines() if line.startswith("prepared ")]


def submit_tree(root: pathlib.Path) -> pathlib.Path:
    (root / "experiments").mkdir(parents=True)
    for name in SUBMIT_INPUTS:
        (root / "experiments" / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(EXPERIMENTS / name, root / "experiments" / name)
    (root / "experiments" / "kernels.txt").write_text("\n".join(ROSTER_KERNELS) + "\n")
    stub(root / "bin", "sbatch", 'touch "${STUB_MARKERS}/sbatch-called"; exit 1')
    # the launcher hardcodes ${SCRATCH}/venv-hpcagent-bench-314/bin/python, so that path must exist
    stub(root / "scratch" / "venv-hpcagent-bench-314" / "bin", "python", f'exec "{sys.executable}" "$@"')
    return root


def clean_env(root: pathlib.Path, **knobs: str) -> dict[str, str]:
    drop = {
        "SUBMIT",
        "LANGUAGES",
        "MODELS",
        "LEGS",
        "OFFLOAD",
        "PACKET",
        "KERNELS_FILE",
        "PROBLEMS_PREFIX",
        "TAG",
        "AMD_CE_ENV_GPU",
        "DEPEND_ON",
        "EXPERIMENT",
        "RECORD_EXPERIMENT",
        "STAMP",
        "CLEAN",
        "DEADLINE",
        "DEADLINE_MARGIN_SECONDS",
        "MIN_AGENT_SECONDS",
        "STAGING_HOURS",
        "BEGIN",
        "TIME_LIMIT",
        "OPT",
        "PY",
        "SCRATCH",
        "PYTHONPATH",
    }
    env = {k: v for k, v in os.environ.items() if k not in drop and not k.startswith("SLURM_")}
    env.update(
        PATH=f"{root / 'bin'}:{env['PATH']}",
        SCRATCH=str(root / "scratch"),
        OPT=str(REPO),
        PYTHONPATH=f"{REPO}:{REPO / 'hpcagent_bench' / 'numpy_translators' / 'src'}",
        STAMP="20260915",
        STUB_MARKERS=str(root),
        **knobs,
    )
    return env


def run_submit(root: pathlib.Path, **knobs: str) -> subprocess.CompletedProcess[str]:
    """SUBMIT defaults to 0: submit_arm_job's own default (unset) calls real sbatch."""
    return subprocess.run(
        ["bash", str(root / "experiments" / "submit-gpu-llr40.sh")],
        env=clean_env(root, **{"SUBMIT": "0", **knobs}),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


#: Every run here feeds the roster through KERNELS_FILE="kernels.txt" (real roster_for(TAG) cannot
#: be redirected into a temp tree), so every env name carries submit_common.sh's
#: kernels_file_suffix("kernels.txt" -> "-kernels") the same way a real owed/subset rerun would.
FILE_SFX = "-kernels"


def arm_env(experiments: pathlib.Path, arm: str) -> pathlib.Path:
    return experiments / f".env.gpu-llr-focus40-qwen38-{arm}{FILE_SFX}"


@pytest.fixture(name="clean", scope="module")
def clean_fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[pathlib.Path, subprocess.CompletedProcess[str]]:
    """One CLEAN=1 wave (plain + skills, language hip) under a deadline far enough out that it never
    shrinks the episode."""
    tmp_path = tmp_path_factory.mktemp("clean")
    root = submit_tree(tmp_path)
    result = run_submit(
        root,
        MODELS="qwen38",
        LANGUAGES="hip",
        KERNELS_FILE="kernels.txt",
        CLEAN="1",
        DEADLINE=deadline(HOURS_AHEAD),
    )
    return root / "experiments", result


def test_a_clean_wave_names_every_arm_and_every_job_with_the_suffix(
    clean: tuple[pathlib.Path, subprocess.CompletedProcess[str]],
) -> None:
    """The suffix is the whole mechanism: nothing else tells the analysis or the board that these
    tasks supersede the ones before them."""
    _, result = clean
    assert result.returncode == 0, result.stderr
    assert sorted(prepared(result)) == ["gpu-llr-focus40-qwen38-hip-clean", "gpu-llr-focus40-qwen38-hip-skills-clean"]


def test_a_clean_arm_keeps_the_identity_the_analysis_pairs_on(
    clean: tuple[pathlib.Path, subprocess.CompletedProcess[str]],
) -> None:
    """The suffix names no condition. An arm that also moved its packet or language would be a new
    condition with nothing to pair against."""
    experiments, _ = clean
    values = env_dict(arm_env(experiments, "hip-skills-clean"))
    assert values["HPCAGENT_BENCH_RECORD_PACKET"] == "lang-skills"
    assert values["HPCAGENT_BENCH_RECORD_LANGUAGE"] == "hip"
    assert values["HPCAGENT_BENCH_RECORD_DEVICE"] == "gpu"
    assert values["HPCAGENT_BENCH_RECORD_EXPERIMENT"] == "llr-focus40"
    assert values["HPCAGENT_BENCH_RECORD_ARM"] == "gpu-llr-focus40-qwen38-hip-skills-clean"


def test_a_deadline_shrinks_the_job_but_never_lengthens_the_episode(
    clean: tuple[pathlib.Path, subprocess.CompletedProcess[str]],
) -> None:
    """A deadline is a bound on the JOB, and a longer episode is a different condition: an arm whose
    agents get more time than the arms it is compared with is not comparable with them, nor with the
    same arm submitted an hour later under the same deadline."""
    experiments, result = clean
    reported = set(prepared(result).values())
    assert len(reported) == 1, reported
    limit = seconds_of(reported.pop())
    assert CONFIGURED_AGENT_SECONDS < limit - STAGING_SECONDS
    assert int(env_dict(arm_env(experiments, "hip-skills-clean"))["AGENT_TIMEOUT_SECONDS"]) == CONFIGURED_AGENT_SECONDS
    # The clock moves while the launcher runs, so the target is an upper bound, not an equality.
    target = HOURS_AHEAD * 3600 - MARGIN_SECONDS
    assert target - 120 <= limit <= target


def test_a_deadline_the_episode_does_not_fit_in_shortens_the_episode(tmp_path: pathlib.Path) -> None:
    """The other half of the same rule: what the job cannot cover, the agents do not get, or the job
    dies at its limit with the last batch ungraded and the arm is partly its own control."""
    root = submit_tree(tmp_path)
    result = run_submit(
        root,
        MODELS="qwen38",
        LANGUAGES="hip",
        LEGS="0",
        KERNELS_FILE="kernels.txt",
        DEADLINE=deadline(HOURS_TIGHT),
    )
    assert result.returncode == 0, result.stderr
    walltime = next(iter(prepared(result).values()))
    env = env_dict(arm_env(root / "experiments", "hip"))
    agent = int(env["AGENT_TIMEOUT_SECONDS"])
    assert agent == seconds_of(walltime) - STAGING_SECONDS < CONFIGURED_AGENT_SECONDS


def test_a_deadline_wave_starts_now_instead_of_waiting(
    clean: tuple[pathlib.Path, subprocess.CompletedProcess[str]],
) -> None:
    """A DEADLINE wave must not be left waiting on a BEGIN nobody set."""
    _, result = clean
    for line in built_lines(result):
        assert " begin " not in line, line


def test_a_deadline_too_close_to_measure_anything_refuses_the_wave(tmp_path: pathlib.Path) -> None:
    """Under an hour of agent time buys a handful of turns per kernel and an arm of build failures;
    the nodes are better left in the queue."""
    root = submit_tree(tmp_path)
    result = run_submit(
        root,
        MODELS="qwen38",
        LANGUAGES="hip",
        LEGS="0",
        KERNELS_FILE="kernels.txt",
        CLEAN="1",
        DEADLINE=deadline(0.9),
    )
    assert result.returncode == 2, result.stdout
    assert "under the 3600s floor" in result.stderr
    assert list((root / "experiments").glob(".env.gpu-llr-focus40-*")) == []


def test_kernels_file_order_is_deterministic_not_the_files_own_line_order(tmp_path: pathlib.Path) -> None:
    """make_problems.py sorts the resolved kernel set; kernels.txt here (used verbatim by this
    fixture's own submit_tree) already happens to list fuse_diamond before tsvc_2_s115 -- a fresh
    file with the opposite order must still come out sorted, not the file's own line order."""
    root = submit_tree(tmp_path)
    reversed_kf = root / "experiments" / "reversed.txt"
    reversed_kf.write_text("tsvc_2_s115\nfuse_diamond\n")
    result = run_submit(root, MODELS="qwen38", LANGUAGES="hip", LEGS="0", KERNELS_FILE="reversed.txt")
    assert result.returncode == 0, result.stderr
    problems = root / "experiments" / "problems-gpu-llr40-qwen38-hip-reversed.jsonl"
    kernels = [json.loads(line)["kernel"].rsplit("/", 1)[-1] for line in problems.read_text().splitlines()]
    assert kernels == sorted(ROSTER_KERNELS)


def test_an_unknown_kernel_name_is_refused_not_silently_dropped(tmp_path: pathlib.Path) -> None:
    root = submit_tree(tmp_path)
    (root / "experiments" / "bad.txt").write_text("fuse_diamond\nnosuchkernel123\n")
    result = run_submit(root, MODELS="qwen38", LANGUAGES="hip", LEGS="0", KERNELS_FILE="bad.txt")
    assert result.returncode != 0
    assert "nosuchkernel123" in result.stderr
    assert not list((root / "experiments").glob(".env.gpu-llr-focus40-*bad*"))
    # make_problems.py writes into problems.jsonl.tmp before the final `mv`; a failed selector never
    # reaches that mv (set -e kills the script first), so the .tmp precursor is expected litter --
    # only the final .jsonl name matters, since nothing else ever reads a .jsonl.tmp file.
    assert not list((root / "experiments").glob("problems-gpu-llr40-*bad*.jsonl"))


def test_walltime_scales_with_the_subsets_own_kernel_count(tmp_path: pathlib.Path) -> None:
    """arm_walltime batches on AGENTS_PER_NODE * AGENT_NODES workers; the real base env's 40 agents
    on 1 node cover both fixture rosters in a single batch, hiding any scaling bug, so this pins
    AGENTS_PER_NODE down to 1 worker to force one batch PER kernel and prove a 3-kernel subset needs
    more wall clock than the 2-kernel one (test_a_wave_without_clean_or_a_deadline_is_unchanged)."""
    root = submit_tree(tmp_path)
    set_base(root / "experiments", "campaign:qwen38", AGENTS_PER_NODE=1)
    three_kf = root / "experiments" / "three.txt"
    three_kf.write_text("fuse_diamond\ntsvc_2_s115\nargmax_with_index\n")
    result = run_submit(root, MODELS="qwen38", LANGUAGES="hip", LEGS="0", KERNELS_FILE="three.txt")
    assert result.returncode == 0, result.stderr
    walltime = next(iter(prepared(result).values()))
    # 1 worker, 3 kernels -> 3 batches of AGENT_TIMEOUT_SECONDS (21600s = 6h) + 3h staging = 21h
    assert walltime == "21:00:00", walltime


def test_a_wave_without_clean_or_a_deadline_is_unchanged(tmp_path: pathlib.Path) -> None:
    """Every wave so far ran without either, and neither default may move an arm name or a limit."""
    root = submit_tree(tmp_path)
    result = run_submit(
        root,
        MODELS="qwen38",
        LANGUAGES="hip",
        LEGS="0",
        KERNELS_FILE="kernels.txt",
    )
    assert result.returncode == 0, result.stderr
    assert sorted(prepared(result)) == ["gpu-llr-focus40-qwen38-hip"]
    # arm_walltime: 2 kernels in 1 batch (AGENTS_PER_NODE=40, AGENT_NODES=1) of AGENT_TIMEOUT_SECONDS
    # (21600/3600=6h) plus the staging allowance (3h).
    assert set(prepared(result).values()) == {"09:00:00"}
    assert " begin " not in built_lines(result)[0]
