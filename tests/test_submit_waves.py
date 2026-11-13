# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""MAX_NODES in hpcagent_bench/cluster/submit_common.sh: plan_waves splits the setups of one submit call into waves
whose summed node count is at most MAX_NODES, and submit_waves chains every wave afterany on all jobs of the previous
one and afterok on the prepare and golden jobs. A stub ``sbatch`` numbers its jobs and records each agent job's name
and dependency; nothing reaches Slurm."""

import pathlib
import re
import shutil
import subprocess
import sys

import pytest

from hpcagent_bench import paths
from tests.bash_stub import bash_stub
from tests.test_submit import submit, tree

BASH = shutil.which("bash")
assert BASH is not None

REPO = paths.ROOT
CLUSTER_DIR = REPO / "hpcagent_bench" / "cluster"

#: The first job id the chain stub prints; every later call prints the next integer.
FIRST_JOB_ID = 101

#: A stub sbatch: prints job ids FIRST_JOB_ID, FIRST_JOB_ID + 1, ... as ``<id>;cluster`` (sbatch --parsable on a
#: federation), records every call's arguments in ``${STUB_MARKERS}/sbatch.calls`` and appends
#: ``<id> <job-name> <dependency or ->`` per agent job to ``${STUB_MARKERS}/sbatch.chain``.
CHAIN_SBATCH = f"""if [[ -n "${{FAIL_FIRST:-}}" && ! -e "${{STUB_MARKERS}}/failed" ]]; then
    touch "${{STUB_MARKERS}}/failed"
    echo "sbatch: error: cli_filter plugin terminated with error" >&2
    exit 1
fi
last=$(cat "${{STUB_MARKERS}}/sbatch.last" 2>/dev/null || echo {FIRST_JOB_ID - 1})
id=$(( last + 1 ))
echo "${{id}}" > "${{STUB_MARKERS}}/sbatch.last"
printf '%s env:REPO=%s PLUGIN_DIRS=%s\\n' "$*" "${{HPCAGENT_BENCH_REPO-unset}}" "${{HPCAGENT_BENCH_USER_PLUGIN_DIRS-unset}}" \
    >> "${{STUB_MARKERS}}/sbatch.calls"
name="" dependency="-"
for arg; do case "${{arg}}" in --job-name=*) name="${{arg#*=}}" ;; --dependency=*) dependency="${{arg#*=}}" ;; esac; done
case "$*" in *services.sbatch*) echo "${{id}} ${{name}} ${{dependency}}" >> "${{STUB_MARKERS}}/sbatch.chain" ;; esac
echo "${{id}};cluster"
"""


def run_bash(tmp_path: pathlib.Path, script: str, **env: str) -> subprocess.CompletedProcess[str]:
    """``script`` after sourcing setup_nodes.sh and submit_common.sh, run in ``tmp_path`` with ``tmp_path/bin``
    first on PATH."""
    probe = tmp_path / "probe.sh"
    probe.write_text(f"set -eu\n. {CLUSTER_DIR / 'setup_nodes.sh'}\n. {CLUSTER_DIR / 'submit_common.sh'}\n{script}\n")
    site = tmp_path / "site.env"
    site.write_text("")
    return subprocess.run(
        [BASH, str(probe)],
        env={
            "PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin",
            "USER": "tester",
            "STUB_MARKERS": str(tmp_path),
            "HPCAGENT_BENCH_HOST_PYTHON": sys.executable,
            "HPCAGENT_BENCH_SITE_ENV": str(site),
            "SBATCH_ACCOUNT": "project",
            "HPCAGENT_BENCH_JOB_GPUS_PER_NODE": "4",
            **env,
        },
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def plan(tmp_path: pathlib.Path, max_nodes: str, *counts: int) -> subprocess.CompletedProcess[str]:
    """plan_waves over ``counts`` under ``max_nodes``; stdout is WAVE_NUMBERS, space-separated."""
    words = " ".join(str(count) for count in counts)
    return run_bash(tmp_path, f'plan_waves "{max_nodes}" {words}\necho "${{WAVE_NUMBERS[*]}}"')


def test_plan_waves_opens_a_wave_when_the_next_setup_would_pass_the_limit(tmp_path: pathlib.Path) -> None:
    done = plan(tmp_path, "32", 12, 12, 8, 20, 4, 32)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["1", "1", "1", "2", "2", "3"]


def test_plan_waves_without_a_limit_is_one_wave(tmp_path: pathlib.Path) -> None:
    done = plan(tmp_path, "", 12, 40, 8)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["1", "1", "1"]


def test_plan_waves_refuses_a_setup_larger_than_the_limit(tmp_path: pathlib.Path) -> None:
    done = plan(tmp_path, "8", 4, 9)
    assert done.returncode == 2
    assert "a setup of 9 nodes exceeds MAX_NODES=8" in done.stderr


@pytest.mark.parametrize("max_nodes", ["0", "-4", "eight", "08"])
def test_plan_waves_refuses_a_limit_that_is_not_a_positive_integer(tmp_path: pathlib.Path, max_nodes: str) -> None:
    done = plan(tmp_path, max_nodes, 4)
    assert done.returncode == 2
    assert f"MAX_NODES is a positive integer, got '{max_nodes}'" in done.stderr


def submit_three_setups(tmp_path: pathlib.Path, **env: str) -> subprocess.CompletedProcess[str]:
    """submit_waves over three 4-node setups ``s1 s2 s3`` under MAX_NODES=8 after job 77, with the chain stub."""
    bash_stub(tmp_path / "bin", "sbatch", CHAIN_SBATCH)
    for name in ("s1", "s2", "s3"):
        (tmp_path / f".env.{name}").write_text("INFERENCE_NODES=2\nAGENT_NODES=1\nJUDGE_NODES=1\n")
    return run_bash(
        tmp_path,
        "WAVE_ENVS=(.env.s1 .env.s2 .env.s3) WAVE_SETUPS=(s1 s2 s3) WAVE_WALLTIMES=(01:00:00 01:00:00 01:00:00)\n"
        'submit_waves 8 77 ""',
        **env,
    )


def test_each_wave_depends_on_every_job_of_the_previous_wave(tmp_path: pathlib.Path) -> None:
    done = submit_three_setups(tmp_path, SUBMIT="1")
    assert done.returncode == 0, done.stderr
    chain = (tmp_path / "sbatch.chain").read_text().splitlines()
    assert chain == ["101 s1 afterany:77", "102 s2 afterany:77", "103 s3 afterany:101:102"]


def test_a_dry_run_prints_the_waves_and_submits_nothing(tmp_path: pathlib.Path) -> None:
    done = submit_three_setups(tmp_path)
    assert done.returncode == 0, done.stderr
    headers = [line for line in done.stdout.splitlines() if line.startswith("wave ")]
    assert headers == [
        "wave 1/2: 8 nodes (MAX_NODES=8), after 77",
        "wave 2/2: 4 nodes (MAX_NODES=8), after wave 1 (s1:s2)",
    ]
    assert not (tmp_path / "sbatch.chain").exists()


def test_the_wave_waits_on_one_prepare_job_rendering_what_its_setups_read(tmp_path: pathlib.Path) -> None:
    """One prepare job renders numba and each setup's reference language (C for hip, which has no CPU reference of
    its own) in the judge image the setups name; one golden job per language grades the references after it; the
    wave waits on all of them beside DEPEND_ON, and a job a failure cancels says so."""
    root = tree(tmp_path)
    bash_stub(root / "bin", "sbatch", CHAIN_SBATCH)
    edf = root / "edf"
    edf.mkdir()
    (edf / "hpcagent-bench-judge-mi300-latest.toml").write_text('image = "judge.sqsh"\n')
    knobs = {"KERNELS_FILE": "subset.txt", "LANGUAGES": "c hip", "PREPARE": "1", "EDF_PATH": str(edf)}
    done = submit(root, "--account", "project", **knobs, DEPEND_ON="77", SUBMIT="1")
    assert done.returncode == 0, done.stderr
    calls = (root / "sbatch.calls").read_text().splitlines() if (root / "sbatch.calls").exists() else []
    prepare = [line for line in done.stdout.splitlines() if line.startswith("submitted prepare-")]
    assert prepare == ["submitted prepare-wave -> 101;cluster: renders numba,c"], (prepare, calls)
    goldens = [line for line in done.stdout.splitlines() if line.startswith("submitted goldens-")]
    assert goldens == ["submitted goldens-wave-c -> 102;cluster", "submitted goldens-wave-hip -> 103;cluster"]
    chain = [line.split() for line in (root / "sbatch.chain").read_text().splitlines()]
    assert [dependency for _, _, dependency in chain] == ["afterany:77,afterok:101:102:103"] * 2
    assert all("--kill-on-invalid-dep=yes" in line for line in calls if "--dependency=afterok:" in line), calls
    assert "--dependency=afterok:101" in calls[1], calls
    # the prep jobs' values ride sbatch's environment, never --export (Slurm cancels such a job at its start)
    prep = [line for line in calls if "prepare.sbatch" in line]
    assert len(prep) == 3, calls
    assert all(line.endswith(" PLUGIN_DIRS=") and "env:REPO=unset" not in line for line in prep), prep
    assert not [line for line in calls if "--export" in line], calls


#: A stub squeue: this user's queued prep jobs of the experiment ``wave`` -- the prepare job (renders numba,c) and the
#: c golden job -- as ``<id> <comment>``, for the job name after ``-n``, queued from the submitting checkout (``OPT``).
QUEUED_SQUEUE = """name=""
while [ $# -gt 0 ]; do [ "$1" = -n ] && name="$2"; shift; done
commit=$(git -C "${OPT}" rev-parse --short=9 HEAD 2>/dev/null || echo unknown)
case "${name}" in
    prepare-wave) echo "55 renders=numba,c;commit=${commit}" ;;
    goldens-wave-c) echo "56 goldens;commit=${commit}" ;;
esac
"""


def test_a_second_submit_waits_on_the_prep_jobs_the_first_queued(tmp_path: pathlib.Path) -> None:
    root = tree(tmp_path)
    bash_stub(root / "bin", "sbatch", CHAIN_SBATCH)
    bash_stub(root / "bin", "squeue", QUEUED_SQUEUE)
    edf = root / "edf"
    edf.mkdir()
    (edf / "hpcagent-bench-judge-mi300-latest.toml").write_text('image = "judge.sqsh"\n')
    knobs = {"KERNELS_FILE": "subset.txt", "LANGUAGES": "c hip", "PREPARE": "1", "EDF_PATH": str(edf)}
    done = submit(root, "--account", "project", **knobs, DEPEND_ON="77", SUBMIT="1")
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert "queued prepare-wave -> 55: renders numba,c (reused)" in lines
    assert "queued goldens-wave-c -> 56 (reused)" in lines
    assert [line for line in lines if line.startswith("submitted goldens-")] == [
        "submitted goldens-wave-hip -> 101;cluster"
    ]
    chain = [line.split() for line in (root / "sbatch.chain").read_text().splitlines()]
    assert {dependency for _, _, dependency in chain} == {"afterany:77,afterok:55:56:101"}


def slurm_outcome(dependency: str, states: dict[str, str]) -> str:
    """What Slurm does with a job of ``--dependency=<dependency>`` once the jobs it names ended as ``states``: ``type:id
    [:id...]`` clauses joined by a comma all hold, and an afterok on a job that did not complete cancels the job."""
    verdict = "RUN"
    for clause in dependency.split(","):
        kind, *ids = clause.split(":")
        assert kind in ("afterany", "afterok") and ids, dependency
        assert not {"afterany", "afterok"} & set(ids), f"not Slurm's grammar: {dependency}"
        if kind == "afterok" and any(states[job] != "COMPLETED" for job in ids):
            verdict = "CANCELLED"
    return verdict


#: One line of a dry run per job it does not submit, with its dependency.
DRY_JOB = re.compile(
    r"^(?:prepared (?P<setup>\S+)|goldens: (?P<lang>\S+)) .* after (?P<dependency>\S+) -- not submitted$"
)


@pytest.mark.parametrize(
    ("failed", "survivors"),
    [("prepare-wave", 0), ("goldens-wave-c", 0), ("goldens-wave-hip", 0), ("wave-qwen38-c", 2)],
    ids=["prepare", "first-golden", "last-golden", "wave-job"],
)
def test_the_dependencies_a_dry_run_prints_stop_every_wave_on_a_failed_prep_job(
    tmp_path: pathlib.Path, failed: str, survivors: int
) -> None:
    """Replay of the printed dependencies in submission order over one job failing: a failed prepare or golden job
    cancels the wave after the first too, which afterany on the cancelled wave alone would have started; a failed wave
    job stops nothing."""
    root = tree(tmp_path)
    knobs = {"KERNELS_FILE": "subset.txt", "LANGUAGES": "c hip", "PREPARE": "1"}
    dry = submit(root, **knobs)
    assert dry.returncode == 0, dry.stderr
    nodes = re.search(r"^prepared \S+ \((\d+) nodes", dry.stdout, re.MULTILINE)
    assert nodes is not None, dry.stdout
    dry = submit(root, **knobs, MAX_NODES=nodes[1])
    assert dry.returncode == 0, dry.stderr
    states = {"prepare-wave": "FAILED" if failed == "prepare-wave" else "COMPLETED"}
    waves = {}
    for line in dry.stdout.splitlines():
        if (job := DRY_JOB.match(line)) is None:
            continue
        name = job["setup"] or f"goldens-wave-{job['lang']}"
        outcome = slurm_outcome(job["dependency"], states)
        states[name] = "FAILED" if name == failed else {"RUN": "COMPLETED", "CANCELLED": "CANCELLED"}[outcome]
        if job["setup"]:
            waves[name] = outcome
    assert len(waves) == 2, dry.stdout
    assert sum(outcome == "RUN" for outcome in waves.values()) == survivors, waves


def transient_then_ok_sbatch(failures: int, message: str) -> str:
    """A stub sbatch failing with ``message`` its first ``failures`` calls, counted in ``sbatch.count``."""
    return f"""n=$(( $(cat "${{STUB_MARKERS}}/sbatch.count" 2>/dev/null || echo 0) + 1 ))
echo "${{n}}" > "${{STUB_MARKERS}}/sbatch.count"
if (( n <= {failures} )); then echo "sbatch: error: {message}" >&2; exit 1; fi
echo "4242;cluster"
"""


def retried(tmp_path: pathlib.Path, failures: int, message: str) -> tuple[subprocess.CompletedProcess[str], int]:
    """``sbatch_retry`` under the stub, without its waits; the result and the sbatch calls made."""
    bash_stub(tmp_path / "bin", "sbatch", transient_then_ok_sbatch(failures, message))
    bash_stub(tmp_path / "bin", "sleep", "exit 0")
    done = run_bash(tmp_path, 'jid=$(sbatch_retry --parsable job.sbatch) || exit $?\necho "jid=${jid}"')
    return done, int((tmp_path / "sbatch.count").read_text())


def test_a_transient_sbatch_failure_is_retried_and_only_the_id_is_captured(tmp_path: pathlib.Path) -> None:
    done, calls = retried(tmp_path, 2, "cli_filter plugin terminated with error")
    assert (done.returncode, calls) == (0, 3), done.stderr
    assert "jid=4242;cluster" in done.stdout.splitlines()
    assert "cli_filter plugin terminated with error" in done.stderr


@pytest.mark.parametrize(
    "message",
    [
        "Batch job submission failed: Invalid account or account/partition combination specified",
        "Batch job submission failed: Invalid partition name specified",
        "Batch job submission failed: Access/permission denied",
        "Batch job submission failed: Socket timed out on send/recv operation",
    ],
)
def test_a_failure_a_retry_cannot_fix_or_may_duplicate_fails_at_once(tmp_path: pathlib.Path, message: str) -> None:
    done, calls = retried(tmp_path, 9, message)
    assert (done.returncode, calls) == (1, 1), done.stderr
    assert message in done.stderr


def test_a_transient_failure_that_persists_gives_up_with_its_status(tmp_path: pathlib.Path) -> None:
    done, calls = retried(tmp_path, 9, "cli_filter plugin terminated with error")
    assert (done.returncode, calls) == (1, 4), done.stderr


def test_a_setup_job_goes_through_the_retry(tmp_path: pathlib.Path) -> None:
    bash_stub(tmp_path / "bin", "sleep", "exit 0")
    (tmp_path / "bin").mkdir(exist_ok=True)
    done = submit_three_setups(tmp_path, SUBMIT="1", FAIL_FIRST="1")
    assert done.returncode == 0, done.stderr
    assert "cli_filter" in done.stderr
    assert [line.split()[1] for line in (tmp_path / "sbatch.chain").read_text().splitlines()] == ["s1", "s2", "s3"]


def test_a_deadline_already_past_is_refused_before_anything_is_staged(tmp_path: pathlib.Path) -> None:
    done = run_bash(tmp_path, 'deadline_setup "2000-01-01 00:00:00 UTC" 300')
    assert done.returncode == 2
    assert "has passed" in done.stderr


@pytest.mark.parametrize("failing", ["squeue", "sacct"])
def test_a_queue_guard_that_cannot_look_says_so(tmp_path: pathlib.Path, failing: str) -> None:
    bash_stub(tmp_path / "bin", "squeue", "echo 7")
    bash_stub(tmp_path / "bin", failing, "exit 1")
    if failing == "squeue":
        bash_stub(tmp_path / "bin", "sacct", "exit 0")
    done = run_bash(tmp_path, f"refuse_if_queue_references {tmp_path}/.env.x")
    assert done.returncode == 0, done.stderr
    assert f"{failing} failed; NOT checking" in done.stderr


def test_a_pin_that_cannot_read_its_file_leaves_it_alone(tmp_path: pathlib.Path) -> None:
    env = tmp_path / ".env.x"
    env.write_text("KEY=old\nOTHER=kept\n")
    bash_stub(tmp_path / "bin", "awk", "exit 2")
    done = run_bash(tmp_path, f". {CLUSTER_DIR / 'pin_env_kv.sh'}\npin_env_kv {env} KEY=new || exit $?")
    assert done.returncode == 2
    assert env.read_text() == "KEY=old\nOTHER=kept\n"
    assert [path.name for path in tmp_path.glob(".env.x*")] == [".env.x"]


def test_the_root_account_submits_no_prepare_job_either(tmp_path: pathlib.Path) -> None:
    root = tree(tmp_path)
    bash_stub(root / "bin", "sbatch", CHAIN_SBATCH)
    done = submit(root, "--account", "root", KERNELS_FILE="subset.txt", PREPARE="1", SUBMIT="1")
    assert done.returncode == 2, done.stderr
    assert "root is not a project account" in done.stderr, done.stderr
    assert not (root / "sbatch.last").exists()


@pytest.mark.smoke
def test_submit_sh_splits_a_study_into_chained_waves(tmp_path: pathlib.Path) -> None:
    """submit.sh with MAX_NODES at two setups' nodes: four setups go out as two waves of two, the second wave
    afterany on both jobs of the first."""
    root = tree(tmp_path)
    bash_stub(root / "bin", "sbatch", CHAIN_SBATCH)
    knobs = {"KERNELS_FILE": "subset.txt", "LANGUAGES": "c hip", "PACKETS": "none lang-skills"}
    dry = submit(root, **knobs)
    assert dry.returncode == 0, dry.stderr
    nodes = {int(count) for count in re.findall(r"^prepared \S+ \((\d+) nodes", dry.stdout, re.MULTILINE)}
    assert len(nodes) == 1, dry.stdout
    max_nodes = str(2 * nodes.pop())

    done = submit(root, "--account", "project", **knobs, MAX_NODES=max_nodes, DEPEND_ON="77", SUBMIT="1")
    assert done.returncode == 0, done.stderr
    headers = [line for line in done.stdout.splitlines() if line.startswith("wave ")]
    assert headers == [
        f"wave 1/2: {max_nodes} nodes (MAX_NODES={max_nodes}), after 77",
        f"wave 2/2: {max_nodes} nodes (MAX_NODES={max_nodes}), after wave 1 (101:102)",
    ]
    chain = [line.split() for line in (root / "sbatch.chain").read_text().splitlines()]
    assert [(job, dependency) for job, _, dependency in chain] == [
        ("101", "afterany:77"),
        ("102", "afterany:77"),
        ("103", "afterany:101:102"),
        ("104", "afterany:101:102"),
    ]
