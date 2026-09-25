# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``experiments/run_cluster.sh``'s FROZEN TREE block: a batch step copies the checkout once and
re-executes from the copy, so a commit landing mid-run never reaches the job (643369: every /score
died on "cannot import name 'decline_kind'"). The block is lifted from the real file and run in a
throwaway git checkout (with the real scripts/cscs/code_snapshot.sh) whose run_cluster.sh stops
right after it. What the copy holds is tests/test_code_snapshot.py's subject.

The FROZEN TREE REMOVAL block deletes that copy at teardown. Its tests run the freeze, the removal
block and the real step teardown (cleanup_steps_on_exit / cleanup_steps_on_signal) inside the same
brace group the real file uses, with a stubbed ``srun`` whose step keeps reading the copy until it
is stopped -- and reads it once more a second AFTER its TERM, so a removal that raced a running step
shows up as a missing read."""

import os
import pathlib
import shutil
import signal
import subprocess
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "experiments" / "run_cluster.sh").read_text()
START = "# FROZEN TREE."
END = "    export HPCAGENT_BENCH_FROZEN=live\nfi\n"
BLOCK = TEXT[TEXT.index(START) : TEXT.index(END) + len(END)]
TEARDOWN_START = "# FROZEN TREE REMOVAL."
TEARDOWN = TEXT[TEXT.index(TEARDOWN_START) : TEXT.index(': "${SLURM_JOB_ID:?')]
CLEANUP_START = "step_pids=()\n"
CLEANUP_END = "trap cleanup_steps_on_signal INT TERM\n"
CLEANUP = TEXT[TEXT.index(CLEANUP_START) : TEXT.index(CLEANUP_END) + len(CLEANUP_END)]
REPORT = (
    'echo "ran from ${SCRIPT_DIR} repo=${HPCAGENT_BENCH_REPO:-} marker=${HPCAGENT_BENCH_FROZEN:-}'
    " commit=${HPCAGENT_BENCH_SNAPSHOT_COMMIT:-}"
    ' packs=${PACK_ROOT:-} matrices=${HPCAGENT_BENCH_CACHE_DIR:-} generated=${HPCAGENT_BENCH_GENERATED_CACHE_HOST:-}"\n'
    'echo "site=${HPCAGENT_BENCH_SITE_ENV:-} cluster_env=${CLUSTER_ENV_FILE:-} gone=${GONE:-}"\n'
)


GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
    "GIT_CONFIG_NOSYSTEM": "1",
    "HOME": "/nonexistent",
}


def git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        env={"PATH": "/usr/bin:/bin", **GIT_ENV},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


SCRIPT_DIR_LINE = 'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'

#: A job's steps, as the teardown tests run them: a role step (``srun``, stubbed to run in place)
#: that reads its tree until stopped, a preparation stand-in run in the FOREGROUND the way
#: prepare_job.sh is, and an extraction stand-in that reads the tree after the steps are up.
JOB = (
    """if [[ -n "${PREPARE:-}" ]]; then
    srun bash "${SCRIPT_DIR}/step.sh"
fi
"""
    + CLEANUP
    + """srun bash "${SCRIPT_DIR}/step.sh" &
step_pids+=("$!")
until grep -q "^started ${SCRIPT_DIR}" "${LOG}" 2>/dev/null; do sleep 0.1; done
cat "${SCRIPT_DIR}/../hpcagent_bench/module.py" >/dev/null
echo "extracted from ${SCRIPT_DIR}" >>"${LOG}"
if [[ "${CRASH:-}" == 1 ]]; then
    false
fi
if [[ -n "${HOLD:-}" ]]; then
    set +e
    wait -n
    set -e
fi
exit "${STATUS:-0}"
"""
)

#: The role step: reads its tree every 0.1 s; on TERM it waits a second, reads the tree ONCE MORE and
#: records whether it could -- a removal that did not wait for it makes that read fail.
STEP = """here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
trap 'sleep 1; if cat "${here}/hpcagent_bench/module.py" >/dev/null; then echo "read after TERM ${here}" >>"${LOG}"; fi; exit 0' TERM
echo "started ${here}/experiments" >>"${LOG}"
while :; do
    cat "${here}/hpcagent_bench/module.py" >/dev/null
    sleep 0.1 &
    wait "$!"
done
"""


def checkout(live: pathlib.Path, commit: bool = True, teardown: bool = False) -> pathlib.Path:
    """A throwaway checkout whose run_cluster.sh is the real FROZEN TREE block and, with
    ``teardown``, the real removal block and step teardown plus :data:`JOB`, all inside the brace
    group the real file wraps itself in."""
    script = live / "experiments" / "run_cluster.sh"
    script.parent.mkdir(parents=True)
    if teardown:
        body = "{\n" + SCRIPT_DIR_LINE + BLOCK + REPORT + TEARDOWN + JOB + "}\n"
        (live / "experiments" / "step.sh").write_text(STEP)
    else:
        body = SCRIPT_DIR_LINE + BLOCK + REPORT
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    (live / "scripts" / "cscs").mkdir(parents=True)
    for name in ("code_snapshot.sh", "frozen_store.py"):
        shutil.copy2(REPO / "scripts" / "cscs" / name, live / "scripts" / "cscs" / name)
    (live / "hpcagent_bench").mkdir()
    (live / "hpcagent_bench" / "module.py").write_text("OLD = 1\n")
    (live / ".gitignore").write_text("core_*\n")
    (live / "core_nid0001_1").write_text("dump")
    if commit:
        git(live.parent, "init", "-q", str(live))
        git(live, "add", "-A")
        git(live, "commit", "-q", "-m", "c")
    return script


def launch(script: pathlib.Path, env: dict[str, str], path: str = "/usr/bin:/bin") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script)], env={"PATH": path, **GIT_ENV, **env}, capture_output=True, text=True, check=True
    )


def fake_rsync(bin_dir: pathlib.Path, rc: int) -> str:
    """A PATH whose rsync does the real copy, then exits ``rc``."""
    bin_dir.mkdir()
    (bin_dir / "rsync").write_text(f'#!/bin/bash\n{shutil.which("rsync")} "$@"\nexit {rc}\n')
    (bin_dir / "rsync").chmod(0o755)
    return f"{bin_dir}:/usr/bin:/bin"


def test_a_batch_step_runs_from_a_copy_beside_its_campaign_that_later_edits_cannot_reach(
    tmp_path: pathlib.Path,
) -> None:
    live, runs = tmp_path / "live", tmp_path / "runs" / "campaign"
    script = checkout(live)
    head = git(live, "rev-parse", "--short", "HEAD")
    out = launch(script, {"SLURM_JOB_ID": "123", "RUN_ROOT": str(runs)}).stdout
    frozen = tmp_path / "runs" / ".frozen" / "job-123"
    assert out.count("frozen tree") == 1, out
    assert f"ran from {frozen}/experiments repo={frozen} marker={frozen} commit={head}" in out, out
    assert f"packs={live}/.cache/packs matrices={live}/hpcagent_bench/.hpcagent_bench_cache" in out, out
    assert f"generated={live}/.cache/generated" in out, out
    (live / "hpcagent_bench" / "module.py").write_text("NEW = 1\n")
    assert (frozen / "hpcagent_bench" / "module.py").read_text() == "OLD = 1\n"
    assert not (frozen / ".git").exists() and not (frozen / "core_nid0001_1").exists()
    assert not runs.exists(), "nothing lands under RUN_ROOT, where extraction globs for databases"


def test_a_variable_naming_a_path_of_the_checkout_names_it_in_the_copy(tmp_path: pathlib.Path) -> None:
    """The judge container mounts only the copy: a site layer exported as a live path aborts
    scripts/site_env.sh there. A path the copy holds (tracked, or an untracked arm snapshot) is
    renamed into it; a data root the copy leaves out, and one it never had, keep their live paths."""
    live, runs = tmp_path / "live", tmp_path / "runs" / "campaign"
    script = checkout(live)
    (live / "experiments" / "layers").mkdir()
    (live / "experiments" / "layers" / "site-cscs.env").write_text('SITE="${SITE:-x}"\n')
    git(live, "add", "-A")
    git(live, "commit", "-q", "-m", "site layer")
    (live / "experiments" / ".rendered").mkdir()
    (live / "experiments" / ".rendered" / "arm.env").write_text("CAMPAIGN_ARM=a\n")
    (live / "hpcagent_bench" / ".hpcagent_bench_cache").mkdir()
    env = {
        "SLURM_JOB_ID": "123",
        "RUN_ROOT": str(runs),
        "HPCAGENT_BENCH_SITE_ENV": f"{live}/experiments/layers/site-cscs.env",
        "CLUSTER_ENV_FILE": f"{live}/experiments/.rendered/arm.env",
        "HPCAGENT_BENCH_CACHE_DIR": f"{live}/hpcagent_bench/.hpcagent_bench_cache",
        "GONE": f"{live}/never/there",
    }
    out = launch(script, env).stdout
    frozen = tmp_path / "runs" / ".frozen" / "job-123"
    assert f"site={frozen}/experiments/layers/site-cscs.env" in out, out
    assert f"cluster_env={frozen}/experiments/.rendered/arm.env" in out, out
    assert f"matrices={live}/hpcagent_bench/.hpcagent_bench_cache" in out, out
    assert f"gone={live}/never/there" in out, out


def test_a_step_that_inherits_the_frozen_tree_does_not_copy_again(tmp_path: pathlib.Path) -> None:
    live, runs = tmp_path / "live", tmp_path / "runs" / "campaign"
    env = {"SLURM_JOB_ID": "123", "RUN_ROOT": str(runs), "HPCAGENT_BENCH_FROZEN": "/elsewhere"}
    out = launch(checkout(live), env).stdout
    assert "frozen tree" not in out and f"ran from {live}/experiments" in out, out
    assert not (tmp_path / "runs").exists()


def test_the_kill_switch_runs_the_job_on_the_live_tree(tmp_path: pathlib.Path) -> None:
    """HPCAGENT_BENCH_FROZEN=live, from the submit env or the arm's .env, skips the copy."""
    live, runs = tmp_path / "live", tmp_path / "runs" / "campaign"
    env = {"SLURM_JOB_ID": "123", "RUN_ROOT": str(runs), "HPCAGENT_BENCH_FROZEN": "live"}
    out = launch(checkout(live), env).stdout
    assert "frozen tree" not in out and f"ran from {live}/experiments repo= marker=live commit=" in out, out
    assert not (tmp_path / "runs").exists()


def test_a_failed_copy_runs_the_job_on_the_live_tree_instead_of_killing_it(tmp_path: pathlib.Path) -> None:
    live, runs = tmp_path / "live", tmp_path / "runs" / "campaign"
    result = launch(checkout(live), {"SLURM_JOB_ID": "123", "RUN_ROOT": str(runs)}, fake_rsync(tmp_path / "bin", 23))
    assert "WARNING: could not freeze" in result.stderr, result.stderr
    assert f"ran from {live}/experiments repo= marker=live commit=" in result.stdout, result.stdout


def test_a_checkout_git_cannot_read_runs_the_job_on_the_live_tree(tmp_path: pathlib.Path) -> None:
    live, runs = tmp_path / "live", tmp_path / "runs" / "campaign"
    result = launch(checkout(live, commit=False), {"SLURM_JOB_ID": "123", "RUN_ROOT": str(runs)})
    assert "WARNING: could not freeze" in result.stderr, result.stderr
    assert "marker=live" in result.stdout, result.stdout


def test_a_file_vanishing_mid_copy_still_freezes(tmp_path: pathlib.Path) -> None:
    live, runs = tmp_path / "live", tmp_path / "runs" / "campaign"
    out = launch(checkout(live), {"SLURM_JOB_ID": "123", "RUN_ROOT": str(runs)}, fake_rsync(tmp_path / "bin", 24))
    assert f"marker={tmp_path}/runs/.frozen/job-123" in out.stdout, out.stdout


def test_a_run_root_inside_the_checkout_is_not_copied_into_itself(tmp_path: pathlib.Path) -> None:
    live = tmp_path / "live"
    result = launch(checkout(live), {"SLURM_JOB_ID": "123", "RUN_ROOT": str(live / "runs" / "campaign")})
    assert "is inside" in result.stderr and "marker=live" in result.stdout, result
    assert not (live / "runs").exists()


# ---- FROZEN TREE REMOVAL ------------------------------------------------------------------------


def test_the_real_file_is_one_brace_group_that_ends_in_exit_and_removes_the_copy_last() -> None:
    """What makes removing the tree the script was read from safe: bash parsed the whole group before
    running any of it and never reads the file again, and the removal is the EXIT trap's last act,
    after the `wait` that reaps every step and after the role dispatch, which no role step passes."""
    code = [line for line in TEXT.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    assert code[:2] == ["set -euo pipefail", "{"], code[:2]
    assert TEXT.endswith('\nexit "${agent_status}"\n}\n')
    exit_trap = CLEANUP[CLEANUP.index("cleanup_steps_on_exit() {") :]
    exit_trap = exit_trap[: exit_trap.index("\n}\n")]
    assert exit_trap.rstrip().endswith("remove_frozen_tree"), exit_trap
    assert exit_trap.index("    wait 2>/dev/null || true") < exit_trap.index("remove_frozen_tree")
    assert TEXT.index("    --agent-node)\n") < TEXT.index(TEARDOWN_START) < TEXT.index(CLEANUP_START)


def srun_stub(bin_dir: pathlib.Path) -> str:
    """A PATH whose srun runs its command in place, the way a step's tasks run the tree's files."""
    bin_dir.mkdir()
    (bin_dir / "srun").write_text('#!/bin/bash\nexec "$@"\n')
    (bin_dir / "srun").chmod(0o755)
    return f"{bin_dir}:/usr/bin:/bin"


def neighbours(tmp_path: pathlib.Path) -> dict[pathlib.Path, str]:
    """Another job's copy and a stray file beside this job's, which no removal may touch."""
    other = tmp_path / "runs" / ".frozen" / "job-122"
    (other / "hpcagent_bench").mkdir(parents=True)
    files = {other / "hpcagent_bench" / "module.py": "OTHER = 1\n", tmp_path / "runs" / ".frozen" / "note": "keep\n"}
    for path, text in files.items():
        path.write_text(text)
    return files


def tree(root: pathlib.Path) -> dict[str, bytes]:
    """Every file under ``root`` but its .git, by relative path."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }


def start(script: pathlib.Path, env: dict[str, str], path: str) -> subprocess.Popen[str]:
    """The job in its own session, so a TERM can reach its whole process group as Slurm's does."""
    return subprocess.Popen(
        ["bash", str(script)],
        env={"PATH": path, **GIT_ENV, **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def await_line(log: pathlib.Path, prefix: str) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if log.exists() and any(line.startswith(prefix) for line in log.read_text().splitlines()):
            return
        time.sleep(0.1)
    raise AssertionError(f"{prefix!r} never reached {log}")


def finish(proc: subprocess.Popen[str]) -> tuple[int, str, str]:
    try:
        out, err = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
    return proc.returncode, out, err


def job(tmp_path: pathlib.Path, **env: str) -> tuple[pathlib.Path, dict[str, str], str]:
    """A committed throwaway checkout running the teardown flow, its environment and its PATH."""
    live = tmp_path / "live"
    script = checkout(live, teardown=True)
    base = {"SLURM_JOB_ID": "123", "RUN_ROOT": str(tmp_path / "runs" / "campaign"), "LOG": str(tmp_path / "log")}
    return script, {**base, **env}, srun_stub(tmp_path / "bin")


def assert_removed_after_its_steps(tmp_path: pathlib.Path, err: str, before: dict[str, bytes]) -> None:
    frozen = tmp_path / "runs" / ".frozen" / "job-123"
    log = (tmp_path / "log").read_text()
    assert f"started {frozen}/experiments" in log, log
    assert f"read after TERM {frozen}" in log, f"a step lost its tree before it stopped:\n{log}\n{err}"
    assert not frozen.exists(), "the copy survived the job"
    assert "No such file" not in err and "syntax error" not in err, err
    assert tree(tmp_path / "live") == before, "the live checkout changed"


@pytest.mark.parametrize(("env", "status"), [({}, 0), ({"STATUS": "3"}, 3), ({"CRASH": "1"}, 1)])
def test_the_copy_is_removed_once_its_steps_are_reaped_whichever_way_the_job_exits(
    tmp_path: pathlib.Path, env: dict[str, str], status: int
) -> None:
    """A normal end, a failing `exit`, and a `set -e` crash: the copy goes, after the step that was
    still reading it has stopped, and nothing beside it goes with it."""
    kept = neighbours(tmp_path)
    script, job_env, path = job(tmp_path, **env)
    before = tree(tmp_path / "live")
    code, out, err = finish(start(script, job_env, path))
    assert code == status, (out, err)
    frozen = tmp_path / "runs" / ".frozen" / "job-123"
    assert out.count(f"frozen tree {frozen}") == 1, out
    assert f"extracted from {frozen}/experiments" in (tmp_path / "log").read_text()
    assert_removed_after_its_steps(tmp_path, err, before)
    assert {path: path.read_text() for path in kept} == kept


def test_a_term_while_the_steps_run_removes_the_copy_after_the_teardown(tmp_path: pathlib.Path) -> None:
    """scancel or the time limit: TERM reaches every process of the batch step; the copy goes once
    the steps are reaped and the script has run on to its own `exit`."""
    kept = neighbours(tmp_path)
    script, job_env, path = job(tmp_path, HOLD="1")
    before = tree(tmp_path / "live")
    proc = start(script, job_env, path)
    await_line(tmp_path / "log", "extracted from")
    os.killpg(proc.pid, signal.SIGTERM)
    code, out, err = finish(proc)
    assert code == 0, (out, err)
    assert_removed_after_its_steps(tmp_path, err, before)
    assert {path: path.read_text() for path in kept} == kept


def test_a_term_during_preparation_waits_for_the_foreground_step_before_removing(tmp_path: pathlib.Path) -> None:
    """Before the role steps start, the step in flight is prepare_job.sh's, in the foreground: the
    copy goes only once it has returned, and the job exits 143 as a TERM would have it."""
    kept = neighbours(tmp_path)
    script, job_env, path = job(tmp_path, PREPARE="1")
    before = tree(tmp_path / "live")
    proc = start(script, job_env, path)
    await_line(tmp_path / "log", "started")
    os.killpg(proc.pid, signal.SIGTERM)
    code, out, err = finish(proc)
    assert code == 143, (out, err)
    assert "extracted" not in (tmp_path / "log").read_text()
    assert_removed_after_its_steps(tmp_path, err, before)
    assert {path: path.read_text() for path in kept} == kept


def decoy(tmp_path: pathlib.Path) -> pathlib.Path:
    """A tree at exactly this job's copy path that this run must NOT have made, so must not remove."""
    path = tmp_path / "runs" / ".frozen" / "job-123" / "hpcagent_bench" / "module.py"
    path.parent.mkdir(parents=True)
    path.write_text("DECOY = 1\n")
    return path


@pytest.mark.parametrize(
    ("env", "rsync_rc"),
    [({"HPCAGENT_BENCH_FROZEN": "live"}, 0), ({}, 23)],
    ids=["kill-switch", "failed-copy"],
)
def test_a_job_on_the_live_tree_removes_nothing(tmp_path: pathlib.Path, env: dict[str, str], rsync_rc: int) -> None:
    """HPCAGENT_BENCH_FROZEN=live and a copy that failed both run on the live checkout: no copy is
    removed -- not even a tree already at this job's copy path -- and the checkout is left as it was."""
    kept = neighbours(tmp_path)
    marker = decoy(tmp_path)
    script, job_env, _ = job(tmp_path, **env)
    path = srun_stub(tmp_path / "srun-bin")
    if rsync_rc:
        path = f"{tmp_path / 'srun-bin'}:{fake_rsync(tmp_path / 'rsync-bin', rsync_rc)}"
    before = tree(tmp_path / "live")
    code, out, err = finish(start(script, job_env, path))
    assert code == 0, (out, err)
    live = tmp_path / "live"
    assert "marker=live" in out and f"extracted from {live}/experiments" in (tmp_path / "log").read_text()
    assert f"read after TERM {live}" in (tmp_path / "log").read_text()
    assert marker.read_text() == "DECOY = 1\n"
    assert tree(live) == before and (live / ".git").is_dir()
    assert {path: path.read_text() for path in kept} == kept


def test_a_process_that_does_not_run_from_the_copy_never_removes_it(tmp_path: pathlib.Path) -> None:
    """A role step inherits HPCAGENT_BENCH_FROZEN and the snapshot commit, but runs from elsewhere:
    the path matches this job's copy exactly, and it is still not this process's to remove."""
    marker = decoy(tmp_path)
    frozen = marker.parent.parent
    script, job_env, path = job(tmp_path, HPCAGENT_BENCH_FROZEN=str(frozen), HPCAGENT_BENCH_SNAPSHOT_COMMIT="abc")
    code, out, err = finish(start(script, job_env, path))
    assert code == 0, (out, err)
    assert "frozen tree" not in out and marker.read_text() == "DECOY = 1\n"
