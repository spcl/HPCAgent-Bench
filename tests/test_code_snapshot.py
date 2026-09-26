# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``experiments/code_snapshot.sh``: the copy of the checkout a cluster job runs from. The live
checkout is fast-forwarded while jobs queue and run, so the copy must hold ONE commit's tracked
files (never a working tree caught mid-checkout or carrying a hand edit), plus the untracked inputs
a job reads from the tree, and none of the caches, run output or dumps. Run on real throwaway git
checkouts."""

import pathlib
import shutil
import subprocess


REPO = pathlib.Path(__file__).resolve().parents[1]
SNAPSHOT = REPO / "experiments" / "code_snapshot.sh"
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


def write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def checkout(live: pathlib.Path) -> str:
    """A committed checkout with a gitignore; returns its short HEAD."""
    write(live / ".gitignore", "*_dace.py\n__pycache__/\n/results/\n/.cache/\n.ruff_cache/\ncore_*\n.env.*\n")
    write(live / "hpcagent_bench" / "module.py", "OLD = 1\n")
    write(live / "experiments" / "run_cluster.sh", "echo run\n")
    git(live.parent, "init", "-q", str(live))
    git(live, "add", "-A")
    git(live, "commit", "-q", "-m", "c")
    return git(live, "rev-parse", "--short", "HEAD")


def snapshot(live: pathlib.Path, dest: pathlib.Path, path: str = "/usr/bin:/bin") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SNAPSHOT), str(live), str(dest)],
        env={"PATH": path, **GIT_ENV},
        capture_output=True,
        text=True,
        check=False,
    )


def test_tracked_files_come_from_the_commit_and_untracked_inputs_from_the_tree(tmp_path: pathlib.Path) -> None:
    """A hand edit or a half-applied fast-forward in the live tree never reaches the copy; the
    generated siblings and arm envs git does not track do, and caches, run output and dumps do not."""
    live, dest = tmp_path / "live", tmp_path / "runs" / ".frozen" / "job-1"
    head = checkout(live)
    write(live / "hpcagent_bench" / "module.py", "EDITED = 1\n")
    write(live / "hpcagent_bench" / "k_dace.py", "generated\n")
    write(live / "experiments" / ".env.arm", "CAMPAIGN_ARM=arm\n")
    write(live / "experiments" / "problems-x.jsonl", "{}\n")
    for junk in (
        "hpcagent_bench/__pycache__/m.pyc",
        "results/r.db",
        ".cache/generated/g.c",
        ".ruff_cache/x",
        "core_nid1_2",
        "experiments/beverin-services-123.out",
        "experiments/mwd-final-v6/p.jsonl",
    ):
        write(live / junk, "junk\n")
    result = snapshot(live, dest)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == head
    assert (dest / "hpcagent_bench" / "module.py").read_text() == "OLD = 1\n"
    assert (dest / "hpcagent_bench" / "k_dace.py").read_text() == "generated\n"
    assert (dest / "experiments" / ".env.arm").exists() and (dest / "experiments" / "problems-x.jsonl").exists()
    copied = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
    assert copied == [
        ".gitignore",
        "experiments/.env.arm",
        "experiments/problems-x.jsonl",
        "experiments/run_cluster.sh",
        "hpcagent_bench/k_dace.py",
        "hpcagent_bench/module.py",
    ]
    assert not dest.with_name("job-1.partial").exists()


def test_a_tracked_file_deleted_mid_checkout_is_still_in_the_copy(tmp_path: pathlib.Path) -> None:
    live, dest = tmp_path / "live", tmp_path / "job-1"
    checkout(live)
    (live / "hpcagent_bench" / "module.py").unlink()
    assert snapshot(live, dest).returncode == 0
    assert (dest / "hpcagent_bench" / "module.py").read_text() == "OLD = 1\n"


def test_submodule_contents_are_copied_without_their_git_link(tmp_path: pathlib.Path) -> None:
    upstream, live, dest = tmp_path / "upstream", tmp_path / "live", tmp_path / "job-1"
    write(upstream / "src" / "model.py", "M = 1\n")
    git(tmp_path, "init", "-q", str(upstream))
    git(upstream, "add", "-A")
    git(upstream, "commit", "-q", "-m", "u")
    checkout(live)
    git(live, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(upstream), "third_party/up")
    git(live, "commit", "-q", "-m", "sub")
    result = snapshot(live, dest)
    assert result.returncode == 0, result.stderr
    assert (dest / "third_party" / "up" / "src" / "model.py").read_text() == "M = 1\n"
    assert not (dest / "third_party" / "up" / ".git").exists()


def test_a_requeued_job_replaces_its_old_copy_whole(tmp_path: pathlib.Path) -> None:
    """A NODE_FAIL requeue reuses the job id, so the copy of the first start must not survive in part."""
    live, dest = tmp_path / "live", tmp_path / "job-1"
    checkout(live)
    write(dest / "stale.py", "old run\n")
    write(dest.with_name("job-1.partial") / "half.py", "killed mid-copy\n")
    assert snapshot(live, dest).returncode == 0
    assert not (dest / "stale.py").exists() and (dest / "hpcagent_bench" / "module.py").exists()
    assert not dest.with_name("job-1.partial").exists()


def test_a_tree_without_git_fails_and_leaves_no_copy(tmp_path: pathlib.Path) -> None:
    live, dest = tmp_path / "live", tmp_path / "job-1"
    write(live / "hpcagent_bench" / "module.py", "OLD = 1\n")
    result = snapshot(live, dest)
    assert result.returncode != 0 and result.stdout == ""
    assert not dest.exists()


def test_a_failed_copy_of_the_untracked_inputs_fails_and_leaves_no_copy(tmp_path: pathlib.Path) -> None:
    live, dest, bin_dir = tmp_path / "live", tmp_path / "job-1", tmp_path / "bin"
    checkout(live)
    bin_dir.mkdir()
    (bin_dir / "rsync").write_text("#!/bin/bash\nexit 23\n")
    (bin_dir / "rsync").chmod(0o755)
    result = snapshot(live, dest, f"{bin_dir}:/usr/bin:/bin")
    assert result.returncode != 0 and "rsync" in result.stderr and result.stdout == ""
    assert not dest.exists() and not dest.with_name("job-1.partial").exists()


def test_a_file_vanishing_mid_copy_still_snapshots(tmp_path: pathlib.Path) -> None:
    """rsync 24: a file vanished between listing and copying (a cache entry replaced)."""
    live, dest, bin_dir = tmp_path / "live", tmp_path / "job-1", tmp_path / "bin"
    checkout(live)
    bin_dir.mkdir()
    (bin_dir / "rsync").write_text(f'#!/bin/bash\n{shutil.which("rsync")} "$@"\nexit 24\n')
    (bin_dir / "rsync").chmod(0o755)
    result = snapshot(live, dest, f"{bin_dir}:/usr/bin:/bin")
    assert result.returncode == 0, result.stderr
    assert (dest / "hpcagent_bench" / "module.py").exists()
