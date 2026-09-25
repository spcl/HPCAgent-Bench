# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""containers/images/dace_refresh.sh, the one dace refresh every job runs at start.

Driven against a local ``origin`` (a file:// repository standing in for spcl/dace) and a stub
``python3``, so no network and no pip install ever happen. The script under test is a copy placed
in a scratch tree next to its own ``pyproject.toml``, whose ``dace-pin`` is where ``pinned`` resolves.
"""

import os
import pathlib
import shutil
import subprocess
import tomllib

from hpcagent_bench import paths

SCRIPT = paths.ROOT / "containers" / "images" / "dace_refresh.sh"


def git(*args: str, cwd: pathlib.Path) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.org")
    env |= {"GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.org"}
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()


def commit(repo: pathlib.Path, text: str) -> str:
    (repo / "dace" / "__init__.py").write_text(f"__version__ = {text!r}\n")
    git("add", "-A", cwd=repo)
    git("commit", "-q", "-m", text, cwd=repo)
    return git("rev-parse", "HEAD", cwd=repo)


def setup(tmp_path: pathlib.Path, pin: str = "") -> tuple[pathlib.Path, pathlib.Path, list[str]]:
    """(the copied script, the checkout it refreshes, [first, second] origin commits on extended).

    The checkout is a clone at the FIRST commit; origin's extended has moved on to the second."""
    origin = tmp_path / "origin"
    (origin / "dace").mkdir(parents=True)
    git("init", "-q", "-b", "extended", cwd=origin)
    git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=origin)
    first = commit(origin, "first")
    checkout = tmp_path / "opt" / "dace"
    subprocess.run(["git", "clone", "-q", f"file://{origin}", str(checkout)], check=True)
    second = commit(origin, "second")
    tree = tmp_path / "tree"
    (tree / "containers" / "images").mkdir(parents=True)
    script = tree / "containers" / "images" / "dace_refresh.sh"
    shutil.copy2(SCRIPT, script)
    (tree / "pyproject.toml").write_text(f'[tool.hpcagent-bench]\ndace-pin = "{pin or first}"\n')
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python3").write_text('#!/bin/sh\necho "python3 $*" >> "$STUB_LOG"\n')
    (bindir / "python3").chmod(0o755)
    return script, checkout, [first, second]


def refresh(tmp_path: pathlib.Path, script: pathlib.Path, checkout: pathlib.Path, *args: str, ref: str | None = None):
    env = dict(os.environ, DACE_DIR=str(checkout), STUB_LOG=str(tmp_path / "stub.log"))
    env["HPCAGENT_BENCH_IMAGE_PYTHON"] = str(tmp_path / "bin" / "python3")
    env.pop("HPCAGENT_BENCH_DACE_REF", None)
    if ref is not None:
        env["HPCAGENT_BENCH_DACE_REF"] = ref
    return subprocess.run(["bash", str(script), *args], env=env, capture_output=True, text=True, timeout=120)


def test_a_job_moves_the_checkout_to_the_branch_tip_and_records_it(tmp_path: pathlib.Path) -> None:
    script, checkout, (first, second) = setup(tmp_path)
    assert git("rev-parse", "HEAD", cwd=checkout) == first

    done = refresh(tmp_path, script, checkout)

    assert done.returncode == 0, done.stderr
    assert git("rev-parse", "HEAD", cwd=checkout) == second
    assert done.stdout.splitlines()[-1] == f"dace-refresh: live commit {second}"
    assert pathlib.Path(f"{checkout}.commit").read_text().strip() == second
    # the editable install is redone with --no-deps, never a resolver run
    assert "-m pip install --no-cache-dir --no-deps -q -e" in (tmp_path / "stub.log").read_text()


def test_pinned_stays_on_the_pin_file_commit_with_no_fetch(tmp_path: pathlib.Path) -> None:
    script, checkout, commits = setup(tmp_path)
    first = commits[0]

    done = refresh(tmp_path, script, checkout, ref="pinned")

    assert done.returncode == 0, done.stderr
    assert git("rev-parse", "HEAD", cwd=checkout) == first
    assert done.stdout.splitlines()[-1] == f"dace-refresh: live commit {first}"
    assert not (tmp_path / "stub.log").exists()


def test_a_commit_sha_moves_the_checkout_to_exactly_that_commit(tmp_path: pathlib.Path) -> None:
    script, checkout, commits = setup(tmp_path)
    second = commits[1]

    done = refresh(tmp_path, script, checkout, ref=second)

    assert done.returncode == 0, done.stderr
    assert git("rev-parse", "HEAD", cwd=checkout) == second


def test_an_unreachable_pin_fails_but_an_unreachable_branch_keeps_the_baked_commit(tmp_path: pathlib.Path) -> None:
    script, checkout, commits = setup(tmp_path)
    first = commits[0]
    git("remote", "set-url", "origin", f"file://{tmp_path / 'gone'}", cwd=checkout)

    branch = refresh(tmp_path, script, checkout)
    pin = refresh(tmp_path, script, checkout, ref="0" * 40)

    assert branch.returncode == 0, branch.stderr
    assert "staying on the baked commit" in branch.stdout
    assert branch.stdout.splitlines()[-1] == f"dace-refresh: live commit {first}"
    assert pin.returncode == 1
    assert "unreachable" in pin.stderr
    assert git("rev-parse", "HEAD", cwd=checkout) == first


def test_resolve_prints_the_pin_and_refuses_a_placeholder_or_a_bad_ref(tmp_path: pathlib.Path) -> None:
    script, checkout, commits = setup(tmp_path)
    first = commits[0]
    assert refresh(tmp_path, script, checkout, "--resolve", ref="pinned").stdout.strip() == first
    assert refresh(tmp_path, script, checkout, "--resolve", ref="f" * 40).stdout.strip() == "f" * 40
    assert refresh(tmp_path, script, checkout, "--resolve", ref="not a ref").returncode == 2

    placeholder = setup(tmp_path / "placeholder", pin="PLACEHOLDER-for-the-release-commit")[0]
    done = refresh(tmp_path, placeholder, checkout, "--resolve", ref="pinned")
    assert done.returncode == 2
    assert "holds no commit sha" in done.stderr


def test_without_a_checkout_it_is_a_no_op_unless_a_commit_is_pinned(tmp_path: pathlib.Path) -> None:
    made = setup(tmp_path)
    script, first = made[0], made[2][0]
    missing = tmp_path / "no-dace"

    assert refresh(tmp_path, script, missing).returncode == 0
    assert refresh(tmp_path, script, missing, ref="pinned").returncode == 1
    assert refresh(tmp_path, script, missing, ref=first).returncode == 1


def test_pyproject_is_the_one_place_the_release_pin_is_written() -> None:
    """Installs and image builds read ``dace-pin`` from pyproject.toml; no other file spells it."""
    with (paths.ROOT / "pyproject.toml").open("rb") as handle:
        assert "dace-pin" in tomllib.load(handle)["tool"]["hpcagent-bench"]
    listed = subprocess.run(
        ["git", "-C", str(paths.ROOT), "grep", "-l", "-E", "^dace-pin ="],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    assert listed == ["pyproject.toml"], listed
