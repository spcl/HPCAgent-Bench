# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""canon_column.sh must not abort with "SCRATCH: parameter null or not set" when $SCRATCH is
unset but $HPCAGENT_BENCH_REPO is -- the shape of a container test run (scripts/run_tests.sh
--container), which has no $SCRATCH mount. This drives `inner` directly, the same way
test_canon_column_kernel_timeout.py and test_canon_column_zero_kernel_rank.py do, with neither
`opt` (arg 6) nor DACE_TREE given and SCRATCH unset: `opt` falls back to HPCAGENT_BENCH_REPO and
dace is the image's checkout (DACE_DIR here), refreshed by containers/images/dace_refresh.sh.
"""

import os
import pathlib
import shutil
import subprocess

from hpcagent_bench import paths
from tests.fake_checkout import install_repo_env

CANON_COLUMN = paths.ROOT / "experiments" / "canon_column.sh"


def stub_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """A checkout-shaped tree with just enough for `inner` to run one real kernel: a no-op
    ``scripts/cache_env.sh`` and a stub ``hpcagent_bench.cli`` that exits 0 immediately.

    Also carries the real ``containers/images/dace_refresh.sh``, which `inner` runs when no
    DACE_TREE is given."""
    repo = tmp_path / "hpcagent-bench"
    (repo / "scripts").mkdir(parents=True)
    (repo / "containers" / "images").mkdir(parents=True)
    shutil.copy2(paths.ROOT / "containers" / "images" / "dace_refresh.sh", repo / "containers" / "images")
    (repo / "scripts" / "cache_env.sh").write_text("# stub cache_env.sh for this test, no-op\n")
    install_repo_env(repo)
    pkg = repo / "hpcagent_bench"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "cli.py").write_text("if __name__ == '__main__':\n    pass\n")
    return repo


def image_dace(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str]:
    """A git checkout holding an importable ``dace``, standing in for the image's /opt/dace, and
    its HEAD commit (pinned below, so the refresh fetches nothing)."""
    dace_dir = tmp_path / "opt-dace"
    (dace_dir / "dace").mkdir(parents=True)
    (dace_dir / "dace" / "__init__.py").write_text("")
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.org")
    env |= {"GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.org"}
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "stub"]):
        subprocess.run(["git", "-C", str(dace_dir), *args], env=env, check=True, capture_output=True)
    head = subprocess.run(
        ["git", "-C", str(dace_dir), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return dace_dir, head


def test_inner_resolves_opt_from_hpcagent_bench_repo_and_dace_from_the_image_with_no_scratch(
    tmp_path: pathlib.Path,
) -> None:
    out_root = tmp_path / "out"
    out_root.mkdir()
    repo = stub_repo(tmp_path)
    dace_dir, head = image_dace(tmp_path)

    env = dict(os.environ, SLURM_PROCID="0", SLURM_NTASKS="1")
    env.pop("SCRATCH", None)
    env.pop("DACE_TREE", None)
    env["HPCAGENT_BENCH_REPO"] = str(repo)
    env |= {"DACE_DIR": str(dace_dir), "HPCAGENT_BENCH_DACE_REF": head}

    # Neither `opt` (arg 6) nor DACE_TREE is passed: opt falls back to HPCAGENT_BENCH_REPO, dace to
    # the image's checkout at the job's commit.
    result = subprocess.run(
        ["bash", str(CANON_COLUMN), "inner", "stubcol", str(out_root), "onlykernel", "fuzzed"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert "SCRATCH: parameter null or not set" not in result.stderr, result.stderr
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert f"dace-refresh: live commit {head}" in result.stdout
    # the label finalize_column merges into canon.db's `build` column names that commit
    assert (out_root / "stubcol.rank0.dace").read_text().startswith(f"dace {head[:7]}")


def test_inner_still_refuses_when_neither_scratch_nor_hpcagent_bench_repo_is_set(
    tmp_path: pathlib.Path,
) -> None:
    """The fallback is SCRATCH else HPCAGENT_BENCH_REPO, not a silent third default -- a genuinely
    unconfigured shell must still fail loudly, with a message naming both variables, not a mystery
    "No such file or directory" three steps later."""
    out_root = tmp_path / "out"
    out_root.mkdir()

    env = dict(os.environ, SLURM_PROCID="0", SLURM_NTASKS="1")
    env.pop("SCRATCH", None)
    env.pop("HPCAGENT_BENCH_REPO", None)
    env.pop("DACE_TREE", None)

    result = subprocess.run(
        ["bash", str(CANON_COLUMN), "inner", "stubcol", str(out_root), "onlykernel", "fuzzed"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 2, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "SCRATCH" in result.stderr and "HPCAGENT_BENCH_REPO" in result.stderr, result.stderr
