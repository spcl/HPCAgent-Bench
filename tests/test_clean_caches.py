# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``hpcagent-bench clean caches``: only allowlisted tool caches and idle build folders go; environments, the uv cache,
spack and image-build trees, run data, a running job's build folder and a symlink never do; a dry run removes nothing and
``clean uv`` is refused (the 2026-11-17 incident)."""

import contextlib
import io
import pathlib
import tempfile

import pytest

from hpcagent_bench.cluster import clean_caches


def tree(root: pathlib.Path, names: list[str]) -> None:
    for name in names:
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / "file").write_text("x")


KEEP = [
    "proj/.venv/lib/__pycache__",
    "x86_64/uv-cache/__pycache__",
    "spack/buildcache/__pycache__",
    "image-build-1116/.cache/__pycache__",
    "hpcagent-bench-runs/job/__pycache__",
    "archives/old/__pycache__",
    "aarch64/pkg/__pycache__",
    "proj/src/keep-me",
]
GO = [
    "proj/src/__pycache__",
    "proj/.ruff_cache",
    "proj/tests/.pytest_cache",
    "proj/.mypy_cache",
    "proj/build/dacecache_x",
]


def test_only_the_allowlisted_caches_are_candidates(tmp_path: pathlib.Path) -> None:
    tree(tmp_path, KEEP + GO)
    (tmp_path / "link").symlink_to(tmp_path / "proj" / "src" / "__pycache__")
    found = sorted(
        str(one.path.relative_to(tmp_path)) for one in clean_caches.candidates([tmp_path], protected=[], busy=[])
    )
    assert found == sorted(GO)


def test_a_protected_root_and_a_running_jobs_build_folder_are_left_alone(tmp_path: pathlib.Path) -> None:
    tree(tmp_path, GO)
    found = [
        one.path.relative_to(tmp_path).as_posix()
        for one in clean_caches.candidates(
            [tmp_path], protected=[tmp_path / "proj" / "tests"], busy=[tmp_path / "proj" / "build"]
        )
    ]
    assert sorted(found) == ["proj/.mypy_cache", "proj/.ruff_cache", "proj/src/__pycache__"]


def run(*args: str) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert clean_caches.main(list(args)) == 0
    return out.getvalue()


def test_a_dry_run_counts_and_removes_nothing_and_apply_removes_only_the_candidates(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree(tmp_path, KEEP + GO)
    monkeypatch.setattr(clean_caches, "running_workdirs", list)
    monkeypatch.setattr(clean_caches.data_guard, "protected_roots", tuple)
    shown = run("caches", "--root", str(tmp_path))
    assert "__pycache__: 1 folders, 2 inodes" in shown and "dry run" in shown
    assert all((tmp_path / name).exists() for name in GO)
    run("caches", "--root", str(tmp_path), "--apply")
    assert not any((tmp_path / name).exists() for name in GO)
    assert all((tmp_path / name / "file").exists() for name in KEEP if name != "proj/src/keep-me")


def test_clean_uv_is_refused() -> None:
    with pytest.raises(SystemExit, match="never cleaned"):
        clean_caches.main(["uv"])


if __name__ == "__main__":

    def scratch() -> pathlib.Path:
        return pathlib.Path(tempfile.mkdtemp())

    test_only_the_allowlisted_caches_are_candidates(scratch())
    test_a_protected_root_and_a_running_jobs_build_folder_are_left_alone(scratch())
    with pytest.MonkeyPatch.context() as patch:
        test_a_dry_run_counts_and_removes_nothing_and_apply_removes_only_the_candidates(scratch(), patch)
    test_clean_uv_is_refused()
