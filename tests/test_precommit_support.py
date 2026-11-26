# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The helpers/scripts/checks/check_*.py pre-commit hooks share these two helpers rather than each copying them.

git_tracked (a `git ls-files` subprocess call) and is_generated_source live here once.
"""

import pathlib

import pytest

from hpcagent_bench.precommit_support import git_tracked, is_generated_source


def test_git_tracked_lists_this_files_own_tracked_module() -> None:
    """A real repo (this one) with a real pattern finds a file known to be tracked."""
    from hpcagent_bench import paths

    found = git_tracked("hpcagent_bench/paths.py", cwd=paths.ROOT)
    assert found == ["hpcagent_bench/paths.py"]


def test_git_tracked_is_empty_outside_any_repo(tmp_path: pathlib.Path) -> None:
    """A hook run somewhere with no git history reports nothing to scan, not a crash."""
    assert git_tracked(cwd=tmp_path) == []


@pytest.mark.parametrize(
    ("content", "generated"),
    [
        ("# hpcagent_bench-autogen -- do not edit\nx = 1\n", True),
        ("x = 1\n", False),
        (None, False),  # a missing file
    ],
    ids=["marker", "handwritten", "missing"],
)
def test_is_generated_source_reads_the_autogen_marker_on_the_first_line(
    tmp_path: pathlib.Path, content: str | None, generated: bool
) -> None:
    path = tmp_path / "file.py"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    assert is_generated_source(path) is generated
