# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The scripts/check_*.py pre-commit hooks share these two helpers rather than each copying them.

git_tracked used to be a `git ls-files` subprocess call rewritten, byte for byte, in three of the
hooks; is_generated_source was rewritten in two.
"""

import pathlib

from hpcagent_bench.precommit_support import git_tracked, is_generated_source


def test_git_tracked_lists_this_files_own_tracked_module(tmp_path: pathlib.Path) -> None:
    """A real repo (this one) with a real pattern finds a file known to be tracked."""
    from hpcagent_bench import paths

    found = git_tracked("hpcagent_bench/paths.py", cwd=paths.ROOT)
    assert found == ["hpcagent_bench/paths.py"]


def test_git_tracked_is_empty_outside_any_repo(tmp_path: pathlib.Path) -> None:
    """A hook run somewhere with no git history reports nothing to scan, not a crash."""
    assert git_tracked(cwd=tmp_path) == []


def test_is_generated_source_reads_the_autogen_marker_on_the_first_line(tmp_path: pathlib.Path) -> None:
    generated = tmp_path / "gen.py"
    generated.write_text("# hpcagent_bench-autogen -- do not edit\nx = 1\n", encoding="utf-8")
    handwritten = tmp_path / "hand.py"
    handwritten.write_text("x = 1\n", encoding="utf-8")

    assert is_generated_source(generated) is True
    assert is_generated_source(handwritten) is False


def test_is_generated_source_is_false_for_a_missing_file(tmp_path: pathlib.Path) -> None:
    assert is_generated_source(tmp_path / "does-not-exist.py") is False
