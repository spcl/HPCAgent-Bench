# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""experiments/roster.sh's roster_for(), wired to experiments/tags.yaml (hpcagent_bench.tags) as
one more fallback ahead of the plain manifest experiment_tags scan.

Every test points HPCAGENT_BENCH_TAGS_FILE at its own temp registry (hpcagent_bench.tags.REGISTRY's
env-var override), so none of them read the real (shipped empty) experiments/tags.yaml and none of
them touch the checkout -- roster_for is invoked for real, subprocess and all, the same way
tests/test_roster.py's own roster_for() helper is.
"""

import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def roster_for(tag: str, registry_text: str, tmp_path: pathlib.Path) -> subprocess.CompletedProcess[str]:
    registry = tmp_path / "tags.yaml"
    registry.write_text(registry_text)
    env = {
        **os.environ,
        "OPT": str(REPO),
        "PY": sys.executable,
        "HPCAGENT_BENCH_TAGS_FILE": str(registry),
    }
    return subprocess.run(
        ["bash", "-c", '. "$OPT/experiments/roster.sh"; roster_for "$1"', "roster", tag],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


def test_a_tags_yaml_only_tag_resolves_through_the_new_branch(tmp_path: pathlib.Path) -> None:
    result = roster_for("mytag", "tags:\n  mytag:\n    list:\n      - explicit:kmp,dfa\n", tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "dfa,kmp"


def test_an_alias_resolves_through_its_target(tmp_path: pathlib.Path) -> None:
    result = roster_for(
        "shortcut",
        "tags:\n  mytag:\n    list:\n      - explicit:kmp,dfa\naliases:\n  shortcut: mytag\n",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "dfa,kmp"


def test_an_existing_kernels_file_still_wins_over_a_tags_yaml_entry_of_the_same_name(
    tmp_path: pathlib.Path,
) -> None:
    """git-scicomp already has a real experiments/kernels-git-scicomp.txt (10 kernels) -- a
    tags.yaml entry of the same name must never shadow it."""
    result = roster_for("git-scicomp", "tags:\n  git-scicomp:\n    list:\n      - explicit:kmp\n", tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.strip().split(",")) == 10


def test_a_tag_naming_neither_a_file_nor_tags_yaml_falls_back_to_the_manifest_scan_unchanged(
    tmp_path: pathlib.Path,
) -> None:
    """A regression check: llr-focus40 (a real, 40-kernel manifest experiment_tags label, no
    kernels-llr-focus40.txt file) must resolve exactly as before once tags.yaml exists but does
    not name it."""
    result = roster_for("llr-focus40", "tags: {}\n", tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.strip().split(",")) == 40


def test_an_unregistered_tag_still_gets_the_existing_clear_refusal(tmp_path: pathlib.Path) -> None:
    result = roster_for("no-such-tag-anywhere", "tags: {}\n", tmp_path)
    assert result.returncode == 2
    assert "matched no kernels" in result.stderr


def test_a_circular_tags_yaml_reference_is_refused_with_a_clear_message(tmp_path: pathlib.Path) -> None:
    result = roster_for("a", "tags:\n  a:\n    union:\n      - 'all@b'\n  b:\n    union:\n      - 'all@a'\n", tmp_path)
    assert result.returncode == 2
    assert "circular" in result.stderr
