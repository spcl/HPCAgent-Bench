# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""experiments/roster.sh: the kernels an experiment tag names, for the launchers and remaining_kernels.py."""

import os
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"

#: How a submit script names the experiment its kernels come from: a TAG or RECORD_EXPERIMENT
#: default, or the kernels file it hands on.
TAG_SPELLINGS = re.compile(r"\b(?:TAG|RECORD_EXPERIMENT):-([\w.-]+)|\bkernels-([\w.-]+)\.txt")


def submit_script_tags() -> list[str]:
    found: set[str] = set()
    for script in sorted(EXPERIMENTS.glob("submit-*.sh")):
        for match in TAG_SPELLINGS.finditer(script.read_text()):
            found.add(match.group(1) or match.group(2))
    return sorted(found)


def roster_for(tag: str) -> list[str]:
    out = subprocess.run(
        ["bash", "-c", '. "$OPT/experiments/roster.sh"; roster_for "$1"', "roster", tag],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "OPT": str(REPO), "PY": sys.executable},
    )
    return [name for name in out.stdout.strip().split(",") if name]


def test_the_scan_finds_the_tags_the_campaign_scripts_run() -> None:
    """The parametrized check below passes vacuously on an empty scan, so the scan itself has to
    be seen finding the rosters the campaigns were launched on."""
    assert {"llr-focus40", "git-scicomp", "scicomp40"} <= set(submit_script_tags()), submit_script_tags()


@pytest.mark.parametrize("tag", submit_script_tags())
def test_every_tag_a_submit_script_uses_resolves_to_a_nonempty_roster(tag: str) -> None:
    """remaining_kernels.py sizes a next wave from this roster. An empty one reads as `names no
    kernels`, and the campaign's owed kernels cannot be computed at all."""
    assert roster_for(tag), f"roster_for {tag!r} is empty"


@pytest.mark.parametrize(
    ("tag", "size", "member"),
    [
        ("git-scicomp", 10, "fv3_dycore"),
        ("scicomp40", 40, "quatrex_rgf"),
        ("harness-focus20", 20, "scan_affine_decay"),
    ],
)
def test_a_tag_with_its_own_kernels_file_is_exactly_that_file(tag: str, size: int, member: str) -> None:
    """The kernels file is what the submit script turned into problems, so the roster must be its
    names: no inline `#` note, no track prefix, and nothing a manifest tag adds on top."""
    names = roster_for(tag)
    assert len(names) == size and member in names, names
