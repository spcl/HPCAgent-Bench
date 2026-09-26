# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""dace hashes iteration order into generated code, so every process runs under PYTHONHASHSEED=0,
set in exactly two places: the job environment (experiments/env.sh) and CI (the workflow's top-level
env). No script, EDF, layer or module sets it again."""

import pathlib
import re
import subprocess

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
SETTING = re.compile(r"PYTHONHASHSEED\s*[=:]\s*['\"]?0|[\"']PYTHONHASHSEED[\"']\s*:")


def test_the_job_environment_exports_the_seed() -> None:
    done = subprocess.run(
        ["bash", "-c", f'. "{REPO}/experiments/env.sh" >/dev/null; printf "%s" "$PYTHONHASHSEED"'],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout == "0"


def test_ci_sets_the_seed_for_every_job() -> None:
    workflow = yaml.safe_load((REPO / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8"))
    assert str(workflow["env"]["PYTHONHASHSEED"]) == "0"


def test_nothing_else_sets_the_seed() -> None:
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True, check=True).stdout
    allowed = {"experiments/env.sh", ".github/workflows/tests.yml"}
    setters = [
        rel
        for rel in tracked.split()
        if rel not in allowed
        and not rel.startswith("tests/")
        and not rel.endswith(".md")
        and (REPO / rel).is_file()
        and SETTING.search((REPO / rel).read_text(encoding="utf-8", errors="replace"))
    ]
    assert setters == []
