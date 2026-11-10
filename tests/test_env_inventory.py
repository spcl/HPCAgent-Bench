# Copyright 2026 the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The environment-name inventory (helpers/scripts/checks/check_env_vars.py) runs on this tree and measures
what the env cleanup keeps count of."""

import re

from hpcagent_bench import paths
from tests.fresh_module import module_at

SCRIPT = paths.ROOT / "helpers" / "scripts" / "checks" / "check_env_vars.py"
COLUMNS = ["# name", "py", "sh", "first_py", "first_sh", "group"]


def test_the_inventory_of_this_tree(tmp_path, capsys) -> None:
    """The TSV has the six columns, names the knobs the cleanup tracks with sane read counts, and its F10-scope
    totals land near the numbers the cleanup was scoped from (python 164 distinct / 127 project-specific,
    shell 462 an upper bound)."""
    check = module_at(SCRIPT, "check_env_vars")
    out = tmp_path / "env_vars.tsv"
    assert check.main(["--root", str(paths.ROOT), "--out", str(out)]) == 0

    rows = [line.split("\t") for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0] == COLUMNS
    table = {row[0]: row for row in rows[1:]}
    assert len(table) > 400

    run_dir = table["RUN_DIR"]
    assert 10 <= int(run_dir[1]) <= 20, "the F10 table says RUN_DIR reads 14 times in python"
    assert 60 <= int(run_dir[2]) <= 120, "the F10 table says RUN_DIR appears 92 times in shell"
    assert run_dir[5] == "project"
    assert table["SLURM_JOB_ID"][5] == "vendor"
    for name in ("AGENT_EFFORT", "CLAUDE_MODEL", "JUDGE_TIMEOUT_SECONDS", "HPCAGENT_BENCH_NCORES", "HPCAGENT_BENCH_REPO"):
        assert 1 <= int(table[name][1]) + int(table[name][2]), f"{name} must appear in the inventory"
        assert table[name][3] or table[name][4], f"{name} must carry a first-read location"

    summary = capsys.readouterr().err
    scoped = re.search(r"the F10 scope\): (\d+) distinct, (\d+) project", summary)
    assert scoped is not None, f"no F10-scope totals in the summary: {summary}"
    assert 140 <= int(scoped.group(1)) <= 190, "distinct python names of hpcagent_bench/ + agent/ (the scan said 164)"
    assert 110 <= int(scoped.group(2)) <= 145, "project-specific python names of hpcagent_bench/ + agent/ (the scan said 127)"
    shell = re.search(r"shell, whole tree: (\d+) distinct", summary)
    assert shell is not None
    assert 430 <= int(shell.group(1)) <= 560, "distinct ${NAME} over the tree's shell"


def test_the_project_split_is_prefix_and_name_based() -> None:
    """Vendor stacks and generic shell names are outside the cleanup's moving set; everything else is in it."""
    check = module_at(SCRIPT, "check_env_vars")
    for name in ("SLURM_JOB_ID", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "UV_INDEX", "LD_LIBRARY_PATH", "PATH"):
        assert not check.is_project(name)
    for name in ("RUN_DIR", "AGENT_EFFORT", "HPCAGENT_BENCH_NCORES", "JUDGE_TIMEOUT_SECONDS", "PROBLEMS_FILE"):
        assert check.is_project(name)
