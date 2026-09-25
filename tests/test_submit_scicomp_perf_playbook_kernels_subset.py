# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""experiments/submit-scicomp-perf-playbook.sh's KERNELS_FILE contract: a subset gets exactly its
own kernels in a deterministic order, its own (never the canonical) env/problems names, AGENT_NODES
and the walltime sized off the SUBSET's own kernel count, and an unknown kernel name refused outright.

Runs from a temp copy of the launcher's inputs (test_submit_scicomp_perf_playbook's fixtures),
SUBMIT unset: nothing reaches sbatch.
"""

import json
import pathlib
import re
import subprocess

from tests.test_submit_scicomp_perf_playbook import env_dict, run_submit, submit_tree

#: Three real scientific_computing kernels, distinct from the fixture's own kernels-scicomp40.txt
#: roster ("kmp", "dfa"), so a differently-NAMED KERNELS_FILE narrows to its own subset.
SUBSET_KERNELS = ("kmp", "dfa", "heat_3d")


def prepared(result: subprocess.CompletedProcess[str]) -> dict[str, str]:
    """``{arm: (nodes, walltime)}`` off the launcher's SUBMIT=0 report."""
    pattern = re.compile(r"^prepared (\S+) \((\d+) nodes, (\d\d:\d\d:\d\d), .*\) -- not submitted")
    return {m.group(1): (m.group(2), m.group(3)) for line in result.stdout.splitlines() if (m := pattern.match(line))}


def test_kernels_file_contains_exactly_the_named_kernels_in_deterministic_order(tmp_path: pathlib.Path) -> None:
    root = submit_tree(tmp_path)
    (root / "experiments" / "subset3.txt").write_text("heat_3d\nkmp\ndfa\n")
    result = run_submit(root, ARMS="plain", KERNELS_FILE="subset3.txt")
    assert result.returncode == 0, result.stderr
    problems = root / "experiments" / "problems-scicomp-perf-playbook-qwen38-plain-subset3.jsonl"
    kernels = [json.loads(line)["kernel"].rsplit("/", 1)[-1] for line in problems.read_text().splitlines()]
    # make_problems.py sorts by the full path key, not the bare stem: kmp and dfa share
    # scientific_computing/finite_state_machine/, heat_3d sits under structured_grids/, so the
    # order is dfa, kmp, heat_3d -- NOT the alphabetical stem order (dfa, heat_3d, kmp).
    assert kernels == ["dfa", "kmp", "heat_3d"]
    assert set(kernels) == set(SUBSET_KERNELS)


def test_subset_env_and_problems_names_never_touch_the_canonical_files(tmp_path: pathlib.Path) -> None:
    root = submit_tree(tmp_path)
    (root / "experiments" / "subset3.txt").write_text("heat_3d\nkmp\ndfa\n")
    result = run_submit(root, ARMS="plain", KERNELS_FILE="subset3.txt")
    assert result.returncode == 0, result.stderr
    experiments = root / "experiments"
    assert (experiments / ".env.scicomp-perf-playbook-qwen38-plain-subset3").is_file()
    assert (experiments / "problems-scicomp-perf-playbook-qwen38-plain-subset3.jsonl").is_file()
    assert not (experiments / ".env.scicomp-perf-playbook-qwen38-plain").exists()
    assert not (experiments / "problems-scicomp-perf-playbook-qwen38-plain.jsonl").exists()


def test_agent_nodes_and_walltime_scale_with_the_subsets_own_kernel_count(tmp_path: pathlib.Path) -> None:
    """arm_walltime must size off the SUBSET's own kernel count, not the fixture's canonical
    kernels-scicomp40.txt roster. AGENT_NODES=1/AGENTS_PER_NODE=1 (both fixed, so AGENT_NODES stays
    put rather than being recomputed from the subset) forces one batch per kernel: 3 kernels -> 3
    batches of the 72000s (20h) budget plus the 3h staging allowance -- a different number than a
    2-kernel subset needs (see test_the_arms_differ_in_their_packet_and_nothing_else's 2-kernel run)."""
    root = submit_tree(tmp_path)
    (root / "experiments" / "subset3.txt").write_text("heat_3d\nkmp\ndfa\n")
    result = run_submit(root, ARMS="plain", KERNELS_FILE="subset3.txt", AGENT_NODES="1", AGENTS_PER_NODE="1")
    assert result.returncode == 0, result.stderr
    arms = prepared(result)
    # the "prepared" report names the ARM, never the file-suffixed env/problems names -- the same
    # convention test_submit_gpu_llr40_clean_dryrun.py's own KERNELS_FILE="kernels.txt" runs rely on.
    # arm_nodes() totals INFERENCE_NODES(1, inherited unchanged from llrbase-c:qwen38) +
    # AGENT_NODES(1, held at the override) + JUDGE_NODES(1, pinned) = 3.
    nodes, walltime = arms["scicomp-perf-playbook-qwen38-plain"]
    assert nodes == "3"
    assert walltime == "63:00:00"  # (72000 * 3 + 3599) // 3600 + 3 staging hours = 63
    env = env_dict(root / "experiments" / ".env.scicomp-perf-playbook-qwen38-plain-subset3")
    assert env["AGENT_NODES"] == "1"  # the override held, not silently recomputed
    env = env_dict(root / "experiments" / ".env.scicomp-perf-playbook-qwen38-plain-subset3")
    assert env["AGENT_NODES"] == "1"


def test_an_unknown_kernel_name_is_refused_not_silently_dropped(tmp_path: pathlib.Path) -> None:
    root = submit_tree(tmp_path)
    (root / "experiments" / "bad.txt").write_text("kmp\nnosuchkernel123\n")
    result = run_submit(root, ARMS="plain", KERNELS_FILE="bad.txt")
    assert result.returncode != 0
    assert "nosuchkernel123" in result.stderr
    assert not list((root / "experiments").glob(".env.scicomp-perf-playbook-*bad*"))
    assert not list((root / "experiments").glob("problems-scicomp-perf-playbook-*bad*"))
