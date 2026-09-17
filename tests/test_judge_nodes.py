# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""experiments/judge_nodes.py: judges are sized by concurrent agents, one rank per five."""

import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

import judge_nodes  # noqa: E402


@pytest.mark.parametrize(
    ("agents", "nodes"),
    [
        (1, 1),  # a one-agent smoke still gets a judge node
        (20, 1),  # exactly one node of 4 ranks x 5 agents
        (21, 2),  # one agent past a node's capacity needs the next node
        (40, 2),  # scicomp40: 2 nodes = 8 ranks = one judge per 5 agents
        (120, 6),
    ],
)
def test_one_judge_rank_serves_five_agents_at_four_ranks_a_node(agents: int, nodes: int) -> None:
    assert judge_nodes.judge_nodes(agents) == nodes


def test_a_wave_without_agents_is_refused_rather_than_sized_to_zero() -> None:
    with pytest.raises(SystemExit):
        judge_nodes.judge_nodes(0)


def test_the_cli_counts_roster_names_times_repeat_and_ignores_notes(tmp_path: pathlib.Path) -> None:
    roster = tmp_path / "kernels.txt"
    roster.write_text("# header\n" + "".join(f"k{i}  # note\n" for i in range(15)) + "\n")
    out = subprocess.run(
        [sys.executable, str(REPO / "experiments" / "judge_nodes.py"), str(roster), "--repeat", "3"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert out.stdout.strip() == "3", out.stdout  # 45 agents -> ceil(45 / 20)
