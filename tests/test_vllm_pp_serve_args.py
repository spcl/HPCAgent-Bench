# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pipeline-parallel serve arguments in run_cluster.sh.

Async scheduling is vLLM's default and is the only caller that runs a collective on
``pp.device_group``; every other use of that group is P2P, which torch serves from per-pair
2-rank communicators. Under lazy init the first decode therefore bootstraps a 4-rank and a
2-rank communicator concurrently, the bootstrap exchanges collide, and rccl reports
"Message truncated : received 1024 bytes instead of 512" (nranks x 256). That killed the
four-node kimi endpoint in 600262, 604463 and 604479 within a minute of the first request.

``run_vllm_node`` cannot be sourced -- it needs Slurm variables, downloads a snapshot and ends
in ``exec`` -- so the argv is pinned against the shipped text.
"""

import pathlib
import re

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "containers/cluster/example-script/run_cluster.sh"
# Indentation-agnostic: the branch already moved one level deeper once when run_vllm_node was
# wrapped, and pinning a literal four spaces made three tests fail over a change that touched none
# of the argv they assert. The backreference keeps the closing `else` matched at the branch's own
# level, so a nested if/else inside it cannot end the match early.
PP_BRANCH = re.compile(r"^(\s*)elif \(\( INFERENCE_NODES > 1 \)\); then$(.*?)^\1else$", re.MULTILINE | re.DOTALL)


def pp_branch() -> str:
    match = PP_BRANCH.search(SCRIPT.read_text())
    assert match, "the INFERENCE_NODES > 1 branch of run_vllm_node moved; re-point this test"
    return match.group(2)


def test_async_scheduling_is_off_on_the_pipeline_path():
    assert "--no-async-scheduling" in pp_branch()


def test_the_flag_is_reachable_but_not_the_default():
    """An operator can re-enable it to re-test upstream, and gets today's behaviour if they do not."""
    branch = pp_branch()
    assert "VLLM_ASYNC_SCHEDULING:-0" in branch
    assert '!= "1"' in branch


def test_single_node_endpoints_do_not_carry_the_flag():
    """A 1-node endpoint has no pp group and no per-pair P2P, so the collision cannot arise and the
    throughput async scheduling buys should be kept."""
    code = [line for line in SCRIPT.read_text().splitlines() if not line.lstrip().startswith("#")]
    hits = sum(line.count("--no-async-scheduling") for line in code)
    assert hits == 1, "the flag leaked outside the pipeline branch"


def test_the_cpu_group_timeout_is_still_set():
    """Separate hardening: the gloo metadata group defaults to 1800 s while
    --distributed-timeout-seconds covers only the device group."""
    assert "--cpu-distributed-timeout-seconds" in pp_branch()
