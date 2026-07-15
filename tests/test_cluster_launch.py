# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure-logic tests for the single-job cluster launcher (no MPI / no cluster).

Covers the rank->role partition, the URL assembly, and the vLLM argv -- the parts that
decide the deployment shape; the MPI orchestration + subprocess lifecycle are exercised
only on a real allocation.
"""
import socket

import pytest

from optarena.harness.cluster_launch import (JUDGE, RankRole, VLLM_HEAD, VLLM_WORKER, assemble_urls,
                                             endpoint_hostport, expected_world, plan_roles, rank_status,
                                             settle_rounds, vllm_command)


class FakeProc:
    """Stand-in for a subprocess.Popen: poll() returns the fixed code (None = still running)."""

    def __init__(self, code):
        self.code = code

    def poll(self):
        return self.code


HEAD = RankRole(VLLM_HEAD, 0, 0, is_driver=True)
WORKER = RankRole(VLLM_WORKER, 0, 0, is_driver=False)


def test_expected_world_is_inference_plus_judge():
    assert expected_world(2, 1, 1) == 3  # 2 single-node endpoints + 1 judge
    assert expected_world(1, 4, 2) == 6  # 1 four-node endpoint + 2 judges
    assert expected_world(3, 2, 4) == 10


def test_plan_single_node_endpoints():
    roles = plan_roles(3, inference_endpoints=2, nodes_per_vllm=1, judge_nodes=1)
    assert [r.role for r in roles] == [VLLM_HEAD, VLLM_HEAD, JUDGE]
    assert [r.endpoint for r in roles] == [0, 1, -1]
    assert roles[0].is_driver and not roles[1].is_driver and not roles[2].is_driver
    # every single-node endpoint is its own head
    assert [r.head_rank for r in roles[:2]] == [0, 1]


def test_plan_multinode_endpoints_group_by_k():
    # I=2 endpoints x K=2 nodes + J=1 judge = 5 nodes
    roles = plan_roles(5, inference_endpoints=2, nodes_per_vllm=2, judge_nodes=1)
    assert [r.role for r in roles] == [VLLM_HEAD, VLLM_WORKER, VLLM_HEAD, VLLM_WORKER, JUDGE]
    assert [r.endpoint for r in roles] == [0, 0, 1, 1, -1]
    # each worker points at its endpoint's head rank so it can join that ray cluster
    assert [r.head_rank for r in roles] == [0, 0, 2, 2, -1]
    assert roles[0].is_driver
    assert not any(r.is_driver for r in roles[1:])


def test_plan_rejects_wrong_world_size():
    with pytest.raises(ValueError, match="world size 4 != I.K . J = 3"):
        plan_roles(4, inference_endpoints=2, nodes_per_vllm=1, judge_nodes=1)


@pytest.mark.parametrize("endpoints,k,judge", [(0, 1, 1), (1, 0, 1), (1, 1, 0), (-1, 1, 1)])
def test_plan_rejects_nonpositive_counts(endpoints, k, judge):
    with pytest.raises(ValueError, match=">= 1"):
        plan_roles(max(endpoints * k + judge, 1), endpoints, k, judge)


def test_assemble_urls_orders_by_endpoint_then_rank():
    # deliberately out of rank order to prove the sort keys, K=2 (rank 1/3 are workers -> no URL)
    gathered = [
        {"rank": 4, "role": JUDGE, "endpoint": -1, "hostname": "nid04"},
        {"rank": 2, "role": VLLM_HEAD, "endpoint": 1, "hostname": "nid02"},
        {"rank": 0, "role": VLLM_HEAD, "endpoint": 0, "hostname": "nid00"},
        {"rank": 1, "role": VLLM_WORKER, "endpoint": 0, "hostname": "nid01"},
        {"rank": 3, "role": VLLM_WORKER, "endpoint": 1, "hostname": "nid03"},
        {"rank": 5, "role": JUDGE, "endpoint": -1, "hostname": "nid05"},
    ]
    vllm_urls, judge_urls = assemble_urls(gathered, vllm_port=8000, judge_port=8800)
    assert vllm_urls == ["http://nid00:8000/v1", "http://nid02:8000/v1"]  # heads only, by endpoint
    assert judge_urls == ["http://nid04:8800", "http://nid05:8800"]       # judges by rank


def test_vllm_command_single_node_has_no_pipeline_or_ray():
    cmd = vllm_command("Qwen/Q", 8000, tensor_parallel=4, pipeline_parallel=1, extra=[])
    assert "--tensor-parallel-size" in cmd and cmd[cmd.index("--tensor-parallel-size") + 1] == "4"
    assert "--pipeline-parallel-size" not in cmd
    assert "ray" not in cmd


def test_vllm_command_multinode_turns_on_ray_pipeline():
    cmd = vllm_command("big/model", 8000, tensor_parallel=4, pipeline_parallel=3, extra=["--max-model-len", "8192"])
    assert cmd[cmd.index("--pipeline-parallel-size") + 1] == "3"
    assert cmd[cmd.index("--distributed-executor-backend") + 1] == "ray"
    assert cmd[-2:] == ["--max-model-len", "8192"]  # passthrough preserved, at the end


def test_endpoint_hostport_parses_v1_suffix_and_bare():
    assert endpoint_hostport("http://nid00:8000/v1") == ("nid00", 8000)
    assert endpoint_hostport("nid07:8800") == ("nid07", 8800)


def test_settle_rounds_normal_and_floored_at_two():
    assert settle_rounds(1800.0, poll_interval=5.0) == 360
    # a sub-2*interval timeout must still get >= 2 rounds (>= 1 poll AFTER a grace sleep), never 1/0,
    # else a spawn that dies just after Popen is misreported pending instead of dead
    assert settle_rounds(8.0, poll_interval=5.0) == 2
    assert settle_rounds(0.0, poll_interval=5.0) == 2
    assert settle_rounds(-3.0, poll_interval=5.0) == 2


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_settle_rounds_non_finite_does_not_crash(bad):
    # argparse's type=float accepts 'inf'/'nan'; these must NOT crash int(), just cap to a long wait
    assert settle_rounds(bad, poll_interval=5.0) == 24 * 3600 // 5


def test_rank_status_dead_when_a_server_exited_nonzero():
    # a server process that exited with a non-zero code is fatal -> the settle loop aborts the run
    st = rank_status(HEAD, [FakeProc(1)], vllm_port=8000, judge_port=8800, hostname="nid00", rank=0)
    assert st["kind"] == "dead" and "rc=1" in st["detail"]


def test_rank_status_worker_ready_while_alive_needs_no_port():
    # a ray worker has no serving port of its own -> ready as soon as its join process is alive
    st = rank_status(WORKER, [FakeProc(None)], vllm_port=8000, judge_port=8800, hostname="nid01", rank=1)
    assert st["kind"] == "ready"


def test_rank_status_pending_until_port_binds_then_ready():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))  # reserve a free port but do NOT listen yet
    port = listener.getsockname()[1]
    try:
        pending = rank_status(HEAD, [FakeProc(None)], vllm_port=port, judge_port=8800, hostname="nid00", rank=0)
        assert pending["kind"] == "pending"  # nothing accepting -> connection refused -> pending
        listener.listen(1)
        ready = rank_status(HEAD, [FakeProc(None)], vllm_port=port, judge_port=8800, hostname="nid00", rank=0)
        assert ready["kind"] == "ready"  # port now accepts -> ready
    finally:
        listener.close()
