# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fabric settings that must reach the inference server inside its container.

The CSCS netstack hook writes its own NCCL defaults into the container after the step environment,
``NCCL_NET_GDR_LEVEL=PHB`` among them, so a layer's ``NCCL_NET_GDR_LEVEL`` only takes effect set on the
server command itself (docs/serving/known-issues.md #5). The role step cannot re-read the layer env inside
its container, so the batch shell hands the value over as ``SERVER_NCCL_NET_GDR_LEVEL``. ``run_vllm_node``
cannot be sourced, so the command is pinned against the shipped text, as tests/test_sglang_pp_serve_args.py does.
"""

import re

from hpcagent_bench import paths

ROOT = paths.ROOT
SCRIPT = ROOT / "hpcagent_bench/cluster/run_cluster.sh"
PP_LAYER = ROOT / "experiments/layers/pp.env"


def test_the_gdr_level_is_set_on_the_server_command_of_both_engines() -> None:
    text = SCRIPT.read_text()
    assert (
        'local -a server_env=(env ${SERVER_NCCL_NET_GDR_LEVEL:+"NCCL_NET_GDR_LEVEL=${SERVER_NCCL_NET_GDR_LEVEL}"})'
        in text
    )
    commands = re.findall(r"command=\(\n\s*(\S+)\n(?:.*\n){0,2}?.*-m (sglang\.launch_server|vllm\.entrypoints)", text)
    assert {engine for first, engine in commands} == {"sglang.launch_server", "vllm.entrypoints"}, commands
    assert all(first == '"${server_env[@]}"' for first, engine in commands), commands


def test_the_pipeline_layer_turns_gpu_direct_rdma_off_by_name() -> None:
    """RCCL reads a number as an old-style path level: ``0`` left GPU-direct RDMA on, ``LOC`` turns it off."""
    values = [
        line.split("=", 1)[1] for line in PP_LAYER.read_text().splitlines() if line.startswith("NCCL_NET_GDR_LEVEL=")
    ]
    assert values == ["LOC"], values


if __name__ == "__main__":
    test_the_gdr_level_is_set_on_the_server_command_of_both_engines()
    test_the_pipeline_layer_turns_gpu_direct_rdma_off_by_name()
